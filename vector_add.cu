#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// error checking
#define CUDA_OK(expr)                                                         \
  do {                                                                        \
    cudaError_t _e = (expr);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                             \
              cudaGetErrorString(_e), __FILE__, __LINE__);                    \
      exit(1);                                                                \
    }                                                                         \
  } while (0)


__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;              // 1,048,576
    const size_t bytes = n * sizeof(float);

    std::vector<float> ha(n, 1.0f), hb(n, 2.0f), hc(n);
    float *da = nullptr, *db = nullptr, *dc = nullptr;

    // cudaMalloc(&da, bytes);
    // cudaMalloc(&db, bytes);
    // cudaMalloc(&dc, bytes);

    // cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice);
    // cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice);

    CUDA_OK(cudaMalloc(&da, bytes));
    CUDA_OK(cudaMalloc(&db, bytes));
    CUDA_OK(cudaMalloc(&dc, bytes));

    CUDA_OK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    cudaEvent_t t0, t1;
    CUDA_OK(cudaEventCreate(&t0));
    CUDA_OK(cudaEventCreate(&t1));
    CUDA_OK(cudaEventRecord(t0));
    vectorAdd<<<grid, block>>>(da, db, dc, n);
    CUDA_OK(cudaGetLastError());            // kernel launch status
    CUDA_OK(cudaEventRecord(t1));
    CUDA_OK(cudaEventSynchronize(t1));      // waits for kernel to finish

    float ms=0.f;
    CUDA_OK(cudaEventElapsedTime(&ms, t0, t1));
    printf("vectorAdd: %.3f ms\n", ms);

    CUDA_OK(cudaEventDestroy(t0));
    CUDA_OK(cudaEventDestroy(t1));

    CUDA_OK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

    // vectorAdd<<<grid, block>>>(da, db, dc, n);
    // cudaDeviceSynchronize();

    // cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost);

    // quick correctness check
    for (int i = 0; i < 5; ++i) std::printf("c[%d]=%.1f\n", i, hc[i]); // expect 3.0
    float max_err = 0.f;
    for (int i = 0; i < n; ++i) max_err = fmaxf(max_err, fabsf(hc[i] - 3.0f));
    std::printf("max error = %g\n", max_err);

    CUDA_OK(cudaFree(da));
    CUDA_OK(cudaFree(db));
    CUDA_OK(cudaFree(dc));

    // cudaFree(da); cudaFree(db); cudaFree(dc);
    return 0;
}