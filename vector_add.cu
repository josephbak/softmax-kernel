#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;              // 1,048,576
    const size_t bytes = n * sizeof(float);

    std::vector<float> ha(n, 1.0f), hb(n, 2.0f), hc(n);
    float *da = nullptr, *db = nullptr, *dc = nullptr;

    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    vectorAdd<<<grid, block>>>(da, db, dc, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost);

    // quick correctness check
    for (int i = 0; i < 5; ++i) std::printf("c[%d]=%.1f\n", i, hc[i]); // expect 3.0
    float max_err = 0.f;
    for (int i = 0; i < n; ++i) max_err = fmaxf(max_err, fabsf(hc[i] - 3.0f));
    std::printf("max error = %g\n", max_err);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    return 0;
}