#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Add near top:
static double theoretical_bw_GBps() {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    // memClockRate in kHz, memBusWidth in bits. GDDR is double data rate → ×2
    const double mem_clock_hz = p.memoryClockRate * 1000.0;
    const double bus_bytes = p.memoryBusWidth / 8.0;
    const double bw_Bps = 2.0 * mem_clock_hz * bus_bytes;
    return bw_Bps / 1e9; // GB/s
}


#define CUDA_OK(expr) do { \
  cudaError_t _e = (expr); \
  if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", \
            cudaGetErrorString(_e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K) {
    // C = A[M×K] * B[K×N]  => C[M×N]
    int row = blockIdx.y * blockDim.y + threadIdx.y; // [0..M)
    int col = blockIdx.x * blockDim.x + threadIdx.x; // [0..N)
    if (row >= M || col >= N) return;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

int main() {

    const int M = 8192, N = 8192, K = 8192;
    // const int M = 4096, N = 4096, K = 4096;
    // const int M = 512, N = 512, K = 512;            // tune later
    const size_t bytesA = (size_t)M*K*sizeof(float);
    const size_t bytesB = (size_t)K*N*sizeof(float);
    const size_t bytesC = (size_t)M*N*sizeof(float);

    std::vector<float> hA(M*K, 1.0f), hB(K*N, 2.0f), hC(M*N, 0.0f);
    float *dA=nullptr, *dB=nullptr, *dC=nullptr;

    CUDA_OK(cudaMalloc(&dA, bytesA));
    CUDA_OK(cudaMalloc(&dB, bytesB));
    CUDA_OK(cudaMalloc(&dC, bytesC));
    CUDA_OK(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);

    cudaEvent_t t0,t1; CUDA_OK(cudaEventCreate(&t0)); CUDA_OK(cudaEventCreate(&t1));
    CUDA_OK(cudaEventRecord(t0));
    matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(t1));
    CUDA_OK(cudaEventSynchronize(t1));

    float ms=0.f; CUDA_OK(cudaEventElapsedTime(&ms, t0, t1));
    std::printf("matmul_naive %dx%dx%d: %.3f ms\n", M,N,K, ms);
    CUDA_OK(cudaEventDestroy(t0)); CUDA_OK(cudaEventDestroy(t1));

    double t = ms / 1e3;
    double flops = 2.0 * M * N * K;
    double gflops = flops / (t * 1e9);
    double bytes = 4.0 * M * N * (2.0 * K + 1.0);
    double gbps = bytes / (t * 1e9);

    std::printf("theoretical BW: %.1f GB/s\n", theoretical_bw_GBps());
    std::printf("achieved: %.1f GFLOP/s, estimated BW: %.1f GB/s, AI: %.3f flop/byte\n", gflops, gbps, flops/bytes);

    CUDA_OK(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    // quick correctness: since A=1, B=2 → each C[row,col] = sum_k 1*2 = 2*K
    float max_err = 0.f;
    for (int i=0;i<M*N;++i) max_err = fmaxf(max_err, fabsf(hC[i] - 2.0f*K));
    std::printf("max error = %g (expected 0)\n", max_err);

    CUDA_OK(cudaFree(dA)); CUDA_OK(cudaFree(dB)); CUDA_OK(cudaFree(dC));
    return (max_err==0.f)? 0:1;
}