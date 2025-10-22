#include <cstdio>
__global__ void hello() {
    printf("Hello from GPU thread %d, block %d!\n", threadIdx.x, blockIdx.x);
}
int main() {
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
