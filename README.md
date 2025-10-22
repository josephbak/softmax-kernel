# Softmax Optimization Project

## Overview

This project implements and optimizes the **Softmax kernel** used in transformer-based attention mechanisms.  
It progresses from foundational CUDA (Compute Unified Device Architecture) kernels toward a high-performance, numerically stable, and memory-efficient Softmax implementation suitable for large-scale model inference.

The current stage focuses on establishing a reproducible GPU (Graphics Processing Unit) development environment, validating baseline CUDA kernels, and preparing a profiling framework for future optimization work.

---

## Phase 1 — GPU Foundations

### Current Progress

| Component | Status | Description |
|------------|---------|-------------|
| CUDA Environment | Completed | Configured RunPod GPU instance with CUDA 12.x and Visual Studio Code Remote SSH |
| GPU Validation | Completed | Verified GPU execution using a simple CUDA kernel (`hello.cu`) |
| Baseline Kernel | Completed | Implemented vector addition (`vector_add.cu`) with proper error checking and timing |
| Profiling | Partial | Nsight Compute CLI restricted; used CUDA event-based timing instead |
| Version Control | Completed | Integrated Git with SSH deploy key for persistent access |
| Next Stage | Planned | Implement matrix multiplication (`matmul.cu`) and reduction (`reduce.cu`) as precursors to Softmax |

---

## Implementation Notes

- **Kernel correctness:** validated memory transfers, thread indexing, and synchronization.  
- **Error handling:** unified through a `CUDA_OK(expr)` macro that reports failures with file and line context.  
- **Performance measurement:** runtime collected via CUDA events (`cudaEventRecord` / `cudaEventElapsedTime`).  
- **Profiling:** hardware counters unavailable in current environment (ERR_NVGPUCTRPERM). Timing metrics are used for baseline validation.

---

## Toolchain

- **GPU:** NVIDIA RTX (SM_89 architecture)
- **OS:** Ubuntu (RunPod container)
- **Compiler:** `nvcc` (NVIDIA CUDA Compiler, CUDA 12.x)
- **Editor:** Visual Studio Code (Remote SSH)
- **Profiler:** Nsight Compute CLI (restricted), CUDA Events

---

## Next Steps — Phase 2

1. Develop a 2D matrix multiplication kernel (`matmul.cu`) using shared memory.  
2. Implement a reduction kernel (`reduce.cu`) to accumulate row maxima and sums.  
3. Integrate the two components into a fused, numerically stable Softmax kernel.  
4. Profile, benchmark, and tune memory access patterns and warp efficiency.

---

## Author

**Joseph Bak**  
