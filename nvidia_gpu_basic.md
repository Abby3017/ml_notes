# GPU Architecture

This guide provides background on the structure of a GPU, how operations are executed, and common limitations with deep learning operations.

## Fundamental

The GPU is a highly parallel processor architecture, composed of processing elements and a memory hierarchy. At a high level, NVIDIA® GPUs consist of a number of Streaming Multiprocessors (SMs), on-chip L2 cache, and high-bandwidth DRAM.
Arithmetic and other instructions are executed by the SMs; data and code are accessed from DRAM via the L2 cache. As an example, an NVIDIA A100 GPU contains 108 SMs, a 40 MB L2 cache, and up to 2039 GB/s bandwidth from 80 GB of HBM2 memory.
