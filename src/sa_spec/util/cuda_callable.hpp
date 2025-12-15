// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CALLABLE __host__ __device__ __forceinline__
#else
#define CUDA_CALLABLE
#define cudaStream_t int
#endif // __CUDACC__
