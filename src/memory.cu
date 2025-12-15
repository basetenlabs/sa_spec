// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#include <string>
#include <span>

#include <cuda_runtime.h>

#include <sa_spec/util/memory.hpp>

#include <sa_spec/util/memory.cuh>


namespace sa_spec::cuda {

extern "C" __attribute__((weak)) void* get_torch_stream_from_python() {
    return nullptr;
}

cudaStream_t get_torch_stream() {
    void* stream_ptr = get_torch_stream_from_python();
    return reinterpret_cast<cudaStream_t>(stream_ptr);
}

void malloc(void** ptr, size_t size) {
    cudaError_t error = cudaMalloc(ptr, size);
    CUDA_CHECK_ERROR(error);
}

void malloc_pinned(void** ptr, size_t size) {
    cudaError_t error = cudaMallocHost(ptr, size);
    CUDA_CHECK_ERROR(error);
}

void free(void* ptr) {
    cudaError_t error = cudaFree(ptr);
    CUDA_CHECK_ERROR(error);
}

void free_pinned(void* ptr) {
    cudaError_t error = cudaFreeHost(ptr);
    CUDA_CHECK_ERROR(error);

}

void memcpy(void* dst, const void* src, size_t size, memcpy_kind kind) {
    cudaMemcpyKind cuda_kind;
    switch (kind) {
        case memcpy_kind::HostToDevice:
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case memcpy_kind::DeviceToHost:
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case memcpy_kind::DeviceToDevice:
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            ASSERT(false);
            break;
    }

    cudaError_t error;
    cudaStream_t stream = get_torch_stream();
    if (stream != nullptr) {
        error = cudaMemcpyAsync(dst, src, size, cuda_kind);
    } else {
        error = cudaMemcpy(dst, src, size, cuda_kind);
    }
    CUDA_CHECK_ERROR(error);

}

void sync() {
    cudaError_t error = cudaDeviceSynchronize();
    CUDA_CHECK_ERROR(error);
}

} // namespace sa_spec::cuda
