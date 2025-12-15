# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Baseten

import logging
import ctypes

import torch
from torch.utils.cpp_extension import load_inline

source = R"""
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

namespace sa_spec::cuda {
__attribute__((visibility("default")))
extern "C"
void* get_torch_stream_from_python() {
    return c10::cuda::getCurrentCUDAStream().stream();
}
} // namespace sa_spec::cuda

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
"""

logging.debug("Compiling sa_spec_torch_adapter...")
sa_spec_torch_module = load_inline(
    name="sa_spec_torch_adapter",
    cpp_sources=[source],
    with_cuda=True,
)

so_path = sa_spec_torch_module.__file__
ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

logging.debug("Loaded sa_spec_torch_adapter")
