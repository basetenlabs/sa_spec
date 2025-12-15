// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#include <cuda_runtime.h>

#include <sa_spec/suffix_automaton.hpp>
#include <sa_spec/util/types.hpp>

#include "utils_for_test.hpp"

namespace sa_spec::cuda::test {

__global__ void assign_kernel(int* a, int val) {
    *a = val;
}


__global__ void push_back_kernel(DynamicBuffer<int, 4>& vec, int val) {
    vec.push_back(val);
}


__global__ void extend_kernel(SuffixAutomaton& slot, Token val) {
    slot.extend(val);
}


void assign_int(int* a, int val) {
    assign_kernel<<<1, 1>>>(a, val);
}


void push_back(managed_ptr_raw<MemLoc::GPU, DynamicBuffer<int, 4>> vec, int val) {
    push_back_kernel<<<1, 1>>>(*vec, val);
}


void push_back(managed_ptr_raw<MemLoc::PINNED, DynamicBuffer<int, 4>> vec, int val) {
    push_back_kernel<<<1, 1>>>(*vec, val);
}


void extend(managed_ptr_raw<MemLoc::GPU, SuffixAutomaton> slot, Token val) {
    extend_kernel<<<1, 1>>>(*slot, val);
}


void extend(managed_ptr_raw<MemLoc::PINNED, SuffixAutomaton> slot, Token val) {
    extend_kernel<<<1, 1>>>(*slot, val);
}

} // namespace sa_spec::cuda::test
