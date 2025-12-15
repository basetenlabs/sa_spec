// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <sa_spec/util/memory.hpp>
#include <sa_spec/util/buffer.hpp>
#include <sa_spec/suffix_automaton.hpp>

namespace sa_spec::cuda::test {

void assign_int(int* a, int val);

void push_back(managed_ptr_raw<MemLoc::GPU, DynamicBuffer<int, 4>> vec, int val);
void push_back(managed_ptr_raw<MemLoc::PINNED, DynamicBuffer<int, 4>> vec, int val);

void extend(managed_ptr_raw<MemLoc::GPU, SuffixAutomaton> slot, Token val);
void extend(managed_ptr_raw<MemLoc::PINNED, SuffixAutomaton> slot, Token val);

} // namespace sa_spec::cuda::test
