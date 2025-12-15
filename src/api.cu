// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#include <sa_spec/api.hpp>

#include <sa_spec/util/assert.hpp>
#include <sa_spec/util/logger.hpp>
#include <sa_spec/util/memory.hpp>
#include <sa_spec/suffix_automaton.hpp>
#include <sa_spec/util/memory.cuh>

namespace sa_spec::cuda {

__global__ void extend_kernel(
    int batch_size,
    int draft_length,
    SuffixAutomaton* slots,
    const BatchIndex::ValueType* batch_indices,
    NumTokens::ValueType* depth_out,
    Token::ValueType* draft_out,
    const Token::ValueType* accepted_in,
    const NumTokens::ValueType* accepted_lens_in) {

    if (threadIdx.x > 0) {
        return;
    }

    ASSERT(slots != nullptr
           && batch_indices != nullptr
           && depth_out != nullptr
           && draft_out != nullptr
           && accepted_in != nullptr
           && accepted_lens_in != nullptr);

    int i = blockIdx.x;
    ASSERT(i < batch_size);

    int batch_index = batch_indices[i];
    ASSERT(batch_index >= 0 && batch_index < Config::MAX_SLOTS);


    SuffixAutomaton& slot = slots[batch_index];

    int num_new_tokens = accepted_lens_in[i];

    LOG_TRACE("START extend_kernel batch_index=%d num_new_tokens=%d tokens.size()=%d", +batch_index, +num_new_tokens, +slot.m_tokens.size());

    for (int j = 0; j < num_new_tokens; j++) {
        slot.extend(Token(accepted_in[i * (draft_length + 1) + j]));
    }

    auto result = slot.lookup();
    if (result.has_value()) {
        depth_out[i] = result->len;
        slot.get_draft_tokens(&draft_out[i * draft_length], draft_length, result->pos);
    } else {
        depth_out[i] = 0;
    }

    LOG_TRACE("END extend_kernel batch_index=%d num_new_tokens=%d depth_out=%d tokens.size()=%d",
              batch_index,
              num_new_tokens,
              depth_out[i],
              +slot.m_tokens.size());
}

void invoke_extend(
    int batch_size,
    int draft_length,
    SuffixAutomaton* slots,
    const BatchIndex::ValueType* batch_indices,
    NumTokens::ValueType* depth_out,
    Token::ValueType* draft_out,
    const Token::ValueType* accepted_in,
    const NumTokens::ValueType* accepted_lens_in) {

    LOG_TRACE("invoke_extend batch_size= %d draft_length= %d", +batch_size, +draft_length);

    auto stream = get_torch_stream();

    extend_kernel<<<batch_size, 1, 0, stream>>>(
        batch_size,
        draft_length,
        slots,
        batch_indices,
        depth_out,
        draft_out,
        accepted_in,
        accepted_lens_in
    );
}

} // namespace sa_spec::cuda
