// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <span>

#include <sa_spec/util/types.hpp>
#include <sa_spec/util/buffer.hpp>
#include <sa_spec/util/named_type.hpp>

namespace sa_spec {

using NumTokens = NamedType<int, struct NumTokensTag>;

struct SuffixAutomaton;

namespace cuda {
void invoke_extend(
    int batch_size,
    int draft_length,
    SuffixAutomaton* slots,
    const BatchIndex::ValueType* batch_indices,
    NumTokens::ValueType* depth_out,
    Token::ValueType* draft_out,
    const Token::ValueType* accepted_in,
    const NumTokens::ValueType* accepted_lens_in
);
} // namespace cuda


struct API {
    static API& get_instance();

    void add_request(RequestID request_id, std::span<const Token> tokens);

    void prepare(std::span<const RequestID> request_ids);

    void extend(
        int batch_size,
        int draft_length,
        NumTokens::ValueType* depth_out, // [batch_size]
        Token::ValueType* draft_out, // [batch_size, draft_length]
        const Token::ValueType* accepted_in, // [batch_size, draft_length+1]
        const NumTokens::ValueType* accepted_lens_in // [batch_size]
    );


    void get_active_tokens_for_test(RequestID request_id,
                                    Token::ValueType* tokens_out,
                                    size_t tokens_out_len);

private:
    API() = default;
};

} // namespace sa_spec
