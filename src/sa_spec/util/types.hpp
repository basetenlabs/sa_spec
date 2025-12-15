// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <cstdint>

#include "named_type.hpp"

namespace sa_spec {


using Token = NamedType<int32_t, struct TokenTag>;

using BatchIndex = NamedType<int, struct BatchIndexTag>;
using RequestID = NamedType<uint64_t, struct RequestIDTag>;

constexpr BatchIndex operator""_bidx(unsigned long long c) {
    return BatchIndex{static_cast<BatchIndex::ValueType>(c)};
}

constexpr RequestID operator""_req(unsigned long long c) {
    return RequestID{static_cast<RequestID::ValueType>(c)};
}

constexpr Token operator""_tok(unsigned long long c) {
    return Token{static_cast<Token::ValueType>(c)};
}

using AcceptLength = NamedType<uint32_t, struct AcceptLengthTag>;
using MatchLength = NamedType<uint32_t, struct MatchLengthTag>;


} // namespace sa_spec
