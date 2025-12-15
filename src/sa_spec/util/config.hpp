// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <cstdlib>
#include <cstddef>
#include <string>
#include <utility>

namespace sa_spec {

#ifndef C_MAX_SEQUENCE_LENGTH
#define C_MAX_SEQUENCE_LENGTH 262144
#endif

#ifndef C_MAX_SLOTS
#define C_MAX_SLOTS 32
#endif


class Config {
public:
    static constexpr size_t MAX_SEQUENCE_LENGTH = C_MAX_SEQUENCE_LENGTH;
    static constexpr size_t MAX_SLOTS = C_MAX_SLOTS;


private:
    Config() = default;
};


} // namespace sa_spec
