// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <functional>

#include "cuda_callable.hpp"

namespace sa_spec {


template <typename T, typename Tag, auto DefaultValue = T()> class NamedType {
public:
    using ValueType = T;
    static constexpr auto Default = DefaultValue;

    constexpr NamedType() {
        static_assert(sizeof(decltype(*this)) == sizeof(T));
    }

    CUDA_CALLABLE explicit constexpr NamedType(const T &value) : value_(value) {}

    CUDA_CALLABLE explicit constexpr NamedType(T &&value) : value_(std::move(value)) {}

    CUDA_CALLABLE constexpr const T &get() const { return value_; }

    CUDA_CALLABLE constexpr T &get() { return value_; }

    CUDA_CALLABLE friend constexpr bool operator==(const NamedType &lhs, const NamedType &rhs) {
        return lhs.value_ == rhs.value_;
    }

    CUDA_CALLABLE friend constexpr bool operator<(const NamedType &lhs, const NamedType &rhs) {
        return lhs.value_ < rhs.value_;
    }

    CUDA_CALLABLE friend constexpr T operator+(const NamedType &lhs) {
        return lhs.value_;
    }

    T value_{DefaultValue};

    static_assert(std::is_trivially_copyable_v<T>);
};


} // namespace sa_spec

// Hash function for std::unordered_map
template <typename T, typename Tag> struct std::hash<sa_spec::NamedType<T, Tag>> {
    std::size_t operator()(const sa_spec::NamedType<T, Tag> &namedType) const {
        return std::hash<T>{}(namedType.get());
    }
};
