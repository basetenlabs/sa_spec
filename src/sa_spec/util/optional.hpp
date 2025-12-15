// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <optional>

#include "cuda_callable.hpp"

namespace sa_spec {

// ABI-consistent alternative to std::optional
template<typename T>
struct optional {
    T m_data;
    bool m_has_value = false;

    CUDA_CALLABLE bool has_value() const {
        return m_has_value;
    }

    CUDA_CALLABLE T& value() {
        return m_data;
    }

    CUDA_CALLABLE const T& value() const {
        return m_data;
    }

    CUDA_CALLABLE T& operator*() {
        return m_data;
    }

    CUDA_CALLABLE const T& operator*() const {
        return m_data;
    }

    CUDA_CALLABLE T* operator->() {
        return &m_data;
    }

    CUDA_CALLABLE const T* operator->() const {
        return &m_data;
    }

    CUDA_CALLABLE optional& operator=(const T& value) {
        m_data = value;
        m_has_value = true;
        return *this;
    }

    CUDA_CALLABLE optional& operator=(T&& value) {
        m_data = std::move(value);
        m_has_value = true;
        return *this;
    }

    CUDA_CALLABLE optional& operator=(std::nullopt_t) {
        m_has_value = false;
        return *this;
    }

    CUDA_CALLABLE optional() : m_has_value(false) {}
    CUDA_CALLABLE optional(const T& value) : m_data(value), m_has_value(true) {}
    CUDA_CALLABLE optional(T&& value) : m_data(std::move(value)), m_has_value(true) {}
    CUDA_CALLABLE optional(std::nullopt_t) : m_has_value(false) {}

    CUDA_CALLABLE void reset() {
        m_has_value = false;
    }
};

} // namespace sa_spec
