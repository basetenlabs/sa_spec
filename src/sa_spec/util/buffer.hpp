// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>

#include "assert.hpp"
#include "cuda_callable.hpp"

using std::byte;

namespace sa_spec {

template <typename T, typename F>
concept is_visitable =
        requires(T t, F &&func) { t.visit_chunks(std::forward<F>(func)); };

template <typename T, size_t Size, typename IndexT = size_t>
struct Buffer {
    const T &at(IndexT, IndexT) const = delete;
    T &at(IndexT, IndexT) = delete;

    CUDA_CALLABLE Buffer(std::initializer_list<T> list) {
        for (size_t i = 0; i < list.size(); i++) {
            m_data[i] = list.begin()[i];
        }
    }

    Buffer() = default;

    CUDA_CALLABLE const T &at(IndexT row) const {
        ASSERT(static_cast<size_t>(+row) < Size);
        return m_data[+row];
    }

    CUDA_CALLABLE T &at(IndexT row) {
        ASSERT(static_cast<size_t>(+row) < Size);
        return m_data[+row];
    }

    struct Iterator {
        const Buffer &vector;
        IndexT index;

        CUDA_CALLABLE Iterator(const Buffer &vector, IndexT index)
                : vector(vector), index(index) {}

        CUDA_CALLABLE const T &operator*() const { return vector.at(index); }

        CUDA_CALLABLE Iterator &operator++() {
            index = IndexT(+index + 1);
            return *this;
        }

        CUDA_CALLABLE bool operator==(const Iterator &other) const {
            return index == other.index;
        }

        CUDA_CALLABLE bool operator!=(const Iterator &other) const {
            return index != other.index;
        }
    };

    CUDA_CALLABLE Iterator begin() const { return Iterator(*this, IndexT(0)); }

    CUDA_CALLABLE Iterator end() const {
        return Iterator(*this, IndexT(Size));
    }

    CUDA_CALLABLE size_t size() const { return Size; }

    bool operator==(const Buffer<T, Size, IndexT> &other) const {
        static_assert(sizeof(decltype(*this)) == sizeof(m_data));
        return std::memcmp(this, &other, sizeof(m_data)) == 0;
    }

    void visit_chunks(auto &&func) const
        requires(!is_visitable<T, decltype(func)>)
    {
        func(static_cast<const void *>(this), sizeof(*this));
    }

    void visit_chunks(auto &&func) const
        requires(is_visitable<T, decltype(func)>)
    {
        for (size_t i = 0; i < Size; i++) {
            m_data[i].visit_chunks(func);
        }
    }

    void clear() {
        memset(static_cast<void*>(&m_data[0]), 0, sizeof(m_data));
    }

    T* data() { return &m_data[0]; }
    const T* data() const { return &m_data[0]; }

    std::array<T, Size> m_data;

    static_assert(std::is_trivially_copyable_v<T>);
};

template <typename T, size_t Size, size_t MAX_COLS,
                    typename IndexColT = size_t, typename IndexRowT = size_t>
struct Buffer2D : Buffer<Buffer<T, MAX_COLS, IndexColT>, Size, IndexRowT> {
    using Base = Buffer<Buffer<T, MAX_COLS, IndexColT>, Size, IndexRowT>;
    using Base::Base;

    CUDA_CALLABLE const T &at(IndexRowT row, IndexColT col) const {
        return Base::at(row).at(col);
    }

    CUDA_CALLABLE T &at(IndexRowT row, IndexColT col) {
        return Base::at(row).at(col);
    }
};

template <typename T, size_t Size, typename IndexT = size_t>
struct DynamicBuffer {
    
    CUDA_CALLABLE DynamicBuffer(std::initializer_list<T> list) {
        m_length = IndexT(list.size());
        m_data = Buffer<T, Size, IndexT>(list);
    }

    DynamicBuffer() = default;


    CUDA_CALLABLE void clear() { m_length = IndexT(0); }

    CUDA_CALLABLE IndexT size() const { return m_length; }

    CUDA_CALLABLE bool empty() const { return +size() == 0; }

    CUDA_CALLABLE void extend(size_t n) {
        m_length = IndexT(+m_length + n);
        ASSERT_THROW(static_cast<size_t>(+m_length) <= Size);
    }

    CUDA_CALLABLE T& push_back(const T &value) {
        ASSERT_THROW(static_cast<size_t>(+m_length) < Size);

        T& result = m_data.at(m_length);
        result = value;
        m_length = IndexT(+m_length + 1);
        return result;
    }

    CUDA_CALLABLE T& push_back(T&& value) {
        ASSERT_THROW(static_cast<size_t>(+m_length) < Size);
        T& result = m_data.at(m_length);
        result = std::move(value);
        m_length = IndexT(+m_length + 1);
        return result;
    }

    CUDA_CALLABLE T& pop_back() {
        ASSERT_THROW(!empty());
        T& result = m_data.at(IndexT(+m_length - 1));
        m_length = IndexT(+m_length - 1);
        return result;
    }

    CUDA_CALLABLE const T& at(IndexT row) const {
        ASSERT(row < m_length);
        return m_data.at(row);
    }

    CUDA_CALLABLE T &at(IndexT row) {
        ASSERT(row < m_length);
        return m_data.at(row);
    }

    void visit_chunks_(auto &&func) const {
        func(static_cast<const void*>(this),
             reinterpret_cast<const std::byte*>(&m_data.m_data[+m_length]) -
             reinterpret_cast<const std::byte*>(this));
    }

    void visit_chunks(auto &&func) const { visit_chunks_(func); }

    [[deprecated("Sparse copy on DynamicBuffer is not supported, falling back to "
                 "dense copy")]]
    void visit_chunks(auto &&func) const
        requires(is_visitable<T, decltype(func)>)
    {
        visit_chunks_(func);
    }

    T* data() { return m_data.data(); }
    const T* data() const { return m_data.data(); }

    CUDA_CALLABLE bool has_capacity() const { return +m_length < Size; }

    // IMPORTANT: pay attention to visit_chunks when modifying below
    IndexT m_length{0};
    Buffer<T, Size, IndexT> m_data;
};

template<typename T, size_t Size, typename IndexT>
auto to_std_vector(const DynamicBuffer<T, Size, IndexT>& vector) {
    std::vector<T> result;
    result.reserve(+vector.size());
    for (IndexT i = IndexT{0}; i < vector.size(); i = IndexT{+i + 1}) {
        result.push_back(vector.at(i));
    }
    return result;
}


} // namespace sa_spec
