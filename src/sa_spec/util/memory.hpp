// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <cstring>
#include <cstdint>
#include <type_traits>
#include <variant>
#include <memory>
#include <deque>

#include <sa_spec/util/config.hpp>
#include <sa_spec/util/assert.hpp>
#include <sa_spec/util/optional.hpp>

namespace sa_spec {


using std::byte;

enum class MemLoc : uint8_t {
    CPU = 0,
    GPU = 1,
    PINNED = 2,
};

namespace cuda {

bool is_torch_enabled();

void malloc_pinned(void **ptr, size_t size);

void malloc(void **ptr, size_t size);

void free(void *ptr);

void free_pinned(void *ptr);

enum class memcpy_kind : uint8_t {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
};

void memcpy(void *dst, const void *src, size_t size, memcpy_kind kind);

void sync();

} // namespace cuda

namespace cpu {

template <typename T> auto make_unique(auto &&...args) {
    return std::make_unique<T>(std::forward<decltype(args)>(args)...);
}

} // namespace cpu

static constexpr bool is_cuda() {
#ifdef __CUDACC__
    return true;
#else
    return false;
#endif
}

template <MemLoc Location, typename T, bool NON_OWNING = false>
struct managed_ptr;

template <MemLoc Location, typename T>
using managed_ptr_raw = managed_ptr<Location, T, true>;

template <MemLoc Location, typename T, bool NON_OWNING>
struct managed_ptr {
    static constexpr MemLoc Location_ = Location;

    using Raw = managed_ptr<Location, T, true>;

    T *ptr_;

    managed_ptr() : ptr_(nullptr) {}

    managed_ptr(T *ptr) requires (NON_OWNING) : ptr_(ptr) {}

    void clear() {
        if (ptr_ == nullptr) {
            return;
        }

        if (NON_OWNING) {
            ptr_ = nullptr;
            return;
        }

        if constexpr (Location == MemLoc::GPU) {
            cuda::free(ptr_);
        } else if constexpr (Location == MemLoc::PINNED) {
            cuda::free_pinned(ptr_);
        } else {
            [[maybe_unused]] auto _ = std::unique_ptr<T>(ptr_);
        }

        ptr_ = nullptr;
    }

    ~managed_ptr() noexcept {
        clear();
    }

    static managed_ptr own(T *ptr) requires(!NON_OWNING) {
        managed_ptr ret;
        ret.ptr_ = ptr;
        return ret;
    }

    template <bool OTHER_IS_RAW>
    managed_ptr(const managed_ptr<Location, T, OTHER_IS_RAW> &other)
        requires(NON_OWNING) : ptr_(other.ptr_) {}

    template <bool OTHER_IS_RAW>
    managed_ptr &operator=(const managed_ptr<Location, T, OTHER_IS_RAW> &other)
        requires(NON_OWNING) {

        ptr_ = other.ptr_;
        return *this;
    }

    managed_ptr(managed_ptr &&other) noexcept requires(!NON_OWNING)
            : ptr_(other.ptr_) {
        other.ptr_ = nullptr;
    }

    T* get() const {
        return ptr_;
    }

    T *operator->() {
        ASSERT(ptr_ != nullptr);

        if constexpr (Location == MemLoc::GPU) {
            ASSERT(is_cuda());
        }
        return ptr_;
    }

    T &operator*() {
        ASSERT(ptr_ != nullptr);

        if constexpr (Location == MemLoc::GPU) {
            ASSERT(is_cuda());
        }
        return *ptr_;
    }

    const T* operator->() const {
        ASSERT(ptr_ != nullptr);

        if constexpr (Location == MemLoc::GPU) {
            ASSERT(is_cuda());
        }
        return ptr_;
    }

    const T &operator*() const {
        ASSERT(ptr_ != nullptr);

        if constexpr (Location == MemLoc::GPU) {
            ASSERT(is_cuda());
        }
        return *ptr_;
    }

    bool operator==(std::nullptr_t) const { return ptr_ == nullptr; }

    template<bool OTHER_IS_RAW>
    bool operator==(const managed_ptr<Location, T, OTHER_IS_RAW> &other) const {
        return ptr_ == other.ptr_;
    }


    managed_ptr(const managed_ptr& other) requires(NON_OWNING) = default;
    managed_ptr& operator=(const managed_ptr& other) requires(NON_OWNING) = default;
    managed_ptr(managed_ptr&& other) noexcept requires(NON_OWNING) = default;
    managed_ptr& operator=(managed_ptr&& other) noexcept requires(NON_OWNING) = default;

    managed_ptr(const managed_ptr& other) requires(!NON_OWNING) = delete;
    managed_ptr& operator=(const managed_ptr& other) requires(!NON_OWNING) = delete;

    managed_ptr& operator=(managed_ptr&& other) noexcept requires(!NON_OWNING) {
        if (this != &other) {
            clear();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
};

template <MemLoc Location, typename T,
                    typename... Args>
managed_ptr<Location, T, /*NON_OWNING=*/false> make_managed(Args &&...args) {
static_assert(std::is_trivially_copyable_v<T>,
                            "T must be trivially copyable");


    // Create a CPU copy
    auto cpu_ptr = sa_spec::cpu::make_unique<T>(std::forward<Args>(args)...);

    // Case 1: CPU memory
    if constexpr (Location == MemLoc::CPU) {
        return managed_ptr<Location, T>::own(cpu_ptr.release());
    }

    // Case 2: CUDA memory

    T *ptr_gpu;

    if constexpr (Location == MemLoc::GPU) {
        // Case 3: GPU memory
        cuda::malloc((void **)&ptr_gpu, sizeof(T));
    } else if constexpr (Location == MemLoc::PINNED) {
        // Case 4: pinned memory
        cuda::malloc_pinned((void **)&ptr_gpu, sizeof(T));
    } else {
        ASSERT_THROW(false && "unreachable");
    }

    // Copy in the cpu copy to the cuda memory.
    // TODO: make this zero-copy for pinned memory.
    cuda::memcpy(ptr_gpu, cpu_ptr.get(), sizeof(T),
                            cuda::memcpy_kind::HostToDevice);

    return managed_ptr<Location, T>::own(ptr_gpu);
}


template <MemLoc DstLocation, MemLoc SrcLocation, typename T,
                    bool DstIsRaw, bool SrcIsRaw>
void sparse_copy(const managed_ptr<DstLocation, T, DstIsRaw> &dst,
                 const managed_ptr<SrcLocation, T, SrcIsRaw> &src)
        requires(SrcLocation != MemLoc::GPU || DstLocation != MemLoc::GPU) {

    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    auto copy_chunk = [&](const void *src_chunk, size_t size) {
        auto dst_offset = static_cast<const byte *>(src_chunk)
                              - reinterpret_cast<const byte *>(src.get());
        void *dst_chunk = reinterpret_cast<byte *>(dst.get()) + dst_offset;

        if constexpr (DstLocation == SrcLocation) {
            ::memcpy(dst_chunk, src_chunk, size);
        } else {
            auto kind = DstLocation == MemLoc::GPU ?
                cuda::memcpy_kind::HostToDevice : cuda::memcpy_kind::DeviceToHost;
            cuda::memcpy(dst_chunk, src_chunk, size, kind);
        }
    };

    src.get()->visit_chunks(copy_chunk);
}


template <MemLoc DstLocation, MemLoc SrcLocation, typename T,
                    bool DstIsRaw, bool SrcIsRaw>
void dense_copy(const managed_ptr<DstLocation, T, DstIsRaw> &dst,
                const managed_ptr<SrcLocation, T, SrcIsRaw> &src)
        requires(SrcLocation == MemLoc::GPU) {

    static_assert(std::is_trivially_copyable_v<T>,
                                "T must be trivially copyable");

    auto kind = DstLocation == MemLoc::GPU ?
                cuda::memcpy_kind::DeviceToDevice : cuda::memcpy_kind::DeviceToHost;
    cuda::memcpy(
        static_cast<void *>(dst.get()),
        static_cast<void *>(src.get()),
        sizeof(T),
        kind
    );
}

} // namespace sa_spec
