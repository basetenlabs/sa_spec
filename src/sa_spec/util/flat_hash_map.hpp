// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <utility>
#include <cstdint>

#include "buffer.hpp"
#include "logger.hpp"
#include "named_type.hpp"
#include "cuda_callable.hpp"

namespace sa_spec {

template<typename Key, typename Value>
struct FlatHashMap {

    struct Bucket {
        Key key;
        Value value;

        // todo: move occupied to a bitset in FlatHashMap
        bool occupied = false;
    };

    const size_t m_capacity = 0;


    CUDA_CALLABLE void clear() {
        std::memset(static_cast<void*>(data()), 0, sizeof(Bucket) * m_capacity);
    }


    CUDA_CALLABLE explicit FlatHashMap(size_t capacity) : m_capacity(capacity) {
        clear();
    }

    CUDA_CALLABLE FlatHashMap& operator=(const FlatHashMap& other) {
        ASSERT(m_capacity == other.m_capacity);
        std::memcpy(static_cast<void*>(data()),
                    static_cast<const void*>(other.data()),
                    sizeof(Bucket) * m_capacity);
        return *this;
    }

    static constexpr size_t size_bytes(size_t capacity) {
        return sizeof(size_t) + sizeof(Bucket) * capacity;
    }

    CUDA_CALLABLE size_t size_bytes() const {
        return size_bytes(m_capacity);
    }

    CUDA_CALLABLE size_t capacity() const {
        return m_capacity;
    }

    CUDA_CALLABLE Bucket* data() {
        return reinterpret_cast<Bucket*>(reinterpret_cast<char*>(this) + sizeof(size_t));
    }

    CUDA_CALLABLE const Bucket* data() const {
        return reinterpret_cast<const Bucket*>(reinterpret_cast<const char*>(this) + sizeof(size_t));
    }

    CUDA_CALLABLE size_t hash(Key key) const {
        return static_cast<size_t>(+key) % m_capacity;
    }

    // returns nullptr if the key is not found or no space to insert
    CUDA_CALLABLE Value* at(Key key, bool insert_if_not_exists = false) {
        size_t index = hash(key);
        size_t original_index = index;
        
        // Linear probe to find the key or an empty slot
        do {
            Bucket& bucket = data()[index];
            
            if (!bucket.occupied) {
                // Empty slot found
                if (insert_if_not_exists) {
                    // Insert new key-value pair
                    bucket.key = key;
                    bucket.value = Value{};
                    bucket.occupied = true;
                    return &bucket.value;
                } else {
                    // Key not found
                    return nullptr;
                }
            } else if (bucket.key == key) {
                // Key found
                return &bucket.value;
            }
            
            // Move to next slot (linear probe)
            index = (index + 1) % m_capacity;
            
        } while (index != original_index);
 
        return nullptr;
    }

    CUDA_CALLABLE const Value* at(Key key) const {
        return const_cast<FlatHashMap*>(this)->at(key);
    }

    struct Iterator {
        const FlatHashMap* map;
        size_t index;

        CUDA_CALLABLE Iterator(const FlatHashMap* map, size_t index) : map(map), index(index) {
            // Skip to first occupied bucket
            while (this->index < map->m_capacity && !map->data()[this->index].occupied) {
                this->index++;
            }
        }

        CUDA_CALLABLE std::pair<const Key&, Value&> operator*() const {
            const Bucket& bucket = map->data()[index];
            return {bucket.key, const_cast<Value&>(bucket.value)};
        }

        CUDA_CALLABLE Iterator& operator++() {
            index++;
            // Skip to next occupied bucket
            while (index < map->m_capacity && !map->data()[index].occupied) {
                index++;
            }
            return *this;
        }

        CUDA_CALLABLE bool operator==(const Iterator& other) const {
            return index == other.index;
        }

        CUDA_CALLABLE bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    CUDA_CALLABLE Iterator begin() const {
        return Iterator(this, 0);
    }

    CUDA_CALLABLE Iterator end() const {
        return Iterator(this, m_capacity);
    }
};


template<typename Key, typename Value, size_t MaxSizeBytes>
struct HashMapAllocator {

    using HashMap = FlatHashMap<Key, Value>;

    using Ptr = NamedType<size_t, struct HashMapPtrTag>;

    
    CUDA_CALLABLE HashMap& at(Ptr ptr) {
        return reinterpret_cast<HashMap&>(m_mem.at(ptr));
    }


    CUDA_CALLABLE Ptr alloc(size_t capacity) {
        // no available pointers, allocate a new one
        auto index = m_mem.size();

        m_mem.extend(HashMap::size_bytes(capacity));

        new (&at(index)) HashMap(capacity);
        at(index).clear();

        return index;
    }

    CUDA_CALLABLE void free(Ptr) {
        // todo
    }

    void visit_chunks(auto&& func) const {
        m_mem.visit_chunks(func);
    }

    CUDA_CALLABLE void clear() {
        m_mem.clear();
    }


    DynamicBuffer<char, MaxSizeBytes, Ptr> m_mem;
};

} // namespace sa_spec
