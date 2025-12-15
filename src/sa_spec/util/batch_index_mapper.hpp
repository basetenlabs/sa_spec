// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <functional>
#include <unordered_map>
#include <span>
#include <map>
#include <optional>

#include "assert.hpp"
#include "types.hpp"
#include "config.hpp"

namespace sa_spec {

template<typename T>
struct LRU {
    void bump(T val) {
        auto it = reverse_lru_.find(val);
        if (it != reverse_lru_.end()) {
            auto gen = it->second;
            auto it2 = lru_.find(gen);
            ASSERT(it2 != lru_.end());
            lru_.erase(it2);
            reverse_lru_.erase(it);
        }

        auto this_gen = next_gen_++;
        lru_[this_gen] = val;
        reverse_lru_[val] = this_gen;

        _maybe_rotate();
    }

    void _maybe_rotate() {
        if (next_gen_ == std::numeric_limits<size_t>::max()) {
            ASSERT(reverse_lru_.size() == lru_.size());
            size_t offset = reverse_lru_.size();

            decltype(lru_) lru_cpy_;
            decltype(reverse_lru_) reverse_lru_cpy_;

            for (auto& [gen, val] : lru_) {
                lru_cpy_[gen - offset] = val;
            }
            for (auto& [val, gen] : reverse_lru_) {
                reverse_lru_cpy_[val] = gen - offset;
            }

            lru_ = std::move(lru_cpy_);
            reverse_lru_ = std::move(reverse_lru_cpy_);

            next_gen_ = lru_.size();
        }
    }

    std::optional<T> pop() {
        if (lru_.empty()) {
            return std::nullopt;
        }

        auto val = lru_.begin()->second;
        lru_.erase(lru_.begin());
        reverse_lru_.erase(val);
        return val;
    }

    size_t next_gen_ = 0;

    std::map<size_t, T> lru_;
    std::unordered_map<T, size_t> reverse_lru_;
};


using GenerationStartCB = std::function<void(RequestID, BatchIndex)>;
using RequestEviectedCB = std::function<void(RequestID, BatchIndex)>;

struct BatchIndexMapper {
    explicit BatchIndexMapper(GenerationStartCB &&generation_start_cb=nullptr,
                               RequestEviectedCB &&request_evicted_cb=nullptr)
        : m_generation_start_cb(std::move(generation_start_cb)),
          m_request_evicted_cb(std::move(request_evicted_cb)) {}
    
    BatchIndex get_batch_index_for_test(RequestID request_id) const {
        ASSERT_THROW(m_mapping.contains(request_id));
        return m_mapping.at(request_id);
    }
    
    BatchIndex prepare(RequestID request_id) {
        m_lru.bump(request_id);
        _ensure_request_prepare(request_id);
        return m_mapping[request_id];
    }

    std::vector<BatchIndex> prepare(std::span<const RequestID> request_ids) {
        for (auto request_id : request_ids) {
            m_lru.bump(request_id);
        }

        for (auto request_id : request_ids) {
            _ensure_request_prepare(request_id);
        }

        return _build_batch_table(request_ids);
    }

    void _ensure_request_prepare(RequestID request_id) {
        if (!m_mapping.contains(request_id)) {

            if (m_rem_slots > 0) {
                m_rem_slots--;
                m_mapping[request_id] = BatchIndex(m_rem_slots);
            } else {
                auto evict_request_id = m_lru.pop();
                ASSERT(evict_request_id.has_value());
                ASSERT(m_mapping.contains(evict_request_id.value()));

                auto batch_index = m_mapping.extract(evict_request_id.value()).mapped();

                if (m_request_evicted_cb) {
                    m_request_evicted_cb(evict_request_id.value(), batch_index);
                }

                m_mapping[request_id] = batch_index;
            }

            if (m_generation_start_cb) {

                m_generation_start_cb(request_id, m_mapping[request_id]);
            }
        }
    }

    std::vector<BatchIndex> _build_batch_table(std::span<const RequestID> request_ids) const {
        std::vector<BatchIndex> batch_table;
        batch_table.reserve(request_ids.size());
        for (size_t i = 0; i < request_ids.size(); ++i) {
            RequestID request_id = request_ids[i];
            ASSERT(m_mapping.contains(request_id));
            batch_table.push_back(m_mapping.at(request_id));
        }
        return batch_table;
    }

    LRU<RequestID> m_lru;
    std::unordered_map<RequestID, BatchIndex> m_mapping;

    GenerationStartCB m_generation_start_cb;
    RequestEviectedCB m_request_evicted_cb;

    size_t m_rem_slots = Config::MAX_SLOTS;
};

} // namespace sa_spec
