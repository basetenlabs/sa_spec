// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#include <thread>
#include <map>
#include <unordered_set>
#include <unordered_map>

#include <sa_spec/api.hpp>
#include <sa_spec/util/lock_release_guard.hpp>

#include <sa_spec/suffix_automaton.hpp>
#include <sa_spec/util/batch_index_mapper.hpp>
#include <sa_spec/util/buffer.hpp>
#include <sa_spec/util/memory.hpp>
#include <sa_spec/util/config.hpp>

namespace sa_spec {

struct APIImpl {
    static APIImpl& get_instance() {
        static APIImpl instance;
        return instance;
    }

    using BatchIndices = Buffer<BatchIndex, Config::MAX_SLOTS>;
    using Workspace = Buffer<SuffixAutomaton, Config::MAX_SLOTS, BatchIndex>;
    using PinnedSuffixAutomaton = managed_ptr<MemLoc::PINNED, SuffixAutomaton>;

    // gpu
    const managed_ptr<MemLoc::GPU, Workspace>
        m_gpu_slots = make_managed<MemLoc::GPU, Workspace>();

    const managed_ptr<MemLoc::GPU, BatchIndices>
        m_gpu_batch_indices = make_managed<MemLoc::GPU, BatchIndices>();


    // pinned
    std::unordered_map<RequestID, PinnedSuffixAutomaton> m_used_host_slots;

    std::vector<PinnedSuffixAutomaton> m_free_host_slots;

    managed_ptr<MemLoc::PINNED, BatchIndices>
        m_pinned_batch_indices = make_managed<MemLoc::PINNED, BatchIndices>();

    // host
    BatchIndexMapper m_batch_index_mapper{
        /*generation_start_cb=*/[this](RequestID request_id, BatchIndex batch_index) {

            LOG_TRACE("generation_start_cb request_id= %ld batch_index= %d", +request_id, +batch_index);

            do {
                auto it = m_used_host_slots.find(request_id);
                if (it == m_used_host_slots.end()) {
                    LOG_INFO("prepare() on request_id=%ld which was not added. "
                                "Using an empty slot.", +request_id);
                    m_used_host_slots[request_id]
                        = make_managed<MemLoc::PINNED, SuffixAutomaton>();
                    m_used_host_slots[request_id]->clear();
                    break;
                }
 
                if (it->second != nullptr) {
                    break;
                }

                auto lck = scoped_gil_release();
                std::this_thread::yield();
            } while (true);

            auto it = m_used_host_slots.find(request_id);

            LOG_TRACE("copying request_id=%ld batch_index=%d tokens.size()=%d", +request_id, +batch_index, +it->second->m_tokens.size());

            managed_ptr_raw<MemLoc::GPU, SuffixAutomaton> dst(
                &const_cast<SuffixAutomaton&>(m_gpu_slots.get()->at(batch_index)));

            sparse_copy(dst, it->second);
        },
        /*request_evicted_cb=*/[this](RequestID request_id, BatchIndex batch_index) {
            (void)batch_index;
            LOG_TRACE("request_evicted_cb request_id=%ld batch_index=%d", +request_id, +batch_index);

           auto node = m_used_host_slots.extract(request_id);
            m_free_host_slots.push_back(std::move(node.mapped()));
        }
    };

    PinnedSuffixAutomaton _alloc_host_slot() {
        if (m_free_host_slots.empty()) {
            m_free_host_slots.emplace_back(
                make_managed<MemLoc::PINNED, SuffixAutomaton>());
        }
        auto slot = std::move(m_free_host_slots.back());
        m_free_host_slots.pop_back();
        slot->clear();
        return slot;
    }

    void add_request(RequestID request_id, std::span<const Token> tokens) {
        ASSERT_THROW(!m_used_host_slots.contains(request_id));

        auto& prepare_slot = m_used_host_slots[request_id];
        ASSERT_THROW(prepare_slot == nullptr);

        auto host_slot = _alloc_host_slot();

        {
            auto lck = scoped_gil_release();
            for (auto token : tokens) {
                host_slot->extend(token);
            }
        }

        prepare_slot = std::move(host_slot);
    }

    void prepare(std::span<const RequestID> request_ids) {
        if (request_ids.size() > Config::MAX_SLOTS) {
            LOG_INFO("prepare() called with request_ids.size()=%ld > %ld. Dropping some requests.",
                      +request_ids.size(), +Config::MAX_SLOTS);
            request_ids = request_ids.subspan(0, Config::MAX_SLOTS);
        }

        std::vector<BatchIndex> batch_indices = m_batch_index_mapper.prepare(request_ids);

        for (size_t i = 0; i < request_ids.size(); ++i) {
            m_pinned_batch_indices->at(i) = batch_indices[i];
        }

        sparse_copy(m_gpu_batch_indices, m_pinned_batch_indices);
    }

    void extend(int batch_size,
                int draft_length,
                NumTokens::ValueType* depth_out,
                Token::ValueType* draft_out,
                const Token::ValueType* accepted_in,
                const NumTokens::ValueType* accepted_lens_in) {

        batch_size = std::min<int>(batch_size, Config::MAX_SLOTS);

        cuda::invoke_extend(
            batch_size,
            draft_length,
            &m_gpu_slots.get()->m_data[0],
            reinterpret_cast<const BatchIndex::ValueType*>(&m_gpu_batch_indices.get()->m_data[0]),
            depth_out,
            draft_out,
            accepted_in,
            accepted_lens_in);
    }

    void get_active_tokens_for_test(RequestID request_id,
                                    Token::ValueType* tokens_out,
                                    size_t tokens_out_len) {

        BatchIndex batch_index = m_batch_index_mapper.get_batch_index_for_test(request_id);

        cuda::memcpy(tokens_out,
                     &m_gpu_slots.get()->at(batch_index).m_tokens.m_data.at(TextIndex(0)),
                     tokens_out_len * sizeof(Token::ValueType),
                     cuda::memcpy_kind::DeviceToDevice);
    }

private:

    APIImpl() = default;
};

API& API::get_instance() {
    static API instance;
    return instance;
}

void API::add_request(RequestID request_id, std::span<const Token> tokens) {
    APIImpl::get_instance().add_request(request_id, tokens);
}

void API::prepare(std::span<const RequestID> request_ids) {
    APIImpl::get_instance().prepare(request_ids);
}

void API::extend(int batch_size,
                 int draft_length,
                 NumTokens::ValueType* depth_out,
                 Token::ValueType* draft_out,
                 const Token::ValueType* accepted_in,
                 const NumTokens::ValueType* accepted_lens_in) {

    APIImpl::get_instance().extend(
        batch_size,
        draft_length,
        depth_out,
        draft_out,
        accepted_in,
        accepted_lens_in);
}


void API::get_active_tokens_for_test(RequestID request_id,
                                     Token::ValueType* tokens_out,
                                     size_t tokens_out_len) {
    APIImpl::get_instance().get_active_tokens_for_test(request_id, tokens_out, tokens_out_len);
}

} // namespace sa_spec
