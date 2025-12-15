// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <cassert>
#include <cstddef>
#include <cstring>
#include <span>
#include <unordered_map>

#include <sa_spec/util/config.hpp>
#include <sa_spec/util/flat_graph.hpp>
#include <sa_spec/util/named_type.hpp>
#include <sa_spec/util/optional.hpp>
#include <sa_spec/util/logger.hpp>
#include <sa_spec/util/types.hpp>

namespace sa_spec {

using TextIndex = NamedType<int, struct TextIndexTag>;

struct SuffixAutomaton {

    struct NodeData {
        int len{0};
        optional<NodeIndex> link;
        optional<TextIndex> pos;
    };

    using Graph = FlatGraph<Token, NodeData, 2 * Config::MAX_SEQUENCE_LENGTH>;

    using TokenVec = DynamicBuffer<Token, Config::MAX_SEQUENCE_LENGTH, TextIndex>;

    Graph m_states;
    TokenVec m_tokens;
    NodeIndex m_last;


    void visit_chunks(auto&& func) const {
        m_states.visit_chunks(func);
        m_tokens.visit_chunks(func);
        func((void*)&m_last, sizeof(m_last));
    }

    SuffixAutomaton() {
        LOG_TRACE("SuffixAutomaton constructor with size %lu", sizeof(SuffixAutomaton));
    }

    // prevent accidental move construction
    SuffixAutomaton(SuffixAutomaton&& other) = delete;

    void clear() {
        m_states.clear();
        m_tokens.clear();
        m_last = NodeIndex(0);
    }

    CUDA_CALLABLE void extend(Token token) {
        if (m_states.empty()) {
            // add root state
            NodeIndex root = m_states.size();
            NodeData& root_data = m_states.push_back();
            root_data = {};
            m_last = root;
        }

        m_tokens.push_back(token);

        NodeIndex cur = m_states.size();
        NodeData& cur_data = m_states.push_back();
        cur_data = NodeData{
            .len = +m_states.at(m_last).len + 1,
            .link = {},
            .pos = TextIndex(+m_tokens.size() - 1),
        };

        auto p_opt = optional<NodeIndex>(m_last);
        while (p_opt.has_value() && m_states.at(*p_opt, token) == nullptr) {
            *m_states.at(*p_opt, token, true) = cur;
            p_opt = m_states.at(*p_opt).link;
        }

        if (!p_opt.has_value()) {
            cur_data.link = NodeIndex(0);
        } else {
            NodeIndex p = p_opt.value();

            NodeIndex* q_ptr = m_states.at(p, token);
            ASSERT(q_ptr != nullptr);
            NodeIndex q = *q_ptr;

            auto &[p_len, p_link, p_pos] = m_states.at(p);

            if (p_len + 1 == m_states.at(q).len) {
                cur_data.link = q;
            } else {
                NodeIndex clone = m_states.size();
                auto& clone_data = m_states.push_back_clone(q);
                clone_data.len = p_len + 1;

                auto p_current_opt = optional<NodeIndex>(p);
                while (p_current_opt.has_value()) {
                    NodeIndex* edge_ptr = m_states.at(*p_current_opt, token);
                    if (edge_ptr == nullptr || *edge_ptr != q) {
                        break;
                    }
                    *edge_ptr = clone;
                    p_current_opt = m_states.at(*p_current_opt).link;
                }

                m_states.at(q).link = clone;
                m_states.at(cur).link = clone;
            }
        }
        m_last = cur;
    }

    struct LookupResult {
        TextIndex pos{0};
        int len{0};
    };

    CUDA_CALLABLE optional<LookupResult> lookup() const {
        if (m_states.empty()) {
            return {};
        }

        NodeIndex state = m_last;
        NodeIndex best_state = NodeIndex(0);

        // Walk up the suffix links to find the longest proper suffix that appears earlier
        while (state != NodeIndex(0)) {
            auto& node_data = m_states.at(state);
            optional<TextIndex> pos_opt = node_data.pos;
            ASSERT(!pos_opt.has_value() || +*pos_opt <= +m_tokens.size());

            bool is_last = pos_opt.has_value() && +*pos_opt + 1 >= +m_tokens.size();

            if (!is_last) {
                best_state = state;
                break;
            }

            auto link_opt = node_data.link;
            if (!link_opt.has_value()) {
                break;
            }
            state = *link_opt;
        }

        if (best_state == NodeIndex(0)) {
            return {};
        }

        auto pos_opt = m_states.at(best_state).pos;
        if (!pos_opt.has_value()) {
            return {};
        }

        ASSERT(*pos_opt < m_tokens.size());

        // Return the position after the suffix match
        TextIndex match_end = TextIndex(+*pos_opt + 1);

        return LookupResult{
            .pos = match_end,
            .len = m_states.at(best_state).len,
        };
    }

    CUDA_CALLABLE void get_draft_tokens(
        Token::ValueType* buf,
        int buf_len,
        TextIndex start_pos) const {

        buf_len = std::min<int>(buf_len, +m_tokens.size() - +start_pos);
        for (int i = 0; i < buf_len; i++) {
            buf[i] = +m_tokens.at(TextIndex(+start_pos + i));
        }
    }
};

static_assert(std::is_trivially_copyable_v<SuffixAutomaton>);

} // namespace sa_spec
