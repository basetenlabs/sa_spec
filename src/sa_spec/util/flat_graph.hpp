// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include "flat_hash_map.hpp"

namespace sa_spec {


using NodeIndex = NamedType<int, struct NodeIndexTag>;

template<typename Key, typename NodeData, size_t MaxSize>
struct FlatGraph {

    using Allocator = HashMapAllocator<
        Key,
        NodeIndex,
        10 * MaxSize * (sizeof(Key) + sizeof(NodeIndex))>;

    using HashMap = Allocator::HashMap;

    struct Node {

        NodeData data;

        bool m_is_empty = true;
        bool m_is_inlined = false;

        Key m_inlined_key;
        NodeIndex m_inlined_node_index;
        Allocator::Ptr edges_ptr;

        CUDA_CALLABLE bool is_empty() {
            return m_is_empty;
        }

        CUDA_CALLABLE bool is_inlined() {
            return m_is_inlined;
        }

        CUDA_CALLABLE void set_inlined_edge(std::pair<Key, NodeIndex> edge) {
            m_is_empty = false;
            m_is_inlined = true;
            m_inlined_key = edge.first;
            m_inlined_node_index = edge.second;
        }

        CUDA_CALLABLE void set_edges_ptr(Allocator::Ptr ptr) {
            m_is_empty = false;
            m_is_inlined = false;
            edges_ptr = ptr;
        }

        static_assert(std::is_trivially_copyable_v<NodeData>);
        static_assert(std::is_trivially_copyable_v<typename Allocator::Ptr>);
        static_assert(std::is_trivially_copyable_v<NodeIndex>);
    };

    static_assert(std::is_trivially_copyable_v<Node>);

    DynamicBuffer<Node, MaxSize, NodeIndex> m_nodes;
    Allocator m_allocator;

    static_assert(std::is_trivially_copyable_v<decltype(m_nodes)>);
    static_assert(std::is_trivially_copyable_v<decltype(m_allocator)>);

    void visit_chunks(auto&& func) const {
        m_nodes.visit_chunks(func);
        m_allocator.visit_chunks(func);
    }

    CUDA_CALLABLE NodeIndex size() const {
        return m_nodes.size();
    }

    CUDA_CALLABLE bool empty() const {
        return m_nodes.empty();
    }

    CUDA_CALLABLE NodeIndex* at(NodeIndex node_id,
                                Key key,
                                bool insert_if_not_exists = false) {

        ASSERT(node_id < m_nodes.size());

        auto& node = m_nodes.at(node_id);

        if (node.is_empty()) {
            if (insert_if_not_exists) {
                node.set_inlined_edge(std::make_pair(key, NodeIndex(0)));
                return &node.m_inlined_node_index;
            } else {
                return nullptr;
            }
        }

        if (node.is_inlined()) {
            if (node.m_inlined_key == key) {
                return &node.m_inlined_node_index;
            }

            if (!insert_if_not_exists) {
                return nullptr;
            }

            auto new_edges_ptr = m_allocator.alloc(6);
            auto& new_edges = m_allocator.at(new_edges_ptr);
            
            auto* val_ptr = new_edges.at(node.m_inlined_key, true);
            ASSERT(val_ptr != nullptr);
            *val_ptr = node.m_inlined_node_index;

            node.set_edges_ptr(new_edges_ptr);
        }

        typename Allocator::Ptr& edges_ptr = node.edges_ptr;
        HashMap& edges = m_allocator.at(edges_ptr);

        NodeIndex* res = nullptr;

        // lookup the key in the hashmap
        res = edges.at(key, insert_if_not_exists);


        // if failed to insert, we need to size up
        if (res == nullptr && insert_if_not_exists) {
            // create a larger hashmap

            // todo: tweak the exponent
            size_t new_capacity = edges.capacity() * 5;

            auto new_edges_ptr = m_allocator.alloc(new_capacity);
            HashMap& new_edges = m_allocator.at(new_edges_ptr);

            // copy to the new hashmap
            for (auto [k, node_idx] : edges) {
                auto* val_ptr = new_edges.at(k, true);
                // must have space
                ASSERT(val_ptr != nullptr);
                *val_ptr = node_idx;
            }

            // insert the new key
            res = new_edges.at(key, true);
            ASSERT(res != nullptr);

            // free the old hashmap
            m_allocator.free(edges_ptr);

            // set pointer to the new hashmap
            edges_ptr = new_edges_ptr;
        }

        return res;
    }

    CUDA_CALLABLE NodeData& at(NodeIndex node_id) {
        return m_nodes.at(node_id).data;
    }

    CUDA_CALLABLE const NodeData& at(NodeIndex node_id) const {
        return m_nodes.at(node_id).data;
    }

    CUDA_CALLABLE NodeData& push_back() {
        // todo: inline the base size of the hashmap
        return m_nodes.push_back(Node{}).data;
    }

    CUDA_CALLABLE NodeData& push_back_clone(NodeIndex old_node_id) {
        ASSERT(old_node_id < m_nodes.size());

        auto& old_node = m_nodes.at(old_node_id);

        if (old_node.is_empty() || old_node.is_inlined()) {
            return m_nodes.push_back(old_node).data;
        }

        auto& old_edges = m_allocator.at(old_node.edges_ptr);

        // allocate hashmap with equal size
        typename Allocator::Ptr new_edges_ptr = m_allocator.alloc(
            old_edges.capacity());

        // copy data
        m_allocator.at(new_edges_ptr) = old_edges;

        Node new_node;
        new_node.data = old_node.data;
        new_node.set_edges_ptr(new_edges_ptr);

        return m_nodes.push_back(std::move(new_node)).data;
    }

    CUDA_CALLABLE void clear() {
        m_nodes.clear();
        m_allocator.clear();
    }
};

} // namespace sa_spec
