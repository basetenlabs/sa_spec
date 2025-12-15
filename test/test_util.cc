// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#include <set>

#include <catch2/catch_test_macros.hpp>

#include <sa_spec/util/config.hpp>
#include <sa_spec/util/batch_index_mapper.hpp>
#include <sa_spec/util/flat_graph.hpp>

using namespace sa_spec;

TEST_CASE("lru test") {
    LRU<int> lru;

    for (int i = 0; i < 4; i++) {
        lru.bump(i);
    }

    REQUIRE(lru.pop() == 0);
    lru.bump(1);
    REQUIRE(lru.pop() == 2);
    REQUIRE(lru.pop() == 3);
    REQUIRE(lru.pop() == 1);
}


TEST_CASE("FlatGraph smoke test") {
    FlatGraph<int, int, 30> map;
    auto node_id1 = map.size();
    auto& data1 = map.push_back();
    auto node_id2 = map.size();
    auto& data2 = map.push_back();

    data1 = 10;
    data2 = 20;

    *map.at(node_id1, 42, true) = node_id2;

    REQUIRE(*map.at(node_id1, 42) == node_id2);

    for (int i = 0; i < 10; i++) {
        auto node_id = map.size();
        map.push_back();
        *map.at(node_id1, i, true) = node_id;
    }

    REQUIRE(*map.at(node_id1, 42) == node_id2);

    for (int i = 0; i < 10; i++) {
        REQUIRE(*map.at(node_id1, i) == NodeIndex(2 + i));
    }

    REQUIRE(map.at(node_id1) == 10);
    REQUIRE(map.at(node_id2) == 20);
}

TEST_CASE("FlatGraph clone test") {
    FlatGraph<int, int, 30> map;

    auto node_id1 = map.size();
    map.push_back();
    auto node_id2 = map.size();
    map.push_back();
    auto node_id3 = map.size();
    map.push_back();
    auto node_id4 = map.size();
    map.push_back();


    int edge1 = 42;
    int edge2 = 43;
    int edge3 = 44;
    int edge4 = 45;

    *map.at(node_id1, edge1, true) = node_id2;
    *map.at(node_id1, edge2, true) = node_id3;
    *map.at(node_id1, edge3, true) = node_id4;

    *map.at(node_id2, edge4, true) = node_id1;

    map.at(node_id1) = 10;

    auto node_id5 = map.size();
    map.push_back_clone(node_id1);

    REQUIRE(map.at(node_id5) == 10);
    REQUIRE(*map.at(node_id5, edge1) == node_id2);
    REQUIRE(*map.at(node_id5, edge2) == node_id3);
    REQUIRE(*map.at(node_id5, edge3) == node_id4);

    REQUIRE(map.at(node_id5, edge4) == nullptr);
}


TEST_CASE("decode mapper", "[cpu][batch]") {
    BatchIndexMapper batch_index_mapper;

    std::unordered_map<RequestID, BatchIndex> request_id_to_batch_index;

    std::vector<RequestID> request_ids_1 = {1_req, 2_req, 3_req};
    auto batch_table_1 = batch_index_mapper.prepare(request_ids_1);

    std::vector<RequestID> request_ids_2 = {1_req, 2_req, 3_req};
    auto batch_table_2 = batch_index_mapper.prepare(request_ids_2);
    ASSERT(batch_table_1 == batch_table_2);

    std::vector<RequestID> request_ids_3 = {4_req, 5_req, 6_req};
    auto batch_table_3 = batch_index_mapper.prepare(request_ids_3);
    std::set<BatchIndex> batch_indices;
    for (auto& table : {batch_table_1, batch_table_2, batch_table_3}) {
        for (auto& batch_index : table) {
            batch_indices.insert(batch_index);
        }
    }
    REQUIRE(batch_indices.size() == 6);

    std::vector<RequestID> request_ids_4 = {1_req,  7_req, 2_req};
    auto batch_table_4 = batch_index_mapper.prepare(request_ids_4);
    batch_indices.clear();
    for (auto& table : {batch_table_1, batch_table_2, batch_table_3, batch_table_4}) {
        for (auto& batch_index : table) {
            batch_indices.insert(batch_index);
        }
    }
    REQUIRE(batch_indices.size() == 7);
}
