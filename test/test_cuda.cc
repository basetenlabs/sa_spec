// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#include <catch2/catch_test_macros.hpp>

#include <sa_spec/suffix_automaton.hpp>

#include "utils_for_test.hpp"

using namespace sa_spec;

TEST_CASE("managed_ptr", "[cuda]") {

    auto vecs = {1, 2, 3, 4};
    auto a = make_managed<MemLoc::CPU, Buffer<int, 4>>(vecs);
    auto b = make_managed<MemLoc::GPU, Buffer<int, 4>>();

    sparse_copy(b, a);

    auto c = make_managed<MemLoc::CPU, Buffer<int, 4>>();
    dense_copy(c, b);

    REQUIRE(c->at(0) == 1);
    REQUIRE(c->at(1) == 2);
    REQUIRE(c->at(2) == 3);
    REQUIRE(c->at(3) == 4);
    REQUIRE(*a == *c);

    cuda::test::assign_int(&b.get()->m_data[0], 5);
    cuda::test::assign_int(&b.get()->m_data[1], 6);
    cuda::test::assign_int(&b.get()->m_data[2], 7);
    cuda::test::assign_int(&b.get()->m_data[3], 8);

    dense_copy(a, b);

    REQUIRE(a->at(0) == 5);
    REQUIRE(a->at(1) == 6);
    REQUIRE(a->at(2) == 7);
    REQUIRE(a->at(3) == 8);

    REQUIRE(c->at(0) == 1);
    REQUIRE(c->at(1) == 2);
    REQUIRE(c->at(2) == 3);
    REQUIRE(c->at(3) == 4);
}

TEST_CASE("DynamicBuffer", "[cuda]") {

    auto a = make_managed<MemLoc::CPU, DynamicBuffer<int, 4>>();

    a->push_back(1);
    a->push_back(2);
    a->push_back(3);

    REQUIRE(a->size() == 3);
    REQUIRE(a->at(0) == 1);
    REQUIRE(a->at(1) == 2);
    REQUIRE(a->at(2) == 3);

    a->m_data.m_data[3] = 4;

    auto b = make_managed<MemLoc::CPU, DynamicBuffer<int, 4>>();
    sparse_copy(b, a);

    REQUIRE(b->size() == 3);
    REQUIRE(b->at(0) == 1);
    REQUIRE(b->at(1) == 2);
    REQUIRE(b->at(2) == 3);
    REQUIRE(b->m_data.m_data[3] != 4);
}

TEST_CASE("Nested Buffer sparse copy", "[cuda]") {
    auto a = make_managed<MemLoc::CPU, Buffer<DynamicBuffer<int, 4>, 4>>();

    a->at(0).push_back(1);
    a->at(0).push_back(2);
    a->at(0).push_back(3);

    a->at(0).m_data.m_data[3] = 4;

    a->at(1).push_back(4);
    a->at(1).push_back(5);
    a->at(1).push_back(6);

    auto b = make_managed<MemLoc::GPU, Buffer<DynamicBuffer<int, 4>, 4>>();
    sparse_copy(b, a);

    auto c = make_managed<MemLoc::CPU, Buffer<DynamicBuffer<int, 4>, 4>>();
    dense_copy(c, b);

    REQUIRE(c->at(0).size() == 3);
    REQUIRE(c->at(1).size() == 3);
    REQUIRE(c->at(0).at(0) == 1);
    REQUIRE(c->at(0).at(1) == 2);
    REQUIRE(c->at(0).at(2) == 3);
    REQUIRE(c->at(0).m_data.m_data[3] != 4);
    REQUIRE(c->at(1).at(0) == 4);
    REQUIRE(c->at(1).at(1) == 5);
    REQUIRE(c->at(1).at(2) == 6);
}


TEST_CASE("sparse_copy", "[cuda]") {

    auto a = make_managed<MemLoc::GPU, DynamicBuffer<int, 4>>();

    cuda::test::push_back(a, 1);
    cuda::test::push_back(a, 2);

    auto b = make_managed<MemLoc::CPU, DynamicBuffer<int, 4>>();
    dense_copy(b, a);

    REQUIRE(b->size() == 2);
    REQUIRE(b->at(0) == 1);
    REQUIRE(b->at(1) == 2);

    b->push_back(3);
    b->m_data.m_data[3] = 4;

    sparse_copy(a, b);

    auto c = make_managed<MemLoc::CPU, DynamicBuffer<int, 4>>();
    dense_copy(c, a);

    REQUIRE(c->at(0) == 1);
    REQUIRE(c->at(1) == 2);
    REQUIRE(c->at(2) == 3);

    // verify that the data was copied sparsely
    REQUIRE(c->m_data.m_data[3] != 4);
}

TEST_CASE("SuffixAutomaton copy", "[suffix_automaton][cuda]") {

    auto cpu_copy = make_managed<MemLoc::CPU, SuffixAutomaton>();

    cpu_copy->extend(1_tok);
    cpu_copy->extend(2_tok);
    cpu_copy->extend(3_tok);
    cpu_copy->extend(4_tok);
    cpu_copy->extend(5_tok);

    cuda::sync();

    auto result = cpu_copy->lookup();
    REQUIRE(!result.has_value());

    cuda::sync();

    auto gpu_copy = make_managed<MemLoc::GPU, SuffixAutomaton>();
    sparse_copy(gpu_copy, cpu_copy);
    cuda::sync();

    cuda::test::extend(gpu_copy, 2_tok);
    cuda::test::extend(gpu_copy, 3_tok);

    dense_copy(cpu_copy, gpu_copy);
    cuda::sync();

    result = cpu_copy->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 3);
    REQUIRE(+result->len == 2);

    sparse_copy(gpu_copy, cpu_copy);
    cuda::test::extend(gpu_copy, 10_tok);
    cuda::test::extend(gpu_copy, 11_tok);

    dense_copy(cpu_copy, gpu_copy);
    cpu_copy->extend(10_tok);

    result = cpu_copy->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 8);
    REQUIRE(+result->len == 1);

    sparse_copy(gpu_copy, cpu_copy);


    using TokensT = decltype(gpu_copy->m_tokens);
    auto tokens = make_managed<MemLoc::CPU, TokensT>();
    dense_copy(tokens, managed_ptr_raw<MemLoc::GPU, TokensT>(&gpu_copy.get()->m_tokens));

    REQUIRE(+tokens->size() == 10);
    REQUIRE(tokens->at(TextIndex(0)) == 1_tok);
    REQUIRE(tokens->at(TextIndex(1)) == 2_tok);
    REQUIRE(tokens->at(TextIndex(2)) == 3_tok);
    REQUIRE(tokens->at(TextIndex(3)) == 4_tok);
    REQUIRE(tokens->at(TextIndex(4)) == 5_tok);
    REQUIRE(tokens->at(TextIndex(5)) == 2_tok);
    REQUIRE(tokens->at(TextIndex(6)) == 3_tok);
    REQUIRE(tokens->at(TextIndex(7)) == 10_tok);
    REQUIRE(tokens->at(TextIndex(8)) == 11_tok);
    REQUIRE(tokens->at(TextIndex(9)) == 10_tok);
}


TEST_CASE("Pinned memory", "[cuda]") {
    auto a = make_managed<MemLoc::PINNED, DynamicBuffer<int, 4>>();

    a->push_back(1);
    a->push_back(2);

    cuda::test::push_back(a, 3);
    cuda::test::push_back(a, 4);

    cuda::sync();

    REQUIRE(a->size() == 4);
    REQUIRE(a->at(0) == 1);
    REQUIRE(a->at(1) == 2);
    REQUIRE(a->at(2) == 3);
    REQUIRE(a->at(3) == 4);
}
