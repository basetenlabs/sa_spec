// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#include <vector>
#include <iostream>

#include <catch2/catch_test_macros.hpp>

#include <sa_spec/util/memory.hpp>
#include <sa_spec/suffix_automaton.hpp>

using namespace sa_spec;

TEST_CASE("SuffixAutomaton Empty", "[suffix_automaton]") {

    auto a = make_managed<MemLoc::CPU, SuffixAutomaton>();

    REQUIRE(!a->lookup().has_value());

    a->extend(1_tok);
    REQUIRE(!a->lookup().has_value());
}

TEST_CASE("SuffixAutomaton Simple Repetition", "[suffix_automaton]") {

    auto a = make_managed<MemLoc::CPU, SuffixAutomaton>();

    a->extend(1_tok);
    a->extend(1_tok);
    a->extend(1_tok);

    auto result = a->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 2);
    REQUIRE(+result->len == 2);

    a->extend(1_tok);
    result = a->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 3);
    REQUIRE(+result->len == 3);
}

TEST_CASE("SuffixAutomaton Alternating", "[suffix_automaton]") {

    auto a = make_managed<MemLoc::CPU, SuffixAutomaton>();

    a->extend(1_tok);
    a->extend(2_tok);
    auto result = a->lookup();
    REQUIRE(!result.has_value());

    a->extend(1_tok);
    result = a->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 1);
    REQUIRE(+result->len == 1);

    a->extend(2_tok);
    result = a->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 2);
    REQUIRE(+result->len == 2);
}

TEST_CASE("SuffixAutomaton Overlapping Repeats", "[suffix_automaton]") {

    auto a = make_managed<MemLoc::CPU, SuffixAutomaton>();

    // "ababa"
    a->extend(1_tok);
    a->extend(2_tok);
    a->extend(1_tok);
    a->extend(2_tok);
    a->extend(1_tok);

    auto result = a->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 3);
    REQUIRE(+result->len == 3);

    // "aaaa"
    auto b = make_managed<MemLoc::CPU, SuffixAutomaton>();
    b->extend(1_tok);
    b->extend(1_tok);
    b->extend(1_tok);
    b->extend(1_tok);

    result = b->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 3);
    REQUIRE(+result->len == 3);
}

TEST_CASE("SuffixAutomaton Substring Reuse", "[suffix_automaton]") {

    auto a = make_managed<MemLoc::CPU, SuffixAutomaton>();

    // "abcab"
    a->extend(1_tok);
    a->extend(2_tok);
    a->extend(3_tok);
    a->extend(1_tok);
    a->extend(2_tok);

    auto result = a->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 2);
    REQUIRE(+result->len == 2);

    // "abcdabc"
    auto b = make_managed<MemLoc::CPU, SuffixAutomaton>();
    b->extend(1_tok);
    b->extend(2_tok);
    b->extend(3_tok);
    b->extend(4_tok);
    b->extend(1_tok);
    b->extend(2_tok);
    b->extend(3_tok);

    result = b->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 3);
    REQUIRE(+result->len == 3);
}

TEST_CASE("SuffixAutomaton Palindrome", "[suffix_automaton]") {

    auto a = make_managed<MemLoc::CPU, SuffixAutomaton>();

    // "racecar"
    a->extend(1_tok);
    a->extend(2_tok);
    a->extend(3_tok);
    a->extend(4_tok);
    a->extend(3_tok);
    a->extend(2_tok);
    a->extend(1_tok);

    auto result = a->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 1);
    REQUIRE(+result->len == 1);
}

TEST_CASE("SuffixAutomaton Periodic String", "[suffix_automaton]") {

    auto a = make_managed<MemLoc::CPU, SuffixAutomaton>();

    // "abababab"
    a->extend(1_tok);
    a->extend(2_tok);
    a->extend(1_tok);
    a->extend(2_tok);
    a->extend(1_tok);
    a->extend(2_tok);
    a->extend(1_tok);
    a->extend(2_tok);

    auto result = a->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 6);
    REQUIRE(+result->len == 6);
}

TEST_CASE("SuffixAutomaton Unique String", "[suffix_automaton]") {

    auto a = make_managed<MemLoc::CPU, SuffixAutomaton>();

    // "abcdef"
    a->extend(1_tok);
    a->extend(2_tok);
    a->extend(3_tok);
    a->extend(4_tok);
    a->extend(5_tok);
    a->extend(6_tok);

    auto result = a->lookup();
    REQUIRE(!result.has_value());
}

TEST_CASE("SuffixAutomaton needle in haystack", "[suffix_automaton]") {

    auto a = make_managed<MemLoc::CPU, SuffixAutomaton>();

    // "abcab123456abc12"
    a->extend(11_tok);
    a->extend(22_tok);
    a->extend(33_tok);
    a->extend(11_tok);
    a->extend(22_tok);
    a->extend(1_tok);
    a->extend(2_tok);
    a->extend(3_tok);
    a->extend(4_tok);
    a->extend(5_tok);
    a->extend(6_tok);
    a->extend(11_tok);
    a->extend(22_tok);
    a->extend(33_tok);
    a->extend(1_tok);
    a->extend(2_tok);

    auto result = a->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->pos == 7);
    REQUIRE(+result->len == 2);
}

TEST_CASE("SuffixAutomaton Large", "[suffix_automaton]") {

    auto a = make_managed<MemLoc::CPU, SuffixAutomaton>();

    // "a" * 1000000
    for (int i = 0; i < 100000; i++) {
        a->extend(1_tok);
    }

    auto result = a->lookup();
    REQUIRE(result.has_value());
    REQUIRE(+result->len == 99999);
}

TEST_CASE("SuffixAutomaton Large with repeats", "[suffix_automaton]") {

    auto a = make_managed<MemLoc::CPU, SuffixAutomaton>();

    int rem_capacity = 100000;
    int cap = 0;
    int last = 0;

    // "a" * 1000000
    for (int i = 0; i < 100000; i++) {
        if (rem_capacity == 0) {
            break;
        }

        if (last == cap && cap > 0) {
            auto result = a->lookup();
            REQUIRE(result.has_value());
            REQUIRE(+result->len == cap);
        }

        a->extend(Token(last));

        if (last == cap) {
            cap++;
            last = 0;
        } else {
            last++;
        }
    }
}
