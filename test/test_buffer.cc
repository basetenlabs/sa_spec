// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#include <memory>
#include <span>

#include <catch2/catch_test_macros.hpp>

#include <sa_spec/util/buffer.hpp>
#include <sa_spec/util/named_type.hpp>

using namespace sa_spec;

TEST_CASE("Buffer smoke test") {
    auto buffer = Buffer<int, 3>();
    buffer.at(0) = 1;
    buffer.at(1) = 2;
    buffer.at(2) = 3;

    REQUIRE(buffer.at(0) == 1);
    REQUIRE(buffer.at(1) == 2);
    REQUIRE(buffer.at(2) == 3);
}

TEST_CASE("Buffer2D int smoke Test") {
    auto buffer = Buffer2D<int, 3, 2>();

    buffer.at(0, 0) = 1;
    buffer.at(0, 1) = 2;
    buffer.at(1, 0) = 3;
    buffer.at(1, 1) = 4;
    buffer.at(2, 0) = 5;
    buffer.at(2, 1) = 6;

    REQUIRE(buffer.at(0, 0) == 1);
    REQUIRE(buffer.at(0, 1) == 2);
    REQUIRE(buffer.at(1, 0) == 3);
    REQUIRE(buffer.at(1, 1) == 4);
    REQUIRE(buffer.at(2, 0) == 5);
    REQUIRE(buffer.at(2, 1) == 6);
}

TEST_CASE("Buffer2D float smoke test") {
    auto buffer = Buffer2D<float, 3, 2>();

    buffer.at(0, 0) = 1.0f;
    buffer.at(0, 1) = 2.0f;

    REQUIRE(buffer.at(0, 0) == 1.0f);
    REQUIRE(buffer.at(0, 1) == 2.0f);
}

TEST_CASE("Buffer2D copy test") {
    auto buffer1 = Buffer2D<int, 3, 2>();

    buffer1.at(0, 0) = 1;
    buffer1.at(0, 1) = 2;
    buffer1.at(1, 0) = 3;
    buffer1.at(1, 1) = 4;
    buffer1.at(2, 0) = 5;
    buffer1.at(2, 1) = 6;

    auto buffer2 = buffer1;

    REQUIRE(buffer2.at(0, 0) == 1);
    REQUIRE(buffer2.at(0, 1) == 2);
    REQUIRE(buffer2.at(1, 0) == 3);
    REQUIRE(buffer2.at(1, 1) == 4);
    REQUIRE(buffer2.at(2, 0) == 5);
    REQUIRE(buffer2.at(2, 1) == 6);
}

TEST_CASE("DynamicBuffer smoke test") {
    auto buffer = DynamicBuffer<int, 10>();

    REQUIRE(buffer.size() == 0);

    buffer.push_back(1);
    buffer.push_back(2);
    buffer.push_back(3);

    REQUIRE(buffer.size() == 3);
    REQUIRE(buffer.at(0) == 1);
    REQUIRE(buffer.at(1) == 2);
    REQUIRE(buffer.at(2) == 3);
}


TEST_CASE("named_type reinterpret cast") {
    using my_type = NamedType<int, struct MyTypeTag>;
    auto buffer = Buffer<my_type, 3>();
    buffer.at(0) = my_type(1);
    buffer.at(1) = my_type(2);
    buffer.at(2) = my_type(3);

    my_type::ValueType* values = reinterpret_cast<my_type::ValueType*>(&buffer.at(0));
    for (size_t i = 0; i < buffer.size(); i++) {
        REQUIRE(values[i] == +buffer.at(i));
    }
}

TEST_CASE("named_type reinterpret cast 2") {
    using my_type = NamedType<int, struct MyTypeTag>;
    std::vector<my_type::ValueType> values;
    values.push_back(1);
    values.push_back(2);
    values.push_back(3);

    std::span<const my_type> values_span(reinterpret_cast<const my_type*>(values.data()), values.size());
    for (size_t i = 0; i < values_span.size(); i++) {
        REQUIRE(+values_span[i] == values[i]);
    }
}
