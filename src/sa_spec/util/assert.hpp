// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <cstdio>
#include <cassert>

#include "logger.hpp"

#ifdef __CUDACC__
    #define _abort() assert(0)
    #define DEBUG_BREAK()
#else
    #include <csignal>
    #define _abort() std::exit(1)
    #define DEBUG_BREAK() std::raise(SIGINT);
#endif

#define ASSERT_IMPL(...)                                                                           \
    do {                                                                                           \
        if (!(__VA_ARGS__)) [[unlikely]] {                                                         \
            LOG_INFO("Assertion failed: %s", #__VA_ARGS__);                                          \
            MAYBE_PRINT_STACK_TRACE; \
            DEBUG_BREAK(); \
            _abort();                                                                              \
        }                                                                                          \
    } while (false)

// Helper: extracts first arg as fmt, rest as format args when provided
#define ASSERT_THROW_GET_FMT(first, ...) first
#define ASSERT_THROW_GET_ARGS(first, ...) __VA_ARGS__

// ASSERT_THROW(cond) or ASSERT_THROW(cond, fmt, ...)
// Supports format strings: ASSERT_THROW(false, "test %d %d", 1, 2)
// When format args provided: constructs format string by concatenating string literals
// When no format args: uses base message only
#define ASSERT_THROW(cond, ...)                                                                    \
    ASSERT_IMPL(cond __VA_OPT__(,) __VA_ARGS__) \

#define CUDA_CHECK_ERROR(error)                                                                     \
    do { \
        if (error != cudaSuccess) { \
            printf("[%s:%d] CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            assert(0); \
        } \
    } while (0)

// disable assertions in production
#ifdef NDEBUG
    #define ASSERT(...)
#else
    #error "SA_SPEC: Assertions are enabled"
    #define ASSERT(...) ASSERT_IMPL(__VA_ARGS__)
#endif
