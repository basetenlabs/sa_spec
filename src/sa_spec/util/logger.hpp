// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 Baseten

#pragma once

#include <cstdio>

#ifdef __CUDACC__
    #define MAYBE_CUDA "[CUDA]"
    #define MAYBE_CUDA_SEP " "
    #define MAYBE_PRINT_STACK_TRACE
#else
    #define MAYBE_CUDA
    #define MAYBE_CUDA_SEP

    #ifdef HAS_CPPTRACE
        #include <cpptrace/cpptrace.hpp>
        #define MAYBE_PRINT_STACK_TRACE cpptrace::generate_trace().print()
    #else
        #define MAYBE_PRINT_STACK_TRACE
    #endif

#endif

#define LOG_IMPL(level, fmt, ...) \
    printf("[SA_SPEC] " level MAYBE_CUDA_SEP MAYBE_CUDA " %s:%d: %s" fmt "\n", \
            __FILE__, __LINE__, " " __VA_OPT__(,) __VA_ARGS__)


#define LOG_TRACE(...) LOG_IMPL("TRACE", __VA_ARGS__);
#define LOG_INFO(...)  LOG_IMPL("INFO", __VA_ARGS__);


#ifdef NDEBUG
    #undef LOG_TRACE
    #define LOG_TRACE(...)
#endif
