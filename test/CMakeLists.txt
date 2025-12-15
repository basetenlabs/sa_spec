# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Baseten

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1
)

FetchContent_MakeAvailable(Catch2)

add_executable(test_runner
               catch_runner.cc
               test_buffer.cc
               test_suffix_automaton.cc
               test_cuda.cc
               test_util.cc
               utils_for_test.cu)

target_link_libraries(test_runner PRIVATE Catch2::Catch2WithMain sa_spec)
