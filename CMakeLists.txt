# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Baseten

cmake_minimum_required(VERSION 3.20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


project(sa_spec LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# enable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra")

# for cuda linkage compatibility
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")


if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    include(FetchContent)
    FetchContent_Declare(
        cpptrace
        GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
        GIT_TAG        v1.0.4 # <HASH or TAG>
    )
    FetchContent_MakeAvailable(cpptrace)
    link_libraries(cpptrace::cpptrace)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAS_CPPTRACE")
endif()


set(LIB_SOURCES
    src/memory.cu
    src/api.cu
    src/api.cc
    src/lock_release_guard.cc
)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

if (SKBUILD)
    # python bindings

    find_package(Python
                REQUIRED COMPONENTS Interpreter Development.Module
                OPTIONAL_COMPONENTS Development.SABIModule)

    # Detect the installed nanobind package and import it into CMake
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(
        _sa_spec_impl
        STABLE_ABI
        src/bind.cc

        ${LIB_SOURCES}
    )

    set_property(TARGET _sa_spec_impl PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)

    install(TARGETS _sa_spec_impl LIBRARY DESTINATION sa_spec)

else()
    # tests
    add_library(sa_spec STATIC ${LIB_SOURCES})
    
    add_subdirectory(test)

endif()
