# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
rapids_cpm_init()
include(rapids-cuda)
include(rapids-find)
include(${rapids-cmake-dir}/cython-core/find_prefix_paths.cmake)

rapids_cuda_init_architectures(rapidsmpf-topo-viz-python)

project(
  rapidsmpf-topo-viz-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

add_library(maybe_asan INTERFACE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)

find_package(CUDAToolkit REQUIRED)

# Find cuCascade (same as the main C++ build)
include(../../cmake/thirdparty/get_cucascade.cmake)

# Find NVML headers
find_path(
  NVML_INCLUDE_DIR
  NAMES nvml.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

include(rapids-cython-core)
rapids_cython_init()

set(cython_lib_dir rapidsmpf_topo_viz)

add_subdirectory(rapidsmpf_topo_viz)
