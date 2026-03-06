# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

# Build the C++ topology library sources as a static library for the Cython module to
# link against.
set(TOPO_VIZ_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp")

add_library(
  rapidsmpf_topology STATIC
  "${TOPO_VIZ_CPP_DIR}/src/topology/json_io.cpp"
  "${TOPO_VIZ_CPP_DIR}/src/topology/bandwidth_discovery.cpp"
  "${TOPO_VIZ_CPP_DIR}/src/topology/system_enrichment.cpp"
  "${TOPO_VIZ_CPP_DIR}/src/topology/topology_viz.cpp"
)
set_target_properties(
  rapidsmpf_topology
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
)
target_include_directories(
  rapidsmpf_topology PUBLIC "${TOPO_VIZ_CPP_DIR}/include"
)
if(NVML_INCLUDE_DIR)
  target_include_directories(rapidsmpf_topology PRIVATE ${NVML_INCLUDE_DIR})
endif()
target_link_libraries(
  rapidsmpf_topology
  PUBLIC cuCascade::cucascade
  PRIVATE CUDA::nvml $<$<TARGET_EXISTS:kvikio::kvikio>:kvikio::kvikio>
)

# Build the Cython module
set(cython_modules topology.pyx)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_modules}"
  LINKED_LIBRARIES rapidsmpf_topology maybe_asan
)
