# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

add_library(bench_utils INTERFACE)
target_sources(bench_utils INTERFACE utils/random_data.cu)
add_executable(bench_shuffle "bench_shuffle.cpp")
set_target_properties(
  bench_shuffle
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_shuffle PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_shuffle PRIVATE rapidsmpf::rapidsmpf ucxx::ucxx $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
                        $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan bench_utils
)
install(
  TARGETS bench_shuffle
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)

add_executable(bench_comm "bench_comm.cpp")
set_target_properties(
  bench_comm
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_comm PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                     "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_comm PRIVATE rapidsmpf::rapidsmpf ucxx::ucxx $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
                     $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan bench_utils
)
install(
  TARGETS bench_comm
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)

add_executable(bench_cuda_ipc "bench_cuda_ipc.cu")
set_target_properties(
  bench_cuda_ipc
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_cuda_ipc PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                         "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_cuda_ipc
  PRIVATE rapidsmpf::rapidsmpf
          CUDA::cudart
          CUDA::cuda_driver
          $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
          $<TARGET_NAME_IF_EXISTS:conda_env>
          maybe_asan
)
install(
  TARGETS bench_cuda_ipc
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)

add_executable(bench_ucp_tag "bench_ucp_tag.cu")
set_target_properties(
  bench_ucp_tag
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_ucp_tag PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_ucp_tag
  PRIVATE rapidsmpf::rapidsmpf
          ucxx::ucxx
          $<TARGET_NAME_IF_EXISTS:ucx::ucp>
          $<TARGET_NAME_IF_EXISTS:ucx::ucs>
          CUDA::cudart
          CUDA::cuda_driver
          $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
          $<TARGET_NAME_IF_EXISTS:conda_env>
          maybe_asan
)
install(
  TARGETS bench_ucp_tag
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)

add_executable(bench_ucp_rma "bench_ucp_rma.cu")
set_target_properties(
  bench_ucp_rma
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_ucp_rma PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                        "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_ucp_rma
  PRIVATE rapidsmpf::rapidsmpf
          ucxx::ucxx
          $<TARGET_NAME_IF_EXISTS:ucx::ucp>
          $<TARGET_NAME_IF_EXISTS:ucx::ucs>
          CUDA::cudart
          CUDA::cuda_driver
          $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
          $<TARGET_NAME_IF_EXISTS:conda_env>
          maybe_asan
)
install(
  TARGETS bench_ucp_rma
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)

add_executable(bench_cuda_p2p "bench_cuda_p2p.cu")
set_target_properties(
  bench_cuda_p2p
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_cuda_p2p PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                         "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_cuda_p2p
  PRIVATE CUDA::cudart CUDA::cuda_driver $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan
)
install(
  TARGETS bench_cuda_p2p
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)

# Find or install GoogleBench
include(${rapids-cmake-dir}/cpm/gbench.cmake)
rapids_cpm_gbench(BUILD_STATIC)

add_executable(bench_partition "bench_partition.cpp")
set_target_properties(
  bench_partition
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_partition PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                          "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_partition
  PRIVATE rapidsmpf::rapidsmpf
          ucxx::ucxx
          benchmark::benchmark
          benchmark::benchmark_main
          $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
          $<TARGET_NAME_IF_EXISTS:conda_env>
          maybe_asan
          bench_utils
)
install(
  TARGETS bench_partition
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)

add_executable(bench_memory_resources "bench_memory_resources.cpp")
set_target_properties(
  bench_memory_resources
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_memory_resources PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                                 "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_memory_resources
  PRIVATE rapidsmpf::rapidsmpf rmm::rmm benchmark::benchmark benchmark::benchmark_main
          $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan bench_utils
)
install(
  TARGETS bench_memory_resources
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)

add_executable(bench_pack "bench_pack.cpp")
set_target_properties(
  bench_pack
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             LINK_FLAGS "-Wl,--allow-shlib-undefined"
)
target_compile_options(
  bench_pack PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                     "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_pack PRIVATE rapidsmpf::rapidsmpf benchmark::benchmark benchmark::benchmark_main
                     $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan bench_utils
)
install(
  TARGETS bench_pack
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)

if(RAPIDSMPF_HAVE_STREAMING)
  add_subdirectory(streaming)
endif()
