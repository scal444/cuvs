# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

add_executable(CUVS_LOW_K_DISTANCE_BENCH low_k_fused_nn.cu)

target_link_libraries(
  CUVS_LOW_K_DISTANCE_BENCH
  PRIVATE benchmark::benchmark
          benchmark::benchmark_main
          cuvs::cuvs_cpp_headers
          raft::raft
          ${CUVS_CTK_MATH_DEPENDENCIES}
          $<TARGET_NAME_IF_EXISTS:nvidia::cutlass::cutlass>
)

set_target_properties(
  CUVS_LOW_K_DISTANCE_BENCH
  PROPERTIES CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
             RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${CUVS_BINARY_DIR}/bench>"
)

target_compile_options(
  CUVS_LOW_K_DISTANCE_BENCH
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUVS_CXX_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${CUVS_CUDA_FLAGS}>"
)

# GCC 15 diagnoses a size-probing snprintf in RAFT's error formatter. Keep cuVS's warning policy
# for this target while avoiding promotion of that dependency-only diagnostic to an error.
if(CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(
    CUVS_LOW_K_DISTANCE_BENCH PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-error=format-truncation>"
  )
endif()

if(CUVS_ENABLE_SM90A_WGMMA)
  set_property(TARGET CUVS_LOW_K_DISTANCE_BENCH PROPERTY CUDA_ARCHITECTURES 90a-real)
  target_compile_definitions(CUVS_LOW_K_DISTANCE_BENCH PRIVATE CUVS_ENABLE_SM90A_WGMMA=1)
endif()
