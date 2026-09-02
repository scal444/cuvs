/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../distance_ops/l2_exp.cuh"

#include <raft/core/kvp.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cute/tensor.hpp>
#include <cutlass/aligned_buffer.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>

#include <cstdint>
#include <limits>

#if defined(__CUDA_ARCH__) &&                                    \
  ((__CUDA_ARCH__ != 900) || !defined(__CUDA_ARCH_SPECIFIC__) || \
   (__CUDA_ARCH_SPECIFIC__ != 900) || !defined(__CUDA_ARCH_FEAT_SM90_ALL))
#error "low_k_tf32_wgmma_sm90a.cuh requires the exact SM90a architecture feature target"
#endif

namespace cuvs::distance::detail {

namespace low_k_wgmma {

using namespace cute;

inline constexpr int tile_m = 64;
inline constexpr int tile_n = 64;

template <int LogicalK>
struct traits {
  static_assert(LogicalK >= 2 && LogicalK <= 5);
  static constexpr int packed_k   = 3 * LogicalK;
  static constexpr int physical_k = ((packed_k + 7) / 8) * 8;
  using element                   = cutlass::tfloat32_t;
  using smem_layout               = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<element>{},
                                             make_shape(Int<tile_m>{}, Int<physical_k>{})));
  using tiled_mma                 = decltype(make_tiled_mma(SM90_64x64x8_F32TF32TF32_SS_TN<>{}));

  struct alignas(128) shared_storage {
    union {
      struct {
        cutlass::AlignedBuffer<element, tile_m * physical_k> a;
        cutlass::AlignedBuffer<element, tile_n * physical_k> b;
      } operands;
      cutlass::AlignedBuffer<float, tile_m * tile_n> dots;
    } storage;
  };
};

template <int LogicalK>
RAFT_KERNEL __launch_bounds__(128) low_k_tf32_wgmma_1nn(raft::KeyValuePair<int32_t, float>* output,
                                                        float const* x,
                                                        float const* y,
                                                        float const* x_norm,
                                                        float const* y_norm,
                                                        int32_t m,
                                                        int32_t n,
                                                        bool sqrt)
{
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  using kernel_traits = traits<LogicalK>;
  using element       = typename kernel_traits::element;
  using smem_layout   = typename kernel_traits::smem_layout;
  using tiled_mma     = typename kernel_traits::tiled_mma;

  extern __shared__ char shared_raw[];
  auto& shared = *reinterpret_cast<typename kernel_traits::shared_storage*>(shared_raw);

  int32_t const global_row = static_cast<int32_t>(blockIdx.x) * tile_m + threadIdx.x;
  float best_distance      = std::numeric_limits<float>::max();
  int32_t best_index       = 0xfffffff0;

  smem_layout layout{};
  tiled_mma mma{};
  cutlass::NumericConverterFastF32<cutlass::FloatRoundStyle::round_toward_zero,
                                   cutlass::FloatRoundStyle::round_half_ulp_truncate>
    converter;

  for (int32_t tile_column = 0; tile_column < n; tile_column += tile_n) {
    auto* shared_a = shared.storage.operands.a.data();
    auto* shared_b = shared.storage.operands.b.data();

    for (int index = threadIdx.x; index < tile_m * kernel_traits::physical_k; index += blockDim.x) {
      int row                     = index / kernel_traits::physical_k;
      int slot                    = index % kernel_traits::physical_k;
      shared_a[layout(row, slot)] = element{0.0f};
      shared_b[layout(row, slot)] = element{0.0f};
    }
    __syncthreads();

    for (int index = threadIdx.x; index < tile_m * LogicalK; index += blockDim.x) {
      int row = index / LogicalK;
      int dim = index % LogicalK;

      int32_t x_row = static_cast<int32_t>(blockIdx.x) * tile_m + row;
      float x_value = x_row < m ? x[static_cast<int64_t>(x_row) * LogicalK + dim] : 0.0f;
      auto x_parts  = converter(x_value);
      shared_a[layout(row, dim)]                = x_parts[0];
      shared_a[layout(row, LogicalK + dim)]     = x_parts[0];
      shared_a[layout(row, 2 * LogicalK + dim)] = x_parts[1];

      int32_t y_row = tile_column + row;
      float y_value = y_row < n ? y[static_cast<int64_t>(y_row) * LogicalK + dim] : 0.0f;
      auto y_parts  = converter(y_value);
      shared_b[layout(row, dim)]                = y_parts[0];
      shared_b[layout(row, LogicalK + dim)]     = y_parts[1];
      shared_b[layout(row, 2 * LogicalK + dim)] = y_parts[0];
    }
    __syncthreads();

    auto tensor_a   = make_tensor(make_smem_ptr(shared_a), layout);
    auto tensor_b   = make_tensor(make_smem_ptr(shared_b), layout);
    auto thread_mma = mma.get_slice(threadIdx.x);
    auto fragment_a = thread_mma.make_fragment_A(thread_mma.partition_A(tensor_a));
    auto fragment_b = thread_mma.make_fragment_B(thread_mma.partition_B(tensor_b));

    auto result_layout =
      make_layout(make_shape(Int<tile_m>{}, Int<tile_n>{}), make_stride(Int<tile_n>{}, Int<1>{}));
    auto tensor_c     = make_tensor(make_smem_ptr(shared.storage.dots.data()), result_layout);
    auto partition_c  = thread_mma.partition_C(tensor_c);
    auto accumulators = thread_mma.make_fragment_C(partition_c);
    clear(accumulators);

    warpgroup_fence_operand(accumulators);
    warpgroup_arrive();
    cute::gemm(mma, fragment_a, fragment_b, accumulators);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(accumulators);
    copy(accumulators, partition_c);
    __syncthreads();

    if (threadIdx.x < tile_m && global_row < m) {
      float const row_norm = x_norm[global_row];
      int valid_columns    = min(tile_n, n - tile_column);
      for (int column = 0; column < valid_columns; ++column) {
        float dot      = shared.storage.dots.data()[threadIdx.x * tile_n + column];
        float distance = row_norm + y_norm[tile_column + column] - 2.0f * dot;
        distance *= !((distance * distance < ops::get_clamp_precision<float, float>()) &&
                      (row_norm == y_norm[tile_column + column]));
        if (sqrt) { distance = raft::sqrt(distance * static_cast<float>(distance > 0.0f)); }
        if (distance < best_distance) {
          best_distance = distance;
          best_index    = tile_column + column;
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x < tile_m && global_row < m) {
    output[global_row] = raft::KeyValuePair<int32_t, float>{best_index, best_distance};
  }
#endif
}

template <int LogicalK>
void launch(raft::KeyValuePair<int32_t, float>* output,
            float const* x,
            float const* y,
            float const* x_norm,
            float const* y_norm,
            int32_t m,
            int32_t n,
            bool sqrt,
            cudaStream_t stream)
{
  using kernel_traits = traits<LogicalK>;
  static_assert(sizeof(typename kernel_traits::shared_storage) == tile_m * tile_n * sizeof(float));
  auto grid  = dim3((m + tile_m - 1) / tile_m);
  auto block = dim3(128);
  low_k_tf32_wgmma_1nn<LogicalK>
    <<<grid, block, sizeof(typename kernel_traits::shared_storage), stream>>>(
      output, x, y, x_norm, y_norm, m, n, sqrt);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace low_k_wgmma

}  // namespace cuvs::distance::detail
