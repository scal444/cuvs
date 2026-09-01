/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cutlass/array.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/default_mma_core.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_conversion.h>

#include <cstdint>
#include <type_traits>

namespace cuvs::gemm::threadblock {

template <int LogicalK, bool IsA>
class LowKTF32InputIterator {
 public:
  using Element     = float;
  using Layout      = std::conditional_t<IsA, cutlass::layout::RowMajor,
                                     cutlass::layout::ColumnMajor>;
  using AccessType  = cutlass::Array<float, 1>;
  using Fragment    = cutlass::Array<float, 1>;
  using TensorCoord = cutlass::MatrixCoord;
  struct Params {
    int64_t stride = 0;
    CUTLASS_HOST_DEVICE Params() = default;
    CUTLASS_HOST_DEVICE explicit Params(int64_t value) : stride(value) {}
  };

 private:
  float const* pointer_ = nullptr;
  int64_t stride_       = 0;
  TensorCoord extent_{};
  TensorCoord offset_{};

 public:
  CUTLASS_DEVICE LowKTF32InputIterator(
    Params const& params, float* pointer, TensorCoord extent, int, TensorCoord offset)
    : pointer_(pointer), stride_(params.stride), extent_(extent), offset_(offset)
  {
  }
  CUTLASS_DEVICE float const* pointer() const { return pointer_; }
  CUTLASS_DEVICE int64_t stride() const { return stride_; }
  CUTLASS_DEVICE int rows() const { return extent_.row(); }
  CUTLASS_DEVICE int columns() const { return extent_.column(); }
  CUTLASS_DEVICE int row_offset() const { return offset_.row(); }
  CUTLASS_DEVICE int column_offset() const { return offset_.column(); }
};

template <int LogicalK, int TileN = 128, int WarpN = 32>
class LowKTF32Mma {
 private:
  // Portable native-MMA implementation for SM80+ targets. More capable architecture-feature
  // targets may override this fallback with WGMMA or tcgen05 backends.
  static_assert(LogicalK >= 2 && LogicalK <= 5);
  static constexpr int PhysicalK = LogicalK == 2 ? 8 : 16;
  static constexpr int StorageK  = 16;
  static constexpr int MmaGroups = PhysicalK / 8;
  using InternalShape = cutlass::gemm::GemmShape<32, TileN, StorageK>;
  using InternalWarpShape = cutlass::gemm::GemmShape<32, WarpN, StorageK>;
  using Core = cutlass::gemm::threadblock::DefaultMmaCore<
    InternalShape,
    InternalWarpShape,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::tfloat32_t,
    cutlass::layout::RowMajor,
    cutlass::tfloat32_t,
    cutlass::layout::RowMajor,
    float,
    cutlass::layout::RowMajor,
    cutlass::arch::OpClassTensorOp,
    2,
    cutlass::arch::OpMultiplyAdd>;

 public:
  using Shape       = cutlass::gemm::GemmShape<32, TileN, 16>;
  using IteratorA   = LowKTF32InputIterator<LogicalK, true>;
  using IteratorB   = LowKTF32InputIterator<LogicalK, false>;
  using Operator    = typename Core::MmaTensorOp;
  using Policy      = typename Core::MmaPolicy;
  using FragmentC   = typename Operator::FragmentC;
  using LayoutC     = cutlass::layout::RowMajor;
  using ArchTag     = typename Operator::ArchTag;
  using WarpCount   = cutlass::gemm::GemmShape<1, TileN / WarpN, 1>;
  static int const kStages = 1;
  static cutlass::ComplexTransform const kTransformA = cutlass::ComplexTransform::kNone;
  static cutlass::ComplexTransform const kTransformB = cutlass::ComplexTransform::kNone;
  struct SharedStorage {
    cutlass::AlignedBuffer<cutlass::tfloat32_t, 32 * StorageK> a;
    cutlass::AlignedBuffer<cutlass::tfloat32_t, StorageK * TileN> b;
  };

 private:
  SharedStorage& storage_;
  int warp_idx_;
  int lane_idx_;

 public:
  CUTLASS_DEVICE LowKTF32Mma(SharedStorage& storage, int, int warp_idx, int lane_idx)
    : storage_(storage), warp_idx_(warp_idx), lane_idx_(lane_idx)
  {
  }

  CUTLASS_DEVICE void operator()(int gemm_k_iterations,
                                 FragmentC& accum,
                                 IteratorA& iterator_a,
                                 IteratorB& iterator_b,
                                 FragmentC const& src_accum)
  {
    accum = src_accum;
    __syncthreads();
    using LayoutA = typename Core::SmemLayoutA;
    using LayoutB = typename Core::SmemLayoutB;
    LayoutA layout_a = LayoutA::packed({32, StorageK});
    LayoutB layout_b = LayoutB::packed({StorageK, TileN});
    using Converter = cutlass::NumericConverterFastF32<
      cutlass::FloatRoundStyle::round_toward_zero,
      cutlass::FloatRoundStyle::round_half_ulp_truncate>;
    Converter converter;

    // FastF32 approximates a*b with big(a)*big(b) + big(a)*small(b) +
    // small(a)*big(b). Pack those three wanted products into the physical K lanes;
    // zero padding makes the MMA reduction contain no cross terms.

    for (int index = threadIdx.x; index < 32 * StorageK; index += blockDim.x) {
      int row  = index >> 4;
      int slot = index & (StorageK - 1);
      if (slot >= 3 * LogicalK) {
        storage_.a.data()[layout_a({row, slot})] = cutlass::tfloat32_t(0.0f);
      }
    }
    for (int index = threadIdx.x; index < StorageK * TileN; index += blockDim.x) {
      int slot = index / TileN;
      int col  = index & (TileN - 1);
      if (slot >= 3 * LogicalK) {
        storage_.b.data()[layout_b({slot, col})] = cutlass::tfloat32_t(0.0f);
      }
    }
    for (int index = threadIdx.x; index < 32 * LogicalK; index += blockDim.x) {
      int row = index / LogicalK;
      int dim = index % LogicalK;
      int global_row = iterator_a.row_offset() + row;
      float value = global_row < iterator_a.rows()
                      ? iterator_a.pointer()[static_cast<int64_t>(global_row) * iterator_a.stride() + dim]
                      : 0.0f;
      auto parts = converter(value);
      storage_.a.data()[layout_a({row, dim})]                = parts[0];
      storage_.a.data()[layout_a({row, LogicalK + dim})]     = parts[0];
      storage_.a.data()[layout_a({row, 2 * LogicalK + dim})] = parts[1];
    }
    for (int index = threadIdx.x; index < LogicalK * TileN; index += blockDim.x) {
      int dim = index / TileN;
      int col = index % TileN;
      int global_col = iterator_b.column_offset() + col;
      float value = global_col < iterator_b.columns()
                      ? iterator_b.pointer()[static_cast<int64_t>(global_col) * iterator_b.stride() + dim]
                      : 0.0f;
      auto parts = converter(value);
      storage_.b.data()[layout_b({dim, col})]                = parts[0];
      storage_.b.data()[layout_b({LogicalK + dim, col})]     = parts[1];
      storage_.b.data()[layout_b({2 * LogicalK + dim, col})] = parts[0];
    }
    __syncthreads();

    cutlass::TensorRef<cutlass::tfloat32_t, LayoutA> ref_a(storage_.a.data(), layout_a);
    cutlass::TensorRef<cutlass::tfloat32_t, LayoutB> ref_b(storage_.b.data(), layout_b);
    typename Operator::IteratorA warp_a(ref_a, lane_idx_);
    typename Operator::IteratorB warp_b(ref_b, lane_idx_);
    int warp_m = warp_idx_ % WarpCount::kM;
    int warp_n = (warp_idx_ / WarpCount::kM) % WarpCount::kN;
    warp_a.add_tile_offset({warp_m, 0});
    warp_b.add_tile_offset({0, warp_n});
    Operator warp_mma;
#pragma unroll
    for (int group = 0; group < MmaGroups; ++group) {
      typename Operator::FragmentA fragment_a;
      typename Operator::FragmentB fragment_b;
      typename Operator::TransformedFragmentA transformed_a;
      typename Operator::TransformedFragmentB transformed_b;
      warp_a.set_kgroup_index(group);
      warp_b.set_kgroup_index(group);
      warp_a.load(fragment_a);
      warp_b.load(fragment_b);
      warp_mma.transform(transformed_a, transformed_b, fragment_a, fragment_b);
      warp_mma(accum, transformed_a, transformed_b, accum);
      ++warp_a;
      ++warp_b;
    }
    __syncthreads();
    (void)gemm_k_iterations;
  }
};

}  // namespace cuvs::gemm::threadblock
