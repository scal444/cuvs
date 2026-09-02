/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/distance/distance.hpp>

#include "../../src/distance/detail/fused_distance_nn/cutlass_base.cuh"
#include "../../src/distance/detail/fused_distance_nn/helper_structs.cuh"
#include "../../src/distance/fused_distance_nn_helpers.cuh"
#include "../../src/distance/unfused_distance_nn.cuh"

#if defined(CUVS_ENABLE_SM90A_WGMMA)
#include "../../src/distance/detail/fused_distance_nn/low_k_tf32_wgmma_sm90a.cuh"
#endif

#include <benchmark/benchmark.h>

#include <raft/core/kvp.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using index_type  = int32_t;
using output_type = raft::KeyValuePair<index_type, float>;

enum class implementation {
  packed_tf32,
  cutlass_fast_f32,
  unfused_fp32,
#if defined(CUVS_ENABLE_SM90A_WGMMA)
  wgmma_tf32,
#endif
};

struct validation_result {
  std::size_t oracle_assignment_mismatches{};
  float oracle_max_distance_error{};
  std::size_t fast_f32_assignment_mismatches{};
  float fast_f32_max_distance_error{};
};

RAFT_KERNEL exact_l2_1nn(
  output_type* output, float const* x, float const* y, index_type m, index_type n, index_type k)
{
  auto row = static_cast<index_type>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= m) { return; }

  float best_distance = std::numeric_limits<float>::infinity();
  index_type best_key = 0;
  for (index_type col = 0; col < n; ++col) {
    float distance = 0.0f;
    for (index_type dim = 0; dim < k; ++dim) {
      float delta =
        x[static_cast<std::size_t>(row) * k + dim] - y[static_cast<std::size_t>(col) * k + dim];
      distance = fmaf(delta, delta, distance);
    }
    if (distance < best_distance) {
      best_distance = distance;
      best_key      = col;
    }
  }
  output[row] = output_type{best_key, best_distance};
}

template <int LogicalK>
inline constexpr int input_alignment = LogicalK == 2 ? 2 : (LogicalK == 4 ? 4 : 1);

class low_k_problem {
 public:
  low_k_problem(index_type m, index_type n, index_type k)
    : m_{m},
      n_{n},
      k_{k},
      stream_{raft::resource::get_cuda_stream(handle_)},
      x_(static_cast<std::size_t>(m) * k, stream_),
      y_(static_cast<std::size_t>(n) * k, stream_),
      x_norm_(m, stream_),
      y_norm_(n, stream_),
      output_(m, stream_),
      oracle_(m, stream_),
      fused_workspace_(static_cast<std::size_t>(m) * sizeof(int), stream_),
      unfused_workspace_(static_cast<std::size_t>(m) * n, stream_)
  {
    initialize_inputs();
    compute_norms_and_oracle();
  }

  void run(implementation impl)
  {
    switch (k_) {
      case 2: run_for_k<2>(impl); break;
      case 3: run_for_k<3>(impl); break;
      case 4: run_for_k<4>(impl); break;
      case 5: run_for_k<5>(impl); break;
      default: throw std::invalid_argument("low-K benchmark supports K=2..5");
    }
  }

  void synchronize() const { RAFT_CUDA_TRY(cudaStreamSynchronize(stream_)); }

  validation_result validate(implementation impl)
  {
    run(impl);
    synchronize();

    std::vector<output_type> actual(m_);
    std::vector<output_type> oracle(m_);
    std::vector<output_type> fast_f32(m_);
    RAFT_CUDA_TRY(cudaMemcpy(
      actual.data(), output_.data(), actual.size() * sizeof(output_type), cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(
      oracle.data(), oracle_.data(), oracle.size() * sizeof(output_type), cudaMemcpyDeviceToHost));
    if (impl == implementation::cutlass_fast_f32) {
      fast_f32 = actual;
    } else {
      run(implementation::cutlass_fast_f32);
      synchronize();
      RAFT_CUDA_TRY(cudaMemcpy(fast_f32.data(),
                               output_.data(),
                               fast_f32.size() * sizeof(output_type),
                               cudaMemcpyDeviceToHost));
    }

    validation_result result;
    for (index_type row = 0; row < m_; ++row) {
      result.oracle_assignment_mismatches += actual[row].key != oracle[row].key;
      result.oracle_max_distance_error =
        std::max(result.oracle_max_distance_error, std::abs(actual[row].value - oracle[row].value));
      result.fast_f32_assignment_mismatches += actual[row].key != fast_f32[row].key;
      result.fast_f32_max_distance_error = std::max(
        result.fast_f32_max_distance_error, std::abs(actual[row].value - fast_f32[row].value));
    }
    return result;
  }

 private:
  void initialize_inputs()
  {
    std::mt19937 generator(31415926);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> host_x(static_cast<std::size_t>(m_) * k_);
    std::vector<float> host_y(static_cast<std::size_t>(n_) * k_);
    std::generate(host_x.begin(), host_x.end(), [&] { return distribution(generator); });
    std::generate(host_y.begin(), host_y.end(), [&] { return distribution(generator); });
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      x_.data(), host_x.data(), host_x.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      y_.data(), host_y.data(), host_y.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
  }

  void compute_norms_and_oracle()
  {
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(x_norm_.data(), x_.data(), k_, m_, stream_);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(y_norm_.data(), y_.data(), k_, n_, stream_);
    exact_l2_1nn<<<(m_ + 255) / 256, 256, 0, stream_>>>(
      oracle_.data(), x_.data(), y_.data(), m_, n_, k_);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    synchronize();
  }

  void initialize_fused_output()
  {
    using reduce_op = cuvs::distance::MinAndDistanceReduceOp<index_type, float>;
    RAFT_CUDA_TRY(cudaMemsetAsync(fused_workspace_.data(), 0, fused_workspace_.size(), stream_));
    cuvs::distance::detail::initKernel<float, output_type, index_type, reduce_op>
      <<<(m_ + 255) / 256, 256, 0, stream_>>>(
        output_.data(), m_, std::numeric_limits<float>::max(), reduce_op{});
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  template <int LogicalK>
  void run_fused(bool packed)
  {
    using reduce_op      = cuvs::distance::MinAndDistanceReduceOp<index_type, float>;
    using pair_reduce_op = cuvs::distance::KVPMinReduce<index_type, float>;
    using cg_reduce_op =
      cuvs::distance::detail::kvp_cg_min_reduce_op<float, index_type, output_type>;
    using distance_op = cuvs::distance::detail::ops::l2_exp_cutlass_op<float, float>;

    initialize_fused_output();
    if (packed) {
      cuvs::distance::detail::cutlassFusedDistanceNNLowKTF32<LogicalK, input_alignment<LogicalK>>(
        x_.data(),
        y_.data(),
        x_norm_.data(),
        y_norm_.data(),
        m_,
        n_,
        output_.data(),
        reinterpret_cast<int*>(fused_workspace_.data()),
        cg_reduce_op{},
        distance_op{false},
        reduce_op{},
        pair_reduce_op{},
        stream_);
    } else {
      cuvs::distance::detail::
        cutlassFusedDistanceNN<float, float, output_type, index_type, input_alignment<LogicalK>>(
          x_.data(),
          y_.data(),
          x_norm_.data(),
          y_norm_.data(),
          m_,
          n_,
          LogicalK,
          LogicalK,
          LogicalK,
          n_,
          output_.data(),
          reinterpret_cast<int*>(fused_workspace_.data()),
          cg_reduce_op{},
          distance_op{false},
          reduce_op{},
          pair_reduce_op{},
          stream_);
    }
  }

  void run_unfused()
  {
    cuvs::distance::unfusedDistanceNNMinReduce<float, float, output_type, index_type>(
      handle_,
      output_.data(),
      x_.data(),
      y_.data(),
      x_norm_.data(),
      y_norm_.data(),
      m_,
      n_,
      k_,
      unfused_workspace_.data(),
      false,
      true,
      true,
      cuvs::distance::DistanceType::L2Expanded,
      0.0f,
      stream_);
  }

#if defined(CUVS_ENABLE_SM90A_WGMMA)
  template <int LogicalK>
  void run_wgmma()
  {
    cuvs::distance::detail::low_k_wgmma::launch<LogicalK>(
      output_.data(), x_.data(), y_.data(), x_norm_.data(), y_norm_.data(), m_, n_, false, stream_);
  }
#endif

  template <int LogicalK>
  void run_for_k(implementation impl)
  {
    switch (impl) {
      case implementation::packed_tf32: run_fused<LogicalK>(true); break;
      case implementation::cutlass_fast_f32: run_fused<LogicalK>(false); break;
      case implementation::unfused_fp32: run_unfused(); break;
#if defined(CUVS_ENABLE_SM90A_WGMMA)
      case implementation::wgmma_tf32: run_wgmma<LogicalK>(); break;
#endif
    }
  }

  index_type m_;
  index_type n_;
  index_type k_;
  raft::resources handle_;
  cudaStream_t stream_;
  rmm::device_uvector<float> x_;
  rmm::device_uvector<float> y_;
  rmm::device_uvector<float> x_norm_;
  rmm::device_uvector<float> y_norm_;
  rmm::device_uvector<output_type> output_;
  rmm::device_uvector<output_type> oracle_;
  rmm::device_uvector<std::byte> fused_workspace_;
  rmm::device_uvector<float> unfused_workspace_;
};

void run_benchmark(benchmark::State& state, implementation impl)
{
  auto m = static_cast<index_type>(state.range(0));
  auto n = static_cast<index_type>(state.range(1));
  auto k = static_cast<index_type>(state.range(2));
  low_k_problem problem(m, n, k);
  auto validation = problem.validate(impl);

  for (auto _ : state) {
    auto start = std::chrono::steady_clock::now();
    problem.run(impl);
    problem.synchronize();
    auto stop = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(stop - start).count());
  }

  state.counters["oracle_assignment_mismatches"] =
    static_cast<double>(validation.oracle_assignment_mismatches);
  state.counters["oracle_max_distance_error"] = validation.oracle_max_distance_error;
  state.counters["fast_f32_assignment_mismatches"] =
    static_cast<double>(validation.fast_f32_assignment_mismatches);
  state.counters["fast_f32_max_distance_error"] = validation.fast_f32_max_distance_error;
}

void packed_tf32(benchmark::State& state) { run_benchmark(state, implementation::packed_tf32); }

void cutlass_fast_f32(benchmark::State& state)
{
  run_benchmark(state, implementation::cutlass_fast_f32);
}

void unfused_fp32(benchmark::State& state) { run_benchmark(state, implementation::unfused_fp32); }

#if defined(CUVS_ENABLE_SM90A_WGMMA)
void wgmma_tf32(benchmark::State& state) { run_benchmark(state, implementation::wgmma_tf32); }
#endif

void add_low_k_arguments(benchmark::internal::Benchmark* registration)
{
  for (int k = 2; k <= 5; ++k) {
    registration->Args({2048, 8192, k});
    registration->Args({2051, 8209, k});
  }
  registration->UseManualTime()->Unit(benchmark::kMicrosecond)->MinTime(1.0);
}

BENCHMARK(packed_tf32)->Apply(add_low_k_arguments);
BENCHMARK(cutlass_fast_f32)->Apply(add_low_k_arguments);
BENCHMARK(unfused_fp32)->Apply(add_low_k_arguments);
#if defined(CUVS_ENABLE_SM90A_WGMMA)
BENCHMARK(wgmma_tf32)->Apply(add_low_k_arguments);
#endif

}  // namespace
