/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_nn_helper.cuh"

#include "../../src/distance/detail/fused_distance_nn/low_k_tf32_dispatch.cuh"
#include "../../src/distance/fused_distance_nn.cuh"

#include <raft/core/kvp.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/norm.cuh>

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace cuvs::neighbors {

struct LowKInput {
  int m;
  int n;
  int k;
};

class LowKTF32Test : public ::testing::TestWithParam<LowKInput> {
 protected:
  using output_type = raft::KeyValuePair<int, float>;

  void run_and_compare()
  {
    auto params = GetParam();
    raft::resources handle;
    auto stream = raft::resource::get_cuda_stream(handle);
    auto x      = raft::make_device_matrix<float, int>(handle, params.m, params.k);
    auto y      = raft::make_device_matrix<float, int>(handle, params.n, params.k);
    auto x_norm = raft::make_device_vector<float, int>(handle, params.m);
    auto y_norm = raft::make_device_vector<float, int>(handle, params.n);
    auto output = raft::make_device_vector<output_type, int>(handle, params.m);
    auto oracle = raft::make_device_vector<output_type, int>(handle, params.m);

    std::mt19937 generator(31415926);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> host_y(static_cast<std::size_t>(params.n) * params.k);
    std::vector<float> host_x(static_cast<std::size_t>(params.m) * params.k);
    for (auto& value : host_y) {
      value = distribution(generator);
    }
    for (int row = 0; row < params.m; ++row) {
      int nearest = (row * 7919) % params.n;
      for (int dim = 0; dim < params.k; ++dim) {
        host_x[static_cast<std::size_t>(row) * params.k + dim] =
          host_y[static_cast<std::size_t>(nearest) * params.k + dim] +
          1.0e-3f * static_cast<float>(dim + 1);
      }
    }

    raft::update_device(x.data_handle(), host_x.data(), host_x.size(), stream);
    raft::update_device(y.data_handle(), host_y.data(), host_y.size(), stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      x_norm.data_handle(), x.data_handle(), params.k, params.m, stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      y_norm.data_handle(), y.data_handle(), params.k, params.n, stream);

    rmm::device_uvector<int> workspace(params.m, stream);
    ref_nn<float, float, output_type, int>(oracle.data_handle(),
                                           x.data_handle(),
                                           y.data_handle(),
                                           params.m,
                                           params.n,
                                           params.k,
                                           false,
                                           DistanceType::L2Expanded,
                                           stream);
    cuvs::distance::fusedDistanceNNMinReduce<float, output_type, int>(output.data_handle(),
                                                                      x.data_handle(),
                                                                      y.data_handle(),
                                                                      x_norm.data_handle(),
                                                                      y_norm.data_handle(),
                                                                      params.m,
                                                                      params.n,
                                                                      params.k,
                                                                      workspace.data(),
                                                                      false,
                                                                      true,
                                                                      true,
                                                                      DistanceType::L2Expanded,
                                                                      0.0f,
                                                                      stream);

    ComparisonSummary summary;
    vector_compare(handle, oracle.data_handle(), output.data_handle(), params.m, summary);
    EXPECT_EQ(summary.n_misses, 0) << summary;
    EXPECT_LT(summary.max_diff, 1.0e-4) << summary;
  }
};

TEST_P(LowKTF32Test, MatchesFP32Oracle) { run_and_compare(); }

INSTANTIATE_TEST_SUITE_P(AlignedAndResidualTiles,
                         LowKTF32Test,
                         ::testing::Values(LowKInput{256, 1024, 2},
                                           LowKInput{259, 1031, 2},
                                           LowKInput{256, 1024, 3},
                                           LowKInput{259, 1031, 3},
                                           LowKInput{256, 1024, 4},
                                           LowKInput{259, 1031, 4},
                                           LowKInput{256, 1024, 5},
                                           LowKInput{259, 1031, 5}));

TEST(LowKTF32Dispatch, SupportedProblems)
{
  using cuvs::distance::detail::is_low_k_tf32_problem;

  EXPECT_FALSE((is_low_k_tf32_problem<float>(128, 256, 1)));
  for (int k = 2; k <= 5; ++k) {
    EXPECT_TRUE((is_low_k_tf32_problem<float>(128, 256, k)));
  }
  EXPECT_FALSE((is_low_k_tf32_problem<float>(128, 256, 6)));
  EXPECT_FALSE((is_low_k_tf32_problem<double>(128, 256, 3)));
  EXPECT_FALSE((is_low_k_tf32_problem<float>(0, 256, 3)));
  EXPECT_FALSE((is_low_k_tf32_problem<float>(128, 0, 3)));
  EXPECT_FALSE((is_low_k_tf32_problem<float, int64_t>(
    static_cast<int64_t>(std::numeric_limits<int>::max()) + 1, 256, 3)));
}

}  // namespace cuvs::neighbors
