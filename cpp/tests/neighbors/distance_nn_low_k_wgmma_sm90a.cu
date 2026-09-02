/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "distance_nn_helper.cuh"

#include "../../src/distance/detail/fused_distance_nn/low_k_tf32_wgmma_sm90a.cuh"

#include <raft/core/kvp.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/norm.cuh>

#include <cstdint>
#include <random>
#include <vector>

namespace cuvs::neighbors {

struct LowKWgmmaInput {
  int m;
  int n;
  int k;
};

class LowKTF32WgmmaTest : public ::testing::TestWithParam<LowKWgmmaInput> {
 protected:
  using output_type = raft::KeyValuePair<int32_t, float>;

  template <int LogicalK>
  void launch(output_type* output,
              float const* x,
              float const* y,
              float const* x_norm,
              float const* y_norm,
              int m,
              int n,
              cudaStream_t stream)
  {
    cuvs::distance::detail::low_k_wgmma::launch<LogicalK>(
      output, x, y, x_norm, y_norm, m, n, false, stream);
  }

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
    std::vector<float> host_x(static_cast<std::size_t>(params.m) * params.k);
    std::vector<float> host_y(static_cast<std::size_t>(params.n) * params.k);
    for (auto& value : host_x) {
      value = distribution(generator);
    }
    for (auto& value : host_y) {
      value = distribution(generator);
    }

    raft::update_device(x.data_handle(), host_x.data(), host_x.size(), stream);
    raft::update_device(y.data_handle(), host_y.data(), host_y.size(), stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      x_norm.data_handle(), x.data_handle(), params.k, params.m, stream);
    raft::linalg::rowNorm<raft::linalg::L2Norm, true>(
      y_norm.data_handle(), y.data_handle(), params.k, params.n, stream);
    ref_nn<float, float, output_type, int>(oracle.data_handle(),
                                           x.data_handle(),
                                           y.data_handle(),
                                           params.m,
                                           params.n,
                                           params.k,
                                           false,
                                           DistanceType::L2Expanded,
                                           stream);

    switch (params.k) {
      case 2:
        launch<2>(output.data_handle(),
                  x.data_handle(),
                  y.data_handle(),
                  x_norm.data_handle(),
                  y_norm.data_handle(),
                  params.m,
                  params.n,
                  stream);
        break;
      case 3:
        launch<3>(output.data_handle(),
                  x.data_handle(),
                  y.data_handle(),
                  x_norm.data_handle(),
                  y_norm.data_handle(),
                  params.m,
                  params.n,
                  stream);
        break;
      case 4:
        launch<4>(output.data_handle(),
                  x.data_handle(),
                  y.data_handle(),
                  x_norm.data_handle(),
                  y_norm.data_handle(),
                  params.m,
                  params.n,
                  stream);
        break;
      case 5:
        launch<5>(output.data_handle(),
                  x.data_handle(),
                  y.data_handle(),
                  x_norm.data_handle(),
                  y_norm.data_handle(),
                  params.m,
                  params.n,
                  stream);
        break;
      default: FAIL() << "unsupported K";
    }

    ComparisonSummary summary;
    vector_compare(handle, oracle.data_handle(), output.data_handle(), params.m, summary);
    EXPECT_EQ(summary.n_misses, 0) << summary;
    EXPECT_LT(summary.max_diff, 1.0e-4) << summary;
  }
};

TEST_P(LowKTF32WgmmaTest, MatchesFP32Oracle) { run_and_compare(); }

INSTANTIATE_TEST_SUITE_P(AlignedAndResidualTiles,
                         LowKTF32WgmmaTest,
                         ::testing::Values(LowKWgmmaInput{64, 128, 2},
                                           LowKWgmmaInput{67, 131, 2},
                                           LowKWgmmaInput{64, 128, 3},
                                           LowKWgmmaInput{67, 131, 3},
                                           LowKWgmmaInput{64, 128, 4},
                                           LowKWgmmaInput{67, 131, 4},
                                           LowKWgmmaInput{64, 128, 5},
                                           LowKWgmmaInput{67, 131, 5}));

}  // namespace cuvs::neighbors
