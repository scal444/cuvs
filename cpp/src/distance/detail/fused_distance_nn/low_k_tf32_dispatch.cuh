/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <limits>
#include <type_traits>

namespace cuvs::distance::detail {

inline constexpr int low_k_tf32_min_k = 2;
inline constexpr int low_k_tf32_max_k = 5;

/**
 * @brief Return whether a problem has a packed-TF32 specialization.
 *
 * This predicate describes only the data type and problem shape. Callers must independently
 * establish that the selected code image supports the MMA backend. Keeping architecture selection
 * outside this predicate avoids confusing the physical device capability with the virtual
 * architecture of the kernel selected from a CUDA fat binary.
 */
template <typename DataT, typename IdxT>
constexpr bool is_low_k_tf32_problem(IdxT m, IdxT n, IdxT k)
{
  static_assert(std::is_integral_v<IdxT>);
  static_assert(std::numeric_limits<IdxT>::max() >= std::numeric_limits<int>::max());

  return std::is_same_v<DataT, float> && m > IdxT{0} && n > IdxT{0} &&
         k >= static_cast<IdxT>(low_k_tf32_min_k) && k <= static_cast<IdxT>(low_k_tf32_max_k) &&
         m <= static_cast<IdxT>(std::numeric_limits<int>::max()) &&
         n <= static_cast<IdxT>(std::numeric_limits<int>::max());
}

}  // namespace cuvs::distance::detail
