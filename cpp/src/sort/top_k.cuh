/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/utilities/traits.hpp>

#include <memory>

namespace cudf {
namespace detail {

/**
 * @brief Returns true if the top-k of `col` can be selected with `cub::DeviceTopK`
 */
inline bool is_cub_top_k_supported(column_view const& col)
{
  return !col.has_nulls() && cudf::is_fixed_width(col.type()) &&
         !cudf::is_floating_point(col.type());  // needs special NaN handling
}

/**
 * @brief Type-dispatcher functor (used with `dispatch_storage_type`) that invokes
 * `fn.template operator()<KeyT>()` with the `cub::DeviceTopK` key type `KeyT` of each type
 * accepted by `is_cub_top_k_supported`
 *
 * @tparam Fn Callable with a templated call operator returning a `std::unique_ptr<column>`
 */
template <typename Fn>
struct dispatch_cub_top_k_key {
  Fn fn;

  template <typename T>
    requires(cudf::is_fixed_width<T>() and !cudf::is_floating_point<T>())
  std::unique_ptr<column> operator()()
  {
    if constexpr (cudf::is_chrono<T>()) {
      return fn.template operator()<typename T::rep>();
    } else {
      return fn.template operator()<T>();
    }
  }

  template <typename T>
    requires(not cudf::is_fixed_width<T>() or cudf::is_floating_point<T>())
  std::unique_ptr<column> operator()()
  {
    CUDF_UNREACHABLE("unexpected type for the cub::DeviceTopK path");
  }
};

}  // namespace detail
}  // namespace cudf
