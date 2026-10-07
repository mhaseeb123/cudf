/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort.hpp"
#include "top_k_dispatch.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_topk.cuh>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Returns whether CUB-based path is optimal for the specified number of rows, number of
 * segments, and `k`
 */
constexpr bool use_cub_based_segmented_top_k(size_type num_rows,
                                             size_type num_segments,
                                             size_type k)
{
  // Empirically measured limits for optimal CUB-based dispatch

  // Max number of segments for optimal CUB-based dispatch
  constexpr size_type max_segments = 64;
  // Min average segment size for optimal CUB-based dispatch
  constexpr size_type min_avg_segment_size = 16'384;
  // Max k, as 1/N of the average segment size, for optimal CUB-based dispatch
  constexpr size_type max_k_fraction = 8;

  // Check if number of segments is out of range
  if (num_segments <= 0 or num_segments > max_segments) { return false; }

  auto const avg_segment_size = num_rows / num_segments;
  return cuda::std::cmp_greater_equal(avg_segment_size, min_avg_segment_size) and
         cuda::std::cmp_less_equal(k, avg_segment_size / max_k_fraction);
}

/**
 * @brief Resolves the k indices per segment
 *
 * Marks values outside the k range to -1 to be removed in a separate step.
 * Rows not covered by any segment are also marked to be removed.
 * Also computes the total number of valid indices for each segment.
 * All elements are used in a segment if it has less than k total elements.
 *
 * @param d_offsets Offsets for each segment
 * @param k Number of values to keep in each segment
 * @param d_indices Mark these indices to be removed
 * @param d_segment_sizes Store actual sizes of each segment
 */
CUDF_KERNEL void resolve_segment_indices(device_span<size_type const> d_offsets,
                                         size_type k,
                                         device_span<size_type> d_indices,
                                         size_type* d_segment_sizes)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= d_indices.size()) { return; }

  auto const sitr = thrust::upper_bound(thrust::seq, d_offsets.begin(), d_offsets.end(), tid);
  // Mark rows outside all segments for removal (offsets need not cover all rows).
  if (sitr == d_offsets.begin() || sitr == d_offsets.end()) {
    d_indices[tid] = -1;
    return;
  }
  auto const segment_start = *(sitr - 1);
  auto const segment_end   = *sitr;
  auto const index         = tid - segment_start;
  if (index >= k) { d_indices[tid] = -1; }  // mark values outside of top k

  if (index == 0) {
    auto const segment_size  = segment_end - segment_start;
    auto const segment_index = cuda::std::distance(d_offsets.begin(), sitr) - 1;
    // segment is k or less elements
    d_segment_sizes[segment_index] = cuda::std::min(k, segment_size);
  }
}

/**
 * @brief Computes top-k indices per segment with a full segmented sort
 */
std::unique_ptr<column> sort_based_segmented_top_k_order(column_view const& col,
                                                         column_view const& segment_offsets,
                                                         size_type k,
                                                         order topk_order,
                                                         cuda::stream_ref stream,
                                                         cudf::memory_resources mr)
{
  auto const size_data_type = data_type{type_to_id<size_type>()};

  auto const nulls   = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto const temp_mr = mr.get_temporary_mr();
  auto const indices = cudf::detail::segmented_sorted_order(
    cudf::table_view({col}), segment_offsets, {topk_order}, {nulls}, stream, temp_mr);
  auto const d_indices = indices->mutable_view().begin<size_type>();

  // Zero-initialized because resolve_segment_indices writes a segment's size only from its
  // first element; an empty segment has none, so its slot must remain 0, not uninitialized.
  auto segment_sizes = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    segment_offsets.size() - 1, stream, temp_mr);
  auto span_indices = device_span<size_type>{d_indices, static_cast<std::size_t>(indices->size())};
  auto const grid   = cudf::detail::grid_1d(indices->size(), 256);
  resolve_segment_indices<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
    segment_offsets, k, span_indices, segment_sizes.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  auto [offsets, total_elements] =
    cudf::detail::make_offsets_child_column(segment_sizes.begin(), segment_sizes.end(), stream, mr);

  auto result = cudf::make_fixed_width_column(
    size_data_type, total_elements, mask_state::UNALLOCATED, stream, mr.get_output_mr());
  auto d_result = result->mutable_view().begin<size_type>();
  // remove the indices marked by resolve_segment_indices
  thrust::remove_copy(
    rmm::exec_policy_nosync(stream, temp_mr), d_indices, d_indices + indices->size(), d_result, -1);

  auto const num_rows = static_cast<size_type>(offsets->size() - 1);
  return make_lists_column(num_rows,
                           std::move(offsets),
                           std::move(result),
                           0,
                           cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

/**
 * @brief Computes top-k indices per segment with one `cub::DeviceTopK` call per segment
 */
template <typename T>
std::unique_ptr<column> cub_based_segmented_top_k_order(column_view const& col,
                                                        std::span<size_type const> h_offsets,
                                                        size_type k,
                                                        order topk_order,
                                                        cuda::stream_ref stream,
                                                        cudf::memory_resources mr)
{
  auto const num_segments = static_cast<size_type>(h_offsets.size()) - 1;
  auto const temp_mr      = mr.get_temporary_mr();

  // Segment indices and sizes
  auto const segments     = std::views::iota(size_type{0}, num_segments);
  auto const segment_size = [&](size_type segment_idx) {
    return h_offsets[segment_idx + 1] - h_offsets[segment_idx];
  };
  auto const max_segment_size = std::ranges::max(segments | std::views::transform(segment_size));

  // Compute output offsets and indices
  auto host_offsets = cudf::detail::make_pinned_vector_async<size_type>(num_segments + 1, stream);
  host_offsets.front() = 0;
  std::transform_inclusive_scan(
    segments.begin(),
    segments.end(),
    host_offsets.begin() + 1,
    std::plus{},
    [&](auto const segment_idx) { return std::min(segment_size(segment_idx), k); });

  auto offsets = std::make_unique<column>(
    cudf::detail::make_device_uvector_async(host_offsets, stream, mr.get_output_mr()),
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
    0);
  auto indices = rmm::device_uvector<size_type>(host_offsets.back(), stream, temp_mr);

  auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                               cuda::execution::output_ordering::unsorted);
  auto env          = cuda::std::execution::env{stream, requirements};

  auto select_top_k = [&](void* tmp, std::size_t& tmp_size, size_type segment_idx, size_type size) {
    auto const keys_in  = col.begin<T>() + h_offsets[segment_idx];
    auto const keys_out = cuda::make_discard_iterator();
    auto const vals_in  = cuda::counting_iterator<size_type>{h_offsets[segment_idx]};
    auto const vals_out = indices.data() + host_offsets[segment_idx];

    if (topk_order == order::ASCENDING) {
      return cub::DeviceTopK::MinPairs(
        tmp, tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env);
    }
    return cub::DeviceTopK::MaxPairs(
      tmp, tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env);
  };

  // Temporary storage grows with the segment size, so the largest segment sizes it for all.
  auto tmp_size = std::size_t{0};
  if (max_segment_size > k) { CUDF_CUDA_TRY(select_top_k(nullptr, tmp_size, 0, max_segment_size)); }
  auto tmp = cuda::device_buffer<std::byte>(stream, temp_mr, tmp_size, cuda::no_init);

  for (auto const segment_idx : segments) {
    auto const size = segment_size(segment_idx);
    if (size == 0) { continue; }
    if (size <= k) {
      thrust::sequence(rmm::exec_policy_nosync(stream, temp_mr),
                       indices.begin() + host_offsets[segment_idx],
                       indices.begin() + host_offsets[segment_idx + 1],
                       h_offsets[segment_idx]);
      continue;
    }
    auto segment_tmp_size = tmp_size;
    CUDF_CUDA_TRY(select_top_k(tmp.data(), segment_tmp_size, segment_idx, size));
  }

  // Sort each segment's selection by value to match the sort-based path.
  auto const indices_view    = column_view(device_span<size_type const>{indices});
  auto const keys            = cudf::detail::gather(cudf::table_view({col}),
                                         indices_view,
                                         out_of_bounds_policy::DONT_CHECK,
                                         negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         memory_resources{temp_mr, temp_mr});
  auto const column_order    = std::vector<order>{topk_order};
  auto const null_precedence = std::vector<null_order>{
    topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE};
  auto sorted = cudf::detail::segmented_sort_by_key(cudf::table_view({indices_view}),
                                                    keys->view(),
                                                    offsets->view(),
                                                    column_order,
                                                    null_precedence,
                                                    stream,
                                                    mr.get_output_mr());

  return make_lists_column(num_segments,
                           std::move(offsets),
                           std::move(sorted->release().front()),
                           0,
                           cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}
}  // namespace

std::unique_ptr<column> segmented_top_k_order(column_view const& col,
                                              column_view const& segment_offsets,
                                              size_type k,
                                              order topk_order,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  auto const size_data_type = data_type{type_to_id<size_type>()};
  if (k == 0 || col.is_empty()) { return cudf::make_empty_lists_column(size_data_type); }

  CUDF_EXPECTS(segment_offsets.size() > 0,
               "segment_offsets must have at least one element",
               std::invalid_argument);

  CUDF_EXPECTS(segment_offsets.type() == size_data_type,
               "segment_offsets must be of type INT32",
               cudf::data_type_error);
  CUDF_EXPECTS(segment_offsets.null_count() == 0,
               "segment_offsets must not have nulls",
               std::invalid_argument);

  // `col.size()` bounds the rows covered by the segments, so this check avoids copying the
  // offsets to host for inputs that cannot take the DeviceTopK path.
  if (auto const num_segments = segment_offsets.size() - 1;
      is_cub_top_k_supported(col) and use_cub_based_segmented_top_k(col.size(), num_segments, k)) {
    auto const h_offsets = cudf::detail::make_pinned_vector(
      device_span<size_type const>{segment_offsets.begin<size_type>(),
                                   static_cast<std::size_t>(num_segments) + 1},
      stream);
    // Malformed offsets keep the sort-based path's failure behavior.
    if (h_offsets.front() >= 0 && h_offsets.back() <= col.size() &&
        cuda::std::is_sorted(h_offsets.begin(), h_offsets.end()) and
        use_cub_based_segmented_top_k(h_offsets.back() - h_offsets.front(), num_segments, k)) {
      return type_dispatcher<dispatch_storage_type>(
        col.type(), dispatch_cub_top_k_key{[&]<typename T>() {
          return cub_based_segmented_top_k_order<T>(
            col, {h_offsets.data(), h_offsets.size()}, k, topk_order, stream, mr);
        }});
    }
  }

  return sort_based_segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
}

std::unique_ptr<column> segmented_top_k(column_view const& col,
                                        column_view const& segment_offsets,
                                        size_type k,
                                        order topk_order,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  if (col.is_empty()) { return cudf::make_empty_column(col.type()); }

  auto ordered =
    cudf::detail::segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
  auto lv = cudf::lists_column_view(ordered->view());
  if (lv.is_empty()) { return cudf::make_empty_lists_column(col.type()); }

  auto result         = cudf::detail::gather(cudf::table_view({col}),
                                     lv.child(),
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  auto offsets        = std::move(ordered->release().children.front());
  auto const num_rows = static_cast<size_type>(offsets->size() - 1);
  return make_lists_column(num_rows,
                           std::move(offsets),
                           std::move(result->release().front()),
                           0,
                           cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

}  // namespace detail

std::unique_ptr<column> segmented_top_k(column_view const& col,
                                        column_view const& segment_offsets,
                                        size_type k,
                                        order topk_order,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_top_k(col, segment_offsets, k, topk_order, stream, mr);
}

std::unique_ptr<column> segmented_top_k_order(column_view const& col,
                                              column_view const& segment_offsets,
                                              size_type k,
                                              order topk_order,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
}
}  // namespace cudf
