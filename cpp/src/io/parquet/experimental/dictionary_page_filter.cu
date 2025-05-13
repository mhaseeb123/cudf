/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hybrid_scan_helpers.hpp"
#include "hybrid_scan_impl.hpp"
#include "io/parquet/parquet_gpu.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>

#include <future>
#include <optional>

namespace cudf::io::parquet::experimental::detail {

using parquet::detail::chunk_page_info;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::decode_error;
using parquet::detail::PageInfo;

constexpr cudf::size_type int96_size        = 12;
constexpr cudf::size_type decode_block_size = 128;

// cuCollections hash set parameters
using key_type                    = cudf::size_type;
auto constexpr empty_key_sentinel = std::numeric_limits<key_type>::max();
auto constexpr set_cg_size        = 1;
auto constexpr bucket_size        = 1;
auto constexpr occupancy_factor   = 70;  // cuCollections suggests targeting a 70% occupancy factor
using storage_type                = cuco::bucket_storage<key_type,
                                          bucket_size,
                                          cuco::extent<std::size_t>,
                                          cudf::detail::cuco_allocator<char>>;
using storage_ref_type            = typename storage_type::ref_type;
using bucket_type                 = typename storage_type::bucket_type;

namespace {

template <typename T>
struct equality_functor {
  cudf::device_span<T> const decoded_data;
  __device__ bool operator()(key_type lhs_idx, key_type rhs_idx) const
  {
    return decoded_data[lhs_idx] == decoded_data[rhs_idx];
  }
};

template <typename T>
struct hash_functor {
  cudf::device_span<T> const decoded_data;
  uint32_t const seed = 0;
  __device__ auto operator()(key_type idx) const
  {
    if constexpr (cudf::is_timestamp<T>() or cudf::is_duration<T>()) {
      return cudf::hashing::detail::MurmurHash3_x86_32<int>::result_type{0};
    } else {
      return cudf::hashing::detail::MurmurHash3_x86_32<T>{seed}(decoded_data[idx]);
    }
  }
};

template <typename T>
__global__
  std::enable_if_t<not std::is_same_v<T, bool> and
                     not(cudf::is_compound<T>() and not std::is_same_v<T, cudf::string_view>),
                   void>
  query_dictionaries(cudf::device_span<T> decoded_data,
                     cudf::device_span<bool*> results,
                     cudf::device_span<ast::generic_scalar_device_view> scalars,
                     cudf::device_span<bucket_type> const set_storage,
                     cudf::device_span<cudf::size_type const> set_offsets,
                     size_t total_row_groups,
                     parquet::Type physical_type)
{
  namespace cg           = cooperative_groups;
  auto const literal_idx = cg::this_grid().block_rank();
  auto const scalar      = scalars[literal_idx];
  auto result            = results[literal_idx];

  using equality_fn_type = equality_functor<T>;
  using hash_fn_type     = hash_functor<T>;
  // Choosing `linear_probing` over `double_hashing` for slighhhtly better performance seen in
  // benchmarks.
  using probing_scheme_type = cuco::linear_probing<set_cg_size, hash_fn_type>;

  for (auto value_idx = cg::this_thread_block().thread_rank(); value_idx < total_row_groups;
       value_idx += cg::this_thread_block().size()) {
    storage_ref_type const storage_ref{set_offsets[value_idx + 1] - set_offsets[value_idx],
                                       set_storage.data() + set_offsets[value_idx]};
    auto hash_set_ref = cuco::static_set_ref{cuco::empty_key{empty_key_sentinel},
                                             equality_fn_type{decoded_data},
                                             probing_scheme_type{hash_fn_type{decoded_data}},
                                             cuco::thread_scope_block,
                                             storage_ref};

    auto set_find_ref  = hash_set_ref.rebind_operators(cuco::contains);
    auto literal_value = scalar.value<T>();

    if constexpr (std::is_same_v<T, cudf::string_view>) {
      if (physical_type == parquet::Type::INT96) {
        auto const int128_key = static_cast<__int128_t>(scalar.value<int64_t>());
        cudf::string_view probe_key{reinterpret_cast<char const*>(&int128_key), int96_size};
        literal_value = probe_key;
      }
    }

    result[value_idx] = set_find_ref.contains(literal_value);
  }
}

__global__ void build_string_dictionaries(PageInfo const* pages,
                                          cudf::device_span<cudf::string_view> decoded_data,
                                          cudf::device_span<bucket_type> const set_storage,
                                          cudf::device_span<cudf::size_type const> set_offsets,
                                          cudf::device_span<cudf::size_type const> value_offsets,
                                          cudf::size_type num_dictionary_columns,
                                          cudf::size_type dictionary_col_idx,
                                          kernel_error::pointer error)
{
  namespace cg    = cooperative_groups;
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const row_group_idx =
    (cg::this_grid().block_rank() * warp.meta_group_size()) + warp.meta_group_rank();
  auto const chunk_idx      = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page          = pages[chunk_idx];
  auto const& page_data     = page.page_data;
  auto const page_data_size = page.uncompressed_page_size;
  auto const value_offset   = value_offsets[row_group_idx];
  storage_ref_type const storage_ref{set_offsets[row_group_idx + 1] - set_offsets[row_group_idx],
                                     set_storage.data() + set_offsets[row_group_idx]};

  using equality_fn_type = equality_functor<cudf::string_view>;
  using hash_fn_type     = hash_functor<cudf::string_view>;
  // Choosing `linear_probing` over `double_hashing` for slighhhtly better performance seen in
  // benchmarks.
  using probing_scheme_type = cuco::linear_probing<set_cg_size, hash_fn_type>;

  auto hash_set_ref = cuco::static_set_ref{cuco::empty_key<key_type>{empty_key_sentinel},
                                           equality_fn_type{decoded_data},
                                           probing_scheme_type{hash_fn_type{decoded_data}},
                                           cuco::thread_scope_thread,
                                           storage_ref};

  auto set_insert_ref = hash_set_ref.rebind_operators(cuco::insert);

  // If empty buffer or no input values, then return early
  if (page.num_input_values == 0 or page_data_size == 0) { return; }

  // Helper to check data stream overrun
  auto const is_stream_overrun = [&](size_type offset, size_type length) {
    return offset + length > page_data_size;
  };

  // Helper to set error
  auto const set_error = [&](decode_error error_value) {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{*error};
    ref.fetch_or(static_cast<kernel_error::value_type>(error_value),
                 cuda::std::memory_order_relaxed);
  };

  // Decode with single warp thread until the value is found or we reach the end of the page
  if (warp.thread_rank() == 0) {
    auto buffer_offset  = 0;
    auto decoded_values = 0;
    while (buffer_offset < page_data_size) {
      if (decoded_values > page.num_input_values or
          is_stream_overrun(buffer_offset, sizeof(int32_t))) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
        break;
      }

      // Decode string length
      auto const string_length = static_cast<int32_t>(*(page_data + buffer_offset));
      buffer_offset += sizeof(int32_t);
      if (is_stream_overrun(buffer_offset, string_length)) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
        break;
      }

      // Decode cudf::string_view value
      auto const decoded_value =
        cudf::string_view{reinterpret_cast<char const*>(page_data + buffer_offset),
                          static_cast<cudf::size_type>(string_length)};

      decoded_data[value_offset + decoded_values] = decoded_value;
      set_insert_ref.insert(value_offset + decoded_values);

      // Otherwise, keep going
      buffer_offset += string_length;
      decoded_values++;
    }
  }
}

template <typename T>
__global__ std::enable_if_t<not std::is_same_v<T, bool> and
                              not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>),
                            void>
build_fixed_width_dictionaries(PageInfo const* pages,
                               cudf::device_span<T> decoded_data,
                               cudf::device_span<bucket_type> const set_storage,
                               cudf::device_span<cudf::size_type const> set_offsets,
                               cudf::device_span<cudf::size_type const> value_offsets,
                               parquet::Type physical_type,
                               cudf::size_type num_dictionary_columns,
                               cudf::size_type dictionary_col_idx,
                               kernel_error::pointer error,
                               cudf::size_type flba_length = 0)
{
  namespace cg             = cooperative_groups;
  auto const group         = cg::this_thread_block();
  auto const row_group_idx = cg::this_grid().block_rank();
  auto const chunk_idx     = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page         = pages[chunk_idx];
  auto const& page_data    = page.page_data;
  auto const value_offset  = value_offsets[row_group_idx];
  storage_ref_type const storage_ref{set_offsets[row_group_idx + 1] - set_offsets[row_group_idx],
                                     set_storage.data() + set_offsets[row_group_idx]};

  using equality_fn_type = equality_functor<T>;
  using hash_fn_type     = hash_functor<T>;
  // Choosing `linear_probing` over `double_hashing` for slighhhtly better performance seen in
  // benchmarks.
  using probing_scheme_type = cuco::linear_probing<set_cg_size, hash_fn_type>;

  auto hash_set_ref = cuco::static_set_ref{cuco::empty_key{empty_key_sentinel},
                                           equality_fn_type{decoded_data},
                                           probing_scheme_type{hash_fn_type{decoded_data}},
                                           cuco::thread_scope_block,
                                           storage_ref};

  auto set_insert_ref = hash_set_ref.rebind_operators(cuco::insert);

  // If empty buffer or no input values, then return early
  if (page.num_input_values == 0 or page.uncompressed_page_size == 0) { return; }

  // Helper to check data stream overrun
  auto const is_stream_overrun = [&](size_type offset, size_type length) {
    return offset + length > page.uncompressed_page_size;
  };

  // Helper to set error
  auto const set_error = [&](decode_error error_value) {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{*error};
    ref.fetch_or(static_cast<kernel_error::value_type>(error_value),
                 cuda::std::memory_order_relaxed);
  };

  auto const is_error_set = [&]() {
    return cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block>{*error}.load(
             cuda::std::memory_order_relaxed) != 0;
  };

  for (auto value_idx = group.thread_rank(); value_idx < page.num_input_values;
       value_idx += group.num_threads()) {
    // Return early if an error has been set
    if (is_error_set()) { return; }

    if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
      // Parquet physical type must be fixed length so either INT96 or FIXED_LEN_BYTE_ARRAY
      switch (physical_type) {
        case parquet::Type::INT96: flba_length = int96_size; [[fallthrough]];
        case parquet::Type::FIXED_LEN_BYTE_ARRAY: {
          // Check if we are overruning the data stream
          if (is_stream_overrun(value_idx * flba_length, flba_length)) {
            set_error(decode_error::DATA_STREAM_OVERRUN);
            return;
          }
          decoded_data[value_offset + value_idx] = cudf::string_view{
            reinterpret_cast<char const*>(page_data) + value_idx * flba_length, flba_length};
          set_insert_ref.insert(value_offset + value_idx);

          break;
        }
        default: {
          // Parquet physical type is not fixed length so set the error code and break early
          set_error(decode_error::INVALID_DATA_TYPE);
          return;
        }
      }
    } else {
      // Check if we are overruning the data stream
      if (is_stream_overrun(value_idx * sizeof(T), sizeof(T))) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
        return;
      }
      // Simply copy over the decoded value bytes from page data
      cuda::std::memcpy(
        &decoded_data[value_offset + value_idx], page_data + (value_idx * sizeof(T)), sizeof(T));
      set_insert_ref.insert(value_offset + value_idx);
    }
  }
}

/**
 * @brief Decode the page information for a given pass.
 *
 * @param chunks Host device span of column chunk descriptors, one per input column chunk
 * @param pages Host device span of empty page headers to fill in, one per input column chunk
 * @param stream CUDA stream
 */
void decode_dictionary_page_headers(cudf::detail::hostdevice_span<ColumnChunkDesc> chunks,
                                    cudf::detail::hostdevice_span<PageInfo> pages,
                                    rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  std::vector<size_t> host_chunk_page_counts(chunks.size() + 1);
  std::transform(
    chunks.host_begin(), chunks.host_end(), host_chunk_page_counts.begin(), [](auto const& chunk) {
      return chunk.num_dict_pages;
    });
  host_chunk_page_counts[chunks.size()] = 0;

  auto chunk_page_counts = cudf::detail::make_device_uvector_async(
    host_chunk_page_counts, stream, cudf::get_current_device_resource_ref());

  thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                         chunk_page_counts.begin(),
                         chunk_page_counts.end(),
                         chunk_page_counts.begin(),
                         size_t{0},
                         thrust::plus<size_t>{});

  rmm::device_uvector<chunk_page_info> d_chunk_page_info(chunks.size(), stream);

  thrust::for_each(rmm::exec_policy_nosync(stream),
                   thrust::counting_iterator<cuda::std::size_t>(0),
                   thrust::counting_iterator(chunks.size()),
                   [cpi               = d_chunk_page_info.begin(),
                    chunk_page_counts = chunk_page_counts.begin(),
                    pages             = pages.device_begin()] __device__(size_t i) {
                     cpi[i].pages = &pages[chunk_page_counts[i]];
                   });

  parquet::kernel_error error_code(stream);

  DecodePageHeaders(
    chunks.device_begin(), d_chunk_page_info.begin(), chunks.size(), error_code.data(), stream);

  if (auto const error = error_code.value_sync(stream); error != 0) {
    CUDF_FAIL("Parquet header parsing failed with code(s) " +
              parquet::kernel_error::to_string(error));
  }

  // Setup dictionary page for each chunk
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   pages.device_begin(),
                   pages.device_end(),
                   [chunks = chunks.device_begin()] __device__(PageInfo const& p) {
                     if (p.flags & parquet::detail::PAGEINFO_FLAGS_DICTIONARY) {
                       chunks[p.chunk_idx].dict_page = &p;
                     }
                   });

  pages.device_to_host_async(stream);
  chunks.device_to_host_async(stream);
  stream.synchronize();
}

template <typename T>
inline T read_le(const uint8_t* buf)
{
  T result;
  std::copy(buf, buf + sizeof(T), reinterpret_cast<uint8_t*>(&result));
  return result;
}

template <typename T,
          CUDF_ENABLE_IF(not cuda::std::is_same_v<T, bool> and not cudf::is_compound<T>())>
std::vector<T> decode_plain_dictionary(uint8_t const* buffer,
                                       size_t length,
                                       parquet::Type type,
                                       uint8_t* d_buffer,
                                       size_type flba_length = 0)
{
  std::vector<T> result;
  size_t offset = 0;
  while (offset < length) {
    switch (type) {
      case parquet::Type::INT32: [[fallthrough]];
      case parquet::Type::INT64: [[fallthrough]];
      case parquet::Type::FLOAT: [[fallthrough]];
      case parquet::Type::DOUBLE: {
        CUDF_EXPECTS(offset + sizeof(T) <= length, "Truncated value");
        auto v = read_le<T>(buffer + offset);
        result.push_back(v);
        offset += sizeof(T);
        break;
      }

      default: throw std::runtime_error("Unsupported type");
    }
  }

  return result;
}

template <typename T,
          CUDF_ENABLE_IF(not cuda::std::is_same_v<T, bool> and not cudf::is_compound<T>())>
bool decode_dictionary_and_evaluate(
  uint8_t const* buffer, size_t length, T literal, parquet::Type type, size_type flba_length = 0)
{
  size_t offset = 0;
  while (offset < length) {
    switch (type) {
      case parquet::Type::INT32: [[fallthrough]];
      case parquet::Type::INT64: [[fallthrough]];
      case parquet::Type::FLOAT: [[fallthrough]];
      case parquet::Type::DOUBLE: {
        CUDF_EXPECTS(offset + sizeof(T) <= length, "Truncated value");
        auto v = read_le<T>(buffer + offset);
        if (v == literal) { return true; }
        offset += sizeof(T);
        break;
      }
      default: throw std::runtime_error("Unsupported type");
    }
  }

  return false;
}

#if 0
template <typename T, CUDF_ENABLE_IF(cuda::std::is_same_v<T, cudf::string_view>)>
std::vector<T> decode_plain_dictionary(uint8_t const* buffer,
                                       size_t length,
                                       parquet::Type type,
                                       uint8_t* d_buffer,
                                       size_type flba_length = 0)
{
  std::vector<T> result;
  size_t offset = 0;
  while (offset < length) {
    switch (type) {
      case parquet::Type::BYTE_ARRAY: {
        CUDF_EXPECTS(offset + 4 <= length, "Truncated byte array length");
        auto len = static_cast<int32_t>(*(buffer + offset));
        offset += 4;
        CUDF_EXPECTS(offset + len <= length, "Truncated byte array data");
        result.emplace_back(
          cudf::string_view{reinterpret_cast<char const*>(d_buffer + offset), len});
        offset += len;
        break;
      }
      case parquet::Type::INT96: {
        CUDF_EXPECTS(flba_length == int96_size, "INT96 must be 12 bytes");
        [[fallthrough]];
      }
      case parquet::Type::FIXED_LEN_BYTE_ARRAY: {
        CUDF_EXPECTS(flba_length > 0, "FLBA length must be > 0");
        CUDF_EXPECTS(offset + flba_length <= length, "Truncated FLBA");
        result.emplace_back(
          cudf::string_view{reinterpret_cast<char const*>(d_buffer + offset), flba_length});
        offset += flba_length;
        break;
      }
      default: throw std::runtime_error("Unsupported type");
    }
  }

  return result;
}

template <typename T>
struct is_equal_to_scalar_value {
  ast::generic_scalar_device_view scalar;
  __device__ bool operator()(T const decoded_value) const
  {
    return decoded_value == scalar.value<T>();
  }
};
#endif

__global__ void evaluate_one_string_literal(PageInfo const* pages,
                                            bool* results,
                                            ast::generic_scalar_device_view scalar,
                                            cudf::size_type num_dictionary_columns,
                                            cudf::size_type dictionary_col_idx,
                                            kernel_error::pointer error)
{
  namespace cg    = cooperative_groups;
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const row_group_idx =
    (cg::this_grid().block_rank() * warp.meta_group_size()) + warp.meta_group_rank();

  auto const chunk_idx      = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page          = pages[chunk_idx];
  auto const& page_data     = page.page_data;
  auto const page_data_size = page.uncompressed_page_size;

  results[row_group_idx] = false;

  // If empty buffer or no input values, then return early
  if (page.num_input_values == 0 or page_data_size == 0) { return; }

  // Helper to check data stream overrun
  auto const is_stream_overrun = [&](size_type offset, size_type length) {
    return offset + length > page_data_size;
  };

  // Helper to set error
  auto const set_error = [&](decode_error error_value) {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{*error};
    ref.fetch_or(static_cast<kernel_error::value_type>(error_value),
                 cuda::std::memory_order_relaxed);
  };

  // Decode with single warp thread until the value is found or we reach the end of the page
  if (warp.thread_rank() == 0) {
    auto buffer_offset  = 0;
    auto decoded_values = 0;
    while (buffer_offset < page_data_size) {
      if (decoded_values > page.num_input_values or
          is_stream_overrun(buffer_offset, sizeof(int32_t))) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
        break;
      }
      // Decode string length
      auto const string_length = static_cast<int32_t>(*(page_data + buffer_offset));
      buffer_offset += sizeof(int32_t);
      if (is_stream_overrun(buffer_offset, string_length)) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
        break;
      }

      // Decode cudf::string_view value
      auto const decoded_value =
        cudf::string_view{reinterpret_cast<char const*>(page_data + buffer_offset),
                          static_cast<cudf::size_type>(string_length)};

      // If the literal is found, set the result to true and break
      if (decoded_value == scalar.value<cudf::string_view>()) {
        results[row_group_idx] = true;
        break;
      }
      // Otherwise, keep going
      buffer_offset += string_length;
      decoded_values++;
    }
  }
}

template <typename T>
__global__ std::enable_if_t<not std::is_same_v<T, bool> and
                              not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>),
                            void>
evaluate_one_fixed_width_literal(PageInfo const* pages,
                                 bool* results,
                                 ast::generic_scalar_device_view scalar,
                                 parquet::Type physical_type,
                                 cudf::size_type num_dictionary_columns,
                                 cudf::size_type dictionary_col_idx,
                                 kernel_error::pointer error,
                                 cudf::size_type flba_length = 0)
{
  namespace cg             = cooperative_groups;
  auto const group         = cg::this_thread_block();
  auto const row_group_idx = cg::this_grid().block_rank();
  auto const chunk_idx     = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
  auto const& page         = pages[chunk_idx];
  auto const& page_data    = page.page_data;

  results[row_group_idx] = false;

  // If empty buffer or no input values, then return early
  if (page.num_input_values == 0 or page.uncompressed_page_size == 0) { return; }

  // Helper to check data stream overrun
  auto const is_stream_overrun = [&](size_type offset, size_type length) {
    return offset + length > page.uncompressed_page_size;
  };

  // Helper to set error
  auto const set_error = [&](decode_error error_value) {
    cuda::atomic_ref<kernel_error::value_type, cuda::thread_scope_block> ref{*error};
    ref.fetch_or(static_cast<kernel_error::value_type>(error_value),
                 cuda::std::memory_order_relaxed);
  };

  auto break_early = false;

  for (auto value_idx = group.thread_rank(); value_idx < page.num_input_values;
       value_idx += group.num_threads()) {
    // If we have already found a match or an error, then return early
    if (break_early) { return; }

    // Placeholder for the decoded value
    auto decoded_value = T{};

    if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
      // Parquet physical type must be fixed length so either INT96 or FIXED_LEN_BYTE_ARRAY
      switch (physical_type) {
        case parquet::Type::INT96: flba_length = int96_size; [[fallthrough]];
        case parquet::Type::FIXED_LEN_BYTE_ARRAY: {
          // Check if we are overruning the data stream
          if (is_stream_overrun(value_idx * flba_length, flba_length)) {
            set_error(decode_error::DATA_STREAM_OVERRUN);
            break_early = true;
            return;
          }
          decoded_value = cudf::string_view{
            reinterpret_cast<char const*>(page_data) + value_idx * flba_length, flba_length};
          break;
        }
        default: {
          // Parquet physical type is not fixed length so set the error code and break early
          set_error(decode_error::INVALID_DATA_TYPE);
          break_early = true;
          return;
        }
      }
    } else {
      // Check if we are overruning the data stream
      if (is_stream_overrun(value_idx * sizeof(T), sizeof(T))) {
        set_error(decode_error::DATA_STREAM_OVERRUN);
        break_early = true;
        return;
      }
      // Simply copy over the decoded value bytes from page data
      cuda::std::memcpy(&decoded_value, page_data + (value_idx * sizeof(T)), sizeof(T));
    }

    // If the decoded value is equal to the scalar value, set the result to true and return early
    if (decoded_value == scalar.value<T>()) {
      results[row_group_idx] = true;
      break_early            = true;
      return;
    }
  }
}

/**
 * @brief Converts dictionary membership results (for each column chunk) to a device column.
 */
struct dictionary_caster {
  cudf::detail::hostdevice_span<parquet::detail::ColumnChunkDesc const> chunks;
  cudf::detail::hostdevice_span<parquet::detail::PageInfo const> pages;
  size_t total_row_groups;
  parquet::Type physical_type;
  cudf::size_type type_length;
  cudf::size_type num_dictionary_columns;
  cudf::size_type dictionary_col_idx;
  rmm::cuda_stream_view stream;

#if 0
  [[maybe_unused]] void host_evaluate_one_string_literal(cudf::device_span<bool> results_span,
                                                         ast::literal* const literal)
  {
    auto host_results = cudf::detail::make_host_vector<bool>(total_row_groups, stream);

    // Copy generic scalar to string scalar
    rmm::device_scalar<cudf::string_view> str_scalar{
      cudf::string_view{}, stream, cudf::get_current_device_resource_ref()};

    cudf::detail::device_single_thread(
      [generic_scalar = literal->get_value(), string_scalar = str_scalar.data()] __device__() {
        *string_scalar = generic_scalar.value<cudf::string_view>();
      },
      stream);

    cudf::string_view host_literal_value = str_scalar.value(stream);
    auto host_literal_bytes              = std::vector<char>(host_literal_value.size_bytes());
    cudf::detail::cuda_memcpy<char>(
      host_literal_bytes,
      cudf::device_span<char const>{host_literal_value.data(),
                                    static_cast<size_t>(host_literal_value.size_bytes())},
      stream);

    // Convert the literal value to std::string_view to use `==` host operator
    auto const literal_value =
      std::string_view{host_literal_bytes.data(), host_literal_bytes.size()};

    auto const find_string_in_dictionary = [&](cudf::size_type row_group_idx) {
      auto const chunk_idx = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
      auto const& chunk    = chunks[chunk_idx];
      auto const& page     = pages[chunk_idx];

      // Copy the dictionary data to host
      auto host_page_data = cudf::detail::make_host_vector<uint8_t>(
        cudf::device_span<uint8_t>(page.page_data, page.uncompressed_page_size), stream);

      // Decode the dictionary data
      host_results[row_group_idx] = [&]() {
        auto const buffer        = host_page_data.data();
        auto const buffer_length = host_page_data.size();

        if (buffer_length == 0) { return false; }

        size_t current_offset = 0;
        while (current_offset < buffer_length) {
          CUDF_EXPECTS(current_offset + sizeof(int32_t) <= buffer_length,
                       "Truncated byte array length");
          auto const string_length = static_cast<int32_t>(*(buffer + current_offset));
          current_offset += sizeof(int32_t);
          CUDF_EXPECTS(current_offset + string_length <= buffer_length,
                       "Truncated byte array data");
          // Decode as std::string_view to use `==` host operator
          auto const decoded_value =
            std::string_view{reinterpret_cast<char const*>(buffer + current_offset),
                             static_cast<size_t>(string_length)};
          // If the literal is found, return true
          if (decoded_value == literal_value) { return true; }
          // Otherwise, keep going
          current_offset += string_length;
        }
        // Literal not found, return false
        return false;
      }();
    };

    // Submit tasks to evaluate string literal in each column chunk's dictionary
    auto string_literal_evaluator_tasks = std::vector<std::future<void>>{};
    string_literal_evaluator_tasks.reserve(total_row_groups);
    std::for_each(
      thrust::counting_iterator<size_t>(0),
      thrust::counting_iterator(total_row_groups),
      [&](auto row_group_idx) {
        string_literal_evaluator_tasks.emplace_back(
          cudf::detail::host_worker_pool().submit_task([&find_string_in_dictionary, row_group_idx] {
            find_string_in_dictionary(row_group_idx);
          }));
      });

    // Wait for all tasks to complete
    std::for_each(string_literal_evaluator_tasks.begin(),
                  string_literal_evaluator_tasks.end(),
                  [&](auto& task) { task.wait(); });

    cudf::detail::cuda_memcpy_async<bool>(results_span, host_results, stream);
  }

#endif

  template <typename T>
  std::enable_if_t<not std::is_same_v<T, bool> and
                     not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>),
                   std::vector<std::unique_ptr<cudf::column>>>
  evaluate_multiple_literals(cudf::host_span<ast::literal* const> literals, size_t total_num_values)
  {
    std::vector<cudf::size_type> set_offsets;
    std::vector<cudf::size_type> value_offsets;
    set_offsets.reserve(total_row_groups + 1);
    value_offsets.reserve(total_row_groups + 1);
    set_offsets.emplace_back(0);
    value_offsets.emplace_back(0);
    std::for_each(
      thrust::counting_iterator<size_t>(0),
      thrust::counting_iterator(total_row_groups),
      [&](auto row_group_idx) {
        auto const chunk_idx = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
        value_offsets.emplace_back(value_offsets.back() + pages[chunk_idx].num_input_values);
        set_offsets.emplace_back(set_offsets.back() +
                                 static_cast<cudf::size_type>(compute_hash_table_size(
                                   pages[chunk_idx].num_input_values, occupancy_factor)));
      });

    auto const total_bucket_storage_size = static_cast<size_t>(set_offsets.back());

    auto const d_set_offsets = cudf::detail::make_device_uvector_async(
      set_offsets, stream, cudf::get_current_device_resource_ref());

    auto const d_value_offsets = cudf::detail::make_device_uvector_async(
      value_offsets, stream, cudf::get_current_device_resource_ref());

    // Create a single bulk storage used by all sub-dictionaries
    auto set_storage = storage_type{
      total_bucket_storage_size,
      cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream}};
    // Initialize storage with the empty key sentinel
    set_storage.initialize_async(empty_key_sentinel, {stream.value()});
    // Device span of the set storage to use in the kernels
    cudf::device_span<bucket_type> const set_storage_data{set_storage.data(),
                                                          total_bucket_storage_size};
    rmm::device_uvector<T> decoded_data{
      total_num_values, stream, cudf::get_current_device_resource_ref()};
    kernel_error error_code(stream);

    auto columns = std::vector<std::unique_ptr<cudf::column>>{};
    columns.reserve(literals.size());

    std::vector<ast::generic_scalar_device_view> generic_scalars;
    generic_scalars.reserve(literals.size());
    std::for_each(literals.begin(), literals.end(), [&](auto const& literal) {
      generic_scalars.push_back(std::move(literal->get_value()));
    });
    auto d_generic_scalars = cudf::detail::make_device_uvector_async(
      generic_scalars, stream, cudf::get_current_device_resource_ref());

    std::vector<rmm::device_buffer> results(literals.size());
    thrust::host_vector<bool*> results_ptrs(literals.size());
    std::for_each(thrust::counting_iterator<size_t>(0),
                  thrust::counting_iterator(literals.size()),
                  [&](auto i) {
                    results[i] = rmm::device_buffer(
                      total_row_groups, stream, cudf::get_current_device_resource_ref());
                    results_ptrs[i] = static_cast<bool*>(results[i].data());
                  });

    auto d_results_ptrs = cudf::detail::make_device_uvector_async(
      results_ptrs, stream, cudf::get_current_device_resource_ref());

    if constexpr (not cuda::std::is_same_v<T, cudf::string_view>) {
      build_fixed_width_dictionaries<T>
        <<<total_row_groups, decode_block_size, 0, stream.value()>>>(pages.device_begin(),
                                                                     decoded_data,
                                                                     set_storage_data,
                                                                     d_set_offsets,
                                                                     d_value_offsets,
                                                                     physical_type,
                                                                     num_dictionary_columns,
                                                                     dictionary_col_idx,
                                                                     error_code.data());

      // Check if there are any errors in data decoding
      if (auto const error = error_code.value_sync(stream); error != 0) {
        CUDF_FAIL("Dictionary decode failed with code(s) " + kernel_error::to_string(error));
      }

      query_dictionaries<T>
        <<<total_row_groups, decode_block_size, 0, stream.value()>>>(decoded_data,
                                                                     d_results_ptrs,
                                                                     d_generic_scalars,
                                                                     set_storage_data,
                                                                     d_set_offsets,
                                                                     total_row_groups,
                                                                     physical_type);

    } else {
      if (physical_type == parquet::Type::INT96 or
          physical_type == parquet::Type::FIXED_LEN_BYTE_ARRAY) {
        // Get flba length from the first column chunk of this column
        auto const flba_length = physical_type == parquet::Type::INT96
                                   ? int96_size
                                   : chunks[dictionary_col_idx].type_length;
        // Check if the fixed width literal is in the dictionaries
        build_fixed_width_dictionaries<T>
          <<<total_row_groups, decode_block_size, 0, stream.value()>>>(pages.device_begin(),
                                                                       decoded_data,
                                                                       set_storage_data,
                                                                       d_set_offsets,
                                                                       d_value_offsets,
                                                                       physical_type,
                                                                       num_dictionary_columns,
                                                                       dictionary_col_idx,
                                                                       error_code.data(),
                                                                       flba_length);
        // Check if there are any errors in data decoding
        if (auto const error = error_code.value_sync(stream); error != 0) {
          CUDF_FAIL("Dictionary decode failed with code(s) " + kernel_error::to_string(error));
        }

        query_dictionaries<T>
          <<<total_row_groups, decode_block_size, 0, stream.value()>>>(decoded_data,
                                                                       d_results_ptrs,
                                                                       d_generic_scalars,
                                                                       set_storage_data,
                                                                       d_set_offsets,
                                                                       total_row_groups,
                                                                       physical_type);
      } else {
        // Check if the fixed width literal is in the dictionaries
        build_string_dictionaries<<<total_row_groups, decode_block_size, 0, stream.value()>>>(
          pages.device_begin(),
          decoded_data,
          set_storage_data,
          d_set_offsets,
          d_value_offsets,
          num_dictionary_columns,
          dictionary_col_idx,
          error_code.data());

        // Check if there are any errors in data decoding
        if (auto const error = error_code.value_sync(stream); error != 0) {
          CUDF_FAIL("Dictionary decode failed with code(s) " + kernel_error::to_string(error));
        }

        query_dictionaries<cudf::string_view>
          <<<total_row_groups, decode_block_size, 0, stream.value()>>>(decoded_data,
                                                                       d_results_ptrs,
                                                                       d_generic_scalars,
                                                                       set_storage_data,
                                                                       d_set_offsets,
                                                                       total_row_groups,
                                                                       physical_type);
      }
    }

    std::transform(results.begin(), results.end(), std::back_inserter(columns), [&](auto& result) {
      return std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::BOOL8},
                                            static_cast<cudf::size_type>(total_row_groups),
                                            std::move(result),
                                            rmm::device_buffer{},
                                            0);
    });

    return columns;
  }

  template <typename T>
  std::enable_if_t<not std::is_same_v<T, bool> and
                     not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>),
                   std::vector<std::unique_ptr<cudf::column>>>
  evaluate_one_literal(ast::literal* const literal)
  {
    rmm::device_buffer results{total_row_groups, stream, cudf::get_current_device_resource_ref()};
    cudf::device_span<bool> results_span{static_cast<bool*>(results.data()), total_row_groups};
    kernel_error error_code(stream);

    auto columns = std::vector<std::unique_ptr<cudf::column>>{};

    if constexpr (not cuda::std::is_same_v<T, cudf::string_view>) {
      // Check if the numeric literal is in the dictionaries
      evaluate_one_fixed_width_literal<T>
        <<<total_row_groups, decode_block_size, 0, stream.value()>>>(pages.device_begin(),
                                                                     results_span.data(),
                                                                     literal->get_value(),
                                                                     physical_type,
                                                                     num_dictionary_columns,
                                                                     dictionary_col_idx,
                                                                     error_code.data());
    } else {
      if (physical_type == parquet::Type::INT96 or
          physical_type == parquet::Type::FIXED_LEN_BYTE_ARRAY) {
        // Get flba length from the first column chunk of this column
        auto const flba_length = physical_type == parquet::Type::INT96
                                   ? int96_size
                                   : chunks[dictionary_col_idx].type_length;
        // Check if the fixed width literal is in the dictionaries
        evaluate_one_fixed_width_literal<T>
          <<<total_row_groups, decode_block_size, 0, stream.value()>>>(pages.device_begin(),
                                                                       results_span.data(),
                                                                       literal->get_value(),
                                                                       physical_type,
                                                                       num_dictionary_columns,
                                                                       dictionary_col_idx,
                                                                       error_code.data(),
                                                                       flba_length);
      } else {
        // Check on host if the string literal is in the dictionaries
        // host_evaluate_one_string_literal(results_span, literal);

        static_assert(decode_block_size % cudf::detail::warp_size == 0,
                      "decode_block_size must be a multiple of warp_size");

        // We need one warp per row group
        size_t const warps_per_block = decode_block_size / cudf::detail::warp_size;
        auto const num_blocks =
          cudf::util::div_rounding_up_safe<size_t>(total_row_groups, warps_per_block);

        // Check if the string literal is in the dictionaries
        evaluate_one_string_literal<<<num_blocks, decode_block_size, 0, stream.value()>>>(
          pages.device_begin(),
          results_span.data(),
          literal->get_value(),
          num_dictionary_columns,
          dictionary_col_idx,
          error_code.data());
      }
    }

    // Check if there are any errors in data decoding
    if (auto const error = error_code.value_sync(stream); error != 0) {
      CUDF_FAIL("Dictionary decode failed with code(s) " + kernel_error::to_string(error));
    }

    // Add the results column to the output
    columns.emplace_back(
      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::BOOL8},
                                     static_cast<cudf::size_type>(total_row_groups),
                                     std::move(results),
                                     rmm::device_buffer{},
                                     0));
    return columns;
  }

  template <typename T>
  std::vector<std::unique_ptr<cudf::column>> operator()(
    cudf::data_type dtype, cudf::host_span<ast::literal* const> literals)
  {
    // Boolean, List, Struct, Dictionary types are not supported
    if constexpr (cuda::std::is_same_v<T, bool> or
                  (cudf::is_compound<T>() and not cuda::std::is_same_v<T, string_view>)) {
      CUDF_FAIL("Dictionaries do not support boolean or compound types");
    } else {
      // Make sure all literals have the same type as the predicate column
      std::for_each(literals.begin(), literals.end(), [&](auto const& literal) {
        // Check if the literal has the same type as the predicate column
        CUDF_EXPECTS(
          dtype == literal->get_data_type() and
            cudf::have_same_types(
              cudf::column_view{dtype, 0, {}, {}, 0, 0, {}},
              cudf::scalar_type_t<T>(T{}, false, stream, cudf::get_current_device_resource_ref())),
          "Mismatched predicate column and literal types");
      });

      // If there is only one literal, then just evaluate expression while decoding dictionary
      // data
      if (literals.size() == 1) {
        return evaluate_one_literal<T>(literals.front());
      }
      // Else, decode dictionaries to `cudf::static_set`s and evaluate all expressions
      else {
        return evaluate_multiple_literals<T>(literals, total_row_groups);
#if 0
        auto columns = std::vector<std::unique_ptr<cudf::column>>{};
        columns.reserve(literals.size());
        kernel_error error_code(stream);

        auto device_data = std::vector<rmm::device_uvector<T>>{};
        device_data.reserve(literals.size());
        std::for_each(
          thrust::counting_iterator<size_t>(0),
          thrust::counting_iterator(total_row_groups),
          [&](auto row_group_idx) {
            auto const chunk_idx = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
            auto const& chunk    = chunks[chunk_idx];
            auto const& page     = pages[chunk_idx];

            // Copy the dictionary data to host
            auto host_page_data = cudf::detail::make_host_vector<uint8_t>(
              cudf::device_span<uint8_t>(page.page_data, page.uncompressed_page_size), stream);

            // Decode the dictionary data
            auto const host_decoded_data = decode_plain_dictionary<T>(host_page_data.data(),
                                                                      host_page_data.size(),
                                                                      chunk.physical_type,
                                                                      page.page_data,
                                                                      chunk.type_length);

            device_data.push_back(cudf::detail::make_device_uvector<T>(
              host_decoded_data, stream, cudf::get_current_device_resource_ref()));
          });

        // Evaluate all expressions now
        std::for_each(literals.begin(), literals.end(), [&](auto const& literal) {
          auto host_results = cudf::detail::make_host_vector<bool>(total_row_groups, stream);
          std::for_each(thrust::counting_iterator<size_t>(0),
                        thrust::counting_iterator(total_row_groups),
                        [&](auto row_group_idx) {
                          auto const& device_decoded_data = device_data[row_group_idx];
                          host_results[row_group_idx] =
                            thrust::any_of(rmm::exec_policy(stream),
                                           device_decoded_data.begin(),
                                           device_decoded_data.end(),
                                           is_equal_to_scalar_value<T>{literal->get_value()});
                        });

          rmm::device_buffer results{
            total_row_groups, stream, cudf::get_current_device_resource_ref()};
          cudf::device_span<bool> results_span{static_cast<bool*>(results.data()),
                                               total_row_groups};

          cudf::detail::cuda_memcpy_async<bool>(results_span, host_results, stream);

          columns.emplace_back(
            std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::BOOL8},
                                           static_cast<cudf::size_type>(total_row_groups),
                                           std::move(results),
                                           rmm::device_buffer{},
                                           0));
        });

        return columns;
#endif
      }
    }
  }
};

/**
 * @brief Converts AST expression to dictionary membership (DictionaryAST) expression.
 * This is used in row group filtering based on equality predicate.
 */
class dictionary_expression_converter : public equality_literals_collector {
 public:
  dictionary_expression_converter(ast::expression const& expr,
                                  size_type num_input_columns,
                                  cudf::host_span<std::vector<ast::literal*> const> literals)
    : _literals{literals}
  {
    // Set the num columns
    _num_input_columns = num_input_columns;

    // Compute and store columns literals offsets
    _col_literals_offsets.reserve(_num_input_columns + 1);
    _col_literals_offsets.emplace_back(0);

    std::transform(literals.begin(),
                   literals.end(),
                   std::back_inserter(_col_literals_offsets),
                   [&](auto const& col_literal_map) {
                     return _col_literals_offsets.back() +
                            static_cast<cudf::size_type>(col_literal_map.size());
                   });

    // Add this visitor
    expr.accept(*this);
  }

  /**
   * @brief Delete equality literals getter as it's not needed in the derived class
   */
  [[nodiscard]] std::vector<std::vector<ast::literal*>> get_equality_literals() && = delete;

  // Bring all overloads of `visit` from equality_predicate_collector into scope
  using equality_literals_collector::visit;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override
  {
    using cudf::ast::ast_operator;
    auto const operands = expr.get_operands();
    auto const op       = expr.get_operator();

    if (auto* v = dynamic_cast<ast::column_reference const*>(&operands[0].get())) {
      // First operand should be column reference, second should be literal.
      CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 2,
                   "Only binary operations are supported on column reference");
      CUDF_EXPECTS(dynamic_cast<ast::literal const*>(&operands[1].get()) != nullptr,
                   "Second operand of binary operation with column reference must be a literal");
      v->accept(*this);

      if (op == ast_operator::EQUAL or op == ast::ast_operator::NOT_EQUAL) {
        // Search the literal in this input column's equality literals list and add to
        // the offset.
        auto const col_idx            = v->get_column_index();
        auto const& equality_literals = _literals[col_idx];
        auto col_literal_offset       = _col_literals_offsets[col_idx];
        auto const literal_iter       = std::find(equality_literals.cbegin(),
                                            equality_literals.cend(),
                                            dynamic_cast<ast::literal const*>(&operands[1].get()));
        CUDF_EXPECTS(literal_iter != equality_literals.end(), "Could not find the literal ptr");
        col_literal_offset += std::distance(equality_literals.cbegin(), literal_iter);

        // Evaluate boolean is_true(value) expression as NOT(NOT(value))
        auto const& value = _dictionary_expr.push(ast::column_reference{col_literal_offset});

        auto const& not_in_dictionary =
          _dictionary_expr.push(ast::operation{ast_operator::NOT, value});

        if (op == ast_operator::EQUAL) {
          _dictionary_expr.push(ast::operation{ast_operator::NOT, not_in_dictionary});
        }
      }
      // For all other expressions, push an always true expression
      else {
        _dictionary_expr.push(
          ast::operation{ast_operator::NOT,
                         _dictionary_expr.push(ast::operation{ast_operator::NOT, _always_true})});
      }
    } else {
      auto new_operands = visit_operands(operands);
      if (cudf::ast::detail::ast_operator_arity(op) == 2) {
        _dictionary_expr.push(ast::operation{op, new_operands.front(), new_operands.back()});
      } else if (cudf::ast::detail::ast_operator_arity(op) == 1) {
        _dictionary_expr.push(ast::operation{op, new_operands.front()});
      }
    }
    return _dictionary_expr.back();
  }

  /**
   * @brief Returns the AST to apply on dictionary membership.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::reference_wrapper<ast::expression const> get_dictionary_expr() const
  {
    return _dictionary_expr.back();
  }

 private:
  std::vector<cudf::size_type> _col_literals_offsets;
  cudf::host_span<std::vector<ast::literal*> const> _literals;
  ast::tree _dictionary_expr;
  cudf::numeric_scalar<bool> _always_true_scalar{true};
  ast::literal const _always_true{_always_true_scalar};
};

}  // namespace

dictionary_literals_collector::dictionary_literals_collector() = default;

dictionary_literals_collector::dictionary_literals_collector(ast::expression const& expr,
                                                             cudf::size_type num_input_columns)
{
  _num_input_columns = num_input_columns;
  _literals.resize(num_input_columns);
  expr.accept(*this);
}

std::reference_wrapper<ast::expression const> dictionary_literals_collector::visit(
  ast::operation const& expr)
{
  using cudf::ast::ast_operator;
  auto const operands = expr.get_operands();
  auto const op       = expr.get_operator();

  if (auto* v = dynamic_cast<ast::column_reference const*>(&operands[0].get())) {
    // First operand should be column reference, second should be literal.
    CUDF_EXPECTS(cudf::ast::detail::ast_operator_arity(op) == 2,
                 "Only binary operations are supported on column reference");
    auto const literal_ptr = dynamic_cast<ast::literal const*>(&operands[1].get());
    CUDF_EXPECTS(literal_ptr != nullptr,
                 "Second operand of binary operation with column reference must be a literal");
    v->accept(*this);

    // Push to the corresponding column's literals and operators list iff EQUAL or NOT_EQUAL
    // operator is seen
    if (op == ast_operator::EQUAL or op == ast::ast_operator::NOT_EQUAL) {
      auto const col_idx = v->get_column_index();
      _literals[col_idx].emplace_back(const_cast<ast::literal*>(literal_ptr));
    }
  } else {
    // Just visit the operands and ignore any output
    std::ignore = visit_operands(operands);
  }

  return expr;
}

std::optional<std::vector<std::vector<size_type>>>
aggregate_reader_metadata::apply_dictionary_filter(
  cudf::detail::hostdevice_span<parquet::detail::ColumnChunkDesc const> chunks,
  cudf::detail::hostdevice_span<parquet::detail::PageInfo const> pages,
  host_span<std::vector<size_type> const> input_row_group_indices,
  host_span<std::vector<ast::literal*> const> literals,
  std::size_t total_row_groups,
  cudf::host_span<data_type const> output_dtypes,
  cudf::host_span<int const> dictionary_col_schemas,
  std::reference_wrapper<ast::expression const> filter,
  rmm::cuda_stream_view stream) const
{
  // Number of input table columns
  auto const num_input_columns = static_cast<cudf::size_type>(output_dtypes.size());
  // Number of dictionary columns
  auto const num_dictionary_columns = static_cast<cudf::size_type>(dictionary_col_schemas.size());

  // Get parquet types for the predicate columns
  auto const parquet_types = get_parquet_types(input_row_group_indices, dictionary_col_schemas);

  // Converts dictionary membership for (in)equality predicate columns to a table
  // containing a column for each `col[i] == literal` or `col[i] != literal` predicate to be
  // evaluated. The table contains #sources * #column_chunks_per_src rows.
  std::vector<std::unique_ptr<cudf::column>> dictionary_membership_columns;
  cudf::size_type dictionary_col_idx = 0;
  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator(output_dtypes.size()),
                [&](auto input_col_idx) {
                  auto const& dtype = output_dtypes[input_col_idx];

                  // Skip if no equality literals for this column
                  if (literals[input_col_idx].empty()) { return; }

                  // Skip if non-comparable (compound) type except string
                  if (cudf::is_compound(dtype) and dtype.id() != cudf::type_id::STRING) { return; }

                  auto const type_length = chunks[dictionary_col_idx].type_length;

                  // Create a bloom filter query table caster
                  dictionary_caster const dictionary_col{chunks,
                                                         pages,
                                                         total_row_groups,
                                                         parquet_types[dictionary_col_idx],
                                                         type_length,
                                                         num_dictionary_columns,
                                                         dictionary_col_idx,
                                                         stream};

                  // Add a column for all literals associated with an equality column
                  auto dict_columns = cudf::type_dispatcher<dispatch_storage_type>(
                    dtype, dictionary_col, dtype, literals[input_col_idx]);

                  dictionary_membership_columns.insert(
                    dictionary_membership_columns.end(),
                    std::make_move_iterator(dict_columns.begin()),
                    std::make_move_iterator(dict_columns.end()));

                  dictionary_col_idx++;
                });

  // Create a table from columns
  [[maybe_unused]] auto dictionary_membership_table =
    cudf::table(std::move(dictionary_membership_columns));

  // Convert AST to DictionaryAST expression with reference to dictionary membership
  // in above `dictionary_membership_table`
  dictionary_expression_converter dictionary_expr{filter.get(), num_input_columns, literals};

  // Filter dictionary membership table with the DictionaryAST expression and collect
  // filtered row group indices
  return parquet::detail::collect_filtered_row_group_indices(dictionary_membership_table,
                                                             dictionary_expr.get_dictionary_expr(),
                                                             input_row_group_indices,
                                                             stream);
}

std::pair<cudf::detail::hostdevice_vector<ColumnChunkDesc>,
          cudf::detail::hostdevice_vector<PageInfo>>
hybrid_scan_reader_impl::prepare_dictionaries(
  cudf::host_span<std::vector<size_type> const> row_group_indices,
  cudf::host_span<rmm::device_buffer> dictionary_page_data,
  cudf::host_span<int const> dictionary_col_schemas,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream)
{
  // Create row group information for the input row group indices
  auto const row_groups_info =
    std::get<2>(_metadata->select_row_groups(row_group_indices, 0, std::nullopt));

  CUDF_EXPECTS(row_groups_info.size() * _input_columns.size() == dictionary_page_data.size(),
               "Dictionary page data size must match the number of row groups times the number of "
               "input columns");

  // Number of input columns
  auto const num_input_columns = _input_columns.size();
  // Number of column chunks
  auto const total_column_chunks = dictionary_page_data.size();

  // Boolean to check if any of the column chunnks have compressed data
  [[maybe_unused]] auto has_compressed_data = false;

  // Initialize column chunk descriptors
  auto chunks = cudf::detail::hostdevice_vector<cudf::io::parquet::detail::ColumnChunkDesc>(
    total_column_chunks, stream);
  auto chunk_idx = 0;

  // For all row groups
  for (auto const& rg : row_groups_info) {
    auto const& row_group = _metadata->get_row_group(rg.index, rg.source_index);

    // For all columns with dictionary page and (in)equality predicate
    for (auto col_schema_idx : dictionary_col_schemas) {
      // look up metadata
      auto& col_meta = _metadata->get_column_metadata(rg.index, rg.source_index, col_schema_idx);
      auto& schema   = _metadata->get_schema(
        _metadata->map_schema_index(col_schema_idx, rg.source_index), rg.source_index);

      // dictionary data buffer for this column chunk
      auto& dict_page_data = dictionary_page_data[chunk_idx];

      // Check if the column chunk has compressed data
      has_compressed_data =
        col_meta.codec != Compression::UNCOMPRESSED and col_meta.total_compressed_size > 0;

      // Create a column chunk
      chunks[chunk_idx] = ColumnChunkDesc(static_cast<int64_t>(dict_page_data.size()),
                                          static_cast<uint8_t*>(dict_page_data.data()),
                                          col_meta.num_values,
                                          schema.type,
                                          schema.type_length,
                                          0,  // Not needed
                                          0,  // Not needed
                                          0,  // Not needed
                                          0,  // Not needed
                                          0,  // Not needed
                                          0,  // Not needed
                                          0,  // Not needed
                                          col_meta.codec,
                                          parquet::LogicalType::UNDEFINED,  // Not needed
                                          0,                                // Not needed
                                          0,                                // Not needed
                                          col_schema_idx,
                                          nullptr,  // Not needed
                                          0.0f,     // Not needed
                                          false,    // Not needed
                                          rg.source_index);
      // Set the number of dictionary and data pages
      chunks[chunk_idx].num_dict_pages = (dict_page_data.size() > 0);
      chunks[chunk_idx].num_data_pages = 0;
      chunk_idx++;
    }
  }

  // Copy the column chunk descriptors to the device
  chunks.host_to_device_async(stream);

  // Create page infos for each column chunk's dictionary page
  cudf::detail::hostdevice_vector<PageInfo> pages(total_column_chunks, stream);

  // Decode dictionary page headers
  decode_dictionary_page_headers(chunks, pages, stream);

  return {std::move(chunks), std::move(pages)};
}

}  // namespace cudf::io::parquet::experimental::detail
