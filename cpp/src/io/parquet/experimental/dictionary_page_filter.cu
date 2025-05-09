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
#include <numeric>
#include <optional>

namespace cudf::io::parquet::experimental::detail {

using parquet::detail::chunk_page_info;
using parquet::detail::ColumnChunkDesc;
using parquet::detail::decode_error;
using parquet::detail::PageInfo;

namespace {
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

  cudf::io::parquet::kernel_error error_code(stream);
  DecodePageHeaders(
    chunks.device_begin(), d_chunk_page_info.begin(), chunks.size(), error_code.data(), stream);

  if (auto const error = error_code.value_sync(stream); error != 0) {
    CUDF_FAIL("Parquet header parsing failed with code(s) " +
              cudf::io::parquet::kernel_error::to_string(error));
  }

  // Setup dictionary page for each chunk
  thrust::for_each(rmm::exec_policy_nosync(stream),
                   pages.device_begin(),
                   pages.device_end(),
                   [chunks = chunks.device_begin()] __device__(PageInfo const& p) {
                     if (p.flags & cudf::io::parquet::detail::PAGEINFO_FLAGS_DICTIONARY) {
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

template <typename T, CUDF_ENABLE_IF(cuda::std::is_same_v<T, string_view>)>
bool decode_dictionary_and_evaluate(
  uint8_t const* buffer, size_t length, T literal, parquet::Type type, size_type flba_length = 0)
{
  size_t offset = 0;
  while (offset < length) {
    switch (type) {
      case parquet::Type::BYTE_ARRAY: {
        CUDF_EXPECTS(offset + 4 <= length, "Truncated byte array length");
        auto len = static_cast<int32_t>(*(buffer + offset));
        offset += 4;
        CUDF_EXPECTS(offset + len <= length, "Truncated byte array data");
        auto v = std::string_view{reinterpret_cast<char const*>(buffer + offset),
                                  static_cast<size_t>(len)};
        auto const compare =
          std::string_view{literal.data(), static_cast<size_t>(literal.size_bytes())};
        if (v == compare) { return true; }
        offset += len;
        break;
      }
      case parquet::Type::INT96: {
        auto constexpr int96_size = 12;
        CUDF_EXPECTS(flba_length == int96_size, "INT96 must be 12 bytes");
        [[fallthrough]];
      }
      case parquet::Type::FIXED_LEN_BYTE_ARRAY: {
        CUDF_EXPECTS(flba_length > 0, "FLBA length must be > 0");
        CUDF_EXPECTS(offset + flba_length <= length, "Truncated FLBA");
        auto v = std::string_view{reinterpret_cast<char const*>(buffer + offset),
                                  static_cast<size_t>(flba_length)};
        auto const compare =
          std::string_view{literal.data(), static_cast<size_t>(literal.size_bytes())};
        if (v == compare) { return true; }
        offset += flba_length;
        break;
      }
      default: throw std::runtime_error("Unsupported type");
    }
  }

  return false;
}

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
        auto constexpr int96_size = 12;
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
  __device__ bool operator()(T const v) const { return v == scalar.value<T>(); }
};

template <typename T>
__global__ std::enable_if_t<not std::is_same_v<T, bool> and
                              not(cudf::is_compound<T>() and not std::is_same_v<T, string_view>),
                            void>
evaluate_fixed_dictionary(PageInfo const* pages,
                          bool* results,
                          ast::generic_scalar_device_view scalar,
                          cudf::size_type num_dictionary_columns,
                          cudf::size_type dictionary_col_idx,
                          cudf::size_type flba_length = 0)
{
  if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
    return;
  } else {
    namespace cg             = cooperative_groups;
    auto const row_group_idx = cg::this_grid().block_rank();
    auto const chunk_idx     = dictionary_col_idx + (row_group_idx * num_dictionary_columns);
    auto const& page         = pages[chunk_idx];

    auto const& page_data = page.page_data;
    auto break_early      = false;

    // Initialize the result to false
    results[row_group_idx] = false;

    auto const group = cg::this_thread_block();
    for (auto value_idx = group.thread_rank(); value_idx < page.num_input_values;
         value_idx += group.num_threads()) {
      if (break_early) {
        break;
      } else {
        // Decode the little endian value from page data
        auto decoded_value = T{};
        cuda::std::memcpy(&decoded_value, page_data + (value_idx * sizeof(T)), sizeof(T));
        // Check if the decoded value is equal to the literal value
        if (is_equal_to_scalar_value<T>{scalar}(decoded_value)) {
          results[row_group_idx] = true;
          break_early            = true;
          break;
        }
      }
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

  template <typename T>
  struct copy_scalar_to_host {
    ast::generic_scalar_device_view d_scalar;
    T* h_scalar;
    __device__ void operator()(int i) const
    {
      if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
        *h_scalar = d_scalar.value<T>();

      } else {
        *h_scalar = d_scalar.value<T>();
      }
    }
  };

  template <typename T>
  std::vector<std::unique_ptr<cudf::column>> operator()(
    cudf::data_type dtype, cudf::host_span<ast::literal* const> literals)
  {
    // Boolean, List, Struct, Dictionary types are not supported
    if constexpr (cuda::std::is_same_v<T, bool> or
                  (cudf::is_compound<T>() and not cuda::std::is_same_v<T, string_view>)) {
      CUDF_FAIL("Dictionaries do not support boolean or compound types");
    } else {
      auto columns = std::vector<std::unique_ptr<cudf::column>>{};
      columns.reserve(literals.size());
      // If there is only one literal, then just evaluate expression while decoding dictionary
      // data
      if (literals.size() == 1) {
        auto& literal = literals.front();
        // Check if the literal has the same type as the predicate column
        CUDF_EXPECTS(
          dtype == literal->get_data_type() and
            cudf::have_same_types(
              cudf::column_view{dtype, 0, {}, {}, 0, 0, {}},
              cudf::scalar_type_t<T>(T{}, false, stream, cudf::get_current_device_resource_ref())),
          "Mismatched predicate column and literal types");

        rmm::device_buffer results{
          total_row_groups, stream, cudf::get_current_device_resource_ref()};
        cudf::device_span<bool> results_span{static_cast<bool*>(results.data()), total_row_groups};

        if constexpr (not cuda::std::is_same_v<T, cudf::string_view>) {
          cudf::detail::grid_1d config(total_row_groups, 256);
          evaluate_fixed_dictionary<T>
            <<<config.num_blocks, config.num_blocks, 0, stream.value()>>>(pages.device_begin(),
                                                                          results_span.data(),
                                                                          literal->get_value(),
                                                                          num_dictionary_columns,
                                                                          dictionary_col_idx);
        } else {
          auto host_results = cudf::detail::make_host_vector<bool>(total_row_groups, stream);
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

              rmm::device_scalar<T> scalar{T{}, stream, cudf::get_current_device_resource_ref()};
              thrust::for_each(rmm::exec_policy(stream),
                               thrust::counting_iterator(0),
                               thrust::counting_iterator(1),
                               copy_scalar_to_host<T>{literal->get_value(), scalar.data()});

              T literal_value = scalar.value(stream);
              std::vector<char> str_data{};
              if constexpr (cuda::std::is_same_v<T, cudf::string_view>) {
                str_data.resize(literal_value.size_bytes());
                cudf::detail::cuda_memcpy<char>(
                  cudf::host_span<char>{str_data.data(),
                                        static_cast<size_t>(literal_value.size_bytes())},
                  cudf::device_span<char const>{literal_value.data(),
                                                static_cast<size_t>(literal_value.size_bytes())},
                  stream);
                literal_value =
                  cudf::string_view{str_data.data(), static_cast<cudf::size_type>(str_data.size())};
              }

              // Decode the dictionary data
              host_results[row_group_idx] = decode_dictionary_and_evaluate<T>(host_page_data.data(),
                                                                              host_page_data.size(),
                                                                              literal_value,
                                                                              chunk.physical_type,
                                                                              chunk.type_length);
            });
          cudf::detail::cuda_memcpy_async<bool>(results_span, host_results, stream);
        }

        columns.emplace_back(
          std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::BOOL8},
                                         static_cast<cudf::size_type>(total_row_groups),
                                         std::move(results),
                                         rmm::device_buffer{},
                                         0));
      }
      // Else, decode dictionaries and then evaluate all expressions
      else {
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
          // Check if the literal has the same type as the predicate column
          CUDF_EXPECTS(
            dtype == literal->get_data_type() and
              cudf::have_same_types(cudf::column_view{dtype, 0, {}, {}, 0, 0, {}},
                                    cudf::scalar_type_t<T>(
                                      T{}, false, stream, cudf::get_current_device_resource_ref())),
            "Mismatched predicate column and literal types");

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
      }
      return columns;
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
  // Number of dictionary pages or number of column chunks
  auto const total_dictionaries = dictionary_page_data.size();

  // Boolean to check if any of the column chunnks have compressed data
  [[maybe_unused]] auto has_compressed_data = false;

  // Initialize column chunk descriptors
  auto chunks = cudf::detail::hostdevice_vector<cudf::io::parquet::detail::ColumnChunkDesc>(
    total_dictionaries, stream);
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

      auto const logical_type = [&]() -> std::optional<LogicalType> {
        auto const column_type_id =
          cudf::io::parquet::detail::to_type_id(schema,
                                                options.is_enabled_convert_strings_to_categories(),
                                                options.get_timestamp_type().id());
        if (schema.logical_type.has_value() and schema.logical_type->type == LogicalType::DECIMAL) {
          // if decimal but not outputting as float or decimal, then convert to no logical type
          if (column_type_id != type_id::FLOAT64 and
              not cudf::is_fixed_point(data_type{column_type_id})) {
            return std::nullopt;
          }
        }
        return schema.logical_type;
      }();

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
                                          schema.max_definition_level,
                                          schema.max_repetition_level,
                                          _metadata->get_output_nesting_depth(col_schema_idx),
                                          0,  // Not needed
                                          0,  // Not needed
                                          col_meta.codec,
                                          logical_type,
                                          0,  // Not needed
                                          0,  // not needed
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
  cudf::detail::hostdevice_vector<PageInfo> pages(total_dictionaries, stream);

  // Decode dictionary page headers
  decode_dictionary_page_headers(chunks, pages, stream);

  return {std::move(chunks), std::move(pages)};
}

#if 0
// TODO: Decompress dictionary pages here

// Plain encoded decompressed dictionary data
auto dict_page_data = cudf::detail::make_host_vector<uint8_t>(
  cudf::device_span<uint8_t>(pages.begin()->page_data, pages.begin()->uncompressed_page_size),
  stream);

auto dict_page_data_int32 = decode_plain_dictionary<int32_t>(
  dict_page_data.data(), dict_page_data.size(), parquet::Type::INT32, chunks.begin()->type_length);

auto d_dict_page_data_int32 = cudf::detail::make_device_uvector_async<int32_t>(
  dict_page_data_int32, stream, cudf::get_current_device_resource_ref());

using hasher_type              = cudf::hashing::detail::default_hash<size_type>;
[[maybe_unused]] auto dict_set = cuco::static_set{
  {compute_hash_table_size(dict_page_data_int32.size())},
  cuco::empty_key<cudf::size_type>{std::numeric_limits<int32_t>::max()},
  cuda::std::equal_to<int32_t>{},
  cuco::linear_probing<1, hasher_type>{},
  {},
  {},
  cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
  stream.value()};

dict_set.insert(d_dict_page_data_int32.begin(), d_dict_page_data_int32.end(), stream.value());

  // A better way to do this would be to create a bulk storage for all chunks of this column and then create sub-hashsets for each chunk
 auto dict_page_data_int32 =
    decode_plain_dictionary(dict_page_data.data(),
                                                               dict_page_data.size(),
                                                               parquet::Type::INT32,
                                                               chunks.begin()->type_length);

  auto d_dict_page_data_int32 = cudf::detail::make_device_uvector_async<int32_t>(
    dict_page_data_int32, stream, cudf::get_current_device_resource_ref());

  using hasher_type = cudf::hashing::detail::default_hash<size_type>;
  using storage_type =
    cuco::bucket_storage<int32_t, 1, cuco::extent<std::size_t>, cudf::detail::cuco_allocator<char>>;
  using storage_ref_type = typename storage_type::ref_type;
  using bucket_type      = typename storage_type::bucket_type;

  auto const num_keys = compute_hash_table_size(dict_page_data_int32.size());
  auto set_storage    = storage_type{
    num_keys, cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream}};

  storage_ref_type const storage_ref{num_keys, set_storage.data()};

  [[maybe_unused]] auto dict_set =
    cuco::static_set_ref{cuco::empty_key<cudf::size_type>{std::numeric_limits<int32_t>::max()},
                         cuda::std::equal_to<int32_t>{},
                         cuco::linear_probing<1, hasher_type>{},
                         cuco::thread_scope_block,
                         storage_ref};

  auto dict_set_insert_ref = dict_set.rebind_operators(cuco::insert);

  dict_set_insert_ref.insert(d_dict_page_data_int32.begin(), d_dict_page_data_int32.end());

// decompress the data pages in this subpass; also decompress the dictionary pages in this pass,
// if this is the first subpass in the pass
// if (has_compressed_data) {
//  [[maybe_unused]] auto [pass_data, subpass_data] =
//    decompress_page_data(dchunks, pages, host_span<PageInfo>{}, _page_mask, _stream);
//}

// [[maybe_unused]] auto str_dict_index =
//   build_string_dict_indices_dict_pages(dchunks, pages, stream);
#endif

}  // namespace cudf::io::parquet::experimental::detail
