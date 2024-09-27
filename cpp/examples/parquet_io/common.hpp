/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#pragma once

#include "../utilities/timer.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <fmt/color.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

/**
 * @brief Create memory resource for libcudf functions
 *
 * @param pool Whether to use a pool memory resource.
 * @return Memory resource instance
 */
std::shared_ptr<rmm::mr::device_memory_resource> create_memory_resource(bool is_pool_used)
{
  auto cuda_mr = std::make_shared<rmm::mr::cuda_memory_resource>();
  if (is_pool_used) {
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      cuda_mr, rmm::percent_of_free_device_memory(50));
  }
  return cuda_mr;
}

/**
 * @brief Get encoding type from the keyword
 *
 * @param name encoding keyword name
 * @return corresponding column encoding type
 */
[[nodiscard]] cudf::io::column_encoding get_encoding_type(std::string name)
{
  using encoding_type = cudf::io::column_encoding;

  static const std::unordered_map<std::string_view, encoding_type> map = {
    {"DEFAULT", encoding_type::USE_DEFAULT},
    {"DICTIONARY", encoding_type::DICTIONARY},
    {"PLAIN", encoding_type::PLAIN},
    {"DELTA_BINARY_PACKED", encoding_type::DELTA_BINARY_PACKED},
    {"DELTA_LENGTH_BYTE_ARRAY", encoding_type::DELTA_LENGTH_BYTE_ARRAY},
    {"DELTA_BYTE_ARRAY", encoding_type::DELTA_BYTE_ARRAY},
  };

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) { return map.at(name); }
  throw std::invalid_argument("FATAL: " + std::string(name) +
                              " is not a valid encoding type.\n\n"
                              "Available encoding types: DEFAULT, DICTIONARY, PLAIN,\n"
                              "DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY,\n"
                              "DELTA_BYTE_ARRAY\n\n");
}

/**
 * @brief Get compression type from the keyword
 *
 * @param name compression keyword name
 * @return corresponding compression type
 */
[[nodiscard]] cudf::io::compression_type get_compression_type(std::string name)
{
  using compression_type = cudf::io::compression_type;

  static const std::unordered_map<std::string_view, compression_type> map = {
    {"NONE", compression_type::NONE},
    {"AUTO", compression_type::AUTO},
    {"SNAPPY", compression_type::SNAPPY},
    {"LZ4", compression_type::LZ4},
    {"ZSTD", compression_type::ZSTD}};

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) { return map.at(name); }
  throw std::invalid_argument("FATAL: " + std::string(name) +
                              " is not a valid compression type.\n\n"
                              "Available compression_type types: NONE, AUTO, SNAPPY,\n"
                              "LZ4, ZSTD\n\n");
}

/**
 * @brief Get boolean from they keyword
 *
 * @param input keyword affirmation string such as: Y, T, YES, TRUE, ON
 * @return true or false
 */
[[nodiscard]] bool get_boolean(std::string input)
{
  std::transform(input.begin(), input.end(), input.begin(), ::toupper);

  // Check if the input string matches to any of the following
  if (not input.compare("ON") or not input.compare("TRUE") or not input.compare("YES") or
      not input.compare("Y") or not input.compare("T")) {
    return true;
  } else {
    return false;
  }
}

/**
 * @brief Check if two tables are identical, throw an error otherwise
 *
 * @param lhs_table View to lhs table
 * @param rhs_table View to rhs table
 */
inline void check_identical_tables(cudf::table_view const& lhs_table,
                                   cudf::table_view const& rhs_table)
{
  try {
    // Left anti-join the original and transcoded tables
    // identical tables should not throw an exception and
    // return an empty indices vector
    auto const indices = cudf::left_anti_join(lhs_table, rhs_table, cudf::null_equality::EQUAL);

    // No exception thrown, check indices
    auto const valid = indices->size() == 0;
    fmt::print(
      fmt::emphasis::bold | fg(fmt::color::green_yellow), "Tables identical: {}\n\n", valid);
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl << std::endl;
    throw std::runtime_error(
      fmt::format(fmt::emphasis::bold | fg(fmt::color::red), "Tables identical: false\n\n"));
  }
}

/**
 * @brief Get io sink type from the string keyword argument
 *
 * @param name io sink type keyword name
 * @return corresponding io sink type type
 */
[[nodiscard]] std::optional<cudf::io::io_type> get_io_sink_type(std::string name)
{
  using io_type = cudf::io::io_type;

  static const std::unordered_map<std::string_view, io_type> map = {
    {"FILEPATH", io_type::FILEPATH},
    {"HOST_BUFFER", io_type::HOST_BUFFER},
    {"PINNED_BUFFER", io_type::HOST_BUFFER},
    {"DEVICE_BUFFER", io_type::DEVICE_BUFFER}};

  std::transform(name.begin(), name.end(), name.begin(), ::toupper);
  if (map.find(name) != map.end()) {
    return {map.at(name)};
  } else {
    fmt::print(
      "{} is not a valid io sink type. Available: FILEPATH,\n"
      "HOST_BUFFER, PINNED_BUFFER, DEVICE_BUFFER. Ignoring\n\n",
      name);
    return std::nullopt;
  }
}

/**
 * @brief Concatenate a vector of tables and return the resultant table
 *
 * @param tables Vector of tables to concatenate
 * @param stream CUDA stream to use
 *
 * @return Unique pointer to the resultant concatenated table.
 */
std::unique_ptr<cudf::table> concatenate_tables(std::vector<std::unique_ptr<cudf::table>> tables,
                                                rmm::cuda_stream_view stream)
{
  if (tables.size() == 1) { return std::move(tables[0]); }

  std::vector<cudf::table_view> table_views;
  table_views.reserve(tables.size());
  std::transform(
    tables.begin(), tables.end(), std::back_inserter(table_views), [&](auto const& tbl) {
      return tbl->view();
    });
  // Construct the final table
  return cudf::concatenate(table_views, stream);
}

/**
 * @brief Thread unsafe function to create a directory for FILEPATH io sink type and return its path
 *
 * @return File path of the created directory
 */
[[nodiscard]] std::string get_default_output_path()
{
  static std::string output_path = std::filesystem::current_path().string();
  if (output_path == std::filesystem::current_path().string()) {
    // Check if output path is a valid directory
    if (std::filesystem::is_directory({output_path})) {
      // Create a new directory in output path if not empty.
      if (not std::filesystem::is_empty({output_path})) {
        output_path +=
          "/output_" + fmt::format("{:%Y-%m-%d-%H-%M-%S}", std::chrono::system_clock::now());
        std::filesystem::create_directory({output_path});
      }
    }
  }
  return output_path;
}
