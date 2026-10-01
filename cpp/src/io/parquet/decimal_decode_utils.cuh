/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>

namespace cudf::io::parquet::detail {

/**
 * @brief Decode a decimal stored as FIXED_LEN_BYTE_ARRAY or BYTE_ARRAY
 *
 * The unscaled value is big-endian two's complement. Values shorter than `T` are sign extended
 * (when `T` is signed), and leading (sign extension) bytes of values longer than `T` are dropped.
 *
 * @tparam T Integer storage type of the decimal column
 * @param data Pointer to the first (most significant) byte of the value
 * @param length Number of bytes in the value
 * @param stride Distance in bytes between consecutive bytes of the value (e.g. the number of
 *        values in a BYTE_STREAM_SPLIT encoded page)
 * @return Decoded unscaled value
 */
template <typename T>
[[nodiscard]] CUDF_HOST_DEVICE inline T decode_big_endian_decimal(uint8_t const* data,
                                                                  int32_t length,
                                                                  size_type stride = 1)
{
  if (length <= 0) { return T{0}; }

  // Keep the trailing (least significant) bytes and fill the rest with the sign
  auto const num_bytes   = cuda::std::min(length, static_cast<int32_t>(sizeof(T)));
  auto const first_byte  = data + static_cast<size_t>(length - num_bytes) * stride;
  auto const is_negative = cudf::is_signed<T>() and (*first_byte & 0x80);
  auto big_endian        = is_negative ? static_cast<T>(~T{0}) : T{0};
  auto const bytes       = reinterpret_cast<uint8_t*>(&big_endian) + sizeof(T) - num_bytes;
  if (stride == 1) {
    cuda::std::memcpy(bytes, first_byte, num_bytes);
  } else {
    for (auto i = 0; i < num_bytes; ++i) {
      bytes[i] = first_byte[static_cast<size_t>(i) * stride];
    }
  }
  return cuda::std::byteswap(big_endian);
}

}  // namespace cudf::io::parquet::detail
