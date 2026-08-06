/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "column_path_helpers.hpp"

#include <cudf/ast/detail/expression_transformer.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cudf::io::parquet::detail {

/**
 * @brief Classification of an AST expression operand
 */
enum class operand_kind : uint8_t { COLUMN_REF = 0, LITERAL = 1, EXPRESSION = 2 };

/**
 * @brief Extracted unary operand from an AST operation
 */
struct unary_operand {
  operand_kind operand_type;
  ast::column_reference const* col_ref;  ///< Non-null only when the operand is COLUMN_REF
};

/**
 * @brief Extracted binary operator and operands from an AST operation
 *
 * For `lit op col` expressions, the input non-commutative operator is inverted and the
 * operands are normalized to `col op lit` form.
 */
struct binary_operands {
  ast::ast_operator op;  ///< Input or inverted operator to normalize the `lit op col` expressions
  operand_kind lhs_type;
  operand_kind rhs_type;
  ast::column_reference const*
    col_ref;  ///< Reliable only when the expression is of the form `col op lit` or `lit op col`
  ast::literal const*
    literal;  ///< Reliable only when the expression is of the form `col op lit` or `lit op col`
};

/**
 * @brief Extracts the unary operand from a unary operation
 */
[[nodiscard]] unary_operand extract_unary_operand(ast::operation const& expr);

/**
 * @brief Decomposes a binary operation into classified parts.
 *
 * When the expression is of the form `lit op col`, the operator is inverted and the result
 * is normalized so that col_ref and literal are set as if the form were `col op lit`.
 */
[[nodiscard]] binary_operands extract_binary_operands(ast::operation const& expr);

/**
 * @brief Specifies how to transform a comparison operator
 */
enum class operator_transform : uint8_t {
  INVERT,  ///< Swap operand sides: `a < b` becomes `b > a`
  NEGATE   ///< Logical negation: `NOT(a < b)` becomes `a >= b`
};

/**
 * @brief Applies the specified transformation to an operator
 *
 * INVERT swaps operand order (e.g. LESS => GREATER) for normalizing `lit op col` to `col op lit`.
 * NEGATE returns the logical complement (e.g. LESS => GREATER_EQUAL) for handling NOT(col op lit).
 *
 * @tparam mode Transformation mode
 *
 * @param op Operator to transform
 * @return Transformed operator or std::nullopt. For INVERT mode, commutative and
 * untransformable operators are returned as is (no std::nullopt)
 */
template <operator_transform mode>
[[nodiscard]] inline std::optional<ast::ast_operator> transform_operator(ast::ast_operator op)
{
  if constexpr (mode == operator_transform::INVERT) {
    switch (op) {
      case ast::ast_operator::LESS: return ast::ast_operator::GREATER;
      case ast::ast_operator::GREATER: return ast::ast_operator::LESS;
      case ast::ast_operator::LESS_EQUAL: return ast::ast_operator::GREATER_EQUAL;
      case ast::ast_operator::GREATER_EQUAL: return ast::ast_operator::LESS_EQUAL;
      default: return std::make_optional(op);
    }
  } else {
    // mode == NEGATE
    switch (op) {
      case ast::ast_operator::LESS: return ast::ast_operator::GREATER_EQUAL;
      case ast::ast_operator::GREATER: return ast::ast_operator::LESS_EQUAL;
      case ast::ast_operator::LESS_EQUAL: return ast::ast_operator::GREATER;
      case ast::ast_operator::GREATER_EQUAL: return ast::ast_operator::LESS;
      case ast::ast_operator::EQUAL: return ast::ast_operator::NOT_EQUAL;
      case ast::ast_operator::NOT_EQUAL: return ast::ast_operator::EQUAL;
      default: return std::nullopt;
    }
  }
}

/**
 * @brief A pruning expression, or std::nullopt when the subtree it was built from cannot be
 * evaluated against the summary columns and is therefore unconstrained
 */
using maybe_pruning_expr = std::optional<std::reference_wrapper<ast::expression const>>;

/**
 * @brief Base for converters that rewrite a filter into an expression over per-row-group or
 * per-page summary columns
 *
 * The built expression answers "might some row in this row group satisfy the filter?" - `true`
 * keeps the row group, `false` prunes it. A subtree that cannot be answered from the summary
 * columns is *relaxed*: `build()` returns std::nullopt for it and the parent treats it as
 * unconstrained. std::nullopt at the top means no pruning is possible at all, and the caller
 * should skip the filter entirely.
 *
 * Because a leaf is an existential rather than the predicate's truth value, only conjunctions and
 * disjunctions combine meaningfully. Every other combining operator relaxes - negating or
 * comparing existential summaries does not answer the same question the filter asks.
 */
class pruning_expression_builder {
 public:
  pruning_expression_builder()                                             = default;
  virtual ~pruning_expression_builder()                                    = default;
  pruning_expression_builder(pruning_expression_builder const&)            = delete;
  pruning_expression_builder& operator=(pruning_expression_builder const&) = delete;

 protected:
  /**
   * @brief Builds the pruning expression for `expr`, relaxing whatever cannot be evaluated
   *
   * @param expr Expression to rewrite
   * @return The pruning expression, or std::nullopt to relax
   */
  [[nodiscard]] maybe_pruning_expr build(ast::expression const& expr);

  /**
   * @brief Rewrites a `col op lit` comparison into an expression over the summary columns
   *
   * `lit op col` forms are normalized to `col op lit` before this is called.
   *
   * @return The pruning expression, or std::nullopt to relax
   */
  [[nodiscard]] virtual maybe_pruning_expr build_comparison(ast::ast_operator op,
                                                            ast::column_reference const& col_ref,
                                                            ast::literal const& literal) = 0;

  /**
   * @brief Rewrites an `op col` unary operation into an expression over the summary columns
   *
   * @return The pruning expression, or std::nullopt to relax
   */
  [[nodiscard]] virtual maybe_pruning_expr build_unary(ast::ast_operator,
                                                       ast::column_reference const&)
  {
    return std::nullopt;
  }

  /**
   * @brief Rewrites `NOT(op col)` into an expression over the summary columns
   *
   * Defaults to relaxing. Override only where the summary column for `op` is *exact* rather than
   * existential, because negating an existential answers a different question than the filter asks.
   *
   * @return The pruning expression, or std::nullopt to relax
   */
  [[nodiscard]] virtual maybe_pruning_expr build_negated_unary(ast::ast_operator,
                                                               ast::column_reference const&)
  {
    return std::nullopt;
  }

  /**
   * @brief Outcome of distributing a negation into its operand
   *
   * `handled` distinguishes "this builder took responsibility for the operand" from "this is not a
   * shape the negation distributes into". It matters because `build_comparison` and
   * `build_unary` have side effects in collecting subclasses: when a negation is handled the
   * operand must *not* be walked again, or the collector records it twice.
   */
  struct negation_result {
    bool handled;             ///< Whether the negation was distributed into the operand
    maybe_pruning_expr expr;  ///< The rewritten expression, or std::nullopt if it relaxed
  };

  /**
   * @brief Rewrites `NOT(operand)` by distributing the negation into `operand`
   *
   * @return Whether the negation was handled, and the resulting expression if so
   */
  [[nodiscard]] negation_result build_negation(ast::expression const& operand);

  /**
   * @brief Checks that a column reference is usable by this converter
   */
  virtual void validate_column_reference(ast::column_reference const& col_ref) const = 0;

  ast::tree _tree;  ///< Owns the nodes of the built expression
};

/**
 * @brief Collects column names from the expression ignoring the `skip_names`
 */
class names_from_expression : public ast::detail::expression_transformer {
 public:
  names_from_expression() = default;

  names_from_expression(std::optional<std::reference_wrapper<ast::expression const>> expr,
                        std::vector<std::string> const& skip_names,
                        cudf::io::parquet_reader_options const& options,
                        std::vector<SchemaElement> const& schema_tree);

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Returns the column names in AST.
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::vector<std::string> to_vector() &&;

 private:
  void visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands);

  std::unordered_map<cudf::size_type, std::string> _column_indices_to_names;
  std::unordered_set<std::string> _column_names;
  column_path_set _skip_names;
};

/**
 * @brief Converts named columns to index reference columns and pushes logical negations down to
 * the leaves of the expression
 *
 * The converted expression is the single expression the reader uses both to prune row groups and
 * pages, and to filter the decoded rows. Every negation rewrite must therefore be an exact
 * equivalence rather than a relaxation - see `push_down_negation()`.
 */
class named_to_reference_converter : public ast::detail::expression_transformer {
 public:
  named_to_reference_converter() = default;

  named_to_reference_converter(std::optional<std::reference_wrapper<ast::expression const>> expr,
                               table_metadata const& metadata,
                               bool case_sensitive_names);

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::literal const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::literal const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::operation const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::operation const& expr) override;

  /**
   * @brief Returns the converted AST expression
   *
   * @return AST operation expression
   */
  [[nodiscard]] std::optional<std::reference_wrapper<ast::expression const>> get_converted_expr()
    const
  {
    return _converted_expr;
  }

 protected:
  std::vector<std::reference_wrapper<ast::expression const>> visit_operands(
    cudf::host_span<std::reference_wrapper<ast::expression const> const> operands);

  /**
   * @brief Rewrites `NOT(operand)` into an equivalent expression with the negation pushed into
   * `operand`'s own operands
   *
   * Only rewrites that are exact in every case cudf's AST evaluates are applied, as the converted
   * expression also filters the decoded rows:
   *
   * - `NOT(NOT(x))` becomes `x`
   * - `NOT(a AND b)` becomes `NOT(a) OR NOT(b)` and the three other De Morgan forms, for both the
   *   null-propagating (`LOGICAL_*`) and the Kleene (`NULL_LOGICAL_*`) operators
   * - `NOT(a == b)` becomes `a != b` and vice versa
   *
   * Ordering comparisons are deliberately **not** complemented. IEEE-754 makes every ordered
   * comparison against a `NaN` false, so `NOT(a < b)` is true exactly where `a >= b` is false.
   * `NOT(IS_NULL(x))` and `NOT(NULL_EQUAL(a, b))` have no complement operator and are left alone.
   *
   * @param operand The operand of the `NOT` operation to rewrite
   * @return The rewritten expression, or std::nullopt if no exact rewrite exists
   */
  [[nodiscard]] std::optional<std::reference_wrapper<ast::expression const>> push_down_negation(
    ast::expression const& operand);

  /**
   * @brief Returns the converted negation of `operand`, pushing the negation down if possible and
   * otherwise wrapping the converted operand in a `NOT`
   */
  [[nodiscard]] std::reference_wrapper<ast::expression const> negate(
    ast::expression const& operand);

  column_path_map<size_type> _column_name_to_index;
  std::optional<std::reference_wrapper<ast::expression const>> _converted_expr;
  // Using std::list or std::deque to avoid reference invalidation
  std::list<ast::column_reference> _col_ref;
  std::list<ast::operation> _operators;
};

/**
 * @brief Collects lists of equality predicate literals in the AST expression, one list per input
 * table column. This is used in row group filtering based on bloom filters.
 */
class equality_literals_collector : public pruning_expression_builder {
 public:
  equality_literals_collector() = default;

  equality_literals_collector(ast::expression const& expr,
                              cudf::host_span<cudf::data_type const> output_dtypes,
                              cudf::host_span<cudf::size_type const> output_column_schemas = {},
                              cudf::host_span<SchemaElement const> schema_tree             = {});

  /**
   * @brief Vectors of equality literals in the AST expression, one per input table column
   *
   * @return Vectors of equality literals, one per input table column
   */
  [[nodiscard]] std::vector<std::vector<ast::literal*>> get_literals() &&;

 protected:
  /**
   * @copydoc pruning_expression_builder::build_comparison
   *
   * Always relaxes - this only records the literals the bloom filter converter will probe.
   */
  [[nodiscard]] maybe_pruning_expr build_comparison(ast::ast_operator op,
                                                    ast::column_reference const& col_ref,
                                                    ast::literal const& literal) override;

  /**
   * @copydoc pruning_expression_builder::validate_column_reference
   */
  void validate_column_reference(ast::column_reference const& col_ref) const override;

  /**
   * @brief Whether a literal for this column would be unprobeable due to a timestamp scale mismatch
   */
  [[nodiscard]] bool has_timestamp_scale_mismatch(cudf::size_type col_idx) const;

  cudf::host_span<cudf::data_type const> _output_dtypes;
  std::vector<std::vector<ast::literal*>> _literals;

 private:
  cudf::host_span<cudf::size_type const> _output_column_schemas;
  cudf::host_span<SchemaElement const> _schema_tree;
};

/**
 * @brief Offsets every column referencein an expression by the specified value
 *
 */
class offset_column_references : public named_to_reference_converter {
 public:
  offset_column_references(std::optional<std::reference_wrapper<ast::expression const>> expr,
                           size_type offset);

  // Use `visit` overloads from named_to_reference_converter
  using named_to_reference_converter::visit;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(ast::column_reference const& expr) override;

  /**
   * @copydoc ast::detail::expression_transformer::visit(ast::column_name_reference const& )
   */
  std::reference_wrapper<ast::expression const> visit(
    ast::column_name_reference const& expr) override;

 private:
  size_type _offset{0};
};

/**
 * @brief Maps indices of (all or selected) columns to their names
 *
 * @param options Parquet reader options
 * @param schema_tree Parquet schema tree
 *
 * @return Map of column indices to their names
 */
[[nodiscard]] std::unordered_map<cudf::size_type, std::string> map_column_indices_to_names(
  cudf::io::parquet_reader_options const& options,
  std::span<SchemaElement const> schema_tree,
  bool case_sensitive_names);

/**
 * @brief Get the column names in expression object
 *
 * @param expr The optional expression object to get the column names from
 * @param skip_names The names of column names to skip in returned column names
 * @param options Reader options
 * @param schema_tree The schema tree describing the file structure
 * @return The column names present in expression object except the skip_names
 */
[[nodiscard]] std::vector<std::string> get_column_names_in_expression(
  std::optional<std::reference_wrapper<ast::expression const>> expr,
  std::vector<std::string> const& skip_names,
  cudf::io::parquet_reader_options const& options,
  std::vector<SchemaElement> const& schema_tree);

/**
 * @brief Filter table using the provided (StatsAST or BloomfilterAST) expression and
 * collect filtered row group indices
 *
 * @param ast_table Table of stats or bloom filter membership columns
 * @param ast_expr StatsAST or BloomfilterAST expression to filter with
 * @param input_row_group_indices Lists of input row groups to read, one per source
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Collected filtered row group indices, one vector per source, if any. A std::nullopt if
 * all row groups are required or if the computed predicate is all nulls
 */
[[nodiscard]] std::optional<std::vector<std::vector<size_type>>> collect_filtered_row_group_indices(
  cudf::table_view ast_table,
  std::reference_wrapper<ast::expression const> ast_expr,
  host_span<std::vector<size_type> const> input_row_group_indices,
  rmm::cuda_stream_view stream);

}  // namespace cudf::io::parquet::detail
