/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "stats_filter_helpers.hpp"

#include "expression_transform_helpers.hpp"

#include <cudf/ast/detail/operators.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

namespace cudf::io::parquet::detail {

stats_columns_collector::stats_columns_collector(ast::expression const& expr,
                                                 cudf::size_type num_columns)
  : _num_columns(num_columns)
{
  _columns_mask.resize(num_columns, false);
  expr.accept(*this);
}

std::reference_wrapper<ast::expression const> stats_columns_collector::visit(
  ast::literal const& expr)
{
  return expr;
}

std::reference_wrapper<ast::expression const> stats_columns_collector::visit(
  ast::column_reference const& expr)
{
  CUDF_EXPECTS(expr.get_table_source() == ast::table_reference::LEFT,
               "Statistics AST supports only left table");
  CUDF_EXPECTS(expr.get_column_index() < _num_columns,
               "Column index cannot be more than number of columns in the table");
  return expr;
}

std::reference_wrapper<ast::expression const> stats_columns_collector::visit(
  ast::column_name_reference const& expr)
{
  CUDF_FAIL("Column name reference is not supported in statistics AST");
}

std::reference_wrapper<ast::expression const> stats_columns_collector::visit(
  ast::operation const& expr)
{
  using cudf::ast::ast_operator;

  auto const input_op       = expr.get_operator();
  auto const operator_arity = cudf::ast::detail::ast_operator_arity(input_op);

  if (operator_arity == 1) {
    auto const [kind, col_ref] = extract_unary_operand(expr);

    if (kind == operand_kind::COLUMN_REF) {
      col_ref->accept(*this);
      if (input_op == ast_operator::IS_NULL) {
        _columns_mask[col_ref->get_column_index()] = true;
        _has_is_null_operator                      = true;
      }
    } else {
      std::ignore = visit_operands(expr.get_operands());
    }
    return expr;
  }

  // Binary operation
  auto const [op, lhs_kind, rhs_kind, col_ref, _] = extract_binary_operands(expr);

  if (lhs_kind == operand_kind::COLUMN_REF and rhs_kind == operand_kind::LITERAL) {
    col_ref->accept(*this);
    if (op == ast_operator::EQUAL or op == ast_operator::NOT_EQUAL or op == ast_operator::LESS or
        op == ast_operator::LESS_EQUAL or op == ast_operator::GREATER or
        op == ast_operator::GREATER_EQUAL) {
      _columns_mask[col_ref->get_column_index()] = true;
    }
  } else {
    // Visit the operands and ignore any output as we only want to build the column mask
    std::ignore = visit_operands(expr.get_operands());
  }
  return expr;
}

std::pair<thrust::host_vector<bool>, bool> stats_columns_collector::get_stats_columns_mask() &&
{
  return {std::move(_columns_mask), _has_is_null_operator};
}

std::vector<std::reference_wrapper<ast::expression const>> stats_columns_collector::visit_operands(
  cudf::host_span<std::reference_wrapper<ast::expression const> const> operands)
{
  std::vector<std::reference_wrapper<ast::expression const>> transformed_operands;
  std::transform(operands.begin(),
                 operands.end(),
                 std::back_inserter(transformed_operands),
                 [t = this](auto& operand) { return operand.get().accept(*t); });

  return transformed_operands;
}

stats_expression_converter::stats_expression_converter(ast::expression const& expr,
                                                       size_type num_columns,
                                                       bool has_is_null_operator)
  : _num_columns{num_columns}, _stats_cols_per_column{has_is_null_operator ? 3 : 2}
{
  _stats_expr = build(expr);
}

void stats_expression_converter::validate_column_reference(
  ast::column_reference const& col_ref) const
{
  CUDF_EXPECTS(col_ref.get_table_source() == ast::table_reference::LEFT,
               "Statistics AST supports only left table");
  CUDF_EXPECTS(col_ref.get_column_index() < _num_columns,
               "Column index cannot be more than number of columns in the table");
}

maybe_pruning_expr stats_expression_converter::build_unary(ast::ast_operator op,
                                                           ast::column_reference const& col_ref)
{
  // The is_null column is tri-state: false when the chunk has no nulls, true when every value is
  // null, and null when it is a mix - so it answers `IS_NULL` directly
  if (op != ast::ast_operator::IS_NULL) { return std::nullopt; }

  CUDF_EXPECTS(std::cmp_equal(_stats_cols_per_column, 3),
               "IS_NULL operator cannot be evaluated without nullability information column");
  return _tree.push(ast::column_reference{vmin_index(col_ref) + 2});
}

maybe_pruning_expr stats_expression_converter::build_negated_unary(
  ast::ast_operator op, ast::column_reference const& col_ref)
{
  // Unlike the min/max columns, the is_null column is *exact* rather than existential - false when
  // the chunk has no nulls, true when every value is null, and null when it is a mix - so negating
  // it still answers the pruning question
  if (op != ast::ast_operator::IS_NULL) { return std::nullopt; }

  auto const is_null = build_unary(op, col_ref);
  if (not is_null.has_value()) { return std::nullopt; }
  return _tree.push(ast::operation{ast::ast_operator::NOT, is_null.value()});
}

maybe_pruning_expr stats_expression_converter::build_comparison(ast::ast_operator op,
                                                                ast::column_reference const& col_ref,
                                                                ast::literal const& literal)
{
  using cudf::ast::ast_operator;

  auto const& lit = _tree.push(literal);

  /* transform to stats conditions
  col == val --> vmin <= val && vmax >= val
  col != val --> !(vmin == val && vmax == val)
  col >  val --> vmax > val
  col <  val --> vmin < val
  col >= val --> vmax >= val
  col <= val --> vmin <= val
  */
  switch (op) {
    case ast_operator::EQUAL: {
      auto const& vmin = _tree.push(ast::column_reference{vmin_index(col_ref)});
      auto const& vmax = _tree.push(ast::column_reference{vmax_index(col_ref)});
      return _tree.push(
        ast::operation{ast_operator::LOGICAL_AND,
                       _tree.push(ast::operation{ast_operator::GREATER_EQUAL, vmax, lit}),
                       _tree.push(ast::operation{ast_operator::LESS_EQUAL, vmin, lit})});
    }
    case ast_operator::NOT_EQUAL: {
      auto const& vmin = _tree.push(ast::column_reference{vmin_index(col_ref)});
      auto const& vmax = _tree.push(ast::column_reference{vmax_index(col_ref)});
      return _tree.push(
        ast::operation{ast_operator::LOGICAL_OR,
                       _tree.push(ast::operation{ast_operator::NOT_EQUAL, vmin, vmax}),
                       _tree.push(ast::operation{ast_operator::NOT_EQUAL, vmax, lit})});
    }
    case ast_operator::LESS: [[fallthrough]];
    case ast_operator::LESS_EQUAL:
      return _tree.push(
        ast::operation{op, _tree.push(ast::column_reference{vmin_index(col_ref)}), lit});
    case ast_operator::GREATER: [[fallthrough]];
    case ast_operator::GREATER_EQUAL:
      return _tree.push(
        ast::operation{op, _tree.push(ast::column_reference{vmax_index(col_ref)}), lit});
    default: return std::nullopt;
  }
}


}  // namespace cudf::io::parquet::detail
