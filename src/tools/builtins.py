"""Deterministic built-in example tools."""

from __future__ import annotations

import ast
import operator

from .models import ToolDefinition


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    raise ValueError("calculator accepts arithmetic expressions only")


def calculator(expression: str) -> str:
    """Evaluate a numeric arithmetic expression without arbitrary code execution."""

    tree = ast.parse(expression, mode="eval")
    value = _evaluate(tree.body)
    return str(int(value) if isinstance(value, float) and value.is_integer() else value)


def calculator_definition() -> ToolDefinition:
    return ToolDefinition(
        name="calculator",
        description="Calculate a numeric arithmetic expression.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
        handler=calculator,
    )
