#!/usr/bin/env python3
"""
Type string normalization utilities for TypeEvalPy runners.

This module provides functions to parse and normalize Python type strings,
handling unions, optionals, generics, and various typing module variations.
"""
from __future__ import annotations
import libcst as cst

# ---------------- Type string normalization ----------------

_RENDER_MODULE = cst.Module(body=())

def _code(expr: cst.BaseExpression) -> str:
    return _RENDER_MODULE.code_for_node(expr).strip()

_NAME_MAP = {
    # Callable
    'typing.Callable': 'callable',
    'collections.abc.Callable': 'callable',
    # Iterator
    'typing.Iterator': 'iterator',
    'collections.abc.Iterator': 'iterator',
    # Generator
    'typing.Generator': 'generator',
    'collections.abc.Generator': 'generator',
    # Container types
    'typing.List': 'list',
    'typing.Dict': 'dict',
    'typing.Set': 'set',
    'typing.Tuple': 'tuple',
    'typing.FrozenSet': 'frozenset',
    # Other typing
    'typing.Type': 'type',
    'types.CodeType': 'code',
    # None
    'None': 'Nonetype',
}

def strip_builtins_prefix(type_str: str) -> str:
    """
    Strip the 'builtins.' prefix from type strings.

    MonkeyType and other tools output fully qualified names like 'builtins.str',
    but builtins are implicit in Python, so we strip this prefix.

    Args:
        type_str: Type string, possibly with 'builtins.' prefix

    Returns:
        Type string with 'builtins.' prefix removed if present
    """
    if type_str.startswith('builtins.'):
        return type_str[len('builtins.'):]
    return type_str


def normalize_types(type_str: str, *, strip_generics: bool = True) -> list[str]:
    """
    Parse type_str and return a list of top-level alternates.
    If strip_generics=True, collapse generics so that e.g. list[int] -> 'list',
    typing.Callable[[int], None] -> 'typing.Callable'.
    """
    s = type_str.strip()
    if not s:
        return [""]

    try:
        expr = cst.parse_expression(s)
    except Exception:
        return [s]

    def _cst_qualified_name(node: cst.BaseExpression) -> str | None:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            parts = []
            cur: cst.BaseExpression | None = node
            while isinstance(cur, cst.Attribute):
                parts.append(cur.attr.value)
                cur = cur.value
            if isinstance(cur, cst.Name):
                parts.append(cur.value)
                return ".".join(reversed(parts))
        return None

    def _is_typing_name(name: str | None, base: str) -> bool:
        return bool(name) and (name == base or name == f"typing.{base}")

    # Split only at TOP-LEVEL unions (| and typing.Union[...])
    def split_top_level(e: cst.BaseExpression) -> list[cst.BaseExpression]:
        if isinstance(e, cst.BinaryOperation) and isinstance(e.operator, cst.BitOr):
            return split_top_level(e.left) + split_top_level(e.right)
        if isinstance(e, cst.Subscript):
            base_name = _cst_qualified_name(e.value)
            if _is_typing_name(base_name, "Union"):
                alts: list[cst.BaseExpression] = []
                for sl in e.slice:
                    if isinstance(sl, cst.SubscriptElement) and isinstance(sl.slice, cst.Index):
                        alts.extend(split_top_level(sl.slice.value))
                return alts
        return [e]

    # Desugar Optional[T] -> [T, None], Annotated[T, ...] -> [T]
    def desugar(e: cst.BaseExpression) -> list[cst.BaseExpression]:
        if isinstance(e, cst.Subscript):
            base_name = _cst_qualified_name(e.value)
            if _is_typing_name(base_name, "Annotated"):
                if e.slice:
                    first = e.slice[0]
                    if isinstance(first, cst.SubscriptElement) and isinstance(first.slice, cst.Index):
                        return desugar(first.slice.value)
                return [e]
            if _is_typing_name(base_name, "Optional"):
                if e.slice:
                    first = e.slice[0]
                    if isinstance(first, cst.SubscriptElement) and isinstance(first.slice, cst.Index):
                        return desugar(first.slice.value) + [cst.Name("None")]
                return [e]
        return [e]

    top_level_parts = split_top_level(expr)
    desugared: list[cst.BaseExpression] = []
    for p in top_level_parts:
        desugared.extend(desugar(p))

    # After desugaring, split unions again (e.g., Optional[Union[A, B]] -> Union[A, B] + None)
    fully_split: list[cst.BaseExpression] = []
    for d in desugared:
        fully_split.extend(split_top_level(d))

    # Optionally collapse generics/subscripts to just their base name
    out: list[str] = []
    for part in fully_split:
        if strip_generics:
            if isinstance(part, cst.Subscript):
                name = _cst_qualified_name(part.value)

                if name is None:
                    name = _code(part)
            else:
                name = _code(part)

            if name in _NAME_MAP:
                name = _NAME_MAP[name]

            if name.startswith("main."):
                name = name[5:]

            out.append(name)
        else:
            out.append(_code(part))
    return out
