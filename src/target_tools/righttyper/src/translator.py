#!/usr/bin/env python3
from __future__ import annotations
import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import libcst as cst

from codeindex import ModuleIndex

# ---------------- Type string normalization ----------------

_RENDER_MODULE = cst.Module(body=())

def _code(expr: cst.BaseExpression) -> str:
    return _RENDER_MODULE.code_for_node(expr).strip()

_NAME_MAP = {
    'typing.Callable': 'callable',
    'typing.Iterator': 'generator',
    'typing.Type': 'type',
    'types.CodeType': 'code',
    'None': 'Nonetype',
}

def normalize_types(type_str: str, *, strip_generics: bool = False) -> list[str]:
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

    # Optionally collapse generics/subscripts to just their base name
    out: list[str] = []
    for part in desugared:
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


# ---------------- Core processing ----------------

def simplify_path(file_str: str, root: Path|None = None) -> str:
    """
    Try to make file path relative to current working directory.
    If not possible, return absolute resolved path.
    """
    p = Path(file_str).resolve()
    if root is None:
        root = Path.cwd()
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def process_annotations(spec: dict, root: Path|None = None, *, strip_generics: bool = False) -> List[dict]:
    out: List[dict] = []

    for file_str, file_info in spec.get("files", {}).items():
        file_path = Path(file_str)
        if not file_path.exists():
            print(f"warning: {file_path} does not exist; skipping", file=sys.stderr)
            continue

        try:
            idx = ModuleIndex.from_source(file_path.read_text(encoding="utf-8"))
        except cst.ParserSyntaxError as e:
            print(f"error: cannot parse {file_path}: {e}", file=sys.stderr)
            continue

        simplified_file = simplify_path(file_str, root)

        def add_item(type_str: str, info: dict) -> None:
            out.append({
                "file": simplified_file,
                **info,
                "type": normalize_types(type_str, strip_generics=strip_generics)
            })

        functions = file_info.get("functions", {})

        # RightTyper only indicates object attributes where they are annotated, but
        # TypeEvalPy sometimes expects them in other code locations where they appear.
        # To work around that, we copy these across methods... this code has some issues:
        # (1) it assumes 'self' is named the same everywhere;
        # (2) it assumes both functions are methods (not functions nested within methods).
        for func_name, func_info in list(functions.items()):
            variables = func_info.get("vars", {})
            if not (attrs := {
                varname: vartype
                for varname, vartype in variables.items()
                if '.' in varname
            }):
                continue

            func_name_base = '.'.join(func_name.split('.')[:-1]) + '.'
            for func_name2 in functions:
                if func_name2 == func_name or not func_name2.startswith(func_name_base):
                    continue

                functions[func_name2]['vars'] = attrs | functions[func_name2].get('vars', {})

        for func_name, func_info in functions.items():
            assert isinstance(func_info, dict)

            if not (func_idx := idx.functions.get(func_name)):
                if func_name != '<lambda>':
                    print(f"Function '{func_name}' not found in index")
                continue

            # Parameters
            for name, type_str in func_info.get("args", {}).items():
                if not (pos := func_idx.params.get(name)):
                    continue

                add_item(type_str, pos.to_item() | {
                    "function": func_name,
                    "parameter": name,
                })

                # Also emit as variables (they'll be there if assigned to)
                if (pos_list := func_idx.vars.get(name)):
                    for pos in pos_list:
                        add_item(type_str, pos.to_item() | {
                            "function": func_name,
                            "variable": name,
                        })

            # Function return
            if (retval := func_info.get("retval")) is not None:
                add_item(retval, func_idx.pos.to_item() | {
                    "function": func_name,
                })

            # Variables
            for name, type_str in func_info.get("vars", {}).items():
                if func_idx.params.get(name):
                    continue    # already emitted above

                if not (pos_list := func_idx.vars.get(name)):
#                    # emit without position... could be from a compiled string
#                    add_item(type_str, {
#                        "function": func_name,
#                        "variable": name,
#                    })
                    continue

                # Emit it for every position in the code...  not sure why the benchmark
                # asks for multiple locations within the same scope.
                for pos in pos_list:
                    add_item(type_str, pos.to_item() | {
                        "function": func_name,
                        "variable": name,
                    })

        # Module variables
        for name, type_str in file_info.get("vars", {}).items():
            if not (pos_list := idx.module_vars.get(name)):
#                # emit without position... could be from a compiled string
#                add_item(type_str, {
#                    "variable": name,
#                })
                continue

            # Emit it for every position in the code...  not sure why the benchmark
            # asks for multiple locations within the same scope.
            for pos in pos_list:
                add_item(type_str, pos.to_item() | {
                    "variable": name,
                })

    return out


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="Read types from JSON; use AST only for positions. Function col_offset = start of function name.")
    ap.add_argument("input_json", help="Path to JSON file with function/type info")
    ap.add_argument("-o", "--output", help="Write output JSON to this file (default: stdout)")
    args = ap.parse_args()

    spec_path = Path(args.input_json)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    records = process_annotations(spec, strip_generics=True)
    out = json.dumps(records, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
    else:
        print(out)

if __name__ == "__main__":
    main()
