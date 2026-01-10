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
from type_normalizer import normalize_types

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


def process_annotations(spec: dict, root: Path|None = None) -> List[dict]:
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
                "type": normalize_types(type_str),
                "full_type": type_str
            })

        functions = file_info.get("functions", {})

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

    records = process_annotations(spec)
    out = json.dumps(records, indent=2, ensure_ascii=False)

    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
    else:
        print(out)

if __name__ == "__main__":
    main()
