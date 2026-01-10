import argparse
import json
import os
from pathlib import Path

from codeindex import ModuleIndex
from type_normalizer import normalize_types


def parse_type_prediction(pred: list[list], id_type=None) -> tuple[list[str], str]:
    """Parse type prediction and return (normalized_types, full_type)"""
    if pred:
        full_type = pred[0][0]
        normalized = normalize_types(full_type)
        return normalized, full_type
    else:
        return ["Unknown"], "Unknown"


def translate_content(data, source_code):
    if not data:
        return []

    # Parse the source code to extract positions using libcst
    try:
        idx = ModuleIndex.from_source(source_code)
    except Exception as e:
        # If parsing fails, abort - don't generate incorrect column offsets
        print(f"ERROR: Could not parse the given source file! Check out its syntax.")
        print(f"Error: {e}")
        raise

    functions = data["response"]["funcs"]
    variables = data["response"]["variables_p"]
    mod_var_ln = data["response"]["mod_var_ln"]
    output = []

    for func in functions:
        name = func["name"]
        fn_lc = func["fn_lc"]
        line_number = fn_lc[0][0]

        # Function entry - get position from index
        func_info = idx.functions.get(name)
        if func_info:
            func_col_offset = func_info.pos.col_offset
        else:
            # Fallback if not found
            func_col_offset = fn_lc[0][1] + 1

        normalized_type, full_type = parse_type_prediction(func.get("ret_type_p"))
        output.append(
            {
                "file": "main.py",
                "line_number": line_number,
                "col_offset": func_col_offset,
                "function": name,
                "type": normalized_type,
                "full_type": full_type,
                "all_type_preds": func.get("ret_type_p"),
            }
        )

        # Function parameters
        params_p = func["params_p"]
        for param, param_type in params_p.items():
            # Skip parameters that don't exist in the source code (like Type4Py's added 'args' and 'kwargs')
            if func_info and param in func_info.params:
                param_col_offset = func_info.params[param].col_offset
                normalized_type, full_type = parse_type_prediction(param_type)
                output.append(
                    {
                        "file": "main.py",
                        "line_number": line_number,
                        "col_offset": param_col_offset,
                        "parameter": param,
                        "function": name,
                        "type": normalized_type,
                        "full_type": full_type,
                        "all_type_preds": param_type,
                    }
                )

    for var, var_type in variables.items():
        var_ln = mod_var_ln.get(var)
        if var_ln:
            line_number = var_ln[0][0]
            # Get variable col_offset from index
            if var in idx.module_vars:
                var_positions = idx.module_vars[var]
                # Find the position that matches this line number
                var_pos = next(
                    (pos for pos in var_positions if pos.line_number == line_number),
                    None
                )
                if var_pos:
                    var_col_offset = var_pos.col_offset
                else:
                    var_col_offset = var_ln[0][1] + 1
            else:
                var_col_offset = var_ln[0][1] + 1

            normalized_type, full_type = parse_type_prediction(var_type)
            output.append(
                {
                    "file": "main.py",
                    "line_number": line_number,
                    "col_offset": var_col_offset,
                    "variable": var,
                    "type": normalized_type,
                    "full_type": full_type,
                    "all_type_preds": var_type,
                }
            )

    inferred_serializable = [
        {k: list(v) if isinstance(v, set) else v for k, v in d.items()} for d in output
    ]

    return inferred_serializable


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bechmark_path",
        help="Specify the benchmark path",
        default="/tmp/micro-benchmark",
    )

    args = parser.parse_args()
    # main_translator(args)
