"""
Translator for QuAC type inference output.
Converts QuAC's JSON format to TypeEvalPy's expected format.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

from codeindex import ModuleIndex
from type_normalizer import normalize_types


logger = logging.getLogger(__name__)


def strip_builtins_prefix(type_str: str) -> str:
    """
    Strip 'builtins.' prefix from type strings since builtins are implicit in Python.
    For example: 'builtins.str' -> 'str', 'builtins.int' -> 'int'
    """
    if type_str.startswith('builtins.'):
        return type_str[len('builtins.'):]
    return type_str


def strip_synthetic_module_prefix(type_str: str) -> str:
    """
    Strip synthetic 'test_*.' module prefixes that were added for QuAC's import mechanism.
    For example: 'test_object.MyClass' -> 'MyClass'

    These prefixes appear because we create temporary modules like 'test_object' to make
    files importable by QuAC, but they shouldn't appear in the final type annotations.
    """
    # Check if type starts with 'test_*.' pattern
    if type_str.startswith('test_'):
        # Find the first dot
        dot_idx = type_str.find('.')
        if dot_idx > 0:
            # Extract the module part
            module_part = type_str[:dot_idx]
            # Verify it matches our synthetic pattern: test_<something>
            if module_part.startswith('test_') and '_' in module_part:
                # Strip the synthetic module prefix
                return type_str[dot_idx + 1:]

    return type_str


def translate_quac_output(
    quac_json_path: Path,
    source_file_path: Path,
    module_name: str
) -> List[Dict]:
    """
    Translate QuAC output JSON to TypeEvalPy format.

    Args:
        quac_json_path: Path to QuAC's output JSON file
        source_file_path: Path to the source Python file
        module_name: The module name used in QuAC output

    Returns:
        List of annotation dictionaries in TypeEvalPy format
    """
    # Read QuAC output
    with open(quac_json_path) as f:
        quac_data = json.load(f)

    # Read source code for position lookup
    with open(source_file_path) as f:
        source_code = f.read()

    # Parse source with ModuleIndex
    try:
        idx = ModuleIndex.from_source(source_code)
    except Exception as e:
        logger.error(f"Failed to parse source file {source_file_path}: {e}")
        return []

    results = []
    file_name = source_file_path.name

    # Get module data from QuAC output
    if module_name not in quac_data:
        logger.warning(f"Module {module_name} not found in QuAC output")
        return []

    module_data = quac_data[module_name]

    # Process global functions
    if "global" in module_data:
        for func_name, func_annotations in module_data["global"].items():
            process_function_annotations(
                func_name=func_name,
                class_name=None,
                func_annotations=func_annotations,
                idx=idx,
                file_name=file_name,
                results=results
            )

    # Process class methods
    for key, value in module_data.items():
        if key != "global" and isinstance(value, dict):
            class_name = key
            for method_name, method_annotations in value.items():
                process_function_annotations(
                    func_name=method_name,
                    class_name=class_name,
                    func_annotations=method_annotations,
                    idx=idx,
                    file_name=file_name,
                    results=results
                )

    return results


def process_function_annotations(
    func_name: str,
    class_name: Optional[str],
    func_annotations: Dict[str, List[str]],
    idx: ModuleIndex,
    file_name: str,
    results: List[Dict]
):
    """
    Process annotations for a single function/method.

    Args:
        func_name: Function or method name
        class_name: Class name if this is a method, None for global functions
        func_annotations: Dictionary of parameter/return annotations from QuAC
        idx: ModuleIndex for position lookup
        file_name: Source file name
        results: List to append results to
    """
    # Construct qualified function name for lookup
    if class_name:
        qualified_name = f"{class_name}.{func_name}"
    else:
        qualified_name = func_name

    # Look up function position
    func_info = idx.functions.get(qualified_name)
    if not func_info:
        logger.warning(f"Function {qualified_name} not found in source index")
        return

    # Process each parameter and return type
    for param_or_return, type_list in func_annotations.items():
        # Skip empty type lists
        if not type_list:
            continue

        # Skip self and cls parameters for methods
        if class_name and param_or_return in ('self', 'cls'):
            continue

        # Skip return for __init__ and __new__
        if class_name and func_name in ('__init__', '__new__') and param_or_return == 'return':
            continue

        # Handle return type
        if param_or_return == "return":
            # Normalize types
            normalized_types = []
            full_types = []
            for type_str in type_list:
                # Strip synthetic module and builtins prefixes before normalization
                type_str_stripped = strip_builtins_prefix(type_str)
                type_str_stripped = strip_synthetic_module_prefix(type_str_stripped)
                normalized = normalize_types(type_str_stripped)
                normalized_types.extend(normalized)
                full_types.append(type_str_stripped)

            # Remove duplicates while preserving order
            normalized_types = list(dict.fromkeys(normalized_types))
            full_type = " | ".join(full_types) if len(full_types) > 1 else full_types[0]

            result = {
                "file": file_name,
                "line_number": func_info.pos.line_number,
                "col_offset": func_info.pos.col_offset,
                "function": func_name,
                "type": normalized_types,
                "full_type": full_type
            }
            results.append(result)

        # Handle parameter type
        else:
            param_name = param_or_return

            # Look up parameter position
            if param_name not in func_info.params:
                logger.warning(
                    f"Parameter {param_name} not found in function {qualified_name}"
                )
                continue

            param_info = func_info.params[param_name]

            # Normalize types
            normalized_types = []
            full_types = []
            for type_str in type_list:
                # Strip synthetic module and builtins prefixes before normalization
                type_str_stripped = strip_builtins_prefix(type_str)
                type_str_stripped = strip_synthetic_module_prefix(type_str_stripped)
                normalized = normalize_types(type_str_stripped)
                normalized_types.extend(normalized)
                full_types.append(type_str_stripped)

            # Remove duplicates while preserving order
            normalized_types = list(dict.fromkeys(normalized_types))
            full_type = " | ".join(full_types) if len(full_types) > 1 else full_types[0]

            result = {
                "file": file_name,
                "line_number": param_info.line_number,
                "col_offset": param_info.col_offset,
                "function": func_name,
                "parameter": param_name,
                "type": normalized_types,
                "full_type": full_type
            }
            results.append(result)
