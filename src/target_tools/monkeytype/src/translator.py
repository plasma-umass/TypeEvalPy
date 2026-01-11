#!/usr/bin/env python3
"""
Translator for MonkeyType stub files.

Parses MonkeyType-generated stub files and translates them to TypeEvalPy format.
Uses libcst to fully qualify imported type names (e.g., Callable -> typing.Callable).
"""

import logging
from pathlib import Path
from typing import List, Dict

import libcst as cst
from libcst.metadata import MetadataWrapper, QualifiedNameProvider

from codeindex import ModuleIndex
from type_normalizer import normalize_types

logger = logging.getLogger(__name__)


class TypeQualifier(cst.CSTVisitor):
    """Visitor that extracts types and qualifies names using import information"""

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self):
        self.types_info = {
            'module_vars': {},  # name -> type
            'classes': {},  # class_name -> {methods: {}, attributes: {}}
            'functions': {}  # func_name -> {params: {}, return: type}
        }

    def _qualify_name(self, name_node):
        """Qualify a single name node, stripping builtins prefix"""
        qualified_names = self.get_metadata(QualifiedNameProvider, name_node, set())

        if qualified_names:
            qname = list(qualified_names)[0]
            qualified_name = qname.name

            # Strip 'builtins.' prefix since builtins are implicit in Python
            if qualified_name.startswith('builtins.'):
                return qualified_name[len('builtins.'):]

            return qualified_name

        # No qualified name found, return the raw name
        if isinstance(name_node, cst.Name):
            return name_node.value
        return None

    def _node_to_code(self, node):
        """Convert a CST node to its code representation"""
        try:
            return cst.Module([]).code_for_node(node)
        except:
            return ""

    def _qualify_annotation(self, annotation_node):
        """Convert an annotation node to a fully qualified string

        Fully qualifies imported names but keeps builtins unqualified.
        For example: 'Callable' from 'typing' -> 'typing.Callable'
        But: 'int', 'str', 'list' stay as-is (not 'builtins.int')

        Handles complex annotations like List[int], Union[str, Callable], etc.
        """
        if annotation_node is None:
            return None

        # Handle simple Name nodes (e.g., 'int', 'Callable')
        if isinstance(annotation_node, cst.Name):
            qualified = self._qualify_name(annotation_node)
            return qualified if qualified else annotation_node.value

        # Handle Subscript nodes (e.g., List[int], Dict[str, int], Optional[Union[...]])
        elif isinstance(annotation_node, cst.Subscript):
            # Qualify the base type (e.g., 'List' in List[int])
            if isinstance(annotation_node.value, cst.Name):
                base = self._qualify_name(annotation_node.value)
                if not base:
                    base = annotation_node.value.value
            else:
                base = self._node_to_code(annotation_node.value)

            # Process the subscript slice
            slice_parts = []
            for slice_elem in annotation_node.slice:
                if isinstance(slice_elem, cst.SubscriptElement):
                    slice_value = slice_elem.slice
                    if isinstance(slice_value, cst.Index):
                        inner = self._qualify_annotation(slice_value.value)
                        if inner:
                            slice_parts.append(inner)
                    else:
                        slice_parts.append(self._node_to_code(slice_value))

            if slice_parts:
                return f"{base}[{', '.join(slice_parts)}]"
            else:
                return base

        # Handle Attribute nodes (e.g., 'typing.List')
        elif isinstance(annotation_node, cst.Attribute):
            return self._node_to_code(annotation_node)

        # For other complex types, generate the code
        else:
            return self._node_to_code(annotation_node)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        """Module-level variable annotation"""
        if isinstance(node.target, cst.Name):
            var_name = node.target.value
            var_type = self._qualify_annotation(node.annotation.annotation)
            self.types_info['module_vars'][var_name] = var_type

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Module-level function or method"""
        func_name = node.name.value

        # Extract return type
        return_type = None
        if node.returns:
            return_type = self._qualify_annotation(node.returns.annotation)

        # Extract parameter types
        params = {}
        for param in node.params.params:
            if param.annotation:
                param_type = self._qualify_annotation(param.annotation.annotation)
                params[param.name.value] = param_type

        func_info = {
            'params': params,
            'return': return_type
        }

        self.types_info['functions'][func_name] = func_info

    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        """Class definition"""
        class_name = node.name.value
        class_info = {
            'methods': {},
            'attributes': {}
        }

        # Process class body
        for item in node.body.body:
            if isinstance(item, cst.FunctionDef):
                # Method
                method_name = item.name.value
                return_type = None
                if item.returns:
                    return_type = self._qualify_annotation(item.returns.annotation)

                params = {}
                for param in item.params.params:
                    if param.annotation:
                        param_type = self._qualify_annotation(param.annotation.annotation)
                        params[param.name.value] = param_type

                method_info = {
                    'params': params,
                    'return': return_type
                }
                class_info['methods'][method_name] = method_info

            elif isinstance(item, cst.SimpleStatementLine):
                # Check for annotated assignments (class attributes)
                for stmt in item.body:
                    if isinstance(stmt, cst.AnnAssign):
                        if isinstance(stmt.target, cst.Name):
                            attr_name = stmt.target.value
                            attr_type = self._qualify_annotation(stmt.annotation.annotation)
                            class_info['attributes'][attr_name] = attr_type

        self.types_info['classes'][class_name] = class_info


def parse_stub_file(stub_content: str) -> Dict:
    """
    Parse a MonkeyType stub file with fully qualified type names.

    Args:
        stub_content: String content of stub file

    Returns:
        dict with keys:
        - module_vars: {var_name: qualified_type_str}
        - functions: {func_name: {'return': qualified_type_str, 'params': {param_name: qualified_type_str}}}
        - classes: {class_name: {'methods': {method_name: {...}}, 'attrs': {attr_name: qualified_type_str}}}
    """
    try:
        # Parse with libcst
        module = cst.parse_module(stub_content)

        # Wrap with metadata to resolve qualified names
        wrapper = MetadataWrapper(module)

        # Visit the tree to extract types
        visitor = TypeQualifier()
        wrapper.visit(visitor)

        return visitor.types_info

    except Exception as e:
        logger.error(f"Failed to parse stub: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'module_vars': {},
            'classes': {},
            'functions': {}
        }


def translate_stub_output(stub_content: str, source_file_path: Path) -> List[Dict]:
    """
    Translate MonkeyType stub output to TypeEvalPy format.

    Args:
        stub_content: Content of MonkeyType stub file
        source_file_path: Path to original source file

    Returns:
        List of annotation dictionaries in TypeEvalPy format
    """
    results = []
    file_name = source_file_path.name

    # Parse the stub file (with fully qualified names)
    stub_info = parse_stub_file(stub_content)

    # Read source code and create index
    try:
        with open(source_file_path) as f:
            source_code = f.read()
        idx = ModuleIndex.from_source(source_code)
    except Exception as e:
        logger.error(f"Failed to parse source file {source_file_path}: {e}")
        return results

    # Process module-level variables
    for var_name, var_type in stub_info['module_vars'].items():
        # Skip special variables
        if var_name.startswith('_'):
            continue

        var_info = idx.variables.get(var_name)
        if not var_info:
            logger.debug(f"Variable {var_name} not found in source index")
            continue

        normalized_types = normalize_types(var_type)

        result = {
            "file": file_name,
            "line_number": var_info.line_number,
            "col_offset": var_info.col_offset,
            "variable": var_name,
            "type": normalized_types,
            "full_type": var_type
        }
        results.append(result)

    # Process module-level functions
    for func_name, func_info in stub_info['functions'].items():
        process_function_annotations(
            results, idx, file_name, func_name, func_info, class_name=None
        )

    # Process classes
    for class_name, class_info in stub_info['classes'].items():
        # Process class methods
        for method_name, method_info in class_info['methods'].items():
            process_function_annotations(
                results, idx, file_name, method_name, method_info, class_name=class_name
            )

    return results


def process_function_annotations(
    results: List[Dict],
    idx: ModuleIndex,
    file_name: str,
    func_name: str,
    func_info: Dict,
    class_name: str = None
):
    """
    Process function/method annotations and add to results.

    Args:
        results: List to append results to
        idx: ModuleIndex for source position lookup
        file_name: Name of source file
        func_name: Name of function/method
        func_info: Function info dict with 'return' and 'params'
        class_name: Class name if this is a method, None if module-level function
    """
    # Build qualified name for lookup
    if class_name:
        qualified_name = f"{class_name}.{func_name}"
    else:
        qualified_name = func_name

    # Look up function in index
    func_pos = idx.functions.get(qualified_name)
    if not func_pos:
        logger.debug(f"Function {qualified_name} not found in source index")
        return

    # Process return type
    return_type = func_info.get('return')
    if return_type and return_type not in ('None', 'NoneType'):
        normalized_types = normalize_types(return_type)

        result = {
            "file": file_name,
            "line_number": func_pos.pos.line_number,
            "col_offset": func_pos.pos.col_offset,
            "function": func_name,
            "type": normalized_types,
            "full_type": return_type
        }
        results.append(result)

    # Process parameters
    for param_name, param_type in func_info.get('params', {}).items():
        # Skip 'self' and 'cls'
        if param_name in ('self', 'cls'):
            continue

        # Check if parameter exists in source
        if param_name not in func_pos.params:
            logger.debug(f"Parameter {param_name} not found in function {qualified_name}")
            continue

        param_pos = func_pos.params[param_name]
        normalized_types = normalize_types(param_type)

        result = {
            "file": file_name,
            "line_number": param_pos.line_number,
            "col_offset": param_pos.col_offset,
            "function": func_name,
            "parameter": param_name,
            "type": normalized_types,
            "full_type": param_type
        }
        results.append(result)
