import argparse
import ast
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from sys import stdout

import libcst as cst
from libcst.metadata import MetadataWrapper, QualifiedNameProvider

import utils
from type_normalizer import normalize_types

# Create a logger
logger = logging.getLogger("runner")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("/tmp/pytype_log.log")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def list_python_files(folder_path):
    python_files = sorted(Path(folder_path).rglob("*.py"))
    return python_files


def run_pytype(file_path, output_dir):
    """Run pytype on a file and return path to generated .pyi file

    Note: pytype generates .pyi stub files which contain:
    - Module-level variable types
    - Function/method return types
    - Class attributes
    But typically NOT parameter types (unless explicitly annotated in source)
    """
    try:
        # Run pytype with output directory
        # Note: Could add --precise-return flag for more accurate return types
        cmd = ["pytype", str(file_path), "-o", output_dir]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            logger.warning(f"pytype had issues with {file_path}: {result.stderr}")

        # Find the generated .pyi file
        file_name = Path(file_path).stem
        pyi_path = Path(output_dir) / "pyi" / f"{file_name}.pyi"

        if pyi_path.exists():
            return pyi_path
        else:
            logger.error(f"No .pyi file generated for {file_path}")
            return None

    except subprocess.TimeoutExpired:
        logger.error(f"pytype timed out on {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error running pytype on {file_path}: {e}")
        return None


class TypeQualifier(cst.CSTVisitor):
    """Visitor that qualifies type names using import information"""

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
        # Create a minimal module wrapper to get the code
        try:
            return cst.Module([]).code_for_node(node)
        except:
            # Fallback: just return empty string
            return ""

    def _qualify_annotation(self, annotation_node):
        """Convert an annotation node to a fully qualified string

        Fully qualifies imported names but keeps builtins unqualified.
        For example: 'count' from 'itertools' -> 'itertools.count'
        But: 'int', 'str', 'list' stay as-is (not 'builtins.int')

        Handles complex annotations like List[int], Union[str, count], etc.
        """
        if annotation_node is None:
            return None

        # Handle simple Name nodes (e.g., 'int', 'count')
        if isinstance(annotation_node, cst.Name):
            qualified = self._qualify_name(annotation_node)
            return qualified if qualified else annotation_node.value

        # Handle Subscript nodes (e.g., List[int], Dict[str, int])
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

        # For other complex types, generate the code and try to qualify names within
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


def parse_pyi_file(pyi_path):
    """Parse a .pyi file and extract type information with fully qualified names"""
    with open(pyi_path) as f:
        pyi_content = f.read()

    try:
        # Parse with libcst
        module = cst.parse_module(pyi_content)

        # Wrap with metadata to resolve qualified names
        wrapper = MetadataWrapper(module)

        # Visit the tree to extract types
        visitor = TypeQualifier()
        wrapper.visit(visitor)

        return visitor.types_info

    except Exception as e:
        logger.error(f"Failed to parse {pyi_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'module_vars': {},
            'classes': {},
            'functions': {}
        }


def match_types_to_source(types_info, source_code, file_name):
    """Match type information from .pyi to source code positions"""
    try:
        from codeindex import ModuleIndex
        idx = ModuleIndex.from_source(source_code)
    except Exception as e:
        logger.error(f"Could not parse source for positions: {e}")
        return []

    results = []

    # Match module-level variables
    for var_name, var_type in types_info['module_vars'].items():
        if var_name in idx.module_vars:
            # Take the first occurrence
            positions = idx.module_vars[var_name]
            if positions:
                pos = positions[0]
                # Skip special variables like __getattr__
                if not var_name.startswith('_'):
                    results.append({
                        "file": file_name,
                        "line_number": pos.line_number,
                        "col_offset": pos.col_offset,
                        "variable": var_name,
                        "type": normalize_types(var_type),
                        "full_type": var_type
                    })

    # Match functions
    for func_name, func_info in types_info['functions'].items():
        if func_name in idx.functions:
            func_pos = idx.functions[func_name]

            # Function return type
            if func_info['return'] and func_info['return'] not in ('None', 'NoneType'):
                results.append({
                    "file": file_name,
                    "line_number": func_pos.pos.line_number,
                    "col_offset": func_pos.pos.col_offset,
                    "function": func_name,
                    "type": normalize_types(func_info['return']),
                    "full_type": func_info['return']
                })

            # Function parameters
            for param_name, param_type in func_info['params'].items():
                if param_name in func_pos.params:
                    param_pos = func_pos.params[param_name]
                    results.append({
                        "file": file_name,
                        "line_number": param_pos.line_number,
                        "col_offset": param_pos.col_offset,
                        "parameter": param_name,
                        "function": func_name,
                        "type": normalize_types(param_type),
                        "full_type": param_type
                    })

    # Match classes
    for class_name, class_info in types_info['classes'].items():
        # Match methods
        for method_name, method_info in class_info['methods'].items():
            full_method_name = f"{class_name}.{method_name}"

            # Try to find the method in the index (ModuleIndex stores with qualified names)
            func_pos = idx.functions.get(full_method_name)
            if not func_pos:
                # Fallback: try without class name
                func_pos = idx.functions.get(method_name)

            if func_pos:
                # Method return type
                # Skip None returns (constructors) but keep Any (unknown return types)
                if method_info['return'] and method_info['return'] not in ('None', 'NoneType'):
                    results.append({
                        "file": file_name,
                        "line_number": func_pos.pos.line_number,
                        "col_offset": func_pos.pos.col_offset,
                        "function": full_method_name,
                        "type": normalize_types(method_info['return']),
                        "full_type": method_info['return']
                    })

                # Method parameters
                for param_name, param_type in method_info['params'].items():
                    if param_name in func_pos.params:
                        param_pos = func_pos.params[param_name]
                        # Skip 'self' parameters without type annotations
                        if param_name != 'self' or param_type not in ('Any',):
                            results.append({
                                "file": file_name,
                                "line_number": param_pos.line_number,
                                "col_offset": param_pos.col_offset,
                                "parameter": param_name,
                                "function": full_method_name,
                                "type": normalize_types(param_type),
                                "full_type": param_type
                            })

        # Match class attributes (instance variables like self.width)
        # These need to be matched differently since they're assignments
        for attr_name, attr_type in class_info['attributes'].items():
            # Look for self.attr_name in the source
            # This is tricky - we need to parse the source and find attribute assignments
            source_tree = ast.parse(source_code)
            for node in ast.walk(source_tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            for stmt in ast.walk(method):
                                if isinstance(stmt, ast.Assign):
                                    for target in stmt.targets:
                                        if isinstance(target, ast.Attribute):
                                            if (isinstance(target.value, ast.Name) and
                                                target.value.id == 'self' and
                                                target.attr == attr_name):
                                                # Found the assignment
                                                full_var_name = f"self.{attr_name}"
                                                results.append({
                                                    "file": file_name,
                                                    "line_number": target.lineno,
                                                    "col_offset": target.col_offset + 1,
                                                    "variable": full_var_name,
                                                    "function": f"{class_name}.{method.name}",
                                                    "type": normalize_types(attr_type),
                                                    "full_type": attr_type
                                                })

    return results


def process_file(file_path, output_dir):
    """Process a single Python file with pytype

    Returns:
        tuple: (results list, pyi_content string or None)
    """
    file_name = Path(file_path).name

    # Read source code
    with open(file_path) as f:
        source_code = f.read()

    # Run pytype to generate .pyi
    pyi_path = run_pytype(file_path, output_dir)
    if not pyi_path:
        logger.error(f"Failed to generate .pyi for {file_path}")
        return [], None

    # Read .pyi content to save later
    pyi_content = None
    try:
        with open(pyi_path) as f:
            pyi_content = f.read()
    except Exception as e:
        logger.warning(f"Could not read .pyi file: {e}")

    # Parse .pyi file
    types_info = parse_pyi_file(pyi_path)

    # Match types to source positions
    results = match_types_to_source(types_info, source_code, file_name)

    return results, pyi_content


def main_runner(args):
    python_files = list_python_files(args.bechmark_path)
    logger.info(f"Found {len(python_files)} python files")
    error_count = 0

    for i, file in enumerate(python_files):
        try:
            logger.info(f"Processing file {i+1}/{len(python_files)}: {file}")

            # Create a temporary directory for pytype output
            with tempfile.TemporaryDirectory() as tmpdir:
                annotations_list, pyi_content = process_file(file, tmpdir)

            # Save result JSON
            json_file_path = str(file).replace(".py", "_result.json")
            with open(json_file_path, "w") as json_file:
                json.dump(annotations_list, json_file, indent=4)

            # Save pytype-generated .pyi file alongside results
            if pyi_content:
                pyi_file_path = str(file).replace(".py", "_pytype.pyi")
                with open(pyi_file_path, "w") as pyi_file:
                    pyi_file.write(pyi_content)
                logger.info(f"Saved pytype stub file: {pyi_file_path}")

            logger.info(f"Successfully processed {file.name}, found {len(annotations_list)} annotations")

        except Exception as e:
            logger.error(f"Error processing file {file}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            error_count += 1

    logger.info(f"Runner finished with {error_count} errors")


if __name__ == "__main__":
    is_running_in_docker = utils.is_running_in_docker()
    if is_running_in_docker:
        print("Python is running inside a Docker container")
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--bechmark_path",
            help="Specify the benchmark path",
            default="/tmp/micro-benchmark",
        )

        args = parser.parse_args()
        main_runner(args)
    else:
        print("Python is not running inside a Docker container")
