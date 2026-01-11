#!/usr/bin/env python3
"""
MonkeyType runner for TypeEvalPy.

MonkeyType is a dynamic type inference tool that collects type information
at runtime by tracing code execution. This runner:
1. Parses Python files to discover functions and classes
2. Generates temporary test files that import and exercise the code
3. Runs test files with MonkeyType tracing (entirely in temp directory)
4. Generates stub files from collected traces
5. Translates stub types to TypeEvalPy format

IMPORTANT: All MonkeyType operations happen in a temporary directory to avoid
modifying source directories.
"""

import argparse
import ast
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Import translator
from translator import translate_stub_output

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def generate_test_file(source_file_path: Path, module_name: str, output_dir: Path) -> Path:
    """
    Generate a test file that imports and exercises code from the target file.

    This is necessary because MonkeyType only traces cross-file function calls,
    not same-file calls.

    Args:
        source_file_path: Path to the original Python file
        module_name: Name of the module (without .py extension)
        output_dir: Directory to write the test file

    Returns:
        Path to the generated test file, or None if failed
    """
    try:
        with open(source_file_path) as f:
            source_code = f.read()

        tree = ast.parse(source_code)
    except Exception as e:
        logger.warning(f"Could not parse {source_file_path}: {e}")
        return None

    test_lines = []
    test_lines.append(f"# Auto-generated test file for {source_file_path.name}")
    test_lines.append(f"import {module_name}")
    test_lines.append("")

    # Discover functions and classes
    functions = []
    classes = []

    for node in ast.walk(tree):
        # Module-level functions
        if isinstance(node, ast.FunctionDef):
            # Check if it's at module level (not inside a class)
            parent_is_module = True
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for child in ast.walk(parent):
                        if child is node:
                            parent_is_module = False
                            break

            if parent_is_module and not node.name.startswith('_'):
                functions.append(node)

        # Classes
        elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
            classes.append(node)

    # Generate test code for functions
    test_lines.append("# Test module-level functions")
    for func in functions:
        param_count = len(func.args.args)

        # Generate dummy arguments
        args = []
        for i, arg in enumerate(func.args.args):
            arg_name = arg.arg
            # Use different dummy values to provide variety
            if 'func' in arg_name.lower() or 'callable' in arg_name.lower():
                args.append("lambda: None")
            elif 'str' in arg_name.lower() or 'name' in arg_name.lower():
                args.append('"test"')
            elif 'int' in arg_name.lower() or 'num' in arg_name.lower():
                args.append("42")
            elif 'list' in arg_name.lower():
                args.append("[1, 2, 3]")
            elif 'dict' in arg_name.lower():
                args.append('{"key": "value"}')
            else:
                # Default: try multiple types
                args.append("None")

        args_str = ", ".join(args)
        test_lines.append(f"try:")
        test_lines.append(f"    {module_name}.{func.name}({args_str})")
        test_lines.append(f"except: pass")

        # Try with alternative arguments if there are parameters
        if param_count > 0:
            # Try with different argument types
            alt_args = []
            for arg in func.args.args:
                alt_args.append('"string"' if len(alt_args) % 2 == 0 else '123')

            alt_args_str = ", ".join(alt_args)
            test_lines.append(f"try:")
            test_lines.append(f"    {module_name}.{func.name}({alt_args_str})")
            test_lines.append(f"except: pass")

    test_lines.append("")

    # Generate test code for classes
    test_lines.append("# Test classes")
    for cls in classes:
        test_lines.append(f"try:")
        test_lines.append(f"    obj_{cls.name} = {module_name}.{cls.name}()")

        # Find methods in the class
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                # Skip __init__ as it's called during instantiation
                if node.name == '__init__':
                    continue

                param_count = len(node.args.args) - 1  # Subtract 'self'

                # Generate dummy arguments (excluding self)
                args = []
                for i, arg in enumerate(node.args.args[1:]):  # Skip 'self'
                    arg_name = arg.arg
                    if 'func' in arg_name.lower() or 'callable' in arg_name.lower():
                        args.append("lambda: None")
                    elif 'str' in arg_name.lower():
                        args.append('"test"')
                    elif 'int' in arg_name.lower():
                        args.append("42")
                    else:
                        args.append("None")

                args_str = ", ".join(args)
                test_lines.append(f"    try:")
                test_lines.append(f"        obj_{cls.name}.{node.name}({args_str})")
                test_lines.append(f"    except: pass")

        test_lines.append(f"except: pass")

    test_lines.append("")

    # Write test file
    test_file_path = output_dir / f"test_{module_name}.py"
    with open(test_file_path, 'w') as f:
        f.write('\n'.join(test_lines))

    logger.info(f"Generated test file: {test_file_path.name}")
    return test_file_path


def run_monkeytype_on_file(file_path: Path, output_dir: Path):
    """
    Run MonkeyType on a Python file to collect type information.

    All operations happen in output_dir (temporary directory) to avoid
    modifying the source directory.

    Args:
        file_path: Path to the original Python file to analyze
        output_dir: Temporary directory for all operations

    Returns:
        stub_content: String content of generated stub file, or None if failed
    """
    file_path = Path(file_path).resolve()
    output_dir = Path(output_dir).resolve()

    module_name = file_path.stem  # e.g., "main" from "main.py"

    # Copy source file to temp directory
    temp_source_file = output_dir / file_path.name
    shutil.copy(file_path, temp_source_file)
    logger.info(f"Copied {file_path.name} to temp directory")

    # Also copy any sibling files that might be imported
    # (e.g., to_import_*.py files in the same directory)
    for sibling in file_path.parent.glob("*.py"):
        if sibling != file_path:
            sibling_temp = output_dir / sibling.name
            try:
                shutil.copy(sibling, sibling_temp)
                logger.debug(f"Copied sibling file: {sibling.name}")
            except Exception as e:
                logger.debug(f"Could not copy sibling {sibling.name}: {e}")

    # MonkeyType stores traces in a SQLite database
    db_path = output_dir / "monkeytype.sqlite3"

    # Set environment variable for MonkeyType database location
    env = {
        'MONKEYTYPE_TRACE_STORE': f'sqlite:///{db_path}',
        'PYTHONPATH': str(output_dir),
    }

    # Step 1: Generate test file that exercises the code
    test_file = generate_test_file(file_path, module_name, output_dir)

    if not test_file:
        logger.warning(f"Could not generate test file for {file_path}")
        return None

    # Step 2: Run the test file with MonkeyType tracing
    logger.info(f"Running MonkeyType trace on test file")

    try:
        # Use monkeytype run to execute and trace the test file
        run_cmd = [
            "monkeytype", "run", str(test_file)
        ]

        run_result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(output_dir),  # Run from temp directory!
            env={**os.environ, **env}
        )

        # If execution fails, log but continue (might still have some traces)
        if run_result.returncode != 0:
            logger.warning(f"Test execution failed: {run_result.stderr}")
            # Some tests might fail but we can still try to get stubs

    except subprocess.TimeoutExpired:
        logger.warning(f"Test execution timeout for {file_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to run MonkeyType: {e}")
        return None

    # Step 3: Generate stub file from traces for the ORIGINAL module
    try:
        stub_cmd = [
            "monkeytype", "stub", module_name
        ]

        stub_result = subprocess.run(
            stub_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(output_dir),  # Run from temp directory!
            env={**os.environ, **env}
        )

        if stub_result.returncode != 0:
            logger.warning(f"Stub generation failed: {stub_result.stderr}")
            return None

        stub_content = stub_result.stdout

        if not stub_content or stub_content.strip() == "":
            logger.warning(f"No stub content generated for {file_path}")
            return None

        logger.info(f"Generated stub file ({len(stub_content)} bytes)")
        return stub_content

    except subprocess.TimeoutExpired:
        logger.warning(f"Stub generation timeout for {file_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to generate stub: {e}")
        return None


def process_file(file_path: Path) -> tuple:
    """
    Process a single Python file with MonkeyType.

    Args:
        file_path: Path to the Python file

    Returns:
        tuple: (results list, stub_content string or None)
    """
    file_path = Path(file_path).resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Run MonkeyType to generate stub
        stub_content = run_monkeytype_on_file(file_path, tmpdir_path)

        if not stub_content:
            logger.warning(f"No stub content for {file_path}")
            return [], None

        # Translate stub to TypeEvalPy format
        try:
            results = translate_stub_output(stub_content, file_path)
            logger.info(f"Extracted {len(results)} annotations from {file_path.name}")
            return results, stub_content
        except Exception as e:
            logger.error(f"Failed to translate stub for {file_path}: {e}")
            return [], stub_content


def main_runner(args):
    """Main runner function."""
    benchmark_path = Path(args.bechmark_path)

    if not benchmark_path.exists():
        logger.error(f"Benchmark path does not exist: {benchmark_path}")
        return

    # Find all Python files
    python_files = sorted(benchmark_path.rglob("*.py"))
    logger.info(f"Found {len(python_files)} Python files")

    for i, file in enumerate(python_files, 1):
        logger.info(f"\n[{i}/{len(python_files)}] Processing: {file}")

        # Process the file
        annotations_list, stub_content = process_file(file)

        # Save result JSON
        json_file_path = str(file).replace(".py", "_result.json")
        with open(json_file_path, "w") as json_file:
            json.dump(annotations_list, json_file, indent=4)
        logger.info(f"Saved results: {json_file_path}")

        # Save MonkeyType-generated stub file alongside results
        if stub_content:
            stub_file_path = str(file).replace(".py", "_monkeytype.pyi")
            with open(stub_file_path, "w") as stub_file:
                stub_file.write(stub_content)
            logger.info(f"Saved stub file: {stub_file_path}")

    logger.info(f"\nProcessed {len(python_files)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MonkeyType runner for TypeEvalPy")
    parser.add_argument(
        "--bechmark_path",
        type=str,
        default="/tmp/micro-benchmark",
        help="Path to the benchmark directory"
    )

    args = parser.parse_args()
    main_runner(args)
