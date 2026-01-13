#!/usr/bin/env python3
"""
MonkeyType runner for TypeEvalPy.

MonkeyType is a dynamic type inference tool that collects type information
at runtime by tracing code execution. This runner:
1. Copies Python files to a temp directory
2. Runs them directly with MonkeyType tracing
3. Generates stub files from collected traces
4. Translates stub types to TypeEvalPy format

IMPORTANT: MonkeyType only captures types for code that actually executes.
Files without module-level execution will generate no traces.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
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


def run_monkeytype_on_file(file_path: Path, output_dir: Path):
    """
    Run MonkeyType on a Python file to collect type information.

    Creates a minimal wrapper that imports the module to trigger cross-module
    tracing. MonkeyType only traces cross-module calls, not intra-module calls.

    IMPORTANT: We only import the module - no synthetic calls that would
    pollute the type traces with incorrect types.

    Args:
        file_path: Path to the original Python file to analyze
        output_dir: Temporary directory for all operations

    Returns:
        stub_content: String content of generated stub file, or None if failed
    """
    file_path = Path(file_path).resolve()
    output_dir = Path(output_dir).resolve()

    module_name = file_path.stem  # e.g., "main" from "main.py"

    # Copy entire directory tree to temp directory
    # This ensures subdirectories (e.g., nested/__init__.py) are available for imports
    source_dir = file_path.parent
    for item in source_dir.iterdir():
        dest = output_dir / item.name
        try:
            if item.is_dir():
                shutil.copytree(item, dest)
                logger.debug(f"Copied directory: {item.name}/")
            else:
                shutil.copy(item, dest)
                logger.debug(f"Copied file: {item.name}")
        except Exception as e:
            logger.debug(f"Could not copy {item.name}: {e}")

    logger.info(f"Copied {source_dir.name}/ contents to temp directory")

    # MonkeyType stores traces in a SQLite database
    db_path = output_dir / "monkeytype.sqlite3"

    # Set environment variable for MonkeyType database location
    env = {
        'MONKEYTYPE_TRACE_STORE': f'sqlite:///{db_path}',
        'PYTHONPATH': str(output_dir),
    }

    # Step 1: Create a minimal wrapper that just imports the module
    # This triggers module-level execution while enabling cross-module tracing
    # IMPORTANT: We do NOT add synthetic function calls - only import
    wrapper_file = output_dir / f"_run_{module_name}.py"
    wrapper_content = f"""# Minimal wrapper to enable cross-module tracing
import {module_name}
"""
    with open(wrapper_file, 'w') as f:
        f.write(wrapper_content)

    logger.info(f"Running MonkeyType trace via import wrapper")

    try:
        run_cmd = [
            "monkeytype", "run", str(wrapper_file)
        ]

        run_result = subprocess.run(
            run_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(output_dir),
            env={**os.environ, **env}
        )

        # If execution fails, log but continue (might still have some traces)
        if run_result.returncode != 0:
            logger.warning(f"Execution failed: {run_result.stderr}")
            # Some code might fail but we can still try to get stubs

    except subprocess.TimeoutExpired:
        logger.warning(f"Execution timeout for {file_path}")
        return None
    except Exception as e:
        logger.error(f"Failed to run MonkeyType: {e}")
        return None

    # Step 2: Generate stub file from traces
    try:
        stub_cmd = [
            "monkeytype", "stub", module_name
        ]

        stub_result = subprocess.run(
            stub_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(output_dir),
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
