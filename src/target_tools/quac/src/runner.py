"""
Runner for QuAC type inference tool.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from translator import translate_quac_output


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s %(filename)s:%(lineno)d] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def run_quac_on_file(file_path: Path, output_json: Path) -> bool:
    """
    Run QuAC on a single Python file.

    Args:
        file_path: Path to the Python file to analyze
        output_json: Path where QuAC should write its output

    Returns:
        True if QuAC ran successfully, False otherwise
    """
    # QuAC requires files to be importable as modules
    # Create a temporary directory structure for this
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a module directory with a unique name based on the test
        # Use the parent directory name as module name to make it unique
        module_name = f"test_{file_path.parent.name.replace('-', '_')}"
        module_dir = tmpdir_path / module_name
        module_dir.mkdir()

        # Copy the file as __init__.py so it becomes the module
        target_file = module_dir / "__init__.py"
        shutil.copy(file_path, target_file)

        # Also need an __init__.py in parent to make it a package
        (tmpdir_path / "__init__.py").touch()

        # Run QuAC - must run main.py from the quac directory for imports to work
        cmd = [
            "python", "/app/quac/quac/main.py",
            "--module-search-path", str(tmpdir_path),
            "--module-prefix", module_name,
            "--output-file", str(output_json)
        ]

        try:
            logger.info(f"Running QuAC: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout per file
                cwd="/app/quac/quac"  # Run from quac directory for imports
            )

            if result.returncode != 0:
                logger.error(f"QuAC failed on {file_path}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                return False

            logger.info(f"QuAC succeeded on {file_path}")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"QuAC timed out on {file_path}")
            return False
        except Exception as e:
            logger.error(f"Exception running QuAC on {file_path}: {e}")
            return False


def process_file(file_path: Path) -> list:
    """
    Process a single Python file with QuAC.

    Args:
        file_path: Path to the Python file

    Returns:
        List of annotation dictionaries
    """
    logger.info(f"Processing {file_path}")

    # Create temporary file for QuAC output
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        quac_output_path = Path(f.name)

    try:
        # Run QuAC
        success = run_quac_on_file(file_path, quac_output_path)

        if not success or not quac_output_path.exists():
            logger.warning(f"QuAC did not produce output for {file_path}")
            return []

        # Read QuAC output to check if it's empty or has no data
        with open(quac_output_path) as f:
            quac_data = json.load(f)

        if not quac_data:
            logger.warning(f"QuAC produced empty output for {file_path}")
            return []

        # Determine module name (from test directory name)
        module_name = f"test_{file_path.parent.name.replace('-', '_')}"

        # Translate QuAC output to TypeEvalPy format
        results = translate_quac_output(
            quac_json_path=quac_output_path,
            source_file_path=file_path,
            module_name=module_name
        )

        logger.info(f"Extracted {len(results)} annotations from {file_path}")
        return results

    finally:
        # Clean up temporary file
        if quac_output_path.exists():
            quac_output_path.unlink()


def main_runner(args):
    """Main entry point for the QuAC runner."""
    benchmark_path = Path(args.bechmark_path).resolve()

    if not benchmark_path.exists():
        logger.error(f"Benchmark path does not exist: {benchmark_path}")
        sys.exit(1)

    # Find all Python files to process
    python_files = sorted(benchmark_path.rglob("*.py"))

    # Filter out ground truth and result files
    python_files = [
        f for f in python_files
        if not f.name.endswith("_gt.py")
        and not f.name.endswith("_result.py")
        and f.name != "setup.py"
    ]

    logger.info(f"Found {len(python_files)} Python files to process")

    # Process each file
    for i, file in enumerate(python_files, 1):
        logger.info(f"Processing file {i}/{len(python_files)}: {file}")

        try:
            annotations_list = process_file(file)

            # Save results
            json_file_path = str(file).replace(".py", "_result.json")
            with open(json_file_path, "w") as json_file:
                json.dump(annotations_list, json_file, indent=4)

            logger.info(f"Saved results to {json_file_path}")

        except Exception as e:
            logger.exception(f"Error processing {file}: {e}")
            # Continue with next file even if this one failed
            continue

    logger.info("QuAC runner completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QuAC type inference on benchmark")
    parser.add_argument(
        "--bechmark_path",
        type=str,
        default="/tmp/micro-benchmark",
        help="Path to the benchmark directory"
    )
    args = parser.parse_args()

    main_runner(args)
