#!/usr/bin/env python3
"""
Compare type inference results across multiple runs for the python_features micro-benchmark.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from tabulate import tabulate


def normalize_annotation(annotation: dict) -> tuple:
    """
    Create a normalized key for comparing annotations.
    Ignores fields like 'all_type_preds' that only exist in results.
    """
    key_fields = ['file', 'line_number', 'col_offset']

    # Determine what kind of annotation this is
    if 'parameter' in annotation:
        key_fields.extend(['function', 'parameter'])
        kind = 'parameter'
    elif 'function' in annotation and 'parameter' not in annotation:
        key_fields.append('function')
        kind = 'function'
    elif 'variable' in annotation:
        key_fields.append('variable')
        kind = 'variable'
    else:
        kind = 'unknown'

    # Create tuple of key fields
    key = tuple(annotation.get(field) for field in key_fields)

    # Get the type (normalized to frozenset for comparison)
    pred_type = frozenset(annotation.get('type', []))

    return (key, kind, pred_type)


def compare_files(gt_path: Path, result_path: Path) -> Tuple[int, int, int]:
    """
    Compare ground truth and result files.
    Returns: (total_facts, correct, missing)
    """
    try:
        with open(gt_path) as f:
            gt_data = json.load(f)

        if not result_path.exists():
            return len(gt_data), 0, len(gt_data)

        with open(result_path) as f:
            result_data = json.load(f)

        # Normalize ground truth annotations
        gt_annotations = {}
        for item in gt_data:
            key, kind, pred_type = normalize_annotation(item)
            gt_annotations[key] = pred_type

        # Normalize result annotations
        result_annotations = {}
        for item in result_data:
            key, kind, pred_type = normalize_annotation(item)
            result_annotations[key] = pred_type

        # Count matches
        total = len(gt_annotations)
        correct = 0
        missing = 0

        for key, gt_type in gt_annotations.items():
            if key in result_annotations:
                if gt_type == result_annotations[key]:
                    correct += 1
            else:
                missing += 1

        return total, correct, missing

    except Exception as e:
        print(f"Error processing {gt_path}: {e}")
        return 0, 0, 0


def analyze_run(results_dir: Path, tool_name: str) -> Dict[str, int]:
    """
    Analyze a single run for a specific tool.
    Returns stats for python_features only.
    """
    tool_dir = results_dir / tool_name / "micro-benchmark" / "python_features"

    if not tool_dir.exists():
        return None

    stats = {
        'total_facts': 0,
        'correct': 0,
        'missing': 0,
        'cases': 0
    }

    # Find all ground truth files in python_features
    for gt_file in tool_dir.rglob("main_gt.json"):
        result_file = gt_file.parent / "main_result.json"

        total, correct, missing = compare_files(gt_file, result_file)
        stats['total_facts'] += total
        stats['correct'] += correct
        stats['missing'] += missing
        stats['cases'] += 1

    if stats['total_facts'] > 0:
        stats['accuracy'] = stats['correct'] / stats['total_facts'] * 100
    else:
        stats['accuracy'] = 0.0

    return stats


def main():
    results_base = Path("/home/juan/project/TypeEvalPy/results")

    if not results_base.exists():
        print(f"Results directory not found: {results_base}")
        return

    # Collect all runs
    all_runs = []

    for run_dir in sorted(results_base.iterdir()):
        if not run_dir.is_dir():
            continue

        run_name = run_dir.name

        # Find what tools were run in this directory
        tools_in_run = []
        for item in run_dir.iterdir():
            if item.is_dir() and item.name not in ['analysis_results']:
                # Check if it has micro-benchmark/python_features
                if (item / "micro-benchmark" / "python_features").exists():
                    tools_in_run.append(item.name)

        if not tools_in_run:
            continue

        # Analyze each tool
        run_data = {
            'run_name': run_name,
            'tools': {}
        }

        for tool in tools_in_run:
            stats = analyze_run(run_dir, tool)
            if stats:
                run_data['tools'][tool] = stats

        if run_data['tools']:
            all_runs.append(run_data)

    # Print comparison table
    print("\n" + "="*100)
    print("Type Inference Comparison - python_features micro-benchmark")
    print("="*100)

    # Get all unique tools across all runs
    all_tools = set()
    for run in all_runs:
        all_tools.update(run['tools'].keys())
    all_tools = sorted(all_tools)

    # Build table data
    table_data = []
    for run in all_runs:
        row = [run['run_name']]
        for tool in all_tools:
            if tool in run['tools']:
                stats = run['tools'][tool]
                acc = stats['accuracy']
                correct = stats['correct']
                total = stats['total_facts']
                row.append(f"{correct}/{total} ({acc:.1f}%)")
            else:
                row.append("")
        table_data.append(row)

    # Print table
    headers = ["Run"] + all_tools
    print("\n" + tabulate(table_data, headers=headers, tablefmt="simple"))

    # Print detailed breakdown for latest run
    if all_runs:
        print("\n" + "="*100)
        print(f"Detailed breakdown for latest run: {all_runs[-1]['run_name']}")
        print("="*100)

        # Build detailed table
        detail_data = []
        for tool, stats in sorted(all_runs[-1]['tools'].items()):
            detail_data.append([
                tool,
                stats['cases'],
                stats['total_facts'],
                stats['correct'],
                stats['missing'],
                f"{stats['accuracy']:.2f}%"
            ])

        detail_headers = ["Tool", "Test Cases", "Total Facts", "Correct", "Missing", "Accuracy"]
        print("\n" + tabulate(detail_data, headers=detail_headers, tablefmt="simple"))

    # Save to CSV
    csv_path = results_base / "comparison_python_features.csv"
    with open(csv_path, 'w') as f:
        # Header
        f.write("Run," + ",".join(all_tools) + "\n")

        # Data rows
        for run in all_runs:
            f.write(f"{run['run_name']},")
            values = []
            for tool in all_tools:
                if tool in run['tools']:
                    stats = run['tools'][tool]
                    values.append(f"{stats['correct']}/{stats['total_facts']} ({stats['accuracy']:.1f}%)")
                else:
                    values.append("")
            f.write(",".join(values))
            f.write("\n")

    print(f"\n\nComparison saved to: {csv_path}")


if __name__ == "__main__":
    main()
