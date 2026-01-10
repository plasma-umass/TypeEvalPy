#!/usr/bin/env python3
"""
Type comparison with semantic equivalence.
Accepts types that are functionally equivalent (can be used interchangeably).
"""

import json
from pathlib import Path
from typing import List, Set
from tabulate import tabulate


# Type equivalence rules - types that are functionally interchangeable
# For non-builtins, requires qualified names (e.g., itertools.count, not just count)
# This is for functional equivalence (e.g., zip is-a iterator), NOT case variations
# (case variations should be handled by each runner's normalizer)
TYPE_EQUIVALENCES = {
    # Builtin iterator types - all are iterators
    # Note: these are builtins, so no qualification needed
    'zip': {'iterator', 'zip'},
    'map': {'iterator', 'map'},
    'filter': {'iterator', 'filter'},
    'enumerate': {'iterator', 'enumerate'},
    'range': {'iterator', 'range'},
    'reversed': {'iterator', 'reversed'},

    # itertools types - require full qualification (itertools.*)
    # Only the qualified version is equivalent to iterator
    'itertools.chain': {'iterator', 'itertools.chain'},
    'itertools.compress': {'iterator', 'itertools.compress'},
    'itertools.count': {'iterator', 'itertools.count'},
    'itertools.cycle': {'iterator', 'itertools.cycle'},
    'itertools.permutations': {'iterator', 'itertools.permutations'},
    'itertools.combinations': {'iterator', 'itertools.combinations'},
    'itertools.product': {'iterator', 'itertools.product'},
    'itertools.repeat': {'iterator', 'itertools.repeat'},
    'itertools.groupby': {'iterator', 'itertools.groupby'},
    'itertools._grouper': {'iterator', 'itertools._grouper'},

    # Iterator itself (and all types that are iterators)
    'iterator': {'iterator', 'zip', 'map', 'filter', 'enumerate', 'range', 'reversed',
                 'itertools.chain', 'itertools.compress', 'itertools.count', 'itertools.cycle',
                 'itertools.permutations', 'itertools.combinations', 'itertools.product',
                 'itertools.repeat', 'itertools.groupby', 'itertools._grouper', 'generator'},

    # Generator is also an iterator
    'generator': {'iterator', 'generator'},
}


def get_equivalent_types(type_str: str) -> Set[str]:
    """
    Get all types equivalent to the given type.
    Returns a set including the type itself and all its equivalents.
    Only considers exact functional equivalence (case variations, etc).
    """
    # Start with the type itself
    equiv_set = {type_str}

    # Check if this type has defined equivalences (case-insensitive lookup)
    type_lower = type_str.lower()
    for key, values in TYPE_EQUIVALENCES.items():
        if key.lower() == type_lower:
            equiv_set.update(values)
            break

    return equiv_set


def compare_types(gt_types: List[str], pred_types: List[str]) -> dict:
    """
    Compare ground truth and predicted types.

    Returns a dict with keys:
    - exact: True if exact match
    - equivalence: True if semantically equivalent (same functionality)

    For union types (multiple types), ALL types must match, not just some.
    E.g., GT=['str', 'int'] requires pred to cover both str and int.
    """
    gt_set = set(gt_types)
    pred_set = set(pred_types)

    result = {
        'exact': False,
        'equivalence': False,
    }

    # 1. Exact match
    if gt_set == pred_set:
        result['exact'] = True
        result['equivalence'] = True
        return result

    # 2. Semantic equivalence
    # For equivalence, check both directions:
    # - All GT types must be covered by predictions (via equivalence)
    # - All prediction types must be covered by GT (via equivalence)
    # This prevents partial matches like GT=['str', 'int'] matching pred=['str']

    # Check: all GT types covered by predictions?
    all_gt_covered = True
    for gt_type in gt_types:
        gt_equiv = get_equivalent_types(gt_type)
        if not (pred_set & gt_equiv):  # No intersection
            all_gt_covered = False
            break

    # Check: all prediction types covered by GT?
    all_pred_covered = True
    for pred_type in pred_types:
        pred_equiv = get_equivalent_types(pred_type)
        if not (gt_set & pred_equiv):  # No intersection
            all_pred_covered = False
            break

    if all_gt_covered and all_pred_covered:
        result['equivalence'] = True

    return result


def analyze_with_normalizations(results_dir: Path, tool_name: str):
    """Analyze tool results with semantic equivalence."""
    tool_dir = results_dir / tool_name / "micro-benchmark" / "python_features"

    if not tool_dir.exists():
        return None

    stats = {
        'total': 0,
        'exact': 0,
        'equivalence': 0,
        'missing': 0,
        'examples': {
            'equivalence_only': [],
        }
    }

    for gt_file in tool_dir.rglob("main_gt.json"):
        result_file = gt_file.parent / "main_result.json"

        with open(gt_file) as f:
            gt_data = json.load(f)

        if not result_file.exists():
            stats['missing'] += len(gt_data)
            stats['total'] += len(gt_data)
            continue

        with open(result_file) as f:
            result_data = json.load(f)

        # Create lookup
        result_lookup = {}
        for item in result_data:
            key = (item['file'], item['line_number'], item['col_offset'])
            result_lookup[key] = item.get('type', [])

        # Compare
        for gt_item in gt_data:
            key = (gt_item['file'], gt_item['line_number'], gt_item['col_offset'])
            gt_types = gt_item.get('type', [])

            stats['total'] += 1

            if key not in result_lookup:
                stats['missing'] += 1
                continue

            pred_types = result_lookup[key]
            comparison = compare_types(gt_types, pred_types)

            if comparison['exact']:
                stats['exact'] += 1
                stats['equivalence'] += 1
            elif comparison['equivalence']:
                stats['equivalence'] += 1

                # Save example
                example = {
                    'file': str(gt_file.relative_to(tool_dir)),
                    'line': gt_item['line_number'],
                    'gt': gt_types,
                    'pred': pred_types
                }

                if len(stats['examples']['equivalence_only']) < 5:
                    stats['examples']['equivalence_only'].append(example)

    return stats


def main():
    results_base = Path("/home/juan/project/TypeEvalPy/results")

    print("\n" + "="*80)
    print("Type Comparison with Semantic Equivalence")
    print("="*80)

    # Automatically discover latest run for each tool
    latest_runs = {}  # tool_name -> run_dir_name

    # Iterate through all run directories (sorted in reverse to get latest first)
    for run_dir in sorted(results_base.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        # Check what tools exist in this run
        for item in run_dir.iterdir():
            if item.is_dir() and item.name not in ['analysis_results']:
                tool_dir = item / "micro-benchmark" / "python_features"
                if tool_dir.exists():
                    # This is a valid tool result
                    tool_name = item.name
                    # Only save if we haven't seen this tool yet (latest)
                    if tool_name not in latest_runs:
                        latest_runs[tool_name] = run_dir.name

    # Sort tools alphabetically, but keep RightTyper at the end
    tool_order = sorted([t for t in latest_runs.keys() if t.lower() != 'righttyper'])
    if 'righttyper' in latest_runs:
        tool_order.append('righttyper')

    print(f"\nDiscovered latest runs:")
    for tool in tool_order:
        print(f"  {tool}: {latest_runs[tool]}")

    for tool in tool_order:
        run = latest_runs[tool]
        run_dir = results_base / run
        if not run_dir.exists():
            continue

        stats = analyze_with_normalizations(run_dir, tool)
        if not stats:
            continue

        print(f"\n{tool.upper()}")
        print("-" * 80)
        print(f"Total facts: {stats['total']}")
        print(f"Missing predictions: {stats['missing']}")
        print()

        # Always use total facts as denominator for fair comparison
        total = stats['total']

        # Build detail table
        detail_data = [
            ["Exact match", f"{stats['exact']}/{total}", f"{stats['exact']/total*100:.2f}%", ""],
            ["+ Semantic equiv", f"{stats['equivalence']}/{total}", f"{stats['equivalence']/total*100:.2f}%", f"+{stats['equivalence'] - stats['exact']}"],
        ]
        print(tabulate(detail_data, headers=["Comparison", "Correct", "Accuracy", "Gain"], tablefmt="simple"))

        # Show examples
        if stats['examples']['equivalence_only']:
            print("\n  Examples fixed by semantic equivalence (same functionality):")
            for ex in stats['examples']['equivalence_only']:
                print(f"    {ex['file']}:{ex['line']}  {ex['gt']} -> {ex['pred']}")

    # Summary
    print("\n" + "="*80)
    print("Summary Table")
    print("="*80)

    # Build table data
    table_data = []
    for tool in tool_order:
        run = latest_runs[tool]
        run_dir = results_base / run
        if not run_dir.exists():
            continue

        stats = analyze_with_normalizations(run_dir, tool)
        if not stats:
            continue

        # Always use total facts as denominator for fair comparison
        total = stats['total']

        row = [
            tool,
            f"{stats['exact']}/{total} ({stats['exact']/total*100:.1f}%)",
            f"{stats['equivalence']}/{total} ({stats['equivalence']/total*100:.1f}%)"
        ]
        table_data.append(row)

    headers = ["Tool", "Exact Match", "With Sem. Equiv."]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))


if __name__ == "__main__":
    main()
