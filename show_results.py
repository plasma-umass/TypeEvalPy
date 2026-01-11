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
    - incomplete: True if predictions are correct but missing some GT types

    For union types (multiple types), ALL types must match, not just some.
    E.g., GT=['str', 'int'] requires pred to cover both str and int.
    """
    gt_set = set(gt_types)
    pred_set = set(pred_types)

    result = {
        'exact': False,
        'equivalence': False,
        'incomplete': False,
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

    # 3. Incomplete match
    # Predictions have some correct types but are incomplete (under-prediction only).
    # NO incorrect types are allowed - all pred types must be valid.
    # This means: all pred types covered by GT, but not all GT types covered by pred.

    # First, verify all prediction types are correct (covered by GT)
    all_pred_covered_by_gt = True
    for pred_type in pred_types:
        pred_equiv = get_equivalent_types(pred_type)
        if not (gt_set & pred_equiv):  # No intersection with GT
            all_pred_covered_by_gt = False
            break

    # Only check for incomplete if all predictions are correct
    if all_pred_covered_by_gt and not all_gt_covered:
        # All predictions are correct, but some GT types are missing
        result['incomplete'] = True

    return result


def analyze_with_normalizations(results_dir: Path, tool_name: str):
    """Analyze tool results with semantic equivalence."""
    tool_dir = results_dir / tool_name / "micro-benchmark" / "python_features"

    if not tool_dir.exists():
        return None

    stats = {
        'total': 0,
        'exact': 0,
        'exact_functions': 0,  # Functions (params + returns)
        'exact_variables': 0,   # Variables
        'total_functions': 0,
        'total_variables': 0,
        'equivalence': 0,
        'incomplete': 0,
        'missing': 0,
        'examples': {
            'equivalence_only': [],
            'incomplete': [],
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

            # Determine if this is a function or variable
            is_function = 'function' in gt_item
            is_variable = 'variable' in gt_item

            stats['total'] += 1
            if is_function:
                stats['total_functions'] += 1
            elif is_variable:
                stats['total_variables'] += 1

            if key not in result_lookup:
                stats['missing'] += 1
                continue

            pred_types = result_lookup[key]
            comparison = compare_types(gt_types, pred_types)

            if comparison['exact']:
                stats['exact'] += 1
                stats['equivalence'] += 1

                # Track category-specific exact matches
                if is_function:
                    stats['exact_functions'] += 1
                elif is_variable:
                    stats['exact_variables'] += 1

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
            elif comparison['incomplete']:
                stats['incomplete'] += 1

                # Save example
                example = {
                    'file': str(gt_file.relative_to(tool_dir)),
                    'line': gt_item['line_number'],
                    'gt': gt_types,
                    'pred': pred_types
                }

                if len(stats['examples']['incomplete']) < 5:
                    stats['examples']['incomplete'].append(example)

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Show type inference results")
    parser.add_argument('--latex', action='store_true', help='Output LaTeX table format')
    args = parser.parse_args()

    results_base = Path("/home/juan/project/TypeEvalPy/results")

    if not args.latex:
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

    # Sort tools alphabetically
    tool_order = sorted(latest_runs.keys())

    # Tool display names
    tool_labels = {
        'gpt-4o': 'GPT-4o',
        'righttyper': 'RightTyper',
        'quac': 'QuAC',
        'type4py': 'Type4Py',
        'monkeytype': 'MonkeyType',
    }

    if not args.latex:
        print(f"\nDiscovered latest runs:")
        for tool in tool_order:
            print(f"  {tool}: {latest_runs[tool]}")

    # Detailed per-tool output
    if not args.latex:
        for tool in tool_order:
            run = latest_runs[tool]
            run_dir = results_base / run
            if not run_dir.exists():
                continue

            stats = analyze_with_normalizations(run_dir, tool)
            if not stats:
                continue

            print(f"\n{tool_labels.get(tool, tool.upper())}")
            print("-" * 80)
            print(f"Total facts: {stats['total']}")
            print(f"Missing predictions: {stats['missing']}")
            print(f"Incomplete matches: {stats['incomplete']}")
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

            if stats['examples']['incomplete']:
                print("\n  Examples of incomplete matches (correct but missing types):")
                for ex in stats['examples']['incomplete']:
                    print(f"    {ex['file']}:{ex['line']}  GT: {ex['gt']}  Pred: {ex['pred']}")

    # Summary
    if not args.latex:
        print("\n" + "="*80)
        print("Summary Table")
        print("="*80)

    # Build table data with raw values
    # (tool_label, exact_funcs_pct, exact_vars_pct, exact_pct, equiv_pct, incomplete_pct,
    #  exact_funcs, total_funcs, exact_vars, total_vars, exact_count, total, equiv_count, incomplete_count)
    table_data_raw = []
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
        total_funcs = stats['total_functions']
        total_vars = stats['total_variables']

        row = [
            tool_labels.get(tool, tool),
            stats['exact_functions']/total_funcs*100 if total_funcs > 0 else 0,  # exact functions percentage
            stats['exact_variables']/total_vars*100 if total_vars > 0 else 0,    # exact variables percentage
            stats['exact']/total*100,                  # total exact percentage
            stats['equivalence']/total*100,            # equiv percentage
            stats['incomplete']/total*100,             # incomplete percentage
            stats['exact_functions'],
            total_funcs,
            stats['exact_variables'],
            total_vars,
            stats['exact'],
            total,
            stats['equivalence'],
            stats['incomplete']
        ]
        table_data_raw.append(row)

    # Sort by Total Semantic (equiv_pct, row[4]) in ascending order
    table_data_raw.sort(key=lambda row: row[4], reverse=False)

    # Find max values in each percentage column
    if table_data_raw:
        max_exact_funcs = max(row[1] for row in table_data_raw)
        max_exact_vars = max(row[2] for row in table_data_raw)
        max_exact = max(row[3] for row in table_data_raw)
        max_equiv = max(row[4] for row in table_data_raw)
        max_incomplete = max(row[5] for row in table_data_raw)
    else:
        max_exact_funcs = max_exact_vars = max_exact = max_equiv = max_incomplete = 0

    if args.latex:
        # LaTeX output - transposed (metrics as rows, tools as columns)
        num_tools = len(table_data_raw)
        col_spec = "l" + "r" * num_tools
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        print(r"\toprule")

        # Header: Metric & Tool1 & Tool2 & ... \\
        tool_names = [row[0] for row in table_data_raw]
        header = "Match & " + " & ".join(tool_names) + r" \\"
        print(header)
        print(r"\midrule")

        # Row 1: Functions (exact)
        func_values = []
        for row in table_data_raw:
            exact_funcs_pct, exact_vars_pct, exact_pct, equiv_pct, incomplete_pct, \
                exact_funcs, total_funcs, exact_vars, total_vars, exact_count, total, equiv_count, incomplete_count = row[1:]
            func_macro = r"\PCT" if exact_funcs_pct == max_exact_funcs else r"\pct"
            func_values.append(f"{func_macro}{{{exact_funcs_pct:.1f}}}")
        print("Functions & " + " & ".join(func_values) + r" \\")

        # Row 2: Variables (exact)
        var_values = []
        for row in table_data_raw:
            exact_funcs_pct, exact_vars_pct, exact_pct, equiv_pct, incomplete_pct, \
                exact_funcs, total_funcs, exact_vars, total_vars, exact_count, total, equiv_count, incomplete_count = row[1:]
            var_macro = r"\PCT" if exact_vars_pct == max_exact_vars else r"\pct"
            var_values.append(f"{var_macro}{{{exact_vars_pct:.1f}}}")
        print("Variables & " + " & ".join(var_values) + r" \\")
        print(r"\midrule")

        # Row 3: Total Exact Match
        exact_values = []
        for row in table_data_raw:
            exact_funcs_pct, exact_vars_pct, exact_pct, equiv_pct, incomplete_pct, \
                exact_funcs, total_funcs, exact_vars, total_vars, exact_count, total, equiv_count, incomplete_count = row[1:]
            exact_macro = r"\PCT" if exact_pct == max_exact else r"\pct"
            exact_values.append(f"{exact_macro}{{{exact_pct:.1f}}}")
        print("Total Exact & " + " & ".join(exact_values) + r" \\")

        # Row 4: Total Semantic Match
        equiv_values = []
        for row in table_data_raw:
            exact_funcs_pct, exact_vars_pct, exact_pct, equiv_pct, incomplete_pct, \
                exact_funcs, total_funcs, exact_vars, total_vars, exact_count, total, equiv_count, incomplete_count = row[1:]
            equiv_macro = r"\PCT" if equiv_pct == max_equiv else r"\pct"
            equiv_values.append(f"{equiv_macro}{{{equiv_pct:.1f}}}")
        print("Total Semantic & " + " & ".join(equiv_values) + r" \\")

        print(r"\bottomrule")
        print(r"\end{tabular}")

        # Add note with totals
        # Get totals from first row (all rows have same totals)
        if table_data_raw:
            _, _, _, _, _, _, total_funcs, _, total_vars, _, total, _, _ = table_data_raw[0][1:]
            print(f"\\\\[0.5em]")
            print(f"\\small Results on {total} type annotations ({total_funcs} functions, {total_vars} variables).")
    else:
        # Normal tabulate output - transposed (metrics as rows, tools as columns)
        tool_names = [row[0] for row in table_data_raw]

        # Build transposed table data
        table_data = []

        # Row 1: Functions (exact)
        func_row = ["Functions"]
        for row in table_data_raw:
            exact_funcs_pct, exact_vars_pct, exact_pct, equiv_pct, incomplete_pct, \
                exact_funcs, total_funcs, exact_vars, total_vars, exact_count, total, equiv_count, incomplete_count = row[1:]
            func_row.append(f"{exact_funcs}/{total_funcs} ({exact_funcs_pct:.1f}%)")
        table_data.append(func_row)

        # Row 2: Variables (exact)
        var_row = ["Variables"]
        for row in table_data_raw:
            exact_funcs_pct, exact_vars_pct, exact_pct, equiv_pct, incomplete_pct, \
                exact_funcs, total_funcs, exact_vars, total_vars, exact_count, total, equiv_count, incomplete_count = row[1:]
            var_row.append(f"{exact_vars}/{total_vars} ({exact_vars_pct:.1f}%)")
        table_data.append(var_row)

        # Row 3: Total Exact Match
        exact_row = ["Total Exact"]
        for row in table_data_raw:
            exact_funcs_pct, exact_vars_pct, exact_pct, equiv_pct, incomplete_pct, \
                exact_funcs, total_funcs, exact_vars, total_vars, exact_count, total, equiv_count, incomplete_count = row[1:]
            exact_row.append(f"{exact_count}/{total} ({exact_pct:.1f}%)")
        table_data.append(exact_row)

        # Row 4: Total Semantic Match
        equiv_row = ["Total Semantic"]
        for row in table_data_raw:
            exact_funcs_pct, exact_vars_pct, exact_pct, equiv_pct, incomplete_pct, \
                exact_funcs, total_funcs, exact_vars, total_vars, exact_count, total, equiv_count, incomplete_count = row[1:]
            equiv_row.append(f"{equiv_count}/{total} ({equiv_pct:.1f}%)")
        table_data.append(equiv_row)

        headers = ["Match"] + tool_names
        print(tabulate(table_data, headers=headers, tablefmt="simple"))


if __name__ == "__main__":
    main()
