#!/usr/bin/env python3
"""
Type comparison using TypeSim for semantic similarity scoring.
"""

import json
import sys
from pathlib import Path
from typing import List
from tabulate import tabulate

# Import TypeSim from sibling directory
sys.path.insert(0, str(Path(__file__).parent.parent / "righttyper-eval" / "src"))
from typesim2 import get_type_similarity


# Reverse mapping for TypeEvalPy's type normalizations
# Original _NAME_MAP in type_normalizer.py strips qualifications; we restore them
TYPEEVALPY_DENORMALIZE = {
    # Callable (was typing.Callable or collections.abc.Callable)
    'callable': 'typing.Callable',
    # Iterator (was typing.Iterator or collections.abc.Iterator)
    'iterator': 'typing.Iterator',
    # Generator (was typing.Generator or collections.abc.Generator)
    'generator': 'typing.Generator',
    # Code (was types.CodeType)
    'code': 'types.CodeType',
    # None (was None, normalized to Nonetype)
    'Nonetype': 'None',
    # Note: 'type' is kept as-is - it's the builtin metaclass, not typing.Type
}


def normalize_type_for_typesim(type_str: str) -> str:
    """
    Reverse TypeEvalPy's type normalizations before passing to TypeSim.

    TypeEvalPy's type_normalizer.py strips module qualifications and lowercases
    some types. We restore them for proper TypeSim comparison.
    """
    return TYPEEVALPY_DENORMALIZE.get(type_str, type_str)


def types_to_annotation(types: List[str]) -> str:
    """Convert a list of types to a single annotation string for TypeSim."""
    if not types:
        return ""
    # Normalize each type
    normalized = [normalize_type_for_typesim(t) for t in types]
    if len(normalized) == 1:
        return normalized[0]
    # Multiple types -> union
    return " | ".join(sorted(normalized))


def compare_types(gt_types: List[str], pred_types: List[str]) -> dict:
    """
    Compare ground truth and predicted types.

    Returns a dict with keys:
    - exact: True if exact string match
    """
    gt_set = set(gt_types)
    pred_set = set(pred_types)

    result = {
        'exact': gt_set == pred_set,
    }

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
        'missing': 0,
        'missing_functions': 0,
        'missing_variables': 0,
        # TypeSim scores (computed over ALL ground truth entries)
        'typesim_total': 0.0,  # Sum of all TypeSim scores (missing = 0)
        'typesim_perfect': 0,   # TypeSim == 1.0
        'typesim_functions_total': 0.0,
        'typesim_variables_total': 0.0,
        'typesim_perfect_functions': 0,
        'typesim_perfect_variables': 0,
        'examples': {
            'typesim_partial': [],  # Examples where TypeSim gave partial credit
            'typesim_zero': [],     # Examples where TypeSim == 0 (complete mismatch)
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
                if is_function:
                    stats['missing_functions'] += 1
                elif is_variable:
                    stats['missing_variables'] += 1
                # Missing prediction = TypeSim score of 0 (already 0, nothing to add)
                continue

            pred_types = result_lookup[key]
            comparison = compare_types(gt_types, pred_types)

            # Compute TypeSim score (over ALL ground truth entries)
            if gt_types:
                if pred_types:
                    gt_annotation = types_to_annotation(gt_types)
                    pred_annotation = types_to_annotation(pred_types)
                    try:
                        typesim_score = get_type_similarity(gt_annotation, pred_annotation)
                    except Exception:
                        typesim_score = 0.0
                else:
                    typesim_score = 0.0  # No prediction = 0 score

                stats['typesim_total'] += typesim_score

                if typesim_score == 1.0:
                    stats['typesim_perfect'] += 1

                if is_function:
                    stats['typesim_functions_total'] += typesim_score
                elif is_variable:
                    stats['typesim_variables_total'] += typesim_score

                # Track TypeSim == 1.0 for functions/variables
                if typesim_score == 1.0:
                    if is_function:
                        stats['typesim_perfect_functions'] += 1
                    elif is_variable:
                        stats['typesim_perfect_variables'] += 1

                # Save examples where TypeSim gives partial credit but not exact
                if 0 < typesim_score < 1.0 and not comparison['exact']:
                    example = {
                        'file': str(gt_file.relative_to(tool_dir)),
                        'line': gt_item['line_number'],
                        'gt': gt_types,
                        'pred': pred_types,
                        'score': typesim_score
                    }
                    if len(stats['examples']['typesim_partial']) < 5:
                        stats['examples']['typesim_partial'].append(example)

                # Save examples where TypeSim == 0 (complete mismatch)
                if typesim_score == 0.0:
                    example = {
                        'file': str(gt_file.relative_to(tool_dir)),
                        'line': gt_item['line_number'],
                        'gt': gt_types,
                        'pred': pred_types,
                    }
                    if len(stats['examples']['typesim_zero']) < 5:
                        stats['examples']['typesim_zero'].append(example)

            if comparison['exact']:
                stats['exact'] += 1

                # Track category-specific exact matches
                if is_function:
                    stats['exact_functions'] += 1
                elif is_variable:
                    stats['exact_variables'] += 1

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Show type inference results")
    parser.add_argument('--latex', action='store_true', help='Output LaTeX table format')
    args = parser.parse_args()

    results_base = Path("/home/juan/project/TypeEvalPy/results")

    if not args.latex:
        print("\n" + "="*80)
        print("Type Comparison Results")
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
            coverage = (stats['total'] - stats['missing']) / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"Total facts: {stats['total']}, Coverage: {coverage:.1f}%")
            print(f"Missing predictions: {stats['missing']}")
            print()

            # Always use total facts as denominator for fair comparison
            total = stats['total']

            # Build detail table
            detail_data = []

            # Add TypeSim rows (computed over all ground truth entries)
            if total > 0:
                avg_typesim = stats['typesim_total'] / total * 100  # As percentage
                detail_data.append(["TypeSim", "", f"{avg_typesim:.2f}%"])
                detail_data.append(["TypeSim == 1.0", f"{stats['typesim_perfect']}/{total}", f"{stats['typesim_perfect']/total*100:.2f}%"])

            detail_data.append(["TypeEvalPy Exact", f"{stats['exact']}/{total}", f"{stats['exact']/total*100:.2f}%"])

            print(tabulate(detail_data, headers=["Metric", "Count", "Percentage"], tablefmt="simple"))

            # Show examples
            if stats['examples']['typesim_partial']:
                print("\n  Examples where TypeSim gave partial credit (0 < score < 1.0):")
                for ex in stats['examples']['typesim_partial']:
                    print(f"    {ex['file']}:{ex['line']}  GT: {ex['gt']}  Pred: {ex['pred']}  (score: {ex['score']:.2f})")

            if stats['examples']['typesim_zero']:
                print("\n  Examples where TypeSim == 0 (complete mismatch):")
                for ex in stats['examples']['typesim_zero']:
                    print(f"    {ex['file']}:{ex['line']}  GT: {ex['gt']}  Pred: {ex['pred']}")

    # Summary
    if not args.latex:
        print("\n" + "="*80)
        print("Summary Table")
        print("="*80)

    # Build table data with raw values
    # Row structure:
    # [0] tool_label
    # [1] exact_funcs_pct, [2] exact_vars_pct, [3] exact_pct
    # [4] total_funcs, [5] total_vars, [6] total
    # [7] coverage_pct, [8] coverage_funcs_pct, [9] coverage_vars_pct
    # [10] typesim_avg_pct, [11] typesim_perfect_pct
    # [12] typesim_funcs_avg_pct, [13] typesim_vars_avg_pct
    # [14] typesim_perfect_funcs_pct, [15] typesim_perfect_vars_pct
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

        # Coverage (percentage of GT items with predictions)
        coverage_pct = (total - stats['missing']) / total * 100 if total > 0 else 0
        coverage_funcs_pct = (total_funcs - stats['missing_functions']) / total_funcs * 100 if total_funcs > 0 else 0
        coverage_vars_pct = (total_vars - stats['missing_variables']) / total_vars * 100 if total_vars > 0 else 0

        # TypeSim stats (computed over ALL ground truth entries, as percentages)
        typesim_avg_pct = stats['typesim_total'] / total * 100 if total > 0 else 0
        typesim_perfect_pct = stats['typesim_perfect'] / total * 100 if total > 0 else 0
        typesim_funcs_avg_pct = stats['typesim_functions_total'] / total_funcs * 100 if total_funcs > 0 else 0
        typesim_vars_avg_pct = stats['typesim_variables_total'] / total_vars * 100 if total_vars > 0 else 0
        typesim_perfect_funcs_pct = stats['typesim_perfect_functions'] / total_funcs * 100 if total_funcs > 0 else 0
        typesim_perfect_vars_pct = stats['typesim_perfect_variables'] / total_vars * 100 if total_vars > 0 else 0

        row = [
            tool_labels.get(tool, tool),                                        # 0: tool label
            stats['exact_functions']/total_funcs*100 if total_funcs > 0 else 0, # 1: exact functions percentage
            stats['exact_variables']/total_vars*100 if total_vars > 0 else 0,   # 2: exact variables percentage
            stats['exact']/total*100,                                           # 3: total exact percentage
            total_funcs,                                                        # 4
            total_vars,                                                         # 5
            total,                                                              # 6
            coverage_pct,                                                       # 7: overall coverage percentage
            coverage_funcs_pct,                                                 # 8: functions coverage percentage
            coverage_vars_pct,                                                  # 9: variables coverage percentage
            typesim_avg_pct,                                                    # 10: average TypeSim as percentage
            typesim_perfect_pct,                                                # 11: TypeSim == 1.0
            typesim_funcs_avg_pct,                                              # 12: functions TypeSim avg as percentage
            typesim_vars_avg_pct,                                               # 13: variables TypeSim avg as percentage
            typesim_perfect_funcs_pct,                                          # 14: TypeSim == 1.0 for functions
            typesim_perfect_vars_pct,                                           # 15: TypeSim == 1.0 for variables
        ]
        table_data_raw.append(row)

    # Sort by TypeSim average (row[10]) in ascending order
    table_data_raw.sort(key=lambda row: row[10], reverse=False)

    # Find max values in each percentage column (for LaTeX bolding)
    if table_data_raw:
        max_coverage = max(row[7] for row in table_data_raw)
        max_coverage_funcs = max(row[8] for row in table_data_raw)
        max_coverage_vars = max(row[9] for row in table_data_raw)
        max_typesim = max(row[10] for row in table_data_raw)
        max_typesim_perfect = max(row[11] for row in table_data_raw)
        max_typesim_funcs = max(row[12] for row in table_data_raw)
        max_typesim_vars = max(row[13] for row in table_data_raw)
        max_typesim_perfect_funcs = max(row[14] for row in table_data_raw)
        max_typesim_perfect_vars = max(row[15] for row in table_data_raw)
    else:
        max_coverage = max_coverage_funcs = max_coverage_vars = 0
        max_typesim = max_typesim_perfect = max_typesim_funcs = max_typesim_vars = 0
        max_typesim_perfect_funcs = max_typesim_perfect_vars = 0

    if args.latex:
        # LaTeX output - transposed with vertical category labels using multirow/rotatebox
        # Requires: \usepackage{multirow}, \usepackage{graphicx}
        num_tools = len(table_data_raw)
        # c for category column, l for metric, then r for each tool
        col_spec = "c l@{\\hspace{4em}}" + " r" * num_tools
        print("begin{table*}")
        print(f"% Requires: \\usepackage{{multirow}}, \\usepackage{{graphicx}}")
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        print(r"\toprule")

        # Header: & Metric & Tool1 & Tool2 & ... \\
        tool_names = [row[0] for row in table_data_raw]
        header = "& Metric & " + " & ".join(tool_names) + r" \\"
        print(header)
        print(r"\midrule")

        # Overall section (3 rows: TypeSim, TypeSim == 1.0, Coverage)
        # Row 1: TypeSim
        typesim_values = []
        for row in table_data_raw:
            typesim_pct = row[10]
            typesim_macro = r"\PCT" if typesim_pct == max_typesim else r"\pct"
            typesim_values.append(f"{typesim_macro}{{{typesim_pct:.1f}}}")
        print(r"\multirow{3}{*}{\rotatebox{90}{\scriptsize overall}} & TypeSim & " + " & ".join(typesim_values) + r" \\")

        # Row 2: TypeSim == 1.0
        typesim_perfect_values = []
        for row in table_data_raw:
            typesim_perfect_pct = row[11]
            typesim_perfect_macro = r"\PCT" if typesim_perfect_pct == max_typesim_perfect else r"\pct"
            typesim_perfect_values.append(f"{typesim_perfect_macro}{{{typesim_perfect_pct:.1f}}}")
        print(r"& TypeSim == 1.0 & " + " & ".join(typesim_perfect_values) + r" \\")

        # Row 3: Coverage
        coverage_values = []
        for row in table_data_raw:
            coverage_pct = row[7]
            coverage_macro = r"\PCT" if coverage_pct == max_coverage else r"\pct"
            coverage_values.append(f"{coverage_macro}{{{coverage_pct:.1f}}}")
        print(r"& Coverage & " + " & ".join(coverage_values) + r" \\")
        print(r"\midrule")

        # Functions section (3 rows: TypeSim, TypeSim == 1.0, Coverage)
        typesim_funcs_values = []
        for row in table_data_raw:
            typesim_funcs_pct = row[12]
            typesim_funcs_macro = r"\PCT" if typesim_funcs_pct == max_typesim_funcs else r"\pct"
            typesim_funcs_values.append(f"{typesim_funcs_macro}{{{typesim_funcs_pct:.1f}}}")
        print(r"\multirow{3}{*}{\rotatebox{90}{\scriptsize funcs.}} & TypeSim & " + " & ".join(typesim_funcs_values) + r" \\")

        typesim_perfect_funcs_values = []
        for row in table_data_raw:
            typesim_perfect_funcs_pct = row[14]
            func_macro = r"\PCT" if typesim_perfect_funcs_pct == max_typesim_perfect_funcs else r"\pct"
            typesim_perfect_funcs_values.append(f"{func_macro}{{{typesim_perfect_funcs_pct:.1f}}}")
        print(r"& TypeSim == 1.0 & " + " & ".join(typesim_perfect_funcs_values) + r" \\")

        func_coverage_values = []
        for row in table_data_raw:
            coverage_funcs_pct = row[8]
            func_coverage_macro = r"\PCT" if coverage_funcs_pct == max_coverage_funcs else r"\pct"
            func_coverage_values.append(f"{func_coverage_macro}{{{coverage_funcs_pct:.1f}}}")
        print(r"& Coverage & " + " & ".join(func_coverage_values) + r" \\")
        print(r"\midrule")

        # Variables section (3 rows: TypeSim, TypeSim == 1.0, Coverage)
        typesim_vars_values = []
        for row in table_data_raw:
            typesim_vars_pct = row[13]
            typesim_vars_macro = r"\PCT" if typesim_vars_pct == max_typesim_vars else r"\pct"
            typesim_vars_values.append(f"{typesim_vars_macro}{{{typesim_vars_pct:.1f}}}")
        print(r"\multirow{3}{*}{\rotatebox{90}{\scriptsize vars.}} & TypeSim & " + " & ".join(typesim_vars_values) + r" \\")

        typesim_perfect_vars_values = []
        for row in table_data_raw:
            typesim_perfect_vars_pct = row[15]
            var_macro = r"\PCT" if typesim_perfect_vars_pct == max_typesim_perfect_vars else r"\pct"
            typesim_perfect_vars_values.append(f"{var_macro}{{{typesim_perfect_vars_pct:.1f}}}")
        print(r"& TypeSim == 1.0 & " + " & ".join(typesim_perfect_vars_values) + r" \\")

        var_coverage_values = []
        for row in table_data_raw:
            coverage_vars_pct = row[9]
            var_coverage_macro = r"\PCT" if coverage_vars_pct == max_coverage_vars else r"\pct"
            var_coverage_values.append(f"{var_coverage_macro}{{{coverage_vars_pct:.1f}}}")
        print(r"& Coverage & " + " & ".join(var_coverage_values) + r" \\")

        print(r"\bottomrule")
        print(r"\end{tabular}")

        # Add note with totals
        # Get totals from first row (all rows have same totals)
        if table_data_raw:
            total_funcs = int(table_data_raw[0][4])
            total_vars = int(table_data_raw[0][5])
            total = int(table_data_raw[0][6])
            print(f"\\\\[0.5em]")
            print(f"\\small Results on {total} type annotations ({total_funcs} functions, {total_vars} variables).")

        print("end{table*}")
    else:
        # Normal tabulate output - transposed (metrics as rows, tools as columns)
        # Row indices:
        # [0] tool_label
        # [1] exact_funcs_pct, [2] exact_vars_pct, [3] exact_pct
        # [4] total_funcs, [5] total_vars, [6] total
        # [7] coverage_pct, [8] coverage_funcs_pct, [9] coverage_vars_pct
        # [10] typesim_avg_pct, [11] typesim_perfect_pct
        # [12] typesim_funcs_avg_pct, [13] typesim_vars_avg_pct
        # [14] typesim_perfect_funcs_pct, [15] typesim_perfect_vars_pct
        tool_names = [row[0] for row in table_data_raw]

        # Build transposed table data
        table_data = []

        # === OVERALL SECTION ===
        # TypeSim
        typesim_row = ["TypeSim"]
        for row in table_data_raw:
            typesim_row.append(f"{row[10]:.1f}%")
        table_data.append(typesim_row)

        # TypeSim == 1.0
        typesim_perfect_row = ["TypeSim == 1.0"]
        for row in table_data_raw:
            typesim_perfect_row.append(f"{row[11]:.1f}%")
        table_data.append(typesim_perfect_row)

        # TypeEvalPy Exact
        exact_row = ["TypeEvalPy Exact"]
        for row in table_data_raw:
            exact_row.append(f"{row[3]:.1f}%")
        table_data.append(exact_row)

        # Coverage
        coverage_row = ["Coverage"]
        for row in table_data_raw:
            coverage_row.append(f"{row[7]:.1f}%")
        table_data.append(coverage_row)

        # Separator row
        table_data.append(["---"] + ["---"] * len(tool_names))

        # === FUNCTIONS SECTION ===
        # TypeSim Funcs
        typesim_funcs_row = ["TypeSim Funcs"]
        for row in table_data_raw:
            typesim_funcs_row.append(f"{row[12]:.1f}%")
        table_data.append(typesim_funcs_row)

        # TypeEvalPy Funcs Exact
        func_exact_row = ["TypeEvalPy Funcs"]
        for row in table_data_raw:
            func_exact_row.append(f"{row[1]:.1f}%")
        table_data.append(func_exact_row)

        # Funcs Coverage
        func_coverage_row = ["Funcs Coverage"]
        for row in table_data_raw:
            func_coverage_row.append(f"{row[8]:.1f}%")
        table_data.append(func_coverage_row)

        # Separator row
        table_data.append(["---"] + ["---"] * len(tool_names))

        # === VARIABLES SECTION ===
        # TypeSim Vars
        typesim_vars_row = ["TypeSim Vars"]
        for row in table_data_raw:
            typesim_vars_row.append(f"{row[13]:.1f}%")
        table_data.append(typesim_vars_row)

        # TypeEvalPy Vars Exact
        var_exact_row = ["TypeEvalPy Vars"]
        for row in table_data_raw:
            var_exact_row.append(f"{row[2]:.1f}%")
        table_data.append(var_exact_row)

        # Vars Coverage
        var_coverage_row = ["Vars Coverage"]
        for row in table_data_raw:
            var_coverage_row.append(f"{row[9]:.1f}%")
        table_data.append(var_coverage_row)

        headers = ["Metric"] + tool_names
        print(tabulate(table_data, headers=headers, tablefmt="simple"))


if __name__ == "__main__":
    main()
