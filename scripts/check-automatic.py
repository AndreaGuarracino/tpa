#!/usr/bin/env python3
"""
Check whether 'automatic' mode selected the winning strategy pair per dataset.

Usage:
    ./test/check-automatic.py [all_results.tsv]

Defaults to test/all_results.tsv.
"""

import sys
import warnings
import pandas as pd
from pathlib import Path


def pick_best(df, metric="ratio"):
    if metric == "ratio":
        return df['ratio_orig_to_tpa'].idxmax()
    if metric == "score":
        df = df.copy()
        df['score'] = df['ratio_orig_to_tpa'] / (df['seek_mode_b_avg_us'] / 10.0)
        return df['score'].idxmax()
    raise ValueError(f"unknown metric {metric}")


def main():
    tsv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test/all_results.tsv")
    if not tsv_path.exists():
        sys.exit(f"TSV not found: {tsv_path}")

    warnings.simplefilter("ignore", pd.errors.ParserWarning)
    df = pd.read_csv(tsv_path, sep="\t", index_col=False, engine="python")
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    if 'layer_first' not in df.columns or 'layer_second' not in df.columns:
        expected_cols = [
            'dataset_name', 'dataset_type', 'original_size_bytes', 'num_records',
            'encoding_type', 'encoding_runtime_sec', 'encoding_memory_mb',
            'tp_file_size_bytes', 'max_complexity', 'complexity_metric',
            'compression_strategy', 'strategy_first', 'strategy_second',
            'layer_first', 'layer_second', 'compression_runtime_sec',
            'compression_memory_mb', 'tpa_size_bytes', 'ratio_orig_to_tp',
            'ratio_tp_to_tpa', 'ratio_orig_to_tpa', 'decompression_runtime_sec',
            'decompression_memory_mb', 'verification_passed',
            'seek_positions_tested', 'seek_iterations_per_position',
            'seek_mode_a_avg_us', 'seek_mode_a_stddev_us', 'seek_mode_b_avg_us',
            'seek_mode_b_stddev_us', 'seek_decode_ratio', 'seek_valid_ratio',
            'file_open_time_us'
        ]
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            names=expected_cols,
            header=0,
            skiprows=1,
            engine="python",
            index_col=False,
        )
    df['is_auto'] = df['compression_strategy'].str.startswith('automatic')
    has_layer_cols = all(col in df.columns for col in ('layer_first', 'layer_second'))

    metrics = ["ratio"]  # ratio best

    for metric in metrics:
        print(f"=== Metric: {metric} ===")
        for ds, g in df.groupby('dataset_name'):
            if g.empty:
                continue
            manual = g[~g['is_auto']]
            if manual.empty:
                print(f"[{ds}] no manual rows to compare against")
                continue
            best_idx = pick_best(manual, metric=metric)
            best = manual.loc[best_idx]
            if isinstance(best, pd.DataFrame):
                best = best.iloc[0]
            autos = g[g['is_auto']]

            best_layer_first = best['layer_first'] if has_layer_cols else "n/a"
            best_layer_second = best['layer_second'] if has_layer_cols else "n/a"
            print(
                f"[{ds}] best: {best.strategy_first}[{best_layer_first}]"
                f"→{best.strategy_second}[{best_layer_second}]"
                f" ({metric}={best['ratio_tp_to_tpa']:.3f}; seek={best['seek_mode_b_avg_us']:.2f} μs)"
            )
            if autos.empty:
                print("  no automatic rows")
                continue

            for auto_name, group in autos.groupby('compression_strategy'):
                for _, row in group.iterrows():
                    row_layer_first = row['layer_first'] if has_layer_cols else "n/a"
                    row_layer_second = row['layer_second'] if has_layer_cols else "n/a"
                    match = (
                        row.strategy_first == best.strategy_first
                        and row.strategy_second == best.strategy_second
                    )
                    if has_layer_cols:
                        match = (
                            match
                            and row_layer_first == best_layer_first
                            and row_layer_second == best_layer_second
                        )
                    mark = "✓" if match else "✗"
                    print(
                        f"  {auto_name}: {row.strategy_first}[{row_layer_first}]"
                        f"→{row.strategy_second}[{row_layer_second}]"
                        f" ratio={row['ratio_tp_to_tpa']:.3f}"
                        f" seek={row['seek_mode_b_avg_us']:.2f} μs {mark}"
                    )
        print()


if __name__ == "__main__":
    main()
