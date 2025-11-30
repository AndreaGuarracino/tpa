#!/usr/bin/env python3
"""
Plot tpa compression test results from all_results.tsv

Usage:
    python3 plot_results.py [path_to_all_results.tsv]

Defaults to: ../test/tpa_all_tests/all_results.tsv (relative to this script)
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

UNKNOWN = "unknown"

def normalize_strategy_pair(row):
    """Return (first, second) using fallbacks when values are missing/unknown."""
    first = str(row.get('strategy_first', '')).strip()
    second = str(row.get('strategy_second', '')).strip()
    combined = str(row.get('compression_strategy', '')).strip()

    def is_unknown(val: str) -> bool:
        return val == "" or val.lower() == UNKNOWN

    if not is_unknown(first) and not is_unknown(second):
        return first, second

    # Fallback: derive from combined strategy string
    if "→" in combined:
        parts = combined.split("→", 1)
        return parts[0], parts[1]
    if ":" in combined:
        parts = combined.split(":", 1)
        return parts[0], parts[1]
    if combined:
        return combined, combined
    return first or UNKNOWN, second or UNKNOWN

def format_bytes(bytes_val):
    """Convert bytes to human-readable format"""
    if bytes_val is None or (isinstance(bytes_val, float) and np.isnan(bytes_val)):
        return "unknown"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"

def plot_dataset_metrics(df, dataset_name, output_dir):
    """Create comprehensive plots for a single dataset"""

    # Filter data for this dataset
    data = df[df['dataset_name'] == dataset_name].copy()

    if len(data) == 0:
        print(f"Warning: No data found for dataset '{dataset_name}'")
        return

    # Sort by compression ratio (best first)
    data = data.sort_values('ratio_orig_to_tpa', ascending=False)

    # Normalize strategy labels (avoid "unknown" when second wasn't parsed)
    norm_pairs = data.apply(normalize_strategy_pair, axis=1, result_type='expand')
    data = data.assign(strategy_first_norm=norm_pairs[0], strategy_second_norm=norm_pairs[1])

    strategies = [f"{f}→{s}" for f, s in zip(data['strategy_first_norm'], data['strategy_second_norm'])]

    # Create figure with 6 subplots (vertical stack)
    fig, axes = plt.subplots(6, 1, figsize=(84, 48))
    fig.suptitle(f'Compression Strategy Performance: {dataset_name}', fontsize=16, fontweight='bold', y=0.997)

    # Original file info
    orig_size = data['original_size_bytes'].iloc[0]
    num_records = data['num_records'].iloc[0]
    dataset_type = data['dataset_type'].iloc[0]

    fig.text(0.5, 0.975, f'Type: {dataset_type} | Records: {num_records:,} | Original Size: {format_bytes(orig_size)}',
             ha='center', fontsize=11, style='italic')

    # Color code by compression layer (based on requested strategy)
    colors = []
    for _, row in data.iterrows():
        strat = row['compression_strategy']
        if strat.endswith('-bgzip'):
            colors.append('#ff7f0e')  # orange for bgzip
        elif strat.endswith('-nocomp'):
            colors.append('#2ca02c')  # green for nocomp
        else:
            colors.append('#1f77b4')  # blue for zstd (default)

    def compact_axis(ax, count: int):
        """Remove extra horizontal padding around bars."""
        ax.set_xlim(-0.5, max(count - 0.5, 0.5))
        ax.margins(x=0)

    # Plot 1: TPA File Size
    ax1 = axes[0]
    bars1 = ax1.bar(range(len(strategies)), data['tpa_size_bytes'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Compression Strategy', fontweight='bold')
    ax1.set_ylabel('TPA File Size (bytes)', fontweight='bold')
    ax1.set_title('Final Compressed File Size', fontweight='bold')
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategies, rotation=90, ha='right', fontsize=8)
    ax1.grid(axis='y', alpha=0.3)
    ax1.ticklabel_format(axis='y', style='plain')
    compact_axis(ax1, len(strategies))

    # Annotate best (smallest)
    best_idx = data['tpa_size_bytes'].idxmin()
    best_size = data.loc[best_idx, 'tpa_size_bytes']
    best_first = data.loc[best_idx, 'strategy_first_norm']
    best_second = data.loc[best_idx, 'strategy_second_norm']
    ax1.text(0.98, 0.98, f'Best: {best_first}→{best_second}\n{format_bytes(best_size)}',
             transform=ax1.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)

    # Plot 2: Compression Ratio (Original → TPA)
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(strategies)), data['ratio_orig_to_tpa'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Compression Strategy', fontweight='bold')
    ax2.set_ylabel('Compression Ratio (x)', fontweight='bold')
    ax2.set_title('End-to-End Compression Ratio (Original → TPA)', fontweight='bold')
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels(strategies, rotation=90, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    compact_axis(ax2, len(strategies))

    # Annotate best (highest)
    best_idx = data['ratio_orig_to_tpa'].idxmax()
    best_ratio = data.loc[best_idx, 'ratio_orig_to_tpa']
    best_first = data.loc[best_idx, 'strategy_first_norm']
    best_second = data.loc[best_idx, 'strategy_second_norm']
    ax2.text(0.98, 0.98, f'Best: {best_first}→{best_second}\n{best_ratio:.2f}x',
             transform=ax2.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)

    # Plot 3: Seek Time (Mode B)
    ax3 = axes[2]
    bars3 = ax3.bar(range(len(strategies)), data['seek_mode_b_avg_us'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Compression Strategy', fontweight='bold')
    ax3.set_ylabel('Average Seek Time (μs)', fontweight='bold')
    ax3.set_title('Seek Performance (Mode B - Standalone Functions)', fontweight='bold')
    ax3.set_xticks(range(len(strategies)))
    ax3.set_xticklabels(strategies, rotation=90, ha='right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    compact_axis(ax3, len(strategies))

    # Annotate best (lowest)
    best_idx = data['seek_mode_b_avg_us'].idxmin()
    best_seek = data.loc[best_idx, 'seek_mode_b_avg_us']
    best_first = data.loc[best_idx, 'strategy_first_norm']
    best_second = data.loc[best_idx, 'strategy_second_norm']
    ax3.text(0.98, 0.98, f'Fastest: {best_first}→{best_second}\n{best_seek:.2f} μs',
             transform=ax3.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=9)

    # Plot 4: Compression Time
    ax4 = axes[3]
    bars4 = ax4.bar(range(len(strategies)), data['compression_runtime_sec'], color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Compression Strategy', fontweight='bold')
    ax4.set_ylabel('Compression Time (seconds)', fontweight='bold')
    ax4.set_title('Compression Runtime', fontweight='bold')
    ax4.set_xticks(range(len(strategies)))
    ax4.set_xticklabels(strategies, rotation=90, ha='right', fontsize=8)
    ax4.grid(axis='y', alpha=0.3)
    compact_axis(ax4, len(strategies))

    # Annotate fastest
    best_idx = data['compression_runtime_sec'].idxmin()
    best_time = data.loc[best_idx, 'compression_runtime_sec']
    best_first = data.loc[best_idx, 'strategy_first_norm']
    best_second = data.loc[best_idx, 'strategy_second_norm']
    ax4.text(0.98, 0.98, f'Fastest: {best_first}→{best_second}\n{best_time:.2f} s',
             transform=ax4.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=9)

    # Plot 5: Verification (checksum) pass/fail
    ax5 = axes[4]
    verification = data['verification_passed'].apply(lambda x: 1.0 if str(x).lower() == 'yes' else 0.0)
    ver_colors = ['#2ca02c' if v >= 0.999 else '#d62728' for v in verification]  # green pass, red fail
    ax5.bar(range(len(strategies)), verification, color=ver_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Compression Strategy', fontweight='bold')
    ax5.set_ylabel('Verification', fontweight='bold')
    ax5.set_title('Checksum Validation (1 = pass, 0 = fail)', fontweight='bold')
    ax5.set_xticks(range(len(strategies)))
    ax5.set_xticklabels(strategies, rotation=90, ha='right', fontsize=8)
    ax5.set_ylim(0, 1.1)
    ax5.grid(axis='y', alpha=0.3)
    compact_axis(ax5, len(strategies))
    if len(verification) > 0:
        avg_ver = verification.mean() * 100.0
        ax5.text(0.98, 0.98, f'Pass rate: {avg_ver:.1f}%', transform=ax5.transAxes,
                 ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), fontsize=9)

    # Plot 6: Tracepoint validation ratio (seek valid_ratio)
    ax6 = axes[5]
    seek_ratio = data['seek_valid_ratio']
    seek_colors = ['#2ca02c' if v >= 0.999 else '#d62728' for v in seek_ratio]  # green pass, red fail
    ax6.bar(range(len(strategies)), seek_ratio, color=seek_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax6.set_xlabel('Compression Strategy', fontweight='bold')
    ax6.set_ylabel('Tracepoint Match Ratio', fontweight='bold')
    ax6.set_title('Tracepoint Validation (Decoded vs Reference)', fontweight='bold')
    ax6.set_xticks(range(len(strategies)))
    ax6.set_xticklabels(strategies, rotation=90, ha='right', fontsize=8)
    ax6.set_ylim(0, 1.05)
    ax6.grid(axis='y', alpha=0.3)
    compact_axis(ax6, len(strategies))
    best_idx = data['seek_valid_ratio'].idxmax()
    best_val = data.loc[best_idx, 'seek_valid_ratio']
    best_first = data.loc[best_idx, 'strategy_first_norm']
    best_second = data.loc[best_idx, 'strategy_second_norm']
    ax6.text(0.98, 0.98, f'Best: {best_first}→{best_second}\n{best_val:.3f}',
             transform=ax6.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)

    # Add legend for compression layers
    zstd_patch = mpatches.Patch(color='#1f77b4', alpha=0.7, label='Zstd (default)')
    bgzip_patch = mpatches.Patch(color='#ff7f0e', alpha=0.7, label='BGZIP')
    nocomp_patch = mpatches.Patch(color='#2ca02c', alpha=0.7, label='No compression')
    fig.legend(handles=[zstd_patch, bgzip_patch, nocomp_patch],
               loc='lower center', ncol=3, frameon=True, fontsize=10,
               bbox_to_anchor=(0.5, 0.0))

    plt.subplots_adjust(left=0.03, right=0.99, wspace=0.15, hspace=0.35)
    plt.tight_layout(rect=[0.005, 0.01, 0.995, 0.985])

    # Save plot
    output_file = output_dir / f'{dataset_name}_metrics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}")
    plt.close()


def generate_summary_table(df, output_dir):
    """Generate a summary markdown table showing top performers per dataset"""

    output_file = output_dir / 'summary_best_strategies.md'

    with open(output_file, 'w') as f:
        f.write("# Best Compression Strategies per Dataset\n\n")
        f.write("Summary of top-performing strategies across all metrics.\n\n")

        for dataset in df['dataset_name'].unique():
            data = df[df['dataset_name'] == dataset].copy()

            f.write(f"## {dataset}\n\n")
            f.write(f"- **Type**: {data['dataset_type'].iloc[0]}\n")
            f.write(f"- **Records**: {data['num_records'].iloc[0]:,}\n")
            f.write(f"- **Original Size**: {format_bytes(data['original_size_bytes'].iloc[0])}\n\n")

            # Best compression ratio
            best_ratio_idx = data['ratio_orig_to_tpa'].idxmax()
            f.write(f"### Best Compression Ratio\n")
            first = data.loc[best_ratio_idx, 'strategy_first']
            second = data.loc[best_ratio_idx, 'strategy_second']
            f.write(f"- **Strategy**: {first}→{second}\n")
            f.write(f"- **Ratio**: {data.loc[best_ratio_idx, 'ratio_orig_to_tpa']:.2f}x\n")
            f.write(f"- **TPA Size**: {format_bytes(data.loc[best_ratio_idx, 'tpa_size_bytes'])}\n\n")

            # Fastest seek
            best_seek_idx = data['seek_mode_b_avg_us'].idxmin()
            f.write(f"### Fastest Seek (Mode B)\n")
            first = data.loc[best_seek_idx, 'strategy_first']
            second = data.loc[best_seek_idx, 'strategy_second']
            f.write(f"- **Strategy**: {first}→{second}\n")
            f.write(f"- **Time**: {data.loc[best_seek_idx, 'seek_mode_b_avg_us']:.2f} μs\n")
            f.write(f"- **Ratio**: {data.loc[best_seek_idx, 'ratio_orig_to_tpa']:.2f}x\n\n")

            # Fastest compression
            best_comp_idx = data['compression_runtime_sec'].idxmin()
            f.write(f"### Fastest Compression\n")
            first = data.loc[best_comp_idx, 'strategy_first']
            second = data.loc[best_comp_idx, 'strategy_second']
            f.write(f"- **Strategy**: {first}→{second}\n")
            f.write(f"- **Time**: {data.loc[best_comp_idx, 'compression_runtime_sec']:.2f} s\n")
            f.write(f"- **Ratio**: {data.loc[best_comp_idx, 'ratio_orig_to_tpa']:.2f}x\n\n")

            # Best overall (balance of ratio and seek time)
            data['score'] = data['ratio_orig_to_tpa'] / (data['seek_mode_b_avg_us'] / 10.0)
            best_overall_idx = data['score'].idxmax()
            f.write(f"### Best Overall Balance (Ratio/Seek)\n")
            first = data.loc[best_overall_idx, 'strategy_first']
            second = data.loc[best_overall_idx, 'strategy_second']
            f.write(f"- **Strategy**: {first}→{second}\n")
            f.write(f"- **Ratio**: {data.loc[best_overall_idx, 'ratio_orig_to_tpa']:.2f}x\n")
            f.write(f"- **Seek**: {data.loc[best_overall_idx, 'seek_mode_b_avg_us']:.2f} μs\n\n")

            f.write("---\n\n")

    print(f"  ✓ Saved: {output_file}")


def main():
    # Determine input file
    if len(sys.argv) > 1:
        tsv_file = Path(sys.argv[1])
    else:
        # Default path relative to script location
        script_dir = Path(__file__).parent
        tsv_file = script_dir / 'tpa_all_tests' / 'all_results.tsv'

    if not tsv_file.exists():
        print(f"Error: TSV file not found: {tsv_file}")
        print(f"\nUsage: {sys.argv[0]} [path_to_all_results.tsv]")
        sys.exit(1)

    print(f"Reading test results from: {tsv_file}")

    # Read TSV (handle legacy exports that include dataset name in the index)
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
        'seek_total_tests', 'seek_mode_a_avg_us', 'seek_mode_a_stddev_us',
        'seek_mode_b_avg_us', 'seek_mode_b_stddev_us', 'seek_decode_ratio', 'seek_valid_ratio'
    ]

    df = pd.read_csv(tsv_file, sep='\t', index_col=False, engine='python')
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    if 'layer_first' not in df.columns:
        layer_candidates = {'zstd', 'bgzip', 'nocomp'}
        if 'compression_runtime_sec' in df.columns and df['compression_runtime_sec'].astype(str).str.lower().isin(layer_candidates).all():
            df = pd.read_csv(tsv_file, sep='\t', names=expected_cols, header=0, skiprows=1, engine='python')
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()

    numeric_cols = [
        'original_size_bytes', 'num_records', 'encoding_runtime_sec', 'encoding_memory_mb',
        'tp_file_size_bytes', 'max_complexity', 'compression_runtime_sec',
        'compression_memory_mb', 'tpa_size_bytes', 'ratio_orig_to_tp', 'ratio_tp_to_tpa',
        'ratio_orig_to_tpa', 'decompression_runtime_sec', 'decompression_memory_mb',
        'seek_positions_tested', 'seek_iterations_per_position', 'seek_total_tests',
        'seek_mode_a_avg_us', 'seek_mode_a_stddev_us', 'seek_mode_b_avg_us',
        'seek_mode_b_stddev_us', 'seek_decode_ratio', 'seek_valid_ratio'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'dataset_name' in df.columns:
        df['dataset_name'] = df['dataset_name'].astype(str)
    if 'dataset_type' in df.columns:
        df['dataset_type'] = df['dataset_type'].astype(str)

    print(f"Loaded {len(df)} test results")
    print(f"Found {df['dataset_name'].nunique()} unique datasets")
    print()

    # Output directory (same as TSV)
    output_dir = tsv_file.parent

    # Generate plots for each dataset
    print("Generating plots...")
    for dataset in sorted(df['dataset_name'].unique()):
        print(f"  Processing: {dataset}")
        plot_dataset_metrics(df, dataset, output_dir)

    print()
    print("Generating summary table...")
    generate_summary_table(df, output_dir)

    print()
    print("=" * 60)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
