#!/bin/bash
set -euo pipefail

# Debug failing strategies by comparing normalized original vs. decompressed PAFs.
# Usage: ./debug_failing_strategies.sh <output_dir_from_comprehensive_test>

OUTPUT_DIR="${1:-}"
if [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <output_dir_from_comprehensive_test>"
    exit 1
fi

LOG_FILE="$OUTPUT_DIR/failing_strategies.log"
if [ ! -f "$LOG_FILE" ]; then
    echo "No failing_strategies.log found in $OUTPUT_DIR"
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NORMALIZE="python3 \"$SCRIPT_DIR/normalize_paf.py\""

CIGZIP_DIR="${CIGZIP_DIR:-/home/guarracino/git/cigzip}"
CIGZIP="${CIGZIP_DIR}/target/release/cigzip"

if [ ! -x "$CIGZIP" ]; then
    echo "cigzip binary not found at $CIGZIP (set CIGZIP_DIR)"
    exit 1
fi

# Cache normalized originals per tracepoint type
declare -A ORIG_NORM_CACHE

normalize_orig() {
    local tp_type="$1"
    local orig_paf="$OUTPUT_DIR/${tp_type}.tp.paf"
    if [ ! -f "$orig_paf" ]; then
        echo "Missing original PAF: $orig_paf" >&2
        return 1
    fi
    if [ -n "${ORIG_NORM_CACHE[$tp_type]:-}" ]; then
        echo "${ORIG_NORM_CACHE[$tp_type]}"
        return 0
    fi
    local tmp
    tmp=$(mktemp)
    eval $NORMALIZE "< \"$orig_paf\" > \"$tmp\""
    ORIG_NORM_CACHE[$tp_type]="$tmp"
    echo "$tmp"
}

# Compare normalized files and print first difference
compare_norm() {
    local norm_a="$1"
    local norm_b="$2"
    python3 - "$norm_a" "$norm_b" <<'PY'
import sys
from itertools import zip_longest

norm_a, norm_b = sys.argv[1], sys.argv[2]
a_lines = open(norm_a).read().splitlines()
b_lines = open(norm_b).read().splitlines()

for idx, (a, b) in enumerate(zip_longest(a_lines, b_lines, fillvalue=None), 1):
    if a != b:
        print(f"diff line {idx}")
        print(f"orig: {a}")
        print(f"decomp: {b}")
        break
else:
    print("no differences")
PY
}

process_block() {
    local tp_type="$1" first="$2" second="$3" key="$4" bpaf="$5" decomp="$6"

    if [ -z "$bpaf" ] || [ -z "$decomp" ] || [ -z "$tp_type" ]; then
        echo "Skipping incomplete block (tp_type/bpaf/decomp missing)" >&2
        return
    fi

    # Ensure we have a fresh decompressed file to compare
    local tmp_decomp
    tmp_decomp=$(mktemp)
    "$CIGZIP" decompress -i "$bpaf" -o "$tmp_decomp"

    local norm_orig norm_decomp
    norm_orig=$(normalize_orig "$tp_type") || return
    norm_decomp=$(mktemp)
    eval $NORMALIZE "< \"$tmp_decomp\" > \"$norm_decomp\""

    echo "=== $key ($first -> $second) ==="
    compare_norm "$norm_orig" "$norm_decomp"
    echo ""
}

# Parse failing_strategies.log blocks
tp_type=""; first=""; second=""; key=""; bpaf=""; decomp=""
while IFS= read -r line || [ -n "$line" ]; do
    if [ "$line" = "---" ]; then
        process_block "$tp_type" "$first" "$second" "$key" "$bpaf" "$decomp"
        tp_type=""; first=""; second=""; key=""; bpaf=""; decomp=""
        continue
    fi
    case "$line" in
        tp_type=*) tp_type="${line#tp_type=}";;
        first=*) first="${line#first=}";;
        second=*) second="${line#second=}";;
        key=*) key="${line#key=}";;
        bpaf=*) bpaf="${line#bpaf=}";;
        decomp=*) decomp="${line#decomp=}";;
    esac
done < "$LOG_FILE"

# Process last block if file didn't end with '---'
if [ -n "$tp_type" ] || [ -n "$bpaf" ]; then
    process_block "$tp_type" "$first" "$second" "$key" "$bpaf" "$decomp"
fi
