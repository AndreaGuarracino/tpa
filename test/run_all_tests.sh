#!/bin/bash
set -e

# Ensure Rust/Cargo tools are available
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
else
    export PATH="/home/node/.cargo/bin:$PATH"
fi

# Wrapper script to run comprehensive tests on multiple PAF files
# Aggregates results into a final report
#
# Usage: ./run_all_tests.sh [-t threads] [num_records] [output_base] [paf1] [paf2] ... [pafN]
#
# Options:
#   -t threads   - Number of parallel threads (default: 1)
#
# Arguments:
#   num_records  - Number of records to test per file (default: 50, 0 = all records)
#   output_base  - Output directory for all test results (default: ./test/bpaf_all_tests)
#   paf1..pafN   - Input PAF files to test (default: 3 standard test files)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPREHENSIVE_TEST="$SCRIPT_DIR/comprehensive_test.sh"

# Parse options
THREADS=1
while getopts "t:" opt; do
    case $opt in
        t) THREADS="$OPTARG" ;;
        *) echo "Usage: $0 [-t threads] [num_records] [output_base] [paf1] ... [pafN]"; exit 1 ;;
    esac
done
shift $((OPTIND-1))

# Parse arguments
NUM_RECORDS="${1:-100}"  # 0 means use ALL records
OUTPUT_BASE="${2:-$SCRIPT_DIR/bpaf_all_tests}"

# Default test files
DEFAULT_FILES=(
    "/home/guarracino/git/_resources/hg002v1.1.pat.PanSN-vs-HG02818_mat_hprc_r2_v1.0.1.p95.Pinf.aln.paf.gz"
    "/home/guarracino/git/_resources/hg002v1.1.pat.PanSN-vs-HG02818_mat_hprc_r2_v1.0.1.sweepga.paf.gz"
    "/home/guarracino/git/_resources/big-from-fg.tp.20k.paf"
)

# Collect input files (from arg 3 onwards, or use defaults)
if [ $# -le 2 ]; then
    # No input files specified, use defaults
    TEST_FILES=("${DEFAULT_FILES[@]}")
else
    # Use provided input files
    TEST_FILES=("${@:3}")
fi

mkdir -p "$OUTPUT_BASE"

# Initialize master TSV file
MASTER_TSV="$OUTPUT_BASE/all_results.tsv"
cat > "$MASTER_TSV" << TSV_HEADER
dataset_name	dataset_type	original_size_bytes	num_records	encoding_type	encoding_runtime_sec	encoding_memory_mb	tp_file_size_bytes	max_complexity	complexity_metric	compression_strategy	strategy_first	strategy_second	compression_runtime_sec	compression_memory_mb	bpaf_size_bytes	ratio_orig_to_tp	ratio_tp_to_bpaf	ratio_orig_to_bpaf	decompression_runtime_sec	decompression_memory_mb	verification_passed	seek_positions_tested	seek_iterations_per_position	seek_total_tests	seek_mode_a_avg_us	seek_mode_a_stddev_us	seek_mode_b_avg_us	seek_mode_b_stddev_us	seek_success_ratio
TSV_HEADER

echo "###################################################################"
echo "# lib_bpaf - Complete Test Suite"
echo "###################################################################"
echo ""
echo "Running comprehensive tests on ${#TEST_FILES[@]} files..."
echo "Output directory: $OUTPUT_BASE"
echo "Records per file: $NUM_RECORDS"
echo "Parallel threads: $THREADS"
echo "Master TSV: $MASTER_TSV"
echo ""

# Track which files exist
VALID_FILES=()
FILE_NAMES=()

for PAF in "${TEST_FILES[@]}"; do
    if [ -f "$PAF" ]; then
        VALID_FILES+=("$PAF")
        BASENAME=$(basename "$PAF" .paf.gz)
        BASENAME=$(basename "$BASENAME" .paf)
        FILE_NAMES+=("$BASENAME")
    else
        echo "Warning: File not found, skipping: $PAF"
    fi
done

if [ ${#VALID_FILES[@]} -eq 0 ]; then
    echo "Error: No valid input files found"
    echo ""
    echo "Usage: $0 [-t threads] [num_records] [output_base] [paf1] [paf2] ... [pafN]"
    echo ""
    echo "Options:"
    echo "  -t threads   - Number of parallel threads (default: 1)"
    echo ""
    echo "Arguments:"
    echo "  num_records  - Number of records to test per file (default: 50)"
    echo "  output_base  - Output directory (default: ./test/bpaf_all_tests)"
    echo "  paf1..pafN   - Input PAF files to test"
    echo ""
    echo "Default test files:"
    echo "  1. p95 CIGAR PAF (compressed)"
    echo "  2. sweepga CIGAR PAF (compressed)"
    echo "  3. big-from-fg tracepoint PAF"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use all defaults"
    echo "  $0 -t 6                               # Use defaults with 6 threads"
    echo "  $0 -t 6 100                           # 6 threads, 100 records per file"
    echo "  $0 -t 6 200 /tmp/results              # 6 threads, custom output"
    echo "  $0 -t 6 50 /tmp/out file1.paf file2.paf  # Custom files"
    exit 1
fi

echo "Testing ${#VALID_FILES[@]} files:"
for i in "${!VALID_FILES[@]}"; do
    echo "  $((i+1)). ${FILE_NAMES[$i]}: ${VALID_FILES[$i]}"
done
echo ""
echo ""

# Run tests on each file
START_TIME=$(date +%s)

for i in "${!VALID_FILES[@]}"; do
    PAF="${VALID_FILES[$i]}"
    NAME="${FILE_NAMES[$i]}"
    OUT_DIR="$OUTPUT_BASE/$NAME"
    
    echo "###################################################################"
    echo "# Test $((i+1))/${#VALID_FILES[@]}: $NAME"
    echo "###################################################################"
    echo ""
    
    $COMPREHENSIVE_TEST "$PAF" "$OUT_DIR" 32 edit-distance "$NUM_RECORDS" dual "$THREADS"

    # Append TSV data (skip header)
    if [ -f "$OUT_DIR/results.tsv" ]; then
        tail -n +2 "$OUT_DIR/results.tsv" >> "$MASTER_TSV"
        echo "✓ Appended results to master TSV"
    fi

    echo ""
    echo ""
done

END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

# Aggregate results
FINAL_REPORT="$OUTPUT_BASE/FINAL_REPORT.md"

cat > "$FINAL_REPORT" << HEADER
# lib_bpaf Complete Test Suite - Aggregated Results

**Test Date:** $(date +%Y-%m-%d)
**Total Files Tested:** ${#VALID_FILES[@]}
**Records Per File:** $NUM_RECORDS
**Total Test Time:** ${TOTAL_TIME}s

---

## Files Tested

HEADER

for i in "${!VALID_FILES[@]}"; do
    NAME="${FILE_NAMES[$i]}"
    PAF="${VALID_FILES[$i]}"
    
    # Detect type
    if [[ "$PAF" == *.gz ]]; then
        FIRST_LINE=$(gzip -cdq "$PAF" | head -1)
    else
        FIRST_LINE=$(head -1 "$PAF")
    fi
    
    if echo "$FIRST_LINE" | grep -q "cg:Z:"; then
        TYPE="CIGAR PAF"
    else
        TYPE="Tracepoint PAF"
    fi
    
    cat >> "$FINAL_REPORT" << FILE_ENTRY
$((i+1)). **$NAME**
   - Path: \`$PAF\`
   - Type: $TYPE
   - Report: [\`$NAME/test_report.md\`](./$NAME/test_report.md)

FILE_ENTRY
done

cat >> "$FINAL_REPORT" << SUMMARY

---

## TSV Data

**Master TSV File:** [\`all_results.tsv\`](./all_results.tsv)

This file contains all test results in tab-separated format with 29 columns:
- Dataset information (name, type, size, records)
- Encoding metrics (type, runtime, memory, output size)
- Compression metrics (strategy, runtime, memory, file sizes, ratios)
- Decompression metrics (runtime, memory, verification)
- Seek performance (100 positions × 100 iterations, avg/stddev/success ratio)

---

## Summary

All individual test reports are available in their respective directories.

### Quick Links

SUMMARY

for NAME in "${FILE_NAMES[@]}"; do
    echo "- [$NAME Results](./$NAME/test_report.md)" >> "$FINAL_REPORT"
done

cat >> "$FINAL_REPORT" << FOOTER

---

## Test Configuration

- **Max Complexity:** 32
- **Complexity Metric:** edit-distance
- **Distance Metric:** gap-affine (penalties: 5,8,2)
- **Tracepoint Types Tested:** standard, variable, mixed (if CIGAR input)
- **Compression Strategies:** raw, zigzag-delta, 2d-delta, rle, bit-packed, delta-of-delta, frame-of-reference, hybrid-rle, offset-joint, xor-delta, dictionary, simple8, stream-vbyte, fastpfor, cascaded, simple8b-full, selective-rle, rice-entropy, huffman-entropy, bgzip
- **Seek Modes:** Mode A (BpafReader), Mode B (standalone functions)

## What Was Tested

For each file and configuration:

1. ✓ Compression time and memory
2. ✓ Decompression time and memory  
3. ✓ Compressed file size and ratio
4. ✓ Seek performance (Mode A & B)
5. ✓ Round-trip verification (input == decompress(compress(input)))

## Verification Method

All tests use improved 3-decimal float normalization:
- Normal decimals: \`0.993724\` → \`0.993\`
- Leading dots: \`.0549\` → \`0.054\`
- Integer floats: \`0\` → \`0.000\`, \`1\` → \`1.000\`

---

**Total Test Time:** ${TOTAL_TIME} seconds
**Report Generated:** $(date)

FOOTER

echo "###################################################################"
echo "# All Tests Complete!"
echo "###################################################################"
echo ""
echo "Tested ${#VALID_FILES[@]} files in ${TOTAL_TIME} seconds"
echo ""
echo "Final Report: $FINAL_REPORT"
echo "Master TSV:   $MASTER_TSV"
echo ""
echo "Individual Reports:"
for NAME in "${FILE_NAMES[@]}"; do
    echo "  - Markdown: $OUTPUT_BASE/$NAME/test_report.md"
    echo "  - TSV:      $OUTPUT_BASE/$NAME/results.tsv"
done
echo ""

# Generate plots
echo "###################################################################"
echo "# Generating Visualization Plots"
echo "###################################################################"
echo ""

if command -v python3 &> /dev/null; then
    python3 "$SCRIPT_DIR/plot_results.py" "$MASTER_TSV" && echo "✓ Plots generated successfully"
else
    echo "⚠ Python3 not found - skipping plot generation"
    echo "  (Install python3, pandas, and matplotlib to enable plots)"
fi
echo ""
