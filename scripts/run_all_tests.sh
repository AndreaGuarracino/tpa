#!/bin/bash
set -e
#
# Batch test runner for TPA compression across multiple PAF files
# Aggregates results into a master TSV and final report
#
# REQUIREMENTS:
#   - cigzip binary (https://github.com/AndreaGuarracino/cigzip)
#   - Set CIGZIP or CIGZIP_DIR environment variable:
#       CIGZIP=/path/to/cigzip/target/release/cigzip  (direct binary path)
#       CIGZIP_DIR=/path/to/cigzip                    (repo root, builds if needed)
#
# USAGE:
#   ./run_all_tests.sh <num_records> <output_base> <num_threads> \
#       --files FILE [FILE ...] --types TYPE [TYPE ...] [--auto ROWS]
#
# ARGUMENTS:
#   num_records  - Number of records to test per file (0 = all records)
#   output_base  - Output directory for all test results
#   num_threads  - Number of parallel threads
#   --files      - Space-separated list of input PAF files
#   --types      - Space-separated list of tracepoint types (standard, variable, mixed)
#
# OPTIONAL:
#   --auto ROWS  - Only test automatic mode with ROWS sample size (0 = full file)
#
# EXAMPLES:
#   # Test 2 files with standard tracepoints, 100 records each, 4 threads
#   CIGZIP=/path/to/cigzip/target/release/cigzip ./run_all_tests.sh 100 /tmp/results 4 \
#       --files data1.paf.gz data2.paf.gz --types standard
#
#   # Test all tracepoint types on one file
#   CIGZIP=/path/to/cigzip/target/release/cigzip ./run_all_tests.sh 500 ./output 8 \
#       --files input.paf --types standard variable mixed
#
#   # Test automatic mode only (10000-record sampling)
#   CIGZIP_DIR=/path/to/cigzip ./run_all_tests.sh 1000 /tmp/auto 4 \
#       --files data.paf --types standard --auto 10000
#
#   # Test automatic mode with full file analysis
#   CIGZIP_DIR=/path/to/cigzip ./run_all_tests.sh 0 /tmp/full 4 \
#       --files data.paf --types standard --auto 0
#
# OUTPUT:
#   output_base/
#   ├── all_results.tsv      # Master TSV with all test results
#   ├── FINAL_REPORT.md      # Aggregated markdown report
#   └── <dataset_name>/      # Per-file results
#       ├── results.tsv
#       └── test_report.md
#

# Ensure Rust/Cargo tools are available
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
else
    export PATH="/home/node/.cargo/bin:$PATH"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPREHENSIVE_TEST="$SCRIPT_DIR/comprehensive_test.sh"

usage() {
    echo "Usage: $0 <num_records> <output_base> <num_threads> --files FILE [FILE ...] --types TYPE [TYPE ...] [--auto ROWS]"
    echo ""
    echo "Arguments (all required):"
    echo "  num_records  - Number of records to test per file (0 = all records)"
    echo "  output_base  - Output directory for all test results"
    echo "  num_threads  - Number of parallel threads"
    echo "  --files      - Space-separated list of input PAF files"
    echo "  --types      - Space-separated list of tracepoint types"
    echo ""
    echo "Optional flags:"
    echo "  --auto ROWS  - Only test automatic mode with ROWS sample size"
    echo "                 (default: 10000, use 0 for full file analysis)"
    echo ""
    echo "Tracepoint types:"
    echo "  standard  - Standard tracepoints (pairs of values)"
    echo "  variable  - Variable tracepoints (optional second value)"
    echo "  mixed     - Mixed tracepoints (interleaved tracepoints and CIGAR ops)"
    echo ""
    echo "Examples:"
    echo "  $0 100 /tmp/results 4 --files data1.paf data2.paf.gz --types standard"
    echo "  $0 500 ./output 8 --files input.paf --types standard variable mixed"
    echo "  $0 100 /tmp/auto 4 --files data.paf --types standard --auto 10000"
    echo "  $0 100 /tmp/full 4 --files data.paf --types standard --auto 0"
    exit 1
}

# Check minimum arguments
if [ $# -lt 6 ]; then
    echo "Error: Missing required arguments"
    echo ""
    usage
fi

# Parse positional arguments
NUM_RECORDS="$1"
OUTPUT_BASE="$2"
THREADS="$3"
shift 3

# Validate numeric arguments
if ! [[ "$NUM_RECORDS" =~ ^[0-9]+$ ]]; then
    echo "Error: num_records must be a non-negative integer"
    usage
fi

if ! [[ "$THREADS" =~ ^[0-9]+$ ]] || [ "$THREADS" -lt 1 ]; then
    echo "Error: num_threads must be a positive integer"
    usage
fi

# Parse --files, --types, and --auto
TEST_FILES=()
TP_TYPES=()
AUTO_ROWS=""  # Empty means run dual mode (all strategies), set to number for auto-only
PARSING_FILES=false
PARSING_TYPES=false

while [ $# -gt 0 ]; do
    case "$1" in
        --files)
            PARSING_FILES=true
            PARSING_TYPES=false
            shift
            ;;
        --types)
            PARSING_FILES=false
            PARSING_TYPES=true
            shift
            ;;
        --auto)
            PARSING_FILES=false
            PARSING_TYPES=false
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --auto requires a number (sample size, 0 for full file)"
                usage
            fi
            if ! [[ "$1" =~ ^[0-9]+$ ]]; then
                echo "Error: --auto requires a non-negative integer"
                usage
            fi
            AUTO_ROWS="$1"
            shift
            ;;
        *)
            if $PARSING_FILES; then
                TEST_FILES+=("$1")
            elif $PARSING_TYPES; then
                # Validate type
                case "$1" in
                    standard|variable|mixed)
                        TP_TYPES+=("$1")
                        ;;
                    *)
                        echo "Error: Invalid tracepoint type '$1'"
                        echo "Valid types: standard, variable, mixed"
                        exit 1
                        ;;
                esac
            else
                echo "Error: Unexpected argument '$1'"
                usage
            fi
            shift
            ;;
    esac
done

# Validate we have files and types
if [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo "Error: No input files specified (use --files)"
    usage
fi

if [ ${#TP_TYPES[@]} -eq 0 ]; then
    echo "Error: No tracepoint types specified (use --types)"
    usage
fi

mkdir -p "$OUTPUT_BASE"

# Initialize master TSV file
MASTER_TSV="$OUTPUT_BASE/all_results.tsv"
cat > "$MASTER_TSV" << TSV_HEADER
dataset_name	dataset_type	original_size_bytes	num_records	encoding_type	encoding_runtime_sec	encoding_memory_mb	tp_file_size_bytes	max_complexity	complexity_metric	compression_strategy	strategy_first	strategy_second	compression_layer_first	compression_layer_second	compression_runtime_sec	compression_memory_mb	tpa_size_bytes	ratio_orig_to_tp	ratio_tp_to_tpa	ratio_orig_to_tpa	decompression_runtime_sec	decompression_memory_mb	verification_passed	seek_positions_tested	seek_iterations_per_position	seek_total_tests	seek_mode_a_avg_us	seek_mode_a_stddev_us	seek_mode_b_avg_us	seek_mode_b_stddev_us	seek_decode_ratio	seek_valid_ratio
TSV_HEADER

echo "###################################################################"
echo "# tpa - Complete Test Suite"
echo "###################################################################"
echo ""
echo "Configuration:"
echo "  Records per file: $NUM_RECORDS"
echo "  Output directory: $OUTPUT_BASE"
echo "  Parallel threads: $THREADS"
echo "  Tracepoint types: ${TP_TYPES[*]}"
if [ -n "$AUTO_ROWS" ]; then
    if [ "$AUTO_ROWS" -eq 0 ]; then
        echo "  Automatic mode:   full file analysis"
    else
        echo "  Automatic mode:   ${AUTO_ROWS}-record sampling"
    fi
else
    echo "  Automatic mode:   disabled (testing all strategies)"
fi
echo "  Master TSV:       $MASTER_TSV"
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

# Convert types array to comma-separated string for comprehensive_test.sh
TP_TYPES_STR=$(IFS=,; echo "${TP_TYPES[*]}")

for i in "${!VALID_FILES[@]}"; do
    PAF="${VALID_FILES[$i]}"
    NAME="${FILE_NAMES[$i]}"
    OUT_DIR="$OUTPUT_BASE/$NAME"

    echo "###################################################################"
    echo "# Test $((i+1))/${#VALID_FILES[@]}: $NAME"
    echo "###################################################################"
    echo ""

    # Pass test mode to comprehensive_test.sh
    if [ -n "$AUTO_ROWS" ]; then
        TEST_MODE="auto:${AUTO_ROWS}"
    else
        TEST_MODE="dual"
    fi
    $COMPREHENSIVE_TEST "$PAF" "$OUT_DIR" 32 edit-distance "$NUM_RECORDS" "$TEST_MODE" "$THREADS" "$TP_TYPES_STR"

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

AUTO_DESC="disabled"
if [ -n "$AUTO_ROWS" ]; then
    if [ "$AUTO_ROWS" -eq 0 ]; then
        AUTO_DESC="full file analysis"
    else
        AUTO_DESC="${AUTO_ROWS}-record sampling"
    fi
fi

cat > "$FINAL_REPORT" << HEADER
# tpa Complete Test Suite - Aggregated Results

**Test Date:** $(date +%Y-%m-%d)
**Total Files Tested:** ${#VALID_FILES[@]}
**Records Per File:** $NUM_RECORDS
**Tracepoint Types:** ${TP_TYPES[*]}
**Automatic Mode:** $AUTO_DESC
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
- **Tracepoint Types Tested:** ${TP_TYPES[*]}
- **Automatic Mode:** $AUTO_DESC
- **Compression Strategies:** raw, zigzag-delta, 2d-delta, rle, bit-packed, delta-of-delta, frame-of-reference, hybrid-rle, offset-joint, xor-delta, dictionary, stream-vbyte, fastpfor, cascaded, simple8b-full, selective-rle, rice, huffman
- **Seek Modes:** Mode A (TpaReader), Mode B (standalone functions)

## What Was Tested

For each file and configuration:

1. ✓ Compression time and memory
2. ✓ Decompression time and memory
3. ✓ Compressed file size and ratio
4. ✓ Seek performance (Mode A & B)
5. ✓ Round-trip verification (input == decompress(compress(input)))

## Verification Method

All tests use 2-decimal float rounding for robust f32 comparison:
- Floats rounded to 2 decimals: \`0.151999995112\` → \`0.15\`
- Handles f32 precision loss correctly

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
