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
#       --files <file_list.txt> --types TYPE [TYPE ...] [--metrics METRIC ...] [--auto ROWS]
#
# ARGUMENTS:
#   num_records  - Number of records to test per file (0 = all records)
#   output_base  - Output directory for all test results
#   num_threads  - Number of parallel threads
#   --files      - Path to a text file containing one PAF file path per line
#   --types      - Space-separated list of tracepoint types (standard, variable, mixed, fastga)
#
# OPTIONAL:
#   --metrics    - Space-separated list of complexity metrics (default: edit-distance)
#                  Note: Ignored for fastga tracepoint type
#   --auto ROWS  - Only test automatic mode with ROWS sample size (0 = full file)
#
# EXAMPLES:
#   # Create a file list
#   echo "/path/to/data1.paf.gz" > files.txt
#   echo "/path/to/data2.paf.gz" >> files.txt
#
#   # Test files from list with standard tracepoints, 100 records each, 4 threads
#   CIGZIP=/path/to/cigzip/target/release/cigzip ./run_all_tests.sh 100 /tmp/results 4 \
#       --files files.txt --types standard
#
#   # Test all tracepoint types using glob pattern to create file list
#   ls /data/*.paf.gz > my_files.txt
#   CIGZIP=/path/to/cigzip/target/release/cigzip ./run_all_tests.sh 500 ./output 8 \
#       --files my_files.txt --types standard variable mixed
#
#   # Test with multiple complexity metrics
#   CIGZIP=/path/to/cigzip/target/release/cigzip ./run_all_tests.sh 100 /tmp/results 4 \
#       --files files.txt --types standard --metrics edit-distance diagonal-distance
#
#   # Test automatic mode only (10000-record sampling)
#   CIGZIP_DIR=/path/to/cigzip ./run_all_tests.sh 1000 /tmp/auto 4 \
#       --files files.txt --types standard --auto 10000
#
#   # Test automatic mode with full file analysis
#   CIGZIP_DIR=/path/to/cigzip ./run_all_tests.sh 0 /tmp/full 4 \
#       --files files.txt --types standard --auto 0
#
# OUTPUT:
#   output_base/
#   ├── all_results.tsv      # Master TSV with all test results
#   └── <dataset_name>/      # Per-file results
#       └── results.tsv
#

# Ensure Rust/Cargo tools are available
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
else
    export PATH="/home/node/.cargo/bin:$PATH"
fi

# Support both CIGZIP (direct binary path) and CIGZIP_DIR (repo root)
if [ -n "${CIGZIP:-}" ] && [ -f "$CIGZIP" ]; then
    # CIGZIP points directly to the binary
    CIGZIP_DIR=""
elif [ -n "${CIGZIP_DIR:-}" ]; then
    # CIGZIP_DIR points to repo root
    CIGZIP="$CIGZIP_DIR/target/release/cigzip"
else
    CIGZIP=""
    CIGZIP_DIR=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPREHENSIVE_TEST="$SCRIPT_DIR/comprehensive_test.sh"

usage() {
    echo "Usage: $0 <num_records> <output_base> <num_threads> --files <file_list.txt> --types TYPE [TYPE ...] [--metrics METRIC ...] [--auto ROWS]"
    echo ""
    echo "Arguments (all required):"
    echo "  num_records  - Number of records to test per file (0 = all records)"
    echo "  output_base  - Output directory for all test results"
    echo "  num_threads  - Number of parallel threads"
    echo "  --files      - Path to a text file containing one PAF file path per line"
    echo "  --types      - Space-separated list of tracepoint types"
    echo ""
    echo "Optional flags:"
    echo "  --metrics    - Space-separated: edit-distance (default), diagonal-distance"
    echo "  --auto ROWS  - Only test automatic mode with ROWS sample size (0=full file)"
    echo ""
    echo "Tracepoint types:"
    echo "  standard  - Standard tracepoints (pairs of values)"
    echo "  variable  - Variable tracepoints (optional second value)"
    echo "  mixed     - Mixed tracepoints (interleaved tracepoints and CIGAR ops)"
    echo "  fastga    - FastGA tracepoints (uses trace_spacing=100, ignores metrics)"
    echo ""
    echo "Examples:"
    echo "  # Create file list"
    echo "  ls /data/*.paf.gz > files.txt"
    echo ""
    echo "  $0 100 /tmp/results 4 --files files.txt --types standard"
    echo "  $0 500 ./output 8 --files files.txt --types standard variable mixed"
    echo "  $0 100 /tmp/results 4 --files files.txt --types standard --metrics edit-distance diagonal-distance"
    echo "  $0 100 /tmp/auto 4 --files files.txt --types standard --auto 10000"
    echo "  $0 100 /tmp/full 4 --files files.txt --types standard --auto 0"
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

# Parse --files, --types, --metrics, and --auto
TEST_FILES=()
TP_TYPES=()
COMPLEXITY_METRICS=()
AUTO_ROWS=""  # Empty means run dual mode (all strategies), set to number for auto-only
FILE_LIST=""
PARSING_MODE=""  # "types" or "metrics"

while [ $# -gt 0 ]; do
    case "$1" in
        --files)
            PARSING_MODE=""
            shift
            if [ $# -eq 0 ]; then
                echo "Error: --files requires a file path"
                usage
            fi
            FILE_LIST="$1"
            shift
            ;;
        --types)
            PARSING_MODE="types"
            shift
            ;;
        --metrics)
            PARSING_MODE="metrics"
            shift
            ;;
        --auto)
            PARSING_MODE=""
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
            if [ "$PARSING_MODE" = "types" ]; then
                # Validate type
                case "$1" in
                    standard|variable|mixed|fastga)
                        TP_TYPES+=("$1")
                        ;;
                    *)
                        echo "Error: Invalid tracepoint type '$1'"
                        echo "Valid types: standard, variable, mixed, fastga"
                        exit 1
                        ;;
                esac
            elif [ "$PARSING_MODE" = "metrics" ]; then
                # Validate metric
                case "$1" in
                    edit-distance|diagonal-distance)
                        COMPLEXITY_METRICS+=("$1")
                        ;;
                    *)
                        echo "Error: Invalid complexity metric '$1'"
                        echo "Valid metrics: edit-distance, diagonal-distance"
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

# Default to edit-distance if no metrics specified
if [ ${#COMPLEXITY_METRICS[@]} -eq 0 ]; then
    COMPLEXITY_METRICS=("edit-distance")
fi

# Validate file list
if [ -z "$FILE_LIST" ]; then
    echo "Error: No file list specified (use --files <file_list.txt>)"
    usage
fi

if [ ! -f "$FILE_LIST" ]; then
    echo "Error: File list not found: $FILE_LIST"
    exit 1
fi

# Read file paths from the list file (one per line, skip empty lines and comments)
while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and lines starting with #
    line="${line%%#*}"  # Remove comments
    line="${line%"${line##*[![:space:]]}"}"  # Trim trailing whitespace
    line="${line#"${line%%[![:space:]]*}"}"  # Trim leading whitespace
    if [ -n "$line" ]; then
        TEST_FILES+=("$line")
    fi
done < "$FILE_LIST"

# Validate we have files and types
if [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo "Error: No files found in $FILE_LIST"
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
dataset_name	dataset_type	original_size_bytes	num_records	encoding_type	encoding_runtime_sec	encoding_memory_mb	tp_file_size_bytes	max_complexity	complexity_metric	compression_strategy	strategy_first	strategy_second	compression_layer_first	compression_layer_second	compression_runtime_sec	compression_memory_mb	tpa_size_bytes	ratio_orig_to_tp	ratio_tp_to_tpa	ratio_orig_to_tpa	decompression_runtime_sec	decompression_memory_mb	verification_passed	positions_tested	iterations_per_position	full_open_avg_us	full_seek_avg_us	direct_open_avg_us	direct_seek_avg_us	seek_decode_ratio	seek_valid_ratio
TSV_HEADER

echo "###################################################################"
echo "# tpa - Complete Test Suite"
echo "###################################################################"
echo ""
echo "Configuration:"
echo "  Records per file: $NUM_RECORDS"
echo "  Output directory: $OUTPUT_BASE"
echo "  Parallel threads: $THREADS"
echo "  File list:        $FILE_LIST (${#TEST_FILES[@]} files)"
echo "  Tracepoint types: ${TP_TYPES[*]}"
echo "  Complexity metrics: ${COMPLEXITY_METRICS[*]}"
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

# Build tools once before running tests
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Check for bgzip (required for baseline comparison)
if ! command -v bgzip &> /dev/null; then
    echo "ERROR: bgzip not found in PATH"
    exit 1
fi

# Build cigzip if needed
if [ -z "$CIGZIP" ] || [ ! -f "$CIGZIP" ]; then
    if [ -z "$CIGZIP_DIR" ]; then
        echo "ERROR: cigzip not found."
        echo "Set one of these environment variables:"
        echo "  CIGZIP=/path/to/cigzip/target/release/cigzip   # direct binary path"
        echo "  CIGZIP_DIR=/path/to/cigzip                     # repo root (will build if needed)"
        exit 1
    fi
    if [ ! -d "$CIGZIP_DIR" ]; then
        echo "ERROR: CIGZIP_DIR=$CIGZIP_DIR does not exist or is not a directory."
        exit 1
    fi
    echo "=== Building cigzip ==="
    (cd "$CIGZIP_DIR" && cargo build --release 2>&1 | tail -3)
    CIGZIP="$CIGZIP_DIR/target/release/cigzip"
fi
export CIGZIP

# Build tpa
echo "=== Building tpa (library, binaries, examples) ==="
(cd "$REPO_DIR" && cargo build --release --examples 2>&1 | tail -3)

# Verify builds succeeded
for tool in tpa-view tpa-analyze; do
    if [ ! -f "$REPO_DIR/target/release/$tool" ]; then
        echo "Error: Failed to build $tool"
        exit 1
    fi
done
for example in seek_bench_reader seek_bench_direct seek_bench_bgzip_paf; do
    if [ ! -f "$REPO_DIR/target/release/examples/$example" ]; then
        echo "Error: Failed to build $example"
        exit 1
    fi
done

echo ""

# Run tests on each file
START_TIME=$(date +%s)

# Convert arrays to comma-separated strings for comprehensive_test.sh
TP_TYPES_STR=$(IFS=,; echo "${TP_TYPES[*]}")
METRICS_STR=$(IFS=,; echo "${COMPLEXITY_METRICS[*]}")

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
    $COMPREHENSIVE_TEST "$PAF" "$OUT_DIR" 32 "$METRICS_STR" "$NUM_RECORDS" "$TEST_MODE" "$THREADS" "$TP_TYPES_STR"

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

echo "###################################################################"
echo "# All Tests Complete!"
echo "###################################################################"
echo ""
echo "Tested ${#VALID_FILES[@]} files in ${TOTAL_TIME} seconds"
echo ""
echo "Master TSV: $MASTER_TSV"
echo ""
echo "Individual Results:"
for NAME in "${FILE_NAMES[@]}"; do
    echo "  - $OUTPUT_BASE/$NAME/results.tsv"
done
echo ""

exit 0

# Generate plots
echo "###################################################################"
echo "# Generating Visualization Plots"
echo "###################################################################"
echo ""

if command -v python3 &> /dev/null; then
    python3 "$SCRIPT_DIR/plot_results.py" "$MASTER_TSV" && echo "✓ Plots generated successfully"
else
    echo "Python3 not found - skipping plot generation"
    echo "  (Install python3, pandas, and matplotlib to enable plots)"
fi
echo ""
