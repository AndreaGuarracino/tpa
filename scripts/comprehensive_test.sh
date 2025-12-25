#!/bin/bash
set -e
#
# Comprehensive test suite for TPA compression
#
# REQUIREMENTS:
#   - cigzip binary (https://github.com/AndreaGuarracino/cigzip)
#   - Set CIGZIP or CIGZIP_DIR environment variable:
#       CIGZIP=/path/to/cigzip/target/release/cigzip  (direct binary path)
#       CIGZIP_DIR=/path/to/cigzip                    (repo root, builds if needed)
#
# USAGE:
#   ./comprehensive_test.sh <input.paf[.gz]> [output_dir] [max_complexity] [metric] [num_records] [test_mode] [threads] [tp_types]
#
# EXAMPLES:
#   # Basic test with defaults (20000 records, single mode)
#   CIGZIP=/path/to/cigzip/target/release/cigzip ./comprehensive_test.sh data.paf.gz
#
#   # Test 1000 records with automatic strategy selection
#   CIGZIP=/path/to/cigzip/target/release/cigzip ./comprehensive_test.sh data.paf.gz /tmp/results 32 edit-distance 1000 auto:0
#
#   # Full dual-strategy matrix (all combinations) with 4 threads
#   CIGZIP_DIR=/path/to/cigzip ./comprehensive_test.sh data.paf.gz /tmp/dual 32 edit-distance 500 dual 4
#
#   # Test multiple tracepoint types
#   CIGZIP=/path/to/cigzip/target/release/cigzip ./comprehensive_test.sh data.paf.gz /tmp/out 32 edit-distance 1000 single 1 standard,variable,mixed
#
# TEST MODES:
#   single    - Test each strategy symmetrically (first==second) [default]
#   dual      - Test all first×second strategy combinations
#   auto:N    - Only test automatic mode with N-record sampling (0=full file)
#

# Ensure Rust/Cargo tools are available
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
else
    export PATH="/home/node/.cargo/bin:$PATH"
fi

# Comprehensive test for a single PAF file
# Automatically detects: CIGAR (compressed/uncompressed) or Tracepoint PAF
# Tests: all tracepoint types + all compression strategies + seek performance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

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
NORMALIZE="python3 $SCRIPT_DIR/normalize_paf.py"

# Safe numeric ratio helper
safe_ratio() {
    local num="$1"
    local den="$2"
    local decimals="${3:-3}"

    python3 - "$num" "$den" "$decimals" <<'PY'
import sys
num, den, dec = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    n = int(num)
    d = int(den)
    if d == 0:
        raise ZeroDivisionError
    print(f"{n / d:.{dec}f}")
except Exception:
    print(f"{0:.{dec}f}")
PY
}

# Helper functions for unit conversion
time_to_seconds() {
    local time_str="$1"
    if [ -z "$time_str" ]; then
        echo "0"
        return
    fi
    # Convert MM:SS.MS format to seconds
    if [[ "$time_str" =~ ^([0-9]+):([0-9]+\.[0-9]+)$ ]]; then
        local minutes="${BASH_REMATCH[1]}"
        local seconds="${BASH_REMATCH[2]}"
        awk "BEGIN {printf \"%.3f\", $minutes * 60 + $seconds}"
    elif [[ "$time_str" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "$time_str"
    else
        echo "0"
    fi
}

kb_to_mb() {
    local kb="$1"
    python3 - "$kb" <<'PY'
import sys
val = sys.argv[1]
try:
    kb = float(val)
    print(f"{kb/1024:.2f}")
except Exception:
    print("0.00")
PY
}

normalize_time_field() {
    local val="$1"
    # Convert common placeholders to numeric zero for TSV
    if [ -z "$val" ] || [ "$val" = "N/A" ]; then
        echo "0"
    else
        echo "$val"
    fi
}

# Portable file size (bytes) using Python to avoid GNU/BSD stat differences
file_size() {
    local path="$1"
    python3 - "$path" <<'PY'
import os, sys
path = sys.argv[1]
try:
    print(os.path.getsize(path))
except OSError:
    print(0)
PY
}

# Parameters
INPUT_PAF="${1}"
OUTPUT_DIR="${2:-/tmp/tpa_test_output}"
MAX_COMPLEXITY="${3:-32}"
COMPLEXITY_METRIC="${4:-edit-distance}"
NUM_RECORDS="${5:-20000}"
TEST_MODE="${6:-single}"  # "single", "dual", or "auto:ROWS" - controls strategy testing mode
THREADS="${7:-1}"  # Number of parallel threads (default: 1)
TP_TYPES_INPUT="${8:-standard}"  # Comma-separated list of tracepoint types

if [ -z "$INPUT_PAF" ] || [ ! -f "$INPUT_PAF" ]; then
    echo "Usage: $0 <input.paf[.gz]> [output_dir] [max_complexity] [complexity_metric] [num_records] [test_mode] [threads] [tp_types]"
    echo ""
    echo "Automatically detects input type:"
    echo "  - CIGAR PAF (compressed or uncompressed)"
    echo "  - Tracepoint PAF"
    echo ""
    echo "Test Modes:"
    echo "  single (default) - Test each strategy symmetrically (first==second)"
    echo "  dual             - Test all strategy combinations (no automatic)"
    echo "  auto:ROWS        - Only test automatic mode with ROWS sample size (0=full file)"
    echo ""
    echo "Threads:"
    echo "  Number of tests to run in parallel (default: 1)"
    echo "  Example: 6 for 6-core CPU"
    echo ""
    echo "Tracepoint Types (comma-separated):"
    echo "  standard  - Standard tracepoints (default)"
    echo "  variable  - Variable tracepoints"
    echo "  mixed     - Mixed tracepoints"
    echo "  Example: standard,variable,mixed"
    echo ""
    echo "Tests:"
    echo "  - Specified tracepoint types"
    echo "  - All compression strategies"
    echo "  - Seek performance (Mode A & B)"
    echo "  - Full verification"
    exit 1
fi

# Parse comma-separated tracepoint types into array
IFS=',' read -ra TP_TYPES_FROM_ARG <<< "$TP_TYPES_INPUT"

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "tpa Comprehensive Test Suite"
echo "========================================="
echo "Input:       $INPUT_PAF"
echo "Output:      $OUTPUT_DIR"
echo "Complexity:  $MAX_COMPLEXITY"
echo "Metric:      $COMPLEXITY_METRIC"
echo "Records:     $NUM_RECORDS"
echo "Test Mode:   $TEST_MODE"
echo "Threads:     $THREADS"
echo "TP Types:    ${TP_TYPES_FROM_ARG[*]}"
echo "========================================="
echo ""

# Detect input type
echo "=== Detecting input type ==="
if [[ "$INPUT_PAF" == *.gz ]]; then
    FIRST_LINE=$(gzip -cdq "$INPUT_PAF" | head -1)
    IS_COMPRESSED=1
else
    FIRST_LINE=$(head -1 "$INPUT_PAF")
    IS_COMPRESSED=0
fi

if echo "$FIRST_LINE" | grep -q "cg:Z:"; then
    INPUT_TYPE="cigar"
    echo "✓ Detected: CIGAR PAF ($([ $IS_COMPRESSED -eq 1 ] && echo 'compressed' || echo 'uncompressed'))"
elif echo "$FIRST_LINE" | grep -q "tp:Z:"; then
    INPUT_TYPE="tracepoint"
    echo "✓ Detected: Tracepoint PAF"
else
    echo "✗ Error: Cannot detect PAF type (no cg:Z: or tp:Z: tags found)"
    exit 1
fi
echo ""

# Extract sample (or all if NUM_RECORDS=0)
if [ "$NUM_RECORDS" -eq 0 ]; then
    echo "=== Using ALL records from input file ==="
    if [ $IS_COMPRESSED -eq 1 ]; then
        gzip -cdq "$INPUT_PAF" > "$OUTPUT_DIR/input_sample.paf"
    else
        cp "$INPUT_PAF" "$OUTPUT_DIR/input_sample.paf"
    fi
else
    echo "=== Extracting $NUM_RECORDS records ==="
    if [ $IS_COMPRESSED -eq 1 ]; then
        gzip -cdq "$INPUT_PAF" | head -n "$NUM_RECORDS" > "$OUTPUT_DIR/input_sample.paf"
    else
        head -n "$NUM_RECORDS" "$INPUT_PAF" > "$OUTPUT_DIR/input_sample.paf"
    fi
fi

EXTRACTED=$(wc -l < "$OUTPUT_DIR/input_sample.paf")
SIZE=$(file_size "$OUTPUT_DIR/input_sample.paf")
echo "Extracted: $EXTRACTED records ($SIZE bytes)"
echo ""

# Build tools if needed
if [ -z "$CIGZIP" ] || [ ! -f "$CIGZIP" ]; then
    if [ -z "$CIGZIP_DIR" ]; then
        echo "ERROR: cigzip not found."
        echo "Set one of these environment variables:"
        echo "  CIGZIP=/path/to/cigzip/target/release/cigzip   # direct binary path"
        echo "  CIGZIP_DIR=/path/to/cigzip                     # repo root (will build if needed)"
        echo ""
        echo "Example:"
        echo "  CIGZIP=/path/to/cigzip/target/release/cigzip $0 $*"
        exit 1
    fi
    if [ ! -d "$CIGZIP_DIR" ]; then
        echo "ERROR: CIGZIP_DIR=$CIGZIP_DIR does not exist or is not a directory."
        exit 1
    fi
    echo "=== Building cigzip ==="
    cd "$CIGZIP_DIR"
    cargo build --release 2>&1 | tail -3
    CIGZIP="$CIGZIP_DIR/target/release/cigzip"
fi

if [ ! -f "$REPO_DIR/target/release/libtpa.rlib" ]; then
    echo "=== Building tpa ==="
    cd "$REPO_DIR"
    cargo build --release 2>&1 | tail -3
fi

# Build seek benchmark examples if needed
SEEK_BENCH_READER="$REPO_DIR/target/release/examples/seek_bench_reader"
SEEK_BENCH_DIRECT="$REPO_DIR/target/release/examples/seek_bench_direct"
SEEK_BENCH_BGZIP_PAF="$REPO_DIR/target/release/examples/seek_bench_bgzip_paf"
OPEN_BENCH="$REPO_DIR/target/release/examples/open_bench"

if [ ! -f "$SEEK_BENCH_READER" ] || [ ! -f "$SEEK_BENCH_DIRECT" ] || [ ! -f "$SEEK_BENCH_BGZIP_PAF" ] || [ ! -f "$OPEN_BENCH" ]; then
    echo "=== Building seek benchmark examples ==="
    cd "$REPO_DIR"
    cargo build --release --examples 2>&1 | tail -3
fi

if [ ! -f "$SEEK_BENCH_READER" ]; then
    echo "✗ Error: Failed to build seek_bench_reader"
    exit 1
fi
if [ ! -f "$SEEK_BENCH_DIRECT" ]; then
    echo "✗ Error: Failed to build seek_bench_direct"
    exit 1
fi
if [ ! -f "$SEEK_BENCH_BGZIP_PAF" ]; then
    echo "✗ Error: Failed to build seek_bench_bgzip_paf"
    exit 1
fi
if [ ! -f "$OPEN_BENCH" ]; then
    echo "✗ Error: Failed to build open_bench"
    exit 1
fi

echo "✓ Benchmark tools ready"
echo ""

# Results storage - declare arrays before use
declare -A ENCODE_TIME ENCODE_MEM
declare -A COMPRESS_TIME COMPRESS_MEM COMPRESS_SIZE
declare -A DECOMPRESS_TIME DECOMPRESS_MEM
declare -A SEEK_A SEEK_B SEEK_A_STDDEV SEEK_B_STDDEV SEEK_DECODE_RATIO SEEK_VALID_RATIO
declare -A VERIFIED
declare -A TP_SIZE
declare -A BGZIP_TIME BGZIP_MEM BGZIP_SIZE
declare -A BGZIP_SEEK_AVG BGZIP_SEEK_STDDEV BGZIP_SEEK_DECODE BGZIP_SEEK_VALID
declare -A BGZIP_OPEN_TIME
declare -A STRATEGY_FIRST STRATEGY_SECOND
declare -A LAYER_FIRST LAYER_SECOND
declare -A FILE_OPEN_TIME

# Use tracepoint types from argument
TP_TYPES=("${TP_TYPES_FROM_ARG[@]}")

# Determine tracepoint types to test
if [ "$INPUT_TYPE" = "cigar" ]; then
    echo "=== Encoding CIGAR to tracepoint types: ${TP_TYPES[*]} ==="
    for TP_TYPE in "${TP_TYPES[@]}"; do
        echo "  Encoding $TP_TYPE..."
        /usr/bin/time -v $CIGZIP encode --paf "$OUTPUT_DIR/input_sample.paf" --threads 1 --type "$TP_TYPE" \
            --max-complexity "$MAX_COMPLEXITY" --complexity-metric "$COMPLEXITY_METRIC" \
            > "$OUTPUT_DIR/${TP_TYPE}.tp.paf" 2> "$OUTPUT_DIR/${TP_TYPE}_encode.log"

        # Extract encoding metrics
        ENCODE_TIME[$TP_TYPE]=$(grep "Elapsed (wall clock)" "$OUTPUT_DIR/${TP_TYPE}_encode.log" | awk '{print $8}')
        ENCODE_MEM[$TP_TYPE]=$(grep "Maximum resident set size" "$OUTPUT_DIR/${TP_TYPE}_encode.log" | awk '{print $6}')
    done
    echo "✓ All encodings complete"
    echo ""
else
    # Already tracepoints - only standard is valid
    if [ ${#TP_TYPES[@]} -gt 1 ] || [ "${TP_TYPES[0]}" != "standard" ]; then
        echo "Warning: Input is already tracepoint PAF, only 'standard' type is valid"
        TP_TYPES=("standard")
    fi
    cp "$OUTPUT_DIR/input_sample.paf" "$OUTPUT_DIR/standard.tp.paf"
    # No encoding needed - use numeric zeroes to keep TSV fields valid
    ENCODE_TIME["standard"]="0"
    ENCODE_MEM["standard"]="0"
    echo "✓ Using existing tracepoint PAF"
    echo ""
fi

# Compress tracepoint PAF files with BGZIP for comparison (if bgzip is available)
if command -v bgzip &> /dev/null; then
    echo "=== Compressing tracepoint PAF files with BGZIP ==="
    for TP_TYPE in "${TP_TYPES[@]}"; do
        TP_PAF="$OUTPUT_DIR/${TP_TYPE}.tp.paf"
        if [ -f "$TP_PAF" ]; then
            echo "  Compressing ${TP_TYPE}.tp.paf with bgzip..."
            /usr/bin/time -v bgzip -c "$TP_PAF" > "${TP_PAF}.gz" 2> "$OUTPUT_DIR/${TP_TYPE}_bgzip.log"

            # Extract compression metrics
            BGZIP_TIME[$TP_TYPE]=$(grep "Elapsed (wall clock)" "$OUTPUT_DIR/${TP_TYPE}_bgzip.log" | awk '{print $8}')
            BGZIP_MEM[$TP_TYPE]=$(grep "Maximum resident set size" "$OUTPUT_DIR/${TP_TYPE}_bgzip.log" | awk '{print $6}')
            BGZIP_SIZE[$TP_TYPE]=$(file_size "${TP_PAF}.gz")
            TP_SIZE[$TP_TYPE]=$(file_size "$TP_PAF")

            ratio_bgzip=$(safe_ratio "${TP_SIZE[$TP_TYPE]}" "${BGZIP_SIZE[$TP_TYPE]}" 2)
            echo "    Uncompressed: $(numfmt --to=iec ${TP_SIZE[$TP_TYPE]})"
            echo "    BGZIP:        $(numfmt --to=iec ${BGZIP_SIZE[$TP_TYPE]}) (${ratio_bgzip}x compression)"

            # Run BGZIP seek benchmark
            echo "    Running BGZIP seek benchmark..."
            bgzip_seek_result=$("$SEEK_BENCH_BGZIP_PAF" "${TP_PAF}.gz" "$EXTRACTED" 10 10 "$TP_TYPE" "$TP_PAF" || echo "0 0 0 0")
            read -r bgzip_avg bgzip_std bgzip_decode bgzip_valid <<< "$bgzip_seek_result"
            BGZIP_SEEK_AVG[$TP_TYPE]="$bgzip_avg"
            BGZIP_SEEK_STDDEV[$TP_TYPE]="$bgzip_std"
            BGZIP_SEEK_DECODE[$TP_TYPE]="$bgzip_decode"
            BGZIP_SEEK_VALID[$TP_TYPE]="$bgzip_valid"
            echo "    Seek avg: ${bgzip_avg}μs, stddev: ${bgzip_std}μs, valid: ${bgzip_valid}"
        fi
    done
    echo "✓ BGZIP compression and seek benchmark complete"
else
    echo "=== BGZIP compression skipped (bgzip not found in PATH) ==="
fi
echo ""

# Compression strategies
# Base strategies (will be tested with +zstd and +bgzip)
BASE_STRATEGIES=(
    "raw"
    "zigzag-delta"
    "2d-delta"
    "rle"
    "bit-packed"
    "delta-of-delta"
    "frame-of-reference"
    "hybrid-rle"
    "xor-delta"
    "dictionary"
    "stream-vbyte"
    "fastpfor"
    "cascaded"
    "simple8b-full"
    "selective-rle"
    "rice"
    "huffman"
    "lz77"
)

LAYER_SUFFIXES=(
    ""
    "-bgzip"
    "-nocomp"
)

AUTO_STRATEGIES=(
    "automatic"
    "automatic,3,0"
)

# Build full strategy list
STRATEGIES=()

# Add all base strategies across layer suffixes
for layer in "${LAYER_SUFFIXES[@]}"; do
    for strategy in "${BASE_STRATEGIES[@]}"; do
        STRATEGIES+=("${strategy}${layer}")
    done
done

# Add automatic strategies: default (2000 samples) and full file (0 = all records)
# Note: automatic strategies are added dynamically based on TEST_MODE
# When TEST_MODE is "auto:ROWS", we use "automatic,3,ROWS" strategy

# Test function
# bgzip_mode: 0 = per-record, 1 = all-records
test_configuration() {
    local tp_type="$1"
    local first_strategy="$2"
    local second_strategy="${3:-$2}"  # Default to first_strategy if not provided (single mode)
    local bgzip_mode="${4:-0}"  # 0=per-record, 1=all-records
    local tp_paf="$OUTPUT_DIR/${tp_type}.tp.paf"
    local is_auto=0
    if [[ "$first_strategy" == automatic* ]]; then
        is_auto=1
    fi

    # Build bgzip flag for cigzip
    local ALL_RECORDS_FLAG=""
    local key_suffix=""
    if [ "$bgzip_mode" -eq 1 ]; then
        ALL_RECORDS_FLAG="--all-records"
        key_suffix="+all-records"
    fi

    # Create key from strategies (add bgzip suffix if enabled)
    local key="${tp_type}_${first_strategy}"
    if [ "$first_strategy" != "$second_strategy" ]; then
        key="${key}_${second_strategy}"
    fi
    key="${key}${key_suffix}"

    echo "    Testing $first_strategy → $second_strategy..."

    # Store tracepoint PAF size (only once per type)
    if [ -z "${TP_SIZE[$tp_type]}" ]; then
        TP_SIZE[$tp_type]=$(file_size "$tp_paf")
    fi

    # Compress - use cigzip for all modes
    if [ "$TEST_MODE" = "dual" ]; then
        if [ $is_auto -eq 1 ]; then
            echo "      [$first_strategy] compress starting..." >&2
            /usr/bin/time -v $CIGZIP compress -i "$tp_paf" -o "$OUTPUT_DIR/${key}.tpa" \
                --type "$tp_type" --max-complexity "$MAX_COMPLEXITY" \
                --complexity-metric "$COMPLEXITY_METRIC" --distance gap-affine --penalties 5,8,2 \
                --strategy "$first_strategy" $ALL_RECORDS_FLAG 2>&1 | \
                tee "$OUTPUT_DIR/${key}_compress.log" >&2
            echo "      [$first_strategy] compress finished" >&2
        else
            # Use cigzip with dual strategies (--strategy and --strategy-second)
            /usr/bin/time -v $CIGZIP compress -i "$tp_paf" -o "$OUTPUT_DIR/${key}.tpa" \
                --type "$tp_type" --max-complexity "$MAX_COMPLEXITY" \
                --complexity-metric "$COMPLEXITY_METRIC" --distance gap-affine --penalties 5,8,2 \
                --strategy "$first_strategy,3" --strategy-second "$second_strategy,3" $ALL_RECORDS_FLAG 2>&1 | \
                tee "$OUTPUT_DIR/${key}_compress.log" >/dev/null
        fi
    else
        # Use cigzip for single/symmetric strategies
        local strategy_arg="$first_strategy"
        if [[ "$first_strategy" == *"-nocomp" ]]; then
            strategy_arg="${first_strategy},0"
        fi

        /usr/bin/time -v $CIGZIP compress -i "$tp_paf" -o "$OUTPUT_DIR/${key}.tpa" \
            --type "$tp_type" --max-complexity "$MAX_COMPLEXITY" \
            --complexity-metric "$COMPLEXITY_METRIC" --distance gap-affine --penalties 5,8,2 \
            --strategy "$strategy_arg" $ALL_RECORDS_FLAG 2>&1 | tee "$OUTPUT_DIR/${key}_compress.log" >/dev/null
    fi

    COMPRESS_TIME[$key]=$(grep "Elapsed (wall clock)" "$OUTPUT_DIR/${key}_compress.log" | awk '{print $8}')
    COMPRESS_MEM[$key]=$(grep "Maximum resident set size" "$OUTPUT_DIR/${key}_compress.log" | awk '{print $6}')
    COMPRESS_SIZE[$key]=$(file_size "$OUTPUT_DIR/${key}.tpa")

    # Extract actual strategies from TPA header using tpa-view
    local strategy_output=$("$REPO_DIR/target/release/tpa-view" --strategies "$OUTPUT_DIR/${key}.tpa" || echo "unknown\tunknown\tunknown\tunknown")
    read -r first_strat second_strat first_layer second_layer <<< "$strategy_output"
    STRATEGY_FIRST[$key]="$first_strat"
    STRATEGY_SECOND[$key]="$second_strat"
    LAYER_FIRST[$key]="$first_layer"
    LAYER_SECOND[$key]="$second_layer"

    # Measure file open time (100 iterations for quick but stable results)
    local open_time=$("$OPEN_BENCH" "$OUTPUT_DIR/${key}.tpa" 100 --simple || echo "0")
    FILE_OPEN_TIME[$key]="$open_time"

    # Decompress
    if [ $is_auto -eq 1 ]; then
        echo "      [$first_strategy] decompress starting..." >&2
        /usr/bin/time -v $CIGZIP decompress -i "$OUTPUT_DIR/${key}.tpa" \
            -o "$OUTPUT_DIR/${key}_decomp.paf" 2>&1 | tee "$OUTPUT_DIR/${key}_decompress.log" >&2
        echo "      [$first_strategy] decompress finished" >&2
    else
        /usr/bin/time -v $CIGZIP decompress -i "$OUTPUT_DIR/${key}.tpa" \
            -o "$OUTPUT_DIR/${key}_decomp.paf" 2>&1 | tee "$OUTPUT_DIR/${key}_decompress.log" >/dev/null
    fi
    
    DECOMPRESS_TIME[$key]=$(grep "Elapsed (wall clock)" "$OUTPUT_DIR/${key}_decompress.log" | awk '{print $8}')
    DECOMPRESS_MEM[$key]=$(grep "Maximum resident set size" "$OUTPUT_DIR/${key}_decompress.log" | awk '{print $6}')
    
    # Verify
    if [ $is_auto -eq 1 ]; then
        echo "      [$first_strategy] normalize (orig) -> ${OUTPUT_DIR}/${key}_normalize_orig.txt" >&2
        if cat "$tp_paf" | $NORMALIZE > "$OUTPUT_DIR/${key}_normalize_orig.txt" 2> "$OUTPUT_DIR/${key}_normalize_orig.err"; then
            echo "      [$first_strategy] normalize (orig) done" >&2
        else
            echo "      [$first_strategy] normalize (orig) failed" >&2
        fi

        echo "      [$first_strategy] normalize (decomp) -> ${OUTPUT_DIR}/${key}_normalize_decomp.txt" >&2
        if cat "$OUTPUT_DIR/${key}_decomp.paf" | $NORMALIZE > "$OUTPUT_DIR/${key}_normalize_decomp.txt" 2> "$OUTPUT_DIR/${key}_normalize_decomp.err"; then
            echo "      [$first_strategy] normalize (decomp) done" >&2
        else
            echo "      [$first_strategy] normalize (decomp) failed" >&2
        fi

        orig_md5=$(md5sum "$OUTPUT_DIR/${key}_normalize_orig.txt" | cut -d' ' -f1)
        decomp_md5=$(md5sum "$OUTPUT_DIR/${key}_normalize_decomp.txt" | cut -d' ' -f1)
        echo "      [$first_strategy] checksum compare: orig=$orig_md5 decomp=$decomp_md5" >&2
    else
        orig_md5=$(cat "$tp_paf" | $NORMALIZE | md5sum | cut -d' ' -f1)
        decomp_md5=$(cat "$OUTPUT_DIR/${key}_decomp.paf" | $NORMALIZE | md5sum | cut -d' ' -f1)
    fi
    
    if [ "$orig_md5" = "$decomp_md5" ]; then
        VERIFIED[$key]="✓"
    else
        VERIFIED[$key]="✗"
    fi
    
    # Seek Mode A: 10 positions × 10 iterations (for quick testing)
    # Output: avg_us stddev_us decode_ratio valid_ratio
    local seek_a_result=$("$SEEK_BENCH_READER" "$OUTPUT_DIR/${key}.tpa" "$EXTRACTED" 10 10 "$tp_type" "$tp_paf" || echo "0 0 0 0")
    read -r seek_a_avg seek_a_std seek_a_decode seek_a_valid <<< "$seek_a_result"
    SEEK_A[$key]="$seek_a_avg"
    SEEK_A_STDDEV[$key]="$seek_a_std"

    # Seek Mode B: 10 positions × 10 iterations (for quick testing)
    # Uses standalone functions (read_*_tracepoints_at_offset / read_*_tracepoints_at_vpos)
    # Supports both per-record mode (raw byte offsets) and all-records mode (virtual positions)
    local seek_b_result=$("$SEEK_BENCH_DIRECT" "$OUTPUT_DIR/${key}.tpa" "$EXTRACTED" 10 10 "$tp_type" "$tp_paf" || echo "0 0 0 0")
    read -r seek_b_avg seek_b_std seek_b_decode seek_b_valid <<< "$seek_b_result"
    SEEK_B[$key]="$seek_b_avg"
    SEEK_B_STDDEV[$key]="$seek_b_std"
    SEEK_DECODE_RATIO[$key]="${seek_b_decode:-0}"
    SEEK_VALID_RATIO[$key]="${seek_b_valid:-0}"

    # Determine pass/fail for logging
    local failure_reasons=()
    if [ "${VERIFIED[$key]}" != "✓" ]; then
        failure_reasons+=("verification:${orig_md5}->${decomp_md5}")
    fi
    local seek_ratio_value="${SEEK_DECODE_RATIO[$key]}"
    if ! printf '%s' "$seek_ratio_value" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
        failure_reasons+=("seek:non-numeric:${seek_ratio_value}")
    else
        if ! awk "BEGIN{exit !($seek_ratio_value >= 1)}"; then
            failure_reasons+=("seek:${seek_ratio_value}")
        fi
    fi

    if [ ${#failure_reasons[@]} -gt 0 ]; then
        {
            echo "tp_type=$tp_type"
            echo "first=$first_strategy"
            echo "second=$second_strategy"
            echo "key=$key"
            echo "tpa=$OUTPUT_DIR/${key}.tpa"
            echo "decomp=$OUTPUT_DIR/${key}_decomp.paf"
            echo "compress_log=$OUTPUT_DIR/${key}_compress.log"
            echo "decompress_log=$OUTPUT_DIR/${key}_decompress.log"
            if [ $is_auto -eq 1 ]; then
                echo "normalize_orig=$OUTPUT_DIR/${key}_normalize_orig.txt"
                echo "normalize_decomp=$OUTPUT_DIR/${key}_normalize_decomp.txt"
            fi
            echo "orig_md5=$orig_md5"
            echo "decomp_md5=$decomp_md5"
            echo "seek_ratio=${seek_ratio_value}"
            echo "seek_a_avg=$seek_a_avg seek_b_avg=$seek_b_avg"
            echo "failure_reason=$(IFS=','; echo \"${failure_reasons[*]}\")"
            echo "---"
        } >> "$OUTPUT_DIR/failing_strategies.log"
    else
        # Cleanup on success
        rm -f "$OUTPUT_DIR/${key}_compress.log" "$OUTPUT_DIR/${key}_decompress.log" "$OUTPUT_DIR/${key}_decomp.paf"
        if [ $is_auto -eq 1 ]; then
            rm -f "$OUTPUT_DIR/${key}_normalize_orig.txt" "$OUTPUT_DIR/${key}_normalize_orig.err" \
                  "$OUTPUT_DIR/${key}_normalize_decomp.txt" "$OUTPUT_DIR/${key}_normalize_decomp.err"
        fi
    fi
}

# TSV output function
output_tsv_row() {
    local tp_type="$1"
    local first_strategy="$2"
    local second_strategy="${3:-$2}"  # Default to first_strategy if not provided
    local bgzip_mode="${4:-0}"  # 0=per-record, 1=all-records
    local tsv_file="$OUTPUT_DIR/results.tsv"

    # Build key suffix for all-records mode
    local key_suffix=""
    if [ "$bgzip_mode" -eq 1 ]; then
        key_suffix="+all-records"
    fi

    # Create key from strategies (must match test_configuration)
    local key="${tp_type}_${first_strategy}"
    if [ "$first_strategy" != "$second_strategy" ]; then
        key="${key}_${second_strategy}"
    fi
    key="${key}${key_suffix}"

    # Strategy label for display (includes all-records mode suffix)
    local strategy_label="$first_strategy"
    if [ "$first_strategy" != "$second_strategy" ]; then
        strategy_label="${first_strategy}→${second_strategy}"
    fi
    strategy_label="${strategy_label}${key_suffix}"

    # Calculate ratios with guards to avoid division by zero or empty values
    local tp_size_bytes=${TP_SIZE[$tp_type]:-0}
    local tpa_size_bytes=${COMPRESS_SIZE[$key]:-0}

    local ratio_orig_to_tp
    ratio_orig_to_tp=$(safe_ratio "$SIZE" "$tp_size_bytes" 3)

    local ratio_tp_to_tpa
    ratio_tp_to_tpa=$(safe_ratio "$tp_size_bytes" "$tpa_size_bytes" 3)

    local ratio_orig_to_tpa
    ratio_orig_to_tpa=$(safe_ratio "$SIZE" "$tpa_size_bytes" 3)

    # Convert times to seconds
    local encode_time_sec=$(time_to_seconds "$(normalize_time_field "${ENCODE_TIME[$tp_type]}")")
    local compress_time_sec=$(time_to_seconds "$(normalize_time_field "${COMPRESS_TIME[$key]}")")
    local decompress_time_sec=$(time_to_seconds "$(normalize_time_field "${DECOMPRESS_TIME[$key]}")")

    # Convert memory to MB
    local encode_mem_mb=$(kb_to_mb "$(normalize_time_field "${ENCODE_MEM[$tp_type]}")")
    local compress_mem_mb=$(kb_to_mb "$(normalize_time_field "${COMPRESS_MEM[$key]}")")
    local decompress_mem_mb=$(kb_to_mb "$(normalize_time_field "${DECOMPRESS_MEM[$key]}")")

    # Verification status
    local verified="${VERIFIED[$key]}"
    local verified_text="no"
    if [ "$verified" = "✓" ]; then
        verified_text="yes"
    fi

    # Seek ratios fallback
    local seek_decode=${SEEK_DECODE_RATIO[$key]:-0}
    local seek_valid=${SEEK_VALID_RATIO[$key]:-0}

    # Dataset name from input file
    local dataset_name=$(basename "$INPUT_PAF" .paf.gz | sed 's/.paf$//')

    # File open time
    local file_open_time="${FILE_OPEN_TIME[$key]:-0}"

    # Output TSV row (34 columns - includes strategy/layer pairs and file open time)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset_name" \
        "$INPUT_TYPE" \
        "$SIZE" \
        "$EXTRACTED" \
        "$tp_type" \
        "$encode_time_sec" \
        "$encode_mem_mb" \
        "$tp_size_bytes" \
        "$MAX_COMPLEXITY" \
        "$COMPLEXITY_METRIC" \
        "$strategy_label" \
        "${STRATEGY_FIRST[$key]}" \
        "${STRATEGY_SECOND[$key]}" \
        "${LAYER_FIRST[$key]}" \
        "${LAYER_SECOND[$key]}" \
        "$compress_time_sec" \
        "$compress_mem_mb" \
        "$tpa_size_bytes" \
        "$ratio_orig_to_tp" \
        "$ratio_tp_to_tpa" \
        "$ratio_orig_to_tpa" \
        "$decompress_time_sec" \
        "$decompress_mem_mb" \
        "$verified_text" \
        "100" \
        "100" \
        "10000" \
        "${SEEK_A[$key]}" \
        "${SEEK_A_STDDEV[$key]}" \
        "${SEEK_B[$key]}" \
        "${SEEK_B_STDDEV[$key]}" \
        "$seek_decode" \
        "$seek_valid" \
        "$file_open_time" \
        >> "$tsv_file"
}

# TSV output function for BGZIP baseline
output_bgzip_baseline_row() {
    local tp_type="$1"
    local tsv_file="$OUTPUT_DIR/results.tsv"

    # Calculate ratios
    local tp_size_bytes=${TP_SIZE[$tp_type]:-0}
    local bgzip_size_bytes=${BGZIP_SIZE[$tp_type]:-0}

    local ratio_orig_to_tp
    ratio_orig_to_tp=$(safe_ratio "$SIZE" "$tp_size_bytes" 3)

    local ratio_tp_to_bgzip
    ratio_tp_to_bgzip=$(safe_ratio "$tp_size_bytes" "$bgzip_size_bytes" 3)

    local ratio_orig_to_bgzip
    ratio_orig_to_bgzip=$(safe_ratio "$SIZE" "$bgzip_size_bytes" 3)

    # Convert times to seconds
    local encode_time_sec=$(time_to_seconds "$(normalize_time_field "${ENCODE_TIME[$tp_type]}")")
    local bgzip_time_sec=$(time_to_seconds "$(normalize_time_field "${BGZIP_TIME[$tp_type]}")")

    # Convert memory to MB
    local encode_mem_mb=$(kb_to_mb "$(normalize_time_field "${ENCODE_MEM[$tp_type]}")")
    local bgzip_mem_mb=$(kb_to_mb "$(normalize_time_field "${BGZIP_MEM[$tp_type]}")")

    # Verification status - based on seek benchmark validity
    local seek_valid=${BGZIP_SEEK_VALID[$tp_type]:-0}
    local verified_text="no"
    # Check if valid ratio >= 1.0 (all seeks validated correctly)
    if awk "BEGIN{exit !($seek_valid >= 1.0)}" 2>/dev/null; then
        verified_text="yes"
    fi

    # Dataset name from input file
    local dataset_name=$(basename "$INPUT_PAF" .paf.gz | sed 's/.paf$//')

    # Output TSV row (34 columns - same structure as TPA rows)
    # For BGZIP baseline:
    #   - compression_strategy: "bgzip-only"
    #   - strategy_first/second: "bgzip"
    #   - layer_first/second: "bgzip"
    #   - tpa_size_bytes column contains BGZIP compressed size
    #   - seek_mode_a: NA (no high-level reader abstraction for BGZIP)
    #   - seek_mode_b: BGZIP seek time (direct BGZF virtual position seeking)
    #   - file_open_time_us: NA (no TpaReader for BGZIP PAF)
    #
    # NOTE: BGZIP seek is placed in Mode B because it uses pre-computed virtual positions
    # and direct bgzf::io::Reader access, which is analogous to TPA Mode B (standalone functions
    # with pre-computed offsets). There's no Mode A equivalent because no "BgzipPafReader"
    # high-level abstraction exists - the bgzf::io::Reader IS the low-level reader.
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset_name" \
        "$INPUT_TYPE" \
        "$SIZE" \
        "$EXTRACTED" \
        "$tp_type" \
        "$encode_time_sec" \
        "$encode_mem_mb" \
        "$tp_size_bytes" \
        "$MAX_COMPLEXITY" \
        "$COMPLEXITY_METRIC" \
        "bgzip-only" \
        "bgzip" \
        "bgzip" \
        "bgzip" \
        "bgzip" \
        "$bgzip_time_sec" \
        "$bgzip_mem_mb" \
        "$bgzip_size_bytes" \
        "$ratio_orig_to_tp" \
        "$ratio_tp_to_bgzip" \
        "$ratio_orig_to_bgzip" \
        "0" \
        "0" \
        "$verified_text" \
        "100" \
        "100" \
        "10000" \
        "NA" \
        "NA" \
        "${BGZIP_SEEK_AVG[$tp_type]:-0}" \
        "${BGZIP_SEEK_STDDEV[$tp_type]:-0}" \
        "${BGZIP_SEEK_DECODE[$tp_type]:-0}" \
        "$seek_valid" \
        "NA" \
        >> "$tsv_file"
}

# Initialize TSV file with header
TSV_FILE="$OUTPUT_DIR/results.tsv"
cat > "$TSV_FILE" << TSV_HEADER
dataset_name	dataset_type	original_size_bytes	num_records	encoding_type	encoding_runtime_sec	encoding_memory_mb	tp_file_size_bytes	max_complexity	complexity_metric	compression_strategy	strategy_first	strategy_second	compression_layer_first	compression_layer_second	compression_runtime_sec	compression_memory_mb	tpa_size_bytes	ratio_orig_to_tp	ratio_tp_to_tpa	ratio_orig_to_tpa	decompression_runtime_sec	decompression_memory_mb	verification_passed	seek_positions_tested	seek_iterations_per_position	seek_total_tests	seek_mode_a_avg_us	seek_mode_a_stddev_us	seek_mode_b_avg_us	seek_mode_b_stddev_us	seek_decode_ratio	seek_valid_ratio	file_open_time_us
TSV_HEADER

# Run all tests
for TP_TYPE in "${TP_TYPES[@]}"; do
    echo "═══════════════════════════════════════════════════"
    echo "Testing Tracepoint Type: ${TP_TYPE^^}"
    echo "═══════════════════════════════════════════════════"

    # Output BGZIP baseline row first (if bgzip was available)
    if [ -n "${BGZIP_SIZE[$TP_TYPE]:-}" ] && [ "${BGZIP_SIZE[$TP_TYPE]}" -gt 0 ]; then
        echo "  Adding BGZIP baseline to results..."
        output_bgzip_baseline_row "$TP_TYPE"
    fi

    if [ "$TEST_MODE" = "dual" ]; then
        # Test all first×second×layer combinations
        total_combos=$((${#BASE_STRATEGIES[@]} * ${#LAYER_SUFFIXES[@]} * ${#BASE_STRATEGIES[@]} * ${#LAYER_SUFFIXES[@]}))
        combo_count=0
        echo "Testing $total_combos dual strategy combinations ((${#BASE_STRATEGIES[@]} strategies × ${#LAYER_SUFFIXES[@]} layers)^2) with $THREADS threads..."

        # Function to run a single test (for parallel execution)
        run_test() {
            local tp_type="$1"
            local first_full="$2"
            local second_full="$3"
            local temp_tsv="$4"

            test_configuration "$tp_type" "$first_full" "$second_full"
            output_tsv_row "$tp_type" "$first_full" "$second_full" >> "$temp_tsv"
        }
        export -f run_test
        export -f test_configuration
        export -f output_tsv_row

        # Create temporary directory for parallel results
        TEMP_DIR="$OUTPUT_DIR/parallel_tmp"
        mkdir -p "$TEMP_DIR"

        # Run tests in parallel
        job_count=0
        for LAYER_FIRST in "${LAYER_SUFFIXES[@]}"; do
            for LAYER_SECOND in "${LAYER_SUFFIXES[@]}"; do
                for FIRST in "${BASE_STRATEGIES[@]}"; do
                    for SECOND in "${BASE_STRATEGIES[@]}"; do
                        combo_count=$((combo_count + 1))
                    FIRST_FULL="${FIRST}${LAYER_FIRST}"
                    SECOND_FULL="${SECOND}${LAYER_SECOND}"
                        TEMP_TSV="$TEMP_DIR/test_${combo_count}.tsv"

                    # Run test in background
                    run_test "$TP_TYPE" "$FIRST_FULL" "$SECOND_FULL" "$TEMP_TSV" &

                    job_count=$((job_count + 1))

                    # Wait if we've reached the thread limit
                    if [ $((job_count % THREADS)) -eq 0 ]; then
                        wait
                        echo "  Progress: $combo_count/$total_combos"
                    fi
                done
                done
            done
        done

        # Wait for remaining jobs
        wait

        # Merge all temporary TSV files
        for tsv_file in "$TEMP_DIR"/test_*.tsv; do
            [ -f "$tsv_file" ] && cat "$tsv_file" >> "$OUTPUT_DIR/results.tsv"
        done

        # Clean up
        rm -rf "$TEMP_DIR"

        # Also test automatic modes (works for all tracepoint types)
        for auto_name in "${AUTO_STRATEGIES[@]}"; do
            echo ""
            echo "Testing ${auto_name} meta-strategy (selects best per stream from $((${#BASE_STRATEGIES[@]} * ${#LAYER_SUFFIXES[@]})) combinations)..."
            test_configuration "$TP_TYPE" "$auto_name"

            # Extract the strategies that automatic mode selected from the TPA header
            auto_tpa="$OUTPUT_DIR/${TP_TYPE}_${auto_name}.tpa"
            selected_strategies=$("$REPO_DIR/target/release/tpa-view" --strategies "$auto_tpa")
            first_selected=$(echo "$selected_strategies" | cut -f1)
            second_selected=$(echo "$selected_strategies" | cut -f2)
            first_layer_selected=$(echo "$selected_strategies" | cut -f3)
            second_layer_selected=$(echo "$selected_strategies" | cut -f4)

            echo "  → ${auto_name} selected: ${first_selected}[${first_layer_selected}] → ${second_selected}[${second_layer_selected}]"

            # Output TSV row for the automatic run (metrics recorded under current key)
            output_tsv_row "$TP_TYPE" "$auto_name"
        done

        echo ""
        echo "✓ Completed ${total_combos} explicit dual combinations (no automatic)"
    elif [[ "$TEST_MODE" == auto:* ]]; then
        # Extract sample size from TEST_MODE (format: auto:ROWS)
        AUTO_SAMPLE_SIZE="${TEST_MODE#auto:}"
        AUTO_STRATEGY="automatic,3,${AUTO_SAMPLE_SIZE}"

        if [ "$AUTO_SAMPLE_SIZE" -eq 0 ]; then
            echo "Testing automatic mode with full file analysis..."
        else
            echo "Testing automatic mode with ${AUTO_SAMPLE_SIZE}-record sampling..."
        fi

        echo ""
        echo "Testing ${AUTO_STRATEGY} meta-strategy (selects best per stream from $((${#BASE_STRATEGIES[@]} * ${#LAYER_SUFFIXES[@]})) combinations)..."
        test_configuration "$TP_TYPE" "$AUTO_STRATEGY"

        # Extract the strategies that automatic mode selected from the TPA header
        auto_tpa="$OUTPUT_DIR/${TP_TYPE}_${AUTO_STRATEGY}.tpa"
        selected_strategies=$("$REPO_DIR/target/release/tpa-view" --strategies "$auto_tpa")
        first_selected=$(echo "$selected_strategies" | cut -f1)
        second_selected=$(echo "$selected_strategies" | cut -f2)
        first_layer_selected=$(echo "$selected_strategies" | cut -f3)
        second_layer_selected=$(echo "$selected_strategies" | cut -f4)

        echo "  → Selected: ${first_selected}[${first_layer_selected}] → ${second_selected}[${second_layer_selected}]"

        # Output TSV row for the automatic run (per-record mode)
        output_tsv_row "$TP_TYPE" "$AUTO_STRATEGY" "$AUTO_STRATEGY" 0
        echo ""
        echo "✓ Completed automatic strategy test (per-record mode)"

        # Test all-records mode (--all-records flag, header/string table plain, records BGZIP-compressed)
        echo ""
        echo "Testing ${AUTO_STRATEGY}+all-records (all-records mode)..."
        test_configuration "$TP_TYPE" "$AUTO_STRATEGY" "$AUTO_STRATEGY" 1

        # Extract the strategies for all-records mode
        auto_all_records_tpa="$OUTPUT_DIR/${TP_TYPE}_${AUTO_STRATEGY}+all-records.tpa"
        all_records_selected_strategies=$("$REPO_DIR/target/release/tpa-view" --strategies "$auto_all_records_tpa")
        all_records_first_selected=$(echo "$all_records_selected_strategies" | cut -f1)
        all_records_second_selected=$(echo "$all_records_selected_strategies" | cut -f2)

        echo "  → Selected: ${all_records_first_selected} → ${all_records_second_selected} [all-records]"

        # Output TSV row for all-records mode
        output_tsv_row "$TP_TYPE" "$AUTO_STRATEGY" "$AUTO_STRATEGY" 1
        echo ""
        echo "✓ Completed all-records mode test"
    else
        # Single mode: test each strategy symmetrically (first==second)
        for STRATEGY in "${STRATEGIES[@]}"; do
            test_configuration "$TP_TYPE" "$STRATEGY"
            output_tsv_row "$TP_TYPE" "$STRATEGY"
        done
    fi
    echo ""
done

# Generate report
REPORT="$OUTPUT_DIR/test_report.md"
cat > "$REPORT" << HEADER
# Comprehensive Test Report

**Input:** $INPUT_PAF
**Type:** $INPUT_TYPE ($([ $IS_COMPRESSED -eq 1 ] && echo 'compressed' || echo 'uncompressed'))
**Records:** $EXTRACTED
**Size:** $SIZE bytes
**Date:** $(date +%Y-%m-%d)

---

HEADER

if [ "$TEST_MODE" = "dual" ]; then
    # For dual mode, skip detailed table (too many rows), just reference TSV
    cat >> "$REPORT" << DUAL_NOTE
## Dual Strategy Testing Mode

Testing all ${#BASE_STRATEGIES[@]}×${#BASE_STRATEGIES[@]}=289 strategy combinations per tracepoint type.

**Results are available in:** \`results.tsv\`

Use \`plot_results.py\` to visualize the data or analyze the TSV file directly.

**Baseline Sizes:**
DUAL_NOTE

    for TP_TYPE in "${TP_TYPES[@]}"; do
        tp_size_bytes=${TP_SIZE[$TP_TYPE]}
            tp_size_mb=$(safe_ratio "$tp_size_bytes" 1048576 2)
        echo "- $TP_TYPE: $tp_size_bytes bytes ($tp_size_mb MB)" >> "$REPORT"
    done
    echo "" >> "$REPORT"
elif [[ "$TEST_MODE" == auto:* ]]; then
    # Extract sample size from TEST_MODE
    AUTO_SAMPLE_SIZE="${TEST_MODE#auto:}"
    if [ "$AUTO_SAMPLE_SIZE" -eq 0 ]; then
        AUTO_DESC="full file analysis"
    else
        AUTO_DESC="${AUTO_SAMPLE_SIZE}-record sampling"
    fi

    cat >> "$REPORT" << AUTO_NOTE
## Automatic Strategy Testing Mode

Testing automatic mode with ${AUTO_DESC}.

**Results are available in:** \`results.tsv\`

**Automatic Strategy Tested:**
- \`automatic,3,${AUTO_SAMPLE_SIZE}\`
AUTO_NOTE
    echo "" >> "$REPORT"
    echo "**Baseline Sizes:**" >> "$REPORT"
    for TP_TYPE in "${TP_TYPES[@]}"; do
        tp_size_bytes=${TP_SIZE[$TP_TYPE]}
        tp_size_mb=$(safe_ratio "$tp_size_bytes" 1048576 2)
        echo "- $TP_TYPE: $tp_size_bytes bytes ($tp_size_mb MB)" >> "$REPORT"
    done
    echo "" >> "$REPORT"
else
    # Single mode: generate detailed markdown table
    for TP_TYPE in "${TP_TYPES[@]}"; do
        # Calculate baseline sizes
        tp_size_bytes=${TP_SIZE[$TP_TYPE]}
            tp_size_mb=$(safe_ratio "$tp_size_bytes" 1048576 2)

        cat >> "$REPORT" << SECTION
## Tracepoint Type: ${TP_TYPE^^}

**Baseline Sizes:**
- Original Input (CIGAR PAF): $SIZE bytes ($(echo "scale=2; $SIZE / 1048576" | bc) MB)
- Tracepoint PAF ($TP_TYPE): $tp_size_bytes bytes ($tp_size_mb MB)

| Strategy | Compressed Size (bytes) | TPA Ratio | End-to-End Ratio | Compress Time | Compress Mem (KB) | Decompress Time | Decompress Mem (KB) | Seek A (μs) | Seek B (μs) | Verified |
|----------|-------------------------|------------|------------------|---------------|-------------------|-----------------|---------------------|-------------|-------------|----------|
SECTION

        for STRATEGY in "${STRATEGIES[@]}"; do
            key="${TP_TYPE}_${STRATEGY}"

            # Calculate both ratios
            tpa_ratio=$(safe_ratio "$tp_size_bytes" "${COMPRESS_SIZE[$key]}" 2)
            e2e_ratio=$(safe_ratio "$SIZE" "${COMPRESS_SIZE[$key]}" 2)

            cat >> "$REPORT" << ROW
| $STRATEGY | ${COMPRESS_SIZE[$key]} | ${tpa_ratio}x | ${e2e_ratio}x | ${COMPRESS_TIME[$key]} | ${COMPRESS_MEM[$key]} | ${DECOMPRESS_TIME[$key]} | ${DECOMPRESS_MEM[$key]} | ${SEEK_A[$key]} | ${SEEK_B[$key]} | ${VERIFIED[$key]} |
ROW
        done

        echo "" >> "$REPORT"
    done
fi

cat >> "$REPORT" << FOOTER

---

## Compression Ratio Legend

- **TPA Ratio:** Tracepoint PAF size / TPA size
  - Measures TPA compression strategy effectiveness only
  - Use this to compare strategy performance
  - Fair comparison across all strategies

- **End-to-End Ratio:** Original input size / TPA size
  - Measures total compression from original input
  - Includes CIGAR→tracepoint conversion gains (if applicable)
  - Higher for CIGAR inputs due to format conversion

**Note:** For CIGAR PAF inputs, End-to-End ratio includes both format conversion (~6-10x) and TPA compression. For tracepoint PAF inputs, both ratios are identical.

## Seek Mode Legend

- **Mode A:** TpaReader with index (general use)
- **Mode B:** Standalone functions (ultimate performance)

## Verification

All tests use 3-decimal float normalization:
- Normal: \`0.993724\` → \`0.993\`
- Leading dot: \`.0549\` → \`0.054\`
- Integer: \`0\` → \`0.000\`, \`1\` → \`1.000\`

FOOTER

echo "========================================="
echo "Tests complete!"
echo "Markdown Report: $REPORT"
echo "TSV Data:        $TSV_FILE"
echo "========================================="
