#!/bin/bash
set -e # Exit on error

# Comprehensive test suite for TPA compression
#
# NOTE: This script is designed to be called from run_all_tests.sh, which handles
#       building tools and setting environment variables. For standalone use,
#       ensure CIGZIP is set and tpa examples are built.
#
# REQUIREMENTS:
#   - CIGZIP environment variable pointing to cigzip binary
#   - tpa examples built (cargo build --release --examples)
#   - bgzip in PATH
#
# USAGE:
#   ./comprehensive_test.sh <input.paf[.gz]> [output_dir] [max_complexity] [metric] [num_records] [test_mode] [threads] [tp_types]
#
# TEST MODES:
#   single    - Test each strategy symmetrically (first==second) [default]
#   dual      - Test all first×second strategy combinations
#   auto:N    - Only test benchmark,3,N (auto-selects best dual encoding)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Normalization script for PAF comparison
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
    # Pass through NA
    if [ "$time_str" = "NA" ]; then
        echo "NA"
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
    # Pass through NA
    if [ "$kb" = "NA" ]; then
        echo "NA"
        return
    fi
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
    # Pass through NA, convert empty/N/A to 0
    if [ "$val" = "NA" ]; then
        echo "NA"
    elif [ -z "$val" ] || [ "$val" = "N/A" ]; then
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

# Get type-specific max_complexity (trace_spacing for fastga)
get_max_complexity() {
    local tp_type="$1"
    if [ "$tp_type" = "fastga" ]; then
        echo "100"
    else
        echo "$MAX_COMPLEXITY"
    fi
}

# Parameters
INPUT_PAF="${1}"
OUTPUT_DIR="${2:-/tmp/tpa_test_output}"
MAX_COMPLEXITY="${3:-32}"
COMPLEXITY_METRICS_INPUT="${4:-edit-distance}"
NUM_RECORDS="${5:-20000}"
TEST_MODE="${6:-single}"
THREADS="${7:-1}"
TP_TYPES_INPUT="${8:-standard}"

# Seek benchmark parameters
SEEK_POSITIONS=100
SEEK_ITERATIONS=100

if [ -z "$INPUT_PAF" ] || [ ! -f "$INPUT_PAF" ]; then
    echo "Usage: $0 <input.paf[.gz]> [output_dir] [max_complexity] [complexity_metrics] [num_records] [test_mode] [threads] [tp_types]"
    echo ""
    echo "Automatically detects input type:"
    echo "  - CIGAR PAF (compressed or uncompressed)"
    echo "  - Tracepoint PAF"
    echo ""
    echo "Test Modes:"
    echo "  single (default) - Test each strategy symmetrically (first==second)"
    echo "  dual             - Test all strategy combinations (no automatic)"
    echo "  auto:ROWS        - Test benchmark,3,N (best dual encoding)"
    echo ""
    echo "Complexity Metrics (comma-separated):"
    echo "  edit-distance     - Edit distance metric (default)"
    echo "  diagonal-distance - Diagonal distance metric"
    echo "  Example: edit-distance,diagonal-distance"
    echo ""
    echo "Tracepoint Types (comma-separated):"
    echo "  standard  - Standard tracepoints (default)"
    echo "  variable  - Variable tracepoints"
    echo "  mixed     - Mixed tracepoints"
    echo "  fastga    - FastGA tracepoints (ignores complexity_metric)"
    echo "  Example: standard,variable,mixed,fastga"
    exit 1
fi

# Parse comma-separated inputs into arrays
IFS=',' read -ra TP_TYPES_FROM_ARG <<< "$TP_TYPES_INPUT"
IFS=',' read -ra COMPLEXITY_METRICS_FROM_ARG <<< "$COMPLEXITY_METRICS_INPUT"

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "TPA Tests Configuration"
echo "========================================="
echo "Input:       $INPUT_PAF"
echo "Output:      $OUTPUT_DIR"
echo "Complexity:  $MAX_COMPLEXITY"
echo "Metrics:     ${COMPLEXITY_METRICS_FROM_ARG[*]}"
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
    echo "Detected: CIGAR PAF ($([ $IS_COMPRESSED -eq 1 ] && echo 'compressed' || echo 'uncompressed'))"
elif echo "$FIRST_LINE" | grep -q "tp:Z:"; then
    INPUT_TYPE="tracepoint"
    echo "Detected: Tracepoint PAF"
else
    echo "Error: Cannot detect PAF type (no cg:Z: or tp:Z: tags found)"
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

# Paths to benchmark binaries
SEEK_BENCH_READER="$REPO_DIR/target/release/examples/seek_bench_reader"
SEEK_BENCH_DIRECT="$REPO_DIR/target/release/examples/seek_bench_direct"
SEEK_BENCH_BGZIP_PAF="$REPO_DIR/target/release/examples/seek_bench_bgzip_paf"

# Results storage - declare arrays before use
declare -A ENCODE_TIME ENCODE_MEM
declare -A COMPRESS_TIME COMPRESS_MEM COMPRESS_SIZE
declare -A DECOMPRESS_TIME DECOMPRESS_MEM
declare -A SEEK_FULL_OPEN SEEK_FULL
declare -A SEEK_DIRECT_OPEN SEEK_DIRECT
declare -A SEEK_DECODE_RATIO SEEK_VALID_RATIO
declare -A VERIFIED
declare -A TP_SIZE
declare -A BGZIP_TIME BGZIP_MEM BGZIP_SIZE
declare -A BGZIP_OPEN_TIME BGZIP_SEEK_AVG BGZIP_SEEK_DECODE BGZIP_SEEK_VALID
declare -A STRATEGY_FIRST STRATEGY_SECOND
declare -A LAYER_FIRST LAYER_SECOND

# Use tracepoint types and metrics from arguments
TP_TYPES=("${TP_TYPES_FROM_ARG[@]}")
COMPLEXITY_METRICS=("${COMPLEXITY_METRICS_FROM_ARG[@]}")

# Build list of (tp_type, metric) encoding configs
# For fastga: metric is NA (not applicable), encode once
# For others: encode for each metric
declare -a ENCODING_CONFIGS=()
for TP_TYPE in "${TP_TYPES[@]}"; do
    if [ "$TP_TYPE" = "fastga" ]; then
        ENCODING_CONFIGS+=("${TP_TYPE}:NA")
    else
        for METRIC in "${COMPLEXITY_METRICS[@]}"; do
            ENCODING_CONFIGS+=("${TP_TYPE}:${METRIC}")
        done
    fi
done

echo "Encoding configs: ${ENCODING_CONFIGS[*]}"
echo ""

# Determine tracepoint types to test
if [ "$INPUT_TYPE" = "cigar" ]; then
    echo "=== Encoding CIGAR to tracepoints ==="
    for CONFIG in "${ENCODING_CONFIGS[@]}"; do
        TP_TYPE="${CONFIG%%:*}"
        METRIC="${CONFIG##*:}"

        tp_max_complexity=$(get_max_complexity "$TP_TYPE")

        # For fastga, use edit-distance (ignored by cigzip) but display NA
        if [ "$TP_TYPE" = "fastga" ]; then
            cigzip_metric="edit-distance"
            echo "  Encoding $TP_TYPE (max_complexity=$tp_max_complexity, metric=NA)..."
        else
            cigzip_metric="$METRIC"
            echo "  Encoding $TP_TYPE (max_complexity=$tp_max_complexity, metric=$METRIC)..."
        fi

        /usr/bin/time -v $CIGZIP encode --paf "$OUTPUT_DIR/input_sample.paf" --threads 1 --type "$TP_TYPE" \
            --max-complexity "$tp_max_complexity" --complexity-metric "$cigzip_metric" \
            > "$OUTPUT_DIR/${CONFIG}.tp.paf" 2> "$OUTPUT_DIR/${CONFIG}_encode.log"

        # Extract encoding metrics
        ENCODE_TIME[$CONFIG]=$(grep "Elapsed (wall clock)" "$OUTPUT_DIR/${CONFIG}_encode.log" | awk '{print $8}')
        ENCODE_MEM[$CONFIG]=$(grep "Maximum resident set size" "$OUTPUT_DIR/${CONFIG}_encode.log" | awk '{print $6}')
    done
    echo ""
else
    # For existing tracepoints, we don't know the original type and metric
    # Use first specified
    FIRST_TYPE="${TP_TYPES[0]}"
    FIRST_METRIC="${COMPLEXITY_METRICS[0]}"
    if [ "$FIRST_TYPE" = "fastga" ]; then
        CONFIG="${FIRST_TYPE}:NA"
    else
        CONFIG="${FIRST_TYPE}:${FIRST_METRIC}"
    fi
    ENCODING_CONFIGS=("$CONFIG")
    cp "$OUTPUT_DIR/input_sample.paf" "$OUTPUT_DIR/${CONFIG}.tp.paf"
    # No encoding needed
    ENCODE_TIME["$CONFIG"]="NA"
    ENCODE_MEM["$CONFIG"]="NA"
    echo "Using existing tracepoint PAF as ${CONFIG}"
    echo ""
fi

# Compress tracepoint PAF files with BGZIP for baseline comparison
echo "=== Compressing tracepoint PAF files with BGZIP ==="
for CONFIG in "${ENCODING_CONFIGS[@]}"; do
    TP_TYPE="${CONFIG%%:*}"
    TP_PAF="$OUTPUT_DIR/${CONFIG}.tp.paf"
    echo "  Compressing ${CONFIG}.tp.paf with bgzip..."
    /usr/bin/time -v bgzip -c "$TP_PAF" > "${TP_PAF}.gz" 2> "$OUTPUT_DIR/${CONFIG}_bgzip.log"

    # Extract compression metrics
    BGZIP_TIME[$CONFIG]=$(grep "Elapsed (wall clock)" "$OUTPUT_DIR/${CONFIG}_bgzip.log" | awk '{print $8}')
    BGZIP_MEM[$CONFIG]=$(grep "Maximum resident set size" "$OUTPUT_DIR/${CONFIG}_bgzip.log" | awk '{print $6}')
    BGZIP_SIZE[$CONFIG]=$(file_size "${TP_PAF}.gz")
    TP_SIZE[$CONFIG]=$(file_size "$TP_PAF")

    ratio_bgzip=$(safe_ratio "${TP_SIZE[$CONFIG]}" "${BGZIP_SIZE[$CONFIG]}" 2)
    echo "    Uncompressed: $(numfmt --to=iec ${TP_SIZE[$CONFIG]})"
    echo "    BGZIP:        $(numfmt --to=iec ${BGZIP_SIZE[$CONFIG]}) (${ratio_bgzip}x compression)"

    # Run BGZIP seek benchmark
    echo "    Running BGZIP seek benchmark..."
    bgzip_seek_result=$("$SEEK_BENCH_BGZIP_PAF" "${TP_PAF}.gz" "$EXTRACTED" "$SEEK_POSITIONS" "$SEEK_ITERATIONS" "$TP_TYPE" "$TP_PAF" || echo "0 0 0 0")
    read -r bgzip_open bgzip_seek bgzip_decode bgzip_valid <<< "$bgzip_seek_result"
    BGZIP_OPEN_TIME[$CONFIG]="$bgzip_open"
    BGZIP_SEEK_AVG[$CONFIG]="$bgzip_seek"
    BGZIP_SEEK_DECODE[$CONFIG]="$bgzip_decode"
    BGZIP_SEEK_VALID[$CONFIG]="$bgzip_valid"
    echo "    Open: ${bgzip_open}μs, Seek: ${bgzip_seek}μs, valid: ${bgzip_valid}"
done
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
    "-zstd"
    "-bgzip"
    "-nocomp"
)

# Build full strategy list
STRATEGIES=()

# Add all base strategies across layer suffixes
for layer in "${LAYER_SUFFIXES[@]}"; do
    for strategy in "${BASE_STRATEGIES[@]}"; do
        STRATEGIES+=("${strategy}${layer}")
    done
done

# Test function
# config: tp_type:metric (e.g., "standard:edit-distance" or "fastga:NA")
# all_records_mode: 0 = per-record, 1 = all-records
test_configuration() {
    local config="$1"
    local first_strategy="$2"
    local second_strategy="${3:-$2}"  # Default to first_strategy if not provided (single mode)
    local all_records_mode="${4:-0}"  # 0=per-record, 1=all-records

    # Extract tp_type and metric from config
    local tp_type="${config%%:*}"
    local metric="${config##*:}"

    local tp_paf="$OUTPUT_DIR/${config}.tp.paf"

    # Build all-records flag for cigzip
    local ALL_RECORDS_FLAG=""
    local key_suffix=""
    if [ "$all_records_mode" -eq 1 ]; then
        ALL_RECORDS_FLAG="--all-records"
        key_suffix="+all-records"
    fi

    # Create key from config and strategies (add bgzip suffix if enabled)
    local key="${config}_${first_strategy}"
    if [ "$first_strategy" != "$second_strategy" ]; then
        key="${key}_${second_strategy}"
    fi
    key="${key}${key_suffix}"

    echo "    Testing $first_strategy/$second_strategy..."

    # Store tracepoint PAF size (only once per config)
    if [ -z "${TP_SIZE[$config]}" ]; then
        TP_SIZE[$config]=$(file_size "$tp_paf")
    fi

    # Get type-specific max_complexity (100 for fastga, user-specified for others)
    local tp_max_complexity=$(get_max_complexity "$tp_type")

    # Get metric for cigzip (edit-distance for fastga since it's ignored)
    local cigzip_metric="$metric"
    if [ "$tp_type" = "fastga" ]; then
        cigzip_metric="edit-distance"
    fi

    # Compress - use cigzip for all modes
    if [ "$TEST_MODE" = "dual" ]; then
        # Use cigzip with dual strategies (first;second format)
        /usr/bin/time -v $CIGZIP compress -i "$tp_paf" -o "$OUTPUT_DIR/${key}.tpa" \
            --type "$tp_type" --max-complexity "$tp_max_complexity" \
            --complexity-metric "$cigzip_metric" --distance gap-affine --penalties 5,8,2 \
            --strategy "${first_strategy},3;${second_strategy},3" $ALL_RECORDS_FLAG 2>&1 | \
            tee "$OUTPUT_DIR/${key}_compress.log" >/dev/null
    else
        # Use cigzip for single/symmetric strategies
        local strategy_arg="$first_strategy"
        if [[ "$first_strategy" == *"-nocomp" ]]; then
            strategy_arg="${first_strategy},0"
        fi

        /usr/bin/time -v $CIGZIP compress -i "$tp_paf" -o "$OUTPUT_DIR/${key}.tpa" \
            --type "$tp_type" --max-complexity "$tp_max_complexity" \
            --complexity-metric "$cigzip_metric" --distance gap-affine --penalties 5,8,2 \
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

    # Decompress
    /usr/bin/time -v $CIGZIP decompress -i "$OUTPUT_DIR/${key}.tpa" \
        -o "$OUTPUT_DIR/${key}_decomp.paf" 2>&1 | tee "$OUTPUT_DIR/${key}_decompress.log" >/dev/null
    
    DECOMPRESS_TIME[$key]=$(grep "Elapsed (wall clock)" "$OUTPUT_DIR/${key}_decompress.log" | awk '{print $8}')
    DECOMPRESS_MEM[$key]=$(grep "Maximum resident set size" "$OUTPUT_DIR/${key}_decompress.log" | awk '{print $6}')
    
    # Verify
    orig_md5=$(cat "$tp_paf" | $NORMALIZE | md5sum | cut -d' ' -f1)
    decomp_md5=$(cat "$OUTPUT_DIR/${key}_decomp.paf" | $NORMALIZE | md5sum | cut -d' ' -f1)
    
    if [ "$orig_md5" = "$decomp_md5" ]; then
        VERIFIED[$key]="yes"
    else
        VERIFIED[$key]="no"
    fi
    
    # Seek Full: TpaReader with full open (header + index + string table)
    # Output: open_us seek_us decode_ratio valid_ratio
    local seek_full_result=$("$SEEK_BENCH_READER" "$OUTPUT_DIR/${key}.tpa" "$EXTRACTED" "$SEEK_POSITIONS" "$SEEK_ITERATIONS" "$tp_type" "$tp_paf" || echo "0 0 0 0")
    read -r seek_full_open seek_full_seek seek_full_decode seek_full_valid <<< "$seek_full_result"
    SEEK_FULL_OPEN[$key]="$seek_full_open"
    SEEK_FULL[$key]="$seek_full_seek"

    # Seek Direct: standalone functions with pre-computed offsets (fastest)
    # Supports both per-record mode (raw byte offsets) and all-records mode (virtual positions)
    # Output: open_us seek_us decode_ratio valid_ratio
    local seek_direct_result=$("$SEEK_BENCH_DIRECT" "$OUTPUT_DIR/${key}.tpa" "$EXTRACTED" "$SEEK_POSITIONS" "$SEEK_ITERATIONS" "$tp_type" "$tp_paf" || echo "0 0 0 0")
    read -r seek_direct_open seek_direct_seek seek_direct_decode seek_direct_valid <<< "$seek_direct_result"
    SEEK_DIRECT_OPEN[$key]="$seek_direct_open"
    SEEK_DIRECT[$key]="$seek_direct_seek"
    SEEK_DECODE_RATIO[$key]="${seek_direct_decode:-0}"
    SEEK_VALID_RATIO[$key]="${seek_direct_valid:-0}"

    # Determine pass/fail for logging
    local failure_reasons=()
    if [ "${VERIFIED[$key]}" != "yes" ]; then
        failure_reasons+=("verification:${orig_md5}->${decomp_md5}")
    fi
    local seek_ratio_value="${SEEK_DECODE_RATIO[$key]}"
    if ! printf '%s' "$seek_ratio_value" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
        failure_reasons+=("seek:non-numeric:${seek_ratio_value}")
    else
        if ! awk "BEGIN{exit !($seek_ratio_value == 1)}"; then
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
    fi
}

# TSV output function
# config: tp_type:metric (e.g., "standard:edit-distance" or "fastga:NA")
output_tsv_row() {
    local config="$1"
    local first_strategy="$2"
    local second_strategy="${3:-$2}"  # Default to first_strategy if not provided
    local all_records_mode="${4:-0}"  # 0=per-record, 1=all-records
    local tsv_file="$OUTPUT_DIR/results.tsv"

    # Extract tp_type and metric from config
    local tp_type="${config%%:*}"
    local metric="${config##*:}"

    # Build key suffix for all-records mode
    local key_suffix=""
    if [ "$all_records_mode" -eq 1 ]; then
        key_suffix="+all-records"
    fi

    # Create key from config and strategies (must match test_configuration)
    local key="${config}_${first_strategy}"
    if [ "$first_strategy" != "$second_strategy" ]; then
        key="${key}_${second_strategy}"
    fi
    key="${key}${key_suffix}"

    # Strategy label for display (includes all-records mode suffix)
    local strategy_label="$first_strategy"
    if [ "$first_strategy" != "$second_strategy" ]; then
        strategy_label="${first_strategy}/${second_strategy}"
    fi
    strategy_label="${strategy_label}${key_suffix}"

    # Calculate ratios with guards to avoid division by zero or empty values
    local tp_size_bytes=${TP_SIZE[$config]:-0}
    local tpa_size_bytes=${COMPRESS_SIZE[$key]:-0}

    local ratio_orig_to_tp
    ratio_orig_to_tp=$(safe_ratio "$SIZE" "$tp_size_bytes" 3)

    local ratio_tp_to_tpa
    ratio_tp_to_tpa=$(safe_ratio "$tp_size_bytes" "$tpa_size_bytes" 3)

    local ratio_orig_to_tpa
    ratio_orig_to_tpa=$(safe_ratio "$SIZE" "$tpa_size_bytes" 3)

    # Convert times to seconds
    local encode_time_sec=$(time_to_seconds "$(normalize_time_field "${ENCODE_TIME[$config]}")")
    local compress_time_sec=$(time_to_seconds "$(normalize_time_field "${COMPRESS_TIME[$key]}")")
    local decompress_time_sec=$(time_to_seconds "$(normalize_time_field "${DECOMPRESS_TIME[$key]}")")

    # Convert memory to MB
    local encode_mem_mb=$(kb_to_mb "$(normalize_time_field "${ENCODE_MEM[$config]}")")
    local compress_mem_mb=$(kb_to_mb "$(normalize_time_field "${COMPRESS_MEM[$key]}")")
    local decompress_mem_mb=$(kb_to_mb "$(normalize_time_field "${DECOMPRESS_MEM[$key]}")")

    # Verification status
    local verified="${VERIFIED[$key]}"

    # Seek ratios fallback
    local seek_decode=${SEEK_DECODE_RATIO[$key]:-0}
    local seek_valid=${SEEK_VALID_RATIO[$key]:-0}

    # Dataset name from input file
    local dataset_name=$(basename "$INPUT_PAF" .paf.gz | sed 's/.paf$//')

    # Get type-specific max_complexity (100 for fastga)
    local tp_max_complexity=$(get_max_complexity "$tp_type")

    # Output TSV row (32 columns)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset_name" \
        "$INPUT_TYPE" \
        "$SIZE" \
        "$EXTRACTED" \
        "$tp_type" \
        "$encode_time_sec" \
        "$encode_mem_mb" \
        "$tp_size_bytes" \
        "$tp_max_complexity" \
        "$metric" \
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
        "$verified" \
        "$SEEK_POSITIONS" \
        "$SEEK_ITERATIONS" \
        "${SEEK_FULL_OPEN[$key]}" \
        "${SEEK_FULL[$key]}" \
        "${SEEK_DIRECT_OPEN[$key]}" \
        "${SEEK_DIRECT[$key]}" \
        "$seek_decode" \
        "$seek_valid" \
        >> "$tsv_file"
}

# TSV output function for BGZIP baseline
# config: tp_type:metric (e.g., "standard:edit-distance" or "fastga:NA")
output_bgzip_baseline_row() {
    local config="$1"
    local tsv_file="$OUTPUT_DIR/results.tsv"

    # Extract tp_type and metric from config
    local tp_type="${config%%:*}"
    local metric="${config##*:}"

    # Calculate ratios
    local tp_size_bytes=${TP_SIZE[$config]:-0}
    local bgzip_size_bytes=${BGZIP_SIZE[$config]:-0}

    local ratio_orig_to_tp
    ratio_orig_to_tp=$(safe_ratio "$SIZE" "$tp_size_bytes" 3)

    local ratio_tp_to_bgzip
    ratio_tp_to_bgzip=$(safe_ratio "$tp_size_bytes" "$bgzip_size_bytes" 3)

    local ratio_orig_to_bgzip
    ratio_orig_to_bgzip=$(safe_ratio "$SIZE" "$bgzip_size_bytes" 3)

    # Convert times to seconds
    local encode_time_sec=$(time_to_seconds "$(normalize_time_field "${ENCODE_TIME[$config]}")")
    local bgzip_time_sec=$(time_to_seconds "$(normalize_time_field "${BGZIP_TIME[$config]}")")

    # Convert memory to MB
    local encode_mem_mb=$(kb_to_mb "$(normalize_time_field "${ENCODE_MEM[$config]}")")
    local bgzip_mem_mb=$(kb_to_mb "$(normalize_time_field "${BGZIP_MEM[$config]}")")

    # Verification status - based on seek benchmark validity
    local seek_valid=${BGZIP_SEEK_VALID[$config]:-0}
    local verified="no"
    # Check if valid ratio == 1.0 (all seeks validated correctly)
    if awk "BEGIN{exit !($seek_valid == 1.0)}" 2>/dev/null; then
        verified="yes"
    fi

    # Dataset name from input file
    local dataset_name=$(basename "$INPUT_PAF" .paf.gz | sed 's/.paf$//')

    # Output TSV row (34 columns)
    # For BGZIP baseline:
    #   - compression_strategy: "bgzip-only"
    #   - strategy_first/second: "bgzip"
    #   - layer_first/second: "bgzip"
    #   - tpa_size_bytes column contains BGZIP compressed size
    #   - seek_full_*: NA (no high-level reader abstraction for BGZIP)
    #   - seek_direct_*: BGZIP seek time (direct BGZF virtual position seeking)

    # Get type-specific max_complexity (100 for fastga)
    local tp_max_complexity=$(get_max_complexity "$tp_type")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$dataset_name" \
        "$INPUT_TYPE" \
        "$SIZE" \
        "$EXTRACTED" \
        "$tp_type" \
        "$encode_time_sec" \
        "$encode_mem_mb" \
        "$tp_size_bytes" \
        "$tp_max_complexity" \
        "$metric" \
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
        "NA" \
        "NA" \
        "$verified" \
        "$SEEK_POSITIONS" \
        "$SEEK_ITERATIONS" \
        "NA" \
        "NA" \
        "${BGZIP_OPEN_TIME[$config]:-0}" \
        "${BGZIP_SEEK_AVG[$config]:-0}" \
        "${BGZIP_SEEK_DECODE[$config]:-0}" \
        "$seek_valid" \
        >> "$tsv_file"
}

# Initialize TSV file with header
TSV_FILE="$OUTPUT_DIR/results.tsv"
cat > "$TSV_FILE" << TSV_HEADER
dataset_name	dataset_type	original_size_bytes	num_records	encoding_type	encoding_runtime_sec	encoding_memory_mb	tp_file_size_bytes	max_complexity	complexity_metric	compression_strategy	strategy_first	strategy_second	compression_layer_first	compression_layer_second	compression_runtime_sec	compression_memory_mb	tpa_size_bytes	ratio_orig_to_tp	ratio_tp_to_tpa	ratio_orig_to_tpa	decompression_runtime_sec	decompression_memory_mb	verification_passed	positions_tested	iterations_per_position	full_open_avg_us	full_seek_avg_us	direct_open_avg_us	direct_seek_avg_us	seek_decode_ratio	seek_valid_ratio
TSV_HEADER

# Run all tests
for CONFIG in "${ENCODING_CONFIGS[@]}"; do
    # Extract tp_type and metric from config
    TP_TYPE="${CONFIG%%:*}"
    METRIC="${CONFIG##*:}"

    echo "═══════════════════════════════════════════════════"
    echo "Testing: ${TP_TYPE^^} (metric: ${METRIC})"
    echo "═══════════════════════════════════════════════════"

    # Output BGZIP baseline row first (if bgzip was available)
    if [ -n "${BGZIP_SIZE[$CONFIG]:-}" ] && [ "${BGZIP_SIZE[$CONFIG]}" -gt 0 ]; then
        echo "  Adding BGZIP baseline to results..."
        output_bgzip_baseline_row "$CONFIG"
    fi

    if [ "$TEST_MODE" = "dual" ]; then
        # Test all first×second×layer combinations
        total_combos=$((${#BASE_STRATEGIES[@]} * ${#LAYER_SUFFIXES[@]} * ${#BASE_STRATEGIES[@]} * ${#LAYER_SUFFIXES[@]}))
        combo_count=0
        echo "Testing $total_combos dual strategy combinations ((${#BASE_STRATEGIES[@]} strategies × ${#LAYER_SUFFIXES[@]} layers)^2) with $THREADS threads..."

        # Function to run a single test (for parallel execution)
        run_test() {
            local config="$1"
            local first_full="$2"
            local second_full="$3"
            local temp_tsv="$4"

            test_configuration "$config" "$first_full" "$second_full"
            output_tsv_row "$config" "$first_full" "$second_full" >> "$temp_tsv"
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
                    run_test "$CONFIG" "$FIRST_FULL" "$SECOND_FULL" "$TEMP_TSV" &

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

        echo ""
        echo "Completed ${total_combos} explicit dual combinations"
    elif [[ "$TEST_MODE" == auto:* ]]; then
        # Extract sample size from TEST_MODE (format: auto:ROWS)
        AUTO_SAMPLE_SIZE="${TEST_MODE#auto:}"
        AUTO_STRATEGY="benchmark,3,${AUTO_SAMPLE_SIZE}"

        if [ "$AUTO_SAMPLE_SIZE" -eq 0 ]; then
            echo "Testing benchmark mode with full file analysis..."
        else
            echo "Testing benchmark mode with ${AUTO_SAMPLE_SIZE}-record sampling..."
        fi

        echo ""
        echo "Testing ${AUTO_STRATEGY} (selects best per stream from $((${#BASE_STRATEGIES[@]} * ${#LAYER_SUFFIXES[@]})) combinations)..."
        test_configuration "$CONFIG" "$AUTO_STRATEGY"

        # Extract the strategies that automatic mode selected from the TPA header
        auto_tpa="$OUTPUT_DIR/${CONFIG}_${AUTO_STRATEGY}.tpa"
        selected_strategies=$("$REPO_DIR/target/release/tpa-view" --strategies "$auto_tpa")
        first_selected=$(echo "$selected_strategies" | cut -f1)
        second_selected=$(echo "$selected_strategies" | cut -f2)
        first_layer_selected=$(echo "$selected_strategies" | cut -f3)
        second_layer_selected=$(echo "$selected_strategies" | cut -f4)

        echo "  Selected: ${first_selected}[${first_layer_selected}]/${second_selected}[${second_layer_selected}]"

        # Output TSV row for the automatic run (per-record mode)
        output_tsv_row "$CONFIG" "$AUTO_STRATEGY" "$AUTO_STRATEGY" 0
        echo ""
        echo "Completed benchmark strategy test (per-record mode)"

        # Test all-records mode (--all-records flag, header/string table plain, records BGZIP-compressed)
        echo ""
        echo "Testing ${AUTO_STRATEGY}+all-records (all-records mode)..."
        test_configuration "$CONFIG" "$AUTO_STRATEGY" "$AUTO_STRATEGY" 1

        # Extract the strategies for all-records mode
        auto_all_records_tpa="$OUTPUT_DIR/${CONFIG}_${AUTO_STRATEGY}+all-records.tpa"
        all_records_selected_strategies=$("$REPO_DIR/target/release/tpa-view" --strategies "$auto_all_records_tpa")
        all_records_first_selected=$(echo "$all_records_selected_strategies" | cut -f1)
        all_records_second_selected=$(echo "$all_records_selected_strategies" | cut -f2)

        echo "  Selected: ${all_records_first_selected}/${all_records_second_selected} [all-records]"

        # Output TSV row for all-records mode
        output_tsv_row "$CONFIG" "$AUTO_STRATEGY" "$AUTO_STRATEGY" 1
        echo ""
        echo "Completed all-records mode test"
    else
        # Single mode: test each strategy symmetrically (first==second)
        for STRATEGY in "${STRATEGIES[@]}"; do
            test_configuration "$CONFIG" "$STRATEGY"
            output_tsv_row "$CONFIG" "$STRATEGY"
        done
    fi
    echo ""
done

echo "========================================="
echo "Tests complete!"
echo "TSV Data: $TSV_FILE"
echo "========================================="
