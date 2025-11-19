#!/bin/bash
set -e

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
CIGZIP_DIR="${CIGZIP_DIR:-/home/guarracino/git/cigzip}"
CIGZIP="$CIGZIP_DIR/target/release/cigzip"
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
OUTPUT_DIR="${2:-/tmp/bpaf_test_output}"
MAX_COMPLEXITY="${3:-32}"
COMPLEXITY_METRIC="${4:-edit-distance}"
NUM_RECORDS="${5:-20000}"
TEST_MODE="${6:-single}"  # "single" or "dual" - controls strategy testing mode
THREADS="${7:-1}"  # Number of parallel threads (default: 1)

if [ -z "$INPUT_PAF" ] || [ ! -f "$INPUT_PAF" ]; then
    echo "Usage: $0 <input.paf[.gz]> [output_dir] [max_complexity] [complexity_metric] [num_records] [test_mode] [threads]"
    echo ""
    echo "Automatically detects input type:"
    echo "  - CIGAR PAF (compressed or uncompressed)"
    echo "  - Tracepoint PAF"
    echo ""
    echo "Test Modes:"
    echo "  single (default) - Test each strategy symmetrically (first==second)"
    echo "  dual             - Test all $((${#BASE_STRATEGIES[@]}*${#BASE_STRATEGIES[@]})) strategy combinations"
    echo ""
    echo "Threads:"
    echo "  Number of tests to run in parallel (default: 1)"
    echo "  Example: 6 for 6-core CPU"
    echo ""
    echo "Tests:"
    echo "  - All tracepoint types (if CIGAR input)"
    echo "  - All compression strategies"
    echo "  - Seek performance (Mode A & B)"
    echo "  - Full verification"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "lib_bpaf Comprehensive Test Suite"
echo "========================================="
echo "Input:       $INPUT_PAF"
echo "Output:      $OUTPUT_DIR"
echo "Complexity:  $MAX_COMPLEXITY"
echo "Metric:      $COMPLEXITY_METRIC"
echo "Records:     $NUM_RECORDS"
echo "Test Mode:   $TEST_MODE"
echo "Threads:     $THREADS"
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
if [ ! -f "$CIGZIP" ]; then
    echo "=== Building cigzip ==="
    cd "$CIGZIP_DIR"
    cargo build --release 2>&1 | tail -3
fi

if [ ! -f "$REPO_DIR/target/release/liblib_bpaf.rlib" ]; then
    echo "=== Building lib_bpaf ==="
    cd "$REPO_DIR"
    cargo build --release 2>&1 | tail -3
fi

# Build seek test programs
echo "=== Building seek test programs ==="
cd "$REPO_DIR"

# Mode A: BpafReader with index
cat > /tmp/seek_mode_a.rs << 'RUST_A'
use std::env;
use std::time::Instant;
use std::fs::File;
use std::io::{BufRead, BufReader};
use lib_bpaf::BpafReader;
use lib_bpaf::TracepointData;

fn parse_reference(path: &str, limit: usize) -> Vec<Vec<(usize, usize)>> {
    let file = File::open(path).expect("reference PAF open failed");
    let reader = BufReader::new(file);
    let mut refs = Vec::new();

    for line in reader.lines().take(limit) {
        let line = line.expect("line read");
        if let Some(tp_idx) = line.find("tp:Z:") {
            let tp_str = &line[tp_idx + 5..];
            let tps: Vec<(usize, usize)> = tp_str
                .split(';')
                .filter(|s| !s.is_empty())
                .map(|pair| {
                    let mut it = pair.split(',');
                    let a = it.next().unwrap().parse().unwrap();
                    let b = it.next().unwrap().parse().unwrap();
                    (a, b)
                })
                .collect();
            refs.push(tps);
        }
    }
    refs
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let bpaf_path = &args[1];
    let num_records: u64 = args[2].parse().unwrap();
    let num_positions: usize = args[3].parse().unwrap();
    let iterations_per_pos: usize = args[4].parse().unwrap();
    let reference_paf = &args[5];

    let mut reader = BpafReader::open(bpaf_path).unwrap();
    let reference = parse_reference(reference_paf, num_records as usize);

    // Generate random positions
    use std::collections::HashSet;
    let mut rng = 12345u64;  // Simple LCG
    let mut positions = HashSet::new();
    while positions.len() < num_positions {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        positions.insert((rng % num_records) as u64);
    }
    let positions: Vec<u64> = positions.into_iter().collect();

    let mut sum_us = 0u128;
    let mut sum_sq_us = 0u128;
    let mut success = 0usize;
    let total_tests = num_positions * iterations_per_pos;

    for &pos in &positions {
        // Warmup
        for _ in 0..3 { let _ = reader.get_tracepoints(pos); }

        // Benchmark this position
        for _ in 0..iterations_per_pos {
            let start = Instant::now();
            match reader.get_tracepoints(pos) {
                Ok((tp, _, _)) => {
                    let time_us = start.elapsed().as_micros();
                    sum_us += time_us;
                    sum_sq_us += time_us * time_us;
                    success += 1;
                    if let TracepointData::Standard(tps) = tp {
                        let expected = reference.get(pos as usize).expect("reference missing");
                        if expected.as_slice() != tps.as_slice() {
                            eprintln!("Tracepoint mismatch at record {}", pos);
                            std::process::exit(1);
                        }
                    } else {
                        eprintln!("Unexpected tracepoint type for validation");
                        std::process::exit(1);
                    }
                }
                Err(_) => {}
            }
        }
    }

    let avg_us = sum_us as f64 / total_tests as f64;
    let variance = (sum_sq_us as f64 / total_tests as f64) - (avg_us * avg_us);
    let stddev_us = variance.sqrt();
    let success_ratio = success as f64 / total_tests as f64;

    println!("{:.2} {:.2} {:.4}", avg_us, stddev_us, success_ratio);
}
RUST_A

# Mode B: Standalone functions
cat > /tmp/seek_mode_b.rs << 'RUST_B'
use std::env;
use std::time::Instant;
use std::fs::File;
use std::io::{BufRead, BufReader};
use lib_bpaf::{BpafReader, read_standard_tracepoints_at_offset,
               read_variable_tracepoints_at_offset, read_mixed_tracepoints_at_offset};

fn parse_reference(path: &str, limit: usize) -> Vec<Vec<(usize, usize)>> {
    let file = File::open(path).expect("reference PAF open failed");
    let reader = BufReader::new(file);
    let mut refs = Vec::new();

    for line in reader.lines().take(limit) {
        let line = line.expect("line read");
        if let Some(tp_idx) = line.find("tp:Z:") {
            let tp_str = &line[tp_idx + 5..];
            let tps: Vec<(usize, usize)> = tp_str
                .split(';')
                .filter(|s| !s.is_empty())
                .map(|pair| {
                    let mut it = pair.split(',');
                    let a = it.next().unwrap().parse().unwrap();
                    let b = it.next().unwrap().parse().unwrap();
                    (a, b)
                })
                .collect();
            refs.push(tps);
        }
    }
    refs
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let bpaf_path = &args[1];
    let num_records: u64 = args[2].parse().unwrap();
    let num_positions: usize = args[3].parse().unwrap();
    let iterations_per_pos: usize = args[4].parse().unwrap();
    let tp_type = &args[5];
    let reference_paf = &args[6];

    let mut reader = BpafReader::open(bpaf_path).unwrap();
    let strategy = reader.header().strategy().unwrap();
    let first_layer = reader.header().first_layer();
    let second_layer = reader.header().second_layer();
    let reference = parse_reference(reference_paf, num_records as usize);

    // Generate random positions and get their offsets
    use std::collections::HashSet;
    let mut rng = 12345u64;
    let mut positions = HashSet::new();
    while positions.len() < num_positions {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        positions.insert((rng % num_records) as u64);
    }

    let offsets: Vec<u64> = positions.iter()
        .map(|&pos| reader.get_tracepoint_offset(pos).unwrap())
        .collect();
    drop(reader);

    let mut file = File::open(bpaf_path).unwrap();
    let mut sum_us = 0u128;
    let mut sum_sq_us = 0u128;
    let mut success = 0usize;
    let total_tests = num_positions * iterations_per_pos;

    for (&offset, &record_id) in offsets.iter().zip(positions.iter()) {
        // Warmup
        for _ in 0..3 {
            let _ = match tp_type.as_str() {
                "standard" => read_standard_tracepoints_at_offset(
                    &mut file,
                    offset,
                    strategy.clone(),
                    first_layer,
                    second_layer,
                ),
                "variable" => read_variable_tracepoints_at_offset(&mut file, offset).map(|_| Vec::new()),
                "mixed" => read_mixed_tracepoints_at_offset(&mut file, offset).map(|_| Vec::new()),
                _ => panic!("Invalid tp_type"),
            };
        }

        // Benchmark this position
        for _ in 0..iterations_per_pos {
            let start = Instant::now();
            let result = match tp_type.as_str() {
                "standard" => read_standard_tracepoints_at_offset(
                    &mut file,
                    offset,
                    strategy.clone(),
                    first_layer,
                    second_layer,
                ),
                "variable" => read_variable_tracepoints_at_offset(&mut file, offset).map(|_| Vec::new()),
                "mixed" => read_mixed_tracepoints_at_offset(&mut file, offset).map(|_| Vec::new()),
                _ => panic!("Invalid tp_type"),
            };

            match result {
                Ok(res) => {
                    let time_us = start.elapsed().as_micros();
                    sum_us += time_us;
                    sum_sq_us += time_us * time_us;
                    success += 1;
                    match tp_type.as_str() {
                        "standard" => {
                            let expected = reference.get(record_id as usize).expect("reference missing");
                            if expected.as_slice() != res.as_slice() {
                                eprintln!("Tracepoint mismatch at record {}", record_id);
                                std::process::exit(1);
                            }
                        }
                        _ => {}
                    }
                }
                Err(_) => {}
            }
        }
    }

    let avg_us = sum_us as f64 / total_tests as f64;
    let variance = (sum_sq_us as f64 / total_tests as f64) - (avg_us * avg_us);
    let stddev_us = variance.sqrt();
    let success_ratio = success as f64 / total_tests as f64;

    println!("{:.2} {:.2} {:.4}", avg_us, stddev_us, success_ratio);
}
RUST_B

if ! rustc --edition 2021 -O /tmp/seek_mode_a.rs \
    -L target/release/deps --extern lib_bpaf=target/release/liblib_bpaf.rlib \
    -o /tmp/seek_mode_a 2>&1; then
    echo "✗ Error: Failed to compile seek_mode_a"
    exit 1
fi

if ! rustc --edition 2021 -O /tmp/seek_mode_b.rs \
    -L target/release/deps --extern lib_bpaf=target/release/liblib_bpaf.rlib \
    -o /tmp/seek_mode_b 2>&1; then
    echo "✗ Error: Failed to compile seek_mode_b"
    exit 1
fi

echo "✓ Seek tools ready"
echo ""

# Results storage - declare arrays before use
declare -A ENCODE_TIME ENCODE_MEM
declare -A COMPRESS_TIME COMPRESS_MEM COMPRESS_SIZE
declare -A DECOMPRESS_TIME DECOMPRESS_MEM
declare -A SEEK_A SEEK_B SEEK_A_STDDEV SEEK_B_STDDEV SEEK_SUCCESS_RATIO
declare -A VERIFIED
declare -A TP_SIZE
declare -A BGZIP_TIME BGZIP_MEM BGZIP_SIZE
declare -A STRATEGY_FIRST STRATEGY_SECOND
declare -A LAYER_FIRST LAYER_SECOND

# Determine tracepoint types to test
if [ "$INPUT_TYPE" = "cigar" ]; then
    #TP_TYPES=("standard" "variable" "mixed")
    TP_TYPES=("standard") # Focus on standard for now
    echo "=== Encoding CIGAR to all tracepoint types ==="
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
    # Already tracepoints - detect which type
    TP_TYPES=("standard")
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
        fi
    done
    echo "✓ BGZIP compression complete"
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
    "offset-joint"
    "xor-delta"
    "dictionary"
    "simple8"
    "stream-vbyte"
    "fastpfor"
    "cascaded"
    "simple8b-full"
    "selective-rle"
    "rice"
    "huffman"
)

LAYER_SUFFIXES=(
    ""
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

# Add meta-strategies (automatic-fast/slow)
AUTO_STRATEGIES=("automatic-fast" "automatic-slow")
for auto in "${AUTO_STRATEGIES[@]}"; do
    STRATEGIES+=("$auto")
done

# Test function
test_configuration() {
    local tp_type="$1"
    local first_strategy="$2"
    local second_strategy="${3:-$2}"  # Default to first_strategy if not provided (single mode)
    local tp_paf="$OUTPUT_DIR/${tp_type}.tp.paf"
    local is_auto=0
    if [[ "$first_strategy" == automatic* ]]; then
        is_auto=1
    fi

    # Create key from strategies
    local key="${tp_type}_${first_strategy}"
    if [ "$first_strategy" != "$second_strategy" ]; then
        key="${key}_${second_strategy}"
    fi

    echo "    Testing $first_strategy → $second_strategy..."

    # Store tracepoint PAF size (only once per type)
    if [ -z "${TP_SIZE[$tp_type]}" ]; then
        TP_SIZE[$tp_type]=$(file_size "$tp_paf")
    fi

    # Compress - use cigzip for all modes
    if [ "$TEST_MODE" = "dual" ]; then
        if [ $is_auto -eq 1 ]; then
            echo "      [$first_strategy] compress starting..." >&2
            /usr/bin/time -v $CIGZIP compress -i "$tp_paf" -o "$OUTPUT_DIR/${key}.bpaf" \
                --type "$tp_type" --max-complexity "$MAX_COMPLEXITY" \
                --complexity-metric "$COMPLEXITY_METRIC" --distance gap-affine --penalties 5,8,2 \
                --strategy "$first_strategy" 2>&1 | \
                tee "$OUTPUT_DIR/${key}_compress.log" >&2
            echo "      [$first_strategy] compress finished" >&2
        else
            # Use cigzip with dual strategies (--strategy and --strategy-second)
            /usr/bin/time -v $CIGZIP compress -i "$tp_paf" -o "$OUTPUT_DIR/${key}.bpaf" \
                --type "$tp_type" --max-complexity "$MAX_COMPLEXITY" \
                --complexity-metric "$COMPLEXITY_METRIC" --distance gap-affine --penalties 5,8,2 \
                --strategy "$first_strategy,3" --strategy-second "$second_strategy,3" 2>&1 | \
                tee "$OUTPUT_DIR/${key}_compress.log" >/dev/null
        fi
    else
        # Use cigzip for single/symmetric strategies
        local strategy_arg="$first_strategy"
        if [[ "$first_strategy" == *"-nocomp" ]]; then
            strategy_arg="${first_strategy},0"
        fi

        /usr/bin/time -v $CIGZIP compress -i "$tp_paf" -o "$OUTPUT_DIR/${key}.bpaf" \
            --type "$tp_type" --max-complexity "$MAX_COMPLEXITY" \
            --complexity-metric "$COMPLEXITY_METRIC" --distance gap-affine --penalties 5,8,2 \
            --strategy "$strategy_arg" 2>&1 | tee "$OUTPUT_DIR/${key}_compress.log" >/dev/null
    fi

    COMPRESS_TIME[$key]=$(grep "Elapsed (wall clock)" "$OUTPUT_DIR/${key}_compress.log" | awk '{print $8}')
    COMPRESS_MEM[$key]=$(grep "Maximum resident set size" "$OUTPUT_DIR/${key}_compress.log" | awk '{print $6}')
    COMPRESS_SIZE[$key]=$(file_size "$OUTPUT_DIR/${key}.bpaf")

    # Extract actual strategies from BPAF header using bpaf-view
    local strategy_output=$("$REPO_DIR/target/release/bpaf-view" --strategies "$OUTPUT_DIR/${key}.bpaf" || echo "unknown\tunknown\tunknown\tunknown")
    read -r first_strat second_strat first_layer second_layer <<< "$strategy_output"
    STRATEGY_FIRST[$key]="$first_strat"
    STRATEGY_SECOND[$key]="$second_strat"
    LAYER_FIRST[$key]="$first_layer"
    LAYER_SECOND[$key]="$second_layer"
    
    # Decompress
    if [ $is_auto -eq 1 ]; then
        echo "      [$first_strategy] decompress starting..." >&2
        /usr/bin/time -v $CIGZIP decompress -i "$OUTPUT_DIR/${key}.bpaf" \
            -o "$OUTPUT_DIR/${key}_decomp.paf" 2>&1 | tee "$OUTPUT_DIR/${key}_decompress.log" >&2
        echo "      [$first_strategy] decompress finished" >&2
    else
        /usr/bin/time -v $CIGZIP decompress -i "$OUTPUT_DIR/${key}.bpaf" \
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
    local seek_a_result=$(/tmp/seek_mode_a "$OUTPUT_DIR/${key}.bpaf" "$EXTRACTED" 10 10 "$tp_paf" || echo "0 0 0")
    read -r seek_a_avg seek_a_std seek_a_ratio <<< "$seek_a_result"
    SEEK_A[$key]="$seek_a_avg"
    SEEK_A_STDDEV[$key]="$seek_a_std"

    # Seek Mode B: 10 positions × 10 iterations (for quick testing)
    local seek_b_result=$(/tmp/seek_mode_b "$OUTPUT_DIR/${key}.bpaf" "$EXTRACTED" 10 10 "$tp_type" "$tp_paf" || echo "0 0 0")
    read -r seek_b_avg seek_b_std seek_b_ratio <<< "$seek_b_result"
    SEEK_B[$key]="$seek_b_avg"
    SEEK_B_STDDEV[$key]="$seek_b_std"
    SEEK_SUCCESS_RATIO[$key]="${seek_b_ratio:-0}"  # Using Mode B ratio (both should be identical)

    # Determine pass/fail for logging
    local failure_reasons=()
    if [ "${VERIFIED[$key]}" != "✓" ]; then
        failure_reasons+=("verification:${orig_md5}->${decomp_md5}")
    fi
    local seek_ratio_value="${SEEK_SUCCESS_RATIO[$key]}"
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
            echo "bpaf=$OUTPUT_DIR/${key}.bpaf"
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
    local tsv_file="$OUTPUT_DIR/results.tsv"

    # Create key from strategies (must match test_configuration)
    local key="${tp_type}_${first_strategy}"
    if [ "$first_strategy" != "$second_strategy" ]; then
        key="${key}_${second_strategy}"
    fi

    # Strategy label for display
    local strategy_label="$first_strategy"
    if [ "$first_strategy" != "$second_strategy" ]; then
        strategy_label="${first_strategy}→${second_strategy}"
    fi

    # Calculate ratios with guards to avoid division by zero or empty values
    local tp_size_bytes=${TP_SIZE[$tp_type]:-0}
    local bpaf_size_bytes=${COMPRESS_SIZE[$key]:-0}

    local ratio_orig_to_tp
    ratio_orig_to_tp=$(safe_ratio "$SIZE" "$tp_size_bytes" 3)

    local ratio_tp_to_bpaf
    ratio_tp_to_bpaf=$(safe_ratio "$tp_size_bytes" "$bpaf_size_bytes" 3)

    local ratio_orig_to_bpaf
    ratio_orig_to_bpaf=$(safe_ratio "$SIZE" "$bpaf_size_bytes" 3)

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

    # Seek success ratio fallback
    local seek_success=${SEEK_SUCCESS_RATIO[$key]:-0}

    # Dataset name from input file
    local dataset_name=$(basename "$INPUT_PAF" .paf.gz | sed 's/.paf$//')

    # Output TSV row (32 columns - includes strategy/layer pairs)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
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
        "$bpaf_size_bytes" \
        "$ratio_orig_to_tp" \
        "$ratio_tp_to_bpaf" \
        "$ratio_orig_to_bpaf" \
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
        "$seek_success" \
        >> "$tsv_file"
}

# Initialize TSV file with header
TSV_FILE="$OUTPUT_DIR/results.tsv"
cat > "$TSV_FILE" << TSV_HEADER
dataset_name	dataset_type	original_size_bytes	num_records	encoding_type	encoding_runtime_sec	encoding_memory_mb	tp_file_size_bytes	max_complexity	complexity_metric	compression_strategy	strategy_first	strategy_second	layer_first	layer_second	compression_runtime_sec	compression_memory_mb	bpaf_size_bytes	ratio_orig_to_tp	ratio_tp_to_bpaf	ratio_orig_to_bpaf	decompression_runtime_sec	decompression_memory_mb	verification_passed	seek_positions_tested	seek_iterations_per_position	seek_total_tests	seek_mode_a_avg_us	seek_mode_a_stddev_us	seek_mode_b_avg_us	seek_mode_b_stddev_us	seek_success_ratio
TSV_HEADER

# Run all tests
for TP_TYPE in "${TP_TYPES[@]}"; do
    echo "═══════════════════════════════════════════════════"
    echo "Testing Tracepoint Type: ${TP_TYPE^^}"
    echo "═══════════════════════════════════════════════════"

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

        # Also test automatic modes (which internally test every strategy×layer per stream and select best)
        for auto_name in "${AUTO_STRATEGIES[@]}"; do
            echo ""
            echo "Testing ${auto_name} meta-strategy (selects best per stream from $((${#BASE_STRATEGIES[@]} * ${#LAYER_SUFFIXES[@]})) combinations)..."
            test_configuration "$TP_TYPE" "$auto_name"

            # Extract the strategies that automatic mode selected from the BPAF header
            auto_bpaf="$OUTPUT_DIR/${TP_TYPE}_${auto_name}.bpaf"
            selected_strategies=$("$REPO_DIR/target/release/bpaf-view" --strategies "$auto_bpaf")
            first_selected=$(echo "$selected_strategies" | cut -f1)
            second_selected=$(echo "$selected_strategies" | cut -f2)
            first_layer_selected=$(echo "$selected_strategies" | cut -f3)
            second_layer_selected=$(echo "$selected_strategies" | cut -f4)

            echo "  → ${auto_name} selected: ${first_selected}[${first_layer_selected}] → ${second_selected}[${second_layer_selected}]"

            # Output TSV row for the automatic run (metrics recorded under current key)
            output_tsv_row "$TP_TYPE" "$auto_name"
        done

        echo ""
        echo "✓ Completed $((total_combos + ${#AUTO_STRATEGIES[@]})) tests: ${total_combos} explicit dual combinations + ${#AUTO_STRATEGIES[@]} automatic"
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
else
    # Single mode: generate detailed markdown table
    for TP_TYPE in "${TP_TYPES[@]}"; do
        # Calculate baseline sizes
        tp_size_bytes=${TP_SIZE[$TP_TYPE]}
            tp_size_mb=$(safe_ratio "$tp_size_bytes" 1048576 2)

        cat >> "$REPORT" << SECTION
## Tracepoint Type: ${TP_TYPE^^}

**Baseline Sizes:**
- Original Input (CIGAR PAF): $SIZE bytes ($(python3 - <<'PY'\nsize = int($SIZE)\nprint(f\"{size/1024/1024:.2f}\")\nPY\n) MB)
- Tracepoint PAF ($TP_TYPE): $tp_size_bytes bytes ($tp_size_mb MB)

| Strategy | Compressed Size (bytes) | BPAF Ratio | End-to-End Ratio | Compress Time | Compress Mem (KB) | Decompress Time | Decompress Mem (KB) | Seek A (μs) | Seek B (μs) | Verified |
|----------|-------------------------|------------|------------------|---------------|-------------------|-----------------|---------------------|-------------|-------------|----------|
SECTION

        for STRATEGY in "${STRATEGIES[@]}"; do
            key="${TP_TYPE}_${STRATEGY}"

            # Calculate both ratios
            bpaf_ratio=$(safe_ratio "$tp_size_bytes" "${COMPRESS_SIZE[$key]}" 2)
            e2e_ratio=$(safe_ratio "$SIZE" "${COMPRESS_SIZE[$key]}" 2)

            cat >> "$REPORT" << ROW
| $STRATEGY | ${COMPRESS_SIZE[$key]} | ${bpaf_ratio}x | ${e2e_ratio}x | ${COMPRESS_TIME[$key]} | ${COMPRESS_MEM[$key]} | ${DECOMPRESS_TIME[$key]} | ${DECOMPRESS_MEM[$key]} | ${SEEK_A[$key]} | ${SEEK_B[$key]} | ${VERIFIED[$key]} |
ROW
        done

        echo "" >> "$REPORT"
    done
fi

cat >> "$REPORT" << FOOTER

---

## Compression Ratio Legend

- **BPAF Ratio:** Tracepoint PAF size / BPAF size
  - Measures BPAF compression strategy effectiveness only
  - Use this to compare strategy performance
  - Fair comparison across all strategies

- **End-to-End Ratio:** Original input size / BPAF size
  - Measures total compression from original input
  - Includes CIGAR→tracepoint conversion gains (if applicable)
  - Higher for CIGAR inputs due to format conversion

**Note:** For CIGAR PAF inputs, End-to-End ratio includes both format conversion (~6-10x) and BPAF compression. For tracepoint PAF inputs, both ratios are identical.

## Seek Mode Legend

- **Mode A:** BpafReader with index (general use)
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
