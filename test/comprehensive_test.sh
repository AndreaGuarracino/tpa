#!/bin/bash
set -e

# Comprehensive test for a single PAF file
# Automatically detects: CIGAR (compressed/uncompressed) or Tracepoint PAF
# Tests: all tracepoint types + all compression strategies + seek performance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CIGZIP_DIR="$(dirname "$REPO_DIR")/cigzip"
CIGZIP="$CIGZIP_DIR/target/release/cigzip"
NORMALIZE="$SCRIPT_DIR/normalize_paf.pl"

# Parameters
INPUT_PAF="${1}"
OUTPUT_DIR="${2:-/tmp/bpaf_test_output}"
MAX_COMPLEXITY="${3:-32}"
COMPLEXITY_METRIC="${4:-edit-distance}"
NUM_RECORDS="${5:-20000}"

if [ -z "$INPUT_PAF" ] || [ ! -f "$INPUT_PAF" ]; then
    echo "Usage: $0 <input.paf[.gz]> [output_dir] [max_complexity] [complexity_metric] [num_records]"
    echo ""
    echo "Automatically detects input type:"
    echo "  - CIGAR PAF (compressed or uncompressed)"
    echo "  - Tracepoint PAF"
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
echo "========================================="
echo ""

# Detect input type
echo "=== Detecting input type ==="
if [[ "$INPUT_PAF" == *.gz ]]; then
    FIRST_LINE=$(zcat "$INPUT_PAF" | head -1)
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

# Extract sample
echo "=== Extracting $NUM_RECORDS records ==="
if [ $IS_COMPRESSED -eq 1 ]; then
    zcat "$INPUT_PAF" | head -n "$NUM_RECORDS" > "$OUTPUT_DIR/input_sample.paf"
else
    head -n "$NUM_RECORDS" "$INPUT_PAF" > "$OUTPUT_DIR/input_sample.paf"
fi

EXTRACTED=$(wc -l < "$OUTPUT_DIR/input_sample.paf")
SIZE=$(stat -c%s "$OUTPUT_DIR/input_sample.paf" 2>/dev/null || stat -f%z "$OUTPUT_DIR/input_sample.paf")
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
use lib_bpaf::BpafReader;

fn main() {
    let args: Vec<String> = env::args().collect();
    let bpaf_path = &args[1];
    let record_id: u64 = args[2].parse().unwrap();
    let iterations: usize = args[3].parse().unwrap();

    let mut reader = BpafReader::open(bpaf_path).unwrap();
    
    // Warmup
    for _ in 0..5 { let _ = reader.get_tracepoints(record_id); }
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        reader.get_tracepoints(record_id).unwrap();
    }
    println!("{}", start.elapsed().as_micros() / iterations as u128);
}
RUST_A

# Mode B: Standalone functions
cat > /tmp/seek_mode_b.rs << 'RUST_B'
use std::env;
use std::time::Instant;
use std::fs::File;
use lib_bpaf::{BpafReader, read_standard_tracepoints_at_offset,
               read_variable_tracepoints_at_offset, read_mixed_tracepoints_at_offset};

fn main() {
    let args: Vec<String> = env::args().collect();
    let bpaf_path = &args[1];
    let record_id: u64 = args[2].parse().unwrap();
    let iterations: usize = args[3].parse().unwrap();
    let tp_type = &args[4];

    let mut reader = BpafReader::open(bpaf_path).unwrap();
    let offset = reader.get_tracepoint_offset(record_id).unwrap();
    let strategy = reader.header().strategy().unwrap();
    drop(reader);

    let mut file = File::open(bpaf_path).unwrap();
    
    // Warmup
    for _ in 0..3 {
        match tp_type.as_str() {
            "standard" => { let _ = read_standard_tracepoints_at_offset(&mut file, offset, strategy); }
            "variable" => { let _ = read_variable_tracepoints_at_offset(&mut file, offset); }
            "mixed" => { let _ = read_mixed_tracepoints_at_offset(&mut file, offset); }
            _ => panic!("Invalid tp_type"),
        }
    }
    
    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        match tp_type.as_str() {
            "standard" => { read_standard_tracepoints_at_offset(&mut file, offset, strategy).unwrap(); }
            "variable" => { read_variable_tracepoints_at_offset(&mut file, offset).unwrap(); }
            "mixed" => { read_mixed_tracepoints_at_offset(&mut file, offset).unwrap(); }
            _ => panic!("Invalid tp_type"),
        }
    }
    println!("{}", start.elapsed().as_micros() / iterations as u128);
}
RUST_B

rustc --edition 2021 -O /tmp/seek_mode_a.rs \
    -L target/release/deps --extern lib_bpaf=target/release/liblib_bpaf.rlib \
    -o /tmp/seek_mode_a 2>/dev/null

rustc --edition 2021 -O /tmp/seek_mode_b.rs \
    -L target/release/deps --extern lib_bpaf=target/release/liblib_bpaf.rlib \
    -o /tmp/seek_mode_b 2>/dev/null

echo "✓ Seek tools ready"
echo ""

# Determine tracepoint types to test
if [ "$INPUT_TYPE" = "cigar" ]; then
    TP_TYPES=("standard" "variable" "mixed")
    echo "=== Encoding CIGAR to all tracepoint types ==="
    for TP_TYPE in "${TP_TYPES[@]}"; do
        echo "  Encoding $TP_TYPE..."
        $CIGZIP encode --paf "$OUTPUT_DIR/input_sample.paf" --threads 1 --type "$TP_TYPE" \
            --max-complexity "$MAX_COMPLEXITY" --complexity-metric "$COMPLEXITY_METRIC" \
            > "$OUTPUT_DIR/${TP_TYPE}.tp.paf" 2>/dev/null
    done
    echo "✓ All encodings complete"
    echo ""
else
    # Already tracepoints - detect which type
    TP_TYPES=("standard")
    cp "$OUTPUT_DIR/input_sample.paf" "$OUTPUT_DIR/standard.tp.paf"
    echo "✓ Using existing tracepoint PAF"
    echo ""
fi

# Compression strategies
STRATEGIES=(
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
)

# Results storage
declare -A COMPRESS_TIME COMPRESS_MEM COMPRESS_SIZE
declare -A DECOMPRESS_TIME DECOMPRESS_MEM
declare -A SEEK_A SEEK_B
declare -A VERIFIED

# Test function
test_configuration() {
    local tp_type="$1"
    local strategy="$2"
    local tp_paf="$OUTPUT_DIR/${tp_type}.tp.paf"
    local key="${tp_type}_${strategy}"
    
    echo "    Testing $strategy..."
    
    # Compress
    /usr/bin/time -v $CIGZIP compress -i "$tp_paf" -o "$OUTPUT_DIR/${key}.bpaf" \
        --type "$tp_type" --max-complexity "$MAX_COMPLEXITY" \
        --complexity-metric "$COMPLEXITY_METRIC" --distance gap-affine --penalties 5,8,2 \
        --strategy "$strategy" 2>&1 | tee "$OUTPUT_DIR/${key}_compress.log" >/dev/null
    
    COMPRESS_TIME[$key]=$(grep "Elapsed (wall clock)" "$OUTPUT_DIR/${key}_compress.log" | awk '{print $8}')
    COMPRESS_MEM[$key]=$(grep "Maximum resident set size" "$OUTPUT_DIR/${key}_compress.log" | awk '{print $6}')
    COMPRESS_SIZE[$key]=$(stat -c%s "$OUTPUT_DIR/${key}.bpaf" 2>/dev/null || stat -f%z "$OUTPUT_DIR/${key}.bpaf")
    
    # Decompress
    /usr/bin/time -v $CIGZIP decompress -i "$OUTPUT_DIR/${key}.bpaf" \
        -o "$OUTPUT_DIR/${key}_decomp.paf" 2>&1 | tee "$OUTPUT_DIR/${key}_decompress.log" >/dev/null
    
    DECOMPRESS_TIME[$key]=$(grep "Elapsed (wall clock)" "$OUTPUT_DIR/${key}_decompress.log" | awk '{print $8}')
    DECOMPRESS_MEM[$key]=$(grep "Maximum resident set size" "$OUTPUT_DIR/${key}_decompress.log" | awk '{print $6}')
    
    # Verify
    local orig_md5=$($NORMALIZE "$tp_paf" | md5sum | cut -d' ' -f1)
    local decomp_md5=$($NORMALIZE "$OUTPUT_DIR/${key}_decomp.paf" | md5sum | cut -d' ' -f1)
    
    if [ "$orig_md5" = "$decomp_md5" ]; then
        VERIFIED[$key]="✓"
    else
        VERIFIED[$key]="✗"
    fi
    
    # Seek Mode A
    local total_a=0
    for i in $(seq 0 100 $((EXTRACTED - 1))); do
        local time_a=$(/tmp/seek_mode_a "$OUTPUT_DIR/${key}.bpaf" "$i" 50 2>/dev/null || echo "0")
        total_a=$((total_a + time_a))
    done
    SEEK_A[$key]=$(echo "scale=1; $total_a / 10" | bc -l)
    
    # Seek Mode B
    local total_b=0
    for i in $(seq 0 100 $((EXTRACTED - 1))); do
        local time_b=$(/tmp/seek_mode_b "$OUTPUT_DIR/${key}.bpaf" "$i" 50 "$tp_type" 2>/dev/null || echo "0")
        total_b=$((total_b + time_b))
    done
    SEEK_B[$key]=$(echo "scale=1; $total_b / 10" | bc -l)
    
    # Cleanup
    rm -f "$OUTPUT_DIR/${key}_compress.log" "$OUTPUT_DIR/${key}_decompress.log" "$OUTPUT_DIR/${key}_decomp.paf"
}

# Run all tests
for TP_TYPE in "${TP_TYPES[@]}"; do
    echo "═══════════════════════════════════════════════════"
    echo "Testing Tracepoint Type: ${TP_TYPE^^}"
    echo "═══════════════════════════════════════════════════"
    
    for STRATEGY in "${STRATEGIES[@]}"; do
        test_configuration "$TP_TYPE" "$STRATEGY"
    done
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

for TP_TYPE in "${TP_TYPES[@]}"; do
    cat >> "$REPORT" << SECTION
## Tracepoint Type: ${TP_TYPE^^}

| Strategy | Compress Time | Compress Mem (KB) | Size (bytes) | Ratio | Decompress Time | Decompress Mem (KB) | Seek A (μs) | Seek B (μs) | Verified |
|----------|---------------|-------------------|--------------|-------|-----------------|---------------------|-------------|-------------|----------|
SECTION

    for STRATEGY in "${STRATEGIES[@]}"; do
        key="${TP_TYPE}_${strategy}"
        ratio=$(echo "scale=2; $SIZE / ${COMPRESS_SIZE[$key]}" | bc -l 2>/dev/null || echo "N/A")
        
        cat >> "$REPORT" << ROW
| $STRATEGY | ${COMPRESS_TIME[$key]} | ${COMPRESS_MEM[$key]} | ${COMPRESS_SIZE[$key]} | ${ratio}x | ${DECOMPRESS_TIME[$key]} | ${DECOMPRESS_MEM[$key]} | ${SEEK_A[$key]} | ${SEEK_B[$key]} | ${VERIFIED[$key]} |
ROW
    done
    
    echo "" >> "$REPORT"
done

cat >> "$REPORT" << FOOTER

---

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
echo "Report: $REPORT"
echo "========================================="
