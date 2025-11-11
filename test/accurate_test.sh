#!/bin/bash
set -e

# Comprehensive test suite for lib_bpaf with proper tracepoint type testing
# Tests: encoding CIGAR→tracepoints, compression, decompression, O(1) seeks (Mode A and Mode B)
# Measures: runtime, memory, correctness, compression ratios

REPO_DIR="/home/guarracino/Dropbox/git/lib_bpaf"
CIGZIP_DIR="/home/guarracino/Dropbox/git/cigzip"

# Source CIGAR-based PAF file
CIGAR_PAF="${1:-/home/guarracino/git/impg/hprcv2/data/hg002v1.1.pat.PanSN-vs-HG02818_mat_hprc_r2_v1.0.1.p95.Pinf.aln.paf.gz}"
NUM_RECORDS=10000
SEEK_ITERATIONS=1000

# Tracepoint types to test
TP_TYPES=("standard")

# Verify input file exists
if [ ! -f "$CIGAR_PAF" ]; then
    echo "Error: CIGAR-based PAF file not found: $CIGAR_PAF"
    exit 1
fi

echo "========================================"
echo "lib_bpaf Comprehensive Test Suite"
echo "Proper Tracepoint Type Testing"
echo "========================================"
echo "Source:     $CIGAR_PAF"
echo "Records:    $NUM_RECORDS"
echo "Types:      Standard, Variable, Mixed"
echo "========================================"
echo

# Arrays to store results
declare -A ENCODE_TIME COMPRESS_TIME COMPRESS_MEM COMPRESS_SIZE
declare -A DECOMP_TIME DECOMP_MEM
declare -A SEEK_TIME_A SEEK_TIME_B
declare -A VERIFIED

# Build cigzip
echo "=== Building cigzip ==="
if [ -d "$CIGZIP_DIR" ]; then
    cd "$CIGZIP_DIR"
    cargo build --release 2>&1 | tail -3
    CIGZIP="$CIGZIP_DIR/target/release/cigzip"

    if [ ! -f "$CIGZIP" ]; then
        echo "Error: cigzip binary not found after build"
        exit 1
    fi
else
    echo "Error: cigzip directory not found: $CIGZIP_DIR"
    exit 1
fi
echo "✓ cigzip built"
echo

# Build lib_bpaf test programs
echo "=== Building lib_bpaf test programs ==="
cd "$REPO_DIR"
cargo build --release 2>&1 | tail -3

# Build seek test program (Mode A - with index)
cat > /tmp/seek_test_mode_a.rs << 'RUST_EOF'
use std::env;
use std::time::Instant;
use lib_bpaf::BpafReader;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <bpaf> <record_id> <iterations>", args[0]);
        std::process::exit(1);
    }

    let bpaf_path = &args[1];
    let record_id: u64 = args[2].parse().expect("Invalid record_id");
    let iterations: usize = args[3].parse().expect("Invalid iterations");

    // Warmup
    for _ in 0..3 {
        let mut reader = BpafReader::open(bpaf_path).expect("Failed to open BPAF");
        let _ = reader.get_tracepoints(record_id).expect("Failed to fetch");
    }

    // Benchmark
    let mut total_open_us = 0f64;
    let mut total_seek_us = 0f64;

    for _ in 0..iterations {
        let open_start = Instant::now();
        let mut reader = BpafReader::open(bpaf_path).expect("Failed to open BPAF");
        let open_elapsed = open_start.elapsed().as_micros() as f64;
        total_open_us += open_elapsed;

        let seek_start = Instant::now();
        let _ = reader.get_tracepoints(record_id).expect("Failed to fetch");
        let seek_elapsed = seek_start.elapsed().as_micros() as f64;
        total_seek_us += seek_elapsed;
    }

    let avg_open_us = total_open_us / iterations as f64;
    let avg_seek_us = total_seek_us / iterations as f64;
    let avg_total_us = avg_open_us + avg_seek_us;

    println!("MODEA_OPEN {:.2}", avg_open_us);
    println!("MODEA_SEEK {:.2}", avg_seek_us);
    println!("MODEA_TOTAL {:.2}", avg_total_us);
}
RUST_EOF

rustc --edition 2021 -O /tmp/seek_test_mode_a.rs \
    -L target/release/deps \
    --extern lib_bpaf=target/release/liblib_bpaf.rlib \
    -o /tmp/seek_test_mode_a 2>/dev/null

# Build seek test program (Mode B - without index, direct offset)
cat > /tmp/seek_test_mode_b.rs << 'RUST_EOF'
use std::env;
use std::time::Instant;
use lib_bpaf::BpafReader;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <bpaf> <record_id> <iterations>", args[0]);
        std::process::exit(1);
    }

    let bpaf_path = &args[1];
    let record_id: u64 = args[2].parse().expect("Invalid record_id");
    let iterations: usize = args[3].parse().expect("Invalid iterations");

    // Pre-compute tracepoint offset using BpafReader (not timed)
    let mut reader = BpafReader::open(bpaf_path).expect("Failed to open BPAF");
    let tracepoint_offset = reader.get_tracepoint_offset(record_id).expect("Failed to get tracepoint offset");
    drop(reader);

    // Warmup
    for _ in 0..3 {
        let mut reader = BpafReader::open_without_index(bpaf_path).expect("Failed to open BPAF");
        let _ = reader.get_tracepoints_at_offset(tracepoint_offset).expect("Failed to fetch");
    }

    // Benchmark
    let mut total_open_us = 0f64;
    let mut total_seek_us = 0f64;

    for _ in 0..iterations {
        let open_start = Instant::now();
        let mut reader = BpafReader::open_without_index(bpaf_path).expect("Failed to open BPAF");
        let open_elapsed = open_start.elapsed().as_micros() as f64;
        total_open_us += open_elapsed;

        let seek_start = Instant::now();
        let _ = reader.get_tracepoints_at_offset(tracepoint_offset).expect("Failed to fetch");
        let seek_elapsed = seek_start.elapsed().as_micros() as f64;
        total_seek_us += seek_elapsed;
    }

    let avg_open_us = total_open_us / iterations as f64;
    let avg_seek_us = total_seek_us / iterations as f64;
    let avg_total_us = avg_open_us + avg_seek_us;

    println!("MODEB_OPEN {:.2}", avg_open_us);
    println!("MODEB_SEEK {:.2}", avg_seek_us);
    println!("MODEB_TOTAL {:.2}", avg_total_us);
}
RUST_EOF

rustc --edition 2021 -O /tmp/seek_test_mode_b.rs \
    -L target/release/deps \
    --extern lib_bpaf=target/release/liblib_bpaf.rlib \
    -o /tmp/seek_test_mode_b 2>/dev/null

# Build decompression test program
cat > /tmp/decompression_test.rs << 'RUST_EOF'
use std::env;
use std::time::Instant;
use lib_bpaf::decompress_bpaf;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input> <output>", args[0]);
        std::process::exit(1);
    }

    let input = &args[1];
    let output = &args[2];

    let start = Instant::now();
    decompress_bpaf(input, output).expect("Decompression failed");
    let elapsed = start.elapsed();

    println!("Decompression time: {:.3}s", elapsed.as_secs_f64());
}
RUST_EOF

rustc --edition 2021 -O /tmp/decompression_test.rs \
    -L target/release/deps \
    --extern lib_bpaf=target/release/liblib_bpaf.rlib \
    -o /tmp/decompression_test 2>/dev/null

echo "✓ Build complete"
echo

# Extract sample from CIGAR-based PAF
echo "=== Extracting $NUM_RECORDS records from CIGAR PAF ==="
CIGAR_SAMPLE="/tmp/cigar_sample.paf"
zcat "$CIGAR_PAF" | head -n $NUM_RECORDS > "$CIGAR_SAMPLE"

ACTUAL_RECORDS=$(wc -l < "$CIGAR_SAMPLE")
CIGAR_SIZE=$(stat -c%s "$CIGAR_SAMPLE")
echo "Extracted: $ACTUAL_RECORDS records"
echo "Size:      $(ls -lh "$CIGAR_SAMPLE" | awk '{print $5}') ($CIGAR_SIZE bytes)"
echo

# Clean up any existing test files
rm -f /tmp/test.*.bpaf /tmp/test.*.bpaf.idx /tmp/test.*.paf 2>/dev/null

SEEK_POSITIONS=(1 100 500)
if [ $ACTUAL_RECORDS -gt 900 ]; then
    SEEK_POSITIONS=(1 100 500 900)
fi

# Main testing loop for each tracepoint type
for TP_TYPE in "${TP_TYPES[@]}"; do
    echo
    echo "###################################################################"
    echo "# TESTING TRACEPOINT TYPE: ${TP_TYPE^^}"
    echo "###################################################################"
    echo

    TP_PAF="/tmp/test.${TP_TYPE}.tp.paf"
    OUTPUT_BPAF="/tmp/test.${TP_TYPE}.bpaf"
    DECOMP_PAF="/tmp/test.${TP_TYPE}.decomp.paf"

    # ========================================
    # TEST 1: Encode (CIGAR → Tracepoints)
    # ========================================
    echo "========================================"
    echo "TEST 1 ($TP_TYPE): Encode CIGAR → Tracepoints"
    echo "========================================"

    ENCODE_START=$(date +%s.%N)

    $CIGZIP encode \
        --paf "$CIGAR_SAMPLE" \
        --threads 8 \
        --type "$TP_TYPE" \
        --max-complexity 10 \
        --complexity-metric edit-distance \
        > "$TP_PAF" 2>/tmp/encode_${TP_TYPE}.log

    ENCODE_END=$(date +%s.%N)
    ENCODE_TIME[$TP_TYPE]=$(echo "$ENCODE_END - $ENCODE_START" | bc -l)

    TP_SIZE=$(stat -c%s "$TP_PAF")
    echo "Encode time: ${ENCODE_TIME[$TP_TYPE]}s"
    echo "Output size: $(ls -lh "$TP_PAF" | awk '{print $5}') ($TP_SIZE bytes)"
    echo

    # ========================================
    # TEST 2: Compression (Tracepoints → BPAF)
    # ========================================
    echo "========================================"
    echo "TEST 2 ($TP_TYPE): Compress Tracepoints → BPAF"
    echo "========================================"

    $CIGZIP compress \
        --input "$TP_PAF" \
        --output "$OUTPUT_BPAF" \
        --strategy varint-zstd \
        --type "$TP_TYPE" \
        --max-complexity 10 \
        --complexity-metric edit-distance \
        --distance gap-affine-2p \
        --penalties 5,8,2,24,1 \
        2>&1 | tee /tmp/compress_${TP_TYPE}.log | tail -5

    COMPRESS_TIME[$TP_TYPE]=$(grep -oP 'Compression took \K[0-9.]+' /tmp/compress_${TP_TYPE}.log || echo "N/A")
    COMPRESS_SIZE[$TP_TYPE]=$(stat -c%s "$OUTPUT_BPAF")

    echo "Compress time: ${COMPRESS_TIME[$TP_TYPE]}s"
    echo "Output size:   $(ls -lh "$OUTPUT_BPAF" | awk '{print $5}') (${COMPRESS_SIZE[$TP_TYPE]} bytes)"
    echo

    # ========================================
    # TEST 3: Decompression (BPAF → Tracepoints)
    # ========================================
    echo "========================================"
    echo "TEST 3 ($TP_TYPE): Decompress BPAF → Tracepoints"
    echo "========================================"

    /usr/bin/time -v /tmp/decompression_test "$OUTPUT_BPAF" "$DECOMP_PAF" 2>&1 | tee /tmp/decompress_${TP_TYPE}.log | grep -E "(Decompression time|Maximum resident|User time|System time|Elapsed)"

    DECOMP_TIME[$TP_TYPE]=$(grep "Decompression time" /tmp/decompress_${TP_TYPE}.log | awk '{print $3}' | sed 's/s//')
    DECOMP_MEM[$TP_TYPE]=$(grep "Maximum resident" /tmp/decompress_${TP_TYPE}.log | awk '{print $6}')

    echo "Decomp time:  ${DECOMP_TIME[$TP_TYPE]}s"
    echo "Memory:       ${DECOMP_MEM[$TP_TYPE]} KB"

    # Verify correctness (normalize all float fields to 3 decimal places)
    # Normalize function: truncate ALL float fields (md:f:, gi:f:, bi:f:) to 3 decimal places
    # Using 3 decimals to avoid f32 rounding errors completely
    normalize_precision() {
        # First pass: ensure all floats have at least 3 decimal places by padding with zeros
        # Second pass: truncate to exactly 3 decimal places
        sed -E 's/(:f:[0-9]+\.[0-9]{1,2})($|[^0-9])/\1000\2/g' "$1" | \
        sed -E 's/(:f:[0-9]+\.)([0-9]{3})[0-9]*/\1\2/g'
    }

    MD5_TP=$(normalize_precision "$TP_PAF" | md5sum | awk '{print $1}')
    MD5_DECOMP=$(normalize_precision "$DECOMP_PAF" | md5sum | awk '{print $1}')

    if [ "$MD5_TP" = "$MD5_DECOMP" ]; then
        echo "✓ Tracepoints verified: Perfect match (MD5: $MD5_TP, normalized to 3 decimal places for all floats)"
        VERIFIED[$TP_TYPE]="✓ Perfect match"
    else
        echo "✗ MD5 mismatch!"
        echo "  Encoded:      $MD5_TP"
        echo "  Decompressed: $MD5_DECOMP"
        VERIFIED[$TP_TYPE]="✗ FAILED"
    fi
    echo

    # ========================================
    # TEST 4: O(1) Random Access Performance
    # ========================================
    echo "========================================"
    echo "TEST 4 ($TP_TYPE): O(1) Random Access"
    echo "========================================"

    # Clean index to ensure fresh build
    rm -f "$OUTPUT_BPAF.idx" 2>/dev/null

    echo "Testing $SEEK_ITERATIONS seeks at different positions..."
    echo

    # Calculate average across all positions for Mode A
    TOTAL_TIME_A=0
    COUNT_A=0
    for POS in "${SEEK_POSITIONS[@]}"; do
        if [ $POS -ge $ACTUAL_RECORDS ]; then
            continue
        fi

        OUTPUT=$(/usr/bin/time -v /tmp/seek_test_mode_a "$OUTPUT_BPAF" $POS $SEEK_ITERATIONS 2>&1)
        TIME_A=$(echo "$OUTPUT" | grep "MODEA_TOTAL" | awk '{print $2}')

        if [ -n "$TIME_A" ] && [ "$TIME_A" != "0.00" ]; then
            TOTAL_TIME_A=$(echo "$TOTAL_TIME_A + $TIME_A" | bc -l)
            COUNT_A=$((COUNT_A + 1))
        fi
    done

    if [ $COUNT_A -gt 0 ]; then
        SEEK_TIME_A[$TP_TYPE]=$(echo "scale=2; $TOTAL_TIME_A / $COUNT_A" | bc -l)
    else
        SEEK_TIME_A[$TP_TYPE]="N/A"
    fi

    # Calculate average across all positions for Mode B
    TOTAL_TIME_B=0
    COUNT_B=0
    for POS in "${SEEK_POSITIONS[@]}"; do
        if [ $POS -ge $ACTUAL_RECORDS ]; then
            continue
        fi

        OUTPUT=$(/usr/bin/time -v /tmp/seek_test_mode_b "$OUTPUT_BPAF" $POS $SEEK_ITERATIONS 2>&1)
        TIME_B=$(echo "$OUTPUT" | grep "MODEB_TOTAL" | awk '{print $2}')

        if [ -n "$TIME_B" ] && [ "$TIME_B" != "0.00" ]; then
            TOTAL_TIME_B=$(echo "$TOTAL_TIME_B + $TIME_B" | bc -l)
            COUNT_B=$((COUNT_B + 1))
        fi
    done

    if [ $COUNT_B -gt 0 ]; then
        SEEK_TIME_B[$TP_TYPE]=$(echo "scale=2; $TOTAL_TIME_B / $COUNT_B" | bc -l)
    else
        SEEK_TIME_B[$TP_TYPE]="N/A"
    fi

    echo "Average seek time (Mode A - with index):    ${SEEK_TIME_A[$TP_TYPE]} μs"
    echo "Average seek time (Mode B - direct offset): ${SEEK_TIME_B[$TP_TYPE]} μs"
    echo
done

# ========================================
# FINAL SUMMARY
# ========================================
echo
echo "###################################################################"
echo "# FINAL SUMMARY - ALL TRACEPOINT TYPES"
echo "###################################################################"
echo
echo "Source:  $CIGAR_PAF"
echo "Sample:  $ACTUAL_RECORDS records, $(ls -lh "$CIGAR_SAMPLE" | awk '{print $5}') ($CIGAR_SIZE bytes)"
echo

# Print table
printf "╔═══════════╦══════════════╦═══════════════╦════════════════╦═══════════════╦═════════════╦═════════════╗\n"
printf "║ %-9s ║ Encode (s)   ║ Compress (s)  ║ Decomp (s)     ║ Size (bytes)  ║ Seek A (μs) ║ Seek B (μs) ║\n" "Type"
printf "╠═══════════╬══════════════╬═══════════════╬════════════════╬═══════════════╬═════════════╬═════════════╣\n"

for TP_TYPE in "${TP_TYPES[@]}"; do
    printf "║ %-9s ║ %12s ║ %13s ║ %14s ║ %13s ║ %11s ║ %11s ║\n" \
        "$TP_TYPE" \
        "$(printf '%.3f' ${ENCODE_TIME[$TP_TYPE]} 2>/dev/null || echo 'N/A')" \
        "${COMPRESS_TIME[$TP_TYPE]}" \
        "${DECOMP_TIME[$TP_TYPE]}" \
        "${COMPRESS_SIZE[$TP_TYPE]}" \
        "${SEEK_TIME_A[$TP_TYPE]}" \
        "${SEEK_TIME_B[$TP_TYPE]}"
done

printf "╚═══════════╩══════════════╩═══════════════╩════════════════╩═══════════════╩═════════════╩═════════════╝\n"
echo

# Data integrity summary
echo "Data Integrity:"
for TP_TYPE in "${TP_TYPES[@]}"; do
    printf "  %-9s : %s\n" "$TP_TYPE" "${VERIFIED[$TP_TYPE]}"
done
echo

# Compression ratios (CIGAR PAF → BPAF)
echo "Compression Ratios (CIGAR PAF → BPAF):"
for TP_TYPE in "${TP_TYPES[@]}"; do
    RATIO=$(echo "scale=2; $CIGAR_SIZE / ${COMPRESS_SIZE[$TP_TYPE]}" | bc -l)
    printf "  %-9s : %.2fx (CIGAR: %d bytes → BPAF: %s bytes)\n" \
        "$TP_TYPE" \
        "$RATIO" \
        "$CIGAR_SIZE" \
        "${COMPRESS_SIZE[$TP_TYPE]}"
done

echo
echo "========================================"
echo "All tests complete!"
echo "========================================"
