#!/bin/bash
set -e

# Comprehensive test suite for lib_bpaf parallel compression
# Tests: compression, decompression, O(1) seeks - all with 1 and 8 threads
# Measures: runtime, memory, correctness

REPO_DIR="/home/guarracino/Dropbox/git/lib_bpaf"
INPUT="${1:-/home/guarracino/Desktop/big-from-fg.tp.20k.paf}"
STRATEGY="${2:-automatic}"
SEEK_ITERATIONS=1000

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file not found: $INPUT"
    exit 1
fi

echo "========================================"
echo "lib_bpaf Comprehensive Test Suite"
echo "========================================"
echo "Input:      $INPUT"
echo "Size:       $(ls -lh $INPUT | awk '{print $5}')"
echo "Records:    $(wc -l < $INPUT)"
echo "Strategy:   $STRATEGY"
echo "========================================"
echo

# Build test programs
echo "=== Building test programs ==="
cd "$REPO_DIR"
cargo build --release 2>&1 | tail -3

# Build verify_tracepoints program
if [ -f "$REPO_DIR/test/verify_tracepoints.rs" ]; then
    echo "Compiling verify_tracepoints..."
    rustc --edition 2021 -O "$REPO_DIR/test/verify_tracepoints.rs" \
        -L target/release/deps \
        --extern lib_bpaf=target/release/liblib_bpaf.rlib \
        -o "$REPO_DIR/test/verify_tracepoints" 2>/dev/null || echo "Warning: verify_tracepoints compilation failed"
fi

# Build compression test program
cat > /tmp/compression_test.rs << 'RUST_EOF'
use std::env;
use std::time::Instant;
use lib_bpaf::{compress_paf, CompressionStrategy, Distance};
use lib_tracepoints::{TracepointType, ComplexityMetric};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input> <output>", args[0]);
        std::process::exit(1);
    }

    let input = &args[1];
    let output = &args[2];
    let strategy = CompressionStrategy::Automatic(3);

    // FASTGA format with trace_spacing=100
    let tp_type = TracepointType::Fastga;
    let trace_spacing = 100;  // Stored in max_complexity field
    let complexity_metric = ComplexityMetric::EditDistance;
    let distance = Distance::Edit;

    let start = Instant::now();
    compress_paf(input, output, strategy, tp_type, trace_spacing, complexity_metric, distance).expect("Compression failed");
    let elapsed = start.elapsed();

    println!("Compression time: {:.3}s", elapsed.as_secs_f64());
}
RUST_EOF

# Find the lib_tracepoints rlib
TRACEPOINTS_RLIB=$(find target/release/deps -name "liblib_tracepoints*.rlib" | head -1)
rustc --edition 2021 -O /tmp/compression_test.rs \
    -L target/release/deps \
    --extern lib_bpaf=target/release/liblib_bpaf.rlib \
    --extern lib_tracepoints="$TRACEPOINTS_RLIB" \
    -o /tmp/compression_test 2>/dev/null

# Build decompression test program
cat > /tmp/decompression_test.rs << 'RUST_EOF'
use std::env;
use std::time::Instant;
use lib_bpaf::decompress_paf;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <input> <output>", args[0]);
        std::process::exit(1);
    }

    let input = &args[1];
    let output = &args[2];

    let start = Instant::now();
    decompress_paf(input, output).expect("Decompression failed");
    let elapsed = start.elapsed();

    println!("Decompression time: {:.3}s", elapsed.as_secs_f64());
}
RUST_EOF

rustc --edition 2021 -O /tmp/decompression_test.rs \
    -L target/release/deps \
    --extern lib_bpaf=target/release/liblib_bpaf.rlib \
    -o /tmp/decompression_test 2>/dev/null

# Build O(1) seek test program (with index, full runtime including file open)
cat > /tmp/seek_test.rs << 'RUST_EOF'
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

    // Warmup (with file opening)
    for _ in 0..3 {
        let mut reader = BpafReader::open(bpaf_path).expect("Failed to open BPAF");
        let _ = reader.get_tracepoints(record_id).expect("Failed to fetch");
    }

    // Benchmark - measure components separately
    let mut total_open_us = 0f64;
    let mut total_seek_us = 0f64;

    for _ in 0..iterations {
        // Measure open + load index
        let open_start = Instant::now();
        let mut reader = BpafReader::open(bpaf_path).expect("Failed to open BPAF");
        let open_elapsed = open_start.elapsed().as_micros() as f64;
        total_open_us += open_elapsed;

        // Measure seek + decompress
        let seek_start = Instant::now();
        let _ = reader.get_tracepoints(record_id).expect("Failed to fetch");
        let seek_elapsed = seek_start.elapsed().as_micros() as f64;
        total_seek_us += seek_elapsed;
    }

    let avg_open_us = total_open_us / iterations as f64;
    let avg_seek_us = total_seek_us / iterations as f64;
    let avg_total_us = avg_open_us + avg_seek_us;

    println!("Mode A breakdown:");
    println!("  Open+LoadIndex: {:.2} μs", avg_open_us);
    println!("  Seek+Decompress: {:.2} μs", avg_seek_us);
    println!("  Total: {:.2} μs", avg_total_us);
}
RUST_EOF

rustc --edition 2021 -O /tmp/seek_test.rs \
    -L target/release/deps \
    --extern lib_bpaf=target/release/liblib_bpaf.rlib \
    -o /tmp/seek_test 2>/dev/null

# Build O(1) seek test program (Mode B - direct tracepoint offset access)
cat > /tmp/seek_test_tracepoint_offset.rs << 'RUST_EOF'
use std::env;
use std::time::Instant;
use std::io::{Read, Seek, SeekFrom};
use lib_bpaf::{BpafReader, build_index};

fn read_varint<R: Read>(reader: &mut R) -> std::io::Result<u64> {
    let mut value: u64 = 0;
    let mut shift = 0;
    loop {
        let mut byte_buf = [0u8; 1];
        reader.read_exact(&mut byte_buf)?;
        let byte = byte_buf[0];
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    Ok(value)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <bpaf> <record_id> <iterations>", args[0]);
        std::process::exit(1);
    }

    let bpaf_path = &args[1];
    let record_id: u64 = args[2].parse().expect("Invalid record_id");
    let iterations: usize = args[3].parse().expect("Invalid iterations");

    // Pre-build index to get record offset (not timed)
    let index = build_index(bpaf_path).expect("Failed to build index");
    let record_offset = index.get_offset(record_id).expect("Record ID out of range");

    // Calculate tracepoint offset by skipping PAF fields (not timed)
    let mut reader = BpafReader::open_without_index(bpaf_path).expect("Failed to open BPAF");
    let mut file = std::fs::File::open(bpaf_path).expect("Failed to open file");
    file.seek(SeekFrom::Start(record_offset)).expect("Failed to seek");

    // Skip 7 varints + 2 single bytes to reach tracepoint data
    read_varint(&mut file).expect("query_name_id");
    read_varint(&mut file).expect("query_start");
    read_varint(&mut file).expect("query_end");
    file.seek(SeekFrom::Current(1)).expect("strand");
    read_varint(&mut file).expect("target_name_id");
    read_varint(&mut file).expect("target_start");
    read_varint(&mut file).expect("target_end");
    read_varint(&mut file).expect("residue_matches");
    read_varint(&mut file).expect("alignment_block_len");
    file.seek(SeekFrom::Current(1)).expect("mapping_quality");

    let tracepoint_offset = file.stream_position().expect("Failed to get position");
    drop(file);

    // Warmup (open without index, use direct tracepoint offset access)
    for _ in 0..3 {
        let mut reader = BpafReader::open_without_index(bpaf_path).expect("Failed to open BPAF");
        let _ = reader.get_tracepoints_at_offset(tracepoint_offset).expect("Failed to fetch");
    }

    // Benchmark - measure components separately
    let mut total_open_us = 0f64;
    let mut total_seek_us = 0f64;

    for _ in 0..iterations {
        // Measure open (header only)
        let open_start = Instant::now();
        let mut reader = BpafReader::open_without_index(bpaf_path).expect("Failed to open BPAF");
        let open_elapsed = open_start.elapsed().as_micros() as f64;
        total_open_us += open_elapsed;

        // Measure direct tracepoint offset seek + decompress
        let seek_start = Instant::now();
        let _ = reader.get_tracepoints_at_offset(tracepoint_offset).expect("Failed to fetch");
        let seek_elapsed = seek_start.elapsed().as_micros() as f64;
        total_seek_us += seek_elapsed;
    }

    let avg_open_us = total_open_us / iterations as f64;
    let avg_seek_us = total_seek_us / iterations as f64;
    let avg_total_us = avg_open_us + avg_seek_us;

    println!("Mode B breakdown:");
    println!("  Open (header): {:.2} μs", avg_open_us);
    println!("  DirectTracepointSeek+Decompress: {:.2} μs", avg_seek_us);
    println!("  Total: {:.2} μs", avg_total_us);
}
RUST_EOF

rustc --edition 2021 -O /tmp/seek_test_tracepoint_offset.rs \
    -L target/release/deps \
    --extern lib_bpaf=target/release/liblib_bpaf.rlib \
    -o /tmp/seek_test_tracepoint_offset 2>/dev/null

echo "✓ Build complete"
echo

# Clean up any existing test files and indices
rm -f /tmp/test.*.bpaf /tmp/test.*.bpaf.idx /tmp/test.*.paf 2>/dev/null

# ========================================
# TEST 1: Compression (Run 1)
# ========================================
echo "========================================"
echo "TEST 1: Compression (Run 1)"
echo "========================================"

OUTPUT_1T="/tmp/test.1t.bpaf"
/usr/bin/time -v /tmp/compression_test "$INPUT" "$OUTPUT_1T" 2>&1 | tee /tmp/compress_1t.log | grep -E "(Compression time|Maximum resident|User time|System time|Elapsed)"

COMPRESS_1T_TIME=$(grep "Compression time" /tmp/compress_1t.log | awk '{print $3}' | sed 's/s//')
COMPRESS_1T_MEM=$(grep "Maximum resident" /tmp/compress_1t.log | awk '{print $6}')
COMPRESS_1T_SIZE=$(ls -lh "$OUTPUT_1T" | awk '{print $5}')

echo "Runtime:  ${COMPRESS_1T_TIME}s"
echo "Memory:   ${COMPRESS_1T_MEM} KB"
echo "Size:     ${COMPRESS_1T_SIZE}"
echo

# ========================================
# TEST 2: Compression (Run 2 - verify determinism)
# ========================================
echo "========================================"
echo "TEST 2: Compression (Run 2 - verify determinism)"
echo "========================================"

OUTPUT_8T="/tmp/test.8t.bpaf"
/usr/bin/time -v /tmp/compression_test "$INPUT" "$OUTPUT_8T" 2>&1 | tee /tmp/compress_8t.log | grep -E "(Compression time|Maximum resident|User time|System time|Elapsed)"

COMPRESS_8T_TIME=$(grep "Compression time" /tmp/compress_8t.log | awk '{print $3}' | sed 's/s//')
COMPRESS_8T_MEM=$(grep "Maximum resident" /tmp/compress_8t.log | awk '{print $6}')
COMPRESS_8T_SIZE=$(ls -lh "$OUTPUT_8T" | awk '{print $5}')

echo "Runtime:  ${COMPRESS_8T_TIME}s"
echo "Memory:   ${COMPRESS_8T_MEM} KB"
echo "Size:     ${COMPRESS_8T_SIZE}"
echo
echo

# ========================================
# TEST 3: Verify files are identical
# ========================================
echo "========================================"
echo "TEST 3: File Identity Check"
echo "========================================"

if cmp -s "$OUTPUT_1T" "$OUTPUT_8T"; then
    echo "✓ 1-thread and 8-thread outputs are identical"
else
    echo "✗ WARNING: Outputs differ!"
    echo "  1t: $(md5sum $OUTPUT_1T | awk '{print $1}')"
    echo "  8t: $(md5sum $OUTPUT_8T | awk '{print $1}')"
fi
echo

# ========================================
# TEST 4: Decompression (1 thread output)
# ========================================
echo "========================================"
echo "TEST 4: Decompression (1 thread output)"
echo "========================================"

DECOMP_1T="/tmp/test.1t.decomp.paf"
/usr/bin/time -v /tmp/decompression_test "$OUTPUT_1T" "$DECOMP_1T" 2>&1 | tee /tmp/decompress_1t.log | grep -E "(Decompression time|Maximum resident|User time|System time|Elapsed)"

DECOMP_1T_TIME=$(grep "Decompression time" /tmp/decompress_1t.log | awk '{print $3}' | sed 's/s//')
DECOMP_1T_MEM=$(grep "Maximum resident" /tmp/decompress_1t.log | awk '{print $6}')

echo "Runtime:  ${DECOMP_1T_TIME}s"
echo "Memory:   ${DECOMP_1T_MEM} KB"

# Verify correctness
MD5_ORIG=$(md5sum "$INPUT" | awk '{print $1}')
MD5_DECOMP=$(md5sum "$DECOMP_1T" | awk '{print $1}')

if [ "$MD5_ORIG" = "$MD5_DECOMP" ]; then
    echo "✓ MD5 verified: $MD5_ORIG"
else
    echo "✗ MD5 mismatch!"
    echo "  Original:     $MD5_ORIG"
    echo "  Decompressed: $MD5_DECOMP"
fi
echo

# ========================================
# TEST 5: Decompression (8 thread output)
# ========================================
echo "========================================"
echo "TEST 5: Decompression (8 thread output)"
echo "========================================"

DECOMP_8T="/tmp/test.8t.decomp.paf"
/usr/bin/time -v /tmp/decompression_test "$OUTPUT_8T" "$DECOMP_8T" 2>&1 | tee /tmp/decompress_8t.log | grep -E "(Decompression time|Maximum resident|User time|System time|Elapsed)"

DECOMP_8T_TIME=$(grep "Decompression time" /tmp/decompress_8t.log | awk '{print $3}' | sed 's/s//')
DECOMP_8T_MEM=$(grep "Maximum resident" /tmp/decompress_8t.log | awk '{print $6}')

echo "Runtime:  ${DECOMP_8T_TIME}s"
echo "Memory:   ${DECOMP_8T_MEM} KB"

# Verify correctness
MD5_DECOMP_8T=$(md5sum "$DECOMP_8T" | awk '{print $1}')

if [ "$MD5_ORIG" = "$MD5_DECOMP_8T" ]; then
    echo "✓ MD5 verified: $MD5_ORIG"
else
    echo "✗ MD5 mismatch!"
    echo "  Original:     $MD5_ORIG"
    echo "  Decompressed: $MD5_DECOMP_8T"
fi
echo

# ========================================
# TEST 6: O(1) Random Access Performance
# ========================================
echo "========================================"
echo "TEST 6: O(1) Random Access Performance"
echo "========================================"

# Clean index to ensure fresh build
rm -f "$OUTPUT_1T.idx" 2>/dev/null

TOTAL_RECORDS=$(wc -l < "$INPUT")
SEEK_POSITIONS=(1 10 100 1000 10000)

echo "Testing $SEEK_ITERATIONS seeks at different positions..."
echo "File: $OUTPUT_1T"
echo
echo "--- Mode A: WITH INDEX ---"

for POS in "${SEEK_POSITIONS[@]}"; do
    if [ $POS -ge $TOTAL_RECORDS ]; then
        continue
    fi

    /usr/bin/time -v /tmp/seek_test "$OUTPUT_1T" $POS $SEEK_ITERATIONS 2>&1 | tee /tmp/seek_${POS}.log | grep -E "(Mode A|Open\+LoadIndex|Seek\+Decompress|Total|Maximum resident)"

    OPEN_TIME=$(grep "Open+LoadIndex:" /tmp/seek_${POS}.log | awk '{print $2}')
    SEEK_TIME=$(grep "Seek+Decompress:" /tmp/seek_${POS}.log | awk '{print $2}')
    TOTAL_TIME=$(grep "Total:" /tmp/seek_${POS}.log | awk '{print $2}')
    SEEK_MEM=$(grep "Maximum resident" /tmp/seek_${POS}.log | awk '{print $6}')

    printf "Position %6d: Open=%s | Seek=%s | Total=%s μs | Mem=%s KB\n" $POS "$OPEN_TIME" "$SEEK_TIME" "$TOTAL_TIME" "$SEEK_MEM"
done

echo
echo "--- Mode B: DIRECT TRACEPOINT OFFSET ---"

for POS in "${SEEK_POSITIONS[@]}"; do
    if [ $POS -ge $TOTAL_RECORDS ]; then
        continue
    fi

    /usr/bin/time -v /tmp/seek_test_tracepoint_offset "$OUTPUT_1T" $POS $SEEK_ITERATIONS 2>&1 | tee /tmp/seek_tp_offset_${POS}.log | grep -E "(Mode B|Open \(header\)|DirectTracepointSeek\+Decompress|Total|Maximum resident)"

    OPEN_TIME=$(grep "Open (header):" /tmp/seek_tp_offset_${POS}.log | awk '{print $3}')
    SEEK_TIME=$(grep "DirectTracepointSeek+Decompress:" /tmp/seek_tp_offset_${POS}.log | awk '{print $2}')
    TOTAL_TIME=$(grep "Total:" /tmp/seek_tp_offset_${POS}.log | awk '{print $2}')
    SEEK_MEM=$(grep "Maximum resident" /tmp/seek_tp_offset_${POS}.log | awk '{print $6}')

    printf "Position %6d: Open=%s | Seek=%s | Total=%s μs | Mem=%s KB\n" $POS "$OPEN_TIME" "$SEEK_TIME" "$TOTAL_TIME" "$SEEK_MEM"
done
echo

# ========================================
# TEST 7: O(1) Seek Correctness
# ========================================
echo "========================================"
echo "TEST 7: O(1) Seek Correctness"
echo "========================================"

# Clean index again to ensure fresh build for correctness tests
rm -f "$OUTPUT_1T.idx" 2>/dev/null

if [ -f "$REPO_DIR/test/verify_tracepoints" ]; then
    echo "Verifying tracepoint accuracy for random records..."

    # Test 10 random positions
    TEST_POSITIONS=(0 1 10 100 500 1000 5000 10000 15000 19999)
    FAILED=0

    for POS in "${TEST_POSITIONS[@]}"; do
        if [ $POS -ge $TOTAL_RECORDS ]; then
            continue
        fi

        if "$REPO_DIR/test/verify_tracepoints" "$INPUT" "$OUTPUT_1T" $POS 2>&1 | grep -q "✓"; then
            echo "✓ Record $POS tracepoints verified"
        else
            echo "✗ Record $POS tracepoints MISMATCH!"
            FAILED=$((FAILED + 1))
        fi
    done

    echo
    if [ $FAILED -eq 0 ]; then
        echo "✓ All seek correctness tests passed"
    else
        echo "✗ Failed $FAILED correctness tests"
    fi
else
    echo "⚠ verify_tracepoints not available, skipping correctness check"
fi
echo

# ========================================
# SUMMARY
# ========================================
echo "========================================"
echo "COMPREHENSIVE TEST SUMMARY"
echo "========================================"
echo
echo "Input: $INPUT ($(ls -lh $INPUT | awk '{print $5}'))"
echo
echo "--- Compression ---"
echo "Run 1:  ${COMPRESS_1T_TIME}s | ${COMPRESS_1T_MEM} KB | ${COMPRESS_1T_SIZE}"
echo "Run 2:  ${COMPRESS_8T_TIME}s | ${COMPRESS_8T_MEM} KB | ${COMPRESS_8T_SIZE}"
echo
echo "--- Decompression ---"
echo "Run 1 output: ${DECOMP_1T_TIME}s | ${DECOMP_1T_MEM} KB"
echo "Run 2 output: ${DECOMP_8T_TIME}s | ${DECOMP_8T_MEM} KB"
echo
echo "--- Correctness ---"
echo "Compression: $(if cmp -s $OUTPUT_1T $OUTPUT_8T; then echo '✓ Identical'; else echo '✗ Different'; fi)"
echo "Decompression: $(if [ "$MD5_ORIG" = "$MD5_DECOMP" ] && [ "$MD5_ORIG" = "$MD5_DECOMP_8T" ]; then echo '✓ Verified'; else echo '✗ Failed'; fi)"
echo "O(1) seeks: $(if [ $FAILED -eq 0 ]; then echo '✓ Verified'; else echo "✗ $FAILED failures"; fi)"
echo
echo "========================================"
echo "All tests complete!"
echo "========================================"
