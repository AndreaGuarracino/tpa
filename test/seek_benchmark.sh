#!/bin/bash
set -e

# Seek benchmark script for PAF (CIGAR) and BPAF (tracepoints)
# Tests random access performance for both formats

REPO_DIR="/home/guarracino/Dropbox/git/lib_bpaf"
SEEK_PER_RECORD="${2:-100}"
STRIDE="${3:-10}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <file.paf|file.paf.gz|file.bpaf> [iterations_per_record=100] [stride=10]"
    echo ""
    echo "Tests seek performance:"
    echo "  - PAF/PAF.gz: Seeks to CIGAR strings using BGZF index"
    echo "  - BPAF: Seeks to tracepoints using Mode A (BpafReader) and Mode B (standalone functions)"
    echo ""
    echo "Parameters:"
    echo "  iterations_per_record: Number of times to seek to each record"
    echo "  stride: Skip records (test every Nth record)"
    exit 1
fi

INPUT_FILE="$1"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File not found: $INPUT_FILE"
    exit 1
fi

# Detect file type
if [[ "$INPUT_FILE" == *.bpaf ]]; then
    FILE_TYPE="bpaf"
elif [[ "$INPUT_FILE" == *.paf.gz ]] || [[ "$INPUT_FILE" == *.paf ]]; then
    FILE_TYPE="paf"
else
    echo "Error: Unsupported file type. Must be .paf, .paf.gz, or .bpaf"
    exit 1
fi

echo "========================================"
echo "Seek Benchmark"
echo "========================================"
echo "File:       $INPUT_FILE"
echo "Type:       $FILE_TYPE"
echo "Iterations: $SEEK_PER_RECORD per record"
echo "Stride:     $STRIDE (test every ${STRIDE}th record)"
echo "========================================"
echo

# Function to test PAF CIGAR seeks
test_paf_seeks() {
    local paf_file="$1"
    local iterations="$2"
    local stride="$3"

    PAF_SEEK_BIN="$REPO_DIR/target/release/examples/paf_seek_bench"

    if [ ! -x "$PAF_SEEK_BIN" ]; then
        echo "Error: paf_seek_bench binary not found at $PAF_SEEK_BIN"
        echo "Build it with: cargo build --release --example paf_seek_bench"
        return 1
    fi

    # Count records
    if [[ "$paf_file" == *.gz ]]; then
        RECORD_COUNT=$(zcat "$paf_file" | wc -l)
    else
        RECORD_COUNT=$(wc -l < "$paf_file")
    fi

    echo "=== PAF BGZF CIGAR Seek Test ==="
    echo "Records: $RECORD_COUNT"
    echo

    PAF_SEEK_OUTPUT=$("$PAF_SEEK_BIN" "$paf_file" "$RECORD_COUNT" "$stride" "$iterations" 2>&1 || true)
    echo "$PAF_SEEK_OUTPUT"

    if [[ $PAF_SEEK_OUTPUT =~ PAF_INDEXED[[:space:]]+([0-9]+) ]]; then
        PAF_INDEXED="${BASH_REMATCH[1]}"
    else
        PAF_INDEXED="0"
    fi

    if [[ $PAF_SEEK_OUTPUT =~ PAF_SEEK[[:space:]]+([0-9.]+) ]]; then
        PAF_SEEK_TIME="${BASH_REMATCH[1]}"
    else
        PAF_SEEK_TIME="N/A"
    fi

    echo
    echo "Results:"
    echo "  Records indexed: $PAF_INDEXED"
    echo "  Average seek time: ${PAF_SEEK_TIME} μs"
}

# Function to test BPAF tracepoint seeks
test_bpaf_seeks() {
    local bpaf_file="$1"
    local iterations="$2"
    local stride="$3"

    # Build seek test programs if needed
    build_seek_programs

    # Detect tracepoint type from file
    TP_TYPE=$(detect_tracepoint_type "$bpaf_file")

    if [ -z "$TP_TYPE" ]; then
        echo "Error: Could not detect tracepoint type from BPAF file"
        return 1
    fi

    # Get record count from BPAF
    BPAF_VIEW="$REPO_DIR/target/release/bpaf-view"
    if [ ! -x "$BPAF_VIEW" ]; then
        echo "Error: bpaf-view binary not found. Building..."
        cd "$REPO_DIR"
        cargo build --release 2>&1 | tail -3
    fi

    RECORD_COUNT=$("$BPAF_VIEW" "$bpaf_file" 2>/dev/null | head -1 | wc -l)
    if [ "$RECORD_COUNT" -eq 0 ]; then
        # Try alternative method using index
        if [ -f "${bpaf_file}.idx" ]; then
            RECORD_COUNT=$(wc -c < "${bpaf_file}.idx" | awk '{print int($1/8)}')
        else
            echo "Error: Could not determine record count"
            return 1
        fi
    fi

    echo "=== BPAF Tracepoint Seek Test ==="
    echo "Tracepoint type: $TP_TYPE"
    echo "Records: $RECORD_COUNT"
    echo

    # Mode A: BpafReader with index
    echo "--- Mode A: BpafReader with index ---"
    TOTAL_TIME_A=0
    COUNT_A=0

    for i in $(seq 0 "$stride" $((RECORD_COUNT - 1))); do
        OUTPUT=$(/tmp/seek_test_mode_a "$bpaf_file" "$i" "$iterations" 2>/dev/null || true)
        if [[ $OUTPUT =~ MODEA_SEEK[[:space:]]+([0-9.]+) ]]; then
            TIME="${BASH_REMATCH[1]}"
            TOTAL_TIME_A=$(echo "$TOTAL_TIME_A + $TIME" | bc -l)
            COUNT_A=$((COUNT_A + 1))
        fi
    done

    if [ "$COUNT_A" -gt 0 ]; then
        SEEK_TIME_A=$(echo "scale=2; $TOTAL_TIME_A / $COUNT_A" | bc -l)
    else
        SEEK_TIME_A="N/A"
    fi

    echo "Average seek time: ${SEEK_TIME_A} μs"
    echo

    # Mode B: Standalone functions
    echo "--- Mode B: Standalone functions ---"
    TOTAL_TIME_B=0
    COUNT_B=0

    for i in $(seq 0 "$stride" $((RECORD_COUNT - 1))); do
        OUTPUT=$(/tmp/seek_test_mode_b "$bpaf_file" "$i" "$iterations" "$TP_TYPE" 2>/dev/null || true)
        if [[ $OUTPUT =~ MODEB_SEEK[[:space:]]+([0-9.]+) ]]; then
            TIME="${BASH_REMATCH[1]}"
            TOTAL_TIME_B=$(echo "$TOTAL_TIME_B + $TIME" | bc -l)
            COUNT_B=$((COUNT_B + 1))
        fi
    done

    if [ "$COUNT_B" -gt 0 ]; then
        SEEK_TIME_B=$(echo "scale=2; $TOTAL_TIME_B / $COUNT_B" | bc -l)
    else
        SEEK_TIME_B="N/A"
    fi

    echo "Average seek time: ${SEEK_TIME_B} μs"
    echo

    echo "Results Summary:"
    echo "  Mode A (BpafReader):       ${SEEK_TIME_A} μs"
    echo "  Mode B (Standalone funcs): ${SEEK_TIME_B} μs"
}

# Detect tracepoint type from BPAF header
detect_tracepoint_type() {
    local bpaf_file="$1"

    # Use bpaf-view to get header info
    BPAF_VIEW="$REPO_DIR/target/release/bpaf-view"
    if [ ! -x "$BPAF_VIEW" ]; then
        return 1
    fi

    HEADER_INFO=$("$BPAF_VIEW" "$bpaf_file" 2>&1 | head -20)

    if echo "$HEADER_INFO" | grep -q "Tracepoint.*Standard"; then
        echo "standard"
    elif echo "$HEADER_INFO" | grep -q "Tracepoint.*Variable"; then
        echo "variable"
    elif echo "$HEADER_INFO" | grep -q "Tracepoint.*Mixed"; then
        echo "mixed"
    elif echo "$HEADER_INFO" | grep -q "Tracepoint.*Fastga"; then
        echo "fastga"
    else
        # Default to standard if can't detect
        echo "standard"
    fi
}

# Build seek test programs
build_seek_programs() {
    if [ -f /tmp/seek_test_mode_a ] && [ -f /tmp/seek_test_mode_b ]; then
        return 0
    fi

    echo "Building seek test programs..."
    cd "$REPO_DIR"
    cargo build --release 2>&1 | tail -3

    # Mode A program
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

    let mut reader = BpafReader::open(bpaf_path).expect("Failed to open BPAF");

    // Warmup
    for _ in 0..3 {
        let _ = reader.get_tracepoints(record_id).expect("Failed to fetch");
    }

    // Benchmark
    let mut total_seek_us = 0f64;
    for _ in 0..iterations {
        let seek_start = Instant::now();
        let _ = reader.get_tracepoints(record_id).expect("Failed to fetch");
        let seek_elapsed = seek_start.elapsed().as_micros() as f64;
        total_seek_us += seek_elapsed;
    }

    let avg_seek_us = total_seek_us / iterations as f64;
    println!("MODEA_SEEK {:.2}", avg_seek_us);
}
RUST_EOF

    # Mode B program
    cat > /tmp/seek_test_mode_b.rs << 'RUST_EOF'
use std::env;
use std::time::Instant;
use std::fs::File;
use lib_bpaf::{BpafReader, read_standard_tracepoints_at_offset,
               read_variable_tracepoints_at_offset, read_mixed_tracepoints_at_offset};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!("Usage: {} <bpaf> <record_id> <iterations> <tp_type>", args[0]);
        std::process::exit(1);
    }

    let bpaf_path = &args[1];
    let record_id: u64 = args[2].parse().expect("Invalid record_id");
    let iterations: usize = args[3].parse().expect("Invalid iterations");
    let tp_type = &args[4];

    // Pre-compute offset and strategy using BpafReader (not timed)
    let mut reader = BpafReader::open(bpaf_path).expect("Failed to open BPAF");
    let offset = reader.get_tracepoint_offset(record_id).expect("Failed to get offset");
    let header = reader.header();
    let strategy = header.strategy().expect("Failed to get strategy");
    drop(reader);

    // Open file once for Mode B
    let mut file = File::open(bpaf_path).expect("Failed to open file");

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
    let mut total_seek_us = 0f64;
    for _ in 0..iterations {
        let seek_start = Instant::now();
        match tp_type.as_str() {
            "standard" => { let _ = read_standard_tracepoints_at_offset(&mut file, offset, strategy).expect("Failed"); }
            "variable" => { let _ = read_variable_tracepoints_at_offset(&mut file, offset).expect("Failed"); }
            "mixed" => { let _ = read_mixed_tracepoints_at_offset(&mut file, offset).expect("Failed"); }
            _ => panic!("Invalid tp_type"),
        }
        let seek_elapsed = seek_start.elapsed().as_micros() as f64;
        total_seek_us += seek_elapsed;
    }

    let avg_seek_us = total_seek_us / iterations as f64;
    println!("MODEB_SEEK {:.2}", avg_seek_us);
}
RUST_EOF

    rustc --edition 2021 -O /tmp/seek_test_mode_a.rs \
        -L target/release/deps \
        --extern lib_bpaf=target/release/liblib_bpaf.rlib \
        -o /tmp/seek_test_mode_a 2>/dev/null

    rustc --edition 2021 -O /tmp/seek_test_mode_b.rs \
        -L target/release/deps \
        --extern lib_bpaf=target/release/liblib_bpaf.rlib \
        -o /tmp/seek_test_mode_b 2>/dev/null

    echo "✓ Seek test programs built"
    echo
}

# Run appropriate test based on file type
if [ "$FILE_TYPE" = "paf" ]; then
    test_paf_seeks "$INPUT_FILE" "$SEEK_PER_RECORD" "$STRIDE"
elif [ "$FILE_TYPE" = "bpaf" ]; then
    test_bpaf_seeks "$INPUT_FILE" "$SEEK_PER_RECORD" "$STRIDE"
fi

echo
echo "========================================"
echo "Seek benchmark complete!"
echo "========================================"
