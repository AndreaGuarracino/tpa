#!/bin/bash
set -e

# Wrapper script to run comprehensive tests on multiple PAF files
# Aggregates results into a final report

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPREHENSIVE_TEST="$SCRIPT_DIR/comprehensive_test.sh"

# Default test files (can override with arguments)
TEST_FILES=(
    "${1:-/home/guarracino/git/impg/hprcv2/data/hg002v1.1.pat.PanSN-vs-HG02818_mat_hprc_r2_v1.0.1.p95.Pinf.aln.paf.gz}"
    "${2:-/home/guarracino/git/impg/hprcv2/data/hg002v1.1.pat.PanSN-vs-HG02818_mat_hprc_r2_v1.0.1.sweepga.paf.gz}"
    "${3:-/tmp/p95_clean.tp.paf}"
    "${4:-/tmp/sweepga_clean.tp.paf}"
)

OUTPUT_BASE="${5:-/tmp/bpaf_all_tests}"
NUM_RECORDS="${6:-20000}"

mkdir -p "$OUTPUT_BASE"

echo "###################################################################"
echo "# lib_bpaf - Complete Test Suite"
echo "###################################################################"
echo ""
echo "Running comprehensive tests on ${#TEST_FILES[@]} files..."
echo "Output directory: $OUTPUT_BASE"
echo "Records per file: $NUM_RECORDS"
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
    echo "Usage: $0 [paf1] [paf2] [paf3] [paf4] [output_dir] [num_records]"
    echo ""
    echo "Default files:"
    echo "  1. p95 CIGAR PAF (compressed)"
    echo "  2. sweepga CIGAR PAF (compressed)"
    echo "  3. p95 tracepoint PAF"
    echo "  4. sweepga tracepoint PAF"
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
    
    $COMPREHENSIVE_TEST "$PAF" "$OUT_DIR" 32 edit-distance "$NUM_RECORDS"
    
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
        FIRST_LINE=$(zcat "$PAF" | head -1)
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
- **Compression Strategies:** raw, zigzag-delta, 2d-delta, rle, bit-packed
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
echo ""
echo "Individual Reports:"
for NAME in "${FILE_NAMES[@]}"; do
    echo "  - $OUTPUT_BASE/$NAME/test_report.md"
done
echo ""
