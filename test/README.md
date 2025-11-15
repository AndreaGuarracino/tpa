# lib_bpaf Test Suite

Comprehensive testing framework for lib_bpaf compression strategies and seek performance.

## Quick Start

### Test a Single File

```bash
# Auto-detects input type (CIGAR compressed/uncompressed, or tracepoints)
./test/comprehensive_test.sh input.paf.gz

# With custom parameters
./test/comprehensive_test.sh input.paf.gz /tmp/output 32 edit-distance 20000
```

### Test Multiple Files

```bash
# Uses default test files
./test/run_all_tests.sh

# With custom files
./test/run_all_tests.sh p95.paf.gz sweepga.paf.gz p95.tp.paf sweepga.tp.paf /tmp/output 20000
```

---

## Test Scripts

### 1. `comprehensive_test.sh` - Universal Single-File Test

**The main testing script** - handles any PAF input automatically.

**Auto-detection:**
- ✓ CIGAR PAF (compressed `.paf.gz` or uncompressed `.paf`)
- ✓ Tracepoint PAF (with `tp:Z:` tags)

**What it tests:**
- **Tracepoint types:** standard, variable, mixed (if CIGAR input)
- **Compression strategies:** raw, zigzag-delta, 2d-delta, rle, bit-packed
- **Performance:** compression/decompression time and memory
- **Seek modes:** Mode A (BpafReader) and Mode B (standalone functions)
- **Verification:** lossless round-trip with 3-decimal float normalization

**Usage:**
```bash
./test/comprehensive_test.sh <input.paf[.gz]> [output_dir] [max_complexity] [metric] [records]
```

**Parameters:**
- `input.paf[.gz]`: Any PAF file (REQUIRED)
- `output_dir`: Output directory (default: `/tmp/bpaf_test_output`)
- `max_complexity`: Max tracepoint complexity (default: `32`)
- `metric`: Complexity metric (default: `edit-distance`)
- `records`: Number of records to test (default: `20000`)

**Examples:**
```bash
# Test compressed CIGAR PAF
./test/comprehensive_test.sh data/alignments.p95.paf.gz

# Test tracepoint PAF with custom output
./test/comprehensive_test.sh data/alignments.tp.paf /tmp/mytest

# Test with 50K records
./test/comprehensive_test.sh data/big.paf.gz /tmp/output 32 edit-distance 50000
```

**Output:**
- Individual test files in `output_dir/`
- Markdown report: `output_dir/test_report.md`
- Compressed files: `output_dir/{type}_{strategy}.bpaf`

---

### 2. `run_all_tests.sh` - Multi-File Test Wrapper

**Runs comprehensive tests on multiple files** and aggregates results.

**Default test suite:**
1. p95 CIGAR PAF (compressed)
2. sweepga CIGAR PAF (compressed)
3. p95 tracepoint PAF
4. sweepga tracepoint PAF

**Usage:**
```bash
./test/run_all_tests.sh [paf1] [paf2] [paf3] [paf4] [output_dir] [records]
```

**Examples:**
```bash
# Use defaults
./test/run_all_tests.sh

# Custom files
./test/run_all_tests.sh \
    /data/file1.paf.gz \
    /data/file2.paf \
    /data/file3.tp.paf \
    /data/file4.tp.paf \
    /tmp/all_results \
    20000
```

**Output:**
- Per-file results: `output_dir/{filename}/test_report.md`
- Aggregated report: `output_dir/FINAL_REPORT.md`
- All test artifacts organized by filename

---

## Legacy Scripts (Deprecated)

### `accurate_test.sh` - Original tracepoint type testing

**Status:** Replaced by `comprehensive_test.sh`

This script is now **deprecated**. Use `comprehensive_test.sh` instead, which includes all its functionality plus:
- Auto-detection of input type
- All compression strategies (not just default)
- Memory measurements
- Better reporting

### `strategy_evaluation.sh` - Original strategy comparison

**Status:** Replaced by `comprehensive_test.sh` + `run_all_tests.sh`

This script is now **deprecated**. The new unified approach provides:
- Support for both CIGAR and tracepoint inputs
- All tracepoint types (not just standard)
- Both seek modes (not just Mode A)
- Better organization and reporting

---

## Utility Scripts

### `normalize_paf.pl`

Normalizes PAF float fields to exactly 3 decimal places for verification.

**Handles all float formats:**
```perl
0.993724        →  0.993    # Normal decimals (truncate)
.0549           →  0.054    # Leading dot (add zero)
0               →  0.000    # Integer (add decimals)
1               →  1.000    # Integer (add decimals)
```

**Usage:**
```bash
./test/normalize_paf.pl input.paf > normalized.paf

# With MD5 verification
./test/normalize_paf.pl file1.paf | md5sum
./test/normalize_paf.pl file2.paf | md5sum
```

---

## Test Results Interpretation

### Seek Modes

**Mode A - BpafReader with index:**
- General-purpose API
- Includes index lookup overhead
- Typical use case for applications
- Times: ~15-25 μs

**Mode B - Standalone functions:**
- Direct file I/O functions
- Pre-computed offsets
- Ultimate performance
- Times: ~10-20 μs

### Compression Strategies

**Raw:**
- Best for: Direct tracepoint data with low redundancy
- Speed: Medium
- Ratio: Best on unstructured data (6-7x)

**ZigzagDelta (default):**
- Best for: General-purpose compression
- Speed: Medium
- Ratio: Good on most data (2-6x)

**2D-Delta:**
- Best for: CIGAR-derived PAF with query/target correlation
- Speed: Medium
- Ratio: Best on CIGAR data (3-3.5x, 16-20% better than zigzag)

**RLE:**
- Best for: Highly repetitive alignment blocks
- Speed: Slower
- Ratio: Variable (needs optimization)

**BitPacked:**
- Best for: Narrow value ranges
- Speed: Fastest compression
- Ratio: Good when values fit in 8-16 bits

---

## Dependencies

**Required:**
- Rust toolchain (for compiling test programs)
- `cigzip` binary (auto-built from `../cigzip`)
- `lib_bpaf` library (auto-built)

**System tools:**
- `perl` (for float normalization)
- `md5sum` (for verification)
- `bc` (for calculations)
- `/usr/bin/time` (for memory measurements)

**Auto-built:**
- Seek test programs (compiled on first run)
- cigzip and lib_bpaf (if not present)

---

## Test Data Recommendations

**For quick tests (5-10 minutes):**
- Use 10,000-20,000 records
- Single file with `comprehensive_test.sh`

**For comprehensive evaluation (20-30 minutes):**
- Use 20,000+ records per file
- Multiple files with `run_all_tests.sh`
- Include both CIGAR and tracepoint inputs

**For production benchmarking:**
- Use full files (100K+ records)
- Test on representative data from your pipeline
- Compare CIGAR vs tracepoint performance

---

## Examples

### Example 1: Quick test of a CIGAR PAF

```bash
# Test first 10K records
./test/comprehensive_test.sh alignments.p95.paf.gz /tmp/test 32 edit-distance 10000

# View results
cat /tmp/test/test_report.md
```

### Example 2: Full evaluation of multiple datasets

```bash
# Run complete test suite
./test/run_all_tests.sh \
    /data/human.p95.paf.gz \
    /data/human.sweepga.paf.gz \
    /data/human.p95.tp.paf \
    /data/human.sweepga.tp.paf \
    /tmp/human_eval \
    50000

# View aggregated results
cat /tmp/human_eval/FINAL_REPORT.md
```

### Example 3: Compare strategies for a specific file

```bash
# Test all strategies on tracepoint PAF
./test/comprehensive_test.sh my_alignments.tp.paf /tmp/strategy_test

# Compare results
grep "| zigzag-delta" /tmp/strategy_test/test_report.md
grep "| 2d-delta" /tmp/strategy_test/test_report.md
grep "| raw" /tmp/strategy_test/test_report.md
```

---

## Troubleshooting

**Build errors:**
```bash
# Manually rebuild cigzip
cd ../cigzip && cargo build --release

# Manually rebuild lib_bpaf
cd .. && cargo build --release
```

**Seek test failures:**
```bash
# Check if binaries exist
ls -la target/release/seek_mode_* /tmp/seek_mode_*

# Rebuild manually
cd test && rm /tmp/seek_mode_* && ./comprehensive_test.sh <file>
```

**Verification failures:**
```bash
# Check normalization
./test/normalize_paf.pl input.paf | head
./test/normalize_paf.pl decompressed.paf | head

# Manual diff
diff <(./test/normalize_paf.pl input.paf) <(./test/normalize_paf.pl decomp.paf)
```

---

## Contributing

When adding new tests:
1. Add to `comprehensive_test.sh` (main test logic)
2. Update this README with new test descriptions
3. Ensure backward compatibility with existing test files
4. Add examples for new functionality

---

**Last Updated:** 2025-11-14
**Version:** 2.0 (Unified testing framework)
