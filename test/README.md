# lib_bpaf Test Suite

Comprehensive test suite for validating lib_bpaf compression, decompression, and random access functionality.

## Test Script

### `accurate_test.sh`

Main test suite that validates all lib_bpaf functionality across 3 tracepoint types (standard, variable, mixed).

**What it tests:**
- CIGAR → Tracepoints encoding
- Tracepoints → BPAF compression (automatic strategy selection)
- BPAF → Tracepoints decompression with MD5 verification
- O(1) random access (Mode A with index, Mode B direct offset)
- Performance and memory efficiency

**Usage:**
```bash
bash test/accurate_test.sh <input_paf> [max_complexity] [complexity_metric]
```

**Parameters:**
- `input_paf`: CIGAR PAF file (gzipped or plain) OR tracepoint PAF file
- `max_complexity`: Encoding complexity threshold (default: 32)
- `complexity_metric`: `edit-distance` or `diagonal-distance` (default: edit-distance)

**Examples:**
```bash
# CIGAR PAF files (will encode to tracepoints)
bash test/accurate_test.sh data/p95.Pinf.aln.paf.gz
bash test/accurate_test.sh data/sweepga.paf.gz

# Tracepoint PAF files (skips encoding, compresses directly)
bash test/accurate_test.sh data/big-from-fg.tp.20k.paf

# Custom parameters
bash test/accurate_test.sh data/p95.paf.gz 10 edit-distance
```

**Features:**
- Auto-detects CIGAR vs tracepoint PAF files
- Handles both gzipped and plain text files
- Tests 20,000 records by default
- Runs 1,000 random seeks per tracepoint type
- Validates data integrity with normalized MD5 checksums

---

## Recent Test Results

All 3 files tested successfully with 100% pass rate (9/9 tests):

### p95 CIGAR PAF (17,184 records)
- Compression: 13-14x
- Seek times: 15-254 μs
- Status: ✅ All types passed

### sweepga CIGAR PAF (20,000 records)
- Compression: 19-26x
- Seek times: 15-295 μs
- Status: ✅ All types passed

### big-from-fg Tracepoint PAF (20,000 records)
- Compression: 2-7x (already tracepoints)
- Seek times: 20-1841 μs
- Status: ✅ All types passed

**Test logs:** `test/logs/test_*.log`

---

## Test Coverage

✅ **Compression:** All tracepoint types, automatic strategy selection, large datasets
✅ **Decompression:** MD5 verification, float normalization, perfect reconstruction
✅ **Random Access:** O(1) seeks, both index modes, consistent accuracy
✅ **Input Formats:** CIGAR PAF, tracepoint PAF, gzipped, plain text
✅ **Performance:** Fast encoding/decoding, low memory usage

---

## Quick Start

```bash
# Clone and build
cd /path/to/lib_bpaf
cargo build --release

# Run test on your PAF file
bash test/accurate_test.sh /path/to/your/file.paf.gz
```

---

## Expected Performance

| Metric | p95 | sweepga | Tracepoint PAF |
|--------|-----|---------|----------------|
| Compression | 13-14x | 19-26x | 2-7x |
| Encoding | 0.1-0.4s | 0.4-0.5s | skipped |
| Decompression | 0.06-0.12s | 0.10-0.12s | 1.3-1.4s |
| Seek (Mode B) | 15-111 μs | 15-112 μs | 20-1611 μs |
| Memory | 5-9 MB | 9 MB | 5 MB |

---

## Input Requirements

**CIGAR PAF files:**
- Must contain `cg:Z:` tags
- Can be gzipped or plain text
- Script will encode to tracepoints automatically

**Tracepoint PAF files:**
- Must contain `tp:Z:` tags
- Can be gzipped or plain text
- Script will skip encoding and compress directly

---

## Troubleshooting

**Test failures:**
- Ensure input file has `cg:Z:` or `tp:Z:` tags
- Check available disk space in `/tmp/` (~500 MB needed)
- Clear old test files: `rm -f /tmp/test.* /tmp/cigar_sample.paf`

**Out of memory:**
- Reduce `NUM_RECORDS` in script (default: 20000)
- Expected memory: 5-10 MB for most datasets

**Compilation errors:**
- Run `cargo build --release` first
- Check that `target/release/liblib_bpaf.rlib` exists
