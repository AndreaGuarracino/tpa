# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tpa` is a Rust library for binary compression and random access of genomic sequence alignments with tracepoints. It provides O(1) random record access through external indexing and supports multiple compression strategies optimized for genomic data.

## Build & Test Commands

### Building
```bash
# Build library
cargo build --release

# Build specific binary
cargo build --release --bin tpa-analyze

# Build all examples
cargo build --release --examples
```

### Testing
```bash
# Run Rust unit/integration tests
cargo test

# Comprehensive test suite (tests dual-strategy combinations across multiple datasets)
# Usage: ./scripts/run_all_tests.sh [num_records] [output_base] [paf1] [paf2] ... [pafN]
./scripts/run_all_tests.sh                    # Use all defaults (50 records, 3 default PAF files)
./scripts/run_all_tests.sh 100                # Test 100 records per file
./scripts/run_all_tests.sh 200 /tmp/results   # Custom output directory
./scripts/run_all_tests.sh 50 /tmp/out file1.paf.gz file2.paf  # Custom files

# Test single file with auto-detection (CIGAR or tracepoint PAF)
./scripts/comprehensive_test.sh input.paf.gz [output_dir] [max_complexity] [metric] [records]

# Example: Test with 1000 records
./scripts/comprehensive_test.sh data.paf.gz /tmp/output 32 edit-distance 1000

# Check if automatic mode selected optimal strategies
./scripts/check-automatic.py [all_results.tsv]
```

The test suite produces:
- Individual test reports: `output_dir/{dataset}/test_report.md`
- Aggregated results: `output_dir/FINAL_REPORT.md`
- TSV data: `output_dir/all_results.tsv` (32 columns including dual strategies, per-stream layers, compression ratios, seek performance, verification status)

### Running Examples
```bash
# Analyze TPA file structure
./target/release/tpa-analyze file.tpa

# View TPA contents as PAF (decompress to stdout)
./target/release/tpa-view file.tpa

# Show which strategies/layers were selected (useful for automatic mode)
./target/release/tpa-view --strategies file.tpa

# Seek performance demo
./target/release/examples/seek_demo file.tpa [record_ids...]

# Offset-based access demo
./target/release/examples/offset_demo file.tpa
```

### Benchmark Tool

Unified seek performance benchmark located in `scripts/`:

```bash
# Build unified benchmark (auto-detects format)
rustc scripts/seek_bench.rs --edition 2021 \
    --extern tpa=target/release/libtpa.rlib \
    --extern noodles=target/release/deps/libnoodles*.rlib \
    -L target/release/deps -o /tmp/seek_bench

# Benchmark any format - auto-detects CIGAR PAF, tracepoint PAF, or TPA
/tmp/seek_bench input.paf.gz 10000 100 100
/tmp/seek_bench input.tpa 10000 100 100
```

## Architecture

### Core Components

**src/lib.rs** - Public API surface
- `TpaReader`: Main reader with O(1) random access via index
- `compress_paf_to_tpa()`: Compression entry point (uses `CompressionConfig` builder)
- `read_*_tracepoints_at_offset()`: Fast standalone seek functions (no TpaReader overhead)
- `build_index()`: Creates `.tpa.idx` files for random access

**src/format.rs** - Data structures and serialization
- `CompressionStrategy` enum: 19 concrete strategies + 1 meta-strategy (`Automatic(level, sample_size)`)
- `CompressionLayer` enum: Zstd, Bgzip, Nocomp - passed explicitly through API
- `TpaHeader`: File metadata (magic, version=1, first/second compression layers, first/second strategy codes, counts, tracepoint type, distance params)
- `TpaFooter`: Crash-safety footer with record/string counts for validation
- `AlignmentRecord`: PAF fields (positions, strand, quality, name IDs)
- `TracepointData` enum: Standard/Variable/Mixed/FastGA representations
- **Dual strategies**: Header stores separate codes for 1st and 2nd values (first_strategy_code, second_strategy_code)

**src/binary.rs** - Binary encoding/decoding
- `SmartDualAnalyzer`: Tests every concrete strategy × layer per stream (19×3 each) and picks independent winners for first/second values
- Strategy-specific encode/decode for tracepoints (19 strategies)
- Varint encoding with delta/zigzag transforms
- Compression layer handling (Zstd/Bgzip/Nocomp)
- Rice and Huffman entropy coding

**src/hybrids.rs** - Advanced compression strategies
- FastPFOR: Patched Frame-of-Reference with exceptions
- Cascaded: Dictionary → RLE chains for low-cardinality data
- Simple8bFull: 16-selector word packing
- SelectiveRLE: Run-length preprocessing with bitmap positions

**src/reader.rs** - High-level reader API
- `TpaReader`: Main reader struct with header, index, string table
- `read_*_tracepoints_at_offset()`: Fast standalone seek functions
- `RecordIterator`: Sequential record iteration

**src/index.rs** - Index building and management
- `TpaIndex`: Index struct with record byte offsets
- `build_index()`: Scans TPA file to create index
- Index save/load for `.tpa.idx` files

**src/utils.rs** - Varint utilities and file helpers
- `write_varint()`, `read_varint()`: Basic varint I/O
- `encode_zigzag()`, `decode_zigzag()`: Signed-to-unsigned transforms
- `open_paf_reader()`: Opens PAF files (handles .gz compression)

### Binary Format

```
[Header] → [StringTable] → [Records...] → [Footer]
```

**Common Prefix (shared by Header and Footer):**
- Magic: "TPA\0" (4 bytes, null-terminated for C compatibility and 32-bit alignment)
- Version: 1 (1 byte)
- Record count: varint
- String count: varint

**Header (Version 1):** Common prefix + header-specific fields:
- First strategy+layer: packed byte (6 bits strategy code 0-18, 2 bits layer)
- Second strategy+layer: packed byte (6 bits strategy code 0-18, 2 bits layer)
- Tracepoint type: 1 byte (Standard/Variable/Mixed/FastGA)
- Complexity metric: 1 byte
- Max complexity/spacing: varint
- Distance parameters: serialized for lib_wfa2

**Footer:** Common prefix + footer_length:
- Footer length: 4 bytes (little-endian u32)

**Index Format (.tpa.idx):**
- Magic: "TPAI" (4 bytes)
- Version: 1 (1 byte)
- Record count: varint
- Byte offsets: varint array (one per record)

## Compression Strategy Selection

### Automatic Strategy
`Automatic(level, sample_size)` - Tests every concrete strategy × layer per stream (19×3=57 combos each), selects independent winners for first/second values.
- `sample_size=1000` (default): Samples first 1,000 records for fast analysis
- `sample_size=0`: Analyzes entire file for best compression (slower)
- `sample_size=N`: Custom sample size

Dual encoding (different strategies for 1st/2nd values) is achieved via `CompressionConfig::dual_strategy()`.

### Concrete Strategies (19 total)
All 19 strategies are considered during automatic analysis:

**High performers:**
1. **2d-delta**: Best for CIGAR-derived alignments (exploits query/target correlation)
2. **stream-vbyte**: SIMD-friendly byte-aligned encoding
3. **simple8b-full**: Complete Simple8b with 16 packing modes + RLE
4. **zigzag-delta**: General-purpose fallback
5. **cascaded**: Multi-level encoding for low-cardinality data

**Specialized:**
- **raw**: Low-complexity data
- **selective-rle**: High-repetition blocks
- **dictionary**: Repeated delta patterns
- **rice**, **huffman**: Entropy coding alternatives

## Key Implementation Details

### Tracepoint Encoding
All strategies preserve **byte-aligned varint encoding** for O(1) random access:
- First values and second values stored separately
- Each value type can use a different compression strategy (dual encoding)
- Delta encoding (when used) maintains position-independence per record
- Compression layers (Zstd/Bgzip/Nocomp) stored per stream and applied after varint encoding
- Layer parameter passed explicitly through all API functions (no thread-local state)

### Random Access Pattern
1. Load index (.tpa.idx) to get byte offsets
2. Seek to record offset in .tpa file
3. Decode tracepoint data using stored strategies from header (first_strategy_code, second_strategy_code)
4. No need to decode previous records

**Dual strategies**: First and second values decoded independently with their respective strategies

### PAF Normalization (Testing)
Test suite uses `scripts/normalize_paf.py` for verification:
- Truncates floats to 3 decimals: `0.993724` → `0.993`
- Handles edge cases: `.0549` → `0.054`, `0` → `0.000`
- Sorts optional fields (13+) alphabetically by tag name to handle field order differences
- TPA outputs `tp:Z:` at the end, which may differ from input PAF field order

## Dependencies

**Core:**
- `zstd`: Compression layer
- `tracepoints`: Tracepoint representations and CIGAR→tracepoint conversion
- `lib_wfa2`: WFA alignment distance parameters

**Testing:**
- `noodles`: BGZF decompression for .paf.gz files
- Python 3 (scripts/normalize_paf.py), md5sum, /usr/bin/time
- **Rust toolchain**: cargo and rustc must be in PATH (sourced via ~/.cargo/env)

## Troubleshooting

### Test script dies at "Building seek test programs"

**Symptom**: `comprehensive_test.sh` hangs or fails silently at "=== Building seek test programs ==="

**Root cause**: `rustc` command not found in PATH

**Solution**: The script now automatically sources `~/.cargo/env` to set up Rust environment. If this fails:
1. Verify Rust is installed: `which rustc`
2. Check that `~/.cargo/env` exists and is valid
3. Manually source before running tests: `source ~/.cargo/env && ./scripts/run_all_tests.sh`

**Prevention**: The script includes proper error checking for rustc compilation failures (scripts/comprehensive_test.sh)

## Recent Development (2025-12-02)

### BGZIP Baseline Comparison

**New Feature:** Added BGZIP-only compression baseline to test suite for comparing TPA format against plain bgzipped tracepoint PAF files.

**Files Added:**
- `examples/seek_bench_bgzip_paf.rs`: Benchmark tool for measuring seek performance on bgzipped PAF using BGZF virtual positions

**Files Modified:**
- `scripts/comprehensive_test.sh`:
  - Builds and runs BGZIP seek benchmark
  - Outputs "bgzip-only" baseline row to TSV with compression metrics and seek performance

**How it works:**
1. BGZIP benchmark parses bgzipped PAF once, capturing BGZF virtual positions for `tp:Z:` fields
2. Randomly seeks to positions using `noodles::bgzf::VirtualPosition`
3. Validates decoded tracepoints against reference PAF
4. Reports: avg_us, stddev_us, decode_ratio, valid_ratio (same format as TPA benchmarks)

**TSV Output:**
The bgzip baseline row uses:
- `compression_strategy`: "bgzip-only"
- `strategy_first/second`: "bgzip"
- `layer_first/second`: "bgzip"
- `tpa_size_bytes`: size of .gz file
- `seek_mode_a_avg_us`: 0 (no high-level reader abstraction for BGZIP)
- `seek_mode_b_avg_us`: BGZF virtual position seek time

**Why Mode B only:**
The BGZIP benchmark uses pre-computed virtual positions and direct `bgzf::io::Reader` access, which is analogous to TPA Mode B (standalone functions with pre-computed offsets). There's no Mode A equivalent because:
- TPA Mode A uses `TpaReader.get_tracepoints()` - a high-level API
- No equivalent "BgzipPafReader" abstraction exists for bgzipped PAF
- The `bgzf::io::Reader` IS the low-level reader (no wrapper needed)

**Comparison Result (100 records, sweepga dataset):**
- BGZIP: 11KB, 44μs seek (Mode B)
- TPA: 12KB, 19μs seek (Mode B)
- TPA provides ~2x faster random access at slight size cost

---

## Earlier Development (2025-11-28)

### Directory Restructure and Cleanup

**Changes Made:**
- Moved test/utility scripts from `test/` to `scripts/` directory
- Removed `tpa-viz` binary (visualization feature removed)
- Added `scripts/check-automatic.py` to verify automatic mode selects optimal strategies

### Header/Footer Format Alignment

**Changes Made:**
- Reordered header fields to match footer: `[magic][version][num_records][num_strings]` prefix is now identical
- Added `write_common_prefix()` and `read_common_prefix()` helper functions in `src/format.rs`
- Both `TpaHeader` and `TpaFooter` now use shared helpers for the common prefix
- Test scripts updated: Added `AUTO_STRATEGIES` array to `scripts/comprehensive_test.sh` to fix automatic mode testing in dual mode

**Benefits:**
- ~25 lines of code reduced through deduplication
- Consistent validation logic for magic/version/counts
- Cleaner separation between common metadata and format-specific fields
- Easier header/footer cross-validation

### Repository Rename: lib_bpaf → tpa

**Changes Made:**
- GitHub repository renamed from `lib_bpaf` to `tpa`
- Updated test scripts to use `tpa_` prefixes instead of `bpaf_`
- TSV column names updated: `bpaf_size_bytes` → `tpa_size_bytes`, `ratio_tp_to_bpaf` → `ratio_tp_to_tpa`, etc.
- Default output directory: `scripts/tpa_all_tests/`
- Rust API already used `TpaReader` (no code changes needed)
- Renamed internal structs: `BinaryPafHeader` → `TpaHeader`, `BinaryPafFooter` → `TpaFooter`

---

## Earlier Development (2025-11-26)

### Code Consolidation and API Cleanup

**Changes Made:**

1. **New CompressionConfig API**: Replaced 4 wrapper functions with single `compress_paf_to_tpa()` using builder pattern
   ```rust
   compress_paf_to_tpa("input.paf", "output.tpa",
       CompressionConfig::new()
           .strategy(CompressionStrategy::ZigzagDelta(3))
           .layer(CompressionLayer::Zstd)
           .from_cigar()
   )?;
   ```

2. **Consolidated zigzag encoding**: Moved `encode_zigzag()` and `decode_zigzag()` to utils.rs for sharing across modules

3. **Consolidated decode match arms**: Combined ZigzagDelta, TwoDimDelta, and OffsetJoint into single match arm (they share identical decode logic)

4. **Removed dead code**: Deleted unused `is_strategy_name()` function, removed incorrect `#[allow(dead_code)]` attributes

**Lines reduced**: 80 lines (5155 → 5075)

---

## Development (2025-11-17)

### Dual Compression Strategy Implementation

**Major Features Completed:**

1. **Dual Strategy Support**: Separate compression strategies for 1st and 2nd values in tracepoint pairs
   - Header stores `first_strategy_code`, `second_strategy_code`, and two compression-layer bytes (version 1 format)
   - Compression layers recorded independently (first/second streams can mix Zstd/Bgzip/Nocomp)
   - When codes differ: creates Dual strategy; when same: single strategy

2. **Unified Automatic Mode**: `Automatic(level, sample_size)` tests 57 combinations per stream (19 strategies × 3 layers)
   - Returns optimal tuple: (first_strategy, first_layer, second_strategy, second_layer)
   - sample_size=1000 (default) for fast sampling; sample_size=0 for entire file
   - Includes Rice and Huffman along with all other concrete codecs

3. **Thread-Local State Removal**: Replaced with explicit parameter passing
   - `layer` parameter added to all public APIs
   - Thread-safe: multiple threads can compress with different layers simultaneously
   - Cleaner architecture: explicit data flow throughout call stack

4. **Tool Consolidation**:
   - Unified seek benchmark: `scripts/seek_bench.rs` auto-detects format (CIGAR PAF, tracepoint PAF, or TPA)
   - Removed redundant tools: compress_paf, tpa_header, separate benchmark tools

5. **Test Suite Enhancements**:
   - Tests all 3,250 combinations per file (3,249 explicit dual combos + 1 automatic)
   - Dual mode: `comprehensive_test.sh` with TEST_MODE=dual parameter
   - Uses `tpa-view --strategies` to extract and track which strategies and layers automatic mode selected
   - Plot visualization: 4×1 vertical layout with improved readability

**Key Files Modified:**
- src/format.rs: Dual strategy support, CompressionLayer enum (21 total strategy variants: 2 meta + 19 concrete)
- src/binary.rs: SmartDualAnalyzer with per-stream 57-combo testing (19 strategies × 3 layers)
- src/lib.rs: Dual strategy APIs, explicit layer parameter
- scripts/comprehensive_test.sh: Dual mode support, strategy tracking
- scripts/run_all_tests.sh: Flexible parameterization

**Dataset-Specific Insights:**
- CIGAR-derived data (p95, sweepga): TwoDimDelta often optimal for 2nd values
- Native tracepoint data (bigfg): Simple strategies (Raw, Simple8) often win
- Compression layer choice is dataset-dependent (Zstd vs Nocomp varies)

---

## Development (2025-12-12)

### Code Consolidation - Phase 1 & 2

**Goal:** Make the codebase more concise, maintainable, and readable while preserving efficiency.

**Phase 1: Format Module Consolidation** (`src/format.rs`)
- Simplified `zstd_level()` by combining 19 match arms into one using `|` patterns
- Added `display_name()` helper method for strategy names
- Simplified `Display` impl from 66 lines to 14 lines using the new helper
- **Reduction:** ~28 lines

**Phase 2: Example File Consolidation**
- Created `examples/common/mod.rs` with shared utilities:
  - `parse_reference()`: Parse tracepoints from PAF for validation (~82 lines)
  - `validate_tracepoints()`: Validate decoded tracepoints against reference
  - `tracepoint_len()`: Count tracepoint elements
- Updated `seek_bench_reader.rs` to use shared module (204 → 93 lines)
- Updated `seek_bench_direct.rs` to use shared module (449 → 359 lines)
- **Reduction:** ~66 lines (with shared code factored out)

**Phase 3: Reader Mode Unification** (reader.rs)
- Added helper functions: `err_not_initialized()`, `err_out_of_bounds()`, `skip_record_header()`
- Replaced duplicated error creation and varint skipping code
- **Reduction:** ~9 lines

**Phase 4: Binary Module Encode/Decode Consolidation** (binary.rs)
- Added `reconstruct_from_zigzag_deltas()` helper for Rice/Huffman decode
- Combined `Raw` and `HybridRLE` encode/decode arms (identical varint logic)
- Simplified Rice and Huffman decode blocks from ~15 lines each to ~3 lines
- **Reduction:** ~22 lines

**Phase 5: Decompress Function Consolidation** (lib.rs)
- Replaced inline PAF writing in `decompress_bgzf_tpa()` with `write_paf_line_with_tracepoints()`
- Removed ~45 lines of duplicate code
- **Reduction:** ~45 lines

**Phase 6: Scripts Consolidation**
- Reviewed scripts; already well-organized
- `run_all_tests.sh` calls `comprehensive_test.sh` directly
- Helper functions specific to comprehensive_test.sh, no duplication

**Total Reduction:** ~236 lines across all phases with cleaner, more maintainable code

---

## Development (2025-12-11)

### Verification and Seek Benchmark Fixes

**Problem:** Test suite was reporting `verification_passed=no` for automatic mode strategies (Rice/TwoDimDelta), and seek benchmarks were panicking with "Record id out of bounds" errors.

**Root Causes:**
1. **PAF field order difference:** TPA outputs `tp:Z:` at the end of optional fields, while input PAF may have it in a different position. The normalize script wasn't accounting for this.
2. **Float precision boundary:** TPA stores floats as f32, but original PAF has f64 values. Boundary cases like `0.994499921799` vs `0.9944999` round differently without proper handling.
3. **Stale index causing bounds errors:** Seek benchmark used CLI argument for record count, but stale indexes could have fewer records.

**Fixes:**
1. **`scripts/normalize_paf.py`:**
   - Added optional field sorting (fields 13+) by tag name before comparison
   - Added f32 conversion before rounding: uses `struct.pack/unpack` to convert floats to f32 precision before rounding to 3 decimals
   - This ensures both original (f64) and decompressed (f32) files normalize to the same values

2. **`examples/seek_bench_direct.rs`:** Uses `reader.header().num_records()` instead of CLI argument
   - Prevents bounds errors when index is stale
   - CLI argument kept for backward compatibility but ignored

**Example of f32 fix:**
- Original: `bi:f:0.994499921799` -> f32: `0.994499921798706` -> round: `0.994`
- Decomp: `bi:f:0.9944999` -> f32: `0.994499921798706` -> round: `0.994`
- Result: Both match after normalization

**Verification:** After fixes, all automatic mode strategies (Rice/TwoDimDelta, bgzip-all) pass verification with matching MD5 checksums.

---

## Development (2025-12-10)

### BGZIP Whole-File Compression Mode

**New Feature:** Added alternative compression mode where the entire TPA file is wrapped in BGZIP for cross-record compression context and block-level random access.

**Trade-offs:**
| Aspect | Per-Record Layers (classic) | BGZIP Whole-File (new) |
|--------|---------------------------|------------------------|
| Compression | Good | Better (cross-record context) |
| Random Access | Fast (direct seek) | Slower (64KB block decompress) |
| Index | Small (varint offsets) | Larger (u64 virtual positions) |

**Usage:**
```rust
// Enable BGZIP whole-file mode
tpa::compress_paf_to_tpa(
    "input.paf",
    "output.tpa",
    tpa::CompressionConfig::new()
        .strategy(tpa::CompressionStrategy::ZigzagDelta(3))
        .whole_file_bgzip()  // <-- Enable BGZIP mode
        .from_cigar(),
)?;

// With automatic strategy selection
tpa::compress_paf_to_tpa(
    "input.paf",
    "output.tpa",
    tpa::CompressionConfig::new()
        .strategy(tpa::CompressionStrategy::Automatic(3, 100))
        .whole_file_bgzip()
        .from_cigar(),
)?;
```

**Key Implementation Details:**
- Same `.tpa` extension with auto-detection via BGZF magic bytes
- `TpaReader::open()` automatically detects classic vs BGZIP mode
- `TpaReader::is_bgzf_mode()` returns true for BGZIP-wrapped files
- Index uses BGZF virtual positions (64-bit: block_offset << 16 | in_block_offset)
- Per-record compression layers set to `Nocomp` (BGZF handles compression)
- `SmartDualAnalyzer::for_whole_file_bgzip()` only tests strategies, not layers

**Standalone Functions (Mode B) for BGZIP:**
- `read_standard_tracepoints_at_vpos()`: Seek to BGZF virtual position and decode standard tracepoints
- `read_variable_tracepoints_at_vpos()`: Seek to BGZF virtual position and decode variable tracepoints
- `read_mixed_tracepoints_at_vpos()`: Seek to BGZF virtual position and decode mixed representations
- Re-exported `noodles::bgzf` module for direct BGZF reader access
- `examples/seek_bench_direct.rs`: Auto-detects mode and uses `_at_offset()` or `_at_vpos()` functions

**Files Modified:**
- `src/format.rs`: `detect_bgzf()`, `CompressionConfig.whole_file_bgzip()`, TpaHeader updates
- `src/index.rs`: `IndexType` enum (RawOffset vs VirtualPosition)
- `src/lib.rs`: `compress_with_bgzip_wrapper()`, `is_tpa_file()` BGZF support, re-exports `bgzf` module
- `src/reader.rs`: Dual-mode `TpaReader` (classic + BGZF), BGZF-aware `_at_vpos()` functions
- `src/binary.rs`: `SmartDualAnalyzer::for_whole_file_bgzip()`
- `scripts/comprehensive_test.sh`: Runs Mode B for both classic and BGZIP modes
- `src/bin/tpa-analyze.rs`: BGZF mode display
- `src/bin/tpa-view.rs`: BGZF mode display

**Test Results (100 records, sweepga dataset):**
- Classic (ZigzagDelta+Zstd): 12,236 bytes
- BGZIP (ZigzagDelta): 9,129 bytes (74.6% of classic)
- BGZIP (Automatic): 7,416 bytes (60.6% of classic) - selected Rice/TwoDimDelta

### Test Suite Fix: BGZIP Mode Seek Benchmarks

**Issue:** Seek benchmark Mode B (standalone offset functions) failed for BGZIP whole-file mode because the index contains BGZF virtual positions, not raw byte offsets.

**Fix:** Updated `scripts/comprehensive_test.sh` to:
1. Skip Mode B benchmark for BGZIP whole-file mode (set to "NA")
2. Use Mode A (TpaReader) results for decode_ratio and valid_ratio
3. TSV now correctly shows seek metrics for all modes

**Modified:** `scripts/comprehensive_test.sh` (lines 550-567)

## Common Patterns

### Adding a New Compression Strategy

1. Add enum variant to `CompressionStrategy` in `src/format.rs`:
   ```rust
   pub enum CompressionStrategy {
       // ...
       MyNewStrategy(i32), // parameter is zstd level
   }
   ```

2. Add encode/decode in `src/binary.rs`:
   ```rust
   CompressionStrategy::MyNewStrategy(_) => {
       // Encode logic - must produce byte-aligned varint stream
   }
   ```

3. Add to strategy list in `src/binary.rs` (~line 160)

4. Add to test script `scripts/comprehensive_test.sh` STRATEGIES array

5. Update `src/lib.rs` strategy resolution if meta-strategy

### Reading Performance-Critical Code

When tracepoints must be decoded in tight loops:
- Use `read_*_tracepoints_at_offset()` functions (src/lib.rs)
- Pre-compute offsets from index
- Avoid `TpaReader` overhead
- See `examples/offset_demo.rs` for pattern

### Verification Failures

If round-trip tests fail:
1. Check float normalization: `python3 ./scripts/normalize_paf.py file.paf | head`
2. Verify strategy preserves byte-alignment
3. Test with smaller dataset to isolate issue
4. Check that delta encoding is position-independent per record
