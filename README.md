# tpa

TracePoint Alignment (TPA) format - binary format for efficient storage and random access of sequence alignments with tracepoints.

## Features

- **O(1) random access**: External index for instant record lookup
- **Fast varint compression**:
  - **Automatic (default)**: Instant lookup-based strategy selection from tracepoint type and complexity metric — no sampling overhead
  - **Benchmark**: Exhaustive testing of every strategy × compression layer per stream (18×3 per stream), selects optimal first/second pair; configurable sample size (default 10000, 0 = entire file)
  - **ZigzagDelta**: Delta + zigzag transform + varint + zstd
  - **Raw**: Plain varints + zstd
  - **Rice / Huffman**: Block-local entropy coding over zigzag deltas, byte-aligned for random seeks
- **Tracepoint support**: Standard, Mixed, Variable, and FastGA representations
- **String deduplication**: Shared sequence name table
- **Byte-aligned encoding**: Enables extremely fast tracepoint extraction
- **BGZIP all-records mode**: Optional whole-file BGZIP wrapping for better cross-record compression
- **Crash-safety footer**: Files carry a footer written at close; missing footers are rejected on read

## Format

```
[Header] → [StringTable] → [Records] → [Footer]
```

### Header (metadata + strategy)
- Magic: `TPA\0` (4 bytes)
- Version: `1` (1 byte)
- Record count: varint
- String count: varint
- First strategy+layer: packed byte — bits 7–6 = layer (`0=Zstd, 1=Bgzip, 2=Nocomp`), bits 5–0 = strategy code (`0-17`)
- Second strategy+layer: packed byte (same format)
- Tracepoint type: `1` byte
- Complexity metric: `1` byte
- Max complexity / spacing: varint
- Distance parameters: serialized to match `lib_wfa2::Distance`

### Footer (written on close)
- Magic: `TPA\0` (4 bytes)
- Version: `1` (1 byte)
- Record count: varint
- String count: varint
- Footer length: `u32` little-endian trailer

Readers validate header/footer agreement and fail fast if the footer is missing or truncated.

### Records (per alignment)
- **PAF fields**: varints for positions, 1-byte strand/quality
- **Tracepoints**: Byte-aligned varint-encoded (first_values, second_values) pairs
- **Tags**: Optional key-value pairs

### String Table
- Follows the header and stores deduplicated sequence names
- Per string: name length (varint) + name bytes (UTF-8) + sequence length (varint)

## Installation

```toml
[dependencies]
tpa = { git = "https://github.com/AndreaGuarracino/tpa" }
```

## Quick Start

### Read with random access

```rust
use tpa::{TpaReader, TracepointData};

// Open with index for O(1) record access
let mut reader = TpaReader::new("alignments.tpa")?;
println!("Total records: {}", reader.len());

// Jump to any record instantly
let record = reader.get_compact_record(1000)?;
let (tracepoints, _, _) = reader.get_tracepoints(1000)?;

match &tracepoints {
    TracepointData::Standard(tps) => println!("{} tracepoints", tps.len()),
    TracepointData::Fastga(tps) => println!("{} FastGA traces", tps.len()),
    TracepointData::Variable(tps) => println!("{} variable segments", tps.len()),
    TracepointData::Mixed(items) => println!("{} mixed items", items.len()),
}
```

### Fast tracepoint access

For maximum speed when you have pre-computed offsets:

```rust
use tpa::{read_standard_tracepoints_at_offset_with_strategies,
               CompressionStrategy, CompressionLayer};
use std::fs::File;

let mut file = File::open("alignments.tpa")?;

// Strategies and layers from header; offset from index
let tps = read_standard_tracepoints_at_offset_with_strategies(
    &mut file,
    123456, // byte offset from index
    CompressionStrategy::ZigzagDelta(3),
    CompressionStrategy::ZigzagDelta(3),
    CompressionLayer::Zstd,
    CompressionLayer::Zstd,
)?;
// Also available: read_variable_tracepoints_at_offset, read_mixed_tracepoints_at_offset
```

Use when you have pre-computed offsets from the index and need tracepoints in tight loops.

### Sequential iteration

```rust
for record in reader.iter_records()? {
    let record = record?;
    println!("{} → {}", record.query_name_id, record.target_name_id);
}
```

### Compression

```rust
use tpa::{paf_to_tpa, CompressionConfig, CompressionStrategy, CompressionLayer};

// Automatic (default): lookup-based strategy selection
paf_to_tpa("alignments.paf", "alignments.tpa", CompressionConfig::new())?;

// Benchmark: exhaustive testing with 500-record sample
paf_to_tpa(
    "alignments.paf",
    "alignments.tpa",
    CompressionConfig::new().strategy(CompressionStrategy::Benchmark(3, 500)),
)?;

// Benchmark with entire file analysis (sample_size = 0)
paf_to_tpa(
    "alignments.paf",
    "alignments.tpa",
    CompressionConfig::new().strategy(CompressionStrategy::Benchmark(3, 0)),
)?;

// ZigzagDelta: Delta + zigzag transform + varint + zstd
paf_to_tpa(
    "alignments.paf",
    "alignments.tpa",
    CompressionConfig::new().strategy(CompressionStrategy::ZigzagDelta(3)),
)?;

// Dual strategy: different strategies for first/second values
paf_to_tpa(
    "alignments.paf",
    "alignments.tpa",
    CompressionConfig::new()
        .dual_strategy(CompressionStrategy::Raw(3), CompressionStrategy::TwoDimDelta(3)),
)?;

// From CIGAR input (converts to tracepoints)
paf_to_tpa(
    "alignments.paf",
    "alignments.tpa",
    CompressionConfig::new().from_cigar(),
)?;

// BGZIP all-records mode (better compression, block-level random access)
paf_to_tpa(
    "alignments.paf",
    "alignments.tpa",
    CompressionConfig::new().all_records(),
)?;
```

**Strategy guide:**
- **Automatic (default)**: Selection based on tracepoint type and complexity metric. Parameter: `(level)`. Best for most use cases
- **Benchmark**: Exhaustive testing of all strategy×layer combinations. Parameters: `(level, sample_size)` where sample_size=10000 is default, 0 = analyze entire file
- **ZigzagDelta**: Use when tracepoint values are mostly increasing
- **TwoDimDelta**: Best for CIGAR-derived alignments (exploits query/target correlation)
- **Raw**: Use when tracepoint values jump frequently
- **Rice/Huffman**: Entropy coding for skewed distributions

### Index management

```rust
use tpa::{build_index, TpaIndex};

// Build index for random access
let index = build_index("alignments.tpa")?;
index.save("alignments.tpa.idx")?;

// Load existing index
let index = TpaIndex::load("alignments.tpa.idx")?;
```

## CLI Tools

```bash
cargo build --release

# Inspect TPA file structure (header, strategies, record count)
./target/release/tpa-analyze file.tpa

# Decompress TPA back to PAF
./target/release/tpa-view file.tpa

# Show selected strategies/layers
./target/release/tpa-view --strategies file.tpa
```

## Examples

```bash
cargo build --release --examples

# O(1) random access demo
./target/release/examples/seek_demo alignments.tpa 0 100 500 1000

# Offset-based access demo
./target/release/examples/offset_demo alignments.tpa
```

## Index Format

`.tpa.idx` file structure:
```
Magic:     TPAI (4 bytes)
Version:   1 (1 byte)
Count:     varint (number of records)
Offsets:   varint[] (byte positions)
```

Index enables O(1) random access without file scanning.

## License

MIT
