# lib_bpaf

Binary format for genomic sequence alignments with tracepoints.

## Features

- **O(1) random access**: External index for instant record lookup
- **Fast varint compression**:
  - **Automatic (default)**: Samples records and tests every concrete strategy × compression layer per stream (19×3 per stream), then locks in the best first/second pair; configurable sample size (default 1000, 0 = entire file)
  - **ZigzagDelta**: Delta + zigzag transform + varint + zstd
  - **Raw**: Plain varints + zstd
  - **Rice / Huffman**: Block-local entropy coding over zigzag deltas, byte-aligned for random seeks
- **Tracepoint support**: Standard, Mixed, Variable, and FastGA representations
- **String deduplication**: Shared sequence name table
- **Byte-aligned encoding**: Enables extremely fast tracepoint extraction
- **Crash-safety footer**: Files carry a footer written at close; missing footers are rejected on read

## Format

```
[Header] → [StringTable] → [Records] → [Footer]
```

### Header (metadata + strategy)
- Magic: `BPAF` (4 bytes)
- Version: `1` (1 byte)
- Strategy bytes (2): bits 7–6 = layer (`0=Zstd, 1=Bgzip, 2=Nocomp`), bits 5–0 = strategy code (`0-18`)
- Record count: varint
- String count: varint
- Tracepoint type: `1` byte
- Complexity metric: `1` byte
- Max complexity / spacing: varint
- Distance parameters: serialized to match `lib_wfa2::Distance`

### Footer (written on close)
- Magic: `BPAF` (4 bytes)
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
lib_bpaf = { git = "https://github.com/AndreaGuarracino/lib_bpaf" }
```

## Quick Start

### Read with random access

```rust
use lib_bpaf::{BpafReader, TracepointType};

// Open with index for O(1) record access
let mut reader = BpafReader::open("alignments.bpaf")?;
println!("Total records: {}", reader.len());

// Jump to any record instantly
let record = reader.get_alignment_record(1000)?;
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
use lib_bpaf::{read_standard_tracepoints_at_offset_with_strategies,
               read_variable_tracepoints_at_offset,
               read_mixed_tracepoints_at_offset,
               CompressionStrategy, CompressionLayer};
use std::fs::File;

// Open file once (reuse for multiple seeks)
let mut file = File::open("alignments.bpaf")?;

// Pre-computed offsets, strategy, and layer from index/header
let offset = 123456;
let first_strategy = CompressionStrategy::ZigzagDelta(3);
let second_strategy = CompressionStrategy::ZigzagDelta(3);
let first_layer = CompressionLayer::Zstd; // or read from BinaryPafHeader
let second_layer = CompressionLayer::Zstd;

// Direct tracepoint decoding - no BpafReader overhead
let standard_tps = read_standard_tracepoints_at_offset_with_strategies(
    &mut file,
    offset,
    first_strategy,
    second_strategy,
    first_layer,
    second_layer,
)?;
let variable_tps = read_variable_tracepoints_at_offset(&mut file, offset)?;
let mixed_tps = read_mixed_tracepoints_at_offset(&mut file, offset)?;
```

**Use when**:
- You have pre-computed tracepoint offsets (from index)
- Only need tracepoints (not full alignment records)
- Processing many records in tight loops

### Sequential iteration

```rust
for record in reader.iter_records() {
    let record = record?;
    println!("{} → {}", record.query_name_id, record.target_name_id);
}
```

### Compression

```rust
use lib_bpaf::{compress_paf_to_bpaf, CompressionConfig, CompressionStrategy, CompressionLayer};

// Automatic (default): samples 1000 records to find best strategy
compress_paf_to_bpaf("alignments.paf", "alignments.bpaf", CompressionConfig::new())?;

// Automatic with custom sample size (500 records)
compress_paf_to_bpaf(
    "alignments.paf",
    "alignments.bpaf",
    CompressionConfig::new().strategy(CompressionStrategy::Automatic(3, 500)),
)?;

// Automatic with entire file analysis (sample_size = 0)
compress_paf_to_bpaf(
    "alignments.paf",
    "alignments.bpaf",
    CompressionConfig::new().strategy(CompressionStrategy::Automatic(3, 0)),
)?;

// ZigzagDelta: Delta + zigzag transform + varint + zstd
compress_paf_to_bpaf(
    "alignments.paf",
    "alignments.bpaf",
    CompressionConfig::new().strategy(CompressionStrategy::ZigzagDelta(3)),
)?;

// Dual strategy: different strategies for first/second values
compress_paf_to_bpaf(
    "alignments.paf",
    "alignments.bpaf",
    CompressionConfig::new()
        .dual_strategy(CompressionStrategy::Raw(3), CompressionStrategy::TwoDimDelta(3)),
)?;

// From CIGAR input (converts to tracepoints)
compress_paf_to_bpaf(
    "alignments.paf",
    "alignments.bpaf",
    CompressionConfig::new().from_cigar(),
)?;
```

**Strategy guide:**
- **Automatic (default)**: Best for most use cases. Parameters: `(level, sample_size)` where sample_size=1000 is default, 0 = analyze entire file
- **ZigzagDelta**: Use when tracepoint values are mostly increasing
- **TwoDimDelta**: Best for CIGAR-derived alignments (exploits query/target correlation)
- **Raw**: Use when tracepoint values jump frequently
- **Rice/Huffman**: Entropy coding for skewed distributions

### Index management

```rust
use lib_bpaf::{build_index, BpafIndex};

// Build index for random access
let index = build_index("alignments.bpaf")?;
index.save("alignments.bpaf.idx")?;

// Load existing index
let index = BpafIndex::load("alignments.bpaf.idx")?;
```

## Examples

```bash
# Build examples
cargo build --release --examples

# Show first 5 records
./target/release/examples/seek_demo alignments.bpaf

# O(1) random access demo
./target/release/examples/seek_demo alignments.bpaf 0 100 500 1000

# Offset-based access demo
./target/release/examples/offset_demo alignments.bpaf
```

## Index Format

`.bpaf.idx` file structure:
```
Magic:     BPAI (4 bytes)
Version:   1 (1 byte)
Count:     varint (number of records)
Offsets:   varint[] (byte positions)
```

Index enables O(1) random access without file scanning.

## License

MIT
