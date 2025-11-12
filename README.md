# lib_bpaf

Binary format for genomic sequence alignments with tracepoints.

## Features

- **O(1) random access**: External index for instant record lookup
- **Fast varint compression**:
  - **Automatic (default)**: Analyzes data to choose delta vs raw encoding + varint + zstd
  - **DeltaVarintZstd**: Delta encoding + varint + zstd
  - **VarintZstd**: No delta encoding + varint + zstd
- **Tracepoint support**: Standard, Mixed, Variable, and FastGA representations
- **String deduplication**: Shared sequence name table
- **Byte-aligned encoding**: Enables extremely fast tracepoint extraction

## Format

```
[Header] → [Records] → [StringTable]
```

### Header (6+ bytes)
- Magic: `BPAF` (4 bytes)
- Version: `1` (1 byte)
- Strategy: `0-1` (1 byte) - 0=varint, 1=varint-raw
- Record count: varint
- String count: varint

### Records (per alignment)
- **PAF fields**: varints for positions, 1-byte strand/quality
- **Tracepoint metadata**: type (1 byte), complexity metric (1 byte), max_complexity (varint)
- **Tracepoints**: Byte-aligned varint-encoded (first_values, second_values) pairs
- **Tags**: Optional key-value pairs

### String Table
- Deduplicated sequence names (length + UTF-8 bytes)

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
let (tracepoints, _, _, _) = reader.get_tracepoints(1000)?;

match &tracepoints {
    TracepointType::Standard(tps) => println!("{} tracepoints", tps.len()),
    TracepointType::Fastga(tps) => println!("{} FastGA traces", tps.len()),
    TracepointType::Variable(tps) => println!("{} variable segments", tps.len()),
    TracepointType::Mixed(items) => println!("{} mixed items", items.len()),
}
```

### Read with file offsets (faster open)

```rust
// Skip index loading if you store offsets externally
let mut reader = BpafReader::open_without_index("alignments.bpaf")?;

// Access by byte offset
let offset = 123456;
let record = reader.get_alignment_record_at_offset(offset)?;
```

### Sequential iteration

```rust
for record in reader.iter_records() {
    let record = record?;
    println!("{} → {}", record.query_name_id, record.target_name_id);
}
```

### Compression

```rust
use lib_bpaf::{compress_paf_with_tracepoints, CompressionStrategy};

// Automatic (default): Analyzes data to choose delta vs raw + varint + zstd
// - Samples first 1000 records to determine optimal encoding
// - Handles all tracepoint types automatically
// - Enables O(1) random tracepoint access
compress_paf_with_tracepoints("alignments.paf", "alignments.bpaf", CompressionStrategy::Automatic(3))?;

// DeltaVarintZstd: Delta encoding + varint + zstd
// - Always uses delta encoding for tracepoints
// - Works well when values are naturally small or monotonic
// - Enables O(1) random tracepoint access
compress_paf_with_tracepoints("alignments.paf", "alignments.bpaf", CompressionStrategy::DeltaVarintZstd(3))?;

// VarintZstd: No delta + varint + zstd
// - All types: raw first values
// - Use if delta encoding doesn't help your data
// - Also enables O(1) random tracepoint access
compress_paf_with_tracepoints("alignments.paf", "alignments.bpaf", CompressionStrategy::VarintZstd(3))?;
```

**Strategy guide:**
- **Automatic (default)**: Best for most use cases, analyzes data to choose optimal encoding
- **DeltaVarintZstd**: Use when you know delta encoding helps your data
- **VarintZstd**: Test if delta encoding creates too many unique symbols for your data

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
