# lib_bpaf

Binary format for genomic sequence alignments with tracepoints.

## Features

- **O(1) random access**: External index for instant record lookup
- **Fast varint compression**:
  - **Automatic (default)**: Samples data to choose between raw or delta+zigzag encoding
  - **ZigzagDelta**: Delta + zigzag transform + varint + zstd
  - **Raw**: Plain varints + zstd
- **Tracepoint support**: Standard, Mixed, Variable, and FastGA representations
- **String deduplication**: Shared sequence name table
- **Byte-aligned encoding**: Enables extremely fast tracepoint extraction

## Format

```
[Header] → [StringTable] → [Records]
```

### Header (metadata + strategy)
- Magic: `BPAF` (4 bytes)
- Version: `1` (1 byte)
- Strategy: `0-1` (1 byte) - 0=Raw, 1=ZigzagDelta (Automatic resolves to one of these)
- Record count: varint
- String count: varint
- Tracepoint type: `1` byte
- Complexity metric: `1` byte
- Max complexity / spacing: varint
- Distance parameters: serialized to match `lib_wfa2::Distance`

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
use lib_bpaf::{read_standard_tracepoints_at_offset,
               read_variable_tracepoints_at_offset,
               read_mixed_tracepoints_at_offset,
               CompressionStrategy, CompressionLayer};
use std::fs::File;

// Open file once (reuse for multiple seeks)
let mut file = File::open("alignments.bpaf")?;

// Pre-computed offsets, strategy, and layer from index/header
let offset = 123456;
let strategy = CompressionStrategy::ZigzagDelta(3);
let layer = CompressionLayer::Zstd; // or read from BinaryPafHeader

// Direct tracepoint decoding - no BpafReader overhead
let standard_tps = read_standard_tracepoints_at_offset(&mut file, offset, strategy, layer)?;
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
use lib_bpaf::{compress_paf_with_tracepoints, CompressionStrategy};

// Automatic (default): Analyzes data to choose delta vs raw + varint + zstd
// - Samples first 1000 records to determine optimal encoding
// - Handles all tracepoint types automatically
// - Enables O(1) random tracepoint access
compress_paf_with_tracepoints("alignments.paf", "alignments.bpaf", CompressionStrategy::Automatic(3))?;

// ZigzagDelta: Delta + zigzag transform (both values) + varint + zstd
// - Always uses delta encoding for tracepoints
// - Works well when values are monotonic or slowly changing
// - Enables O(1) random tracepoint access
compress_paf_with_tracepoints("alignments.paf", "alignments.bpaf", CompressionStrategy::ZigzagDelta(3))?;

// Raw: No delta, direct varints + zstd compression
// - Ideal when deltas are noisy or large
// - Also enables O(1) random tracepoint access
compress_paf_with_tracepoints("alignments.paf", "alignments.bpaf", CompressionStrategy::Raw(3))?;

// BlockwiseAdaptive: Adaptive selection per tracepoint record
// - Samples 1% of data and tests 4 sub-strategies
// - Selects optimal encoding (FOR, DeltaOfDelta, XORDelta, Dictionary)
// - Enables O(1) random tracepoint access
compress_paf_with_tracepoints("alignments.paf", "alignments.bpaf", CompressionStrategy::BlockwiseAdaptive(32))?;
```

**Strategy guide:**
- **Automatic (default)**: Best for most use cases, analyzes data to choose Raw or ZigzagDelta
- **ZigzagDelta**: Use when tracepoint values are mostly increasing
- **Raw**: Use when tracepoint values jump frequently
- **BlockwiseAdaptive**: Adaptive selection per record using advanced codecs

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
