# lib_bpaf

Binary format for genomic sequence alignments with tracepoints.

## Features

- **O(1) random access**: External index for instant record lookup
- **Fast varint compression**:
  - **Automatic-Fast (default)**: Samples the first 1,000 records and tests every concrete strategy × compression layer per stream, then locks in the best first/second combination
  - **Automatic-Slow**: Runs the same exhaustive search but evaluates all records (highest compression fidelity)
  - **ZigzagDelta**: Delta + zigzag transform + varint + zstd
  - **Raw**: Plain varints + zstd
  - **Rice / Huffman**: Block-local entropy coding over zigzag deltas, still byte-aligned for random seeks
- **Tracepoint support**: Standard, Mixed, Variable, and FastGA representations
- **String deduplication**: Shared sequence name table
- **Byte-aligned encoding**: Enables extremely fast tracepoint extraction

## Format

```
[Header] → [StringTable] → [Records]
```

### Header (metadata + strategy)
- Magic: `BPAF` (4 bytes)
- Version: `2` (1 byte)
- First compression layer: `0=Zstd, 1=Bgzip, 2=Nocomp` (1 byte)
- Second compression layer: `0=Zstd, 1=Bgzip, 2=Nocomp` (1 byte)
- First strategy code: `0-20` (1 byte)
- Second strategy code: `0-20` (1 byte)
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
let first_layer = CompressionLayer::Zstd; // or read from BinaryPafHeader
let second_layer = CompressionLayer::Zstd;

// Direct tracepoint decoding - no BpafReader overhead
let standard_tps = read_standard_tracepoints_at_offset(
    &mut file,
    offset,
    strategy,
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
use lib_bpaf::{compress_paf_with_tracepoints, CompressionStrategy};

// Automatic-Fast (default): Analyzes data and tries every strategy/layer per stream
// - Samples first 1000 records to determine optimal encoding
// - Handles all tracepoint types automatically
// - Enables O(1) random tracepoint access
compress_paf_with_tracepoints("alignments.paf", "alignments.bpaf", CompressionStrategy::AutomaticFast(3))?;

// Automatic-Slow: same search but uses ALL records for scoring
compress_paf_with_tracepoints("alignments.paf", "alignments.bpaf", CompressionStrategy::AutomaticSlow(3))?;

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

// Rice entropy coding: Zigzag + delta + Rice (Golomb) bitstream per block
compress_paf_with_tracepoints("alignments.paf", "alignments.bpaf", CompressionStrategy::Rice(3))?;

// Huffman entropy coding: Zigzag + delta + canonical Huffman per block
compress_paf_with_tracepoints("alignments.paf", "alignments.bpaf", CompressionStrategy::Huffman(3))?;
```

**Strategy guide:**
- **Automatic-Fast (default)**: Best for most use cases, analyzes the first/second streams independently (all strategies × layers) on a 1,000-record sample and stores the winning combination
- **Automatic-Slow**: Use when you want the absolute best compression and can afford analyzing the entire file
- **ZigzagDelta**: Use when tracepoint values are mostly increasing
- **Raw**: Use when tracepoint values jump frequently
- **BlockwiseAdaptive**: Adaptive selection per record using advanced codecs
- **Rice**: Skewed, small zigzag deltas where Golomb coding beats varints (still byte-aligned)
- **Huffman**: Highly skewed distributions with a few hot values/zeros

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
