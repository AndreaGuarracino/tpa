# lib_bpaf

Efficient binary storage and random access for genomic sequence alignments with tracepoints.

## Features

- **Fast compression**: Delta encoding + varint + zstd (6x compression ratio, ~1min for 4GB)
- **O(1) random access**: External index for instant record lookup
- **Tracepoint support**: Standard, Mixed, Variable, and FastGA representations
- **String deduplication**: Shared sequence name table
- **Backwards compatible**: Reads all format versions

## Format

```
[Header] → [Records] → [StringTable]
```

- **Header**: Magic "BPAF" + version + metadata
- **Records**: PAF fields + compressed tracepoints + tags
- **StringTable**: Deduplicated sequence names with lengths

## Installation

```toml
[dependencies]
lib_bpaf = { git = "https://github.com/AndreaGuarracino/lib_bpaf" }
```

## Quick Start

### Read with random access

```rust
use lib_bpaf::BpafReader;

// Open with index for O(1) record access
let mut reader = BpafReader::open("alignments.bpaf")?;
println!("Total records: {}", reader.len());

// Jump to any record instantly
let record = reader.get_alignment_record(1000)?;
let (tracepoints, tp_type, _, _) = reader.get_tracepoints(1000)?;

match &tracepoints {
    TracepointData::Standard(tps) => println!("{} tracepoints", tps.len()),
    TracepointData::Fastga(tps) => println!("{} FastGA traces", tps.len()),
    _ => {}
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
use lib_bpaf::compress_paf;

// Compress PAF with tp:Z: tags to binary
compress_paf("alignments.paf", "alignments.bpaf")?;
```

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
