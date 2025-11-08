# lib_bpaf

Binary PAF (BPAF) format library for efficient storage and random access of sequence alignments with tracepoints.

## Features

- **Compressed storage**: Delta encoding + zstd compression
- **O(1) random access**: External index for instant seek to any record
- **Tracepoint support**: Standard, Mixed, Variable, and FastGA representations
- **Seekable format**: Jump directly to any alignment without sequential scan
- **String deduplication**: Shared string table for sequence names

## Format

```
[Header] → [Records] → [StringTable]
```

- **Header**: Magic "BPAF" + version + metadata
- **Records**: Core PAF fields + compressed tracepoints
- **StringTable**: Deduplicated sequence names with lengths

## Compression Modes

### Default (recommended)
- Delta encoding + zstd compression
- Single-pass encoding

### Adaptive (maximum compression)
- Huffman coding + delta encoding + zstd
- Two-pass encoding (training + encoding)

## Usage

### Reading BPAF files

```rust
use lib_bpaf::BpafReader;

// Option 1: Open with index (for record ID access)
let mut reader = BpafReader::open("alignments.bpaf")?;
println!("Total records: {}", reader.len());

// O(1) random access by record ID
let record = reader.get_alignment_record(1000)?;
let (tracepoints, tp_type, complexity_metric, max_complexity) =
    reader.get_tracepoints(1000)?;

// Option 2: Open without index (for offset-based access only)
// Use this if you have your own offset storage (like impg)
// Skips index loading - much faster open time
let mut reader = BpafReader::open_without_index("alignments.bpaf")?;

// Access by file offset (no index needed)
let offset = 123456;
let record = reader.get_alignment_record_at_offset(offset)?;
let (tracepoints, tp_type, complexity_metric, max_complexity) =
    reader.get_tracepoints_at_offset(offset)?;

match &tracepoints {
    TracepointData::Standard(tps) => {
        println!("Standard tracepoints: {} items", tps.len());
    }
    TracepointData::Fastga(tps) => {
        println!("FastGA tracepoints: {} items", tps.len());
    }
    // ... handle other types
}

// Multiple accesses - just loop
for &record_id in &[0, 100, 500, 1000] {
    let (tps, _, _, _) = reader.get_tracepoints(record_id)?;
    // Process tracepoints...
}

// Sequential iteration
for record in reader.iter_records() {
    let record = record?;
    // Process record...
}
```

### Index Management

```rust
use lib_bpaf::{build_index, BpafIndex};

// Manual index creation
let index = build_index("alignments.bpaf")?;
index.save("alignments.bpaf.idx")?;

// Load existing index
let index = BpafIndex::load("alignments.bpaf.idx")?;
println!("Index has {} records", index.len());
```

## Examples

Run the seek demo to see O(1) random access in action:

```bash
# Build the example
cargo build --release --examples

# Show first 5 records
./target/release/examples/seek_demo alignments.bpaf

# Jump to specific records (O(1) access!)
./target/release/examples/seek_demo alignments.bpaf 0 100 500 1000
```

## Index Format

The `.bpaf.idx` index file contains:
- Magic: `BPAI` (4 bytes)
- Version: 1 (1 byte)
- Number of records: varint
- Record offsets: array of varints (byte positions)

Index enables O(1) random access without scanning the file.

## License

MIT
