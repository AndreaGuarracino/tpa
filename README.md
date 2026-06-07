# tpa

TracePoint Alignment (TPA) binary format for compact storage and O(1) random access of sequence alignments encoded with [tracepoints](https://github.com/AndreaGuarracino/tracepoints).

## Overview

TPA stores tracepoint alignments as a binary file with an external index to allow O(1) random access. Use this library when you want to read or write `.tpa` files into your project.

## Installation

```toml
[dependencies]
tpa = { git = "https://github.com/AndreaGuarracino/tpa" }
```

## Usage

Read a `.tpa` file:

```rust
use tpa::TpaReader;

fn main() -> std::io::Result<()> {
    let mut reader = TpaReader::new("alignments.tpa")?; // opens via the .tpa.idx index
    println!("{} alignments", reader.len());

    let _record = reader.get_record(1000)?;             // O(1) lookup, no full scan
    for record in reader.iter_records()? {              // or iterate sequentially
        let _record = record?;
    }
    Ok(())
}
```

Write a `.tpa` from a PAF file with tracepoints:

```rust
use tpa::{paf_to_tpa, CompressionConfig};

paf_to_tpa("alignments.tp.paf", "alignments.tpa", CompressionConfig::new())?; // automatic strategy
```

See `examples/seek_demo.rs` and `examples/offset_demo.rs` for runnable demos.

## Format

```
[Header] → [StringTable] → [Records] → [Footer]
```

### Header
- Magic: `TPA\0` (4 bytes)
- Version: `1` (1 byte)
- Record count: varint
- String count: varint
- First strategy+layer: packed byte, bits 7-6 = layer (`0=Zstd, 1=Bgzip, 2=Nocomp`), bits 5-0 = strategy code
- Second strategy+layer: packed byte (same format)
- Tracepoint type: 1 byte
- Complexity metric: 1 byte
- Max complexity / spacing: varint
- Distance parameters: serialized to match `lib_wfa2::Distance`
- All-records mode flag: 1 byte (`1` if the records section is wrapped in whole-file BGZIP, else `0`)

### String table
Per string: name length (varint) + name bytes (UTF-8) + sequence length (varint).

### Records (per alignment)
- PAF fields: varint name ids, coordinates, matching bases and block length; 1-byte strand and mapping quality
- Tracepoints: a varint segment count, then two length-prefixed value streams, each encoded with the strategy and layer named in the header
- Tags: optional key/value pairs (2-byte key, 1-byte type, then the value)

### Footer
Magic `TPA\0`, version, record count, string count, then a `u32` little-endian footer-length trailer. Readers validate the footer against the header and fail fast if it is missing or truncated.

## Index format

`.tpa.idx`:

```
Magic:          TPAI (4 bytes)
Version:        2 (1 byte)
IndexType:      1 byte (0 = raw byte offsets, 1 = BGZF virtual positions)
BgzfSectStart:  u64 LE (0 for per-record mode)
Count:          varint (number of records)
Positions:      varint[] (per-record mode) or u64 LE[] (all-records mode)
```

## Related repositories

- **[tracepoints](https://github.com/AndreaGuarracino/tracepoints)**: the core library implementing tracepoint sampling and reconstruction.
- **[cigzip](https://github.com/AndreaGuarracino/cigzip)**: the command-line tool that writes and reads `.tpa` files.

## License

MIT
