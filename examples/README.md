# BPAF Visualizer

A minimal, efficient genome alignment dot plot viewer for BPAF files.

## Features

- **Native BPAF support** - Direct visualization of binary PAF files
- **Interactive navigation** - Pan and zoom through alignments
- **Color-coded orientations** - Green for forward, red for reverse alignments
- **Scaffold boundaries** - Visual markers for sequence boundaries
- **Cursor tracking** - Real-time position and sequence information

## Building

```bash
# bpaf-viz is now a binary (not example) with optional feature flag
cargo build --bin bpaf-viz --features viz --release
```

## Usage

```bash
# Run directly
cargo run --bin bpaf-viz --features viz --release -- <file.bpaf>

# Or use the compiled binary
./target/release/bpaf-viz <file.bpaf>
```

## Controls

- **Pan**: Click and drag
- **Zoom**: Mouse scroll wheel
- **Cursor info**: Bottom status bar shows current position

## Example

```bash
# First, create a BPAF file from PAF
cargo run --release -- compress -i alignments.paf -o alignments.bpaf

# Then visualize it
cargo run --example bpaf-viz --features viz --release -- alignments.bpaf
```

## Architecture

The visualizer uses a minimal design extracted from the alnviz project:

1. **Plot** - Stores alignment segments with genome-wide coordinates
2. **ViewState** - Manages pan/zoom transformations (x, y, scale)
3. **egui** - Immediate mode GUI for efficient rendering

### Coordinate Systems

- **Local**: Position within a single sequence (0 to sequence_length)
- **Genome-wide**: Cumulative position across all scaffolds
- **Screen**: Pixel positions on canvas

All segments are stored in genome-wide coordinates for efficient rendering.

## Performance

- **Load time**: O(n) where n is the number of alignment records
- **Render time**: O(v) where v is the number of visible segments
- **Memory**: All segments loaded in memory (no spatial indexing yet)

For very large datasets (>10M segments), consider:
- Filtering by region before visualization
- Adding spatial indexing (R*-tree) for viewport queries
- Progressive loading with view-dependent LOD

## License

MIT
