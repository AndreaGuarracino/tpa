//! Data structures for TracePoint Alignment (TPA) format

use crate::{utils::*, Distance};
use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::str::FromStr;
use tracepoints::{ComplexityMetric, TracepointData, TracepointType};

pub const TPA_VERSION: u8 = 1;
pub const TPA_MAGIC: &[u8; 4] = b"TPA\0";

/// Write the common prefix shared by header and footer: magic, version, record_count, string_count
fn write_common_prefix<W: Write>(
    writer: &mut W,
    num_records: u64,
    num_strings: u64,
) -> io::Result<()> {
    writer.write_all(TPA_MAGIC)?;
    writer.write_all(&[TPA_VERSION])?;
    write_varint(writer, num_records)?;
    write_varint(writer, num_strings)?;
    Ok(())
}

/// Read and validate the common prefix: magic, version, record_count, string_count
fn read_common_prefix<R: Read>(reader: &mut R) -> io::Result<(u8, u64, u64)> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != TPA_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid TPA magic",
        ));
    }

    let mut version_buf = [0u8; 1];
    reader.read_exact(&mut version_buf)?;
    let version = version_buf[0];
    if version != TPA_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported TPA version: {}", version),
        ));
    }

    let num_records = read_varint(reader)?;
    let num_strings = read_varint(reader)?;

    Ok((version, num_records, num_strings))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionLayer {
    /// Zstd compression (default)
    Zstd,
    /// BGZF (blocked gzip) compression
    Bgzip,
    /// No compression (store raw encoded data)
    Nocomp,
}

impl CompressionLayer {
    pub fn to_u8(self) -> u8 {
        match self {
            CompressionLayer::Zstd => 0,
            CompressionLayer::Bgzip => 1,
            CompressionLayer::Nocomp => 2,
        }
    }

    pub fn from_u8(value: u8) -> Result<Self, String> {
        match value {
            0 => Ok(CompressionLayer::Zstd),
            1 => Ok(CompressionLayer::Bgzip),
            2 => Ok(CompressionLayer::Nocomp),
            _ => Err(format!("Invalid compression layer code: {}", value)),
        }
    }

    /// Return a human-readable label for this layer
    pub fn as_str(self) -> &'static str {
        match self {
            CompressionLayer::Zstd => "zstd",
            CompressionLayer::Bgzip => "bgzip",
            CompressionLayer::Nocomp => "nocomp",
        }
    }

    /// Enumerate all supported compression layers
    pub fn all() -> [CompressionLayer; 3] {
        [
            CompressionLayer::Zstd,
            CompressionLayer::Bgzip,
            CompressionLayer::Nocomp,
        ]
    }
}

/// Configuration for PAF compression
#[derive(Clone, Debug)]
pub struct CompressionConfig {
    /// Strategy for first values in tracepoint pairs
    pub first_strategy: CompressionStrategy,
    /// Strategy for second values in tracepoint pairs
    pub second_strategy: CompressionStrategy,
    /// Compression layer for first stream
    pub first_layer: CompressionLayer,
    /// Compression layer for second stream
    pub second_layer: CompressionLayer,
    /// Tracepoint representation type
    pub tp_type: TracepointType,
    /// Maximum complexity/spacing parameter
    pub max_complexity: u32,
    /// Metric used for complexity calculation
    pub complexity_metric: ComplexityMetric,
    /// Distance parameters for alignment
    pub distance: Distance,
    /// When true, compress all records together with BGZIP
    pub all_records: bool,
    /// BGZIP compression level for all-records mode (0-9)
    pub all_records_level: u32,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            first_strategy: CompressionStrategy::Automatic(3, 10000),
            second_strategy: CompressionStrategy::Automatic(3, 10000),
            first_layer: CompressionLayer::Zstd,
            second_layer: CompressionLayer::Zstd,
            tp_type: TracepointType::Standard,
            max_complexity: 32,
            complexity_metric: ComplexityMetric::EditDistance,
            distance: Distance::Edit,
            all_records: false,
            all_records_level: 6,
        }
    }
}

impl CompressionConfig {
    /// Create a new config with sensible defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the compression strategy (used for both first and second values)
    pub fn strategy(mut self, s: CompressionStrategy) -> Self {
        self.first_strategy = s;
        self.second_strategy = s;
        self
    }

    /// Set separate strategies for first and second values
    pub fn dual_strategy(
        mut self,
        first: CompressionStrategy,
        second: CompressionStrategy,
    ) -> Self {
        self.first_strategy = first;
        self.second_strategy = second;
        self
    }

    /// Set the compression layer (used for both first and second streams)
    pub fn layer(mut self, l: CompressionLayer) -> Self {
        self.first_layer = l;
        self.second_layer = l;
        self
    }

    /// Set separate compression layers for first and second streams
    pub fn dual_layer(mut self, first: CompressionLayer, second: CompressionLayer) -> Self {
        self.first_layer = first;
        self.second_layer = second;
        self
    }

    /// Set the tracepoint type
    pub fn tp_type(mut self, t: TracepointType) -> Self {
        self.tp_type = t;
        self
    }

    /// Set maximum complexity/spacing
    pub fn max_complexity(mut self, c: u32) -> Self {
        self.max_complexity = c;
        self
    }

    /// Set complexity metric
    pub fn complexity_metric(mut self, m: ComplexityMetric) -> Self {
        self.complexity_metric = m;
        self
    }

    /// Set distance parameters
    pub fn distance(mut self, d: Distance) -> Self {
        self.distance = d;
        self
    }

    /// Enable all-records compression mode.
    /// Header and string table are plain bytes, all records are BGZIP-compressed together.
    /// Enables cross-record compression context with fast file opening.
    /// Random access uses BGZF virtual positions (block-level, ~64KB blocks).
    pub fn all_records(mut self) -> Self {
        self.all_records = true;
        self
    }

    /// Set compression level for all-records mode (0-9, default 6)
    pub fn all_records_level(mut self, level: u32) -> Self {
        self.all_records_level = level.min(9);
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CompressionStrategy {
    /// Automatic strategy selection
    /// - Tests every concrete strategy × layer on sampled records
    /// - Parameters: (zstd_level, sample_size) where 0 = entire file
    /// - Default: level=3, sample_size=10000
    Automatic(i32, usize),
    /// Raw encoding (no preprocessing) + Zstd
    /// - Optimal for low complexity data
    /// - Configurable compression level (max 22, default: 3)
    Raw(i32),
    /// Zigzag + delta encoding + Zstd
    /// - Optimal for high complexity data
    /// - Configurable compression level (max 22, default: 3)
    ZigzagDelta(i32),
    /// 2D-Delta encoding: query raw, target as delta from query + Zstd
    /// - Exploits correlation between query and target advances
    /// - Optimal for CIGAR-derived alignments (6-20% better)
    /// - Configurable compression level (max 22, default: 3)
    TwoDimDelta(i32),
    /// Run-Length Encoding + optional delta + Zstd
    /// - Encodes (value, run_length) pairs
    /// - Optimal for high-repetition data (30-40% better)
    /// - Configurable compression level (max 22, default: 3)
    RunLength(i32),
    /// Bit packing for small value ranges
    /// - Packs values into minimal bits
    /// - Optimal when values fit in <8 bits (20-30% better)
    /// - Configurable compression level (max 22, default: 3)
    BitPacked(i32),
    /// Delta-of-Delta encoding (Gorilla-style)
    /// - Applies delta encoding twice for regularly spaced coordinates
    /// - Optimal for evenly-spaced tracepoints (10-12x compression)
    /// - Configurable compression level (max 22, default: 3)
    DeltaOfDelta(i32),
    /// Frame-of-Reference encoding
    /// - Stores min value + bit-packed offsets
    /// - True O(1) random access without prefix-sum dependencies
    /// - Optimal for fast random access (7+ billion ints/sec decompression)
    /// - Configurable compression level (max 22, default: 3)
    FrameOfReference(i32),
    /// Hybrid RLE-Varint
    /// - RLE for target stream (high zero%), varint for query
    /// - Optimal for direct tracepoints with 30%+ zero deltas
    /// - Configurable compression level (max 22, default: 3)
    HybridRLE(i32),
    /// XOR-based differential encoding (Gorilla)
    /// - XOR each value with predecessor, encode meaningful bits
    /// - Optimal for highly correlated coordinates (12x compression)
    /// - Configurable compression level (max 22, default: 3)
    XORDelta(i32),
    /// Dictionary coding for low cardinality
    /// - Build dictionary of common deltas, encode positions
    /// - Optimal for repeated delta patterns (5.4x average compression)
    /// - Configurable compression level (max 22, default: 3)
    Dictionary(i32),
    /// Stream VByte byte-aligned encoding
    /// - Separated control and data bytes for SIMD acceleration
    /// - Optimal for byte-range data (1.1-4.0 billion ints/sec)
    /// - Configurable compression level (max 22, default: 3)
    StreamVByte(i32),
    /// FastPFOR - Patched Frame-of-Reference with exceptions
    /// - Groups integers into 128-value blocks with optimal bit-width
    /// - Stores exceptions separately for outliers
    /// - Achieves 3.8-6.2 bits/int at 1.1-3.1 billion ints/sec
    /// - Optimal for data with 82-100% values under 256
    /// - Configurable compression level (max 22, default: 3)
    FastPFOR(i32),
    /// Cascaded compression - Multi-level encoding
    /// - Dictionary → RLE → FastBP128 cascade for low-cardinality runs
    /// - Dictionary → FastPFOR for medium cardinality
    /// - RLE → FastPFOR for high-run varied-magnitude data
    /// - Achieves up to 13,000x compression on low-cardinality data
    /// - Configurable compression level (max 22, default: 3)
    Cascaded(i32),
    /// Full Simple8b-RLE with all 16 packing modes
    /// - Complete implementation with 16 selector modes
    /// - Dedicated RLE mode for runs up to 2^28 repetitions
    /// - 35% zero deltas compress to 8 bytes per run
    /// - Achieves 4+ billion ints/sec for runs, 1.5-2.0 billion for varied
    /// - Configurable compression level (max 22, default: 3)
    Simple8bFull(i32),
    /// Selective RLE preprocessing with bitmap positions
    /// - Detects runs of 8+ identical values before main compression
    /// - Stores run positions in bitmap for skip-ahead decompression
    /// - Achieves 3-8x improvement on high-zero data (35%+ zeros)
    /// - Negligible overhead when runs are absent
    /// - Configurable compression level (max 22, default: 3)
    SelectiveRLE(i32),
    /// Rice / Golomb coding with block-local parameter
    /// - Zigzag deltas encoded as unary quotient + fixed remainder bits
    /// - Chooses k per block to minimize total bits
    /// - Byte-aligned output keeps O(1) random access
    /// - Configurable compression level (max 22, default: 3)
    Rice(i32),
    /// Canonical Huffman entropy coder
    /// - Builds per-block codes over zigzag deltas
    /// - Stores compact code lengths; canonical decoding avoids storing tree
    /// - Byte-aligned output keeps O(1) random access
    /// - Configurable compression level (max 22, default: 3)
    Huffman(i32),
    /// LZ77-style sequence matching
    /// - Finds repeated sequences in sliding window
    /// - Encodes as literals + (offset, length) matches
    /// - Captures multi-value patterns that Dictionary misses
    /// - Optimal for data with repeated subsequences
    /// - Configurable compression level (max 22, default: 3)
    LZ77(i32),
}

impl CompressionStrategy {
    /// Return all encoding strategies (excludes Automatic meta-strategy)
    pub fn all(level: i32) -> Vec<Self> {
        vec![
            CompressionStrategy::Raw(level),
            CompressionStrategy::ZigzagDelta(level),
            CompressionStrategy::TwoDimDelta(level),
            CompressionStrategy::RunLength(level),
            CompressionStrategy::BitPacked(level),
            CompressionStrategy::DeltaOfDelta(level),
            CompressionStrategy::FrameOfReference(level),
            CompressionStrategy::HybridRLE(level),
            CompressionStrategy::XORDelta(level),
            CompressionStrategy::Dictionary(level),
            CompressionStrategy::StreamVByte(level),
            CompressionStrategy::FastPFOR(level),
            CompressionStrategy::Cascaded(level),
            CompressionStrategy::Simple8bFull(level),
            CompressionStrategy::SelectiveRLE(level),
            CompressionStrategy::Rice(level),
            CompressionStrategy::Huffman(level),
            CompressionStrategy::LZ77(level),
        ]
    }

    /// Parse strategy from string with compression layer
    /// Formats:
    ///   - Single: "strategy", "strategy,level", "strategy-bgzip", "strategy-nocomp"
    ///   - Automatic: "automatic", "automatic,level", "automatic,level,sample_size"
    pub fn from_str_with_layer(s: &str) -> Result<(Self, CompressionLayer), String> {
        let parts: Vec<&str> = s.split(',').collect();
        let mut strategy_name = parts[0].to_lowercase();

        // Check for suffix: -bgzip or -nocomp
        let layer = if strategy_name.ends_with("-bgzip") {
            strategy_name = strategy_name.trim_end_matches("-bgzip").to_string();
            CompressionLayer::Bgzip
        } else if strategy_name.ends_with("-nocomp") {
            strategy_name = strategy_name.trim_end_matches("-nocomp").to_string();
            CompressionLayer::Nocomp // No compression - store raw encoded data
        } else {
            CompressionLayer::Zstd
        };

        let default_level = if layer == CompressionLayer::Nocomp {
            0
        } else {
            3
        };
        let compression_level = Self::parse_level(&parts, default_level)?;

        // Special handling for automatic strategy with optional sample_size
        let strategy = if strategy_name == "automatic" {
            let sample_size = match parts.get(2) {
                Some(raw) => raw.trim().parse::<usize>().map_err(|_| {
                    format!(
                        "Invalid sample size '{}'. Must be a non-negative integer (0 = entire file).",
                        raw
                    )
                })?,
                None => 10000, // Default sample size
            };
            CompressionStrategy::Automatic(compression_level, sample_size)
        } else {
            Self::parse_single_strategy(&strategy_name, compression_level)?
        };

        Ok((strategy, layer))
    }

    /// Parse compression level from split input and validate range
    fn parse_level(parts: &[&str], default: i32) -> Result<i32, String> {
        let level = match parts.get(1) {
            Some(raw) => raw.trim().parse::<i32>().map_err(|_| {
                format!(
                    "Invalid compression level '{}'. Must be a number between 0 and 22.",
                    raw
                )
            })?,
            None => default,
        };

        if (0..=22).contains(&level) {
            Ok(level)
        } else {
            Err(format!(
                "Compression level {} is out of range. Must be between 0 and 22.",
                level
            ))
        }
    }

    /// Parse a single strategy name into a CompressionStrategy enum
    fn parse_single_strategy(name: &str, level: i32) -> Result<CompressionStrategy, String> {
        match name {
            "automatic" => Ok(CompressionStrategy::Automatic(level, 10000)),
            "raw" => Ok(CompressionStrategy::Raw(level)),
            "zigzag-delta" => Ok(CompressionStrategy::ZigzagDelta(level)),
            "2d-delta" => Ok(CompressionStrategy::TwoDimDelta(level)),
            "rle" => Ok(CompressionStrategy::RunLength(level)),
            "bit-packed" => Ok(CompressionStrategy::BitPacked(level)),
            "delta-of-delta" => Ok(CompressionStrategy::DeltaOfDelta(level)),
            "frame-of-reference" | "for" => Ok(CompressionStrategy::FrameOfReference(level)),
            "hybrid-rle" => Ok(CompressionStrategy::HybridRLE(level)),
            "xor-delta" => Ok(CompressionStrategy::XORDelta(level)),
            "dictionary" | "dict" => Ok(CompressionStrategy::Dictionary(level)),
            "stream-vbyte" | "streamvbyte" => Ok(CompressionStrategy::StreamVByte(level)),
            "fastpfor" | "fast-pfor" => Ok(CompressionStrategy::FastPFOR(level)),
            "cascaded" => Ok(CompressionStrategy::Cascaded(level)),
            "simple8b-full" | "simple8bfull" => Ok(CompressionStrategy::Simple8bFull(level)),
            "selective-rle" | "selectiverle" => Ok(CompressionStrategy::SelectiveRLE(level)),
            "rice" => Ok(CompressionStrategy::Rice(level)),
            "huffman" => Ok(CompressionStrategy::Huffman(level)),
            "lz77" => Ok(CompressionStrategy::LZ77(level)),
            _ => Err(format!(
                "Unsupported compression strategy '{}'. Use --help to see all available strategies.",
                name
            )),
        }
    }

    /// Get all available strategies
    pub fn variants() -> &'static [&'static str] {
        &[
            "automatic",
            "raw",
            "zigzag-delta",
            "2d-delta",
            "rle",
            "bit-packed",
            "delta-of-delta",
            "frame-of-reference",
            "hybrid-rle",
            "xor-delta",
            "dictionary",
            "stream-vbyte",
            "fastpfor",
            "cascaded",
            "simple8b-full",
            "selective-rle",
            "rice",
            "huffman",
            "lz77",
        ]
    }

    /// Convert to strategy code for file header
    /// Strategy codes: 0-17 (18 strategies)
    fn to_code(self) -> io::Result<u8> {
        match self {
            CompressionStrategy::Automatic(_, _) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Automatic strategy must be resolved before writing",
            )),
            CompressionStrategy::Raw(_) => Ok(0),
            CompressionStrategy::ZigzagDelta(_) => Ok(1),
            CompressionStrategy::TwoDimDelta(_) => Ok(2),
            CompressionStrategy::RunLength(_) => Ok(3),
            CompressionStrategy::BitPacked(_) => Ok(4),
            CompressionStrategy::DeltaOfDelta(_) => Ok(5),
            CompressionStrategy::FrameOfReference(_) => Ok(6),
            CompressionStrategy::HybridRLE(_) => Ok(7),
            CompressionStrategy::XORDelta(_) => Ok(8),
            CompressionStrategy::Dictionary(_) => Ok(9),
            CompressionStrategy::StreamVByte(_) => Ok(10),
            CompressionStrategy::FastPFOR(_) => Ok(11),
            CompressionStrategy::Cascaded(_) => Ok(12),
            CompressionStrategy::Simple8bFull(_) => Ok(13),
            CompressionStrategy::SelectiveRLE(_) => Ok(14),
            CompressionStrategy::Rice(_) => Ok(15),
            CompressionStrategy::Huffman(_) => Ok(16),
            CompressionStrategy::LZ77(_) => Ok(17),
        }
    }

    /// Parse from strategy code
    /// Strategy codes: 0-17 (18 strategies)
    fn from_code(code: u8) -> io::Result<Self> {
        match code {
            0 => Ok(CompressionStrategy::Raw(3)),
            1 => Ok(CompressionStrategy::ZigzagDelta(3)),
            2 => Ok(CompressionStrategy::TwoDimDelta(3)),
            3 => Ok(CompressionStrategy::RunLength(3)),
            4 => Ok(CompressionStrategy::BitPacked(3)),
            5 => Ok(CompressionStrategy::DeltaOfDelta(3)),
            6 => Ok(CompressionStrategy::FrameOfReference(3)),
            7 => Ok(CompressionStrategy::HybridRLE(3)),
            8 => Ok(CompressionStrategy::XORDelta(3)),
            9 => Ok(CompressionStrategy::Dictionary(3)),
            10 => Ok(CompressionStrategy::StreamVByte(3)),
            11 => Ok(CompressionStrategy::FastPFOR(3)),
            12 => Ok(CompressionStrategy::Cascaded(3)),
            13 => Ok(CompressionStrategy::Simple8bFull(3)),
            14 => Ok(CompressionStrategy::SelectiveRLE(3)),
            15 => Ok(CompressionStrategy::Rice(3)),
            16 => Ok(CompressionStrategy::Huffman(3)),
            17 => Ok(CompressionStrategy::LZ77(3)),
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Unsupported compression strategy code: {}", code),
            )),
        }
    }

    /// Get zstd compression level for this strategy.
    /// Note: This level is used for the final Zstd/Bgzip compression layer,
    /// not for the pre-encoding step (which is strategy-specific).
    pub fn zstd_level(&self) -> i32 {
        match self {
            CompressionStrategy::Automatic(level, _)
            | CompressionStrategy::Raw(level)
            | CompressionStrategy::ZigzagDelta(level)
            | CompressionStrategy::TwoDimDelta(level)
            | CompressionStrategy::RunLength(level)
            | CompressionStrategy::BitPacked(level)
            | CompressionStrategy::DeltaOfDelta(level)
            | CompressionStrategy::FrameOfReference(level)
            | CompressionStrategy::HybridRLE(level)
            | CompressionStrategy::XORDelta(level)
            | CompressionStrategy::Dictionary(level)
            | CompressionStrategy::StreamVByte(level)
            | CompressionStrategy::FastPFOR(level)
            | CompressionStrategy::Cascaded(level)
            | CompressionStrategy::Simple8bFull(level)
            | CompressionStrategy::SelectiveRLE(level)
            | CompressionStrategy::Rice(level)
            | CompressionStrategy::Huffman(level)
            | CompressionStrategy::LZ77(level) => *level,
        }
    }

    /// Get the display name of this strategy
    pub fn display_name(&self) -> &'static str {
        match self {
            CompressionStrategy::Automatic(_, _) => "Automatic",
            CompressionStrategy::Raw(_) => "Raw",
            CompressionStrategy::ZigzagDelta(_) => "ZigzagDelta",
            CompressionStrategy::TwoDimDelta(_) => "TwoDimDelta",
            CompressionStrategy::RunLength(_) => "RunLength",
            CompressionStrategy::BitPacked(_) => "BitPacked",
            CompressionStrategy::DeltaOfDelta(_) => "DeltaOfDelta",
            CompressionStrategy::FrameOfReference(_) => "FrameOfReference",
            CompressionStrategy::HybridRLE(_) => "HybridRLE",
            CompressionStrategy::XORDelta(_) => "XORDelta",
            CompressionStrategy::Dictionary(_) => "Dictionary",
            CompressionStrategy::StreamVByte(_) => "StreamVByte",
            CompressionStrategy::FastPFOR(_) => "FastPFOR",
            CompressionStrategy::Cascaded(_) => "Cascaded",
            CompressionStrategy::Simple8bFull(_) => "Simple8bFull",
            CompressionStrategy::SelectiveRLE(_) => "SelectiveRLE",
            CompressionStrategy::Rice(_) => "Rice",
            CompressionStrategy::Huffman(_) => "Huffman",
            CompressionStrategy::LZ77(_) => "LZ77",
        }
    }
}

impl FromStr for CompressionStrategy {
    type Err = String;

    /// Parse strategy from string (format: "strategy" or "strategy,level")
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (strategy, _layer) = Self::from_str_with_layer(s)?;
        Ok(strategy)
    }
}

impl std::fmt::Display for CompressionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Special case for Automatic which has additional sample_size info
        if let CompressionStrategy::Automatic(level, sample_size) = self {
            return if *sample_size == 0 {
                write!(f, "Automatic (level {}, all records)", level)
            } else {
                write!(f, "Automatic (level {}, {} records)", level, sample_size)
            };
        }
        // All other strategies: "Name (level N)"
        write!(f, "{} (level {})", self.display_name(), self.zstd_level())
    }
}

const STRATEGY_MASK: u8 = 0b0011_1111;
const LAYER_SHIFT: u8 = 6;

fn encode_strategy_with_layer(code: u8, layer: CompressionLayer) -> u8 {
    (layer.to_u8() << LAYER_SHIFT) | (code & STRATEGY_MASK)
}

fn decode_strategy_with_layer(value: u8) -> io::Result<(CompressionLayer, u8)> {
    let layer_bits = value >> LAYER_SHIFT;
    let layer = CompressionLayer::from_u8(layer_bits)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok((layer, value & STRATEGY_MASK))
}

pub struct TpaHeader {
    pub(crate) version: u8,
    pub(crate) first_strategy_code: u8,
    pub(crate) second_strategy_code: u8,
    pub(crate) first_layer: CompressionLayer,
    pub(crate) second_layer: CompressionLayer,
    pub(crate) num_records: u64,
    pub(crate) num_strings: u64,
    pub(crate) tracepoint_type: TracepointType,
    pub(crate) complexity_metric: ComplexityMetric,
    pub(crate) max_complexity: u32, // For Standard/Mixed/Variable: max_value; For FASTGA: trace_spacing
    pub(crate) distance: Distance,
    pub(crate) all_records: bool, // When true, all records compressed together with BGZIP
}

impl TpaHeader {
    /// Create header with strategy
    pub(crate) fn new(
        num_records: u64,
        num_strings: u64,
        first_strategy: CompressionStrategy,
        second_strategy: CompressionStrategy,
        first_layer: CompressionLayer,
        second_layer: CompressionLayer,
        tracepoint_type: TracepointType,
        complexity_metric: ComplexityMetric,
        max_complexity: u32,
        distance: Distance,
        all_records: bool,
    ) -> io::Result<Self> {
        let first_strategy_code = first_strategy.to_code()?;
        let second_strategy_code = second_strategy.to_code()?;

        Ok(Self {
            version: TPA_VERSION,
            first_strategy_code,
            second_strategy_code,
            first_layer,
            second_layer,
            num_records,
            num_strings,
            tracepoint_type,
            complexity_metric,
            max_complexity,
            distance,
            all_records,
        })
    }

    /// Get format version
    pub fn version(&self) -> u8 {
        self.version
    }

    /// Get number of records
    pub fn num_records(&self) -> u64 {
        self.num_records
    }

    /// Get number of unique strings
    pub fn num_strings(&self) -> u64 {
        self.num_strings
    }

    /// Get compression strategies (first, second)
    pub fn strategies(&self) -> io::Result<(CompressionStrategy, CompressionStrategy)> {
        Ok((
            CompressionStrategy::from_code(self.first_strategy_code)?,
            CompressionStrategy::from_code(self.second_strategy_code)?,
        ))
    }

    /// Get strategy for first values
    pub fn first_strategy(&self) -> io::Result<CompressionStrategy> {
        CompressionStrategy::from_code(self.first_strategy_code)
    }

    /// Get strategy for second values
    pub fn second_strategy(&self) -> io::Result<CompressionStrategy> {
        CompressionStrategy::from_code(self.second_strategy_code)
    }

    /// Get tracepoint type
    pub fn tp_type(&self) -> TracepointType {
        self.tracepoint_type
    }

    /// Get distance mode
    pub fn distance(&self) -> Distance {
        self.distance
    }

    /// Get complexity metric
    pub fn complexity_metric(&self) -> ComplexityMetric {
        self.complexity_metric
    }

    /// Get max complexity
    pub fn max_complexity(&self) -> u32 {
        self.max_complexity
    }

    /// Compression layer for first values
    pub fn first_layer(&self) -> CompressionLayer {
        self.first_layer
    }

    /// Compression layer for second values
    pub fn second_layer(&self) -> CompressionLayer {
        self.second_layer
    }

    /// Whether all records are compressed together with BGZIP (header/string table stay plain)
    pub fn all_records(&self) -> bool {
        self.all_records
    }

    pub(crate) fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        // Common prefix (same as footer)
        write_common_prefix(writer, self.num_records, self.num_strings)?;

        // Header-specific fields
        writer.write_all(&[encode_strategy_with_layer(
            self.first_strategy_code,
            self.first_layer,
        )])?;
        writer.write_all(&[encode_strategy_with_layer(
            self.second_strategy_code,
            self.second_layer,
        )])?;
        writer.write_all(&[self.tracepoint_type.to_u8()])?;
        writer.write_all(&[self.complexity_metric.to_u8()])?;
        write_varint(writer, self.max_complexity as u64)?;
        write_distance(writer, &self.distance)?;
        // All-records mode flag
        writer.write_all(&[if self.all_records { 1 } else { 0 }])?;
        Ok(())
    }

    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        // Common prefix (same as footer)
        let (version, num_records, num_strings) = read_common_prefix(reader)?;

        // Header-specific fields
        let mut strategy_bytes = [0u8; 2];
        reader.read_exact(&mut strategy_bytes)?;
        let (first_layer, first_strategy_code) = decode_strategy_with_layer(strategy_bytes[0])?;
        let (second_layer, second_strategy_code) = decode_strategy_with_layer(strategy_bytes[1])?;

        let mut tp_type_buf = [0u8; 1];
        reader.read_exact(&mut tp_type_buf)?;
        let tracepoint_type = TracepointType::from_u8(tp_type_buf[0])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut metric_buf = [0u8; 1];
        reader.read_exact(&mut metric_buf)?;
        let complexity_metric = ComplexityMetric::from_u8(metric_buf[0])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let max_complexity = read_varint(reader)? as u32;

        let distance = read_distance(reader)?;

        // All-records mode flag
        let mut all_records_flag = [0u8; 1];
        reader.read_exact(&mut all_records_flag)?;
        let all_records = all_records_flag[0] != 0;

        Ok(Self {
            version,
            first_strategy_code,
            second_strategy_code,
            first_layer,
            second_layer,
            num_records,
            num_strings,
            tracepoint_type,
            complexity_metric,
            max_complexity,
            distance,
            all_records,
        })
    }
}

pub struct TpaFooter {
    pub(crate) num_records: u64,
    pub(crate) num_strings: u64,
}

impl TpaFooter {
    pub fn new(num_records: u64, num_strings: u64) -> Self {
        Self {
            num_records,
            num_strings,
        }
    }

    /// Write footer as: [common_prefix][footer_len_le]
    pub fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let mut buf = Vec::new();
        write_common_prefix(&mut buf, self.num_records, self.num_strings)?;

        let footer_len: u32 = buf
            .len()
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Footer too large"))?;

        writer.write_all(&buf)?;
        writer.write_all(&footer_len.to_le_bytes())?;
        Ok(())
    }

    /// Read footer from the end of the file, returning an error if the footer
    /// is missing or malformed.
    pub fn read_from_end<R: Read + Seek>(reader: &mut R) -> io::Result<Self> {
        let file_len = reader.seek(SeekFrom::End(0))?;
        if file_len < 4 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "File too small to contain TPA footer length",
            ));
        }

        reader.seek(SeekFrom::End(-4))?;
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let footer_len = u32::from_le_bytes(len_buf) as u64;

        if footer_len + 4 > file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid TPA footer length",
            ));
        }

        let back = i64::try_from(footer_len + 4).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "TPA footer length does not fit in i64",
            )
        })?;
        reader.seek(SeekFrom::End(-back))?;

        let mut buf = vec![0u8; footer_len as usize];
        reader.read_exact(&mut buf)?;

        let mut cursor = std::io::Cursor::new(buf);
        let (_version, num_records, num_strings) = read_common_prefix(&mut cursor)?;

        Ok(Self {
            num_records,
            num_strings,
        })
    }

    pub fn validate_against(&self, header: &TpaHeader) -> io::Result<()> {
        if self.num_records != header.num_records {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Footer/header record mismatch: header={}, footer={}",
                    header.num_records, self.num_records
                ),
            ));
        }
        if self.num_strings != header.num_strings {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Footer/header string mismatch: header={}, footer={}",
                    header.num_strings, self.num_strings
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod footer_tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn footer_roundtrip() {
        let footer = TpaFooter::new(10, 5);
        let mut data = Vec::new();
        footer.write(&mut data).unwrap();
        let mut cursor = Cursor::new(data);
        let read_back = TpaFooter::read_from_end(&mut cursor).unwrap();
        assert_eq!(read_back.num_records, 10);
        assert_eq!(read_back.num_strings, 5);
    }

    #[test]
    fn open_validates() {
        // Build a minimal fake file: header (magic + fixed bytes), string table (empty), footer.
        // Header layout: magic(4) + version(1) + num_records(varint) + num_strings(varint)
        // + first_strategy_with_layer(1) + second_strategy_with_layer(1)
        // + tp_type(1) + complexity_metric(1) + max_complexity(varint) + distance + bgzip_all_records(1)
        let mut file_bytes = Vec::new();

        // Common prefix (same order as footer)
        file_bytes.extend_from_slice(TPA_MAGIC);
        file_bytes.push(TPA_VERSION);
        write_varint(&mut file_bytes, 1).unwrap(); // num_records
        write_varint(&mut file_bytes, 0).unwrap(); // num_strings

        // Header-specific fields
        file_bytes.push(encode_strategy_with_layer(
            CompressionStrategy::Raw(3).to_code().unwrap(),
            CompressionLayer::Zstd,
        ));
        file_bytes.push(encode_strategy_with_layer(
            CompressionStrategy::Raw(3).to_code().unwrap(),
            CompressionLayer::Zstd,
        ));
        file_bytes.push(TracepointType::Standard.to_u8());
        file_bytes.push(ComplexityMetric::EditDistance.to_u8());
        write_varint(&mut file_bytes, 0).unwrap(); // max_complexity
        write_distance(&mut file_bytes, &Distance::Edit).unwrap();
        file_bytes.push(0); // all_records = false

        // string table is empty

        // footer
        let footer = TpaFooter::new(1, 0);
        footer.write(&mut file_bytes).unwrap();

        let mut cursor = Cursor::new(file_bytes);
        crate::binary::read_header(&mut cursor).unwrap();
    }
}

/// String table for deduplicating sequence names
#[derive(Clone)]
pub struct StringTable {
    strings: Vec<String>,
    lengths: Vec<u64>,
    index: HashMap<String, u64>,
}

impl Default for StringTable {
    fn default() -> Self {
        Self::new()
    }
}

impl StringTable {
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            lengths: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Get existing ID for string, or insert and return new ID
    pub fn get_or_insert_id(&mut self, s: &str, length: u64) -> u64 {
        if let Some(&id) = self.index.get(s) {
            id
        } else {
            let id = self.strings.len() as u64;
            self.strings.push(s.to_string());
            self.lengths.push(length);
            self.index.insert(s.to_string(), id);
            id
        }
    }

    pub fn get(&self, id: u64) -> Option<&str> {
        self.strings.get(id as usize).map(|s| s.as_str())
    }

    pub fn get_length(&self, id: u64) -> Option<u64> {
        self.lengths.get(id as usize).copied()
    }

    pub fn get_name_and_len(&self, id: u64) -> Option<(&str, u64)> {
        let name = self.strings.get(id as usize)?;
        let len = self.lengths.get(id as usize)?;
        Some((name.as_str(), *len))
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }

    pub(crate) fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        for (s, &len) in self.strings.iter().zip(self.lengths.iter()) {
            write_varint(writer, s.len() as u64)?;
            writer.write_all(s.as_bytes())?;
            write_varint(writer, len)?;
        }
        Ok(())
    }

    pub fn read<R: Read>(reader: &mut R, num_strings: u64) -> io::Result<Self> {
        let num_strings = num_strings as usize;
        let mut strings = Vec::with_capacity(num_strings);
        let mut lengths = Vec::with_capacity(num_strings);
        let mut index = HashMap::new();

        for id in 0..num_strings {
            let len = read_varint(reader)? as usize;
            let mut buf = vec![0u8; len];
            reader.read_exact(&mut buf)?;
            let s = String::from_utf8(buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let seq_len = read_varint(reader)?;
            index.insert(s.clone(), id as u64);
            strings.push(s);
            lengths.push(seq_len);
        }
        Ok(Self {
            strings,
            lengths,
            index,
        })
    }

    /// Skip over string table entries without loading them into memory.
    /// Used for all-records mode where we need to find the BGZF section start.
    pub fn skip<R: Read>(reader: &mut R, num_strings: u64) -> io::Result<()> {
        for _ in 0..num_strings {
            let name_len = read_varint(reader)? as usize;
            // Skip over the name bytes
            io::copy(&mut reader.take(name_len as u64), &mut io::sink())?;
            // Skip the sequence length varint
            read_varint(reader)?;
        }
        Ok(())
    }
}

pub struct CompactRecord {
    pub query_name_id: u64,
    pub query_start: u64,
    pub query_end: u64,
    pub strand: char,
    pub target_name_id: u64,
    pub target_start: u64,
    pub target_end: u64,
    pub residue_matches: u64,
    pub alignment_block_len: u64,
    pub mapping_quality: u8,
    pub tracepoints: TracepointData,
    pub tags: Vec<Tag>,
}

#[derive(Debug)]
pub struct Tag {
    pub key: [u8; 2],
    pub tag_type: u8,
    pub value: TagValue,
}

#[derive(Debug)]
pub enum TagValue {
    Int(i32),
    Float(f32),
    String(String),
}
impl Tag {
    pub(crate) fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&self.key)?;
        writer.write_all(&[self.tag_type])?;
        match &self.value {
            TagValue::Int(v) => writer.write_all(&v.to_le_bytes())?,
            TagValue::Float(v) => writer.write_all(&v.to_le_bytes())?,
            TagValue::String(s) => {
                write_varint(writer, s.len() as u64)?;
                writer.write_all(s.as_bytes())?;
            }
        }
        Ok(())
    }

    pub(crate) fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut key = [0u8; 2];
        reader.read_exact(&mut key)?;
        let mut tag_type = [0u8; 1];
        reader.read_exact(&mut tag_type)?;
        let value = match tag_type[0] {
            b'i' => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                TagValue::Int(i32::from_le_bytes(buf))
            }
            b'f' => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                TagValue::Float(f32::from_le_bytes(buf))
            }
            b'Z' => {
                let len = read_varint(reader)?;
                let mut buf = vec![0u8; len as usize];
                reader.read_exact(&mut buf)?;
                TagValue::String(String::from_utf8(buf).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Invalid UTF-8 in tag: {}", e),
                    )
                })?)
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown tag type: {}", tag_type[0] as char),
                ))
            }
        };
        Ok(Self {
            key,
            tag_type: tag_type[0],
            value,
        })
    }
}

pub fn parse_tag(field: &str) -> Option<Tag> {
    let parts: Vec<&str> = field.splitn(3, ':').collect();
    if parts.len() != 3 || parts[0].len() < 2 || parts[1].is_empty() {
        return None;
    }
    let key = [parts[0].as_bytes()[0], parts[0].as_bytes()[1]];
    let tag_type = parts[1].as_bytes()[0];
    let value = match tag_type {
        b'i' => match parts[2].parse::<i32>() {
            Ok(v) => TagValue::Int(v),
            Err(e) => {
                log::warn!("Failed to parse integer tag '{}': {}", field, e);
                return None;
            }
        },
        b'f' => match parts[2].parse::<f32>() {
            Ok(v) => TagValue::Float(v),
            Err(e) => {
                log::warn!("Failed to parse float tag '{}': {}", field, e);
                return None;
            }
        },
        b'Z' => TagValue::String(parts[2].to_string()),
        _ => return None,
    };
    Some(Tag {
        key,
        tag_type,
        value,
    })
}

pub fn format_tag(tag: &Tag) -> String {
    let key = String::from_utf8_lossy(&tag.key);
    let tag_type = tag.tag_type as char;
    match &tag.value {
        TagValue::Int(v) => format!("{}:{}:{}", key, tag_type, v),
        // Use enough precision to round-trip f32 values correctly
        // f32 has ~7 significant digits, so we use 8 to ensure round-trip
        TagValue::Float(v) => format!("{}:{}:{:.8}", key, tag_type, v),
        TagValue::String(s) => format!("{}:Z:{}", key, s),
    }
}
