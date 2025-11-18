//! Data structures for Binary PAF format

use crate::binary::BINARY_MAGIC;
use crate::{utils::*, Distance};
use lib_tracepoints::{ComplexityMetric, TracepointData, TracepointType};
use std::collections::HashMap;
use std::io::{self, Read, Write};

/// Compression layer to use for final compression
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
}

#[derive(Clone, Debug)]
pub enum CompressionStrategy {
    /// Automatic strategy selection based on data analysis
    /// - Runs heuristic to choose optimal strategy
    /// - Configurable compression level (max 22, default: 3)
    Automatic(i32),
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
    /// Offset-based Joint Encoding
    /// - Encode target as offset from query (simpler than 2D-Delta)
    /// - Optimal for high correlation data (ρ > 0.95)
    /// - Configurable compression level (max 22, default: 3)
    OffsetJoint(i32),
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
    /// Simple8b-RLE style encoding
    /// - Pack multiple small integers into 64-bit words with RLE mode
    /// - Optimal for runs and small values (4+ billion ints/sec)
    /// - Configurable compression level (max 22, default: 3)
    Simple8(i32),
    /// Stream VByte byte-aligned encoding
    /// - Separated control and data bytes for SIMD acceleration
    /// - Optimal for byte-range data (1.1-4.0 billion ints/sec)
    /// - Configurable compression level (max 22, default: 3)
    StreamVByte(i32),
    /// Adaptive Correlation-based strategy selection
    /// - Automatically switches between strategies based on correlation
    /// - Measures ρ per block: >0.95 → joint, >0.50 → delta, <0.50 → FOR
    /// - Optimal for heterogeneous data patterns
    /// - Configurable compression level (max 22, default: 3)
    AdaptiveCorrelation(i32),
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
    /// Dual-strategy: apply different strategies to first_values and second_values
    /// - first_strategy for first_values stream
    /// - second_strategy for second_values stream
    /// - Optimal when streams have different statistical properties (5-15% improvement on CIGAR data)
    /// - Configurable compression level (max 22, default: 3)
    Dual(Box<CompressionStrategy>, Box<CompressionStrategy>, i32),
}

impl CompressionStrategy {
    /// Helper to check if a string is a valid strategy name
    #[allow(dead_code)]
    fn is_strategy_name(s: &str) -> bool {
        matches!(
            s,
            "automatic"
                | "raw"
                | "zigzag-delta"
                | "2d-delta"
                | "rle"
                | "bit-packed"
                | "delta-of-delta"
                | "frame-of-reference"
                | "for"
                | "hybrid-rle"
                | "offset-joint"
                | "xor-delta"
                | "dictionary"
                | "dict"
                | "simple8"
                | "stream-vbyte"
                | "streamvbyte"
                | "adaptive-correlation"
                | "adaptive"
                | "fastpfor"
                | "fast-pfor"
                | "cascaded"
                | "simple8b-full"
                | "simple8bfull"
                | "selective-rle"
                | "selectiverle"
        )
    }

    /// Parse strategy from string with compression layer
    /// Formats:
    ///   - Single: "strategy", "strategy,level", "strategy-bgzip", "strategy-nocomp"
    ///   - Dual: "strategy1:strategy2", "strategy1:strategy2,level", "strategy1:strategy2-bgzip"
    pub fn from_str_with_layer(s: &str) -> Result<(Self, CompressionLayer), String> {
        // Check if this is a dual strategy (contains ':')
        if s.contains(':') {
            return Self::parse_dual_strategy(s);
        }

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

        let compression_level = if parts.len() > 1 {
            parts[1].trim().parse::<i32>().map_err(|_| {
                format!(
                    "Invalid compression level '{}'. Must be a number between 0 and 22.",
                    parts[1]
                )
            })?
        } else if strategy_name.ends_with("-nocomp") || s.contains("-nocomp") {
            0 // nocomp means level 0
        } else {
            3 // Default compression level
        };

        // Validate compression level range (allow 0 for nocomp)
        if compression_level < 0 || compression_level > 22 {
            return Err(format!(
                "Compression level {} is out of range. Must be between 0 and 22.",
                compression_level
            ));
        }

        let strategy = match strategy_name.as_str() {
            "automatic" => Ok(CompressionStrategy::Automatic(compression_level)),
            "raw" => Ok(CompressionStrategy::Raw(compression_level)),
            "zigzag-delta" => Ok(CompressionStrategy::ZigzagDelta(compression_level)),
            "2d-delta" => Ok(CompressionStrategy::TwoDimDelta(compression_level)),
            "rle" => Ok(CompressionStrategy::RunLength(compression_level)),
            "bit-packed" => Ok(CompressionStrategy::BitPacked(compression_level)),
            "delta-of-delta" => Ok(CompressionStrategy::DeltaOfDelta(compression_level)),
            "frame-of-reference" | "for" => Ok(CompressionStrategy::FrameOfReference(compression_level)),
            "hybrid-rle" => Ok(CompressionStrategy::HybridRLE(compression_level)),
            "offset-joint" => Ok(CompressionStrategy::OffsetJoint(compression_level)),
            "xor-delta" => Ok(CompressionStrategy::XORDelta(compression_level)),
            "dictionary" | "dict" => Ok(CompressionStrategy::Dictionary(compression_level)),
            "simple8" => Ok(CompressionStrategy::Simple8(compression_level)),
            "stream-vbyte" | "streamvbyte" => Ok(CompressionStrategy::StreamVByte(compression_level)),
            "adaptive-correlation" | "adaptive" => Ok(CompressionStrategy::AdaptiveCorrelation(compression_level)),
            "fastpfor" | "fast-pfor" => Ok(CompressionStrategy::FastPFOR(compression_level)),
            "cascaded" => Ok(CompressionStrategy::Cascaded(compression_level)),
            "simple8b-full" | "simple8bfull" => Ok(CompressionStrategy::Simple8bFull(compression_level)),
            "selective-rle" | "selectiverle" => Ok(CompressionStrategy::SelectiveRLE(compression_level)),
            _ => Err(format!(
                "Unsupported compression strategy '{}'. Use --help to see all available strategies.",
                strategy_name
            )),
        }?;

        Ok((strategy, layer))
    }

    /// Parse dual strategy from string (format: "strategy1:strategy2" or "strategy1:strategy2,level")
    fn parse_dual_strategy(s: &str) -> Result<(Self, CompressionLayer), String> {
        // Split by comma first to extract level/layer
        let parts: Vec<&str> = s.split(',').collect();
        let strategy_part = parts[0];

        // Extract compression layer from suffix
        let (clean_part, layer) = if strategy_part.ends_with("-bgzip") {
            (
                strategy_part.trim_end_matches("-bgzip"),
                CompressionLayer::Bgzip,
            )
        } else if strategy_part.ends_with("-nocomp") {
            (
                strategy_part.trim_end_matches("-nocomp"),
                CompressionLayer::Nocomp,
            )
        } else {
            (strategy_part, CompressionLayer::Zstd)
        };

        // Split by colon to get both strategies
        let strat_parts: Vec<&str> = clean_part.split(':').collect();
        if strat_parts.len() != 2 {
            return Err(format!(
                "Dual strategy must be in format 'strategy1:strategy2', got '{}'",
                s
            ));
        }

        let first_name = strat_parts[0].trim().to_lowercase();
        let second_name = strat_parts[1].trim().to_lowercase();

        // Parse compression level
        let compression_level = if parts.len() > 1 {
            parts[1].trim().parse::<i32>().map_err(|_| {
                format!(
                    "Invalid compression level '{}'. Must be a number between 0 and 22.",
                    parts[1]
                )
            })?
        } else {
            3 // Default
        };

        // Validate compression level
        if compression_level < 0 || compression_level > 22 {
            return Err(format!(
                "Compression level {} is out of range. Must be between 0 and 22.",
                compression_level
            ));
        }

        // Parse first strategy
        let first_strategy = Self::parse_single_strategy(&first_name, compression_level)?;
        // Parse second strategy
        let second_strategy = Self::parse_single_strategy(&second_name, compression_level)?;

        Ok((
            CompressionStrategy::Dual(
                Box::new(first_strategy),
                Box::new(second_strategy),
                compression_level,
            ),
            layer,
        ))
    }

    /// Parse a single strategy name into a CompressionStrategy enum
    fn parse_single_strategy(name: &str, level: i32) -> Result<CompressionStrategy, String> {
        match name {
            "automatic" => Ok(CompressionStrategy::Automatic(level)),
            "raw" => Ok(CompressionStrategy::Raw(level)),
            "zigzag-delta" => Ok(CompressionStrategy::ZigzagDelta(level)),
            "2d-delta" => Ok(CompressionStrategy::TwoDimDelta(level)),
            "rle" => Ok(CompressionStrategy::RunLength(level)),
            "bit-packed" => Ok(CompressionStrategy::BitPacked(level)),
            "delta-of-delta" => Ok(CompressionStrategy::DeltaOfDelta(level)),
            "frame-of-reference" | "for" => Ok(CompressionStrategy::FrameOfReference(level)),
            "hybrid-rle" => Ok(CompressionStrategy::HybridRLE(level)),
            "offset-joint" => Ok(CompressionStrategy::OffsetJoint(level)),
            "xor-delta" => Ok(CompressionStrategy::XORDelta(level)),
            "dictionary" | "dict" => Ok(CompressionStrategy::Dictionary(level)),
            "simple8" => Ok(CompressionStrategy::Simple8(level)),
            "stream-vbyte" | "streamvbyte" => Ok(CompressionStrategy::StreamVByte(level)),
            "adaptive-correlation" | "adaptive" => {
                Ok(CompressionStrategy::AdaptiveCorrelation(level))
            }
            "fastpfor" | "fast-pfor" => Ok(CompressionStrategy::FastPFOR(level)),
            "cascaded" => Ok(CompressionStrategy::Cascaded(level)),
            "simple8b-full" | "simple8bfull" => Ok(CompressionStrategy::Simple8bFull(level)),
            "selective-rle" | "selectiverle" => Ok(CompressionStrategy::SelectiveRLE(level)),
            _ => Err(format!(
                "Unsupported compression strategy '{}'. Use --help to see all available strategies.",
                name
            )),
        }
    }

    /// Parse strategy from string (format: "strategy" or "strategy,level") - defaults to Zstd
    /// Also sets the thread-local compression layer based on suffix (-bgzip/-nocomp)
    pub fn from_str(s: &str) -> Result<Self, String> {
        let (strategy, _layer) = Self::from_str_with_layer(s)?;
        // Note: Layer is now passed explicitly through API calls, not via thread-local state
        Ok(strategy)
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
            "offset-joint",
            "xor-delta",
            "dictionary",
            "simple8",
            "stream-vbyte",
            "adaptive-correlation",
        ]
    }

    /// Convert to strategy code for file header
    fn to_code(&self) -> u8 {
        match self {
            CompressionStrategy::Automatic(_) | CompressionStrategy::AdaptiveCorrelation(_) => {
                panic!("Automatic strategies must be resolved before writing")
            }
            CompressionStrategy::Dual(_, _, _) => {
                254 // Special code indicating dual strategy (sub-strategies follow)
            }
            CompressionStrategy::Raw(_) => 0,
            CompressionStrategy::ZigzagDelta(_) => 1,
            CompressionStrategy::TwoDimDelta(_) => 2,
            CompressionStrategy::RunLength(_) => 3,
            CompressionStrategy::BitPacked(_) => 4,
            CompressionStrategy::DeltaOfDelta(_) => 5,
            CompressionStrategy::FrameOfReference(_) => 6,
            CompressionStrategy::HybridRLE(_) => 7,
            CompressionStrategy::OffsetJoint(_) => 8,
            CompressionStrategy::XORDelta(_) => 9,
            CompressionStrategy::Dictionary(_) => 10,
            CompressionStrategy::Simple8(_) => 11,
            CompressionStrategy::StreamVByte(_) => 12,
            CompressionStrategy::FastPFOR(_) => 13,
            CompressionStrategy::Cascaded(_) => 14,
            CompressionStrategy::Simple8bFull(_) => 15,
            CompressionStrategy::SelectiveRLE(_) => 16,
        }
    }

    /// Parse from strategy code
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
            8 => Ok(CompressionStrategy::OffsetJoint(3)),
            9 => Ok(CompressionStrategy::XORDelta(3)),
            10 => Ok(CompressionStrategy::Dictionary(3)),
            11 => Ok(CompressionStrategy::Simple8(3)),
            12 => Ok(CompressionStrategy::StreamVByte(3)),
            13 => Ok(CompressionStrategy::FastPFOR(3)),
            14 => Ok(CompressionStrategy::Cascaded(3)),
            15 => Ok(CompressionStrategy::Simple8bFull(3)),
            16 => Ok(CompressionStrategy::SelectiveRLE(3)),
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Unsupported compression strategy code: {}", code),
            )),
        }
    }

    /// Get zstd compression level for this strategy
    pub fn zstd_level(&self) -> i32 {
        match self {
            CompressionStrategy::Automatic(level) => *level,
            CompressionStrategy::Dual(_, _, level) => *level,
            CompressionStrategy::Raw(level) => *level,
            CompressionStrategy::ZigzagDelta(level) => *level,
            CompressionStrategy::TwoDimDelta(level) => *level,
            CompressionStrategy::RunLength(level) => *level,
            CompressionStrategy::BitPacked(level) => *level,
            CompressionStrategy::DeltaOfDelta(level) => *level,
            CompressionStrategy::FrameOfReference(level) => *level,
            CompressionStrategy::HybridRLE(level) => *level,
            CompressionStrategy::OffsetJoint(level) => *level,
            CompressionStrategy::XORDelta(level) => *level,
            CompressionStrategy::Dictionary(level) => *level,
            CompressionStrategy::Simple8(level) => *level,
            CompressionStrategy::StreamVByte(level) => *level,
            CompressionStrategy::AdaptiveCorrelation(level) => *level,
            CompressionStrategy::FastPFOR(level) => *level,
            CompressionStrategy::Cascaded(level) => *level,
            CompressionStrategy::Simple8bFull(level) => *level,
            CompressionStrategy::SelectiveRLE(level) => *level,
        }
    }
}

impl std::fmt::Display for CompressionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionStrategy::Automatic(level) => {
                write!(f, "Automatic (level {})", level)
            }
            CompressionStrategy::Dual(first, second, level) => {
                write!(f, "Dual({} : {}, level {})", first, second, level)
            }
            CompressionStrategy::Raw(level) => {
                write!(f, "Raw (level {})", level)
            }
            CompressionStrategy::ZigzagDelta(level) => {
                write!(f, "ZigzagDelta (level {})", level)
            }
            CompressionStrategy::TwoDimDelta(level) => {
                write!(f, "TwoDimDelta (level {})", level)
            }
            CompressionStrategy::RunLength(level) => {
                write!(f, "RunLength (level {})", level)
            }
            CompressionStrategy::BitPacked(level) => {
                write!(f, "BitPacked (level {})", level)
            }
            CompressionStrategy::DeltaOfDelta(level) => {
                write!(f, "DeltaOfDelta (level {})", level)
            }
            CompressionStrategy::FrameOfReference(level) => {
                write!(f, "FrameOfReference (level {})", level)
            }
            CompressionStrategy::HybridRLE(level) => {
                write!(f, "HybridRLE (level {})", level)
            }
            CompressionStrategy::OffsetJoint(level) => {
                write!(f, "OffsetJoint (level {})", level)
            }
            CompressionStrategy::XORDelta(level) => {
                write!(f, "XORDelta (level {})", level)
            }
            CompressionStrategy::Dictionary(level) => {
                write!(f, "Dictionary (level {})", level)
            }
            CompressionStrategy::Simple8(level) => {
                write!(f, "Simple8 (level {})", level)
            }
            CompressionStrategy::StreamVByte(level) => {
                write!(f, "StreamVByte (level {})", level)
            }
            CompressionStrategy::AdaptiveCorrelation(level) => {
                write!(f, "AdaptiveCorrelation (level {})", level)
            }
            CompressionStrategy::FastPFOR(level) => {
                write!(f, "FastPFOR (level {})", level)
            }
            CompressionStrategy::Cascaded(level) => {
                write!(f, "Cascaded (level {})", level)
            }
            CompressionStrategy::Simple8bFull(level) => {
                write!(f, "Simple8bFull (level {})", level)
            }
            CompressionStrategy::SelectiveRLE(level) => {
                write!(f, "SelectiveRLE (level {})", level)
            }
        }
    }
}

pub struct BinaryPafHeader {
    pub(crate) version: u8,
    pub(crate) first_strategy_code: u8,
    pub(crate) second_strategy_code: u8,
    pub(crate) compression_layer: CompressionLayer,
    pub(crate) num_records: u64,
    pub(crate) num_strings: u64,
    pub(crate) tracepoint_type: TracepointType,
    pub(crate) complexity_metric: ComplexityMetric,
    pub(crate) max_complexity: u64, // For Standard/Mixed/Variable: max_value; For FASTGA: trace_spacing
    pub(crate) distance: Distance,
}

impl BinaryPafHeader {
    /// Create header with strategy
    pub(crate) fn new(
        num_records: u64,
        num_strings: u64,
        strategy: CompressionStrategy,
        compression_layer: CompressionLayer,
        tracepoint_type: TracepointType,
        complexity_metric: ComplexityMetric,
        max_complexity: u64,
        distance: Distance,
    ) -> Self {
        let (first_strategy_code, second_strategy_code) = match &strategy {
            CompressionStrategy::Dual(first, second, _level) => (first.to_code(), second.to_code()),
            _ => {
                let code = strategy.to_code();
                (code, code) // For single strategies, write same code twice
            }
        };

        Self {
            version: 1,
            first_strategy_code,
            second_strategy_code,
            compression_layer,
            num_records,
            num_strings,
            tracepoint_type,
            complexity_metric,
            max_complexity,
            distance,
        }
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

    /// Get compression strategy
    pub fn strategy(&self) -> io::Result<CompressionStrategy> {
        if self.first_strategy_code != self.second_strategy_code {
            // Dual strategy: different codes for first and second values
            let first_strat = CompressionStrategy::from_code(self.first_strategy_code)?;
            let second_strat = CompressionStrategy::from_code(self.second_strategy_code)?;

            // Use default level 3 when reading (level only matters for encoding)
            Ok(CompressionStrategy::Dual(
                Box::new(first_strat),
                Box::new(second_strat),
                3,
            ))
        } else {
            // Single strategy: same code for both values
            CompressionStrategy::from_code(self.first_strategy_code)
        }
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
        self.distance.clone()
    }

    /// Get complexity metric
    pub fn complexity_metric(&self) -> ComplexityMetric {
        self.complexity_metric
    }

    /// Get max complexity
    pub fn max_complexity(&self) -> u64 {
        self.max_complexity
    }

    /// Get compression layer
    pub fn compression_layer(&self) -> CompressionLayer {
        self.compression_layer
    }

    pub(crate) fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(BINARY_MAGIC)?;
        writer.write_all(&[
            self.version,
            self.compression_layer.to_u8(),
            self.first_strategy_code,
            self.second_strategy_code,
        ])?;

        write_varint(writer, self.num_records)?;
        write_varint(writer, self.num_strings)?;
        writer.write_all(&[self.tracepoint_type.to_u8()])?;
        writer.write_all(&[self.complexity_metric.to_u8()])?;
        write_varint(writer, self.max_complexity)?;
        write_distance(writer, &self.distance)?;
        Ok(())
    }

    pub fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != BINARY_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic"));
        }

        // Fixed header layout (version 1):
        // [version][compression_layer][first_strategy][second_strategy]
        let mut header_bytes = [0u8; 4];
        reader.read_exact(&mut header_bytes)?;
        let version = header_bytes[0];
        if version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported format version: {}", version),
            ));
        }
        let compression_layer = CompressionLayer::from_u8(header_bytes[1])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let first_strategy_code = header_bytes[2];
        let second_strategy_code = header_bytes[3];

        let num_records = read_varint(reader)?;
        let num_strings = read_varint(reader)?;

        // Read tracepoint metadata from header
        let mut tp_type_buf = [0u8; 1];
        reader.read_exact(&mut tp_type_buf)?;
        let tracepoint_type = TracepointType::from_u8(tp_type_buf[0])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut metric_buf = [0u8; 1];
        reader.read_exact(&mut metric_buf)?;
        let complexity_metric = ComplexityMetric::from_u8(metric_buf[0])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let max_complexity = read_varint(reader)?;

        let distance = read_distance(reader)?;

        Ok(Self {
            version,
            first_strategy_code,
            second_strategy_code,
            compression_layer,
            num_records,
            num_strings,
            tracepoint_type,
            complexity_metric,
            max_complexity,
            distance,
        })
    }
}

/// String table for deduplicating sequence names
#[derive(Clone)]
pub struct StringTable {
    strings: Vec<String>,
    lengths: Vec<u64>,
    index: HashMap<String, u64>,
}

impl StringTable {
    pub fn new() -> Self {
        Self {
            strings: Vec::new(),
            lengths: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub fn intern(&mut self, s: &str, length: u64) -> u64 {
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
}

pub struct AlignmentRecord {
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
    if parts.len() != 3 {
        return None;
    }
    let key = [parts[0].as_bytes()[0], parts[0].as_bytes()[1]];
    let tag_type = parts[1].as_bytes()[0];
    let value = match tag_type {
        b'i' => parts[2].parse::<i32>().ok().map(TagValue::Int)?,
        b'f' => parts[2].parse::<f32>().ok().map(TagValue::Float)?,
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
        TagValue::Float(v) => format!("{}:{}:{}", key, tag_type, v),
        TagValue::String(s) => format!("{}:Z:{}", key, s),
    }
}
