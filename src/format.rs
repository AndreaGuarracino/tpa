//! Data structures for Binary PAF format

use crate::binary::{complexity_metric_from_u8, complexity_metric_to_u8, BINARY_MAGIC};
use crate::{utils::*, Distance};
use lib_tracepoints::{ComplexityMetric, MixedRepresentation, TracepointType};
use std::collections::HashMap;
use std::io::{self, Read, Write};

#[derive(Clone, Copy, Debug)]
pub enum CompressionStrategy {
    /// Automatic encoding decision + Zstd
    /// - Analyze first 1000 records to decide delta vs raw encoding
    /// - Configurable compression level (max 22, default: 3)
    Automatic(i32),
    /// Raw encoding (no delta) + Varint + Zstd
    /// - No delta encoding, stores absolute values
    /// - Configurable compression level (max 22, default: 3)
    VarintZstd(i32),
    /// Delta encoding + Varint + Zstd
    /// - Always uses delta encoding for tracepoints
    /// - Works well when values are naturally small or monotonic
    /// - Configurable compression level (max 22, default: 3)
    DeltaVarintZstd(i32),
}

impl CompressionStrategy {
    /// Parse strategy from string (format: "strategy" or "strategy,level")
    pub fn from_str(s: &str) -> Result<Self, String> {
        let parts: Vec<&str> = s.split(',').collect();
        let strategy_name = parts[0].to_lowercase();
        let compression_level = if parts.len() > 1 {
            parts[1].trim().parse::<i32>().map_err(|_| {
                format!(
                    "Invalid compression level '{}'. Must be a number between 1 and 22.",
                    parts[1]
                )
            })?
        } else {
            3 // Default compression level
        };

        // Validate compression level range
        if compression_level < 1 || compression_level > 22 {
            return Err(format!(
                "Compression level {} is out of range. Must be between 1 and 22.",
                compression_level
            ));
        }

        match strategy_name.as_str() {
            "automatic" => Ok(CompressionStrategy::Automatic(compression_level)),
            "varint-zstd" => Ok(CompressionStrategy::VarintZstd(compression_level)),
            "delta-varint-zstd" => Ok(CompressionStrategy::DeltaVarintZstd(compression_level)),
            _ => Err(format!(
                "Unsupported compression strategy '{}'. Supported: 'automatic', 'varint-zstd', 'delta-varint-zstd'.",
                strategy_name
            )),
        }
    }

    /// Get all available strategies
    pub fn variants() -> &'static [&'static str] {
        &["automatic", "varint-zstd", "delta-varint-zstd"]
    }

    /// Convert to strategy code for file header
    fn to_code(&self) -> u8 {
        match self {
            CompressionStrategy::Automatic(_) => 0,
            CompressionStrategy::VarintZstd(_) => 1,
            CompressionStrategy::DeltaVarintZstd(_) => 2,
        }
    }

    /// Parse from strategy code
    fn from_code(code: u8) -> io::Result<Self> {
        match code {
            0 => Ok(CompressionStrategy::Automatic(3)),
            1 => Ok(CompressionStrategy::VarintZstd(3)),
            2 => Ok(CompressionStrategy::DeltaVarintZstd(3)),
            _ => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "Unsupported compression strategy code: {}. Supported codes: 0=automatic, 1=varint-zstd, 2=delta-varint-zstd.",
                    code
                ),
            )),
        }
    }

    /// Get zstd compression level for this strategy
    pub fn zstd_level(&self) -> i32 {
        match self {
            CompressionStrategy::Automatic(level) => *level,
            CompressionStrategy::VarintZstd(level) => *level,
            CompressionStrategy::DeltaVarintZstd(level) => *level,
        }
    }
}

impl std::fmt::Display for CompressionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionStrategy::Automatic(level) => {
                if *level == 3 {
                    write!(f, "Automatic")
                } else {
                    write!(f, "Automatic,{}", level)
                }
            }
            CompressionStrategy::VarintZstd(level) => {
                if *level == 3 {
                    write!(f, "Varint-zstd")
                } else {
                    write!(f, "Varint-zstd,{}", level)
                }
            }
            CompressionStrategy::DeltaVarintZstd(level) => {
                if *level == 3 {
                    write!(f, "Delta-varint-zstd")
                } else {
                    write!(f, "Delta-varint-zstd,{}", level)
                }
            }
        }
    }
}

pub struct BinaryPafHeader {
    pub(crate) version: u8,
    pub(crate) flags: u8,
    pub(crate) num_records: u64,
    pub(crate) num_strings: u64,
    pub(crate) tracepoint_type: TracepointType,
    pub(crate) complexity_metric: ComplexityMetric,
    pub(crate) max_complexity: u64, // For Standard/Mixed/Variable: max_value; For FASTGA: trace_spacing
    pub(crate) distance: Distance,
}

impl BinaryPafHeader {
    /// Create header with strategy and optional automatic encoding flags
    pub(crate) fn new(
        num_records: u64,
        num_strings: u64,
        strategy: CompressionStrategy,
        use_delta_first: bool,
        use_delta_second: bool,
        tracepoint_type: TracepointType,
        complexity_metric: ComplexityMetric,
        max_complexity: u64,
        distance: Distance,
    ) -> Self {
        let mut flags = strategy.to_code() & 0x07; // Strategy in bits 0-2
        if matches!(strategy, CompressionStrategy::Automatic(_)) {
            if use_delta_first {
                flags |= 0x08; // Bit 3
            }
            if use_delta_second {
                flags |= 0x10; // Bit 4
            }
        }
        Self {
            version: 1,
            flags,
            num_records,
            num_strings,
            tracepoint_type,
            complexity_metric,
            max_complexity,
            distance,
        }
    }

    /// Get delta encoding flag for first values (Automatic mode only)
    pub(crate) fn use_delta_first(&self) -> bool {
        (self.flags & 0x08) != 0
    }

    /// Get delta encoding flag for second values (Automatic mode only)
    pub(crate) fn use_delta_second(&self) -> bool {
        (self.flags & 0x10) != 0
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
        CompressionStrategy::from_code(self.flags & 0x07)
    }

    /// Get delta encoding flag for first values (Automatic mode only)
    pub fn delta_first(&self) -> bool {
        (self.flags & 0x08) != 0
    }

    /// Get delta encoding flag for second values (Automatic mode only)
    pub fn delta_second(&self) -> bool {
        (self.flags & 0x10) != 0
    }

    /// Get tracepoint type
    pub fn tp_type(&self) -> TracepointType {
        self.tracepoint_type
    }

    /// Get distance mode
    pub fn distance(&self) -> &Distance {
        &self.distance
    }

    /// Get complexity metric
    pub fn complexity_metric(&self) -> ComplexityMetric {
        self.complexity_metric
    }

    /// Get max complexity
    pub fn max_complexity(&self) -> u64 {
        self.max_complexity
    }

    pub(crate) fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(BINARY_MAGIC)?;
        writer.write_all(&[self.version, self.flags])?;
        write_varint(writer, self.num_records)?;
        write_varint(writer, self.num_strings)?;
        writer.write_all(&[self.tracepoint_type.to_u8()])?;
        writer.write_all(&[complexity_metric_to_u8(&self.complexity_metric)])?;
        write_varint(writer, self.max_complexity)?;
        write_distance(writer, &self.distance)?;
        Ok(())
    }

    pub(crate) fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != BINARY_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic"));
        }
        let mut ver_flags = [0u8; 2];
        reader.read_exact(&mut ver_flags)?;
        let version = ver_flags[0];
        let flags = ver_flags[1];

        let num_records = read_varint(reader)?;
        let num_strings = read_varint(reader)?;

        // Read tracepoint metadata from header
        let mut tp_type_buf = [0u8; 1];
        reader.read_exact(&mut tp_type_buf)?;
        let tracepoint_type = TracepointType::from_u8(tp_type_buf[0])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut metric_buf = [0u8; 1];
        reader.read_exact(&mut metric_buf)?;
        let complexity_metric = complexity_metric_from_u8(metric_buf[0])?;

        let max_complexity = read_varint(reader)?;

        let distance = read_distance(reader)?;

        Ok(Self {
            version,
            flags,
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
        write_varint(writer, self.strings.len() as u64)?;
        for (s, &len) in self.strings.iter().zip(self.lengths.iter()) {
            write_varint(writer, s.len() as u64)?;
            writer.write_all(s.as_bytes())?;
            write_varint(writer, len)?;
        }
        Ok(())
    }

    pub(crate) fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let num_strings = read_varint(reader)? as usize;
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
    pub tp_type: TracepointType,
    pub complexity_metric: ComplexityMetric,
    pub max_complexity: u64,
    pub tracepoints: TracepointData,
    pub tags: Vec<Tag>,
}

pub enum TracepointData {
    Standard(Vec<(u64, u64)>),
    Mixed(Vec<MixedTracepointItem>),
    Variable(Vec<(u64, Option<u64>)>),
    Fastga(Vec<(u64, u64)>),
}

#[derive(Debug, Clone)]
pub enum MixedTracepointItem {
    Tracepoint(u64, u64),
    CigarOp(u64, char),
}

impl From<&MixedRepresentation> for MixedTracepointItem {
    fn from(item: &MixedRepresentation) -> Self {
        match item {
            MixedRepresentation::Tracepoint(a, b) => {
                MixedTracepointItem::Tracepoint(*a as u64, *b as u64)
            }
            MixedRepresentation::CigarOp(len, op) => MixedTracepointItem::CigarOp(*len as u64, *op),
        }
    }
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

/// CIGAR operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CigarOp {
    Match = 0,
    Mismatch = 1,
    Insertion = 2,
    Deletion = 3,
}
