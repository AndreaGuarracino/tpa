/// Binary PAF format for efficient storage of sequence alignments with tracepoints
///
/// Format: [Header] → [Records] → [StringTable]
/// - Header: Magic "BPAF" + version + metadata
/// - Records: Core PAF fields + compressed tracepoints
/// - StringTable: Deduplicated sequence names with lengths
///
/// Compression strategy:
/// - Delta encoding: positions stored as deltas from previous position
/// - Varint encoding: variable-length integer encoding
/// - Zstd level 3: fast compression with good ratios
use flate2::read::MultiGzDecoder;
use lib_tracepoints::{
    cigar_to_mixed_tracepoints, cigar_to_tracepoints, cigar_to_tracepoints_fastga,
    cigar_to_variable_tracepoints, ComplexityMetric, MixedRepresentation, TracepointType,
};
use log::{debug, error, info};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

// ============================================================================
// COMPRESSION STRATEGY
// ============================================================================

/// Compression strategy for binary PAF format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
                format!("Invalid compression level '{}'. Must be a number between 1 and 22.", parts[1])
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
                    write!(f, "automatic")
                } else {
                    write!(f, "automatic,{}", level)
                }
            }
            CompressionStrategy::VarintZstd(level) => {
                if *level == 3 {
                    write!(f, "varint-zstd")
                } else {
                    write!(f, "varint-zstd,{}", level)
                }
            }
            CompressionStrategy::DeltaVarintZstd(level) => {
                if *level == 3 {
                    write!(f, "delta-varint-zstd")
                } else {
                    write!(f, "delta-varint-zstd,{}", level)
                }
            }
        }
    }
}

// ============================================================================
// VARINT ENCODING (LEB128)
// ============================================================================

/// Encode an unsigned integer as a varint
fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // Set continuation bit
        }
        bytes.push(byte);
        if value == 0 {
            break;
        }
    }
    bytes
}

/// Write a varint to a writer
fn write_varint<W: Write>(writer: &mut W, value: u64) -> io::Result<usize> {
    let bytes = encode_varint(value);
    writer.write_all(&bytes)?;
    Ok(bytes.len())
}

/// Read a varint from a reader
#[inline]
fn read_varint<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut value: u64 = 0;
    let mut shift = 0;
    loop {
        let mut byte_buf = [0u8; 1];
        reader.read_exact(&mut byte_buf)?;
        let byte = byte_buf[0];
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Varint too long",
            ));
        }
    }
    Ok(value)
}


// ============================================================================
// BINARY PAF FORMAT
// ============================================================================

const BINARY_MAGIC: &[u8; 4] = b"BPAF";

/// Binary PAF header
#[derive(Debug)]
pub struct BinaryPafHeader {
    version: u8,
    flags: u8,
    num_records: u64,
    num_strings: u64,
    tracepoint_type: TracepointType,
    complexity_metric: ComplexityMetric,
    max_complexity: u64,
}

impl BinaryPafHeader {
    /// Create header with strategy and optional automatic encoding flags
    fn new(
        num_records: u64,
        num_strings: u64,
        strategy: CompressionStrategy,
        use_delta_first: bool,
        use_delta_second: bool,
        tracepoint_type: TracepointType,
        complexity_metric: ComplexityMetric,
        max_complexity: u64,
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
        }
    }

    /// Get compression strategy from flags
    fn strategy(&self) -> io::Result<CompressionStrategy> {
        CompressionStrategy::from_code(self.flags & 0x07)
    }

    /// Get delta encoding flag for first values (Automatic mode only)
    fn use_delta_first(&self) -> bool {
        (self.flags & 0x08) != 0
    }

    /// Get delta encoding flag for second values (Automatic mode only)
    fn use_delta_second(&self) -> bool {
        (self.flags & 0x10) != 0
    }

    fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(BINARY_MAGIC)?;
        writer.write_all(&[self.version, self.flags])?;
        write_varint(writer, self.num_records)?;
        write_varint(writer, self.num_strings)?;
        writer.write_all(&[self.tracepoint_type.to_u8()])?;
        writer.write_all(&[complexity_metric_to_u8(&self.complexity_metric)])?;
        write_varint(writer, self.max_complexity)?;
        Ok(())
    }

    fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
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

        Ok(Self {
            version,
            flags,
            num_records,
            num_strings,
            tracepoint_type,
            complexity_metric,
            max_complexity,
        })
    }
}

/// String table for deduplicating sequence names
#[derive(Debug, Default, Clone)]
pub struct StringTable {
    strings: Vec<String>,
    lengths: Vec<u64>,
    index: HashMap<String, u64>,
}

impl StringTable {
    pub fn new() -> Self {
        Self::default()
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

    fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        write_varint(writer, self.strings.len() as u64)?;
        for (s, &len) in self.strings.iter().zip(self.lengths.iter()) {
            write_varint(writer, s.len() as u64)?;
            writer.write_all(s.as_bytes())?;
            write_varint(writer, len)?;
        }
        Ok(())
    }

    fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
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

#[derive(Debug, Clone)]
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
    fn from(mr: &MixedRepresentation) -> Self {
        match mr {
            MixedRepresentation::Tracepoint(a, b) => Self::Tracepoint(*a as u64, *b as u64),
            MixedRepresentation::CigarOp(len, op) => Self::CigarOp(*len as u64, *op),
        }
    }
}

#[derive(Debug)]
pub struct Tag {
    pub key: [u8; 2],
    pub tag_type: u8,
    pub value: TagValue,
}

#[derive(Debug)]
pub enum TagValue {
    Int(i64),
    Float(f32),
    String(String),
}

/// Delta encode positions
#[inline]
fn delta_encode(values: &[u64]) -> Vec<i64> {
    let mut deltas = Vec::with_capacity(values.len());
    if values.is_empty() {
        return deltas;
    }
    deltas.push(values[0] as i64);
    for i in 1..values.len() {
        deltas.push(values[i] as i64 - values[i - 1] as i64);
    }
    deltas
}

/// Delta decode back to positions
#[inline]
fn delta_decode(deltas: &[i64]) -> Vec<u64> {
    let mut values = Vec::with_capacity(deltas.len());
    if deltas.is_empty() {
        return values;
    }
    values.push(deltas[0] as u64);
    for i in 1..deltas.len() {
        values.push((values[i - 1] as i64 + deltas[i]) as u64);
    }
    values
}

/// Lightweight heuristic analysis - samples records and uses statistics
/// Returns (use_delta_first, use_delta_second)
fn analyze_smart_light_compression(records: &[AlignmentRecord]) -> (bool, bool) {
    const SAMPLE_SIZE: usize = 1000;

    let mut all_first_vals = Vec::new();
    let mut all_second_vals = Vec::new();

    // Sample first N records (or all if fewer)
    let sample_count = records.len().min(SAMPLE_SIZE);
    for record in records.iter().take(sample_count) {
        if let TracepointData::Standard(tps) | TracepointData::Fastga(tps) = &record.tracepoints {
            for (first, second) in tps {
                all_first_vals.push(*first);
                all_second_vals.push(*second);
            }
        }
    }

    if all_first_vals.is_empty() {
        return (false, false);
    }

    // Analyze first_values
    let use_delta_first = should_use_delta(&all_first_vals);
    let use_delta_second = should_use_delta(&all_second_vals);

    info!(
        "Automatic analysis: sampled {} records, {} tracepoints - first_values: use_delta={}, second_values: use_delta={}",
        sample_count,
        all_first_vals.len(),
        use_delta_first,
        use_delta_second
    );

    (use_delta_first, use_delta_second)
}

/// Heuristic to decide if delta encoding is beneficial
fn should_use_delta(values: &[u64]) -> bool {
    if values.len() < 2 {
        return false;
    }

    // Check monotonicity and delta statistics
    let mut monotonic = true;
    let mut total_delta: u64 = 0;
    let mut max_delta: i64 = 0;
    let mut negative_count = 0;

    for i in 1..values.len() {
        let delta = values[i] as i64 - values[i - 1] as i64;

        if delta < 0 {
            monotonic = false;
            negative_count += 1;
        }

        total_delta += delta.abs() as u64;
        max_delta = max_delta.max(delta.abs());
    }

    // Average delta magnitude
    let avg_delta = total_delta / (values.len() - 1) as u64;

    // Average raw value
    let avg_value = values.iter().sum::<u64>() / values.len() as u64;

    // Heuristic: Use delta if:
    // 1. Values are mostly monotonic (< 10% negative deltas)
    // 2. Average delta is significantly smaller than average value (< 50%)
    // 3. Max delta is not too large (< 10x average)

    let negative_ratio = negative_count as f64 / (values.len() - 1) as f64;
    let delta_ratio = avg_delta as f64 / avg_value.max(1) as f64;
    let max_delta_ratio = max_delta as f64 / avg_delta.max(1) as f64;

    let use_delta = monotonic
        || (negative_ratio < 0.1 && delta_ratio < 0.5 && max_delta_ratio < 10.0);

    debug!(
        "Delta heuristic: mono={}, neg_ratio={:.2}, delta_ratio={:.2}, max_ratio={:.2} -> {}",
        monotonic, negative_ratio, delta_ratio, max_delta_ratio, use_delta
    );

    use_delta
}

impl AlignmentRecord {
    fn write<W: Write>(&self, writer: &mut W, use_delta: bool, strategy: CompressionStrategy) -> io::Result<()> {
        write_varint(writer, self.query_name_id)?;
        write_varint(writer, self.query_start)?;
        write_varint(writer, self.query_end)?;
        writer.write_all(&[self.strand as u8])?;
        write_varint(writer, self.target_name_id)?;
        write_varint(writer, self.target_start)?;
        write_varint(writer, self.target_end)?;
        write_varint(writer, self.residue_matches)?;
        write_varint(writer, self.alignment_block_len)?;
        writer.write_all(&[self.mapping_quality])?;
        if use_delta {
            self.write_tracepoints(writer, strategy)?;
        } else {
            self.write_tracepoints_raw(writer, strategy)?;
        }
        write_varint(writer, self.tags.len() as u64)?;
        for tag in &self.tags {
            tag.write(writer)?;
        }
        Ok(())
    }

    fn write_automatic<W: Write>(
        &self,
        writer: &mut W,
        use_delta_first: bool,
        use_delta_second: bool,
        strategy: CompressionStrategy,
    ) -> io::Result<()> {
        write_varint(writer, self.query_name_id)?;
        write_varint(writer, self.query_start)?;
        write_varint(writer, self.query_end)?;
        writer.write_all(&[self.strand as u8])?;
        write_varint(writer, self.target_name_id)?;
        write_varint(writer, self.target_start)?;
        write_varint(writer, self.target_end)?;
        write_varint(writer, self.residue_matches)?;
        write_varint(writer, self.alignment_block_len)?;
        writer.write_all(&[self.mapping_quality])?;
        self.write_tracepoints_automatic(writer, use_delta_first, use_delta_second, strategy)?;
        write_varint(writer, self.tags.len() as u64)?;
        for tag in &self.tags {
            tag.write(writer)?;
        }
        Ok(())
    }

    fn write_tracepoints<W: Write>(&self, writer: &mut W, strategy: CompressionStrategy) -> io::Result<()> {
        match &self.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                write_varint(writer, tps.len() as u64)?;
                if tps.is_empty() {
                    return Ok(());
                }
                let (first_vals, second_vals): (Vec<u64>, Vec<u64>) = tps.iter().copied().unzip();

                // Delta encoding selection based on tracepoint type:
                // - FastGA: num_differences are naturally small, use raw values
                // - Standard: query_bases are incremental, use delta encoding
                let use_delta = !matches!(self.tp_type, TracepointType::Fastga);
                let first_vals_encoded = if use_delta {
                    delta_encode(&first_vals)
                } else {
                    first_vals.iter().map(|&v| v as i64).collect()
                };

                let mut first_val_buf = Vec::with_capacity(first_vals_encoded.len() * 2);
                let mut second_val_buf = Vec::with_capacity(second_vals.len() * 2);

                for &val in &first_vals_encoded {
                    let zigzag = ((val << 1) ^ (val >> 63)) as u64;
                    write_varint(&mut first_val_buf, zigzag)?;
                }
                for &val in &second_vals {
                    write_varint(&mut second_val_buf, val)?;
                }

                let first_compressed = zstd::encode_all(&first_val_buf[..], strategy.zstd_level())?;
                let second_compressed = zstd::encode_all(&second_val_buf[..], strategy.zstd_level())?;

                write_varint(writer, first_compressed.len() as u64)?;
                writer.write_all(&first_compressed)?;
                write_varint(writer, second_compressed.len() as u64)?;
                writer.write_all(&second_compressed)?;
            }
            TracepointData::Variable(tps) => {
                write_varint(writer, tps.len() as u64)?;
                for (a, b_opt) in tps {
                    write_varint(writer, *a)?;
                    if let Some(b) = b_opt {
                        writer.write_all(&[1])?;
                        write_varint(writer, *b)?;
                    } else {
                        writer.write_all(&[0])?;
                    }
                }
            }
            TracepointData::Mixed(items) => {
                write_varint(writer, items.len() as u64)?;
                for item in items {
                    match item {
                        MixedTracepointItem::Tracepoint(a, b) => {
                            writer.write_all(&[0])?;
                            write_varint(writer, *a)?;
                            write_varint(writer, *b)?;
                        }
                        MixedTracepointItem::CigarOp(len, op) => {
                            writer.write_all(&[1])?;
                            write_varint(writer, *len)?;
                            writer.write_all(&[*op as u8])?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Write tracepoints WITHOUT delta encoding (for raw strategies)
    fn write_tracepoints_raw<W: Write>(&self, writer: &mut W, strategy: CompressionStrategy) -> io::Result<()> {
        match &self.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                write_varint(writer, tps.len() as u64)?;
                if tps.is_empty() {
                    return Ok(());
                }
                let (first_vals, second_vals): (Vec<u64>, Vec<u64>) = tps.iter().copied().unzip();

                // No delta encoding - use raw values for all types
                let mut first_val_buf = Vec::with_capacity(first_vals.len() * 2);
                let mut second_val_buf = Vec::with_capacity(second_vals.len() * 2);

                for &val in &first_vals {
                    write_varint(&mut first_val_buf, val)?;
                }
                for &val in &second_vals {
                    write_varint(&mut second_val_buf, val)?;
                }

                let first_compressed = zstd::encode_all(&first_val_buf[..], strategy.zstd_level())?;
                let second_compressed = zstd::encode_all(&second_val_buf[..], strategy.zstd_level())?;

                write_varint(writer, first_compressed.len() as u64)?;
                writer.write_all(&first_compressed)?;
                write_varint(writer, second_compressed.len() as u64)?;
                writer.write_all(&second_compressed)?;
            }
            TracepointData::Variable(tps) => {
                write_varint(writer, tps.len() as u64)?;
                for (a, b_opt) in tps {
                    write_varint(writer, *a)?;
                    if let Some(b) = b_opt {
                        writer.write_all(&[1])?;
                        write_varint(writer, *b)?;
                    } else {
                        writer.write_all(&[0])?;
                    }
                }
            }
            TracepointData::Mixed(items) => {
                write_varint(writer, items.len() as u64)?;
                for item in items {
                    match item {
                        MixedTracepointItem::Tracepoint(a, b) => {
                            writer.write_all(&[0])?;
                            write_varint(writer, *a)?;
                            write_varint(writer, *b)?;
                        }
                        MixedTracepointItem::CigarOp(len, op) => {
                            writer.write_all(&[1])?;
                            write_varint(writer, *len)?;
                            writer.write_all(&[*op as u8])?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Write tracepoints with automatic encoding (independent delta flags for first/second values)
    fn write_tracepoints_automatic<W: Write>(
        &self,
        writer: &mut W,
        use_delta_first: bool,
        use_delta_second: bool,
        strategy: CompressionStrategy,
    ) -> io::Result<()> {
        match &self.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                write_varint(writer, tps.len() as u64)?;
                if tps.is_empty() {
                    return Ok(());
                }
                let (first_vals, second_vals): (Vec<u64>, Vec<u64>) = tps.iter().copied().unzip();

                // Encode first_values (delta or raw)
                let mut first_val_buf = Vec::with_capacity(first_vals.len() * 2);
                if use_delta_first {
                    let first_vals_encoded = delta_encode(&first_vals);
                    for &val in &first_vals_encoded {
                        let zigzag = ((val << 1) ^ (val >> 63)) as u64;
                        write_varint(&mut first_val_buf, zigzag)?;
                    }
                } else {
                    for &val in &first_vals {
                        write_varint(&mut first_val_buf, val)?;
                    }
                }

                // Encode second_values (delta or raw)
                let mut second_val_buf = Vec::with_capacity(second_vals.len() * 2);
                if use_delta_second {
                    let second_vals_encoded = delta_encode(&second_vals);
                    for &val in &second_vals_encoded {
                        let zigzag = ((val << 1) ^ (val >> 63)) as u64;
                        write_varint(&mut second_val_buf, zigzag)?;
                    }
                } else {
                    for &val in &second_vals {
                        write_varint(&mut second_val_buf, val)?;
                    }
                }

                let first_compressed = zstd::encode_all(&first_val_buf[..], strategy.zstd_level())?;
                let second_compressed = zstd::encode_all(&second_val_buf[..], strategy.zstd_level())?;

                write_varint(writer, first_compressed.len() as u64)?;
                writer.write_all(&first_compressed)?;
                write_varint(writer, second_compressed.len() as u64)?;
                writer.write_all(&second_compressed)?;
            }
            TracepointData::Variable(tps) => {
                write_varint(writer, tps.len() as u64)?;
                for (a, b_opt) in tps {
                    write_varint(writer, *a)?;
                    if let Some(b) = b_opt {
                        writer.write_all(&[1])?;
                        write_varint(writer, *b)?;
                    } else {
                        writer.write_all(&[0])?;
                    }
                }
            }
            TracepointData::Mixed(items) => {
                write_varint(writer, items.len() as u64)?;
                for item in items {
                    match item {
                        MixedTracepointItem::Tracepoint(a, b) => {
                            writer.write_all(&[0])?;
                            write_varint(writer, *a)?;
                            write_varint(writer, *b)?;
                        }
                        MixedTracepointItem::CigarOp(len, op) => {
                            writer.write_all(&[1])?;
                            write_varint(writer, *len)?;
                            writer.write_all(&[*op as u8])?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn read<R: Read>(
        reader: &mut R,
        tp_type: TracepointType,
        complexity_metric: ComplexityMetric,
        max_complexity: u64,
    ) -> io::Result<Self> {
        let query_name_id = read_varint(reader)?;
        let query_start = read_varint(reader)?;
        let query_end = read_varint(reader)?;
        let mut strand_buf = [0u8; 1];
        reader.read_exact(&mut strand_buf)?;
        let strand = strand_buf[0] as char;
        let target_name_id = read_varint(reader)?;
        let target_start = read_varint(reader)?;
        let target_end = read_varint(reader)?;
        let residue_matches = read_varint(reader)?;
        let alignment_block_len = read_varint(reader)?;
        let mut mapq_buf = [0u8; 1];
        reader.read_exact(&mut mapq_buf)?;
        let mapping_quality = mapq_buf[0];
        let tracepoints = Self::read_tracepoints(reader, tp_type)?;
        let num_tags = read_varint(reader)? as usize;
        let mut tags = Vec::with_capacity(num_tags);
        for _ in 0..num_tags {
            tags.push(Tag::read(reader)?);
        }
        Ok(Self {
            query_name_id,
            query_start,
            query_end,
            strand,
            target_name_id,
            target_start,
            target_end,
            residue_matches,
            alignment_block_len,
            mapping_quality,
            tp_type,
            complexity_metric,
            max_complexity,
            tracepoints,
            tags,
        })
    }

    fn read_tracepoints<R: Read>(
        reader: &mut R,
        tp_type: TracepointType,
    ) -> io::Result<TracepointData> {
        let num_items = read_varint(reader)? as usize;
        match tp_type {
            TracepointType::Standard | TracepointType::Fastga => {
                if num_items == 0 {
                    return Ok(match tp_type {
                        TracepointType::Standard => TracepointData::Standard(Vec::new()),
                        _ => TracepointData::Fastga(Vec::new()),
                    });
                }
                let pos_len = read_varint(reader)? as usize;
                let mut pos_compressed = vec![0u8; pos_len];
                reader.read_exact(&mut pos_compressed)?;
                let score_len = read_varint(reader)? as usize;
                let mut score_compressed = vec![0u8; score_len];
                reader.read_exact(&mut score_compressed)?;

                let pos_buf = zstd::decode_all(&pos_compressed[..])?;
                let score_buf = zstd::decode_all(&score_compressed[..])?;

                let mut pos_reader = &pos_buf[..];
                let mut pos_values = Vec::with_capacity(num_items);
                for _ in 0..num_items {
                    let zigzag = read_varint(&mut pos_reader)?;
                    let val = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
                    pos_values.push(val);
                }

                // Delta encoding selection based on tracepoint type
                let positions: Vec<u64> = if matches!(tp_type, TracepointType::Fastga) {
                    // FastGA: raw values (no delta)
                    pos_values.iter().map(|&v| v as u64).collect()
                } else {
                    // Standard: delta-encoded
                    delta_decode(&pos_values)
                };

                let mut score_reader = &score_buf[..];
                let mut scores = Vec::with_capacity(num_items);
                for _ in 0..num_items {
                    scores.push(read_varint(&mut score_reader)?);
                }

                let tps: Vec<(u64, u64)> = positions.into_iter().zip(scores).collect();
                Ok(match tp_type {
                    TracepointType::Standard => TracepointData::Standard(tps),
                    _ => TracepointData::Fastga(tps),
                })
            }
            TracepointType::Variable => {
                let mut tps = Vec::with_capacity(num_items);
                for _ in 0..num_items {
                    let a = read_varint(reader)?;
                    let mut flag = [0u8; 1];
                    reader.read_exact(&mut flag)?;
                    let b_opt = if flag[0] == 1 {
                        Some(read_varint(reader)?)
                    } else {
                        None
                    };
                    tps.push((a, b_opt));
                }
                Ok(TracepointData::Variable(tps))
            }
            TracepointType::Mixed => {
                let mut items = Vec::with_capacity(num_items);
                for _ in 0..num_items {
                    let mut item_type = [0u8; 1];
                    reader.read_exact(&mut item_type)?;
                    match item_type[0] {
                        0 => {
                            let a = read_varint(reader)?;
                            let b = read_varint(reader)?;
                            items.push(MixedTracepointItem::Tracepoint(a, b));
                        }
                        1 => {
                            let len = read_varint(reader)?;
                            let mut op = [0u8; 1];
                            reader.read_exact(&mut op)?;
                            items.push(MixedTracepointItem::CigarOp(len, op[0] as char));
                        }
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "Invalid mixed item type",
                            ))
                        }
                    }
                }
                Ok(TracepointData::Mixed(items))
            }
        }
    }
}

impl Tag {
    fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
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

    fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
        let mut key = [0u8; 2];
        reader.read_exact(&mut key)?;
        let mut tag_type = [0u8; 1];
        reader.read_exact(&mut tag_type)?;
        let value = match tag_type[0] {
            b'i' => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                TagValue::Int(i64::from_le_bytes(buf))
            }
            b'f' => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                TagValue::Float(f32::from_le_bytes(buf))
            }
            b'Z' => {
                let len = read_varint(reader)? as usize;
                let mut buf = vec![0u8; len];
                reader.read_exact(&mut buf)?;
                let s = String::from_utf8(buf)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                TagValue::String(s)
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid tag type",
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

// ============================================================================
// PUBLIC API
// ============================================================================

/// Detect if file is binary PAF
pub fn is_binary_paf(path: &str) -> io::Result<bool> {
    if path == "-" {
        return Ok(false);
    }
    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    match file.read_exact(&mut magic) {
        Ok(()) => Ok(&magic == BINARY_MAGIC),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e),
    }
}

/// Open PAF file for reading, supporting plain text, gzip, and bgzip formats
///
/// Handles three input types:
/// - `-`: Read from stdin
/// - `.gz` or `.bgz` files: Decompress with gzip decoder
/// - Plain text: Read directly
fn open_paf_reader(input_path: &str) -> io::Result<Box<dyn BufRead>> {
    if input_path == "-" {
        Ok(Box::new(BufReader::new(io::stdin())))
    } else if input_path.ends_with(".gz") || input_path.ends_with(".bgz") {
        let file = File::open(input_path).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("Failed to open input file '{}': {}", input_path, e),
            )
        })?;
        let decoder = MultiGzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        let file = File::open(input_path).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("Failed to open input file '{}': {}", input_path, e),
            )
        })?;
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Encode PAF with CIGAR to binary with tracepoints
/// Encode CIGAR to tracepoints and write binary format
pub fn encode_cigar_to_binary(
    input_path: &str,
    output_path: &str,
    tp_type: &TracepointType,
    max_complexity: usize,
    complexity_metric: &ComplexityMetric,
    strategy: CompressionStrategy,
) -> io::Result<()> {
    info!("Encoding CIGAR with {} strategy...", strategy);

    let input = open_paf_reader(input_path)?;
    let mut string_table = StringTable::new();
    let mut records = Vec::new();

    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        match parse_paf_with_cigar(
            &line,
            &mut string_table,
            tp_type,
            max_complexity,
            complexity_metric,
        ) {
            Ok(record) => records.push(record),
            Err(e) => {
                error!("Line {}: {}", line_num + 1, e);
                return Err(e);
            }
        }
    }

    write_binary(output_path, &records, &string_table, strategy)?;

    info!(
        "Encoded {} records ({} unique names) with {} strategy",
        records.len(),
        string_table.len(),
        strategy
    );
    Ok(())
}

/// Convert binary PAF to text format
pub fn decompress_paf(input_path: &str, output_path: &str) -> io::Result<()> {
    info!("Decompressing {} to text format...", input_path);

    let input = File::open(input_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to open input file '{}': {}", input_path, e),
        )
    })?;
    let mut reader = BufReader::new(input);

    let header = BinaryPafHeader::read(&mut reader)?;

    if header.version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported format version: {}", header.version),
        ));
    }

    let strategy = header.strategy()?;
    info!(
        "Reading {} records ({} unique names) [{}]",
        header.num_records, header.num_strings, strategy
    );

    decompress_varint(reader, output_path, &header, strategy)
}

/// Decompress varint-encoded records (streaming with direct seek)
fn decompress_varint<R: Read>(
    mut reader: R,
    output_path: &str,
    header: &BinaryPafHeader,
    strategy: CompressionStrategy,
) -> io::Result<()> {
    // Read string table
    let string_table = StringTable::read(&mut reader)?;

    // Stream write records
    let output: Box<dyn Write> = if output_path == "-" {
        Box::new(io::stdout())
    } else {
        Box::new(File::create(output_path).map_err(|e| {
            io::Error::new(e.kind(), format!("Failed to create output file '{}': {}", output_path, e))
        })?)
    };
    let mut writer = BufWriter::new(output);

    match strategy {
        CompressionStrategy::Automatic(_) => {
            let use_delta_first = header.use_delta_first();
            let use_delta_second = header.use_delta_second();
            for _ in 0..header.num_records {
                let record = read_record_automatic(
                    &mut reader,
                    use_delta_first,
                    use_delta_second,
                    header.tracepoint_type,
                    header.complexity_metric,
                    header.max_complexity,
                )?;
                write_paf_line(&mut writer, &record, &string_table)?;
            }
        }
        CompressionStrategy::VarintZstd(_) => {
            for _ in 0..header.num_records {
                let record = read_record_varint(
                    &mut reader,
                    false,
                    header.tracepoint_type,
                    header.complexity_metric,
                    header.max_complexity,
                )?;
                write_paf_line(&mut writer, &record, &string_table)?;
            }
        }
        CompressionStrategy::DeltaVarintZstd(_) => {
            for _ in 0..header.num_records {
                let record = AlignmentRecord::read(
                    &mut reader,
                    header.tracepoint_type,
                    header.complexity_metric,
                    header.max_complexity,
                )?;
                write_paf_line(&mut writer, &record, &string_table)?;
            }
        }
    }
    writer.flush()?;

    info!("Decompressed {} records", header.num_records);
    Ok(())
}



// Read record for version 4 (VarintRaw) - always use raw values, no delta
fn read_record_varint<R: Read>(
    reader: &mut R,
    _use_delta: bool, // Parameter for consistency, always false for VarintRaw
    tp_type: TracepointType,
    complexity_metric: ComplexityMetric,
    max_complexity: u64,
) -> io::Result<AlignmentRecord> {
    // Read PAF fields (same as version 1)
    let query_name_id = read_varint(reader)?;
    let query_start = read_varint(reader)?;
    let query_end = read_varint(reader)?;
    let mut strand_buf = [0u8; 1];
    reader.read_exact(&mut strand_buf)?;
    let strand = strand_buf[0] as char;
    let target_name_id = read_varint(reader)?;
    let target_start = read_varint(reader)?;
    let target_end = read_varint(reader)?;
    let residue_matches = read_varint(reader)?;
    let alignment_block_len = read_varint(reader)?;
    let mut mapq_buf = [0u8; 1];
    reader.read_exact(&mut mapq_buf)?;
    let mapping_quality = mapq_buf[0];

    // Read tracepoints (raw, no delta)
    let tracepoints = read_tracepoints_raw(reader, tp_type)?;

    let num_tags = read_varint(reader)? as usize;
    let mut tags = Vec::with_capacity(num_tags);
    for _ in 0..num_tags {
        tags.push(Tag::read(reader)?);
    }

    Ok(AlignmentRecord {
        query_name_id,
        query_start,
        query_end,
        strand,
        target_name_id,
        target_start,
        target_end,
        residue_matches,
        alignment_block_len,
        mapping_quality,
        tp_type,
        complexity_metric,
        max_complexity,
        tracepoints,
        tags,
    })
}

// Read record for Automatic mode with per-field delta encoding
fn read_record_automatic<R: Read>(
    reader: &mut R,
    use_delta_first: bool,
    use_delta_second: bool,
    tp_type: TracepointType,
    complexity_metric: ComplexityMetric,
    max_complexity: u64,
) -> io::Result<AlignmentRecord> {
    let query_name_id = read_varint(reader)?;
    let query_start = read_varint(reader)?;
    let query_end = read_varint(reader)?;
    let mut strand_buf = [0u8; 1];
    reader.read_exact(&mut strand_buf)?;
    let strand = strand_buf[0] as char;
    let target_name_id = read_varint(reader)?;
    let target_start = read_varint(reader)?;
    let target_end = read_varint(reader)?;
    let residue_matches = read_varint(reader)?;
    let alignment_block_len = read_varint(reader)?;
    let mut mapq_buf = [0u8; 1];
    reader.read_exact(&mut mapq_buf)?;
    let mapping_quality = mapq_buf[0];

    // Read tracepoints with automatic encoding
    let tracepoints = read_tracepoints_automatic(reader, tp_type, use_delta_first, use_delta_second)?;

    let num_tags = read_varint(reader)? as usize;
    let mut tags = Vec::with_capacity(num_tags);
    for _ in 0..num_tags {
        tags.push(Tag::read(reader)?);
    }

    Ok(AlignmentRecord {
        query_name_id,
        query_start,
        query_end,
        strand,
        target_name_id,
        target_start,
        target_end,
        residue_matches,
        alignment_block_len,
        mapping_quality,
        tp_type,
        complexity_metric,
        max_complexity,
        tracepoints,
        tags,
    })
}

// Read tracepoints without delta encoding (version 4)
fn read_tracepoints_raw<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
) -> io::Result<TracepointData> {
    let num_items = read_varint(reader)? as usize;
    if num_items == 0 {
        return Ok(match tp_type {
            TracepointType::Standard => TracepointData::Standard(Vec::new()),
            _ => TracepointData::Fastga(Vec::new()),
        });
    }

    let pos_len = read_varint(reader)? as usize;
    let mut pos_compressed = vec![0u8; pos_len];
    reader.read_exact(&mut pos_compressed)?;
    let score_len = read_varint(reader)? as usize;
    let mut score_compressed = vec![0u8; score_len];
    reader.read_exact(&mut score_compressed)?;

    let pos_buf = zstd::decode_all(&pos_compressed[..])?;
    let score_buf = zstd::decode_all(&score_compressed[..])?;

    // Read raw varint values (no zigzag, no delta)
    let mut pos_reader = &pos_buf[..];
    let mut positions = Vec::with_capacity(num_items);
    for _ in 0..num_items {
        positions.push(read_varint(&mut pos_reader)?);
    }

    let mut score_reader = &score_buf[..];
    let mut scores = Vec::with_capacity(num_items);
    for _ in 0..num_items {
        scores.push(read_varint(&mut score_reader)?);
    }

    let tps: Vec<(u64, u64)> = positions.into_iter().zip(scores).collect();
    Ok(match tp_type {
        TracepointType::Standard => TracepointData::Standard(tps),
        _ => TracepointData::Fastga(tps),
    })
}

// Read tracepoints with automatic encoding (independent delta for first/second values)
fn read_tracepoints_automatic<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
    use_delta_first: bool,
    use_delta_second: bool,
) -> io::Result<TracepointData> {
    let num_items = read_varint(reader)? as usize;
    if num_items == 0 {
        return Ok(match tp_type {
            TracepointType::Standard => TracepointData::Standard(Vec::new()),
            _ => TracepointData::Fastga(Vec::new()),
        });
    }

    let first_len = read_varint(reader)? as usize;
    let mut first_compressed = vec![0u8; first_len];
    reader.read_exact(&mut first_compressed)?;
    let second_len = read_varint(reader)? as usize;
    let mut second_compressed = vec![0u8; second_len];
    reader.read_exact(&mut second_compressed)?;

    let first_buf = zstd::decode_all(&first_compressed[..])?;
    let second_buf = zstd::decode_all(&second_compressed[..])?;

    // Decode first values (delta or raw)
    let mut first_reader = &first_buf[..];
    let first_vals = if use_delta_first {
        // Read zigzag-encoded deltas
        let mut deltas = Vec::with_capacity(num_items);
        for _ in 0..num_items {
            let zigzag = read_varint(&mut first_reader)?;
            let val = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
            deltas.push(val);
        }
        delta_decode(&deltas)
    } else {
        // Read raw values
        let mut vals = Vec::with_capacity(num_items);
        for _ in 0..num_items {
            vals.push(read_varint(&mut first_reader)?);
        }
        vals
    };

    // Decode second values (delta or raw)
    let mut second_reader = &second_buf[..];
    let second_vals = if use_delta_second {
        // Read zigzag-encoded deltas
        let mut deltas = Vec::with_capacity(num_items);
        for _ in 0..num_items {
            let zigzag = read_varint(&mut second_reader)?;
            let val = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
            deltas.push(val);
        }
        delta_decode(&deltas)
    } else {
        // Read raw values
        let mut vals = Vec::with_capacity(num_items);
        for _ in 0..num_items {
            vals.push(read_varint(&mut second_reader)?);
        }
        vals
    };

    let tps: Vec<(u64, u64)> = first_vals.into_iter().zip(second_vals).collect();
    Ok(match tp_type {
        TracepointType::Standard => TracepointData::Standard(tps),
        _ => TracepointData::Fastga(tps),
    })
}


/// Compress PAF with tracepoints to binary format
///
/// Uses delta encoding + varint + zstd compression for optimal balance
/// of speed and compression ratio on genomic alignment data.
/// Compress PAF with tracepoints to binary format
pub fn compress_paf(input_path: &str, output_path: &str, strategy: CompressionStrategy) -> io::Result<()> {
    info!("Compressing PAF with {} strategy...", strategy);

    // Pass 1: Build string table + collect sample for analysis
    let mut string_table = StringTable::new();
    let mut sample = Vec::new();
    let mut record_count = 0u64;

    let input = open_paf_reader(input_path)?;
    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') { continue; }

        let record = parse_paf_with_tracepoints(&line, &mut string_table)
            .map_err(|e| { error!("Line {}: {}", line_num + 1, e); e })?;

        if sample.len() < 1000 { sample.push(record); }
        record_count += 1;
    }

    // Extract tracepoint metadata from first record (assuming homogeneous file)
    if sample.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "No records found in PAF file"));
    }
    let tp_type = sample[0].tp_type;
    let complexity_metric = sample[0].complexity_metric;
    let max_complexity = sample[0].max_complexity;

    // Analyze sample for Automatic mode
    let (use_delta_first, use_delta_second) = match strategy {
        CompressionStrategy::Automatic(_) => analyze_smart_light_compression(&sample),
        _ => (false, false),
    };

    // Pass 2: Stream write - Header → StringTable → Records
    let mut output = File::create(output_path).map_err(|e| {
        io::Error::new(e.kind(), format!("Failed to create output file '{}': {}", output_path, e))
    })?;

    // Write header
    let header = BinaryPafHeader::new(record_count, string_table.len() as u64, strategy, use_delta_first, use_delta_second, tp_type, complexity_metric, max_complexity);
    header.write(&mut output)?;

    // Write string table
    string_table.write(&mut output)?;

    // Write records
    let mut writer = BufWriter::new(&mut output);
    let input = open_paf_reader(input_path)?;
    let mut temp_table = string_table.clone();
    for line_result in input.lines() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') { continue; }

        let record = parse_paf_with_tracepoints(&line, &mut temp_table)?;
        match strategy {
            CompressionStrategy::Automatic(_) => record.write_automatic(&mut writer, use_delta_first, use_delta_second, strategy)?,
            CompressionStrategy::VarintZstd(_) => record.write(&mut writer, false, strategy)?,
            CompressionStrategy::DeltaVarintZstd(_) => record.write(&mut writer, true, strategy)?,
        }
    }
    writer.flush()?;

    info!("Compressed {} records ({} unique names) with {} strategy", record_count, string_table.len(), strategy);
    Ok(())
}

/// Write records to binary PAF format
fn write_binary(
    output_path: &str,
    records: &[AlignmentRecord],
    string_table: &StringTable,
    strategy: CompressionStrategy,
) -> io::Result<()> {
    let mut output = File::create(output_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{}': {}", output_path, e),
        )
    })?;

    // Extract tracepoint metadata from first record (assuming homogeneous file)
    if records.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "No records provided"));
    }
    let tp_type = records[0].tp_type;
    let complexity_metric = records[0].complexity_metric;
    let max_complexity = records[0].max_complexity;

    // Analyze data for Automatic mode
    let (use_delta_first, use_delta_second) = match strategy {
        CompressionStrategy::Automatic(_) => analyze_smart_light_compression(records),
        _ => (false, false),
    };

    // Write header
    let header = BinaryPafHeader::new(
        records.len() as u64,
        string_table.len() as u64,
        strategy,
        use_delta_first,
        use_delta_second,
        tp_type,
        complexity_metric,
        max_complexity,
    );
    header.write(&mut output)?;

    // Write string table
    string_table.write(&mut output)?;

    // Write records
    let mut writer = BufWriter::new(&mut output);
    match strategy {
        CompressionStrategy::Automatic(_) => {
            for record in records {
                record.write_automatic(&mut writer, use_delta_first, use_delta_second, strategy)?;
            }
        }
        CompressionStrategy::VarintZstd(_) => {
            for record in records {
                record.write(&mut writer, false, strategy)?;
            }
        }
        CompressionStrategy::DeltaVarintZstd(_) => {
            for record in records {
                record.write(&mut writer, true, strategy)?;
            }
        }
    }
    writer.flush()?;

    Ok(())
}

fn write_paf_output(
    output_path: &str,
    records: &[AlignmentRecord],
    string_table: &StringTable,
) -> io::Result<()> {
    let output: Box<dyn Write> = if output_path == "-" {
        Box::new(io::stdout())
    } else {
        Box::new(File::create(output_path).map_err(|e| {
            io::Error::new(
                e.kind(),
                format!("Failed to create output file '{}': {}", output_path, e),
            )
        })?)
    };
    let mut writer = BufWriter::new(output);

    for record in records {
        write_paf_line(&mut writer, record, string_table)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_paf_line<W: Write>(
    writer: &mut W,
    record: &AlignmentRecord,
    string_table: &StringTable,
) -> io::Result<()> {
    let query_name = string_table.get(record.query_name_id).unwrap();
    let target_name = string_table.get(record.target_name_id).unwrap();
    let query_len = string_table.get_length(record.query_name_id).unwrap();
    let target_len = string_table.get_length(record.target_name_id).unwrap();

    write!(
        writer,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        query_name,
        query_len,
        record.query_start,
        record.query_end,
        record.strand,
        target_name,
        target_len,
        record.target_start,
        record.target_end,
        record.residue_matches,
        record.alignment_block_len,
        record.mapping_quality
    )?;

    for tag in &record.tags {
        write!(writer, "\t{}", format_tag(tag))?;
    }
    write!(writer, "\ttp:Z:{}", format_tracepoints(&record.tracepoints))?;
    writeln!(writer)?;
    Ok(())
}

fn format_tracepoints(tps: &TracepointData) -> String {
    match tps {
        TracepointData::Standard(items) | TracepointData::Fastga(items) => items
            .iter()
            .map(|(a, b)| format!("{},{}", a, b))
            .collect::<Vec<_>>()
            .join(";"),
        TracepointData::Variable(items) => items
            .iter()
            .map(|(a, b_opt)| match b_opt {
                Some(b) => format!("{},{}", a, b),
                None => a.to_string(),
            })
            .collect::<Vec<_>>()
            .join(";"),
        TracepointData::Mixed(items) => items
            .iter()
            .map(|item| match item {
                MixedTracepointItem::Tracepoint(a, b) => format!("{},{}", a, b),
                MixedTracepointItem::CigarOp(len, op) => format!("{}{}", len, op),
            })
            .collect::<Vec<_>>()
            .join(";"),
    }
}

fn parse_tag(field: &str) -> Option<Tag> {
    let parts: Vec<&str> = field.splitn(3, ':').collect();
    if parts.len() != 3 {
        return None;
    }
    let key = [parts[0].as_bytes()[0], parts[0].as_bytes()[1]];
    let tag_type = parts[1].as_bytes()[0];
    let value = match tag_type {
        b'i' => parts[2].parse::<i64>().ok().map(TagValue::Int)?,
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

fn format_tag(tag: &Tag) -> String {
    let key = String::from_utf8_lossy(&tag.key);
    match &tag.value {
        TagValue::Int(v) => format!("{}:i:{}", key, v),
        TagValue::Float(v) => format!("{}:f:{}", key, v),
        TagValue::String(s) => format!("{}:Z:{}", key, s),
    }
}

fn parse_usize(s: &str, field: &str) -> io::Result<u64> {
    s.parse().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid {}: {}", field, s),
        )
    })
}

fn parse_u8(s: &str, field: &str) -> io::Result<u8> {
    s.parse().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid {}: {}", field, s),
        )
    })
}

/// Convert ComplexityMetric to u8 for serialization
#[inline]
fn complexity_metric_to_u8(metric: &ComplexityMetric) -> u8 {
    match metric {
        ComplexityMetric::EditDistance => 0,
        ComplexityMetric::DiagonalDistance => 1,
    }
}

/// Convert u8 to ComplexityMetric for deserialization
#[inline]
fn complexity_metric_from_u8(byte: u8) -> io::Result<ComplexityMetric> {
    match byte {
        0 => Ok(ComplexityMetric::EditDistance),
        1 => Ok(ComplexityMetric::DiagonalDistance),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid complexity metric",
        )),
    }
}

fn parse_paf_with_cigar(
    line: &str,
    string_table: &mut StringTable,
    tp_type: &TracepointType,
    max_complexity: usize,
    complexity_metric: &ComplexityMetric,
) -> io::Result<AlignmentRecord> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 12 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PAF line has fewer than 12 fields",
        ));
    }

    let query_len = parse_usize(fields[1], "query_len")?;
    let query_name_id = string_table.intern(fields[0], query_len);
    let query_start = parse_usize(fields[2], "query_start")?;
    let query_end = parse_usize(fields[3], "query_end")?;
    let strand = fields[4].chars().next().unwrap_or('+');
    let target_len = parse_usize(fields[6], "target_len")?;
    let target_name_id = string_table.intern(fields[5], target_len);
    let target_start = parse_usize(fields[7], "target_start")?;
    let target_end = parse_usize(fields[8], "target_end")?;
    let residue_matches = parse_usize(fields[9], "residue_matches")?;
    let alignment_block_len = parse_usize(fields[10], "alignment_block_len")?;
    let mapping_quality = parse_u8(fields[11], "mapping_quality")?;

    let mut cigar = None;
    let mut tags = Vec::new();

    for field in &fields[12..] {
        if let Some(stripped) = field.strip_prefix("cg:Z:") {
            cigar = Some(stripped);
        } else if !field.starts_with("tp:Z:") {
            if let Some(tag) = parse_tag(field) {
                tags.push(tag);
            }
        }
    }

    let cigar =
        cigar.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing cg:Z: tag"))?;

    let tracepoints = match tp_type {
        TracepointType::Standard => {
            let tps = cigar_to_tracepoints(cigar, max_complexity, *complexity_metric);
            TracepointData::Standard(tps.into_iter().map(|(a, b)| (a as u64, b as u64)).collect())
        }
        TracepointType::Mixed => {
            let tps = cigar_to_mixed_tracepoints(cigar, max_complexity, *complexity_metric);
            TracepointData::Mixed(tps.iter().map(MixedTracepointItem::from).collect())
        }
        TracepointType::Variable => {
            let tps = cigar_to_variable_tracepoints(cigar, max_complexity, *complexity_metric);
            TracepointData::Variable(
                tps.into_iter()
                    .map(|(a, b_opt)| (a as u64, b_opt.map(|b| b as u64)))
                    .collect(),
            )
        }
        TracepointType::Fastga => {
            let complement = strand == '-';
            let segments = cigar_to_tracepoints_fastga(
                cigar,
                max_complexity,
                query_start as usize,
                query_end as usize,
                query_len as usize,
                target_start as usize,
                target_end as usize,
                target_len as usize,
                complement,
            );
            if let Some((tps, _coords)) = segments.first() {
                TracepointData::Fastga(tps.iter().map(|(a, b)| (*a as u64, *b as u64)).collect())
            } else {
                TracepointData::Fastga(Vec::new())
            }
        }
    };

    let tp_type_enum = *tp_type;

    Ok(AlignmentRecord {
        query_name_id,
        query_start,
        query_end,
        strand,
        target_name_id,
        target_start,
        target_end,
        residue_matches,
        alignment_block_len,
        mapping_quality,
        tp_type: tp_type_enum,
        complexity_metric: *complexity_metric,
        max_complexity: max_complexity as u64,
        tracepoints,
        tags,
    })
}

fn parse_paf_with_tracepoints(
    line: &str,
    string_table: &mut StringTable,
) -> io::Result<AlignmentRecord> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 12 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PAF line has fewer than 12 fields",
        ));
    }

    let query_len = parse_usize(fields[1], "query_len")?;
    let query_name_id = string_table.intern(fields[0], query_len);
    let query_start = parse_usize(fields[2], "query_start")?;
    let query_end = parse_usize(fields[3], "query_end")?;
    let strand = fields[4].chars().next().unwrap_or('+');
    let target_len = parse_usize(fields[6], "target_len")?;
    let target_name_id = string_table.intern(fields[5], target_len);
    let target_start = parse_usize(fields[7], "target_start")?;
    let target_end = parse_usize(fields[8], "target_end")?;
    let residue_matches = parse_usize(fields[9], "residue_matches")?;
    let alignment_block_len = parse_usize(fields[10], "alignment_block_len")?;
    let mapping_quality = parse_u8(fields[11], "mapping_quality")?;

    let mut tp_str = None;
    let mut tags = Vec::new();

    for field in &fields[12..] {
        if let Some(stripped) = field.strip_prefix("tp:Z:") {
            tp_str = Some(stripped);
        } else if let Some(tag) = parse_tag(field) {
            tags.push(tag);
        }
    }

    let tp_str =
        tp_str.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing tp:Z: tag"))?;
    let (tracepoints, tp_type) = parse_tracepoints_auto(tp_str)?;

    Ok(AlignmentRecord {
        query_name_id,
        query_start,
        query_end,
        strand,
        target_name_id,
        target_start,
        target_end,
        residue_matches,
        alignment_block_len,
        mapping_quality,
        tp_type,
        complexity_metric: ComplexityMetric::EditDistance,
        max_complexity: 100,
        tracepoints,
        tags,
    })
}

fn parse_tracepoints_auto(tp_str: &str) -> io::Result<(TracepointData, TracepointType)> {
    if tp_str.is_empty() {
        return Ok((
            TracepointData::Standard(Vec::new()),
            TracepointType::Standard,
        ));
    }

    let items: Vec<&str> = tp_str.split(';').collect();
    let has_cigar = items.iter().any(|s| {
        s.chars()
            .last()
            .map(|c| matches!(c, 'M' | 'D' | 'I' | 'X' | '='))
            .unwrap_or(false)
    });

    if has_cigar {
        let mut mixed = Vec::new();
        for item in items {
            if item
                .chars()
                .last()
                .map(|c| matches!(c, 'M' | 'D' | 'I' | 'X' | '='))
                .unwrap_or(false)
            {
                let op = item.chars().last().unwrap();
                let len_str = &item[..item.len() - 1];
                let len: u64 = len_str.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid CIGAR length")
                })?;
                mixed.push(MixedTracepointItem::CigarOp(len, op));
            } else {
                let parts: Vec<&str> = item.split(',').collect();
                if parts.len() == 2 {
                    let a: u64 = parts[0].parse().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid tracepoint value")
                    })?;
                    let b: u64 = parts[1].parse().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid tracepoint value")
                    })?;
                    mixed.push(MixedTracepointItem::Tracepoint(a, b));
                }
            }
        }
        Ok((TracepointData::Mixed(mixed), TracepointType::Mixed))
    } else {
        let has_single_values = items.iter().any(|s| !s.contains(','));
        if has_single_values {
            let mut variable = Vec::new();
            for item in items {
                if item.contains(',') {
                    let parts: Vec<&str> = item.split(',').collect();
                    if parts.len() == 2 {
                        let a: u64 = parts[0].parse().map_err(|_| {
                            io::Error::new(io::ErrorKind::InvalidData, "Invalid tracepoint value")
                        })?;
                        let b: u64 = parts[1].parse().map_err(|_| {
                            io::Error::new(io::ErrorKind::InvalidData, "Invalid tracepoint value")
                        })?;
                        variable.push((a, Some(b)));
                    }
                } else {
                    let a: u64 = item.parse().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid tracepoint value")
                    })?;
                    variable.push((a, None));
                }
            }
            Ok((TracepointData::Variable(variable), TracepointType::Variable))
        } else {
            let mut pairs = Vec::new();
            for item in items {
                let parts: Vec<&str> = item.split(',').collect();
                if parts.len() == 2 {
                    let a: u64 = parts[0].parse().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid tracepoint value")
                    })?;
                    let b: u64 = parts[1].parse().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid tracepoint value")
                    })?;
                    pairs.push((a, b));
                }
            }
            Ok((TracepointData::Standard(pairs), TracepointType::Standard))
        }
    }
}

// ============================================================================
// SEEKABLE READER WITH INDEX
// ============================================================================

/// Index for O(1) random access to records in a BPAF file
#[derive(Debug)]
pub struct BpafIndex {
    /// File offset for each record (byte position in .bpaf file)
    offsets: Vec<u64>,
}

impl BpafIndex {
    const INDEX_MAGIC: &'static [u8; 4] = b"BPAI";

    /// Save index to .bpaf.idx file
    pub fn save(&self, idx_path: &str) -> io::Result<()> {
        let mut file = File::create(idx_path)?;

        // Write magic and version
        file.write_all(Self::INDEX_MAGIC)?;
        file.write_all(&[1u8])?; // Version 1

        // Write number of offsets
        write_varint(&mut file, self.offsets.len() as u64)?;

        // Write all offsets
        for &offset in &self.offsets {
            write_varint(&mut file, offset)?;
        }

        Ok(())
    }

    /// Load index from .bpaf.idx file
    pub fn load(idx_path: &str) -> io::Result<Self> {
        let file = File::open(idx_path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != Self::INDEX_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid index magic",
            ));
        }

        // Read version
        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported index version: {}", version[0]),
            ));
        }

        // Read number of offsets
        let num_offsets = read_varint(&mut reader)? as usize;

        // Read all offsets
        let mut offsets = Vec::with_capacity(num_offsets);
        for _ in 0..num_offsets {
            offsets.push(read_varint(&mut reader)?);
        }

        Ok(Self { offsets })
    }

    /// Get number of records in index
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

/// Build index by scanning through BPAF file (for existing files without .idx)
pub fn build_index(bpaf_path: &str) -> io::Result<BpafIndex> {
    info!("Building index for {}", bpaf_path);

    let file = File::open(bpaf_path)?;
    let mut reader = BufReader::with_capacity(131072, file); // 128KB buffer)

    let header = BinaryPafHeader::read(&mut reader)?;
    StringTable::read(&mut reader)?; // Read string table to advance past it

    let mut offsets = Vec::with_capacity(header.num_records as usize);
    for _ in 0..header.num_records {
        offsets.push(reader.stream_position()?);
        skip_record(&mut reader, false)?;
    }

    info!("Index built: {} records", offsets.len());
    Ok(BpafIndex { offsets })
}

/// Skip a record without parsing (for building index)
#[inline]
fn skip_record<R: Read + Seek>(reader: &mut R, _is_adaptive: bool) -> io::Result<()> {
    // Skip core PAF fields (varints)
    read_varint(reader)?; // query_name_id
    read_varint(reader)?; // query_start
    read_varint(reader)?; // query_end
    reader.seek(SeekFrom::Current(1))?; // strand
    read_varint(reader)?; // target_name_id
    read_varint(reader)?; // target_start
    read_varint(reader)?; // target_end
    read_varint(reader)?; // residue_matches
    read_varint(reader)?; // alignment_block_len
    reader.seek(SeekFrom::Current(1))?; // mapping_quality

    // Skip tracepoints
    let num_items = read_varint(reader)? as usize;
    if num_items > 0 {
        // Skip compressed position data
        let pos_len = read_varint(reader)? as usize;
        reader.seek(SeekFrom::Current(pos_len as i64))?;

        // Skip compressed score data
        let score_len = read_varint(reader)? as usize;
        reader.seek(SeekFrom::Current(score_len as i64))?;
    }

    // Skip tags
    let num_tags = read_varint(reader)? as usize;
    for _ in 0..num_tags {
        reader.seek(SeekFrom::Current(2))?; // key
        let mut tag_type = [0u8; 1];
        reader.read_exact(&mut tag_type)?;
        match tag_type[0] {
            b'i' => reader.seek(SeekFrom::Current(8))?,
            b'f' => reader.seek(SeekFrom::Current(4))?,
            b'Z' => {
                let len = read_varint(reader)? as usize;
                reader.seek(SeekFrom::Current(len as i64))?
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid tag type",
                ))
            }
        };
    }

    Ok(())
}

/// Seekable reader for BPAF files with O(1) random access
pub struct BpafReader {
    file: File,
    index: BpafIndex,
    header: BinaryPafHeader,
    string_table: StringTable,
}

impl BpafReader {
    /// Open a BPAF file with index (builds index if .bpaf.idx doesn't exist)
    pub fn open(bpaf_path: &str) -> io::Result<Self> {
        let idx_path = format!("{}.idx", bpaf_path);

        // Load or build index
        let index = if Path::new(&idx_path).exists() {
            info!("Loading existing index: {}", idx_path);
            BpafIndex::load(&idx_path)?
        } else {
            info!("No index found, building...");
            let idx = build_index(bpaf_path)?;
            idx.save(&idx_path)?;
            debug!("Index saved to {}", idx_path);
            idx
        };

        // Open file and read header
        let mut file = File::open(bpaf_path)?;
        let header = BinaryPafHeader::read(&mut file)?;


        // Lazy-load string table only when needed (expensive for large files)
        let string_table = StringTable::new();

        Ok(Self {
            file,
            index,
            header,
            string_table,
        })
    }

    /// Open a BPAF file without index (for offset-based access only)
    ///
    /// Use this if you have your own offset storage (like impg) and only need:
    /// - get_alignment_record_at_offset()
    /// - get_tracepoints_at_offset()
    ///
    /// This skips index loading entirely - much faster open time.
    pub fn open_without_index(bpaf_path: &str) -> io::Result<Self> {
        // Open file and read header
        let mut file = File::open(bpaf_path)?;
        let header = BinaryPafHeader::read(&mut file)?;


        // Empty index - not used for offset-based access
        let index = BpafIndex {
            offsets: Vec::new(),
        };
        let string_table = StringTable::new();

        Ok(Self {
            file,
            index,
            header,
            string_table,
        })
    }

    /// Load string table (call this if you need sequence names)
    pub fn load_string_table(&mut self) -> io::Result<()> {
        if !self.string_table.is_empty() {
            return Ok(()); // Already loaded
        }

        // Seek to string table (immediately after header at start of file)
        self.file.seek(SeekFrom::Start(0))?;
        BinaryPafHeader::read(&mut self.file)?; // Skip header
        self.string_table = StringTable::read(&mut self.file)?;
        Ok(())
    }

    /// Get number of records
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Check if reader is empty
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Get header information
    pub fn header(&self) -> &BinaryPafHeader {
        &self.header
    }

    /// Get string table (loads on first access if needed)
    pub fn string_table(&mut self) -> io::Result<&StringTable> {
        if self.string_table.is_empty() {
            self.load_string_table()?;
        }
        Ok(&self.string_table)
    }

    /// Get immutable reference to string table (must be loaded first with load_string_table)
    pub fn string_table_ref(&self) -> &StringTable {
        &self.string_table
    }

    /// Get full alignment record by ID - O(1) random access
    pub fn get_alignment_record(&mut self, record_id: u64) -> io::Result<AlignmentRecord> {
        if record_id >= self.index.len() as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Record ID {} out of range (max: {})",
                    record_id,
                    self.index.len() - 1
                ),
            ));
        }

        let offset = self.index.offsets[record_id as usize];
        self.get_alignment_record_at_offset(offset)
    }

    /// Get alignment record by file offset (for impg compatibility)
    pub fn get_alignment_record_at_offset(&mut self, offset: u64) -> io::Result<AlignmentRecord> {
        self.file.seek(SeekFrom::Start(offset))?;

        let strategy = self.header.strategy()?;
        match strategy {
            CompressionStrategy::Automatic(_) => {
                let use_delta_first = self.header.use_delta_first();
                let use_delta_second = self.header.use_delta_second();
                read_record_automatic(
                    &mut self.file,
                    use_delta_first,
                    use_delta_second,
                    self.header.tracepoint_type,
                    self.header.complexity_metric,
                    self.header.max_complexity,
                )
            }
            CompressionStrategy::VarintZstd(_) => read_record_varint(
                &mut self.file,
                false,
                self.header.tracepoint_type,
                self.header.complexity_metric,
                self.header.max_complexity,
            ),
            CompressionStrategy::DeltaVarintZstd(_) => AlignmentRecord::read(
                &mut self.file,
                self.header.tracepoint_type,
                self.header.complexity_metric,
                self.header.max_complexity,
            ),
        }
    }

    /// Get tracepoints only (optimized) - O(1) random access by record ID
    /// Returns: (tracepoints, tp_type, complexity_metric, max_complexity)
    ///
    /// Optimized for tracepoint-only access:
    /// - Default compression: skips unnecessary fields
    /// - Adaptive compression: reads full record but avoids field extraction overhead
    pub fn get_tracepoints(
        &mut self,
        record_id: u64,
    ) -> io::Result<(TracepointData, TracepointType, ComplexityMetric, u64)> {
        if record_id >= self.index.len() as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Record ID {} out of range (max: {})",
                    record_id,
                    self.index.len() - 1
                ),
            ));
        }

        let offset = self.index.offsets[record_id as usize];
        self.get_tracepoints_at_offset(offset)
    }

    /// Get tracepoints by file offset (for impg compatibility)
    /// Returns: (tracepoints, tp_type, complexity_metric, max_complexity)
    ///
    /// Use this if you have stored the actual file offsets - skips index lookup.
    pub fn get_tracepoints_at_offset(
        &mut self,
        offset: u64,
    ) -> io::Result<(TracepointData, TracepointType, ComplexityMetric, u64)> {
        self.file.seek(SeekFrom::Start(offset))?;

        // Skip core PAF fields
        read_varint(&mut self.file)?; // query_name_id - SKIP
        read_varint(&mut self.file)?; // query_start - SKIP
        read_varint(&mut self.file)?; // query_end - SKIP
        self.file.seek(SeekFrom::Current(1))?; // strand - SKIP
        read_varint(&mut self.file)?; // target_name_id - SKIP
        read_varint(&mut self.file)?; // target_start - SKIP
        read_varint(&mut self.file)?; // target_end - SKIP
        read_varint(&mut self.file)?; // residue_matches - SKIP
        read_varint(&mut self.file)?; // alignment_block_len - SKIP
        self.file.seek(SeekFrom::Current(1))?; // mapping_quality - SKIP
        // Get metadata from header
        let tp_type = self.header.tracepoint_type;
        let complexity_metric = self.header.complexity_metric;
        let max_complexity = self.header.max_complexity;

        // Read tracepoints - dispatch based on strategy
        let tracepoints = match self.header.strategy()? {
            CompressionStrategy::Automatic(_) => {
                read_tracepoints_automatic(
                    &mut self.file,
                    tp_type,
                    self.header.use_delta_first(),
                    self.header.use_delta_second(),
                )?
            }
            CompressionStrategy::VarintZstd(_) => {
                read_tracepoints_raw(&mut self.file, tp_type)?
            }
            CompressionStrategy::DeltaVarintZstd(_) => {
                AlignmentRecord::read_tracepoints(&mut self.file, tp_type)?
            }
        };

        Ok((tracepoints, tp_type, complexity_metric, max_complexity))
    }

    /// Iterator over all records (sequential access)
    pub fn iter_records(&mut self) -> RecordIterator<'_> {
        RecordIterator {
            reader: self,
            current_id: 0,
        }
    }
}

/// Iterator for sequential record access
pub struct RecordIterator<'a> {
    reader: &'a mut BpafReader,
    current_id: u64,
}

impl<'a> Iterator for RecordIterator<'a> {
    type Item = io::Result<AlignmentRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_id >= self.reader.len() as u64 {
            return None;
        }

        let result = self.reader.get_alignment_record(self.current_id);
        self.current_id += 1;
        Some(result)
    }
}
