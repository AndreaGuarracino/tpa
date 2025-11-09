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
const FLAG_COMPRESSED: u8 = 0x01;

/// Binary PAF header
#[derive(Debug)]
pub struct BinaryPafHeader {
    version: u8,
    flags: u8,
    num_records: u64,
    num_strings: u64,
}

impl BinaryPafHeader {
    fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(BINARY_MAGIC)?;
        writer.write_all(&[self.version, self.flags])?;
        write_varint(writer, self.num_records)?;
        write_varint(writer, self.num_strings)?;
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
        Ok(Self {
            version: ver_flags[0],
            flags: ver_flags[1],
            num_records: read_varint(reader)?,
            num_strings: read_varint(reader)?,
        })
    }
}

/// String table for deduplicating sequence names
#[derive(Debug, Default)]
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

impl AlignmentRecord {
    fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
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
        writer.write_all(&[self.tp_type.to_u8()])?;
        writer.write_all(&[complexity_metric_to_u8(&self.complexity_metric)])?;
        write_varint(writer, self.max_complexity)?;
        self.write_tracepoints(writer)?;
        write_varint(writer, self.tags.len() as u64)?;
        for tag in &self.tags {
            tag.write(writer)?;
        }
        Ok(())
    }

    fn write_tracepoints<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        match &self.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                write_varint(writer, tps.len() as u64)?;
                if tps.is_empty() {
                    return Ok(());
                }
                let (positions, scores): (Vec<u64>, Vec<u64>) = tps.iter().copied().unzip();
                let pos_deltas = delta_encode(&positions);

                let mut pos_buf = Vec::with_capacity(pos_deltas.len() * 2);
                let mut score_buf = Vec::with_capacity(scores.len() * 2);

                for &delta in &pos_deltas {
                    let zigzag = ((delta << 1) ^ (delta >> 63)) as u64;
                    write_varint(&mut pos_buf, zigzag)?;
                }
                for &score in &scores {
                    write_varint(&mut score_buf, score)?;
                }

                let pos_compressed = zstd::encode_all(&pos_buf[..], 3)?;
                let score_compressed = zstd::encode_all(&score_buf[..], 3)?;

                write_varint(writer, pos_compressed.len() as u64)?;
                writer.write_all(&pos_compressed)?;
                write_varint(writer, score_compressed.len() as u64)?;
                writer.write_all(&score_compressed)?;
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

    fn read<R: Read>(reader: &mut R) -> io::Result<Self> {
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
        let mut tp_type_buf = [0u8; 1];
        reader.read_exact(&mut tp_type_buf)?;
        let tp_type = TracepointType::from_u8(tp_type_buf[0])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut metric_buf = [0u8; 1];
        reader.read_exact(&mut metric_buf)?;
        let complexity_metric = complexity_metric_from_u8(metric_buf[0])?;
        let max_complexity = read_varint(reader)?;
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
                let mut pos_deltas = Vec::with_capacity(num_items);
                for _ in 0..num_items {
                    let zigzag = read_varint(&mut pos_reader)?;
                    let delta = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
                    pos_deltas.push(delta);
                }
                let positions = delta_decode(&pos_deltas);

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
        let file = File::open(input_path)?;
        let decoder = MultiGzDecoder::new(file);
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(File::open(input_path)?)))
    }
}

/// Encode PAF with CIGAR to binary with tracepoints
pub fn encode_cigar_to_binary(
    input_path: &str,
    output_path: &str,
    tp_type: &TracepointType,
    max_complexity: usize,
    complexity_metric: &ComplexityMetric,
) -> io::Result<()> {
    info!("Encoding CIGAR to tracepoints and writing binary format...");

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

    write_binary(output_path, &records, &string_table)?;
    debug!(
        "Encoded {} records ({} unique names)",
        records.len(),
        string_table.len()
    );
    Ok(())
}

/// Convert binary PAF to text format
pub fn decompress_paf(input_path: &str, output_path: &str) -> io::Result<()> {
    info!("Decompressing {} to text format...", input_path);

    let input = File::open(input_path)?;
    let mut reader = BufReader::new(input);

    let header = BinaryPafHeader::read(&mut reader)?;
    debug!(
        "Reading {} records ({} unique names)",
        header.num_records, header.num_strings
    );

    if header.version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported format version: {}", header.version),
        ));
    }

    decompress_default(reader, output_path, &header)
}

fn decompress_default<R: Read>(
    mut reader: R,
    output_path: &str,
    header: &BinaryPafHeader,
) -> io::Result<()> {
    let mut records = Vec::with_capacity(header.num_records as usize);
    for _ in 0..header.num_records {
        records.push(AlignmentRecord::read(&mut reader)?);
    }
    let string_table = StringTable::read(&mut reader)?;
    write_paf_output(output_path, &records, &string_table)?;
    debug!("Decompressed {} records", header.num_records);
    Ok(())
}

/// Compress PAF with tracepoints to binary format
///
/// Uses delta encoding + varint + zstd compression for optimal balance
/// of speed and compression ratio on genomic alignment data.
pub fn compress_paf(input_path: &str, output_path: &str) -> io::Result<()> {
    info!("Compressing PAF with tracepoints to binary format...");

    let input = open_paf_reader(input_path)?;
    let mut string_table = StringTable::new();
    let mut records = Vec::new();

    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        match parse_paf_with_tracepoints(&line, &mut string_table) {
            Ok(record) => records.push(record),
            Err(e) => {
                error!("Line {}: {}", line_num + 1, e);
                return Err(e);
            }
        }
    }

    write_binary(output_path, &records, &string_table)?;
    debug!(
        "Compressed {} records ({} unique names)",
        records.len(),
        string_table.len()
    );
    Ok(())
}

/// Write records to binary PAF format
///
/// Uses delta encoding, varint, and zstd compression for efficient storage.
fn write_binary(
    output_path: &str,
    records: &[AlignmentRecord],
    string_table: &StringTable,
) -> io::Result<()> {
    let output = File::create(output_path)?;
    let mut writer = BufWriter::new(output);

    let header = BinaryPafHeader {
        version: 1,
        flags: FLAG_COMPRESSED,
        num_records: records.len() as u64,
        num_strings: string_table.len() as u64,
    };

    header.write(&mut writer)?;
    for record in records {
        record.write(&mut writer)?;
    }
    string_table.write(&mut writer)?;
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
        Box::new(File::create(output_path)?)
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

/// Build index by scanning through BPAF file
pub fn build_index(bpaf_path: &str) -> io::Result<BpafIndex> {
    info!("Building index for {}", bpaf_path);

    let mut file = File::open(bpaf_path)?;
    let header = BinaryPafHeader::read(&mut file)?;

    let mut offsets = Vec::with_capacity(header.num_records as usize);

    // Record offset of each record
    for _ in 0..header.num_records {
        offsets.push(file.stream_position()?);
        skip_record(&mut file)?;
    }

    debug!("Index built: {} records", offsets.len());
    Ok(BpafIndex { offsets })
}

/// Skip a record without parsing (for building index)
fn skip_record<R: Read + Seek>(reader: &mut R) -> io::Result<()> {
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
    reader.seek(SeekFrom::Current(1))?; // tp_type
    reader.seek(SeekFrom::Current(1))?; // complexity_metric
    read_varint(reader)?; // max_complexity

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

        // Seek to string table (at end of file, after all records)
        if !self.index.offsets.is_empty() {
            self.file.seek(SeekFrom::Start(
                self.index.offsets[self.index.offsets.len() - 1],
            ))?;
            skip_record(&mut self.file)?;
        }

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
        AlignmentRecord::read(&mut self.file)
    }

    /// Get tracepoints only (optimized) - O(1) random access by record ID
    /// Returns: (tracepoints, tp_type, complexity_metric, max_complexity)
    ///
    /// Optimized for tracepoint-only access - skips unnecessary fields
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

        // Skip fields we don't need
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

        // Read only what we need
        let mut tp_type_buf = [0u8; 1];
        self.file.read_exact(&mut tp_type_buf)?;
        let tp_type = TracepointType::from_u8(tp_type_buf[0])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut metric_buf = [0u8; 1];
        self.file.read_exact(&mut metric_buf)?;
        let complexity_metric = complexity_metric_from_u8(metric_buf[0])?;

        let max_complexity = read_varint(&mut self.file)?;

        // Read tracepoints
        let tracepoints = AlignmentRecord::read_tracepoints(&mut self.file, tp_type)?;

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
