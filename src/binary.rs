//! Binary I/O operations for BPAF format

use lib_tracepoints::{ComplexityMetric, TracepointType};
use log::{debug, info};
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::format::*;
use crate::utils::*;

// ============================================================================
// CONSTANTS
// ============================================================================

pub const BINARY_MAGIC: &[u8; 4] = b"BPAF";

// ============================================================================
// COMPLEXITY METRIC SERIALIZATION
// ============================================================================

pub fn complexity_metric_to_u8(metric: &ComplexityMetric) -> u8 {
    match metric {
        ComplexityMetric::EditDistance => 0,
        ComplexityMetric::DiagonalDistance => 1,
    }
}

pub fn complexity_metric_from_u8(byte: u8) -> io::Result<ComplexityMetric> {
    match byte {
        0 => Ok(ComplexityMetric::EditDistance),
        1 => Ok(ComplexityMetric::DiagonalDistance),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid complexity metric code: {}", byte),
        )),
    }
}

// ============================================================================
// DELTA ENCODING
// ============================================================================

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

// ============================================================================
// ZIGZAG ENCODING
// ============================================================================

/// Zigzag encode a signed value to unsigned
#[inline]
fn encode_zigzag(val: i64) -> u64 {
    ((val << 1) ^ (val >> 63)) as u64
}

/// Helper function to encode varint inline (same as utils::encode_varint)
#[inline]
fn encode_varint_inline(mut value: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        bytes.push(byte);
        if value == 0 {
            break;
        }
    }
    bytes
}

/// Empirical strategy selection by actually compressing samples
/// Returns true if ZigzagDelta should be used (produces smaller output)
pub(crate) fn analyze_smart_compression(records: &[AlignmentRecord], zstd_level: i32) -> bool {
    // Collect sample tracepoints
    let mut all_first_vals = Vec::new();
    let mut all_second_vals = Vec::new();

    let sample_count = records.len();
    for record in records.iter() {
        if let TracepointData::Standard(tps) | TracepointData::Fastga(tps) = &record.tracepoints {
            for (first, second) in tps {
                all_first_vals.push(*first);
                all_second_vals.push(*second);
            }
        }
    }

    if all_first_vals.is_empty() {
        info!("Empirical analysis: No tracepoints found, defaulting to Raw");
        return false;  // Default to Raw
    }

    // Compress with Raw strategy (varint + zstd)
    let raw_size = {
        let mut first_val_buf = Vec::with_capacity(all_first_vals.len() * 2);
        let mut second_val_buf = Vec::with_capacity(all_second_vals.len() * 2);

        // Write first values as varint
        for &val in &all_first_vals {
            first_val_buf.extend_from_slice(&encode_varint_inline(val));
        }
        // Write second values as varint
        for &val in &all_second_vals {
            second_val_buf.extend_from_slice(&encode_varint_inline(val));
        }

        // Compress with zstd
        let first_compressed = zstd::encode_all(&first_val_buf[..], zstd_level).unwrap();
        let second_compressed = zstd::encode_all(&second_val_buf[..], zstd_level).unwrap();

        first_compressed.len() + second_compressed.len()
    };

    // Compress with ZigzagDelta strategy (delta + zigzag + varint + zstd)
    let zigzag_size = {
        let mut first_val_buf = Vec::with_capacity(all_first_vals.len() * 2);
        let mut second_val_buf = Vec::with_capacity(all_second_vals.len() * 2);

        // Process first values: delta + zigzag + varint
        if !all_first_vals.is_empty() {
            let first_val = all_first_vals[0];
            first_val_buf.extend_from_slice(&encode_varint_inline(encode_zigzag(first_val as i64)));

            for i in 1..all_first_vals.len() {
                let delta = all_first_vals[i] as i64 - all_first_vals[i-1] as i64;
                first_val_buf.extend_from_slice(&encode_varint_inline(encode_zigzag(delta)));
            }
        }

        // Process second values: just varint (NO delta, NO zigzag)
        for &val in &all_second_vals {
            second_val_buf.extend_from_slice(&encode_varint_inline(val));
        }

        // Compress with zstd
        let first_compressed = zstd::encode_all(&first_val_buf[..], zstd_level).unwrap();
        let second_compressed = zstd::encode_all(&second_val_buf[..], zstd_level).unwrap();

        first_compressed.len() + second_compressed.len()
    };

    let use_zigzag = zigzag_size < raw_size;
    let winner = if use_zigzag { "ZigzagDelta" } else { "Raw" };
    let diff_pct = if use_zigzag {
        ((raw_size - zigzag_size) as f64 / raw_size as f64) * 100.0
    } else {
        ((zigzag_size - raw_size) as f64 / zigzag_size as f64) * 100.0
    };

    info!(
        "Empirical analysis: sampled {} records, {} tracepoints - Raw: {} bytes, ZigzagDelta: {} bytes -> {} wins ({:.2}% better)",
        sample_count,
        all_first_vals.len(),
        raw_size,
        zigzag_size,
        winner,
        diff_pct
    );

    use_zigzag
}

// ============================================================================
// PAF WRITING
// ============================================================================

pub fn write_paf_line_with_tracepoints<W: Write>(
    writer: &mut W,
    record: &AlignmentRecord,
    string_table: &StringTable,
) -> io::Result<()> {
    let query_name = string_table
        .get(record.query_name_id)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid query name ID"))?;
    let query_len = string_table
        .get_length(record.query_name_id)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid query length"))?;
    let target_name = string_table
        .get(record.target_name_id)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid target name ID"))?;
    let target_len = string_table
        .get_length(record.target_name_id)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid target length"))?;

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

    // Write other tags first
    for tag in &record.tags {
        write!(writer, "\t{}", format_tag(tag))?;
    }

    // Write tracepoints as tp:Z: tag last
    let tp_str = format_tracepoints(&record.tracepoints);
    write!(writer, "\ttp:Z:{}", tp_str)?;

    writeln!(writer)?;
    Ok(())
}

fn format_tracepoints(tps: &TracepointData) -> String {
    match tps {
        TracepointData::Standard(tps) | TracepointData::Fastga(tps) => tps
            .iter()
            .map(|(a, b)| format!("{},{}", a, b))
            .collect::<Vec<_>>()
            .join(";"),
        TracepointData::Variable(tps) => tps
            .iter()
            .map(|(a, b)| {
                if let Some(b_val) = b {
                    format!("{},{}", a, b_val)
                } else {
                    format!("{}", a)
                }
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

// ============================================================================
// DECOMPRESSION
// ============================================================================

pub(crate) fn decompress_varint<R: Read>(
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
            io::Error::new(
                e.kind(),
                format!("Failed to create output file '{}': {}", output_path, e),
            )
        })?)
    };
    let mut writer = BufWriter::new(output);

    // Read records with strategy from header
    for _ in 0..header.num_records {
        let record = read_record_automatic(
            &mut reader,
            strategy,
            header.tracepoint_type,
            header.complexity_metric,
            header.max_complexity,
        )?;
        write_paf_line_with_tracepoints(&mut writer, &record, &string_table)?;
    }
    writer.flush()?;

    info!("Decompressed {} records", header.num_records);
    Ok(())
}

fn read_record_automatic<R: Read>(
    reader: &mut R,
    strategy: CompressionStrategy,
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

    let tracepoints = read_tracepoints_automatic(reader, tp_type, strategy)?;

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

/// Read tracepoints with compression strategy
fn read_tracepoints_automatic<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
    strategy: CompressionStrategy,
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

            let first_len = read_varint(reader)? as usize;
            let mut first_compressed = vec![0u8; first_len];
            reader.read_exact(&mut first_compressed)?;
            let second_len = read_varint(reader)? as usize;
            let mut second_compressed = vec![0u8; second_len];
            reader.read_exact(&mut second_compressed)?;

            let first_buf = zstd::decode_all(&first_compressed[..])?;
            let second_buf = zstd::decode_all(&second_compressed[..])?;

            // Helper to decode values based on strategy
            let decode_values = |buf: &[u8], strategy: CompressionStrategy| -> io::Result<Vec<u64>> {
                let mut reader = buf;
                match strategy {
                    CompressionStrategy::Raw(_) => {
                        // Raw varints
                        let mut vals = Vec::with_capacity(num_items);
                        for _ in 0..num_items {
                            vals.push(read_varint(&mut reader)?);
                        }
                        Ok(vals)
                    }
                    CompressionStrategy::ZigzagDelta(_) => {
                        // Zigzag + delta
                        let mut deltas = Vec::with_capacity(num_items);
                        for _ in 0..num_items {
                            let zigzag = read_varint(&mut reader)?;
                            let val = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
                            deltas.push(val);
                        }
                        Ok(delta_decode(&deltas))
                    }
                    CompressionStrategy::Automatic(_) => {
                        panic!("Automatic strategy must be resolved before decoding")
                    }
                }
            };

            let first_vals = decode_values(&first_buf, strategy)?;
            let second_vals = decode_values(&second_buf, strategy)?;

            let tps: Vec<(u64, u64)> = first_vals.into_iter().zip(second_vals).collect();
            Ok(match tp_type {
                TracepointType::Standard => TracepointData::Standard(tps),
                _ => TracepointData::Fastga(tps),
            })
        }
        TracepointType::Variable => {
            let tps = read_variable_tracepoint_items(reader, num_items)?;
            Ok(TracepointData::Variable(tps))
        }
        TracepointType::Mixed => {
            let items = read_mixed_tracepoint_items(reader, num_items)?;
            Ok(TracepointData::Mixed(items))
        }
    }
}

fn read_variable_tracepoint_items<R: Read>(
    reader: &mut R,
    num_items: usize,
) -> io::Result<Vec<(u64, Option<u64>)>> {
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
    Ok(tps)
}

fn read_mixed_tracepoint_items<R: Read>(
    reader: &mut R,
    num_items: usize,
) -> io::Result<Vec<MixedTracepointItem>> {
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
    Ok(items)
}

// ============================================================================
// ALIGNMENT RECORD I/O
// ============================================================================

impl AlignmentRecord {
    pub(crate) fn write_automatic<W: Write>(
        &self,
        writer: &mut W,
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
        self.write_tracepoints_automatic(writer, strategy)?;
        write_varint(writer, self.tags.len() as u64)?;
        for tag in &self.tags {
            tag.write(writer)?;
        }
        Ok(())
    }

    fn write_tracepoints_automatic<W: Write>(
        &self,
        writer: &mut W,
        strategy: CompressionStrategy,
    ) -> io::Result<()> {
        match &self.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                write_varint(writer, tps.len() as u64)?;
                if tps.is_empty() {
                    return Ok(());
                }
                let (first_vals, second_vals): (Vec<u64>, Vec<u64>) = tps.iter().copied().unzip();

                // Helper to encode values based on strategy
                let encode_values = |vals: &[u64], strategy: CompressionStrategy| -> io::Result<Vec<u8>> {
                    let mut buf = Vec::with_capacity(vals.len() * 2);
                    match strategy {
                        CompressionStrategy::Raw(_) => {
                            // Raw varints
                            for &val in vals {
                                write_varint(&mut buf, val)?;
                            }
                        }
                        CompressionStrategy::ZigzagDelta(_) => {
                            // Zigzag + delta
                            let deltas = delta_encode(vals);
                            for &val in &deltas {
                                let zigzag = ((val << 1) ^ (val >> 63)) as u64;
                                write_varint(&mut buf, zigzag)?;
                            }
                        }
                        CompressionStrategy::Automatic(_) => {
                            panic!("Automatic strategy must be resolved before encoding")
                        }
                    }
                    Ok(buf)
                };

                let first_val_buf = encode_values(&first_vals, strategy)?;
                let second_val_buf = encode_values(&second_vals, strategy)?;

                let first_compressed = zstd::encode_all(&first_val_buf[..], strategy.zstd_level())?;
                let second_compressed =
                    zstd::encode_all(&second_val_buf[..], strategy.zstd_level())?;

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
}

// ============================================================================
// RANDOM ACCESS
// ============================================================================

pub struct BpafIndex {
    /// File offset for each record (byte position in .bpaf file)
    offsets: Vec<u64>,
}

impl BpafIndex {
    const INDEX_MAGIC: &'static [u8; 4] = b"BPAI";

    /// Save index to .bpaf.idx file
    pub fn save(&self, idx_path: &str) -> io::Result<()> {
        let mut file = File::create(idx_path)?;

        file.write_all(Self::INDEX_MAGIC)?;
        file.write_all(&[1u8])?; // Version 1

        write_varint(&mut file, self.offsets.len() as u64)?;

        for &offset in &self.offsets {
            write_varint(&mut file, offset)?;
        }

        Ok(())
    }

    /// Load index from .bpaf.idx file
    pub fn load(idx_path: &str) -> io::Result<Self> {
        let file = File::open(idx_path)?;
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != Self::INDEX_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid index magic",
            ));
        }

        let mut version = [0u8; 1];
        reader.read_exact(&mut version)?;
        if version[0] != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported index version: {}", version[0]),
            ));
        }

        let num_offsets = read_varint(&mut reader)? as usize;

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

    /// Get offset for a specific record ID
    pub fn get_offset(&self, record_id: u64) -> Option<u64> {
        self.offsets.get(record_id as usize).copied()
    }
}

pub fn build_index(bpaf_path: &str) -> io::Result<BpafIndex> {
    info!("Building index for {}", bpaf_path);

    let file = File::open(bpaf_path)?;
    let mut reader = BufReader::with_capacity(131072, file);

    let header = BinaryPafHeader::read(&mut reader)?;
    StringTable::read(&mut reader)?;

    let tp_type = header.tracepoint_type;
    let mut offsets = Vec::with_capacity(header.num_records as usize);
    for _ in 0..header.num_records {
        offsets.push(reader.stream_position()?);
        skip_record(&mut reader, tp_type)?;
    }

    info!("Index built: {} records", offsets.len());
    Ok(BpafIndex { offsets })
}

#[inline]
fn skip_record<R: Read + Seek>(reader: &mut R, tp_type: TracepointType) -> io::Result<()> {
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

    skip_tracepoints(reader, tp_type)?;

    // Skip tags
    let num_tags = read_varint(reader)? as usize;
    for _ in 0..num_tags {
        reader.seek(SeekFrom::Current(2))?; // key
        let mut tag_type = [0u8; 1];
        reader.read_exact(&mut tag_type)?;
        match tag_type[0] {
            b'i' => reader.seek(SeekFrom::Current(4))?,
            b'f' => reader.seek(SeekFrom::Current(4))?,
            b'Z' => {
                let len = read_varint(reader)? as usize;
                reader.seek(SeekFrom::Current(len as i64))?
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Invalid tag type: '{}' (byte: {})",
                        tag_type[0] as char, tag_type[0]
                    ),
                ))
            }
        };
    }

    Ok(())
}

fn skip_tracepoints<R: Read + Seek>(reader: &mut R, tp_type: TracepointType) -> io::Result<()> {
    let num_items = read_varint(reader)? as usize;
    match tp_type {
        TracepointType::Standard | TracepointType::Fastga => {
            if num_items == 0 {
                return Ok(());
            }
            // Read first_len and skip first_data
            let first_len = read_varint(reader)?;
            let first_len = i64::try_from(first_len).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Compressed block too large to skip",
                )
            })?;
            reader.seek(SeekFrom::Current(first_len))?;

            // Read second_len and skip second_data
            let second_len = read_varint(reader)?;
            let second_len = i64::try_from(second_len).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Compressed block too large to skip",
                )
            })?;
            reader.seek(SeekFrom::Current(second_len))?;
        }
        TracepointType::Variable => {
            for _ in 0..num_items {
                read_varint(reader)?;
                let mut flag = [0u8; 1];
                reader.read_exact(&mut flag)?;
                if flag[0] == 1 {
                    read_varint(reader)?;
                }
            }
        }
        TracepointType::Mixed => {
            for _ in 0..num_items {
                let mut item_type = [0u8; 1];
                reader.read_exact(&mut item_type)?;
                match item_type[0] {
                    0 => {
                        read_varint(reader)?;
                        read_varint(reader)?;
                    }
                    1 => {
                        read_varint(reader)?;
                        let mut op = [0u8; 1];
                        reader.read_exact(&mut op)?;
                    }
                    _ => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Invalid mixed item type",
                        ))
                    }
                }
            }
        }
    }
    Ok(())
}

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

        let mut file = File::open(bpaf_path)?;
        let header = BinaryPafHeader::read(&mut file)?;

        let string_table = StringTable::new();

        Ok(Self {
            file,
            index,
            header,
            string_table,
        })
    }

    /// Open a BPAF file without index (for offset-based access only)
    pub fn open_without_index(bpaf_path: &str) -> io::Result<Self> {
        let mut file = File::open(bpaf_path)?;
        let header = BinaryPafHeader::read(&mut file)?;

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
            return Ok(());
        }

        self.file.seek(SeekFrom::Start(0))?;
        BinaryPafHeader::read(&mut self.file)?;
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

        // Read with strategy from header
        let strategy = self.header.strategy()?;
        read_record_automatic(
            &mut self.file,
            strategy,
            self.header.tracepoint_type,
            self.header.complexity_metric,
            self.header.max_complexity,
        )
    }

    /// Get tracepoints only (optimized) - O(1) random access by record ID
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

        let tracepoint_offset = self.get_tracepoint_offset(record_id)?;
        self.get_tracepoints_at_offset(tracepoint_offset)
    }

    /// Get tracepoint offset for a specific record ID
    /// Returns the byte offset where tracepoint data starts within the record
    /// Use this to build external indexes that store tracepoint offsets for fastest access
    pub fn get_tracepoint_offset(&mut self, record_id: u64) -> io::Result<u64> {
        let record_offset = self.index.offsets[record_id as usize];
        self.file.seek(SeekFrom::Start(record_offset))?;

        // Skip 7 varints + 2 single bytes to reach tracepoint data
        read_varint(&mut self.file)?; // query_name_id
        read_varint(&mut self.file)?; // query_start
        read_varint(&mut self.file)?; // query_end
        self.file.seek(SeekFrom::Current(1))?; // strand
        read_varint(&mut self.file)?; // target_name_id
        read_varint(&mut self.file)?; // target_start
        read_varint(&mut self.file)?; // target_end
        read_varint(&mut self.file)?; // residue_matches
        read_varint(&mut self.file)?; // alignment_block_len
        self.file.seek(SeekFrom::Current(1))?; // mapping_quality

        self.file.stream_position()
    }

    /// Get tracepoints by tracepoint offset (fastest access)
    /// Seeks directly to tracepoint data within a record
    /// Use this when your external index stores tracepoint offsets for O(1) access
    pub fn get_tracepoints_at_offset(
        &mut self,
        tracepoint_offset: u64,
    ) -> io::Result<(TracepointData, TracepointType, ComplexityMetric, u64)> {
        self.file.seek(SeekFrom::Start(tracepoint_offset))?;

        let tp_type = self.header.tracepoint_type;
        let complexity_metric = self.header.complexity_metric;
        let max_complexity = self.header.max_complexity;

        // Read tracepoints with strategy from header
        let strategy = self.header.strategy()?;
        let tracepoints = read_tracepoints_automatic(
            &mut self.file,
            tp_type,
            strategy,
        )?;

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
