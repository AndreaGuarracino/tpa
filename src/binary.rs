//! Binary I/O operations for BPAF format

use lib_tracepoints::{ComplexityMetric, MixedRepresentation, TracepointData, TracepointType};
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
#[allow(dead_code)]
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

/// Analyze query-target correlation for adaptive strategy selection
pub(crate) fn analyze_correlation(records: &[AlignmentRecord]) -> f64 {
    // Collect tracepoint pairs
    let mut query_vals: Vec<i64> = Vec::new();
    let mut target_vals: Vec<i64> = Vec::new();

    for record in records.iter() {
        if let TracepointData::Standard(tps) | TracepointData::Fastga(tps) = &record.tracepoints {
            for (q, t) in tps {
                query_vals.push(*q as i64);
                target_vals.push(*t as i64);
            }
        }
    }

    if query_vals.len() < 2 {
        return 0.0; // Not enough data
    }

    // Calculate Pearson correlation coefficient
    let n = query_vals.len() as f64;
    let sum_q: f64 = query_vals.iter().map(|&x| x as f64).sum();
    let sum_t: f64 = target_vals.iter().map(|&x| x as f64).sum();
    let sum_qq: f64 = query_vals.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let sum_tt: f64 = target_vals.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let sum_qt: f64 = query_vals.iter().zip(target_vals.iter())
        .map(|(&q, &t)| (q as f64) * (t as f64))
        .sum();

    let numerator = n * sum_qt - sum_q * sum_t;
    let denominator = ((n * sum_qq - sum_q * sum_q) * (n * sum_tt - sum_t * sum_t)).sqrt();

    if denominator.abs() < 1e-10 {
        return 0.0; // Avoid division by zero
    }

    (numerator / denominator).abs() // Return absolute value of correlation
}

/// Empirical strategy selection by actually compressing a subset of records
pub(crate) fn analyze_smart_compression(records: &[AlignmentRecord], zstd_level: i32) -> bool {
    // Collect sample tracepoints
    let mut all_first_vals = Vec::new();
    let mut all_second_vals = Vec::new();

    let sample_count = records.len();
    for record in records.iter() {
        if let TracepointData::Standard(tps) | TracepointData::Fastga(tps) = &record.tracepoints {
            for (first, second) in tps {
                all_first_vals.push(*first as u64);
                all_second_vals.push(*second as u64);
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
            let deltas = delta_encode(&all_first_vals);
            for delta in deltas {
                first_val_buf.extend_from_slice(&encode_varint_inline(encode_zigzag(delta)));
            }
        }

        // Process second values: delta + zigzag + varint
        if !all_second_vals.is_empty() {
            let deltas = delta_encode(&all_second_vals);
            for delta in deltas {
                second_val_buf.extend_from_slice(&encode_varint_inline(encode_zigzag(delta)));
            }
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

    debug!(
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
    let tp_str = record.tracepoints.to_tp_tag();
    write!(writer, "\ttp:Z:{}", tp_str)?;

    writeln!(writer)?;
    Ok(())
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
    let string_table = StringTable::read(&mut reader, header.num_strings())?;

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
        let record = read_record(
            &mut reader,
            strategy,
            header.tracepoint_type,
        )?;
        write_paf_line_with_tracepoints(&mut writer, &record, &string_table)?;
    }
    writer.flush()?;

    info!("Decompressed {} records", header.num_records);
    Ok(())
}

fn read_record<R: Read>(
    reader: &mut R,
    strategy: CompressionStrategy,
    tp_type: TracepointType,
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

    let tracepoints = read_tracepoints(reader, tp_type, strategy)?;

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
        tracepoints,
        tags,
    })
}

fn read_tracepoints<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
    strategy: CompressionStrategy,
) -> io::Result<TracepointData> {
    let num_items = read_varint(reader)? as usize;
    match tp_type {
        TracepointType::Standard | TracepointType::Fastga => {
            let tps = decode_standard_tracepoints(reader, num_items, strategy)?;
            Ok(match tp_type {
                TracepointType::Standard => TracepointData::Standard(tps),
                _ => TracepointData::Fastga(tps),
            })
        }
        TracepointType::Variable => {
            let tps = decode_variable_tracepoints(reader, num_items)?;
            Ok(TracepointData::Variable(tps))
        }
        TracepointType::Mixed => {
            let items = decode_mixed_tracepoints(reader, num_items)?;
            Ok(TracepointData::Mixed(items))
        }
    }
}

/// Encode tracepoint values based on compression strategy
#[inline]
fn encode_tracepoint_values(
    vals: &[u64],
    strategy: CompressionStrategy,
) -> io::Result<Vec<u8>> {
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
        CompressionStrategy::TwoDimDelta(_) => {
            // For first values: same as ZigzagDelta
            // For second values: will be handled specially in write_tracepoints
            let deltas = delta_encode(vals);
            for &val in &deltas {
                let zigzag = ((val << 1) ^ (val >> 63)) as u64;
                write_varint(&mut buf, zigzag)?;
            }
        }
        CompressionStrategy::RunLength(_) => {
            // RLE: encode (value, run_length) pairs
            if vals.is_empty() {
                return Ok(buf);
            }

            let mut run_val = vals[0];
            let mut run_len = 1u64;

            for &val in &vals[1..] {
                if val == run_val {
                    run_len += 1;
                } else {
                    // Write (value, length) pair
                    write_varint(&mut buf, run_val)?;
                    write_varint(&mut buf, run_len)?;
                    run_val = val;
                    run_len = 1;
                }
            }
            // Write final run
            write_varint(&mut buf, run_val)?;
            write_varint(&mut buf, run_len)?;
        }
        CompressionStrategy::BitPacked(_) => {
            // Bit packing: find max value, determine bits needed, pack
            if vals.is_empty() {
                return Ok(buf);
            }

            let max_val = *vals.iter().max().unwrap();
            let bits_needed = if max_val == 0 { 1 } else { 64 - max_val.leading_zeros() };

            // Store bits_needed (1 byte)
            buf.push(bits_needed as u8);

            // Simple byte-aligned packing for now (can optimize later)
            if bits_needed <= 8 {
                for &val in vals {
                    buf.push(val as u8);
                }
            } else if bits_needed <= 16 {
                for &val in vals {
                    buf.extend_from_slice(&(val as u16).to_le_bytes());
                }
            } else if bits_needed <= 32 {
                for &val in vals {
                    buf.extend_from_slice(&(val as u32).to_le_bytes());
                }
            } else {
                for &val in vals {
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
        }
        CompressionStrategy::DeltaOfDelta(_) => {
            // Delta-of-delta (Gorilla-style): delta twice for regularly spaced coordinates
            if vals.is_empty() {
                return Ok(buf);
            }
            let deltas = delta_encode(vals);
            let delta_deltas = delta_encode(&deltas.iter().map(|&x| x as u64).collect::<Vec<_>>());
            for &val in &delta_deltas {
                let zigzag = ((val << 1) ^ (val >> 63)) as u64;
                write_varint(&mut buf, zigzag)?;
            }
        }
        CompressionStrategy::FrameOfReference(_) => {
            // Frame-of-Reference: min + bit-packed offsets
            if vals.is_empty() {
                return Ok(buf);
            }
            let min_val = *vals.iter().min().unwrap();
            let max_offset = vals.iter().map(|&v| v - min_val).max().unwrap();
            let bits_needed = if max_offset == 0 { 1 } else { 64 - max_offset.leading_zeros() };

            // Write min value and bits_needed
            write_varint(&mut buf, min_val)?;
            buf.push(bits_needed as u8);

            // Write offsets
            for &val in vals {
                let offset = val - min_val;
                if bits_needed <= 8 {
                    buf.push(offset as u8);
                } else if bits_needed <= 16 {
                    buf.extend_from_slice(&(offset as u16).to_le_bytes());
                } else if bits_needed <= 32 {
                    buf.extend_from_slice(&(offset as u32).to_le_bytes());
                } else {
                    buf.extend_from_slice(&offset.to_le_bytes());
                }
            }
        }
        CompressionStrategy::HybridRLE(_) => {
            // Will be handled specially in write_tracepoints (RLE for target, varint for query)
            // For now, use regular varint
            for &val in vals {
                write_varint(&mut buf, val)?;
            }
        }
        CompressionStrategy::OffsetJoint(_) => {
            // For first values: regular zigzag delta
            // For second values: will be handled specially (as offset from first)
            let deltas = delta_encode(vals);
            for &val in &deltas {
                let zigzag = ((val << 1) ^ (val >> 63)) as u64;
                write_varint(&mut buf, zigzag)?;
            }
        }
        CompressionStrategy::XORDelta(_) => {
            // XOR-based differential (Gorilla): XOR with predecessor
            if vals.is_empty() {
                return Ok(buf);
            }
            // First value
            write_varint(&mut buf, vals[0])?;
            // XOR subsequent values with predecessor
            for i in 1..vals.len() {
                let xor_val = vals[i] ^ vals[i - 1];
                write_varint(&mut buf, xor_val)?;
            }
        }
        CompressionStrategy::Dictionary(_) => {
            // Dictionary coding: build dictionary, encode indices
            if vals.is_empty() {
                return Ok(buf);
            }
            use std::collections::HashMap;
            let mut dict: Vec<u64> = Vec::new();
            let mut dict_map: HashMap<u64, u32> = HashMap::new();
            let mut indices: Vec<u32> = Vec::new();

            // Build dictionary
            for &val in vals {
                if !dict_map.contains_key(&val) {
                    dict_map.insert(val, dict.len() as u32);
                    dict.push(val);
                }
                indices.push(*dict_map.get(&val).unwrap());
            }

            // Write dictionary size
            write_varint(&mut buf, dict.len() as u64)?;
            // Write dictionary values
            for &val in &dict {
                write_varint(&mut buf, val)?;
            }
            // Write indices
            for &idx in &indices {
                write_varint(&mut buf, idx as u64)?;
            }
        }
        CompressionStrategy::Simple8(_) => {
            // Simple8-style: pack multiple small integers into 64-bit words
            if vals.is_empty() {
                return Ok(buf);
            }
            // Simplified version: use varint for now (full Simple8 needs selector logic)
            for &val in vals {
                write_varint(&mut buf, val)?;
            }
        }
        CompressionStrategy::StreamVByte(_) => {
            // Stream VByte: separated control and data bytes
            if vals.is_empty() {
                return Ok(buf);
            }
            let mut control_bytes: Vec<u8> = Vec::new();
            let mut data_bytes: Vec<u8> = Vec::new();

            for &val in vals {
                let bytes_needed = if val == 0 { 1 } else { ((64 - val.leading_zeros() + 7) / 8) as usize };
                // 2 bits per value: 00=1 byte, 01=2 bytes, 10=3 bytes, 11=4 bytes
                let control = (bytes_needed - 1).min(3) as u8;
                control_bytes.push(control);

                // Write data bytes
                for i in 0..bytes_needed {
                    data_bytes.push(((val >> (i * 8)) & 0xFF) as u8);
                }
            }

            // Pack control bytes (4 values per byte, 2 bits each)
            let mut packed_control: Vec<u8> = Vec::new();
            for chunk in control_bytes.chunks(4) {
                let mut byte = 0u8;
                for (i, &ctrl) in chunk.iter().enumerate() {
                    byte |= ctrl << (i * 2);
                }
                packed_control.push(byte);
            }

            // Write control length, control bytes, then data bytes
            write_varint(&mut buf, packed_control.len() as u64)?;
            buf.extend_from_slice(&packed_control);
            buf.extend_from_slice(&data_bytes);
        }
        CompressionStrategy::FastPFOR(_) => {
            // FastPFOR: Patched Frame-of-Reference with exceptions
            buf = crate::hybrids::encode_fastpfor(vals)?;
        }
        CompressionStrategy::Cascaded(_) => {
            // Cascaded compression: Dictionary → RLE → base
            buf = crate::hybrids::encode_cascaded(vals)?;
        }
        CompressionStrategy::Simple8bFull(_) => {
            // Simple8b-RLE Full: pack into 64-bit words
            let words = crate::hybrids::encode_simple8b_full(vals)?;
            for word in words {
                buf.extend_from_slice(&word.to_le_bytes());
            }
        }
        CompressionStrategy::SelectiveRLE(_) => {
            // Selective RLE: detect and encode runs
            buf = crate::hybrids::encode_selective_rle(vals)?;
        }
        CompressionStrategy::Automatic(_) | CompressionStrategy::AdaptiveCorrelation(_) => {
            panic!("Automatic strategies must be resolved before encoding")
        }
    }
    Ok(buf)
}

/// Encode second values for 2D-Delta strategy (as delta from first values)
#[inline]
fn encode_2d_delta_second_values(
    first_vals: &[u64],
    second_vals: &[u64],
) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(second_vals.len() * 2);

    // Compute differences: second - first
    let diffs: Vec<i64> = first_vals.iter()
        .zip(second_vals.iter())
        .map(|(f, s)| *s as i64 - *f as i64)
        .collect();

    // Apply zigzag encoding to differences
    for diff in diffs {
        let zigzag = ((diff << 1) ^ (diff >> 63)) as u64;
        write_varint(&mut buf, zigzag)?;
    }

    Ok(buf)
}

/// Decode tracepoint values based on compression strategy
#[inline]
#[allow(dead_code)]
fn decode_tracepoint_values(
    buf: &[u8],
    num_items: usize,
    strategy: CompressionStrategy,
) -> io::Result<Vec<u64>> {
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
        CompressionStrategy::ZigzagDelta(_) | CompressionStrategy::TwoDimDelta(_) => {
            // Zigzag + delta decode
            let mut vals = Vec::with_capacity(num_items);

            // First value
            let zigzag = read_varint(&mut reader)?;
            let first = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
            vals.push(first as u64);

            // Remaining values: zigzag decode + delta accumulate in one pass
            for _ in 1..num_items {
                let zigzag = read_varint(&mut reader)?;
                let delta = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
                let prev = *vals.last().unwrap() as i64;
                vals.push((prev + delta) as u64);
            }
            Ok(vals)
        }
        CompressionStrategy::RunLength(_) => {
            // RLE: decode (value, run_length) pairs
            let mut vals = Vec::with_capacity(num_items);

            while vals.len() < num_items {
                let value = read_varint(&mut reader)?;
                let run_len = read_varint(&mut reader)? as usize;

                for _ in 0..run_len {
                    vals.push(value);
                    if vals.len() >= num_items {
                        break;
                    }
                }
            }
            Ok(vals)
        }
        CompressionStrategy::BitPacked(_) => {
            // Bit packing: read bits_needed, then unpack
            if num_items == 0 {
                return Ok(Vec::new());
            }

            let bits_needed = reader[0] as u32;
            reader = &reader[1..];

            let mut vals = Vec::with_capacity(num_items);

            if bits_needed <= 8 {
                for _ in 0..num_items {
                    vals.push(reader[0] as u64);
                    reader = &reader[1..];
                }
            } else if bits_needed <= 16 {
                for _ in 0..num_items {
                    let val = u16::from_le_bytes([reader[0], reader[1]]) as u64;
                    vals.push(val);
                    reader = &reader[2..];
                }
            } else if bits_needed <= 32 {
                for _ in 0..num_items {
                    let val = u32::from_le_bytes([reader[0], reader[1], reader[2], reader[3]]) as u64;
                    vals.push(val);
                    reader = &reader[4..];
                }
            } else {
                for _ in 0..num_items {
                    let val = u64::from_le_bytes([
                        reader[0], reader[1], reader[2], reader[3],
                        reader[4], reader[5], reader[6], reader[7],
                    ]);
                    vals.push(val);
                    reader = &reader[8..];
                }
            }
            Ok(vals)
        }
        CompressionStrategy::DeltaOfDelta(_) => {
            // Delta-of-delta decode
            if num_items == 0 {
                return Ok(Vec::new());
            }
            let mut delta_deltas = Vec::with_capacity(num_items);
            for _ in 0..num_items {
                let zigzag = read_varint(&mut reader)?;
                let val = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
                delta_deltas.push(val);
            }
            // Apply delta decode twice
            let deltas = delta_decode(&delta_deltas);
            let deltas_i64: Vec<i64> = deltas.iter().map(|&x| x as i64).collect();
            Ok(delta_decode(&deltas_i64))
        }
        CompressionStrategy::FrameOfReference(_) => {
            // Frame-of-Reference decode
            if num_items == 0 {
                return Ok(Vec::new());
            }
            let min_val = read_varint(&mut reader)?;
            let bits_needed = reader[0] as u32;
            reader = &reader[1..];

            let mut vals = Vec::with_capacity(num_items);
            if bits_needed <= 8 {
                for _ in 0..num_items {
                    vals.push(min_val + reader[0] as u64);
                    reader = &reader[1..];
                }
            } else if bits_needed <= 16 {
                for _ in 0..num_items {
                    let offset = u16::from_le_bytes([reader[0], reader[1]]) as u64;
                    vals.push(min_val + offset);
                    reader = &reader[2..];
                }
            } else if bits_needed <= 32 {
                for _ in 0..num_items {
                    let offset = u32::from_le_bytes([reader[0], reader[1], reader[2], reader[3]]) as u64;
                    vals.push(min_val + offset);
                    reader = &reader[4..];
                }
            } else {
                for _ in 0..num_items {
                    let offset = u64::from_le_bytes([
                        reader[0], reader[1], reader[2], reader[3],
                        reader[4], reader[5], reader[6], reader[7],
                    ]);
                    vals.push(min_val + offset);
                    reader = &reader[8..];
                }
            }
            Ok(vals)
        }
        CompressionStrategy::HybridRLE(_) => {
            // Decode as regular varint (special handling for target in decode_standard_tracepoints)
            let mut vals = Vec::with_capacity(num_items);
            for _ in 0..num_items {
                vals.push(read_varint(&mut reader)?);
            }
            Ok(vals)
        }
        CompressionStrategy::OffsetJoint(_) => {
            // Same as ZigzagDelta for first values
            let mut vals = Vec::with_capacity(num_items);
            if num_items == 0 {
                return Ok(vals);
            }
            let zigzag = read_varint(&mut reader)?;
            let first = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
            vals.push(first as u64);
            for _ in 1..num_items {
                let zigzag = read_varint(&mut reader)?;
                let delta = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
                let prev = *vals.last().unwrap() as i64;
                vals.push((prev + delta) as u64);
            }
            Ok(vals)
        }
        CompressionStrategy::XORDelta(_) => {
            // XOR decode
            if num_items == 0 {
                return Ok(Vec::new());
            }
            let mut vals = Vec::with_capacity(num_items);
            vals.push(read_varint(&mut reader)?);
            for _ in 1..num_items {
                let xor_val = read_varint(&mut reader)?;
                let prev = *vals.last().unwrap();
                vals.push(prev ^ xor_val);
            }
            Ok(vals)
        }
        CompressionStrategy::Dictionary(_) => {
            // Dictionary decode
            let dict_size = read_varint(&mut reader)? as usize;
            let mut dict = Vec::with_capacity(dict_size);
            for _ in 0..dict_size {
                dict.push(read_varint(&mut reader)?);
            }
            let mut vals = Vec::with_capacity(num_items);
            for _ in 0..num_items {
                let idx = read_varint(&mut reader)? as usize;
                vals.push(dict[idx]);
            }
            Ok(vals)
        }
        CompressionStrategy::Simple8(_) => {
            // Simple8 decode (simplified: just varint)
            let mut vals = Vec::with_capacity(num_items);
            for _ in 0..num_items {
                vals.push(read_varint(&mut reader)?);
            }
            Ok(vals)
        }
        CompressionStrategy::StreamVByte(_) => {
            // Stream VByte decode
            if num_items == 0 {
                return Ok(Vec::new());
            }
            let control_len = read_varint(&mut reader)? as usize;
            let mut control_bytes = vec![0u8; control_len];
            reader.read_exact(&mut control_bytes)?;

            // Unpack control bytes
            let mut controls = Vec::new();
            for &packed in &control_bytes {
                for i in 0..4 {
                    controls.push((packed >> (i * 2)) & 0x03);
                    if controls.len() >= num_items {
                        break;
                    }
                }
            }

            // Read data bytes according to control
            let mut vals = Vec::with_capacity(num_items);
            for &ctrl in controls.iter().take(num_items) {
                let bytes_needed = (ctrl + 1) as usize;
                let mut val = 0u64;
                for i in 0..bytes_needed {
                    val |= (reader[0] as u64) << (i * 8);
                    reader = &reader[1..];
                }
                vals.push(val);
            }
            Ok(vals)
        }
        CompressionStrategy::FastPFOR(_) => {
            // FastPFOR decode
            crate::hybrids::decode_fastpfor(buf)
        }
        CompressionStrategy::Cascaded(_) => {
            // Cascaded decode
            crate::hybrids::decode_cascaded(buf)
        }
        CompressionStrategy::Simple8bFull(_) => {
            // Simple8b-RLE Full decode
            // Convert bytes back to u64 words
            let mut words = Vec::new();
            let mut i = 0;
            while i + 8 <= buf.len() {
                let word = u64::from_le_bytes([
                    buf[i], buf[i+1], buf[i+2], buf[i+3],
                    buf[i+4], buf[i+5], buf[i+6], buf[i+7],
                ]);
                words.push(word);
                i += 8;
            }
            crate::hybrids::decode_simple8b_full(&words)
        }
        CompressionStrategy::SelectiveRLE(_) => {
            // Selective RLE decode
            crate::hybrids::decode_selective_rle(buf)
        }
        CompressionStrategy::Automatic(_) | CompressionStrategy::AdaptiveCorrelation(_) => {
            panic!("Automatic strategies must be resolved before decoding")
        }
    }
}

/// Decode second values for 2D-Delta strategy (from diff values and first values)
#[inline]
fn decode_2d_delta_second_values(
    buf: &[u8],
    first_vals: &[u64],
) -> io::Result<Vec<u64>> {
    let mut reader = buf;
    let mut second_vals = Vec::with_capacity(first_vals.len());

    // Decode zigzag-encoded differences and add to first values
    for &first in first_vals {
        let zigzag = read_varint(&mut reader)?;
        let diff = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
        let second = (first as i64 + diff) as u64;
        second_vals.push(second);
    }

    Ok(second_vals)
}

/// Standard/Fastga tracepoint decoding
#[inline(always)]
fn decode_standard_tracepoints<R: Read>(
    reader: &mut R,
    num_items: usize,
    strategy: CompressionStrategy,
) -> io::Result<Vec<(usize, usize)>> {
    // Read compressed blocks
    let first_len = read_varint(reader)? as usize;
    let mut first_compressed = vec![0u8; first_len];
    reader.read_exact(&mut first_compressed)?;

    let second_len = read_varint(reader)? as usize;
    let mut second_compressed = vec![0u8; second_len];
    reader.read_exact(&mut second_compressed)?;

    // Decompress
    let first_buf = zstd::decode_all(&first_compressed[..])?;
    let second_buf = zstd::decode_all(&second_compressed[..])?;

    // Decode directly to (usize, usize) tuples
    let mut first_reader = &first_buf[..];
    let mut second_reader = &second_buf[..];

    let mut tps = Vec::with_capacity(num_items);

    match strategy {
        CompressionStrategy::Raw(_) => {
            // Raw varints - direct decode to usize
            for _ in 0..num_items {
                let a = read_varint(&mut first_reader)? as usize;
                let b = read_varint(&mut second_reader)? as usize;
                tps.push((a, b));
            }
        }
        CompressionStrategy::ZigzagDelta(_) => {
            // First values
            let zigzag_a = read_varint(&mut first_reader)?;
            let a = (((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64)) as usize;

            let zigzag_b = read_varint(&mut second_reader)?;
            let b = (((zigzag_b >> 1) as i64) ^ -((zigzag_b & 1) as i64)) as usize;

            tps.push((a, b));

            // Remaining values: zigzag decode + delta accumulate
            for _ in 1..num_items {
                let zigzag_a = read_varint(&mut first_reader)?;
                let delta_a = ((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64);
                let prev_a = tps.last().unwrap().0 as i64;
                let a = (prev_a + delta_a) as usize;

                let zigzag_b = read_varint(&mut second_reader)?;
                let delta_b = ((zigzag_b >> 1) as i64) ^ -((zigzag_b & 1) as i64);
                let prev_b = tps.last().unwrap().1 as i64;
                let b = (prev_b + delta_b) as usize;

                tps.push((a, b));
            }
        }
        CompressionStrategy::TwoDimDelta(_) => {
            // First values: same as ZigzagDelta
            let mut first_vals = Vec::with_capacity(num_items);
            let zigzag_a = read_varint(&mut first_reader)?;
            let a = ((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64);
            first_vals.push(a as usize);

            for _ in 1..num_items {
                let zigzag_a = read_varint(&mut first_reader)?;
                let delta_a = ((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64);
                let prev_a = *first_vals.last().unwrap() as i64;
                first_vals.push((prev_a + delta_a) as usize);
            }

            // Second values: decode as differences from first values
            for &a in &first_vals {
                let zigzag_diff = read_varint(&mut second_reader)?;
                let diff = ((zigzag_diff >> 1) as i64) ^ -((zigzag_diff & 1) as i64);
                let b = (a as i64 + diff) as usize;
                tps.push((a, b));
            }
        }
        CompressionStrategy::RunLength(_) => {
            // RLE decoding for both first and second values
            let mut first_vals = Vec::with_capacity(num_items);
            while first_vals.len() < num_items {
                let value = read_varint(&mut first_reader)? as usize;
                let run_len = read_varint(&mut first_reader)? as usize;
                for _ in 0..run_len {
                    first_vals.push(value);
                    if first_vals.len() >= num_items {
                        break;
                    }
                }
            }

            let mut second_vals = Vec::with_capacity(num_items);
            while second_vals.len() < num_items {
                let value = read_varint(&mut second_reader)? as usize;
                let run_len = read_varint(&mut second_reader)? as usize;
                for _ in 0..run_len {
                    second_vals.push(value);
                    if second_vals.len() >= num_items {
                        break;
                    }
                }
            }

            for (a, b) in first_vals.into_iter().zip(second_vals.into_iter()) {
                tps.push((a, b));
            }
        }
        CompressionStrategy::BitPacked(_) => {
            // Bit packing decoding
            let bits_needed_a = first_reader[0] as u32;
            first_reader = &first_reader[1..];

            let bits_needed_b = second_reader[0] as u32;
            second_reader = &second_reader[1..];

            let mut first_vals = Vec::with_capacity(num_items);
            let mut second_vals = Vec::with_capacity(num_items);

            // Decode first values
            if bits_needed_a <= 8 {
                for _ in 0..num_items {
                    first_vals.push(first_reader[0] as usize);
                    first_reader = &first_reader[1..];
                }
            } else if bits_needed_a <= 16 {
                for _ in 0..num_items {
                    let val = u16::from_le_bytes([first_reader[0], first_reader[1]]) as usize;
                    first_vals.push(val);
                    first_reader = &first_reader[2..];
                }
            } else if bits_needed_a <= 32 {
                for _ in 0..num_items {
                    let val = u32::from_le_bytes([first_reader[0], first_reader[1], first_reader[2], first_reader[3]]) as usize;
                    first_vals.push(val);
                    first_reader = &first_reader[4..];
                }
            } else {
                for _ in 0..num_items {
                    let val = u64::from_le_bytes([
                        first_reader[0], first_reader[1], first_reader[2], first_reader[3],
                        first_reader[4], first_reader[5], first_reader[6], first_reader[7],
                    ]) as usize;
                    first_vals.push(val);
                    first_reader = &first_reader[8..];
                }
            }

            // Decode second values
            if bits_needed_b <= 8 {
                for _ in 0..num_items {
                    second_vals.push(second_reader[0] as usize);
                    second_reader = &second_reader[1..];
                }
            } else if bits_needed_b <= 16 {
                for _ in 0..num_items {
                    let val = u16::from_le_bytes([second_reader[0], second_reader[1]]) as usize;
                    second_vals.push(val);
                    second_reader = &second_reader[2..];
                }
            } else if bits_needed_b <= 32 {
                for _ in 0..num_items {
                    let val = u32::from_le_bytes([second_reader[0], second_reader[1], second_reader[2], second_reader[3]]) as usize;
                    second_vals.push(val);
                    second_reader = &second_reader[4..];
                }
            } else {
                for _ in 0..num_items {
                    let val = u64::from_le_bytes([
                        second_reader[0], second_reader[1], second_reader[2], second_reader[3],
                        second_reader[4], second_reader[5], second_reader[6], second_reader[7],
                    ]) as usize;
                    second_vals.push(val);
                    second_reader = &second_reader[8..];
                }
            }

            for (a, b) in first_vals.into_iter().zip(second_vals.into_iter()) {
                tps.push((a, b));
            }
        }
        CompressionStrategy::DeltaOfDelta(_) | CompressionStrategy::FrameOfReference(_) |
        CompressionStrategy::XORDelta(_) | CompressionStrategy::Dictionary(_) |
        CompressionStrategy::Simple8(_) | CompressionStrategy::StreamVByte(_) |
        CompressionStrategy::FastPFOR(_) | CompressionStrategy::Cascaded(_) |
        CompressionStrategy::Simple8bFull(_) | CompressionStrategy::SelectiveRLE(_) => {
            // These strategies use standard encoding for both streams
            let first_vals = decode_tracepoint_values(&first_buf, num_items, strategy)?;
            let second_vals = decode_tracepoint_values(&second_buf, num_items, strategy)?;
            for (a, b) in first_vals.into_iter().zip(second_vals.into_iter()) {
                tps.push((a as usize, b as usize));
            }
        }
        CompressionStrategy::HybridRLE(_) => {
            // Query as varint, target as RLE
            let mut first_vals = Vec::with_capacity(num_items);
            for _ in 0..num_items {
                first_vals.push(read_varint(&mut first_reader)? as usize);
            }

            // Decode target with RLE
            let mut second_vals = Vec::with_capacity(num_items);
            while second_vals.len() < num_items {
                let value = read_varint(&mut second_reader)? as usize;
                let run_len = read_varint(&mut second_reader)? as usize;
                for _ in 0..run_len {
                    second_vals.push(value);
                    if second_vals.len() >= num_items {
                        break;
                    }
                }
            }

            for (a, b) in first_vals.into_iter().zip(second_vals.into_iter()) {
                tps.push((a, b));
            }
        }
        CompressionStrategy::OffsetJoint(_) => {
            // First values: zigzag delta
            let mut first_vals = Vec::with_capacity(num_items);
            let zigzag_a = read_varint(&mut first_reader)?;
            let a = ((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64);
            first_vals.push(a as usize);

            for _ in 1..num_items {
                let zigzag_a = read_varint(&mut first_reader)?;
                let delta_a = ((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64);
                let prev_a = *first_vals.last().unwrap() as i64;
                first_vals.push((prev_a + delta_a) as usize);
            }

            // Second values: as offset from first
            for &a in &first_vals {
                let offset = read_varint(&mut second_reader)? as usize;
                let b = a + offset;
                tps.push((a, b));
            }
        }
        CompressionStrategy::Automatic(_) | CompressionStrategy::AdaptiveCorrelation(_) => {
            panic!("Automatic strategies must be resolved before decoding")
        }
    }

    Ok(tps)
}

/// Variable tracepoint decoding (only raw varints)
#[inline(always)]
fn decode_variable_tracepoints<R: Read>(
    reader: &mut R,
    num_items: usize,
) -> io::Result<Vec<(usize, Option<usize>)>> {
    let mut tps = Vec::with_capacity(num_items);
    for _ in 0..num_items {
        let a = read_varint(reader)? as usize;
        let mut flag = [0u8; 1];
        reader.read_exact(&mut flag)?;
        let b_opt = if flag[0] == 1 {
            Some(read_varint(reader)? as usize)
        } else {
            None
        };
        tps.push((a, b_opt));
    }
    Ok(tps)
}

/// Mixed tracepoint decoding (only raw varints)
#[inline(always)]
fn decode_mixed_tracepoints<R: Read>(
    reader: &mut R,
    num_items: usize,
) -> io::Result<Vec<MixedRepresentation>> {
    let mut items = Vec::with_capacity(num_items);
    for _ in 0..num_items {
        let mut item_type = [0u8; 1];
        reader.read_exact(&mut item_type)?;
        match item_type[0] {
            0 => {
                let a = read_varint(reader)? as usize;
                let b = read_varint(reader)? as usize;
                items.push(MixedRepresentation::Tracepoint(a, b));
            }
            1 => {
                let len = read_varint(reader)? as usize;
                let mut op = [0u8; 1];
                reader.read_exact(&mut op)?;
                items.push(MixedRepresentation::CigarOp(len, op[0] as char));
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
    pub(crate) fn write<W: Write>(
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
        self.write_tracepoints(writer, strategy)?;
        write_varint(writer, self.tags.len() as u64)?;
        for tag in &self.tags {
            tag.write(writer)?;
        }
        Ok(())
    }

    fn write_tracepoints<W: Write>(
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
                let (first_vals, second_vals): (Vec<u64>, Vec<u64>) = tps
                    .iter()
                    .map(|(a, b)| (*a as u64, *b as u64))
                    .unzip();

                let first_val_buf = encode_tracepoint_values(&first_vals, strategy)?;
                let second_val_buf = match strategy {
                    CompressionStrategy::TwoDimDelta(_) => {
                        // Encode second as delta from first
                        encode_2d_delta_second_values(&first_vals, &second_vals)?
                    }
                    CompressionStrategy::OffsetJoint(_) => {
                        // Encode second as offset from first (simpler than 2D-Delta)
                        let mut buf = Vec::with_capacity(second_vals.len() * 2);
                        for (f, s) in first_vals.iter().zip(second_vals.iter()) {
                            let offset = s - f; // No zigzag, just raw offset
                            write_varint(&mut buf, offset)?;
                        }
                        buf
                    }
                    CompressionStrategy::HybridRLE(_) => {
                        // Encode target with RLE
                        let mut buf = Vec::with_capacity(second_vals.len() * 2);
                        if !second_vals.is_empty() {
                            let mut run_val = second_vals[0];
                            let mut run_len = 1u64;
                            for &val in &second_vals[1..] {
                                if val == run_val {
                                    run_len += 1;
                                } else {
                                    write_varint(&mut buf, run_val)?;
                                    write_varint(&mut buf, run_len)?;
                                    run_val = val;
                                    run_len = 1;
                                }
                            }
                            write_varint(&mut buf, run_val)?;
                            write_varint(&mut buf, run_len)?;
                        }
                        buf
                    }
                    _ => {
                        // Standard encoding
                        encode_tracepoint_values(&second_vals, strategy)?
                    }
                };

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
                    write_varint(writer, *a as u64)?;
                    if let Some(b) = b_opt {
                        writer.write_all(&[1])?;
                        write_varint(writer, *b as u64)?;
                    } else {
                        writer.write_all(&[0])?;
                    }
                }
            }
            TracepointData::Mixed(items) => {
                write_varint(writer, items.len() as u64)?;
                for item in items {
                    match item {
                        MixedRepresentation::Tracepoint(a, b) => {
                            writer.write_all(&[0])?;
                            write_varint(writer, *a as u64)?;
                            write_varint(writer, *b as u64)?;
                        }
                        MixedRepresentation::CigarOp(len, op) => {
                            writer.write_all(&[1])?;
                            write_varint(writer, *len as u64)?;
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
    if header.version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported format version: {}", header.version),
        ));
    }
    StringTable::read(&mut reader, header.num_strings)?;

    let mut offsets = Vec::with_capacity(header.num_records as usize);
    for _ in 0..header.num_records {
        offsets.push(reader.stream_position()?);
        skip_record(&mut reader, header.tracepoint_type)?;
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
            debug!("Loading existing index: {}", idx_path);
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
        if header.version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported format version: {}", header.version),
            ));
        }

        let string_table = StringTable::new();

        Ok(Self {
            file,
            index,
            header,
            string_table,
        })
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

    /// Load string table (call this if you need sequence names)
    pub fn load_string_table(&mut self) -> io::Result<()> {
        if !self.string_table.is_empty() {
            return Ok(());
        }

        self.file.seek(SeekFrom::Start(0))?;
        let header = BinaryPafHeader::read(&mut self.file)?;
        self.string_table = StringTable::read(&mut self.file, header.num_strings())?;
        Ok(())
    }

    /// Get immutable reference to string table (must be loaded first with load_string_table)
    pub fn string_table_ref(&self) -> &StringTable {
        &self.string_table
    }

    /// Get full alignment record by ID - O(1) random access
    pub fn get_alignment_record(&mut self, record_id: u64) -> io::Result<AlignmentRecord> {
        let offset = self.index.offsets[record_id as usize];
        self.get_alignment_record_at_offset(offset)
    }

    /// Get alignment record by file offset (for impg compatibility)
    pub fn get_alignment_record_at_offset(&mut self, offset: u64) -> io::Result<AlignmentRecord> {
        self.file.seek(SeekFrom::Start(offset))?;

        // Read with strategy from header
        let strategy = self.header.strategy()?;
        read_record(
            &mut self.file,
            strategy,
            self.header.tracepoint_type,
        )
    }

    /// Get tracepoints by record ID
    pub fn get_tracepoints(
        &mut self,
        record_id: u64,
    ) -> io::Result<(TracepointData, ComplexityMetric, u64)> {
        let tracepoint_offset = self.get_tracepoint_offset(record_id)?;
        self.get_tracepoints_at_offset(tracepoint_offset)
    }

    /// Get tracepoint offset by record ID
    /// Returns the byte offset where tracepoint data starts within the record
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

    /// Get tracepoints by tracepoint offset
    /// Seeks directly to tracepoint data within a record
    pub fn get_tracepoints_at_offset(
        &mut self,
        tracepoint_offset: u64,
    ) -> io::Result<(TracepointData, ComplexityMetric, u64)> {
        self.file.seek(SeekFrom::Start(tracepoint_offset))?;

        let tp_type = self.header.tracepoint_type;
        let complexity_metric = self.header.complexity_metric;
        let max_complexity = self.header.max_complexity;

        // Read tracepoints with strategy from header
        let strategy = self.header.strategy()?;
        let tracepoints = read_tracepoints(
            &mut self.file,
            tp_type,
            strategy,
        )?;

        Ok((tracepoints, complexity_metric, max_complexity))
    }

    /// Iterator over all records (sequential access)
    pub fn iter_records(&mut self) -> RecordIterator<'_> {
        RecordIterator {
            reader: self,
            current_id: 0,
        }
    }
}

// ============================================================================
// STANDALONE FUNCTIONS (NO BPAFREADER OVERHEAD)
// ============================================================================

/// Fastest access: decode standard tracepoints directly from file at offset.
/// Requires pre-computed offset and compression strategy.
#[inline]
pub fn read_standard_tracepoints_at_offset<R: Read + Seek>(
    file: &mut R,
    offset: u64,
    strategy: CompressionStrategy,
) -> io::Result<Vec<(usize, usize)>> {
    file.seek(SeekFrom::Start(offset))?;
    // Buffer small reads (varints + compressed block lengths)
    let mut buffered = BufReader::with_capacity(64, file);
    let num_items = read_varint(&mut buffered)? as usize;
    decode_standard_tracepoints(&mut buffered, num_items, strategy)
}

/// Fastest access: decode variable tracepoints directly from file at offset.
/// Requires pre-computed offset.
#[inline]
pub fn read_variable_tracepoints_at_offset<R: Read + Seek>(
    file: &mut R,
    offset: u64,
) -> io::Result<Vec<(usize, Option<usize>)>> {
    file.seek(SeekFrom::Start(offset))?;
    // Buffer small reads (varints + compressed block lengths)
    let mut buffered = BufReader::with_capacity(64, file);
    let num_items = read_varint(&mut buffered)? as usize;
    decode_variable_tracepoints(&mut buffered, num_items)
}

/// Fastest access: decode mixed tracepoints directly from file at offset.
/// Requires pre-computed offset.
#[inline]
pub fn read_mixed_tracepoints_at_offset<R: Read + Seek>(
    file: &mut R,
    offset: u64,
) -> io::Result<Vec<MixedRepresentation>> {
    file.seek(SeekFrom::Start(offset))?;
    // Buffer small reads (varints + compressed block lengths)
    let mut buffered = BufReader::with_capacity(64, file);
    let num_items = read_varint(&mut buffered)? as usize;
    decode_mixed_tracepoints(&mut buffered, num_items)
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
