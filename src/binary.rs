//! Binary I/O operations for TPA format

use log::info;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use tracepoints::{MixedRepresentation, TracepointData, TracepointType};

use crate::format::*;
use crate::utils::*;

// ============================================================================
// COMPRESSION LAYER ABSTRACTION
// ============================================================================

/// Compress data using the specified compression layer
fn compress_with_layer(data: &[u8], layer: CompressionLayer, level: i32) -> io::Result<Vec<u8>> {
    match layer {
        CompressionLayer::Zstd => zstd::encode_all(data, level)
            .map_err(|e| io::Error::other(format!("Zstd compression failed: {}", e))),
        CompressionLayer::Bgzip => {
            use bgzip::write::BGZFWriter;
            let mut compressed = Vec::new();
            {
                let compression = bgzip::Compression::new(level as u32).map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Invalid bgzip compression level: {}", e),
                    )
                })?;
                let mut writer = BGZFWriter::new(&mut compressed, compression);
                writer.write_all(data)?;
            } // BGZFWriter flushes on drop
            Ok(compressed)
        }
        CompressionLayer::Nocomp => Ok(data.to_vec()),
    }
}

/// Decompress data using the specified compression layer
fn decompress_with_layer(data: &[u8], layer: CompressionLayer) -> io::Result<Vec<u8>> {
    match layer {
        CompressionLayer::Zstd => zstd::decode_all(data)
            .map_err(|e| io::Error::other(format!("Zstd decompression failed: {}", e))),
        CompressionLayer::Bgzip => {
            use bgzip::read::BGZFReader;
            let mut reader = BGZFReader::new(data)
                .map_err(|e| io::Error::other(format!("BGZF decompression failed: {}", e)))?;
            let mut decompressed = Vec::new();
            reader.read_to_end(&mut decompressed)?;
            Ok(decompressed)
        }
        CompressionLayer::Nocomp => Ok(data.to_vec()),
    }
}

// ============================================================================
// DELTA ENCODING
// ============================================================================

/// Delta encode positions
#[inline]
fn delta_encode(values: &[u64]) -> Vec<i64> {
    let mut deltas = Vec::with_capacity(values.len());
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
    values.push(deltas[0] as u64);
    for i in 1..deltas.len() {
        values.push((values[i - 1] as i64 + deltas[i]) as u64);
    }
    values
}

/// Reconstruct original values from zigzag-encoded deltas.
#[inline]
fn reconstruct_from_zigzag_deltas(zigzagged: &[u64]) -> Vec<u64> {
    let mut vals = Vec::with_capacity(zigzagged.len());
    vals.push(decode_zigzag(zigzagged[0]) as u64);
    for &enc in zigzagged.iter().skip(1) {
        let delta = decode_zigzag(enc);
        let prev = *vals.last().unwrap() as i64;
        vals.push((prev + delta) as u64);
    }
    vals
}

// ============================================================================
// STRATEGY ANALYSIS
// ============================================================================

pub(crate) struct StrategyAnalyzer {
    first_states: Vec<StreamState>,
    second_states: Vec<StreamState>,
    layers: [CompressionLayer; 3],
    sample_limit: usize, // 0 = no limit
    processed_records: usize,
    processed_tracepoints: usize,
}

impl StrategyAnalyzer {
    /// Create analyzer. If `test_layers` is true, tests all 3 compression layers (Zstd, Bgzip, Nocomp).
    /// If false, only tests Nocomp (for all-records mode).
    pub fn new(zstd_level: i32, sample_limit: usize, test_layers: bool) -> Self {
        let strategies = CompressionStrategy::all(zstd_level);
        let layers = CompressionLayer::all();
        let num_layers_to_test = if test_layers { 3 } else { 1 };
        let first_states = strategies
            .iter()
            .cloned()
            .map(|s| StreamState::new(s, num_layers_to_test))
            .collect();
        let second_states = strategies
            .into_iter()
            .map(|s| StreamState::new(s, num_layers_to_test))
            .collect();

        Self {
            first_states,
            second_states,
            layers,
            sample_limit,
            processed_records: 0,
            processed_tracepoints: 0,
        }
    }

    pub fn analyze_record(&mut self, record: &CompactRecord) -> io::Result<()> {
        if self.sample_limit > 0 && self.processed_records >= self.sample_limit {
            return Ok(());
        }

        match &record.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                self.processed_records += 1;
                self.processed_tracepoints += tps.len();

                let mut first_vals = Vec::with_capacity(tps.len());
                let mut second_vals = Vec::with_capacity(tps.len());
                for &(first, second) in tps {
                    first_vals.push(first as u64);
                    second_vals.push(second as u64);
                }

                process_stream_states(
                    &mut self.first_states,
                    &self.layers,
                    &first_vals,
                    &second_vals,
                    "First",
                    |first, _second, strategy| encode_first_stream(first, strategy),
                );
                process_stream_states(
                    &mut self.second_states,
                    &self.layers,
                    &first_vals,
                    &second_vals,
                    "Second",
                    encode_second_stream,
                );
            }
            TracepointData::Variable(tps) => {
                self.processed_records += 1;
                self.processed_tracepoints += tps.len();

                // For Variable: first values are always present, second values are optional
                let first_vals: Vec<u64> = tps.iter().map(|(a, _)| *a as u64).collect();
                // Only include present second values for analysis
                let second_vals: Vec<u64> = tps
                    .iter()
                    .filter_map(|(_, b)| b.map(|v| v as u64))
                    .collect();

                process_stream_states(
                    &mut self.first_states,
                    &self.layers,
                    &first_vals,
                    &second_vals,
                    "First",
                    |first, _second, strategy| encode_first_stream(first, strategy),
                );
                // Only analyze second stream if there are any second values
                if !second_vals.is_empty() {
                    process_stream_states(
                        &mut self.second_states,
                        &self.layers,
                        &first_vals,
                        &second_vals,
                        "Second",
                        encode_second_stream,
                    );
                }
            }
            TracepointData::Mixed(items) => {
                self.processed_records += 1;
                self.processed_tracepoints += items.len();

                // For Mixed: separate tracepoint values from CIGAR op lengths
                // First stream: tracepoint first values + CIGAR lengths
                // Second stream: tracepoint second values only
                let (tp_first, tp_second): (Vec<u64>, Vec<u64>) = items
                    .iter()
                    .filter_map(|i| match i {
                        tracepoints::MixedRepresentation::Tracepoint(a, b) => {
                            Some((*a as u64, *b as u64))
                        }
                        _ => None,
                    })
                    .unzip();

                let cigar_lens: Vec<u64> = items
                    .iter()
                    .filter_map(|i| match i {
                        tracepoints::MixedRepresentation::CigarOp(len, _) => Some(*len as u64),
                        _ => None,
                    })
                    .collect();

                // Combine tp_first and cigar_lens for first stream analysis
                let mut first_vals = tp_first;
                first_vals.extend(cigar_lens);

                process_stream_states(
                    &mut self.first_states,
                    &self.layers,
                    &first_vals,
                    &tp_second,
                    "First",
                    |first, _second, strategy| encode_first_stream(first, strategy),
                );
                // Only analyze second stream if there are tracepoint second values
                if !tp_second.is_empty() {
                    process_stream_states(
                        &mut self.second_states,
                        &self.layers,
                        &first_vals,
                        &tp_second,
                        "Second",
                        encode_second_stream,
                    );
                }
            }
        }
        Ok(())
    }

    pub fn select_best(
        self,
    ) -> (
        CompressionStrategy,
        CompressionStrategy,
        CompressionLayer,
        CompressionLayer,
    ) {
        // Collect (strategy, layer, size) for each stream
        let score = |c: &(CompressionStrategy, CompressionLayer, usize)| {
            (c.2, layer_to_rank(c.1), strategy_to_rank(&c.0))
        };

        let first_best = self
            .first_states
            .into_iter()
            .flat_map(|state| {
                state
                    .totals
                    .into_iter()
                    .enumerate()
                    .map(move |(i, size)| (state.strategy, self.layers[i], size))
            })
            .min_by_key(score)
            .unwrap();

        let second_best = self
            .second_states
            .into_iter()
            .flat_map(|state| {
                state
                    .totals
                    .into_iter()
                    .enumerate()
                    .map(move |(i, size)| (state.strategy, self.layers[i], size))
            })
            .min_by_key(score)
            .unwrap();

        info!(
            "Automatic: {} [{}] → {} [{}]",
            first_best.0,
            first_best.1.as_str(),
            second_best.0,
            second_best.1.as_str()
        );

        (first_best.0, second_best.0, first_best.1, second_best.1)
    }
}

struct StreamState {
    strategy: CompressionStrategy,
    totals: Vec<usize>,
}

#[inline]
fn strategy_to_rank(strategy: &CompressionStrategy) -> u8 {
    // Lower = simpler/faster decode; tie-breaker when sizes match
    match strategy {
        CompressionStrategy::Raw(_) => 0,
        CompressionStrategy::ZigzagDelta(_) => 1,
        CompressionStrategy::TwoDimDelta(_) => 2,
        CompressionStrategy::RunLength(_) => 3,
        CompressionStrategy::BitPacked(_) => 4,
        CompressionStrategy::FrameOfReference(_) => 5,
        CompressionStrategy::HybridRLE(_) => 6,
        CompressionStrategy::XORDelta(_) => 7,
        CompressionStrategy::DeltaOfDelta(_) => 8,
        CompressionStrategy::Simple8bFull(_) => 9,
        CompressionStrategy::StreamVByte(_) => 10,
        CompressionStrategy::Dictionary(_) => 11,
        CompressionStrategy::Rice(_) => 12,
        CompressionStrategy::Huffman(_) => 13,
        CompressionStrategy::FastPFOR(_) => 14,
        CompressionStrategy::Cascaded(_) => 15,
        CompressionStrategy::SelectiveRLE(_) => 16,
        CompressionStrategy::LZ77(_) => 17,
        CompressionStrategy::Automatic(_, _) => 255,
    }
}

#[inline]
fn layer_to_rank(layer: CompressionLayer) -> u8 {
    // Lower = less CPU overhead; tie-breaker when sizes match
    match layer {
        CompressionLayer::Nocomp => 0,
        CompressionLayer::Zstd => 1,
        CompressionLayer::Bgzip => 2,
    }
}

impl StreamState {
    fn new(strategy: CompressionStrategy, layer_count: usize) -> Self {
        Self {
            strategy,
            totals: vec![0usize; layer_count],
        }
    }

    fn process_sample<F>(
        &mut self,
        layers: &[CompressionLayer; 3],
        first_vals: &[u64],
        second_vals: &[u64],
        encode_fn: &F,
        stream_label: &str,
    ) where
        F: Fn(&[u64], &[u64], &CompressionStrategy) -> io::Result<Vec<u8>>,
    {
        let encoded = encode_fn(first_vals, second_vals, &self.strategy).unwrap_or_else(|err| {
            panic!(
                "{}-stream strategy {} failed to encode sample: {}",
                stream_label, self.strategy, err
            )
        });

        let level = self.strategy.zstd_level();
        for (idx, layer) in layers.iter().take(self.totals.len()).enumerate() {
            let compressed =
                compress_with_layer(&encoded[..], *layer, level).unwrap_or_else(|err| {
                    panic!(
                        "{}-stream strategy {} with layer {:?} failed to compress sample: {}",
                        stream_label, self.strategy, layer, err
                    )
                });
            let len = compressed.len();
            let var_len = varint_size(len as u64) as usize;
            self.totals[idx] += len + var_len;
        }
    }
}

fn process_stream_states<F>(
    states: &mut [StreamState],
    layers: &[CompressionLayer; 3],
    first_vals: &[u64],
    second_vals: &[u64],
    stream_label: &'static str,
    encode_fn: F,
) where
    F: Fn(&[u64], &[u64], &CompressionStrategy) -> io::Result<Vec<u8>> + Sync,
{
    states.par_iter_mut().for_each(|state| {
        state.process_sample(layers, first_vals, second_vals, &encode_fn, stream_label);
    });
}

fn encode_first_stream(values: &[u64], strategy: &CompressionStrategy) -> io::Result<Vec<u8>> {
    match strategy {
        CompressionStrategy::Automatic(_, _) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Automatic strategy must be resolved before encoding",
        )),
        _ => encode_tracepoint_values(values, *strategy),
    }
}

fn encode_second_stream(
    first_vals: &[u64],
    second_vals: &[u64],
    strategy: &CompressionStrategy,
) -> io::Result<Vec<u8>> {
    match strategy {
        CompressionStrategy::Automatic(_, _) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Automatic strategy must be resolved before encoding",
        )),
        CompressionStrategy::TwoDimDelta(_) => {
            encode_2d_delta_second_values(first_vals, second_vals)
        }
        CompressionStrategy::HybridRLE(_) => encode_rle_values(second_vals),
        _ => encode_tracepoint_values(second_vals, *strategy),
    }
}

// ============================================================================
// PAF WRITING
// ============================================================================

pub fn write_paf_line_with_tracepoints<W: Write>(
    writer: &mut W,
    record: &CompactRecord,
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
    header: &TpaHeader,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
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
            first_strategy,
            second_strategy,
            header.tracepoint_type,
            header.first_layer,
            header.second_layer,
        )?;
        write_paf_line_with_tracepoints(&mut writer, &record, &string_table)?;
    }
    writer.flush()?;

    info!("Decompressed {} records", header.num_records);
    Ok(())
}

pub(crate) fn read_record<R: Read>(
    reader: &mut R,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    tp_type: TracepointType,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<CompactRecord> {
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

    let tracepoints = read_tracepoints(
        reader,
        tp_type,
        first_strategy,
        second_strategy,
        first_layer,
        second_layer,
    )?;

    let num_tags = read_varint(reader)? as usize;
    let mut tags = Vec::with_capacity(num_tags);
    for _ in 0..num_tags {
        tags.push(Tag::read(reader)?);
    }

    Ok(CompactRecord {
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

pub(crate) fn read_tracepoints<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<TracepointData> {
    let num_items = read_varint(reader)? as usize;
    if num_items == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Encountered tracepoint block with zero entries",
        ));
    }
    match tp_type {
        TracepointType::Standard | TracepointType::Fastga => {
            let tps = decode_standard_tracepoints(
                reader,
                num_items,
                first_strategy,
                second_strategy,
                first_layer,
                second_layer,
            )?;
            Ok(match tp_type {
                TracepointType::Standard => TracepointData::Standard(tps),
                _ => TracepointData::Fastga(tps),
            })
        }
        TracepointType::Variable => {
            let tps = decode_variable_tracepoints(
                reader,
                num_items,
                first_strategy,
                second_strategy,
                first_layer,
                second_layer,
            )?;
            Ok(TracepointData::Variable(tps))
        }
        TracepointType::Mixed => {
            let items = decode_mixed_tracepoints(
                reader,
                num_items,
                first_strategy,
                second_strategy,
                first_layer,
                second_layer,
            )?;
            Ok(TracepointData::Mixed(items))
        }
    }
}

// ============================================================================
// SHARED HELPERS (HEADER/FOOTER, TRACEPOINTS, SKIPS)
// ============================================================================

/// Read and validate header (checks footer for crash-safety), positions reader after header.
pub(crate) fn read_header<R: Read + Seek>(reader: &mut R) -> io::Result<TpaHeader> {
    let header = TpaHeader::read(reader)?;
    if header.version != TPA_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported format version: {}", header.version),
        ));
    }

    let after_header = reader.stream_position()?;
    let footer = TpaFooter::read_from_end(reader)?;
    footer.validate_against(&header)?;
    reader.seek(SeekFrom::Start(after_header))?;
    Ok(header)
}

/// Decode tracepoints from a known offset (seeks before reading).
pub(crate) fn read_tracepoints_at_offset<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    tp_type: TracepointType,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<TracepointData> {
    reader.seek(SeekFrom::Start(offset))?;
    read_tracepoints(
        reader,
        tp_type,
        first_strategy,
        second_strategy,
        first_layer,
        second_layer,
    )
}

/// Skip over a record without allocating.
pub(crate) fn skip_record<R: Read + Seek>(
    reader: &mut R,
    tp_type: TracepointType,
) -> io::Result<()> {
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
    let _num_items = read_varint(reader)?;

    // Helper to read length and skip that many bytes
    let skip_block = |reader: &mut R| -> io::Result<()> {
        let len = read_varint(reader)?;
        let len = i64::try_from(len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Compressed block too large to skip",
            )
        })?;
        reader.seek(SeekFrom::Current(len))?;
        Ok(())
    };

    match tp_type {
        TracepointType::Standard | TracepointType::Fastga => {
            // Format: [first_len][first_data][second_len][second_data]
            skip_block(reader)?;
            skip_block(reader)?;
        }
        TracepointType::Variable => {
            // Format: [presence_bitmap_len][presence_bitmap][first_len][first_data][second_len][second_data]
            skip_block(reader)?; // presence bitmap
            skip_block(reader)?; // first values
            skip_block(reader)?; // second values
        }
        TracepointType::Mixed => {
            // Format: [type_bitmap_len][type_bitmap][tp_first_len][tp_first][tp_second_len][tp_second]
            //         [cigar_lens_len][cigar_lens][cigar_ops_len][cigar_ops]
            skip_block(reader)?; // type bitmap
            skip_block(reader)?; // tp first values
            skip_block(reader)?; // tp second values
            skip_block(reader)?; // cigar lengths
            skip_block(reader)?; // cigar ops
        }
    }
    Ok(())
}

/// Skip over a record using only Read (no Seek) - for BGZF sequential scanning.
pub(crate) fn skip_record_sequential<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
) -> io::Result<()> {
    // Helper to skip n bytes by reading into a buffer
    fn skip_bytes<R: Read>(reader: &mut R, n: usize) -> io::Result<()> {
        io::copy(&mut reader.take(n as u64), &mut io::sink())?;
        Ok(())
    }

    // Helper to read length-prefixed block and skip its data
    fn skip_block<R: Read>(reader: &mut R) -> io::Result<()> {
        let len = read_varint(reader)? as usize;
        skip_bytes(reader, len)
    }

    read_varint(reader)?; // query_name_id
    read_varint(reader)?; // query_start
    read_varint(reader)?; // query_end
    skip_bytes(reader, 1)?; // strand
    read_varint(reader)?; // target_name_id
    read_varint(reader)?; // target_start
    read_varint(reader)?; // target_end
    read_varint(reader)?; // residue_matches
    read_varint(reader)?; // alignment_block_len
    skip_bytes(reader, 1)?; // mapping_quality

    // Skip tracepoints
    let _num_items = read_varint(reader)?;
    match tp_type {
        TracepointType::Standard | TracepointType::Fastga => {
            skip_block(reader)?;
            skip_block(reader)?;
        }
        TracepointType::Variable => {
            skip_block(reader)?; // presence bitmap
            skip_block(reader)?; // first values
            skip_block(reader)?; // second values
        }
        TracepointType::Mixed => {
            skip_block(reader)?; // type bitmap
            skip_block(reader)?; // tp first values
            skip_block(reader)?; // tp second values
            skip_block(reader)?; // cigar lengths
            skip_block(reader)?; // cigar ops
        }
    }

    // Skip tags
    let num_tags = read_varint(reader)? as usize;
    for _ in 0..num_tags {
        skip_bytes(reader, 2)?; // key
        let mut tag_type = [0u8; 1];
        reader.read_exact(&mut tag_type)?;
        match tag_type[0] {
            b'i' | b'f' => skip_bytes(reader, 4)?,
            b'Z' => {
                let len = read_varint(reader)? as usize;
                skip_bytes(reader, len)?;
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
        }
    }

    Ok(())
}

// ============================================================================
// BITSTREAM HELPERS
// ============================================================================

#[derive(Default)]
struct BitWriter {
    buffer: Vec<u8>,
    current: u8,
    bits_filled: u8,
}

impl BitWriter {
    fn write_bit(&mut self, bit: bool) {
        self.current <<= 1;
        if bit {
            self.current |= 1;
        }
        self.bits_filled += 1;
        if self.bits_filled == 8 {
            self.buffer.push(self.current);
            self.current = 0;
            self.bits_filled = 0;
        }
    }

    fn write_unary(&mut self, count: u64) {
        for _ in 0..count {
            self.write_bit(true);
        }
        self.write_bit(false);
    }

    fn write_bits(&mut self, value: u64, bits: u8) {
        if bits == 0 {
            return;
        }
        for shift in (0..bits).rev() {
            let bit = ((value >> (shift as u32)) & 1) != 0;
            self.write_bit(bit);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits_filled > 0 {
            self.current <<= 8 - self.bits_filled;
            self.buffer.push(self.current);
        }
        self.buffer
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_idx: usize,
    bits_left: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_idx: 0,
            bits_left: 0,
        }
    }

    fn read_bit(&mut self) -> io::Result<bool> {
        if self.bits_left == 0 {
            if self.byte_idx >= self.data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "bit reader exhausted",
                ));
            }
            self.bits_left = 8;
        }
        let byte = self.data[self.byte_idx];
        let bit = (byte & (1 << (self.bits_left - 1))) != 0;
        self.bits_left -= 1;
        if self.bits_left == 0 {
            self.byte_idx += 1;
        }
        Ok(bit)
    }

    fn read_bits(&mut self, bits: u8) -> io::Result<u64> {
        let mut value = 0u64;
        for _ in 0..bits {
            value = (value << 1) | (self.read_bit()? as u64);
        }
        Ok(value)
    }

    fn read_unary(&mut self) -> io::Result<u64> {
        let mut count = 0u64;
        loop {
            let bit = self.read_bit()?;
            if bit {
                count += 1;
            } else {
                break;
            }
        }
        Ok(count)
    }
}

// ============================================================================
// RICE / GOLOMB CODING
// ============================================================================

fn choose_rice_k(values: &[u64]) -> u8 {
    if values.is_empty() {
        return 0;
    }
    let mut best_k = 0;
    let mut best_bits = u64::MAX;
    for k in 0..=16 {
        let mut bits = 0u64;
        for &v in values {
            bits += (v >> (k as u32)) + 1 + k as u64; // unary quotient + stop bit + remainder
        }
        if bits < best_bits {
            best_bits = bits;
            best_k = k;
        }
    }
    best_k
}

fn encode_rice_values(values: &[u64]) -> io::Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let k = choose_rice_k(values);
    let mut writer = BitWriter::default();
    let mask = if k == 0 { 0 } else { (1u64 << (k as u32)) - 1 };
    for &v in values {
        let q = v >> (k as u32);
        let r = v & mask;
        writer.write_unary(q);
        writer.write_bits(r, k);
    }
    let mut out = Vec::with_capacity(1 + values.len() / 2);
    out.push(k);
    out.extend_from_slice(&writer.finish());
    Ok(out)
}

fn decode_rice_values(buf: &[u8], num_items: usize) -> io::Result<Vec<u64>> {
    if num_items == 0 {
        return Ok(Vec::new());
    }
    if buf.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "missing Rice header",
        ));
    }
    let k = buf[0];
    let mut reader = BitReader::new(&buf[1..]);
    let mut values = Vec::with_capacity(num_items);
    for _ in 0..num_items {
        let q = reader.read_unary()?;
        let r = if k > 0 { reader.read_bits(k)? } else { 0 };
        values.push((q << (k as u32)) | r);
    }
    Ok(values)
}

// ============================================================================
// HUFFMAN CODING
// ============================================================================

#[derive(Clone)]
struct HuffmanCode {
    value: u64,
    len: u8,
    code: u64,
}

#[derive(Clone)]
enum HuffmanNode {
    Leaf(u64),
    Internal(Box<HuffmanNode>, Box<HuffmanNode>),
}

fn build_huffman_tree(freqs: &[(u64, usize)]) -> HuffmanNode {
    let mut nodes: Vec<(usize, HuffmanNode)> = freqs
        .iter()
        .map(|(v, f)| (*f, HuffmanNode::Leaf(*v)))
        .collect();
    while nodes.len() > 1 {
        nodes.sort_by_key(|(f, _)| *f);
        let (f1, n1) = nodes.remove(0);
        let (f2, n2) = nodes.remove(0);
        nodes.push((f1 + f2, HuffmanNode::Internal(Box::new(n1), Box::new(n2))));
    }
    nodes.pop().map(|(_, n)| n).unwrap()
}

fn gather_lengths(node: &HuffmanNode, depth: u8, lens: &mut Vec<(u64, u8)>) {
    match node {
        HuffmanNode::Leaf(v) => {
            // Ensure single-symbol trees still get a length of 1
            lens.push((*v, depth.max(1)));
        }
        HuffmanNode::Internal(left, right) => {
            gather_lengths(left, depth + 1, lens);
            gather_lengths(right, depth + 1, lens);
        }
    }
}

fn build_canonical_codes(lengths: &[(u64, u8)]) -> io::Result<Vec<HuffmanCode>> {
    let mut entries = lengths.to_vec();
    entries.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut codes = Vec::with_capacity(entries.len());
    let mut code = 0u64;
    let mut prev_len = 0u8;
    for (value, len) in entries {
        if len == 0 || len > 63 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid Huffman code length: {}", len),
            ));
        }
        if prev_len > 0 {
            code <<= (len - prev_len) as u32;
        }
        codes.push(HuffmanCode { value, len, code });
        code += 1;
        prev_len = len;
    }
    Ok(codes)
}

fn huffman_encode(values: &[u64]) -> io::Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    use std::collections::HashMap;
    let mut freq: HashMap<u64, usize> = HashMap::new();
    for &v in values {
        *freq.entry(v).or_insert(0) += 1;
    }
    let mut freq_vec: Vec<(u64, usize)> = freq.into_iter().collect();
    freq_vec.sort_by_key(|(_, c)| *c);

    let tree = build_huffman_tree(&freq_vec);
    let mut lengths = Vec::with_capacity(freq_vec.len());
    gather_lengths(&tree, 0, &mut lengths);
    let codes = build_canonical_codes(&lengths)?;

    let mut code_lookup = std::collections::HashMap::new();
    for c in &codes {
        code_lookup.insert(c.value, (c.code, c.len));
    }

    // Serialize header: symbol count + (value, length) pairs in canonical order
    let mut out = Vec::new();
    write_varint(&mut out, codes.len() as u64)?;
    let mut lengths_sorted: Vec<(u64, u8)> = codes.iter().map(|c| (c.value, c.len)).collect();
    lengths_sorted.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    for (value, len) in lengths_sorted.iter() {
        write_varint(&mut out, *value)?;
        out.push(*len);
    }

    // Encode payload
    let mut writer = BitWriter::default();
    for &v in values {
        let (code, len) = code_lookup[&v];
        writer.write_bits(code, len);
    }
    out.extend_from_slice(&writer.finish());
    Ok(out)
}

fn huffman_decode(buf: &[u8], num_items: usize) -> io::Result<Vec<u64>> {
    if num_items == 0 {
        return Ok(Vec::new());
    }
    let mut reader = buf;
    let symbol_count = read_varint(&mut reader)? as usize;
    if symbol_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Huffman stream has zero symbols",
        ));
    }
    let mut lengths: Vec<(u64, u8)> = Vec::with_capacity(symbol_count);
    for _ in 0..symbol_count {
        let value = read_varint(&mut reader)?;
        if reader.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Huffman header truncated",
            ));
        }
        let len = reader[0];
        reader = &reader[1..];
        lengths.push((value, len));
    }
    let codes = build_canonical_codes(&lengths)?;

    #[derive(Default)]
    struct DecodeNode {
        left: Option<Box<DecodeNode>>,
        right: Option<Box<DecodeNode>>,
        value: Option<u64>,
    }

    let mut root = DecodeNode::default();
    for c in &codes {
        let mut node = &mut root;
        for shift in (0..c.len).rev() {
            let bit = (c.code >> (shift as u32)) & 1;
            if bit == 0 {
                if node.left.is_none() {
                    node.left = Some(Box::new(DecodeNode::default()));
                }
                node = node.left.as_mut().unwrap();
            } else {
                if node.right.is_none() {
                    node.right = Some(Box::new(DecodeNode::default()));
                }
                node = node.right.as_mut().unwrap();
            }
        }
        node.value = Some(c.value);
    }

    let mut bit_reader = BitReader::new(reader);
    let mut output = Vec::with_capacity(num_items);
    for _ in 0..num_items {
        let mut node = &root;
        loop {
            if let Some(v) = node.value {
                output.push(v);
                break;
            }
            let bit = bit_reader.read_bit()?;
            node = if !bit {
                node.left.as_deref().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid Huffman bitstream")
                })?
            } else {
                node.right.as_deref().ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid Huffman bitstream")
                })?
            };
        }
    }

    Ok(output)
}

// ============================================================================
// LZ77 ENCODING/DECODING
// ============================================================================

/// LZ77 token markers
const LZ77_LITERAL: u8 = 0;
const LZ77_MATCH: u8 = 1;

/// Minimum match length for LZ77 (must be >= 3 for compression benefit)
const LZ77_MIN_MATCH: usize = 3;

/// Default window size for LZ77 (256 values)
const LZ77_WINDOW_SIZE: usize = 256;

/// Find the longest match in the sliding window
#[inline]
fn lz77_find_longest_match(values: &[u64], pos: usize, window_size: usize) -> (usize, usize) {
    let start = pos.saturating_sub(window_size);
    let mut best_offset = 0;
    let mut best_length = 0;

    // Search through the window for matches
    for offset in 1..=(pos - start) {
        let match_start = pos - offset;
        let mut len = 0;

        // Match can extend beyond original match position (overlapping match)
        while pos + len < values.len() {
            let src_idx = match_start + (len % offset);
            if values[src_idx] != values[pos + len] {
                break;
            }
            len += 1;
        }

        if len >= LZ77_MIN_MATCH && len > best_length {
            best_offset = offset;
            best_length = len;
        }
    }

    (best_offset, best_length)
}

/// Encode values using LZ77-style sequence matching
fn lz77_encode(vals: &[u64]) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(vals.len() * 2);
    let mut i = 0;
    while i < vals.len() {
        let (best_offset, best_length) = lz77_find_longest_match(vals, i, LZ77_WINDOW_SIZE);

        if best_length >= LZ77_MIN_MATCH {
            // Emit match token: marker + offset + length
            buf.push(LZ77_MATCH);
            write_varint(&mut buf, best_offset as u64)?;
            write_varint(&mut buf, best_length as u64)?;
            i += best_length;
        } else {
            // Emit literal token: marker + value
            buf.push(LZ77_LITERAL);
            write_varint(&mut buf, vals[i])?;
            i += 1;
        }
    }

    Ok(buf)
}

/// Decode LZ77-encoded values
fn lz77_decode(buf: &[u8], num_items: usize) -> io::Result<Vec<u64>> {
    if num_items == 0 {
        return Ok(Vec::new());
    }

    let mut reader = buf;
    let mut values = Vec::with_capacity(num_items);

    while values.len() < num_items {
        if reader.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "LZ77 decode: unexpected end of data",
            ));
        }

        let marker = reader[0];
        reader = &reader[1..];

        if marker == LZ77_LITERAL {
            // Literal: read single value
            let value = read_varint(&mut reader)?;
            values.push(value);
        } else if marker == LZ77_MATCH {
            // Match: read offset and length, then copy from history
            let offset = read_varint(&mut reader)? as usize;
            let length = read_varint(&mut reader)? as usize;

            if offset == 0 || offset > values.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "LZ77 decode: invalid offset {} (history size {})",
                        offset,
                        values.len()
                    ),
                ));
            }

            let start = values.len() - offset;
            // Copy values, handling overlapping matches
            for j in 0..length {
                let src_idx = start + (j % offset);
                values.push(values[src_idx]);
            }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("LZ77 decode: invalid marker byte {}", marker),
            ));
        }
    }

    Ok(values)
}

/// Encode tracepoint values based on compression strategy
#[inline]
fn encode_tracepoint_values(vals: &[u64], strategy: CompressionStrategy) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(vals.len() * 2);
    match strategy {
        CompressionStrategy::Raw(_) | CompressionStrategy::HybridRLE(_) => {
            // Raw varints (HybridRLE: first values as varint, second uses RLE in encode_second_stream)
            for &val in vals {
                write_varint(&mut buf, val)?;
            }
        }
        CompressionStrategy::ZigzagDelta(_) | CompressionStrategy::TwoDimDelta(_) => {
            // Zigzag + delta encode (for first values; second values handled specially)
            let deltas = delta_encode(vals);
            for &val in &deltas {
                let zigzag = encode_zigzag(val);
                write_varint(&mut buf, zigzag)?;
            }
        }
        CompressionStrategy::RunLength(_) => {
            return encode_rle_values(vals);
        }
        CompressionStrategy::BitPacked(_) => {
            // Bit packing: find max value, determine bits needed, pack tightly
            if vals.is_empty() {
                return Ok(buf);
            }

            let max_val = *vals.iter().max().unwrap();
            let bits_needed = if max_val == 0 {
                1
            } else {
                (64 - max_val.leading_zeros()) as u8
            };

            // Store bits_needed (1 byte)
            buf.push(bits_needed);

            // Bit-packing using BitWriter
            let mut writer = BitWriter::default();
            for &val in vals {
                writer.write_bits(val, bits_needed);
            }
            buf.extend_from_slice(&writer.finish());
        }
        CompressionStrategy::DeltaOfDelta(_) => {
            // Delta-of-delta (Gorilla-style): delta twice for regularly spaced coordinates
            let deltas = delta_encode(vals);
            let delta_deltas = delta_encode(&deltas.iter().map(|&x| x as u64).collect::<Vec<_>>());
            for &val in &delta_deltas {
                let zigzag = encode_zigzag(val);
                write_varint(&mut buf, zigzag)?;
            }
        }
        CompressionStrategy::FrameOfReference(_) => {
            // Frame-of-Reference: min + bit-packed offsets
            let min_val = *vals.iter().min().unwrap();
            let max_offset = vals.iter().map(|&v| v - min_val).max().unwrap();
            let bits_needed = if max_offset == 0 {
                1
            } else {
                64 - max_offset.leading_zeros()
            };

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
            // Dictionary coding: build dictionary (capped at 256), encode indices as u8
            if vals.is_empty() {
                return Ok(buf);
            }

            const MAX_DICT_SIZE: usize = 256;

            use std::collections::HashMap;
            let mut dict: Vec<u64> = Vec::new();
            let mut dict_map: HashMap<u64, u32> = HashMap::new();
            let mut indices: Vec<u8> = Vec::new();
            let mut exceeded = false;

            // Build dictionary with cap
            for &val in vals {
                if let std::collections::hash_map::Entry::Vacant(e) = dict_map.entry(val) {
                    if dict.len() >= MAX_DICT_SIZE {
                        exceeded = true;
                        break;
                    }
                    e.insert(dict.len() as u32);
                    dict.push(val);
                }
                if !exceeded {
                    indices.push(*dict_map.get(&val).unwrap() as u8);
                }
            }

            if exceeded {
                // Fallback: mode byte 0 + raw varints
                buf.push(0); // Mode: raw fallback
                for &val in vals {
                    write_varint(&mut buf, val)?;
                }
            } else {
                // Dictionary mode: mode byte 1 + dict_size + dict + u8 indices
                buf.push(1); // Mode: dictionary
                buf.push(dict.len() as u8); // Dict size (1-256, stored as 0-255)
                for &val in &dict {
                    write_varint(&mut buf, val)?;
                }
                // Indices as raw u8 bytes (compact)
                buf.extend_from_slice(&indices);
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
                let bytes_needed = if val == 0 {
                    1
                } else {
                    (64 - val.leading_zeros()).div_ceil(8) as usize
                };
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
        CompressionStrategy::Rice(_) => {
            // Rice/Golomb on zigzagged deltas
            let deltas = delta_encode(vals);
            let zigzagged: Vec<u64> = deltas.iter().map(|&v| encode_zigzag(v)).collect();
            buf = encode_rice_values(&zigzagged)?;
        }
        CompressionStrategy::Huffman(_) => {
            // ANS (rANS) on zigzagged deltas - replaces Huffman for better compression
            let deltas = delta_encode(vals);
            let zigzagged: Vec<u64> = deltas.iter().map(|&v| encode_zigzag(v)).collect();
            buf = huffman_encode(&zigzagged)?;
        }
        CompressionStrategy::LZ77(_) => {
            // LZ77-style sequence matching: find repeated sequences
            buf = lz77_encode(vals)?;
        }
        CompressionStrategy::Automatic(_, _) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Automatic strategy must be resolved before encoding",
            ));
        }
    }
    Ok(buf)
}

/// Encode values using RLE (value, run_length) pairs
#[inline]
fn encode_rle_values(vals: &[u64]) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(vals.len() * 2);
    if !vals.is_empty() {
        let mut run_val = vals[0];
        let mut run_len = 1u64;
        for &val in &vals[1..] {
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
    Ok(buf)
}

/// Encode second values for 2D-Delta strategy (as delta from first values)
#[inline]
fn encode_2d_delta_second_values(first_vals: &[u64], second_vals: &[u64]) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(second_vals.len() * 2);
    for (f, s) in first_vals.iter().zip(second_vals.iter()) {
        let diff = *s as i64 - *f as i64;
        write_varint(&mut buf, encode_zigzag(diff))?;
    }
    Ok(buf)
}

/// Decode tracepoint values based on compression strategy
#[inline]
fn decode_tracepoint_values(
    buf: &[u8],
    num_items: usize,
    strategy: CompressionStrategy,
) -> io::Result<Vec<u64>> {
    let mut reader = buf;
    match strategy {
        CompressionStrategy::Raw(_) | CompressionStrategy::HybridRLE(_) => {
            // Raw varints (HybridRLE: first values as varint, second uses RLE separately)
            let mut vals = Vec::with_capacity(num_items);
            for _ in 0..num_items {
                vals.push(read_varint(&mut reader)?);
            }
            Ok(vals)
        }
        CompressionStrategy::ZigzagDelta(_) | CompressionStrategy::TwoDimDelta(_) => {
            // Zigzag + delta decode
            let mut vals = Vec::with_capacity(num_items);
            if num_items == 0 {
                return Ok(vals);
            }

            // First value
            let zigzag = read_varint(&mut reader)?;
            let first = decode_zigzag(zigzag);
            vals.push(first as u64);

            // Remaining values: zigzag decode + delta accumulate in one pass
            for _ in 1..num_items {
                let zigzag = read_varint(&mut reader)?;
                let delta = decode_zigzag(zigzag);
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
            // Bit packing: read bits_needed, then unpack using BitReader
            if num_items == 0 {
                return Ok(Vec::new());
            }

            let bits_needed = reader[0];
            reader = &reader[1..];

            let mut bit_reader = BitReader::new(reader);
            let mut vals = Vec::with_capacity(num_items);
            for _ in 0..num_items {
                vals.push(bit_reader.read_bits(bits_needed)?);
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
                delta_deltas.push(decode_zigzag(zigzag));
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
                    let offset =
                        u32::from_le_bytes([reader[0], reader[1], reader[2], reader[3]]) as u64;
                    vals.push(min_val + offset);
                    reader = &reader[4..];
                }
            } else {
                for _ in 0..num_items {
                    let offset = u64::from_le_bytes([
                        reader[0], reader[1], reader[2], reader[3], reader[4], reader[5],
                        reader[6], reader[7],
                    ]);
                    vals.push(min_val + offset);
                    reader = &reader[8..];
                }
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
            // Dictionary decode with mode byte
            if num_items == 0 {
                return Ok(Vec::new());
            }

            let mode = reader[0];
            reader = &reader[1..];

            if mode == 0 {
                // Raw fallback mode
                let mut vals = Vec::with_capacity(num_items);
                for _ in 0..num_items {
                    vals.push(read_varint(&mut reader)?);
                }
                Ok(vals)
            } else {
                // Dictionary mode
                let dict_size = reader[0] as usize;
                reader = &reader[1..];

                // Handle special case: dict_size 0 means 256
                let dict_size = if dict_size == 0 { 256 } else { dict_size };

                let mut dict = Vec::with_capacity(dict_size);
                for _ in 0..dict_size {
                    dict.push(read_varint(&mut reader)?);
                }

                // Read u8 indices
                let mut vals = Vec::with_capacity(num_items);
                for _ in 0..num_items {
                    let idx = reader[0] as usize;
                    reader = &reader[1..];
                    if idx >= dict.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "Dictionary index {} out of bounds (size {})",
                                idx,
                                dict.len()
                            ),
                        ));
                    }
                    vals.push(dict[idx]);
                }
                Ok(vals)
            }
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
            crate::hybrids::decode_fastpfor(buf, num_items)
        }
        CompressionStrategy::Cascaded(_) => {
            // Cascaded decode
            crate::hybrids::decode_cascaded(buf, num_items)
        }
        CompressionStrategy::Simple8bFull(_) => {
            // Simple8b-RLE Full decode
            // Convert bytes back to u64 words
            let mut words = Vec::new();
            let mut i = 0;
            while i + 8 <= buf.len() {
                let word = u64::from_le_bytes([
                    buf[i],
                    buf[i + 1],
                    buf[i + 2],
                    buf[i + 3],
                    buf[i + 4],
                    buf[i + 5],
                    buf[i + 6],
                    buf[i + 7],
                ]);
                words.push(word);
                i += 8;
            }
            crate::hybrids::decode_simple8b_full(&words, num_items)
        }
        CompressionStrategy::SelectiveRLE(_) => {
            // Selective RLE decode
            crate::hybrids::decode_selective_rle(buf)
        }
        CompressionStrategy::Rice(_) => {
            let zigzagged = decode_rice_values(buf, num_items)?;
            Ok(reconstruct_from_zigzag_deltas(&zigzagged))
        }
        CompressionStrategy::Huffman(_) => {
            let zigzagged = huffman_decode(buf, num_items)?;
            Ok(reconstruct_from_zigzag_deltas(&zigzagged))
        }
        CompressionStrategy::LZ77(_) => {
            // LZ77-style sequence matching decode
            lz77_decode(buf, num_items)
        }
        CompressionStrategy::Automatic(_, _) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Automatic strategy must be resolved before decoding",
        )),
    }
}

/// Decode second values for 2D-Delta strategy (from diff values and first values)
#[inline]
fn decode_2d_delta_second_values(buf: &[u8], first_vals: &[u64]) -> io::Result<Vec<u64>> {
    let mut reader = buf;
    let mut second_vals = Vec::with_capacity(first_vals.len());

    // Decode zigzag-encoded differences and add to first values
    for &first in first_vals {
        let zigzag = read_varint(&mut reader)?;
        let diff = decode_zigzag(zigzag);
        let second = safe_signed_add(first as i64, diff)?;
        second_vals.push(second as u64);
    }

    // Ensure we consumed exactly what we expected; leftover bytes indicate corruption
    if !reader.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("2d-delta decoding left {} unused bytes", reader.len()),
        ));
    }

    Ok(second_vals)
}

#[inline]
fn safe_signed_add(base: i64, delta: i64) -> io::Result<i64> {
    let val = base.checked_add(delta).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tracepoint delta overflow: base={base} delta={delta}"),
        )
    })?;
    if val < 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tracepoint delta underflow: base={base} delta={delta}"),
        ));
    }
    Ok(val)
}

/// Standard/Fastga tracepoint decoding
#[inline(always)]
pub(crate) fn decode_standard_tracepoints<R: Read>(
    reader: &mut R,
    num_items: usize,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<Vec<(usize, usize)>> {
    // Read compressed blocks
    let first_len = read_varint(reader)? as usize;
    let mut first_compressed = vec![0u8; first_len];
    reader.read_exact(&mut first_compressed)?;

    let second_len = read_varint(reader)? as usize;
    let mut second_compressed = vec![0u8; second_len];
    reader.read_exact(&mut second_compressed)?;

    // Decompress using explicit layers
    let first_buf = decompress_with_layer(&first_compressed[..], first_layer)?;
    let second_buf = decompress_with_layer(&second_compressed[..], second_layer)?;

    // Decode directly to (usize, usize) tuples
    let mut tps = Vec::with_capacity(num_items);

    let first_vals_u64 = decode_tracepoint_values(&first_buf, num_items, first_strategy)?;
    let first_vals: Vec<usize> = first_vals_u64.iter().map(|v| *v as usize).collect();

    let second_vals_u64: Vec<u64> = match second_strategy {
        CompressionStrategy::TwoDimDelta(_) => {
            decode_2d_delta_second_values(&second_buf, &first_vals_u64)?
        }
        CompressionStrategy::HybridRLE(_) => {
            let mut reader = &second_buf[..];
            let mut vals = Vec::with_capacity(num_items);
            while vals.len() < num_items {
                let value = read_varint(&mut reader)? as u64;
                let run_len = read_varint(&mut reader)? as usize;
                for _ in 0..run_len {
                    vals.push(value);
                    if vals.len() >= num_items {
                        break;
                    }
                }
            }
            if vals.len() != num_items {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "hybrid-rle decode length mismatch: expected {} got {}",
                        num_items,
                        vals.len()
                    ),
                ));
            }
            if !reader.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("hybrid-rle residual bytes after decode: {}", reader.len()),
                ));
            }
            vals
        }
        CompressionStrategy::Automatic(_, _) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Automatic strategy must be resolved before decoding",
            ))
        }
        other => decode_tracepoint_values(&second_buf, num_items, other)?,
    };

    if second_vals_u64.len() != num_items || first_vals.len() != num_items {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tracepoint decode length mismatch",
        ));
    }

    for (a, b) in first_vals.into_iter().zip(second_vals_u64.into_iter()) {
        tps.push((a, b as usize));
    }

    Ok(tps)
}

/// Variable tracepoint decoding with strategy support
#[inline(always)]
pub(crate) fn decode_variable_tracepoints<R: Read>(
    reader: &mut R,
    num_items: usize,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<Vec<(usize, Option<usize>)>> {
    // Read presence bitmap
    let bitmap_len = read_varint(reader)? as usize;
    let mut presence_bytes = vec![0u8; bitmap_len];
    reader.read_exact(&mut presence_bytes)?;
    let presence = crate::utils::unpack_bitmap(&presence_bytes, num_items);

    // Read first stream
    let first_len = read_varint(reader)? as usize;
    let mut first_compressed = vec![0u8; first_len];
    reader.read_exact(&mut first_compressed)?;
    let first_buf = decompress_with_layer(&first_compressed, first_layer)?;
    let first_vals = decode_tracepoint_values(&first_buf, num_items, first_strategy)?;

    // Count how many have second values
    let second_count = presence.iter().filter(|&&b| b).count();

    // Read second stream
    let second_len = read_varint(reader)? as usize;
    let second_vals = if second_len == 0 || second_count == 0 {
        Vec::new()
    } else {
        let mut second_compressed = vec![0u8; second_len];
        reader.read_exact(&mut second_compressed)?;
        let second_buf = decompress_with_layer(&second_compressed, second_layer)?;
        decode_tracepoint_values(&second_buf, second_count, second_strategy)?
    };

    // Reconstruct
    let mut second_iter = second_vals.into_iter();
    let tps: Vec<(usize, Option<usize>)> = first_vals
        .into_iter()
        .zip(presence)
        .map(|(first, has_second)| {
            let second = if has_second {
                second_iter.next().map(|v| v as usize)
            } else {
                None
            };
            (first as usize, second)
        })
        .collect();

    Ok(tps)
}

/// Mixed tracepoint decoding with strategy support
#[inline(always)]
pub(crate) fn decode_mixed_tracepoints<R: Read>(
    reader: &mut R,
    num_items: usize,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<Vec<MixedRepresentation>> {
    // Read type bitmap
    let bitmap_len = read_varint(reader)? as usize;
    let mut type_bytes = vec![0u8; bitmap_len];
    reader.read_exact(&mut type_bytes)?;
    let types = crate::utils::unpack_bitmap(&type_bytes, num_items);

    // Count tracepoints and CIGARs
    let cigar_count = types.iter().filter(|&&is_cigar| is_cigar).count();
    let tp_count = num_items - cigar_count;

    // Read tracepoint first stream
    let tp_first_len = read_varint(reader)? as usize;
    let tp_first_vals = if tp_first_len == 0 || tp_count == 0 {
        Vec::new()
    } else {
        let mut tp_first_compressed = vec![0u8; tp_first_len];
        reader.read_exact(&mut tp_first_compressed)?;
        let tp_first_buf = decompress_with_layer(&tp_first_compressed, first_layer)?;
        decode_tracepoint_values(&tp_first_buf, tp_count, first_strategy)?
    };

    // Read tracepoint second stream
    let tp_second_len = read_varint(reader)? as usize;
    let tp_second_vals = if tp_second_len == 0 || tp_count == 0 {
        Vec::new()
    } else {
        let mut tp_second_compressed = vec![0u8; tp_second_len];
        reader.read_exact(&mut tp_second_compressed)?;
        let tp_second_buf = decompress_with_layer(&tp_second_compressed, second_layer)?;
        decode_tracepoint_values(&tp_second_buf, tp_count, second_strategy)?
    };

    // Read CIGAR lengths stream
    let cigar_len_len = read_varint(reader)? as usize;
    let cigar_lens = if cigar_len_len == 0 || cigar_count == 0 {
        Vec::new()
    } else {
        let mut cigar_len_compressed = vec![0u8; cigar_len_len];
        reader.read_exact(&mut cigar_len_compressed)?;
        let cigar_len_buf = decompress_with_layer(&cigar_len_compressed, first_layer)?;
        decode_tracepoint_values(&cigar_len_buf, cigar_count, first_strategy)?
    };

    // Read CIGAR ops (raw bytes)
    let cigar_ops_len = read_varint(reader)? as usize;
    let mut cigar_ops = vec![0u8; cigar_ops_len];
    if cigar_ops_len > 0 {
        reader.read_exact(&mut cigar_ops)?;
    }

    // Reconstruct items in original order
    let mut tp_iter = tp_first_vals.into_iter().zip(tp_second_vals);
    let mut cigar_iter = cigar_lens.into_iter().zip(cigar_ops);

    let items: Vec<MixedRepresentation> = types
        .into_iter()
        .map(|is_cigar| {
            if is_cigar {
                let (len, op) = cigar_iter.next().unwrap_or((0, b'?'));
                MixedRepresentation::CigarOp(len as usize, op as char)
            } else {
                let (a, b) = tp_iter.next().unwrap_or((0, 0));
                MixedRepresentation::Tracepoint(a as usize, b as usize)
            }
        })
        .collect();

    Ok(items)
}

// ============================================================================
// ALIGNMENT RECORD I/O
// ============================================================================

impl CompactRecord {
    pub(crate) fn write<W: Write>(
        &self,
        writer: &mut W,
        first_strategy: CompressionStrategy,
        second_strategy: CompressionStrategy,
        first_layer: CompressionLayer,
        second_layer: CompressionLayer,
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
        self.write_tracepoints(
            writer,
            first_strategy,
            second_strategy,
            first_layer,
            second_layer,
        )?;
        write_varint(writer, self.tags.len() as u64)?;
        for tag in &self.tags {
            tag.write(writer)?;
        }
        Ok(())
    }

    fn write_tracepoints<W: Write>(
        &self,
        writer: &mut W,
        first_strategy: CompressionStrategy,
        second_strategy: CompressionStrategy,
        first_layer: CompressionLayer,
        second_layer: CompressionLayer,
    ) -> io::Result<()> {
        match &self.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                write_varint(writer, tps.len() as u64)?;
                let (first_vals, second_vals): (Vec<u64>, Vec<u64>) =
                    tps.iter().map(|(a, b)| (*a as u64, *b as u64)).unzip();

                let first_val_buf = encode_tracepoint_values(&first_vals, first_strategy)?;
                let second_val_buf = match &second_strategy {
                    CompressionStrategy::TwoDimDelta(_) => {
                        encode_2d_delta_second_values(&first_vals, &second_vals)?
                    }
                    CompressionStrategy::HybridRLE(_) => encode_rle_values(&second_vals)?,
                    _ => encode_tracepoint_values(&second_vals, second_strategy)?,
                };

                // Use the explicitly passed layer parameter
                let first_compressed = compress_with_layer(
                    &first_val_buf[..],
                    first_layer,
                    first_strategy.zstd_level(),
                )?;
                let second_compressed = compress_with_layer(
                    &second_val_buf[..],
                    second_layer,
                    second_strategy.zstd_level(),
                )?;

                write_varint(writer, first_compressed.len() as u64)?;
                writer.write_all(&first_compressed)?;
                write_varint(writer, second_compressed.len() as u64)?;
                writer.write_all(&second_compressed)?;
            }
            TracepointData::Variable(tps) => {
                write_varint(writer, tps.len() as u64)?;

                // Build presence bitmap (1 bit per item: has_second)
                let presence: Vec<bool> = tps.iter().map(|(_, b)| b.is_some()).collect();
                let presence_bytes = crate::utils::pack_bitmap(&presence);
                write_varint(writer, presence_bytes.len() as u64)?;
                writer.write_all(&presence_bytes)?;

                // Collect first values (all items have first)
                let first_vals: Vec<u64> = tps.iter().map(|(a, _)| *a as u64).collect();

                // Collect second values (only present ones)
                let second_vals: Vec<u64> = tps
                    .iter()
                    .filter_map(|(_, b)| b.map(|v| v as u64))
                    .collect();

                // Encode with strategies
                let first_buf = encode_tracepoint_values(&first_vals, first_strategy)?;
                let second_buf = if second_vals.is_empty() {
                    Vec::new()
                } else {
                    encode_tracepoint_values(&second_vals, second_strategy)?
                };

                // Compress with layers
                let first_compressed =
                    compress_with_layer(&first_buf, first_layer, first_strategy.zstd_level())?;
                let second_compressed = if second_buf.is_empty() {
                    Vec::new()
                } else {
                    compress_with_layer(&second_buf, second_layer, second_strategy.zstd_level())?
                };

                write_varint(writer, first_compressed.len() as u64)?;
                writer.write_all(&first_compressed)?;
                write_varint(writer, second_compressed.len() as u64)?;
                writer.write_all(&second_compressed)?;
            }
            TracepointData::Mixed(items) => {
                write_varint(writer, items.len() as u64)?;

                // Build type bitmap (0=Tracepoint, 1=CigarOp)
                let types: Vec<bool> = items
                    .iter()
                    .map(|i| matches!(i, MixedRepresentation::CigarOp(_, _)))
                    .collect();
                let type_bytes = crate::utils::pack_bitmap(&types);
                write_varint(writer, type_bytes.len() as u64)?;
                writer.write_all(&type_bytes)?;

                // Separate by type
                let (tp_first, tp_second): (Vec<u64>, Vec<u64>) = items
                    .iter()
                    .filter_map(|i| match i {
                        MixedRepresentation::Tracepoint(a, b) => Some((*a as u64, *b as u64)),
                        _ => None,
                    })
                    .unzip();

                let (cigar_lens, cigar_ops): (Vec<u64>, Vec<u8>) = items
                    .iter()
                    .filter_map(|i| match i {
                        MixedRepresentation::CigarOp(len, op) => Some((*len as u64, *op as u8)),
                        _ => None,
                    })
                    .unzip();

                // Encode tracepoint streams with strategies (if any tracepoints)
                let tp_first_buf = if tp_first.is_empty() {
                    Vec::new()
                } else {
                    encode_tracepoint_values(&tp_first, first_strategy)?
                };
                let tp_second_buf = if tp_second.is_empty() {
                    Vec::new()
                } else {
                    encode_tracepoint_values(&tp_second, second_strategy)?
                };

                // Encode CIGAR lengths with first strategy (if any CIGARs)
                let cigar_len_buf = if cigar_lens.is_empty() {
                    Vec::new()
                } else {
                    encode_tracepoint_values(&cigar_lens, first_strategy)?
                };

                // Compress all streams
                let tp_first_compressed = if tp_first_buf.is_empty() {
                    Vec::new()
                } else {
                    compress_with_layer(&tp_first_buf, first_layer, first_strategy.zstd_level())?
                };
                let tp_second_compressed = if tp_second_buf.is_empty() {
                    Vec::new()
                } else {
                    compress_with_layer(&tp_second_buf, second_layer, second_strategy.zstd_level())?
                };
                let cigar_len_compressed = if cigar_len_buf.is_empty() {
                    Vec::new()
                } else {
                    compress_with_layer(&cigar_len_buf, first_layer, first_strategy.zstd_level())?
                };

                // Write all streams
                write_varint(writer, tp_first_compressed.len() as u64)?;
                writer.write_all(&tp_first_compressed)?;
                write_varint(writer, tp_second_compressed.len() as u64)?;
                writer.write_all(&tp_second_compressed)?;
                write_varint(writer, cigar_len_compressed.len() as u64)?;
                writer.write_all(&cigar_len_compressed)?;
                // CIGAR ops are raw bytes (no strategy)
                write_varint(writer, cigar_ops.len() as u64)?;
                writer.write_all(&cigar_ops)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod compression_tests {
    use super::*;

    #[test]
    fn rice_roundtrip_varied() {
        let vals = vec![5u64, 7, 9, 15, 15, 200, 210, 205, 300, 301];
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::Rice(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::Rice(3)).unwrap();
        assert_eq!(vals, decoded);
    }

    #[test]
    fn huffman_roundtrip_non_monotonic() {
        let vals = vec![10u64, 8, 8, 12, 8, 10, 9, 20, 18, 18, 18];
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::Huffman(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::Huffman(3)).unwrap();
        assert_eq!(vals, decoded);
    }

    #[test]
    fn lz77_roundtrip_with_repeats() {
        // Values with repeated patterns - should compress well with LZ77
        let vals = vec![10u64, 20, 10, 20, 10, 20, 15, 25];
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::LZ77(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::LZ77(3)).unwrap();
        assert_eq!(vals, decoded);
    }

    #[test]
    fn lz77_roundtrip_no_repeats() {
        // Values without patterns - should still work, just less compression
        let vals = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::LZ77(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::LZ77(3)).unwrap();
        assert_eq!(vals, decoded);
    }

    #[test]
    fn lz77_roundtrip_long_repeat() {
        // Long repeated sequence - tests overlapping match handling
        let vals = vec![42u64; 100];
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::LZ77(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::LZ77(3)).unwrap();
        assert_eq!(vals, decoded);
    }

    #[test]
    fn bitpacked_roundtrip() {
        // Values that need 5 bits (0-31) - should use exactly 5 bits per value
        let vals: Vec<u64> = (0..100).map(|i| i % 32).collect();
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::BitPacked(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::BitPacked(3)).unwrap();
        assert_eq!(vals, decoded);

        // Verify compression: 100 values × 5 bits = 500 bits = 63 bytes (+ 1 header)
        // Old byte-aligned would be 100 bytes + 1 header = 101 bytes
        // Bit-packing should produce around 64 bytes (1 header + ~63 data)
        assert!(
            buf.len() < 80,
            "BitPacked should use bit-packing, got {} bytes (expected ~64)",
            buf.len()
        );
    }

    #[test]
    fn bitpacked_roundtrip_various_widths() {
        // Test different bit widths
        for bits in [1, 4, 7, 12, 20, 32, 48] {
            let max_val = (1u64 << bits) - 1;
            let vals: Vec<u64> = (0..50).map(|i| i % (max_val + 1)).collect();
            let buf = encode_tracepoint_values(&vals, CompressionStrategy::BitPacked(3)).unwrap();
            let decoded =
                decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::BitPacked(3))
                    .unwrap();
            assert_eq!(vals, decoded, "Failed roundtrip for {} bits", bits);
        }

        // Test 64-bit separately (can't do modulo on max + 1)
        let vals: Vec<u64> = (0..50).map(|i| i * 1000000000000).collect();
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::BitPacked(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::BitPacked(3)).unwrap();
        assert_eq!(vals, decoded, "Failed roundtrip for 64 bits");
    }

    #[test]
    fn dictionary_roundtrip_low_cardinality() {
        // Low cardinality data (< 256 unique values) - should use dictionary mode
        let vals: Vec<u64> = (0..500).map(|i| i % 10).collect(); // Only 10 unique values
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::Dictionary(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::Dictionary(3)).unwrap();
        assert_eq!(vals, decoded);

        // Verify dictionary mode was used (mode byte = 1)
        assert_eq!(buf[0], 1, "Should use dictionary mode for low cardinality");
    }

    #[test]
    fn dictionary_roundtrip_high_cardinality() {
        // High cardinality data (> 256 unique values) - should fall back to raw mode
        let vals: Vec<u64> = (0..500).collect(); // 500 unique values
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::Dictionary(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::Dictionary(3)).unwrap();
        assert_eq!(vals, decoded);

        // Verify raw mode was used (mode byte = 0)
        assert_eq!(
            buf[0], 0,
            "Should fall back to raw mode for high cardinality"
        );
    }

    #[test]
    fn dictionary_roundtrip_exactly_256() {
        // Exactly 256 unique values - should still use dictionary mode
        let vals: Vec<u64> = (0..256).collect();
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::Dictionary(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::Dictionary(3)).unwrap();
        assert_eq!(vals, decoded);

        // Verify dictionary mode was used
        assert_eq!(
            buf[0], 1,
            "Should use dictionary mode for exactly 256 unique values"
        );
        // dict_size should be 0 (meaning 256)
        assert_eq!(buf[1], 0, "Dict size 256 should be stored as 0");
    }

    #[test]
    fn simple8b_roundtrip() {
        // Test basic roundtrip
        let vals: Vec<u64> = (0..100).collect();
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::Simple8bFull(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::Simple8bFull(3))
                .unwrap();
        assert_eq!(vals, decoded);
    }

    #[test]
    fn simple8b_small_values_optimization() {
        // 7 values that fit in 1 bit - should use mode (60, 1) for efficiency
        // Before fix: would use (7, 8) = 56 bits for data
        // After fix: uses (60, 1) = 7 bits for data
        let vals = vec![0u64, 1, 0, 1, 0, 1, 0];
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::Simple8bFull(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::Simple8bFull(3))
                .unwrap();
        assert_eq!(vals, decoded);

        // The encoded data should be very compact (1 word = 8 bytes + varint overhead)
        assert!(
            buf.len() <= 16,
            "Simple8b should efficiently pack small values, got {} bytes",
            buf.len()
        );
    }

    #[test]
    fn huffman_vs_rice_size_comparison() {
        // Compare Huffman vs Rice for typical tracepoint data
        let test_cases: Vec<(&str, Vec<u64>)> = vec![
            ("uniform_100", vec![100u64; 50]),
            ("small_range", (95..105).cycle().take(50).collect()),
            (
                "typical_tp",
                vec![
                    100, 101, 99, 102, 100, 98, 103, 100, 101, 99, 100, 102, 101, 100, 99, 98, 100,
                    101, 102, 100, 100, 101, 99, 102, 100, 98, 103, 100, 101, 99,
                ],
            ),
        ];

        println!("\n=== Huffman vs Rice vs ZigzagDelta Size ===");
        println!(
            "{:<15} {:>8} {:>8} {:>8} {:>10}",
            "Test", "Huffman", "Rice", "Zigzag", "Best"
        );
        println!("{}", "-".repeat(55));

        for (name, vals) in test_cases {
            let huffman = encode_tracepoint_values(&vals, CompressionStrategy::Huffman(3))
                .unwrap()
                .len();
            let rice = encode_tracepoint_values(&vals, CompressionStrategy::Rice(3))
                .unwrap()
                .len();
            let zigzag = encode_tracepoint_values(&vals, CompressionStrategy::ZigzagDelta(3))
                .unwrap()
                .len();

            let best = if huffman <= rice && huffman <= zigzag {
                "Huffman"
            } else if rice <= zigzag {
                "Rice"
            } else {
                "Zigzag"
            };

            println!(
                "{:<15} {:>8} {:>8} {:>8} {:>10}",
                name, huffman, rice, zigzag, best
            );
        }
    }
}
