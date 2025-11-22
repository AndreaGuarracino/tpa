//! Binary I/O operations for BPAF format

use lib_tracepoints::{MixedRepresentation, TracepointData, TracepointType};
use log::{debug, info};
use rayon::prelude::*;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};

use crate::ensure_tracepoints;
use crate::format::*;
use crate::utils::*;

// ============================================================================
// COMPRESSION LAYER ABSTRACTION
// ============================================================================

/// Compress data using the specified compression layer
fn compress_with_layer(data: &[u8], layer: CompressionLayer, level: i32) -> io::Result<Vec<u8>> {
    match layer {
        CompressionLayer::Zstd => zstd::encode_all(data, level).map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Zstd compression failed: {}", e),
            )
        }),
        CompressionLayer::Bgzip => {
            // Use bgzip crate for BGZF compression
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
        CompressionLayer::Nocomp => {
            // No compression - return data as-is
            Ok(data.to_vec())
        }
    }
}

/// Decompress data using the explicitly recorded compression layer
fn decompress_with_layer(data: &[u8], layer: CompressionLayer) -> io::Result<Vec<u8>> {
    match layer {
        CompressionLayer::Zstd => {
            if data.is_empty() {
                return Ok(Vec::new());
            }
            zstd::decode_all(data).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("Zstd decompression failed: {}", e),
                )
            })
        }
        CompressionLayer::Bgzip => {
            if data.is_empty() {
                return Ok(Vec::new());
            }
            use bgzip::read::BGZFReader;
            let mut reader = BGZFReader::new(data).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("BGZF decompression failed: {}", e),
                )
            })?;
            let mut decompressed = Vec::new();
            reader.read_to_end(&mut decompressed)?;
            Ok(decompressed)
        }
        CompressionLayer::Nocomp => Ok(data.to_vec()),
    }
}

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
#[allow(dead_code)]
fn encode_zigzag(val: i64) -> u64 {
    ((val << 1) ^ (val >> 63)) as u64
}

pub(crate) struct SmartDualAnalyzer {
    first_states: Vec<StreamState>,
    second_states: Vec<StreamState>,
    layers: [CompressionLayer; 3],
    parallel: bool,
    sample_limit: Option<usize>,
    processed_records: usize,
    processed_tracepoints: usize,
    zstd_level: i32,
}

impl SmartDualAnalyzer {
    pub fn new(zstd_level: i32, sample_limit: Option<usize>, parallel: bool) -> Self {
        let strategies = CompressionStrategy::concrete_strategies(zstd_level);
        let layers = CompressionLayer::all();
        let first_states = strategies
            .iter()
            .cloned()
            .map(|s| StreamState::new(s, layers.len()))
            .collect();
        let second_states = strategies
            .into_iter()
            .map(|s| StreamState::new(s, layers.len()))
            .collect();

        Self {
            first_states,
            second_states,
            layers,
            parallel,
            sample_limit,
            processed_records: 0,
            processed_tracepoints: 0,
            zstd_level,
        }
    }

    pub fn ingest(&mut self, record: &AlignmentRecord) {
        if let Some(limit) = self.sample_limit {
            if self.processed_records >= limit {
                return;
            }
        }

        match &record.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                ensure_tracepoints(&record.tracepoints);
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
                    self.parallel,
                    "First",
                    |first, _second, strategy| encode_first_stream(first, strategy),
                );
                process_stream_states(
                    &mut self.second_states,
                    &self.layers,
                    &first_vals,
                    &second_vals,
                    self.parallel,
                    "Second",
                    |first, second, strategy| encode_second_stream(first, second, strategy),
                );
            }
            _ => {
                panic!("Automatic strategy analysis requires standard or FASTGA tracepoints");
            }
        }
    }

    pub fn finalize(self) -> (CompressionStrategy, CompressionLayer, CompressionLayer) {
        if self.processed_records == 0 {
            panic!("Automatic strategy analysis requires at least one tracepoint");
        }

        let mut first_candidates: Vec<(CompressionStrategy, CompressionLayer, usize)> = Vec::new();
        let mut second_candidates: Vec<(CompressionStrategy, CompressionLayer, usize)> = Vec::new();
        let mut best_first: Option<(CompressionStrategy, CompressionLayer, usize)> = None;
        let mut best_second: Option<(CompressionStrategy, CompressionLayer, usize)> = None;
        let mut worst_first: Option<usize> = None;
        let mut worst_second: Option<usize> = None;
        let strategy_count = self.first_states.len();
        let layer_count = self.layers.len();

        for state in self.first_states.into_iter() {
            for (layer_idx, maybe_size) in state.totals.into_iter().enumerate() {
                if let Some(size) = maybe_size {
                    let layer = self.layers[layer_idx];
                    first_candidates.push((state.strategy.clone(), layer, size));
                    best_first = match best_first.take() {
                        Some(current) if current.2 <= size => Some(current),
                        _ => Some((state.strategy.clone(), layer, size)),
                    };
                    worst_first = Some(worst_first.map(|w| w.max(size)).unwrap_or(size));
                }
            }
        }

        for state in self.second_states.into_iter() {
            for (layer_idx, maybe_size) in state.totals.into_iter().enumerate() {
                if let Some(size) = maybe_size {
                    let layer = self.layers[layer_idx];
                    second_candidates.push((state.strategy.clone(), layer, size));
                    best_second = match best_second.take() {
                        Some(current) if current.2 <= size => Some(current),
                        _ => Some((state.strategy.clone(), layer, size)),
                    };
                    worst_second = Some(worst_second.map(|w| w.max(size)).unwrap_or(size));
                }
            }
        }

        if first_candidates.is_empty() || second_candidates.is_empty() {
            panic!("Automatic strategy analysis failed: no valid candidates");
        }

        let (best_first_strategy, best_first_layer, best_first_size) = best_first.unwrap();
        let (best_second_strategy, best_second_layer, best_second_size) = best_second.unwrap();

        let mut final_first = best_first_strategy.clone();
        let mut final_first_layer = best_first_layer;
        let mut final_second = best_second_strategy.clone();
        let mut final_second_layer = best_second_layer;
        let mut best_total_size = best_first_size + best_second_size;

        let mut combined_best: Option<(
            CompressionStrategy,
            CompressionLayer,
            CompressionStrategy,
            CompressionLayer,
            usize,
        )> = None;
        let mut dependent_combo_count = 0usize;

        for (first_strat, first_layer, first_size) in &first_candidates {
            for (second_strat, second_layer, second_size) in &second_candidates {
                if !second_depends_on_first(second_strat) {
                    continue;
                }
                dependent_combo_count += 1;
                let total = first_size + second_size;
                match &mut combined_best {
                    Some((_, _, _, _, best_total)) if total >= *best_total => {}
                    _ => {
                        combined_best = Some((
                            first_strat.clone(),
                            *first_layer,
                            second_strat.clone(),
                            *second_layer,
                            total,
                        ));
                    }
                }
            }
        }

        if let Some((
            combo_first,
            combo_first_layer,
            combo_second,
            combo_second_layer,
            combo_size,
        )) = combined_best
        {
            if combo_size < best_total_size {
                info!(
                    "Automatic combined search improved size: {} bytes → {} bytes",
                    best_total_size, combo_size
                );
                final_first = combo_first;
                final_first_layer = combo_first_layer;
                final_second = combo_second;
                final_second_layer = combo_second_layer;
                best_total_size = combo_size;
            }
        }

        let dual_strategy = CompressionStrategy::Dual(
            Box::new(final_first.clone()),
            Box::new(final_second.clone()),
            self.zstd_level,
        );

        let total_combos = strategy_count * layer_count;
        info!(
            "Dual empirical analysis: sampled {} records, {} tracepoints - tested {} combinations per stream ({} strategies × {} layers)",
            self.processed_records,
            self.processed_tracepoints,
            total_combos,
            strategy_count,
            layer_count
        );
        info!(
            "  Evaluated {} dependent dual combinations",
            dependent_combo_count
        );
        if let Some(worst) = worst_first {
            let improvement = if worst > best_first_size {
                ((worst - best_first_size) as f64 / worst as f64) * 100.0
            } else {
                0.0
            };
            info!(
                "  First stream winner: {} [{}] = {} bytes ({:.2}% smaller than worst)",
                best_first_strategy,
                best_first_layer.as_str(),
                best_first_size,
                improvement
            );
        }
        if let Some(worst) = worst_second {
            let improvement = if worst > best_second_size {
                ((worst - best_second_size) as f64 / worst as f64) * 100.0
            } else {
                0.0
            };
            info!(
                "  Second stream winner: {} [{}] = {} bytes ({:.2}% smaller than worst)",
                best_second_strategy,
                best_second_layer.as_str(),
                best_second_size,
                improvement
            );
        }
        info!(
            "Automatic: Selected {} [{}] → {} [{}]",
            final_first,
            final_first_layer.as_str(),
            final_second,
            final_second_layer.as_str()
        );
        debug!(
            "Automatic combined size estimate: {} bytes",
            best_total_size
        );

        (dual_strategy, final_first_layer, final_second_layer)
    }
}

struct StreamState {
    strategy: CompressionStrategy,
    totals: Vec<Option<usize>>,
    failed: bool,
}

impl StreamState {
    fn new(strategy: CompressionStrategy, layer_count: usize) -> Self {
        Self {
            strategy,
            totals: vec![Some(0usize); layer_count],
            failed: false,
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
        if self.failed || self.totals.iter().all(|slot| slot.is_none()) {
            return;
        }

        let encoded = match encode_fn(first_vals, second_vals, &self.strategy) {
            Ok(buf) => buf,
            Err(err) => {
                debug!(
                    "{}-stream strategy {} failed to encode sample: {}",
                    stream_label, self.strategy, err
                );
                self.failed = true;
                self.totals.iter_mut().for_each(|slot| *slot = None);
                return;
            }
        };

        let level = self.strategy.zstd_level();
        for (idx, layer) in layers.iter().enumerate() {
            if let Some(total) = self.totals[idx] {
                match compress_with_layer(&encoded[..], *layer, level) {
                    Ok(compressed) => {
                        let len = compressed.len();
                        let var_len = varint_size(len as u64) as usize;
                        self.totals[idx] = Some(total + len + var_len);
                    }
                    Err(err) => {
                        debug!(
                            "{}-stream strategy {} with layer {:?} failed to compress sample: {}",
                            stream_label, self.strategy, layer, err
                        );
                        self.totals[idx] = None;
                    }
                }
            }
        }
    }
}

fn process_stream_states<F>(
    states: &mut [StreamState],
    layers: &[CompressionLayer; 3],
    first_vals: &[u64],
    second_vals: &[u64],
    parallel: bool,
    stream_label: &'static str,
    encode_fn: F,
) where
    F: Fn(&[u64], &[u64], &CompressionStrategy) -> io::Result<Vec<u8>> + Sync,
{
    let worker = |state: &mut StreamState| {
        state.process_sample(layers, first_vals, second_vals, &encode_fn, stream_label);
    };
    if parallel {
        states.par_iter_mut().for_each(worker);
    } else {
        states.iter_mut().for_each(worker);
    }
}

fn encode_first_stream(values: &[u64], strategy: &CompressionStrategy) -> io::Result<Vec<u8>> {
    match strategy {
        CompressionStrategy::AutomaticFast(_)
        | CompressionStrategy::AutomaticSlow(_)
        | CompressionStrategy::Dual(_, _, _) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Automatic strategies must be resolved before encoding",
        )),
        _ => encode_tracepoint_values(values, strategy.clone()),
    }
}

fn encode_second_stream(
    first_vals: &[u64],
    second_vals: &[u64],
    strategy: &CompressionStrategy,
) -> io::Result<Vec<u8>> {
    match strategy {
        CompressionStrategy::AutomaticFast(_)
        | CompressionStrategy::AutomaticSlow(_)
        | CompressionStrategy::Dual(_, _, _) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Automatic strategies must be resolved before encoding",
        )),
        CompressionStrategy::TwoDimDelta(_) => {
            encode_2d_delta_second_values(first_vals, second_vals)
        }
        CompressionStrategy::OffsetJoint(_) => {
            let mut buf = Vec::with_capacity(second_vals.len() * 2);
            for (f, s) in first_vals.iter().zip(second_vals.iter()) {
                let diff = *s as i64 - *f as i64;
                let zigzag = ((diff << 1) ^ (diff >> 63)) as u64;
                write_varint(&mut buf, zigzag)?;
            }
            Ok(buf)
        }
        CompressionStrategy::HybridRLE(_) => {
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
            Ok(buf)
        }
        _ => encode_tracepoint_values(second_vals, strategy.clone()),
    }
}

fn second_depends_on_first(strategy: &CompressionStrategy) -> bool {
    matches!(
        strategy,
        CompressionStrategy::TwoDimDelta(_) | CompressionStrategy::OffsetJoint(_)
    )
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
            strategy.clone(),
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
    strategy: CompressionStrategy,
    tp_type: TracepointType,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
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

    let tracepoints = read_tracepoints(reader, tp_type, strategy, first_layer, second_layer)?;

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

pub(crate) fn read_tracepoints<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
    strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<TracepointData> {
    let num_items = read_varint(reader)? as usize;
    if num_items == 0 {
        panic!("Encountered tracepoint block with zero entries");
    }
    match tp_type {
        TracepointType::Standard | TracepointType::Fastga => {
            let tps = decode_standard_tracepoints(
                reader,
                num_items,
                strategy,
                first_layer,
                second_layer,
            )?;
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

// ============================================================================
// SHARED HELPERS (HEADER/FOOTER, TRACEPOINTS, SKIPS)
// ============================================================================

/// Read header, validate footer, and reset position to just after the header.
pub(crate) fn read_header_and_footer<R: Read + Seek>(
    reader: &mut R,
) -> io::Result<(BinaryPafHeader, u64)> {
    let header = BinaryPafHeader::read(reader)?;
    if header.version != BPAF_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported format version: {}", header.version),
        ));
    }

    let after_header = reader.stream_position()?;
    let footer = BinaryPafFooter::read_from_end(reader)?;
    footer.validate_against(&header)?;
    reader.seek(SeekFrom::Start(after_header))?;
    Ok((header, after_header))
}

/// Decode tracepoints from a known offset (seeks before reading).
pub(crate) fn read_tracepoints_at_offset<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    tp_type: TracepointType,
    strategy: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<TracepointData> {
    reader.seek(SeekFrom::Start(offset))?;
    read_tracepoints(reader, tp_type, strategy, first_layer, second_layer)
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
    out.push(k as u8);
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

/// Encode tracepoint values based on compression strategy
#[inline]
fn encode_tracepoint_values(vals: &[u64], strategy: CompressionStrategy) -> io::Result<Vec<u8>> {
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
            let bits_needed = if max_val == 0 {
                1
            } else {
                64 - max_val.leading_zeros()
            };

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
                let bytes_needed = if val == 0 {
                    1
                } else {
                    ((64 - val.leading_zeros() + 7) / 8) as usize
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
            // Canonical Huffman on zigzagged deltas
            let deltas = delta_encode(vals);
            let zigzagged: Vec<u64> = deltas.iter().map(|&v| encode_zigzag(v)).collect();
            buf = huffman_encode(&zigzagged)?;
        }
        CompressionStrategy::AutomaticFast(_) | CompressionStrategy::AutomaticSlow(_) => {
            panic!("Automatic strategies must be resolved before encoding")
        }
        CompressionStrategy::Dual(_, _, _) => {
            panic!("Dual strategies must be resolved before encoding")
        }
    }
    Ok(buf)
}

/// Encode second values for 2D-Delta strategy (as delta from first values)
#[inline]
fn encode_2d_delta_second_values(first_vals: &[u64], second_vals: &[u64]) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(second_vals.len() * 2);

    // Compute differences: second - first
    let diffs: Vec<i64> = first_vals
        .iter()
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
                    let val =
                        u32::from_le_bytes([reader[0], reader[1], reader[2], reader[3]]) as u64;
                    vals.push(val);
                    reader = &reader[4..];
                }
            } else {
                for _ in 0..num_items {
                    let val = u64::from_le_bytes([
                        reader[0], reader[1], reader[2], reader[3], reader[4], reader[5],
                        reader[6], reader[7],
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
            crate::hybrids::decode_simple8b_full(&words)
        }
        CompressionStrategy::SelectiveRLE(_) => {
            // Selective RLE decode
            crate::hybrids::decode_selective_rle(buf)
        }
        CompressionStrategy::Rice(_) => {
            if num_items == 0 {
                return Ok(Vec::new());
            }
            let zigzagged = decode_rice_values(buf, num_items)?;
            let mut vals = Vec::with_capacity(num_items);
            // First value
            let first = ((zigzagged[0] >> 1) as i64) ^ -((zigzagged[0] & 1) as i64);
            vals.push(first as u64);
            for &enc in zigzagged.iter().skip(1) {
                let delta = ((enc >> 1) as i64) ^ -((enc & 1) as i64);
                let prev = *vals.last().unwrap() as i64;
                vals.push((prev + delta) as u64);
            }
            Ok(vals)
        }
        CompressionStrategy::Huffman(_) => {
            if num_items == 0 {
                return Ok(Vec::new());
            }
            let zigzagged = huffman_decode(buf, num_items)?;
            let mut vals = Vec::with_capacity(num_items);
            let first = ((zigzagged[0] >> 1) as i64) ^ -((zigzagged[0] & 1) as i64);
            vals.push(first as u64);
            for &enc in zigzagged.iter().skip(1) {
                let delta = ((enc >> 1) as i64) ^ -((enc & 1) as i64);
                let prev = *vals.last().unwrap() as i64;
                vals.push((prev + delta) as u64);
            }
            Ok(vals)
        }
        CompressionStrategy::AutomaticFast(_) | CompressionStrategy::AutomaticSlow(_) => {
            panic!("Automatic strategies must be resolved before decoding")
        }
        CompressionStrategy::Dual(_, _, _) => {
            panic!("Dual strategies must be resolved before decoding")
        }
    }
}

/// Decode second values for 2D-Delta strategy (from diff values and first values)
#[inline]
#[allow(dead_code)]
fn decode_2d_delta_second_values(buf: &[u8], first_vals: &[u64]) -> io::Result<Vec<u64>> {
    let mut reader = buf;
    let mut second_vals = Vec::with_capacity(first_vals.len());

    // Decode zigzag-encoded differences and add to first values
    for &first in first_vals {
        let zigzag = read_varint(&mut reader)?;
        let diff = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
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
    strategy: CompressionStrategy,
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
            let a =
                safe_signed_add(0, ((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64))? as usize;

            let zigzag_b = read_varint(&mut second_reader)?;
            let b =
                safe_signed_add(0, ((zigzag_b >> 1) as i64) ^ -((zigzag_b & 1) as i64))? as usize;

            tps.push((a, b));

            // Remaining values: zigzag decode + delta accumulate
            for _ in 1..num_items {
                let zigzag_a = read_varint(&mut first_reader)?;
                let delta_a = ((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64);
                let prev_a = tps.last().unwrap().0 as i64;
                let a = safe_signed_add(prev_a, delta_a)? as usize;

                let zigzag_b = read_varint(&mut second_reader)?;
                let delta_b = ((zigzag_b >> 1) as i64) ^ -((zigzag_b & 1) as i64);
                let prev_b = tps.last().unwrap().1 as i64;
                let b = safe_signed_add(prev_b, delta_b)? as usize;

                tps.push((a, b));
            }
        }
        CompressionStrategy::TwoDimDelta(_) => {
            // First values: same as ZigzagDelta
            let mut first_vals = Vec::with_capacity(num_items);
            let zigzag_a = read_varint(&mut first_reader)?;
            let a = safe_signed_add(0, ((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64))?;
            first_vals.push(a as usize);

            for _ in 1..num_items {
                let zigzag_a = read_varint(&mut first_reader)?;
                let delta_a = ((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64);
                let prev_a = *first_vals.last().unwrap() as i64;
                first_vals.push(safe_signed_add(prev_a, delta_a)? as usize);
            }

            // Second values: decode as differences from first values
            for &a in &first_vals {
                let zigzag_diff = read_varint(&mut second_reader)?;
                let diff = ((zigzag_diff >> 1) as i64) ^ -((zigzag_diff & 1) as i64);
                let b = safe_signed_add(a as i64, diff)? as usize;
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
                    let val = u32::from_le_bytes([
                        first_reader[0],
                        first_reader[1],
                        first_reader[2],
                        first_reader[3],
                    ]) as usize;
                    first_vals.push(val);
                    first_reader = &first_reader[4..];
                }
            } else {
                for _ in 0..num_items {
                    let val = u64::from_le_bytes([
                        first_reader[0],
                        first_reader[1],
                        first_reader[2],
                        first_reader[3],
                        first_reader[4],
                        first_reader[5],
                        first_reader[6],
                        first_reader[7],
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
                    let val = u32::from_le_bytes([
                        second_reader[0],
                        second_reader[1],
                        second_reader[2],
                        second_reader[3],
                    ]) as usize;
                    second_vals.push(val);
                    second_reader = &second_reader[4..];
                }
            } else {
                for _ in 0..num_items {
                    let val = u64::from_le_bytes([
                        second_reader[0],
                        second_reader[1],
                        second_reader[2],
                        second_reader[3],
                        second_reader[4],
                        second_reader[5],
                        second_reader[6],
                        second_reader[7],
                    ]) as usize;
                    second_vals.push(val);
                    second_reader = &second_reader[8..];
                }
            }

            for (a, b) in first_vals.into_iter().zip(second_vals.into_iter()) {
                tps.push((a, b));
            }
        }
        CompressionStrategy::DeltaOfDelta(_)
        | CompressionStrategy::FrameOfReference(_)
        | CompressionStrategy::XORDelta(_)
        | CompressionStrategy::Dictionary(_)
        | CompressionStrategy::Simple8(_)
        | CompressionStrategy::StreamVByte(_)
        | CompressionStrategy::FastPFOR(_)
        | CompressionStrategy::Cascaded(_)
        | CompressionStrategy::Simple8bFull(_)
        | CompressionStrategy::SelectiveRLE(_)
        | CompressionStrategy::Rice(_)
        | CompressionStrategy::Huffman(_) => {
            // These strategies use standard encoding for both streams
            let first_vals = decode_tracepoint_values(&first_buf, num_items, strategy.clone())?;
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
            if second_vals.len() != num_items {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "hybrid-rle decode length mismatch: expected {} got {}",
                        num_items,
                        second_vals.len()
                    ),
                ));
            }

            for (a, b) in first_vals.into_iter().zip(second_vals.into_iter()) {
                tps.push((a, b));
            }
        }
        CompressionStrategy::OffsetJoint(_) => {
            // First values: zigzag delta
            let mut first_vals = Vec::with_capacity(num_items);
            let zigzag_a = read_varint(&mut first_reader)?;
            let a = safe_signed_add(0, ((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64))?;
            first_vals.push(a as usize);

            for _ in 1..num_items {
                let zigzag_a = read_varint(&mut first_reader)?;
                let delta_a = ((zigzag_a >> 1) as i64) ^ -((zigzag_a & 1) as i64);
                let prev_a = *first_vals.last().unwrap() as i64;
                first_vals.push(safe_signed_add(prev_a, delta_a)? as usize);
            }

            // Second values: as offset from first
            for &a in &first_vals {
                let offset_zigzag = read_varint(&mut second_reader)?;
                let offset = ((offset_zigzag >> 1) as i64) ^ -((offset_zigzag & 1) as i64);
                let b = safe_signed_add(a as i64, offset)? as usize;
                tps.push((a, b));
            }
        }
        CompressionStrategy::AutomaticFast(_) | CompressionStrategy::AutomaticSlow(_) => {
            panic!("Automatic strategies must be resolved before decoding")
        }
        CompressionStrategy::Dual(first_strat, second_strat, _) => {
            // Dual strategy: decode first_vals and second_vals independently
            let first_vals = decode_tracepoint_values(&first_buf, num_items, *first_strat.clone())?;

            // Decode second, with special handling for 2d-delta and offset-joint which depend on first_vals
            let second_vals: Vec<u64> = match *second_strat.clone() {
                CompressionStrategy::TwoDimDelta(_) => {
                    decode_2d_delta_second_values(&second_buf, &first_vals)?
                }
                CompressionStrategy::OffsetJoint(_) => {
                    let mut reader = &second_buf[..];
                    let mut vals = Vec::with_capacity(num_items);
                    for a in &first_vals {
                        let zigzag = read_varint(&mut reader)?;
                        let diff = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
                        let b = safe_signed_add(*a as i64, diff)?;
                        vals.push(b as u64);
                    }
                    if !reader.is_empty() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("offset-joint residual bytes after decode: {}", reader.len()),
                        ));
                    }
                    vals
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
                                "hybrid-rle dual decode length mismatch: expected {} got {}",
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
                _ => decode_tracepoint_values(&second_buf, num_items, *second_strat.clone())?,
            };

            for (a, b) in first_vals.into_iter().zip(second_vals.into_iter()) {
                tps.push((a as usize, b as usize));
            }
        }
    }

    Ok(tps)
}

/// Variable tracepoint decoding (only raw varints)
#[inline(always)]
pub(crate) fn decode_variable_tracepoints<R: Read>(
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
pub(crate) fn decode_mixed_tracepoints<R: Read>(
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
        self.write_tracepoints(writer, strategy, first_layer, second_layer)?;
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
        first_layer: CompressionLayer,
        second_layer: CompressionLayer,
    ) -> io::Result<()> {
        ensure_tracepoints(&self.tracepoints);
        match &self.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                write_varint(writer, tps.len() as u64)?;
                let (first_vals, second_vals): (Vec<u64>, Vec<u64>) =
                    tps.iter().map(|(a, b)| (*a as u64, *b as u64)).unzip();

                let (first_val_buf, second_val_buf) = match &strategy {
                    CompressionStrategy::Dual(first_strat, second_strat, _) => {
                        // Dual strategy: encode first_vals and second_vals independently
                        let first_buf =
                            encode_tracepoint_values(&first_vals, *first_strat.clone())?;
                        let second_buf = match *second_strat.clone() {
                            CompressionStrategy::TwoDimDelta(_) => {
                                encode_2d_delta_second_values(&first_vals, &second_vals)?
                            }
                            CompressionStrategy::OffsetJoint(_) => {
                                let mut buf = Vec::with_capacity(second_vals.len() * 2);
                                for (f, s) in first_vals.iter().zip(second_vals.iter()) {
                                    let diff = *s as i64 - *f as i64;
                                    let zigzag = ((diff << 1) ^ (diff >> 63)) as u64;
                                    write_varint(&mut buf, zigzag)?;
                                }
                                buf
                            }
                            CompressionStrategy::HybridRLE(_) => {
                                // Encode target with RLE (value, run_len)
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
                            _ => encode_tracepoint_values(&second_vals, *second_strat.clone())?,
                        };
                        (first_buf, second_buf)
                    }
                    _ => {
                        // Single strategy
                        let first_buf = encode_tracepoint_values(&first_vals, strategy.clone())?;
                        let second_buf = match &strategy {
                            CompressionStrategy::TwoDimDelta(_) => {
                                // Encode second as delta from first
                                encode_2d_delta_second_values(&first_vals, &second_vals)?
                            }
                            CompressionStrategy::OffsetJoint(_) => {
                                // Encode second as signed offset from first (zigzag)
                                let mut buf = Vec::with_capacity(second_vals.len() * 2);
                                for (f, s) in first_vals.iter().zip(second_vals.iter()) {
                                    let diff = *s as i64 - *f as i64;
                                    let zigzag = ((diff << 1) ^ (diff >> 63)) as u64;
                                    write_varint(&mut buf, zigzag)?;
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
                                encode_tracepoint_values(&second_vals, strategy.clone())?
                            }
                        };
                        (first_buf, second_buf)
                    }
                };

                // Use the explicitly passed layer parameter
                let first_compressed =
                    compress_with_layer(&first_val_buf[..], first_layer, strategy.zstd_level())?;
                let second_compressed =
                    compress_with_layer(&second_val_buf[..], second_layer, strategy.zstd_level())?;

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
    fn empty_rice_and_huffman() {
        let empty: Vec<u64> = Vec::new();
        let r = encode_tracepoint_values(&empty, CompressionStrategy::Rice(3)).unwrap();
        let h = encode_tracepoint_values(&empty, CompressionStrategy::Huffman(3)).unwrap();
        assert!(r.is_empty());
        assert!(h.is_empty());
    }
}
