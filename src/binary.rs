//! Binary I/O operations for TPA format

use log::info;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use tracepoints::{ComplexityMetric, MixedRepresentation, TracepointData, TracepointType};

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
            let level = u32::try_from(level)
                .ok()
                .filter(|l| *l <= 9)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "BGZIP compression level must be between 0 and 9",
                    )
                })?;
            let compression = bgzip::Compression::new(level).map_err(io::Error::other)?;
            let mut compressed = Vec::new();
            {
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
    huffman_samples: [HuffmanSamples; 2],
    layers: [CompressionLayer; 3],
    level: i32,
    sample_limit: usize, // 0 = no limit
    processed_records: usize,
}

impl StrategyAnalyzer {
    /// Create analyzer. With `forced_layer = Some(l)`, tests every strategy but pins the layer to `l`
    /// (e.g. "benchmark-nocomp"). Otherwise, if `test_layers` is true tests all 3 layers and picks the
    /// smallest; if false tests only the first layer (all-records mode).
    pub fn new(
        zstd_level: i32,
        sample_limit: usize,
        test_layers: bool,
        forced_layer: Option<CompressionLayer>,
    ) -> Self {
        let strategies = CompressionStrategy::all(zstd_level);
        // When a layer is forced, test exactly that one layer (index 0, which select_best reads back).
        let (layers, num_layers_to_test) = match (test_layers, forced_layer) {
            (false, _) => ([CompressionLayer::Nocomp; 3], 1),
            (_, Some(layer)) => ([layer; 3], 1),
            _ => (CompressionLayer::all(), 3),
        };
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
            huffman_samples: Default::default(),
            layers,
            level: zstd_level,
            sample_limit,
            processed_records: 0,
        }
    }

    pub fn analyze_record(&mut self, record: &CompactRecord) -> io::Result<()> {
        if self.sample_limit > 0 && self.processed_records >= self.sample_limit {
            return Ok(());
        }

        match &record.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                self.processed_records += 1;

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
                )?;
                process_stream_states(
                    &mut self.second_states,
                    &self.layers,
                    &first_vals,
                    &second_vals,
                    "Second",
                    encode_second_stream,
                )?;
                self.huffman_samples[0].push(&first_vals)?;
                self.huffman_samples[1].push(&second_vals)?;
            }
            TracepointData::FastgaNoDiff(tps) => {
                self.processed_records += 1;
                let vals: Vec<u64> = tps.iter().map(|&v| v as u64).collect();
                process_stream_states(
                    &mut self.first_states,
                    &self.layers,
                    &vals,
                    &[],
                    "First",
                    |first, _second, strategy| encode_first_stream(first, strategy),
                )?;
                self.huffman_samples[0].push(&vals)?;
            }
            TracepointData::Variable(tps) => {
                self.processed_records += 1;

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
                )?;
                // Only analyze second stream if there are any second values
                if !second_vals.is_empty() {
                    process_stream_states(
                        &mut self.second_states,
                        &self.layers,
                        &first_vals,
                        &second_vals,
                        "Second",
                        encode_second_stream,
                    )?;
                }
            }
            TracepointData::Mixed(items) => {
                self.processed_records += 1;

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
                )?;
                // Only analyze second stream if there are tracepoint second values
                if !tp_second.is_empty() {
                    process_stream_states(
                        &mut self.second_states,
                        &self.layers,
                        &first_vals,
                        &tp_second,
                        "Second",
                        encode_second_stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    pub fn select_best(
        mut self,
    ) -> (
        CompressionStrategy,
        CompressionStrategy,
        CompressionLayer,
        CompressionLayer,
    ) {
        if let Some(state) = huffman_state(
            &self.huffman_samples[0],
            &self.layers,
            self.level,
            self.first_states[0].totals.len(),
        ) {
            self.first_states.push(state);
        }
        if let Some(state) = huffman_state(
            &self.huffman_samples[1],
            &self.layers,
            self.level,
            self.second_states[0].totals.len(),
        ) {
            self.second_states.push(state);
        }

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
            "Benchmark: {} [{}] → {} [{}]",
            first_best.0,
            first_best.1.as_str(),
            second_best.0,
            second_best.1.as_str()
        );

        (first_best.0, second_best.0, first_best.1, second_best.1)
    }
}

/// Automatic strategy selection based on tracepoint type and complexity metric.
///
/// Returns (first_strategy, first_layer, second_strategy, second_layer).
/// The choices follow from structural properties of each segmentation algorithm:
/// - Edit-distance: right-skewed segment lengths → Rice coding for first stream
/// - Diagonal-distance: fewer segments per record → Raw varint (Rice overhead too high)
/// - Fastga (FL-TP): many segments per record amortizes Rice overhead
/// - Second stream: query/target advances are correlated → TwoDimDelta (except FL-TP)
pub(crate) fn automatic_select(
    tp_type: TracepointType,
    complexity_metric: ComplexityMetric,
    level: i32,
) -> (
    CompressionStrategy,
    CompressionStrategy,
    CompressionLayer,
    CompressionLayer,
) {
    let nocomp = CompressionLayer::Nocomp;
    match tp_type {
        TracepointType::Standard => match complexity_metric {
            ComplexityMetric::DiagonalDistance => (
                CompressionStrategy::Raw(level),
                CompressionStrategy::TwoDimDelta(level),
                nocomp,
                nocomp,
            ),
            ComplexityMetric::EditDistance => (
                CompressionStrategy::Rice(level),
                CompressionStrategy::TwoDimDelta(level),
                nocomp,
                nocomp,
            ),
        },
        TracepointType::Fastga => (
            CompressionStrategy::EliasFano(level),
            CompressionStrategy::LocalHuffman(level),
            nocomp,
            nocomp,
        ),
        TracepointType::FastgaNoDiff => (
            CompressionStrategy::LocalHuffman(level),
            CompressionStrategy::Raw(level),
            nocomp,
            nocomp,
        ),
        TracepointType::Variable | TracepointType::Mixed => (
            CompressionStrategy::ZigzagDelta(level),
            CompressionStrategy::TwoDimDelta(level),
            nocomp,
            nocomp,
        ),
    }
}

struct StreamState {
    strategy: CompressionStrategy,
    totals: Vec<usize>,
}

#[derive(Default)]
struct HuffmanSamples {
    encoded: Vec<u8>,
    frequencies: std::collections::HashMap<u64, usize>,
}

impl HuffmanSamples {
    fn push(&mut self, values: &[u64]) -> io::Result<()> {
        write_varint(&mut self.encoded, values.len() as u64)?;
        for &value in values {
            write_varint(&mut self.encoded, value)?;
            *self.frequencies.entry(value).or_insert(0) += 1;
        }
        Ok(())
    }
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
        CompressionStrategy::LocalHuffman(_) => 13,
        CompressionStrategy::FastPFOR(_) => 14,
        CompressionStrategy::Cascaded(_) => 15,
        CompressionStrategy::SelectiveRLE(_) => 16,
        CompressionStrategy::LZ77(_) => 17,
        CompressionStrategy::EliasFano(_) => 18,
        CompressionStrategy::Huffman(_) => 19,
        CompressionStrategy::Automatic(_) | CompressionStrategy::Benchmark(_, _, _) => 255,
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
    ) -> io::Result<()>
    where
        F: Fn(&[u64], &[u64], &CompressionStrategy) -> io::Result<Vec<u8>>,
    {
        let encoded = encode_fn(first_vals, second_vals, &self.strategy).map_err(|err| {
            io::Error::new(
                err.kind(),
                format!(
                    "{stream_label}-stream strategy {} failed to encode sample: {err}",
                    self.strategy
                ),
            )
        })?;
        self.add_encoded(layers, &encoded, stream_label)
    }

    fn add_encoded(
        &mut self,
        layers: &[CompressionLayer; 3],
        encoded: &[u8],
        stream_label: &str,
    ) -> io::Result<()> {
        let level = self.strategy.zstd_level();
        for (total, layer) in self.totals.iter_mut().zip(layers) {
            let compressed = compress_with_layer(encoded, *layer, level).map_err(|err| {
                io::Error::new(
                    err.kind(),
                    format!(
                        "{stream_label}-stream strategy {} with layer {layer:?} failed: {err}",
                        self.strategy
                    ),
                )
            })?;
            let len = compressed.len();
            *total = total
                .checked_add(len + varint_size(len as u64) as usize)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "sample size overflow")
                })?;
        }
        Ok(())
    }
}

fn huffman_state(
    samples: &HuffmanSamples,
    layers: &[CompressionLayer; 3],
    level: i32,
    layer_count: usize,
) -> Option<StreamState> {
    let table = HuffmanTable::from_frequencies(&samples.frequencies).ok()?;
    let mut state = StreamState::new(CompressionStrategy::Huffman(level), layer_count);
    let mut table_bytes = Vec::new();
    table.write(&mut table_bytes).ok()?;
    state
        .totals
        .iter_mut()
        .for_each(|n| *n += table_bytes.len());
    let mut input = samples.encoded.as_slice();
    while !input.is_empty() {
        let len = usize::try_from(read_varint(&mut input).ok()?).ok()?;
        let values = (0..len)
            .map(|_| read_varint(&mut input))
            .collect::<io::Result<Vec<_>>>()
            .ok()?;
        state
            .add_encoded(layers, &table.encode(&values).ok()?, "Huffman")
            .ok()?;
    }
    Some(state)
}

fn process_stream_states<F>(
    states: &mut [StreamState],
    layers: &[CompressionLayer; 3],
    first_vals: &[u64],
    second_vals: &[u64],
    stream_label: &'static str,
    encode_fn: F,
) -> io::Result<()>
where
    F: Fn(&[u64], &[u64], &CompressionStrategy) -> io::Result<Vec<u8>> + Sync,
{
    states.par_iter_mut().try_for_each(|state| {
        state.process_sample(layers, first_vals, second_vals, &encode_fn, stream_label)
    })
}

fn encode_first_stream(values: &[u64], strategy: &CompressionStrategy) -> io::Result<Vec<u8>> {
    match strategy {
        CompressionStrategy::Automatic(_) | CompressionStrategy::Benchmark(_, _, _) => {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Automatic/Benchmark must be resolved before encoding",
            ))
        }
        _ => encode_tracepoint_values(values, *strategy),
    }
}

fn encode_second_stream(
    first_vals: &[u64],
    second_vals: &[u64],
    strategy: &CompressionStrategy,
) -> io::Result<Vec<u8>> {
    match strategy {
        CompressionStrategy::Automatic(_) | CompressionStrategy::Benchmark(_, _, _) => {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Automatic/Benchmark must be resolved before encoding",
            ))
        }
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
    tables: &HuffmanTables,
) -> io::Result<()> {
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
            tables,
        )?;
        write_paf_line_with_tracepoints(&mut writer, &record, &string_table)?;
    }
    writer.flush()?;

    info!("Decompressed {} records", header.num_records);
    Ok(())
}

fn encode_stream(
    vals: &[u64],
    strategy: CompressionStrategy,
    table: Option<&HuffmanTable>,
) -> io::Result<Vec<u8>> {
    match table {
        Some(t) if strategy.uses_file_table() => t.encode(vals),
        _ => encode_tracepoint_values(vals, strategy),
    }
}

fn decode_stream(
    buf: &[u8],
    num_items: usize,
    strategy: CompressionStrategy,
    table: Option<&HuffmanTable>,
) -> io::Result<Vec<u64>> {
    match table {
        Some(t) if strategy.uses_file_table() => t.decode(buf, num_items),
        _ => decode_tracepoint_values(buf, num_items, strategy),
    }
}

/// The pair of file-level Huffman tables (one per stream). Empty for version-1 files.
#[derive(Default)]
pub struct HuffmanTables {
    first: Option<HuffmanTable>,
    second: Option<HuffmanTable>,
}

impl HuffmanTables {
    fn validate_presence(header: &TpaHeader, present: [bool; 2]) -> io::Result<()> {
        if present != header.required_tables() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Huffman table blocks do not match the declared strategies",
            ));
        }
        Ok(())
    }

    pub(crate) fn new(first: Option<HuffmanTable>, second: Option<HuffmanTable>) -> Self {
        Self { first, second }
    }

    /// Symbol counts of the two tables (0 when absent)
    pub fn symbol_counts(&self) -> (usize, usize) {
        (
            self.first.as_ref().map_or(0, |t| t.lengths.len()),
            self.second.as_ref().map_or(0, |t| t.lengths.len()),
        )
    }

    pub(crate) fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        for t in [&self.first, &self.second] {
            match t {
                Some(t) => t.write(writer)?,
                None => {
                    write_varint(writer, 0)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn read<R: Read>(reader: &mut R, header: &TpaHeader) -> io::Result<Self> {
        if header.version() != crate::format::TPA_VERSION_HUFFMAN_TABLE {
            return Ok(Self::default());
        }
        let first = HuffmanTable::read(reader)?;
        let second = HuffmanTable::read(reader)?;
        Self::validate_presence(header, [first.is_some(), second.is_some()])?;
        Ok(Self { first, second })
    }

    /// Skip the section and return its byte length.
    pub fn skip<R: Read + Seek>(reader: &mut R, header: &TpaHeader) -> io::Result<u64> {
        if header.version() != crate::format::TPA_VERSION_HUFFMAN_TABLE {
            return Ok(0);
        }
        let start = reader.stream_position()?;
        let mut present = [false; 2];
        for item in &mut present {
            let len = read_varint(reader)?;
            *item = len != 0;
            let copied = io::copy(&mut reader.by_ref().take(len), &mut io::sink())?;
            if copied != len {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("Huffman table section truncated: wanted {len} bytes, got {copied}"),
                ));
            }
        }
        Self::validate_presence(header, present)?;
        Ok(reader.stream_position()? - start)
    }

    /// Load the tables of a TPA file once, for reuse across standalone random-access calls.
    pub fn load(path: &str) -> io::Result<Self> {
        let mut file = File::open(path)?;
        Ok(read_header_and_tables(&mut file)?.1)
    }
}

pub(crate) fn read_record<R: Read>(
    reader: &mut R,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    tp_type: TracepointType,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
    tables: &HuffmanTables,
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
        tables,
        first_layer,
        second_layer,
    )?;

    let mut tags = Vec::new();
    for _ in 0..read_varint(reader)? {
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn read_tracepoints<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    tables: &HuffmanTables,
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
                tables,
                first_layer,
                second_layer,
            )?;
            Ok(match tp_type {
                TracepointType::Standard => TracepointData::Standard(tps),
                _ => TracepointData::Fastga(tps),
            })
        }
        TracepointType::FastgaNoDiff => {
            // Single stream: [first_len][first_data]
            let first_len = read_varint(reader)? as usize;
            let mut first_compressed = vec![0u8; first_len];
            reader.read_exact(&mut first_compressed)?;
            let first_buf = decompress_with_layer(&first_compressed[..], first_layer)?;
            let vals = decode_stream(&first_buf, num_items, first_strategy, tables.first.as_ref())?;
            Ok(TracepointData::FastgaNoDiff(
                vals.into_iter().map(|v| v as usize).collect(),
            ))
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

/// Read a validated header and leave the reader at the string table.
pub(crate) fn read_header<R: Read + Seek>(reader: &mut R) -> io::Result<TpaHeader> {
    let header = read_validated_header(reader)?;
    HuffmanTables::skip(reader, &header)?;
    Ok(header)
}

/// Read a validated header and its file-level tables.
pub(crate) fn read_header_and_tables<R: Read + Seek>(
    reader: &mut R,
) -> io::Result<(TpaHeader, HuffmanTables)> {
    let header = read_validated_header(reader)?;
    let tables = HuffmanTables::read(reader, &header)?;
    Ok((header, tables))
}

fn read_validated_header<R: Read + Seek>(reader: &mut R) -> io::Result<TpaHeader> {
    let header = TpaHeader::read(reader)?;
    let after_header = reader.stream_position()?;
    let footer = TpaFooter::read_from_end(reader)?;
    footer.validate_against(&header)?;
    reader.seek(SeekFrom::Start(after_header))?;
    Ok(header)
}

/// Decode tracepoints from a known offset (seeks before reading).
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_tracepoints_at_offset<R: Read + Seek>(
    reader: &mut R,
    offset: u64,
    tp_type: TracepointType,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    tables: &HuffmanTables,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
) -> io::Result<TracepointData> {
    reader.seek(SeekFrom::Start(offset))?;
    read_tracepoints(
        reader,
        tp_type,
        first_strategy,
        second_strategy,
        tables,
        first_layer,
        second_layer,
    )
}

fn skip_exact<R: Read>(reader: &mut R, len: u64) -> io::Result<()> {
    if io::copy(&mut reader.take(len), &mut io::sink())? == len {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "length-prefixed field is truncated",
        ))
    }
}

fn skip_block<R: Read>(reader: &mut R) -> io::Result<()> {
    let len = read_varint(reader)?;
    skip_exact(reader, len)
}

fn skip_tracepoints_and_tags<R: Read>(reader: &mut R, tp_type: TracepointType) -> io::Result<()> {
    read_varint(reader)?; // tracepoint count
    let blocks = match tp_type {
        TracepointType::Standard | TracepointType::Fastga => 2,
        TracepointType::FastgaNoDiff => 1,
        TracepointType::Variable => 3,
        TracepointType::Mixed => 5,
    };
    for _ in 0..blocks {
        skip_block(reader)?;
    }

    for _ in 0..read_varint(reader)? {
        skip_exact(reader, 2)?; // tag key
        let mut tag_type = [0];
        reader.read_exact(&mut tag_type)?;
        match tag_type[0] {
            b'i' | b'f' => skip_exact(reader, 4)?,
            b'Z' => skip_block(reader)?,
            value => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid tag type: '{}' (byte: {value})", value as char),
                ))
            }
        }
    }
    Ok(())
}

/// Skip over a record using only Read (no Seek) - for BGZF sequential scanning.
pub(crate) fn skip_record_sequential<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
) -> io::Result<()> {
    read_varint(reader)?; // query_name_id
    read_varint(reader)?; // query_start
    read_varint(reader)?; // query_end
    skip_exact(reader, 1)?; // strand
    read_varint(reader)?; // target_name_id
    read_varint(reader)?; // target_start
    read_varint(reader)?; // target_end
    read_varint(reader)?; // residue_matches
    read_varint(reader)?; // alignment_block_len
    skip_exact(reader, 1)?; // mapping_quality
    skip_tracepoints_and_tags(reader, tp_type)
}

/// Read record header fields (coordinates, strand, mapq) and skip tracepoints+tags.
/// Advances the stream to the start of the next record. Only requires `Read` (no `Seek`),
/// making it suitable for sequential streaming over both File and BGZF readers.
pub(crate) fn read_record_metadata<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
) -> io::Result<CompactRecord> {
    // READ header fields
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

    skip_tracepoints_and_tags(reader, tp_type)?;

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
        tracepoints: TracepointData::Standard(Vec::new()),
        tags: Vec::new(),
    })
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

    fn has_only_padding(&self) -> bool {
        if self.bits_left == 0 {
            return self.byte_idx == self.data.len();
        }
        let padding_mask = (1u8 << self.bits_left) - 1;
        self.byte_idx + 1 == self.data.len() && self.data[self.byte_idx] & padding_mask == 0
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
// ELIAS-FANO CODING
// ============================================================================
//
// Prefix-sums the stream into a monotone sequence, then stores it in the
// quasi-succinct Elias-Fano layout: 1 header byte `l = floor(log2(u/n))`, then
// n*l low bits, then n unary high-bit gaps (~ n*l + 2n bits). Decode recovers
// the cumulative values and returns their successive differences.

fn encode_eliasfano_values(values: &[u64]) -> io::Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let n = values.len() as u64;
    let mut cum = Vec::with_capacity(values.len());
    let mut acc: u64 = 0;
    for &v in values {
        acc += v;
        cum.push(acc);
    }
    // l = floor(log2(u/n)), u = max+1; number of low bits per value.
    let un = (acc + 1) / n;
    let l: u8 = if un < 1 {
        0
    } else {
        (63 - un.leading_zeros()) as u8
    };
    let low_mask = if l == 0 { 0 } else { (1u64 << (l as u32)) - 1 };
    let mut writer = BitWriter::default();
    // Low bits: n * l bits, in order.
    for &c in &cum {
        writer.write_bits(c & low_mask, l);
    }
    // High bits: unary gaps between successive high parts (high[-1] = 0).
    let mut prev_high: u64 = 0;
    for &c in &cum {
        let high = c >> (l as u32);
        writer.write_unary(high - prev_high); // `gap` ones then a zero
        prev_high = high;
    }
    let mut out = Vec::with_capacity(1 + values.len());
    out.push(l);
    out.extend_from_slice(&writer.finish());
    Ok(out)
}

fn decode_eliasfano_values(buf: &[u8], num_items: usize) -> io::Result<Vec<u64>> {
    if num_items == 0 {
        return Ok(Vec::new());
    }
    if buf.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "missing Elias-Fano header",
        ));
    }
    let l = buf[0];
    let mut reader = BitReader::new(&buf[1..]);
    // Low bits.
    let mut lows = Vec::with_capacity(num_items);
    for _ in 0..num_items {
        lows.push(if l > 0 { reader.read_bits(l)? } else { 0 });
    }
    // High bits (unary gaps) -> cumulative sequence -> successive differences.
    let mut out = Vec::with_capacity(num_items);
    let mut high: u64 = 0;
    let mut prev_cum: u64 = 0;
    for low in lows {
        high += reader.read_unary()?;
        let c = (high << (l as u32)) | low;
        out.push(c - prev_cum);
        prev_cum = c;
    }
    Ok(out)
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

/// O(n log n) Huffman lengths with deterministic symbol-based tie-breaking.
fn huffman_code_lengths(freqs: &[(u64, usize)]) -> Vec<(u64, u8)> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    if freqs.is_empty() {
        return Vec::new();
    }
    if freqs.len() == 1 {
        return vec![(freqs[0].0, 1)];
    }

    enum Node {
        Leaf(u64),
        Internal(usize, usize),
    }
    let mut nodes: Vec<Node> = Vec::with_capacity(freqs.len() * 2);
    let mut heap: BinaryHeap<Reverse<(usize, u64, usize)>> = BinaryHeap::with_capacity(freqs.len());
    for &(value, count) in freqs {
        nodes.push(Node::Leaf(value));
        heap.push(Reverse((count, value, nodes.len() - 1)));
    }
    while heap.len() > 1 {
        let Reverse((c1, v1, i1)) = heap.pop().unwrap();
        let Reverse((c2, v2, i2)) = heap.pop().unwrap();
        nodes.push(Node::Internal(i1, i2));
        heap.push(Reverse((c1 + c2, v1.min(v2), nodes.len() - 1)));
    }

    let root = heap.pop().unwrap().0 .2;
    let mut lengths = Vec::with_capacity(freqs.len());
    let mut stack = vec![(root, 0u8)];
    while let Some((idx, depth)) = stack.pop() {
        match nodes[idx] {
            Node::Leaf(value) => lengths.push((value, depth.max(1))),
            Node::Internal(l, r) => {
                stack.push((l, depth + 1));
                stack.push((r, depth + 1));
            }
        }
    }
    lengths
}

fn build_canonical_codes(lengths: &mut [(u64, u8)]) -> io::Result<Vec<HuffmanCode>> {
    lengths.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut codes = Vec::with_capacity(lengths.len());
    let mut code = 0u64;
    let mut prev_len = 0u8;
    for &(value, len) in lengths.iter() {
        if len == 0 || len > 63 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid Huffman code length: {}", len),
            ));
        }
        if prev_len > 0 {
            code <<= (len - prev_len) as u32;
        }
        // Enforce the Kraft bound.
        if code >= (1u64 << len) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("oversubscribed Huffman code lengths at value {value} (len {len})"),
            ));
        }
        codes.push(HuffmanCode { value, len, code });
        code += 1;
        prev_len = len;
    }
    Ok(codes)
}

#[derive(Clone, Copy, Default)]
struct DecodeRange {
    first_code: u64,
    first_symbol: usize,
    count: usize,
}

struct HuffmanDecoder {
    ranges: Vec<DecodeRange>,
    symbols: Vec<u64>,
    max_len: u8,
}

impl HuffmanDecoder {
    fn new(codes: &[HuffmanCode]) -> Self {
        let max_len = codes.last().map_or(0, |code| code.len);
        let mut ranges = vec![DecodeRange::default(); max_len as usize + 1];
        for (index, code) in codes.iter().enumerate() {
            let range = &mut ranges[code.len as usize];
            if range.count == 0 {
                range.first_code = code.code;
                range.first_symbol = index;
            }
            range.count += 1;
        }
        Self {
            ranges,
            symbols: codes.iter().map(|code| code.value).collect(),
            max_len,
        }
    }
}

fn build_encode_lookup(codes: &[HuffmanCode]) -> std::collections::HashMap<u64, (u64, u8)> {
    codes.iter().map(|c| (c.value, (c.code, c.len))).collect()
}

fn huffman_lengths(freq: &std::collections::HashMap<u64, usize>) -> Vec<(u64, u8)> {
    let freq_vec: Vec<(u64, usize)> = freq.iter().map(|(v, c)| (*v, *c)).collect();
    huffman_code_lengths(&freq_vec)
}

fn write_code_lengths(out: &mut Vec<u8>, lengths: &[(u64, u8)]) -> io::Result<()> {
    write_varint(out, lengths.len() as u64)?;
    for (value, len) in lengths {
        write_varint(out, *value)?;
        out.push(*len);
    }
    Ok(())
}

fn read_code_lengths<R: Read>(reader: &mut R, available: u64) -> io::Result<Vec<(u64, u8)>> {
    let symbol_count = usize::try_from(read_varint(reader)?).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Huffman symbol count does not fit in memory",
        )
    })?;
    if symbol_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Huffman stream has zero symbols",
        ));
    }
    // Every entry needs at least a one-byte value and length.
    if symbol_count as u64 > available / 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Huffman table claims {symbol_count} symbols but only {available} bytes remain"
            ),
        ));
    }
    let mut lengths: Vec<(u64, u8)> = Vec::with_capacity(symbol_count);
    let mut symbols = std::collections::HashSet::with_capacity(symbol_count);
    for _ in 0..symbol_count {
        let value = read_varint(reader)?;
        if !symbols.insert(value) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("duplicate Huffman symbol: {value}"),
            ));
        }
        let mut len = [0];
        reader.read_exact(&mut len)?;
        lengths.push((value, len[0]));
    }
    Ok(lengths)
}

fn huffman_encode_payload(
    values: &[u64],
    lookup: &std::collections::HashMap<u64, (u64, u8)>,
) -> io::Result<Vec<u8>> {
    let mut writer = BitWriter::default();
    for v in values {
        let (code, len) = *lookup.get(v).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("value {} is missing from the Huffman table", v),
            )
        })?;
        writer.write_bits(code, len);
    }
    Ok(writer.finish())
}

fn huffman_decode_payload(
    buf: &[u8],
    num_items: usize,
    decoder: &HuffmanDecoder,
) -> io::Result<Vec<u64>> {
    if num_items > buf.len().saturating_mul(8) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Huffman item count exceeds the payload capacity",
        ));
    }
    let mut bit_reader = BitReader::new(buf);
    let mut output = Vec::with_capacity(num_items);
    for _ in 0..num_items {
        let mut code = 0u64;
        let mut symbol = None;
        for len in 1..=decoder.max_len {
            code = (code << 1) | bit_reader.read_bit()? as u64;
            let range = decoder.ranges[len as usize];
            if code >= range.first_code && code - range.first_code < range.count as u64 {
                let index = range.first_symbol + (code - range.first_code) as usize;
                symbol = Some(decoder.symbols[index]);
                break;
            }
        }
        output.push(symbol.ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "invalid Huffman bitstream")
        })?);
    }
    if !bit_reader.has_only_padding() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Huffman payload has trailing data",
        ));
    }
    Ok(output)
}

/// Per-record Huffman: [code lengths][payload]
fn local_huffman_encode(values: &[u64]) -> io::Result<Vec<u8>> {
    if values.is_empty() {
        return Ok(Vec::new());
    }
    let mut freq: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    for &v in values {
        *freq.entry(v).or_insert(0) += 1;
    }
    let mut lengths = huffman_lengths(&freq);
    let codes = build_canonical_codes(&mut lengths)?;
    let lookup = build_encode_lookup(&codes);

    let mut out = Vec::new();
    write_code_lengths(&mut out, &lengths)?;
    out.extend_from_slice(&huffman_encode_payload(values, &lookup)?);
    Ok(out)
}

fn local_huffman_decode(buf: &[u8], num_items: usize) -> io::Result<Vec<u64>> {
    if num_items == 0 {
        return Ok(Vec::new());
    }
    let mut reader = buf;
    let mut lengths = read_code_lengths(&mut reader, buf.len() as u64)?;
    let codes = build_canonical_codes(&mut lengths)?;
    let decoder = HuffmanDecoder::new(&codes);
    huffman_decode_payload(reader, num_items, &decoder)
}

// ============================================================================
// GLOBAL (FILE-LEVEL) HUFFMAN CODEBOOK
// ============================================================================

/// A Huffman code table stored once per file instead of once per record.
///
/// Records still decode independently: the table is read at open time and shared, so
/// random access is preserved while short records stop paying for their own table.
enum HuffmanCodec {
    Encoder(std::collections::HashMap<u64, (u64, u8)>),
    Decoder(HuffmanDecoder),
}

pub(crate) struct HuffmanTable {
    lengths: Vec<(u64, u8)>,
    codec: HuffmanCodec,
}

impl HuffmanTable {
    pub(crate) fn from_frequencies(
        freq: &std::collections::HashMap<u64, usize>,
    ) -> io::Result<Self> {
        if freq.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot build a Huffman table from an empty histogram",
            ));
        }
        let mut lengths = huffman_lengths(freq);
        let codes = build_canonical_codes(&mut lengths)?;
        Ok(Self {
            lengths,
            codec: HuffmanCodec::Encoder(build_encode_lookup(&codes)),
        })
    }

    fn from_lengths(mut lengths: Vec<(u64, u8)>) -> io::Result<Self> {
        let codes = build_canonical_codes(&mut lengths)?;
        Ok(Self {
            lengths,
            codec: HuffmanCodec::Decoder(HuffmanDecoder::new(&codes)),
        })
    }

    pub(crate) fn write<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        let mut buf = Vec::new();
        write_code_lengths(&mut buf, &self.lengths)?;
        write_varint(writer, buf.len() as u64)?;
        writer.write_all(&buf)?;
        Ok(())
    }

    pub(crate) fn read<R: Read>(reader: &mut R) -> io::Result<Option<Self>> {
        let len = read_varint(reader)?;
        if len == 0 {
            return Ok(None);
        }
        // Buffered: the table is parsed varint by varint, and an unbuffered File would cost one
        // syscall per byte (110 ms for a 42k-symbol table). take() still bounds what we read.
        let mut section = io::BufReader::new(reader.take(len));
        let lengths = read_code_lengths(&mut section, len)?;
        if section.get_ref().limit() != 0 || !section.buffer().is_empty() {
            let remaining = section.get_ref().limit() + section.buffer().len() as u64;
            let mut byte = [0];
            let kind = if section.read(&mut byte)? == 0 {
                io::ErrorKind::UnexpectedEof
            } else {
                io::ErrorKind::InvalidData
            };
            return Err(io::Error::new(
                kind,
                format!("Huffman table has {remaining} unparsed bytes"),
            ));
        }
        Ok(Some(Self::from_lengths(lengths)?))
    }

    fn encode(&self, values: &[u64]) -> io::Result<Vec<u8>> {
        match &self.codec {
            HuffmanCodec::Encoder(lookup) => huffman_encode_payload(values, lookup),
            HuffmanCodec::Decoder(_) => Err(io::Error::other("Huffman table is decode-only")),
        }
    }

    fn decode(&self, buf: &[u8], num_items: usize) -> io::Result<Vec<u64>> {
        match &self.codec {
            HuffmanCodec::Decoder(decoder) => huffman_decode_payload(buf, num_items, decoder),
            HuffmanCodec::Encoder(_) => Err(io::Error::other("Huffman table is encode-only")),
        }
    }
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
        CompressionStrategy::LocalHuffman(_) => {
            // Canonical Huffman on zigzagged deltas
            let deltas = delta_encode(vals);
            let zigzagged: Vec<u64> = deltas.iter().map(|&v| encode_zigzag(v)).collect();
            buf = local_huffman_encode(&zigzagged)?;
        }
        CompressionStrategy::Huffman(_) => {
            // Handled by the caller, which owns the file-level table
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "huffman requires a file-level table",
            ));
        }
        CompressionStrategy::LZ77(_) => {
            // LZ77-style sequence matching: find repeated sequences
            buf = lz77_encode(vals)?;
        }
        CompressionStrategy::EliasFano(_) => {
            // Elias-Fano on the prefix-summed (monotone) stream
            buf = encode_eliasfano_values(vals)?;
        }
        CompressionStrategy::Automatic(_) | CompressionStrategy::Benchmark(_, _, _) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Automatic/Benchmark must be resolved before encoding",
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
        CompressionStrategy::LocalHuffman(_) => {
            let zigzagged = local_huffman_decode(buf, num_items)?;
            Ok(reconstruct_from_zigzag_deltas(&zigzagged))
        }
        CompressionStrategy::Huffman(_) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "huffman requires a file-level table",
        )),
        CompressionStrategy::LZ77(_) => {
            // LZ77-style sequence matching decode
            lz77_decode(buf, num_items)
        }
        CompressionStrategy::EliasFano(_) => decode_eliasfano_values(buf, num_items),
        CompressionStrategy::Automatic(_) | CompressionStrategy::Benchmark(_, _, _) => {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Automatic/Benchmark must be resolved before decoding",
            ))
        }
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_standard_tracepoints<R: Read>(
    reader: &mut R,
    num_items: usize,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    tables: &HuffmanTables,
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

    let first_vals_u64 =
        decode_stream(&first_buf, num_items, first_strategy, tables.first.as_ref())?;
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
        CompressionStrategy::Automatic(_) | CompressionStrategy::Benchmark(_, _, _) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Automatic/Benchmark must be resolved before decoding",
            ))
        }
        other => decode_stream(&second_buf, num_items, other, tables.second.as_ref())?,
    };

    if second_vals_u64.len() != num_items || first_vals.len() != num_items {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "tracepoint decode length mismatch",
        ));
    }

    for (a, b) in first_vals.into_iter().zip(second_vals_u64) {
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write<W: Write>(
        &self,
        writer: &mut W,
        first_strategy: CompressionStrategy,
        second_strategy: CompressionStrategy,
        first_layer: CompressionLayer,
        second_layer: CompressionLayer,
        tables: &HuffmanTables,
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
            tables,
        )?;
        write_varint(writer, self.tags.len() as u64)?;
        for tag in &self.tags {
            tag.write(writer)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_tracepoints<W: Write>(
        &self,
        writer: &mut W,
        first_strategy: CompressionStrategy,
        second_strategy: CompressionStrategy,
        first_layer: CompressionLayer,
        second_layer: CompressionLayer,
        tables: &HuffmanTables,
    ) -> io::Result<()> {
        match &self.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                write_varint(writer, tps.len() as u64)?;
                let (first_vals, second_vals): (Vec<u64>, Vec<u64>) =
                    tps.iter().map(|(a, b)| (*a as u64, *b as u64)).unzip();

                let first_val_buf =
                    encode_stream(&first_vals, first_strategy, tables.first.as_ref())?;
                let second_val_buf = match &second_strategy {
                    CompressionStrategy::TwoDimDelta(_) => {
                        encode_2d_delta_second_values(&first_vals, &second_vals)?
                    }
                    CompressionStrategy::HybridRLE(_) => encode_rle_values(&second_vals)?,
                    _ => encode_stream(&second_vals, second_strategy, tables.second.as_ref())?,
                };

                // Apply the layer.
                let first_tmp;
                let first_compressed: &[u8] = if matches!(first_layer, CompressionLayer::Nocomp) {
                    &first_val_buf
                } else {
                    first_tmp = compress_with_layer(
                        &first_val_buf[..],
                        first_layer,
                        first_strategy.zstd_level(),
                    )?;
                    &first_tmp
                };
                let second_tmp;
                let second_compressed: &[u8] = if matches!(second_layer, CompressionLayer::Nocomp) {
                    &second_val_buf
                } else {
                    second_tmp = compress_with_layer(
                        &second_val_buf[..],
                        second_layer,
                        second_strategy.zstd_level(),
                    )?;
                    &second_tmp
                };

                write_varint(writer, first_compressed.len() as u64)?;
                writer.write_all(first_compressed)?;
                write_varint(writer, second_compressed.len() as u64)?;
                writer.write_all(second_compressed)?;
            }
            TracepointData::FastgaNoDiff(tps) => {
                // Single stream: [num_items][first_len][first_data]
                write_varint(writer, tps.len() as u64)?;
                let vals: Vec<u64> = tps.iter().map(|&v| v as u64).collect();
                let first_val_buf = encode_stream(&vals, first_strategy, tables.first.as_ref())?;
                let first_compressed = compress_with_layer(
                    &first_val_buf[..],
                    first_layer,
                    first_strategy.zstd_level(),
                )?;
                write_varint(writer, first_compressed.len() as u64)?;
                writer.write_all(&first_compressed)?;
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
        let buf = encode_tracepoint_values(&vals, CompressionStrategy::LocalHuffman(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::LocalHuffman(3))
                .unwrap();
        assert_eq!(vals, decoded);
    }

    #[test]
    fn file_table_roundtrip_and_size() {
        // FASTGA trace advances: near-constant, so a single file-level table beats a
        // per-record one, which is the whole point of the file-level table.
        let vals: Vec<u64> = (0..5_000)
            .map(|i| match i % 7 {
                0 => 97,
                1 => 109,
                2 => 92,
                _ => 100,
            })
            .collect();
        let mut freq = std::collections::HashMap::new();
        for &v in &vals {
            *freq.entry(v).or_insert(0) += 1;
        }
        let cb = HuffmanTable::from_frequencies(&freq).unwrap();
        assert_eq!(cb.lengths.len(), 4);

        let payload = cb.encode(&vals).unwrap();

        // the payload carries no table, so it must be smaller than the per-record encoding
        let per_record =
            encode_tracepoint_values(&vals, CompressionStrategy::LocalHuffman(3)).unwrap();
        assert!(payload.len() < per_record.len());

        // serialization round-trip
        let mut buf = Vec::new();
        cb.write(&mut buf).unwrap();
        let mut slice = &buf[..];
        let back = HuffmanTable::read(&mut slice).unwrap().unwrap();
        assert_eq!(back.decode(&payload, vals.len()).unwrap(), vals);
    }

    #[test]
    fn benchmark_samples_are_compact() {
        let values = vec![7; 10_000];
        let mut samples = HuffmanSamples::default();
        samples.push(&values).unwrap();
        assert!(samples.encoded.len() < values.len() * size_of::<u64>());
        assert!(huffman_state(&samples, &[CompressionLayer::Nocomp; 3], 3, 1).is_some());
    }

    #[test]
    fn code_lengths_ignore_input_order() {
        // the heap key must be a total order, so the order entries arrive in cannot matter
        let base: Vec<(u64, usize)> = vec![
            (5, 7),
            (9, 7),
            (2, 7),
            (11, 3),
            (4, 3),
            (8, 12),
            (1, 1),
            (3, 1),
            (6, 1),
        ];
        let mut expected = huffman_code_lengths(&base);
        expected.sort();
        for rot in 1..base.len() {
            let mut permuted = base.clone();
            permuted.rotate_left(rot);
            let mut got = huffman_code_lengths(&permuted);
            got.sort();
            assert_eq!(got, expected, "rotation by {rot} changed the code lengths");
        }
    }

    #[test]
    fn code_lengths_are_deterministic_and_optimal() {
        use std::collections::HashMap;
        // equal frequencies: HashMap iteration order must not leak into the code lengths
        let freq: HashMap<u64, usize> = (0..6u64).map(|v| (v, 10)).collect();
        let mut first = huffman_lengths(&freq);
        for _ in 0..20 {
            let again: HashMap<u64, usize> = (0..6u64).map(|v| (v, 10)).collect();
            assert_eq!(
                huffman_lengths(&again),
                first,
                "code lengths must be reproducible"
            );
        }
        // and still a valid, non-oversubscribed code
        assert!(build_canonical_codes(&mut first).is_ok());

        // skewed distribution: the frequent symbol must get the shortest code
        let skewed: HashMap<u64, usize> = HashMap::from([(7, 1000), (8, 5), (9, 3), (10, 2)]);
        let mut lens = huffman_lengths(&skewed);
        let len_of = |v: u64| lens.iter().find(|(x, _)| *x == v).unwrap().1;
        assert!(len_of(7) < len_of(9), "frequent symbol should be shorter");
        assert!(build_canonical_codes(&mut lens).is_ok());
    }

    #[test]
    fn oversubscribed_code_lengths_are_rejected() {
        // three symbols cannot all have 1-bit codes
        let err = build_canonical_codes(&mut [(10, 1), (20, 1), (30, 1)])
            .err()
            .expect("oversubscribed lengths must be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        // the valid shape for three symbols still works
        assert!(build_canonical_codes(&mut [(10, 1), (20, 2), (30, 2)]).is_ok());
    }

    #[test]
    fn canonical_decoder_rejects_unused_codes() {
        let codes = build_canonical_codes(&mut [(7, 2)]).unwrap();
        let decoder = HuffmanDecoder::new(&codes);
        assert!(huffman_decode_payload(&[0b1000_0000], 1, &decoder).is_err());
        assert!(huffman_decode_payload(&[0, 0], 1, &decoder).is_err());
        assert!(huffman_decode_payload(&[1], 1, &decoder).is_err());
        assert!(huffman_decode_payload(&[0], usize::MAX, &decoder).is_err());
    }

    #[test]
    fn malformed_table_length_is_rejected() {
        // outer length claims a terabyte, only three bytes follow
        let mut buf = Vec::new();
        write_varint(&mut buf, 1u64 << 40).unwrap();
        buf.extend_from_slice(&[1, 2, 3]);
        let mut slice = &buf[..];
        let err = match HuffmanTable::read(&mut slice) {
            Err(e) => e,
            Ok(_) => panic!("a bogus table length must not parse"),
        };
        assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn exact_skip_rejects_short_input() {
        let mut input = &b"abc"[..];
        assert_eq!(
            skip_exact(&mut input, 4).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn malformed_symbol_count_is_rejected() {
        // outer length is honest, but the symbol count is far larger than the bytes allow
        let mut inner = Vec::new();
        write_varint(&mut inner, 1_000_000u64).unwrap();
        inner.extend_from_slice(&[0u8; 8]);
        let mut buf = Vec::new();
        write_varint(&mut buf, inner.len() as u64).unwrap();
        buf.extend_from_slice(&inner);
        let mut slice = &buf[..];
        let err = match HuffmanTable::read(&mut slice) {
            Err(e) => e,
            Ok(_) => panic!("a bogus symbol count must not parse"),
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn malformed_table_contents_are_rejected() {
        for inner in [
            vec![2, 7, 1, 7, 2], // duplicate symbol
            vec![1, 7, 1, 0],    // trailing byte
        ] {
            let mut buf = Vec::new();
            write_varint(&mut buf, inner.len() as u64).unwrap();
            buf.extend_from_slice(&inner);
            let err = match HuffmanTable::read(&mut &buf[..]) {
                Err(e) => e,
                Ok(_) => panic!("malformed table contents must not parse"),
            };
            assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn absent_table_marker() {
        let mut buf = Vec::new();
        write_varint(&mut buf, 0).unwrap();
        let mut slice = &buf[..];
        assert!(HuffmanTable::read(&mut slice).unwrap().is_none());
    }

    #[test]
    fn huffman_without_table_is_an_error() {
        let vals = vec![1u64, 2, 3];
        assert!(encode_tracepoint_values(&vals, CompressionStrategy::Huffman(3)).is_err());
        assert!(decode_tracepoint_values(&[0u8; 4], 3, CompressionStrategy::Huffman(3)).is_err());
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

    fn ef_roundtrip(vals: &[u64]) {
        let buf = encode_tracepoint_values(vals, CompressionStrategy::EliasFano(3)).unwrap();
        let decoded =
            decode_tracepoint_values(&buf, vals.len(), CompressionStrategy::EliasFano(3)).unwrap();
        assert_eq!(vals, &decoded[..], "EF round-trip mismatch");
    }

    #[test]
    fn eliasfano_roundtrip_cases() {
        ef_roundtrip(&[]); // empty
        ef_roundtrip(&[0]); // single zero
        ef_roundtrip(&[42]); // single nonzero
        ef_roundtrip(&[5, 7, 9, 15, 15, 200, 210, 205, 300, 301]); // varied
        ef_roundtrip(&[0, 0, 0, 0]); // all zeros (equal cumulative)
        ef_roundtrip(&[1, 0, 2, 0, 3, 0]); // interleaved zeros (pure-indel segments)
        ef_roundtrip(&[100, 100, 100, 100]); // dense, constant
        ef_roundtrip(&[1, 1, 1, 1, 1, 1, 1, 1]); // dense unit steps
        ef_roundtrip(&[0, 1_000_000, 5, 2_000_000]); // large gaps
        ef_roundtrip(&[97, 103, 99, 101, 100, 98, 102, 100, 100]); // FL-TP-like diffs
    }

    #[test]
    fn eliasfano_known_answer() {
        // vals [0,2,3] -> cumulative [0,2,5], n=3, u=6, l=floor(log2(2))=1.
        // low bits (1 each): 0,0,1 ; high gaps unary: 0 | 10 | 10 -> bitstream 00101010 = 0x2A.
        // header byte l=1, then one data byte 0x2A.
        let buf =
            encode_tracepoint_values(&[0u64, 2, 3], CompressionStrategy::EliasFano(3)).unwrap();
        assert_eq!(buf, vec![1u8, 0x2A]);
        let decoded = decode_tracepoint_values(&buf, 3, CompressionStrategy::EliasFano(3)).unwrap();
        assert_eq!(decoded, vec![0u64, 2, 3]);
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
            let huffman = encode_tracepoint_values(&vals, CompressionStrategy::LocalHuffman(3))
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
