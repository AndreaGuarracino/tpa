//! TracePoint Alignment (TPA) format for efficient storage of sequence alignments with tracepoints
//!
//! Format: `Header -> Huffman tables (v2 only) -> StringTable -> Records -> Footer`
//! - Header: Magic "TPA\0" + version + metadata
//! - StringTable: Deduplicated sequence names with lengths
//! - Records: PAF fields + compressed tracepoints
//! - Footer: Crash-safety marker with record/string counts

mod binary;
mod cigar;
mod format;
mod hybrids;
mod index;
mod reader;
mod utils;

use log::{info, warn};
use rayon::prelude::*;

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, Write};

// Re-export public types
pub use lib_wfa2::affine_wavefront::Distance;
use std::collections::HashMap;
use tracepoints::{
    cigar_to_mixed_tracepoints, cigar_to_tracepoints, cigar_to_tracepoints_fastga,
    cigar_to_tracepoints_fastga_nodiff, cigar_to_tracepoints_fastga_with_contigs,
    cigar_to_variable_tracepoints,
};

pub use format::{
    CompressionConfig, CompressionLayer, CompressionStrategy, StringTable, Tag, TagValue,
    TpaHeader, TPA_MAGIC,
};
pub use tracepoints::{ComplexityMetric, MixedRepresentation, TracepointData, TracepointType};

use crate::format::{parse_tag, CompactRecord, TpaFooter};
use crate::utils::{parse_u8, parse_usize};

pub use index::{build_index_all_records, build_index_per_record, TpaIndex};
pub use reader::{
    read_mixed_tracepoints_at_offset, read_mixed_tracepoints_at_vpos,
    read_standard_tracepoints_at_offset_with_strategies,
    read_standard_tracepoints_at_offset_with_tables, read_standard_tracepoints_at_vpos,
    read_standard_tracepoints_at_vpos_with_tables, read_variable_tracepoints_at_offset,
    read_variable_tracepoints_at_vpos, AlignmentRecord, AlignmentRecordIterator, RecordIterator,
    StreamingMetadataIterator, TpaReader,
};

// Re-export cigar module types and functions
pub use cigar::{
    calculate_alignment_score, reconstruct_cigar, reconstruct_cigar_with_aligner,
    reconstruct_cigar_with_heuristic, CigarStats,
};

// Re-export aligner type for reconstruct_cigar_with_aligner API
pub use lib_wfa2::affine_wavefront::AffineWavefronts;

// Re-export noodles bgzf for BGZF mode standalone access
pub use noodles::bgzf;

// Re-export utility functions for external tools
pub use utils::{read_varint, varint_size};

pub use binary::{encode_two_dim_delta, HuffmanStreamCodec, HuffmanTables};

use crate::binary::{
    automatic_select, decompress_varint, read_header_and_tables, read_record,
    write_paf_line_with_tracepoints, HuffmanTable, StrategyAnalyzer,
};

use crate::utils::{is_compressed_paf, open_paf_reader};

type HuffmanHistograms = [HashMap<u64, usize>; 2];

#[derive(Clone, Copy)]
struct CompressionPlan {
    first: CompressionStrategy,
    second: CompressionStrategy,
    first_layer: CompressionLayer,
    second_layer: CompressionLayer,
    tp_type: TracepointType,
    metric: ComplexityMetric,
    max_complexity: u32,
    distance: Distance,
    all_records_level: u32,
}

impl CompressionPlan {
    fn header(self, records: u64, strings: u64, all_records: bool) -> io::Result<TpaHeader> {
        TpaHeader::new(
            records,
            strings,
            [
                (self.first, self.first_layer),
                (self.second, self.second_layer),
            ],
            (self.tp_type, self.metric, self.max_complexity),
            self.distance,
            all_records,
        )
    }
}

fn reject_same_file(input: &str, output: &str) -> io::Result<()> {
    if input == "-" || output == "-" {
        return Ok(());
    }
    let mut same = input == output
        || matches!(
            (std::fs::canonicalize(input), std::fs::canonicalize(output)),
            (Ok(input), Ok(output)) if input == output
        );
    #[cfg(unix)]
    if let (Ok(input), Ok(output)) = (std::fs::metadata(input), std::fs::metadata(output)) {
        use std::os::unix::fs::MetadataExt;
        same |= input.dev() == output.dev() && input.ino() == output.ino();
    }
    if same {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "input and output must be different files",
        ));
    }
    Ok(())
}

struct TempInput(std::path::PathBuf);

impl Drop for TempInput {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn spool_stdin() -> io::Result<TempInput> {
    use std::fs::OpenOptions;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    loop {
        let path = std::env::temp_dir().join(format!(
            "tpa-stdin-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(mut file) => {
                let input = TempInput(path);
                io::copy(&mut io::stdin().lock(), &mut file)?;
                return Ok(input);
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

fn plain_ranges(size: u64) -> Vec<(u64, u64)> {
    const MIN_RANGE_BYTES: u64 = 64 * 1024;
    let workers = (rayon::current_num_threads() as u64)
        .min(size.div_ceil(MIN_RANGE_BYTES))
        .max(1);
    let boundary = |i| ((size as u128 * i as u128) / workers as u128) as u64;
    (0..workers)
        .map(|i| (boundary(i), boundary(i + 1)))
        .filter(|(beg, end)| beg < end)
        .collect()
}

/// Visit every complete line whose first byte is in `[beg, end)`.
fn for_each_plain_line<F>(input_path: &str, beg: u64, end: u64, mut f: F) -> io::Result<()>
where
    F: FnMut(u64, &str) -> io::Result<()>,
{
    let mut file = File::open(input_path)?;
    let start = beg.saturating_sub(1);
    file.seek(io::SeekFrom::Start(start))?;
    let mut reader = BufReader::new(file);
    let mut buf = Vec::new();
    let mut pos = start;
    if beg != 0 {
        pos += reader.read_until(b'\n', &mut buf)? as u64;
    }
    while pos < end {
        buf.clear();
        let line_start = pos;
        let n = reader.read_until(b'\n', &mut buf)?;
        if n == 0 {
            break;
        }
        pos += n as u64;
        while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
            buf.pop();
        }
        if let Ok(line) = std::str::from_utf8(&buf) {
            if !line.trim().is_empty() && !line.starts_with('#') {
                f(line_start, line)?;
            }
        }
    }
    Ok(())
}

fn validate_cigar(cigar: &str, tp_type: TracepointType) -> io::Result<()> {
    let special = matches!(
        tp_type,
        TracepointType::Mixed | TracepointType::Fastga | TracepointType::FastgaNoDiff
    );
    let mut len = 0usize;
    let mut digits = false;
    let mut total = 0usize;
    for byte in cigar.bytes() {
        if byte.is_ascii_digit() {
            len = len
                .checked_mul(10)
                .and_then(|n| n.checked_add((byte - b'0') as usize))
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "CIGAR length overflow")
                })?;
            digits = true;
        } else {
            let valid = matches!(byte, b'M' | b'I' | b'D' | b'=' | b'X')
                || special && matches!(byte, b'H' | b'N' | b'P' | b'S');
            if !digits || len == 0 || !valid {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid CIGAR"));
            }
            total = total.checked_add(len).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "CIGAR length overflow")
            })?;
            len = 0;
            digits = false;
        }
    }
    if digits || total == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid CIGAR"));
    }
    Ok(())
}

fn validate_interval(start: u64, end: u64, len: u64, name: &str) -> io::Result<()> {
    if start > end || end > len || usize::try_from(len).is_err() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid {name} interval {start}..{end} for length {len}"),
        ));
    }
    Ok(())
}

// ============================================================================
// PUBLIC API
// ============================================================================

pub fn is_tpa_file(path: &str) -> io::Result<bool> {
    if path == "-" {
        return Ok(false);
    }

    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    match file.read_exact(&mut magic) {
        Ok(()) => Ok(&magic == TPA_MAGIC),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e),
    }
}

fn intern_valid_paf_names(
    line: &str,
    table: &mut StringTable,
    tp_type: TracepointType,
) -> io::Result<()> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 12 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("PAF line has {} fields, expected at least 12", fields.len()),
        ));
    }
    let query_len = parse_usize(fields[1], "query_len")?;
    let query_start = parse_usize(fields[2], "query_start")?;
    let query_end = parse_usize(fields[3], "query_end")?;
    validate_interval(query_start, query_end, query_len, "query")?;
    if !matches!(fields[4], "+" | "-") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Strand must be '+' or '-'",
        ));
    }
    let target_len = parse_usize(fields[6], "target_len")?;
    let target_start = parse_usize(fields[7], "target_start")?;
    let target_end = parse_usize(fields[8], "target_end")?;
    validate_interval(target_start, target_end, target_len, "target")?;
    for (value, name) in [
        (fields[9], "residue_matches"),
        (fields[10], "alignment_block_len"),
    ] {
        parse_usize(value, name)?;
    }
    parse_u8(fields[11], "mapping_quality")?;

    let mut cigar = None;
    let mut tracepoints = None;
    for field in &fields[12..] {
        if let Some(value) = field.strip_prefix("cg:Z:") {
            cigar = Some(value);
        }
        if let Some(value) = field.strip_prefix("tp:Z:") {
            tracepoints = Some(value);
        }
    }
    if let Some(value) = tracepoints {
        parse_tracepoints(value, tp_type)?;
    } else if let Some(value) = cigar {
        validate_cigar(value, tp_type)?;
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PAF line has neither tp:Z: (tracepoints) nor cg:Z: (CIGAR) tag",
        ));
    }

    table.get_or_insert_id(fields[0], query_len);
    table.get_or_insert_id(fields[5], target_len);
    Ok(())
}

/// Parallel pass 1; ordered merging preserves deterministic string IDs.
#[allow(clippy::too_many_arguments)]
fn parallel_pass1_plain(
    input_path: &str,
    tp_type: TracepointType,
    max_complexity: u32,
    complexity_metric: ComplexityMetric,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
    huffman_streams: [bool; 2],
) -> io::Result<(u64, StringTable, HuffmanHistograms)> {
    let size = std::fs::metadata(input_path)?.len();
    if size == 0 {
        return Ok((0, StringTable::new(), [HashMap::new(), HashMap::new()]));
    }
    let ranges = plain_ranges(size);

    let partials: Vec<io::Result<(u64, StringTable, HuffmanHistograms)>> = ranges
        .par_iter()
        .map(
            |&(beg, end)| -> io::Result<(u64, StringTable, HuffmanHistograms)> {
                let mut local = StringTable::new();
                let mut hist = [HashMap::new(), HashMap::new()];
                let mut count = 0u64;
                let full_parse = huffman_streams.contains(&true)
                    || contig_table.is_some() && tp_type == TracepointType::Fastga;
                for_each_plain_line(input_path, beg, end, |_, line| {
                    if full_parse {
                        if let Ok(records) = parse_paf(
                            line,
                            &mut local,
                            tp_type,
                            max_complexity,
                            complexity_metric,
                            contig_table,
                        ) {
                            count += records.len() as u64;
                            accumulate_huffman_histograms(&mut hist, &records, huffman_streams);
                        }
                    } else if intern_valid_paf_names(line, &mut local, tp_type).is_ok() {
                        count += 1;
                    }
                    Ok(())
                })?;
                Ok((count, local, hist))
            },
        )
        .collect();

    let mut total = 0u64;
    let mut merged = StringTable::new();
    let mut hist = [HashMap::new(), HashMap::new()];
    for r in partials {
        let (c, local, local_hist) = r?;
        total += c;
        for id in 0..local.len() as u64 {
            if let Some((name, len)) = local.get_name_and_len(id) {
                merged.get_or_insert_id(name, len);
            }
        }
        for (stream, counts) in local_hist.into_iter().enumerate() {
            for (value, count) in counts {
                *hist[stream].entry(value).or_insert(0) += count;
            }
        }
    }
    Ok((total, merged, hist))
}

/// Convert a PAF file to TPA format
pub fn paf_to_tpa(
    input_path: &str,
    output_path: &str,
    config: CompressionConfig,
) -> io::Result<()> {
    reject_same_file(input_path, output_path)?;
    if matches!(
        config.tp_type,
        TracepointType::Fastga | TracepointType::FastgaNoDiff
    ) && config.max_complexity == 0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Fastga trace spacing must be greater than zero",
        ));
    }
    if config.all_records && config.all_records_level > 9 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "BGZIP compression level must be between 0 and 9",
        ));
    }
    info!(
        "Converting PAF to TPA using strategies {} / {}...",
        config.first_strategy, config.second_strategy
    );

    let (uses_automatic, benchmark) = match (config.first_strategy, config.second_strategy) {
        (CompressionStrategy::Automatic(a), CompressionStrategy::Automatic(b)) if a == b => {
            (true, None)
        }
        (CompressionStrategy::Benchmark(a, an, al), CompressionStrategy::Benchmark(b, bn, bl))
            if a == b && an == bn && al == bl =>
        {
            (false, Some((a, an, al)))
        }
        (a, b)
            if matches!(
                a,
                CompressionStrategy::Automatic(_) | CompressionStrategy::Benchmark(_, _, _)
            ) || matches!(
                b,
                CompressionStrategy::Automatic(_) | CompressionStrategy::Benchmark(_, _, _)
            ) =>
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "both streams must use the same automatic or benchmark configuration",
            ));
        }
        _ => (false, None),
    };

    // Variable and mixed layouts do not expose the two file-table streams.
    // Resolve Automatic here: the histogram pre-pass below must know whether the strategy
    // it will end up choosing needs a file-level table, otherwise the table is never built.
    let requested_huffman_streams = if uses_automatic {
        let (a, b, _, _) = binary::automatic_select(
            config.tp_type,
            config.complexity_metric,
            config.first_strategy.zstd_level(),
        );
        [a.uses_file_table(), b.uses_file_table()]
    } else {
        [
            config.first_strategy.uses_file_table(),
            config.second_strategy.uses_file_table(),
        ]
    };
    let uses_file_table = requested_huffman_streams[0] || requested_huffman_streams[1];
    if uses_file_table
        && matches!(
            config.tp_type,
            TracepointType::Variable | TracepointType::Mixed
        )
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "huffman is not supported for {:?} tracepoints",
                config.tp_type
            ),
        ));
    }
    if input_path == "-" {
        let input = spool_stdin()?;
        return paf_to_tpa(input.0.to_string_lossy().as_ref(), output_path, config);
    }

    // Pass 1: Build string table + collect sample for analysis
    let mut string_table = StringTable::new();
    let mut record_count = 0u64;

    // For all-records mode, we skip layer testing since layers are not used
    let test_layers = !config.all_records;
    let mut analyzer = benchmark
        .map(|(level, samples, layer)| StrategyAnalyzer::new(level, samples, test_layers, layer));
    let histogram_streams = if uses_file_table {
        requested_huffman_streams
    } else if benchmark.is_some()
        && matches!(
            config.tp_type,
            TracepointType::Standard | TracepointType::Fastga | TracepointType::FastgaNoDiff
        )
    {
        [
            true,
            !matches!(config.tp_type, TracepointType::FastgaNoDiff),
        ]
    } else {
        [false; 2]
    };
    let collect_huffman_histogram = histogram_streams.contains(&true);
    let mut huffman_histogram = [HashMap::new(), HashMap::new()];

    let contig_table = config.contig_table.as_ref();
    let multi_record = contig_table.is_some() && config.tp_type == TracepointType::Fastga;

    let parse_record =
        |line: &str, line_num: usize, table: &mut StringTable| -> Option<Vec<CompactRecord>> {
            match parse_paf(
                line,
                table,
                config.tp_type,
                config.max_complexity,
                config.complexity_metric,
                contig_table,
            ) {
                Ok(r) => Some(r),
                Err(e) => {
                    warn!("Skipping malformed line {}: {}", line_num + 1, e);
                    None
                }
            }
        };

    // No analyzer (no order-dependent sampling) + plain seekable input => parallel byte-range pass 1.
    // Gz/bgz/stdin are not seekable into, and the benchmark analyzer samples the first N records in
    // order, so those keep the sequential scan below.
    let plain_seekable = input_path != "-" && !is_compressed_paf(input_path);
    if analyzer.is_none() && plain_seekable {
        let (rc, st, hist) = parallel_pass1_plain(
            input_path,
            config.tp_type,
            config.max_complexity,
            config.complexity_metric,
            contig_table,
            if collect_huffman_histogram {
                histogram_streams
            } else {
                [false; 2]
            },
        )?;
        record_count = rc;
        string_table = st;
        huffman_histogram = hist;
    } else {
        let mut input = open_paf_reader(input_path)?;
        let mut buf: Vec<u8> = Vec::new(); // reused line buffer (avoid a String alloc per line)
        let mut line_num = 0usize;
        loop {
            buf.clear();
            if input.read_until(b'\n', &mut buf)? == 0 {
                break;
            }
            let cur = line_num;
            line_num += 1;
            while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
                buf.pop();
            }
            let line = match std::str::from_utf8(&buf) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            if analyzer.is_none() && !collect_huffman_histogram && !multi_record {
                match intern_valid_paf_names(line, &mut string_table, config.tp_type) {
                    Ok(()) => record_count += 1,
                    Err(e) => warn!("Skipping malformed line {}: {}", cur + 1, e),
                }
                continue;
            }

            let Some(records) = parse_record(line, cur, &mut string_table) else {
                continue;
            };
            if collect_huffman_histogram {
                accumulate_huffman_histograms(&mut huffman_histogram, &records, histogram_streams);
            }
            if let Some(an) = analyzer.as_mut() {
                for record in &records {
                    an.analyze_record(record)?;
                }
            }
            record_count += records.len() as u64;
        }
    }

    // Verify file is not empty
    if record_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No records found in PAF file",
        ));
    }

    // Choose strategy: Automatic, Benchmark, or explicit
    let (chosen_first, chosen_second, first_layer, second_layer) = if uses_automatic {
        let level = config.first_strategy.zstd_level();
        let result = automatic_select(config.tp_type, config.complexity_metric, level);
        info!(
            "Automatic: {} [{}] → {} [{}]",
            result.0,
            result.2.as_str(),
            result.1,
            result.3.as_str()
        );
        result
    } else {
        match analyzer {
            Some(analyzer) => analyzer.select_best(),
            None => (
                config.first_strategy,
                config.second_strategy,
                config.first_layer,
                config.second_layer,
            ),
        }
    };
    let chosen_huffman_streams = [
        chosen_first.uses_file_table(),
        chosen_second.uses_file_table(),
    ];
    let tables = if chosen_huffman_streams[0] || chosen_huffman_streams[1] {
        finish_huffman_tables(
            huffman_histogram,
            chosen_huffman_streams,
            config.tp_type,
            input_path,
        )?
    } else {
        HuffmanTables::default()
    };
    let plan = CompressionPlan {
        first: chosen_first,
        second: chosen_second,
        first_layer,
        second_layer,
        tp_type: config.tp_type,
        metric: config.complexity_metric,
        max_complexity: config.max_complexity,
        distance: config.distance,
        all_records_level: config.all_records_level,
    };

    if config.all_records {
        // All-records mode: header/string table plain, records BGZIP-compressed all together
        compress_all_records(
            input_path,
            output_path,
            record_count,
            &string_table,
            plan,
            config.contig_table.as_ref(),
            tables,
        )
    } else {
        // Per-record compression with layers
        compress_per_record(
            input_path,
            output_path,
            record_count,
            &string_table,
            plan,
            config.contig_table.as_ref(),
            tables,
        )
    }
}

pub fn tpa_to_paf(input_path: &str, output_path: &str) -> io::Result<()> {
    reject_same_file(input_path, output_path)?;
    info!("Converting {} to PAF format...", input_path);

    let mut file = File::open(input_path)?;
    let header = TpaHeader::read(&mut file)?;
    drop(file);

    if header.all_records() {
        decompress_all_records(input_path, output_path)
    } else {
        decompress_per_record(input_path, output_path)
    }
}

/// Parse a tracepoint string (tp:Z: field value) into TracepointData
pub fn parse_tracepoints(tp_str: &str, tp_type: TracepointType) -> io::Result<TracepointData> {
    match tp_type {
        TracepointType::Standard | TracepointType::Fastga => {
            let mut tps = Vec::new();
            for part in tp_str.split(';') {
                let pair = parse_tracepoint_pair(part, true)?;
                tps.push(pair);
            }

            Ok(match tp_type {
                TracepointType::Standard => TracepointData::Standard(tps),
                TracepointType::Fastga => TracepointData::Fastga(tps),
                _ => unreachable!(),
            })
        }
        TracepointType::FastgaNoDiff => {
            let mut tps = Vec::new();
            for part in tp_str.split(';') {
                tps.push(parse_usize_value(part, "Invalid FastgaNoDiff value")?);
            }
            Ok(TracepointData::FastgaNoDiff(tps))
        }
        TracepointType::Mixed => {
            let mut items = Vec::new();

            for part in tp_str.split(';') {
                if part.contains(',') {
                    let (first, second) = parse_tracepoint_pair(part, true)?;
                    items.push(MixedRepresentation::Tracepoint(first, second));
                } else {
                    let op = part.chars().last().ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Mixed tracepoint entry missing operation",
                        )
                    })?;
                    let len_str = part.strip_suffix(op).map(str::trim).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Invalid mixed tracepoint: '{}'", part),
                        )
                    })?;
                    if len_str.is_empty() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Mixed tracepoint entry missing length",
                        ));
                    }
                    let len = parse_usize_value(len_str, "Invalid CIGAR length")?;
                    items.push(MixedRepresentation::CigarOp(len, op));
                }
            }

            Ok(TracepointData::Mixed(items))
        }
        TracepointType::Variable => {
            let mut tps = Vec::new();
            for part in tp_str.split(';') {
                if part.contains(',') {
                    let (first, second) = parse_tracepoint_pair(part, true)?;
                    tps.push((first, Some(second)));
                } else {
                    let first = parse_usize_value(part, "Invalid first value")?;
                    tps.push((first, None));
                }
            }
            Ok(TracepointData::Variable(tps))
        }
    }
}

/// Format tracepoints as a string (inverse of parse_tracepoints)
///
/// Handles all TracepointData variants:
/// - Standard/Fastga: "a1,b1;a2,b2;..."
/// - Mixed: "a,b;10I;c,d;5D;..."
/// - Variable: "a1,b1;a2;a3,b3;..." (second value optional)
///
/// Format fastga/standard `(a,b)` tracepoints as `"a,b;a,b;..."` in a single allocation, from a
/// slice. Avoids cloning the Vec into a `TracepointData` and the per-element `String` that
/// `format_tracepoints` allocates. Hot in fastga encode (millions of tracepoints per file).
pub fn format_ab_tracepoints(tps: &[(usize, usize)]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(tps.len() * 8);
    for (i, (a, b)) in tps.iter().enumerate() {
        if i > 0 {
            s.push(';');
        }
        let _ = write!(s, "{},{}", a, b);
    }
    s
}

pub fn format_tracepoints(data: &TracepointData) -> String {
    match data {
        TracepointData::Standard(tps) | TracepointData::Fastga(tps) => format_ab_tracepoints(tps),
        TracepointData::FastgaNoDiff(tps) => tps
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>()
            .join(";"),
        TracepointData::Mixed(items) => items
            .iter()
            .map(|tp| match tp {
                MixedRepresentation::Tracepoint(a, b) => format!("{},{}", a, b),
                MixedRepresentation::CigarOp(len, op) => format!("{}{}", len, op),
            })
            .collect::<Vec<String>>()
            .join(";"),
        TracepointData::Variable(tps) => tps
            .iter()
            .map(|(a, b_opt)| match b_opt {
                Some(b) => format!("{},{}", a, b),
                None => format!("{}", a),
            })
            .collect::<Vec<String>>()
            .join(";"),
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn accumulate_huffman_histograms(
    hist: &mut HuffmanHistograms,
    records: &[CompactRecord],
    streams: [bool; 2],
) {
    for record in records {
        match &record.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                for &(first, second) in tps {
                    if streams[0] {
                        *hist[0].entry(first as u64).or_insert(0) += 1;
                    }
                    if streams[1] {
                        *hist[1].entry(second as u64).or_insert(0) += 1;
                    }
                }
            }
            TracepointData::FastgaNoDiff(values) if streams[0] => {
                for &value in values {
                    *hist[0].entry(value as u64).or_insert(0) += 1;
                }
            }
            _ => {}
        }
    }
}

fn finish_huffman_tables(
    hist: HuffmanHistograms,
    streams: [bool; 2],
    tp_type: TracepointType,
    input_path: &str,
) -> io::Result<HuffmanTables> {
    let second_stream_used = matches!(tp_type, TracepointType::Standard | TracepointType::Fastga);
    for stream in 0..2 {
        if streams[stream] && (stream == 0 || second_stream_used) && hist[stream].is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "huffman: no values found for stream {stream} while scanning '{input_path}'"
                ),
            ));
        }
    }

    let mut tables = [None, None];
    for (stream, table) in tables.iter_mut().enumerate() {
        if streams[stream] && !hist[stream].is_empty() {
            *table = Some(HuffmanTable::from_frequencies(&hist[stream])?);
        }
    }
    let [first, second] = tables;
    let tables = HuffmanTables::new(first, second);
    let (first_count, second_count) = tables.symbol_counts();
    info!(
        "Built file-level Huffman tables: first {} symbols, second {} symbols",
        first_count, second_count
    );
    Ok(tables)
}

#[derive(Default)]
struct EncodedRange {
    bytes: Vec<u8>,
    offsets: Vec<u64>,
}

impl EncodedRange {
    #[inline(always)]
    fn extend(
        &mut self,
        records: &[CompactRecord],
        plan: CompressionPlan,
        tables: &HuffmanTables,
    ) -> io::Result<()> {
        for record in records {
            self.offsets.push(self.bytes.len() as u64);
            record.write(
                &mut self.bytes,
                plan.first,
                plan.second,
                plan.first_layer,
                plan.second_layer,
                tables,
            )?;
        }
        Ok(())
    }
}

fn encode_plain_range(
    input_path: &str,
    range: (u64, u64),
    string_table: &StringTable,
    plan: CompressionPlan,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
    tables: &HuffmanTables,
) -> io::Result<EncodedRange> {
    // Avoid transient doubling of large per-range buffers.
    let mut encoded = EncodedRange {
        bytes: Vec::with_capacity(((range.1 - range.0) / 3) as usize),
        offsets: Vec::new(),
    };
    for_each_plain_line(input_path, range.0, range.1, |pos, line| {
        let records = match parse_paf_readonly(
            line,
            string_table,
            plan.tp_type,
            plan.max_complexity,
            plan.metric,
            contig_table,
        ) {
            Ok(records) => records,
            Err(e) => {
                warn!("Skipping malformed line at byte {}: {}", pos, e);
                return Ok(());
            }
        };
        encoded.extend(&records, plan, tables)?;
        Ok(())
    })?;
    Ok(encoded)
}

fn compress_plain_records<W: Write>(
    input_path: &str,
    writer: &mut W,
    mut base: u64,
    string_table: &StringTable,
    plan: CompressionPlan,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
    tables: &HuffmanTables,
) -> io::Result<Vec<u64>> {
    let ranges: Vec<_> = plain_ranges(std::fs::metadata(input_path)?.len())
        .into_par_iter()
        .map(|range| {
            encode_plain_range(input_path, range, string_table, plan, contig_table, tables)
        })
        .collect();

    let mut byte_offsets = Vec::with_capacity(
        ranges
            .iter()
            .filter_map(|range| range.as_ref().ok())
            .map(|range| range.offsets.len())
            .sum(),
    );
    for range in ranges {
        let range = range?;
        byte_offsets.extend(range.offsets.into_iter().map(|offset| base + offset));
        writer.write_all(&range.bytes)?;
        base += range.bytes.len() as u64;
    }
    Ok(byte_offsets)
}

fn compress_per_record(
    input_path: &str,
    output_path: &str,
    record_count: u64,
    string_table: &StringTable,
    plan: CompressionPlan,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
    tables: HuffmanTables,
) -> io::Result<()> {
    let header = plan.header(record_count, string_table.len() as u64, false)?;

    let mut output = File::create(output_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{}': {}", output_path, e),
        )
    })?;
    header.write(&mut output)?;
    if header.uses_file_table() {
        tables.write(&mut output)?;
    }

    // Write string table
    string_table.write(&mut output)?;

    let records_start = output.stream_position()?;
    let mut writer = BufWriter::new(&mut output);
    let byte_offsets = if !is_compressed_paf(input_path) {
        compress_plain_records(
            input_path,
            &mut writer,
            records_start,
            string_table,
            plan,
            contig_table,
            &tables,
        )?
    } else {
        let mut byte_offsets = Vec::with_capacity(record_count as usize);
        let mut cur_off = records_start;
        let mut input = open_paf_reader(input_path)?;
        let mut buf = Vec::new();
        let mut line_num = 0usize;
        let chunk_size = rayon::current_num_threads().max(1) * 2;
        let mut chunk: Vec<(usize, String)> = Vec::with_capacity(chunk_size);
        let mut eof = false;
        while !eof {
            chunk.clear();
            while chunk.len() < chunk_size {
                buf.clear();
                if input.read_until(b'\n', &mut buf)? == 0 {
                    eof = true;
                    break;
                }
                let cur = line_num;
                line_num += 1;
                while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
                    buf.pop();
                }
                let line = match std::str::from_utf8(&buf) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                if !line.trim().is_empty() && !line.starts_with('#') {
                    chunk.push((cur, line.to_string()));
                }
            }
            let packed: Vec<EncodedRange> = chunk
                .par_iter()
                .map(|(line_num, line)| -> io::Result<EncodedRange> {
                    match parse_paf_readonly(
                        line,
                        string_table,
                        plan.tp_type,
                        plan.max_complexity,
                        plan.metric,
                        contig_table,
                    ) {
                        Ok(records) => {
                            let mut encoded = EncodedRange::default();
                            encoded.extend(&records, plan, &tables)?;
                            Ok(encoded)
                        }
                        Err(e) => {
                            warn!("Skipping malformed line {}: {}", line_num + 1, e);
                            Ok(EncodedRange::default())
                        }
                    }
                })
                .collect::<io::Result<_>>()?;
            for encoded in packed {
                byte_offsets.extend(encoded.offsets.into_iter().map(|offset| cur_off + offset));
                cur_off += encoded.bytes.len() as u64;
                writer.write_all(&encoded.bytes)?;
            }
        }
        byte_offsets
    };
    writer.flush()?;
    drop(writer);

    if byte_offsets.len() as u64 != record_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input changed between compression passes",
        ));
    }

    // Append footer to mark the file as complete
    let footer = TpaFooter::for_header(&header);
    footer.write(&mut output)?;
    output.sync_all()?; // Ensure data is flushed to disk

    // Save index with byte offsets
    let idx_path = format!("{}.idx", output_path);
    let index = index::TpaIndex::new_raw(byte_offsets);
    index.save(&idx_path)?;

    info!(
        "Converted {} records ({} unique sequence names) with strategies {} / {} [per-record mode]",
        record_count,
        string_table.len(),
        plan.first,
        plan.second
    );
    Ok(())
}

/// All-records compression mode: BGZIP all records together (header/string table/footer as plain bytes)
///
/// This enables fast file open since header can be read directly without decompressing
/// a BGZF block. Random access via virtual position index (.tpa.idx file, rebuilt if missing).
/// Virtual positions can also be stored externally and used with standalone seek functions.
fn compress_all_records(
    input_path: &str,
    output_path: &str,
    record_count: u64,
    string_table: &StringTable,
    plan: CompressionPlan,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
    tables: HuffmanTables,
) -> io::Result<()> {
    use noodles::bgzf;

    let mut output_file = File::create(output_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{}': {}", output_path, e),
        )
    })?;

    // Write header as plain bytes
    let mut stored = plan;
    stored.first_layer = CompressionLayer::Nocomp;
    stored.second_layer = CompressionLayer::Nocomp;
    let header = stored.header(record_count, string_table.len() as u64, true)?;
    header.write(&mut output_file)?;

    if header.uses_file_table() {
        tables.write(&mut output_file)?;
    }

    // Write string table as plain bytes
    string_table.write(&mut output_file)?;

    // Record BGZF section start for virtual position adjustment
    let bgzf_section_start = output_file.stream_position()?;

    // Create BGZF writer starting at current position (records section)
    let level = bgzf::io::writer::CompressionLevel::new(plan.all_records_level as u8)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid BGZIP level"))?;
    let mut bgzf_writer = bgzf::io::writer::Builder::default()
        .set_compression_level(level)
        .build_from_writer(output_file);

    // Track virtual positions for index
    let mut virtual_positions = Vec::with_capacity(record_count as usize);

    // Write records through BGZF
    let input = open_paf_reader(input_path)?;
    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        let records = match parse_paf_readonly(
            &line,
            string_table,
            plan.tp_type,
            plan.max_complexity,
            plan.metric,
            contig_table,
        ) {
            Ok(r) => r,
            Err(e) => {
                warn!("Skipping malformed line {}: {}", line_num + 1, e);
                continue;
            }
        };

        // One input line may yield several records (contig-aware fastga); write each with its own vpos.
        for record in &records {
            // Capture virtual position before writing record
            // Adjust to absolute virtual positions (block_offset includes bgzf_section_start)
            let relative_vpos = u64::from(bgzf_writer.virtual_position());
            let block_offset = relative_vpos >> 16;
            let uncompressed_offset = relative_vpos & 0xFFFF;
            let absolute_block_offset = block_offset + bgzf_section_start;
            let absolute_vpos = (absolute_block_offset << 16) | uncompressed_offset;
            virtual_positions.push(absolute_vpos);

            // Write with strategy encoding but no per-record compression layer since BGZF handles compression
            record.write(
                &mut bgzf_writer,
                plan.first,
                plan.second,
                format::CompressionLayer::Nocomp,
                format::CompressionLayer::Nocomp,
                &tables,
            )?;
        }
    }

    // Finalize BGZF (flushes and writes EOF marker)
    let mut output_file = bgzf_writer.finish()?;

    if virtual_positions.len() as u64 != record_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "input changed between compression passes",
        ));
    }

    // Write footer as plain bytes (after BGZF EOF)
    let footer = TpaFooter::for_header(&header);
    footer.write(&mut output_file)?;
    output_file.sync_all()?; // Ensure data is flushed to disk

    // Save index with virtual positions and bgzf_section_start for fast file open
    let idx_path = format!("{}.idx", output_path);
    let index =
        index::TpaIndex::new_virtual_with_section_start(virtual_positions, bgzf_section_start);
    index.save(&idx_path)?;

    // Log compression level (default for noodles is 6)
    info!(
        "Converted {} records ({} unique sequence names) with strategies {} / {} [all-records mode, level={}]",
        record_count,
        string_table.len(),
        plan.first,
        plan.second,
        plan.all_records_level
    );
    Ok(())
}

/// Parallel decode for a seekable .tpa with a per-record (raw-offset) index. Reads records in
/// chunks by byte-range, decodes + formats each chunk's records in parallel into ordered buffers,
/// and writes them in order (byte-identical to the sequential decoder). Returns Ok(false) to fall
/// back to the sequential stream decoder (stdin, missing/incompatible index).
fn decompress_per_record_parallel(
    input_path: &str,
    output_path: &str,
    header: &TpaHeader,
    tables: &HuffmanTables,
    file: &mut File,
) -> io::Result<bool> {
    if input_path == "-" {
        return Ok(false);
    }
    let idx_path = format!("{}.idx", input_path);
    let index = match TpaIndex::load(&idx_path) {
        Ok(ix)
            if ix.index_type() == crate::index::IndexType::RawOffset
                && ix.len() as u64 == header.num_records =>
        {
            ix
        }
        _ => return Ok(false),
    };
    let n = header.num_records as usize;
    let (first_strategy, second_strategy) = header.strategies()?;

    // Validate external offsets before creating output.
    let string_table_start = file.stream_position()?;
    let string_table = StringTable::read(file, header.num_strings())?;
    let records_start = file.stream_position()?;
    let file_size = std::fs::metadata(input_path)?.len();
    if !index.validates(
        crate::index::IndexType::RawOffset,
        header.num_records,
        records_start,
        file_size,
    ) {
        file.seek(io::SeekFrom::Start(string_table_start))?;
        return Ok(false);
    }

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

    let tp_type = header.tracepoint_type;
    let (fl, sl) = (header.first_layer, header.second_layer);
    let st = &string_table;
    let index = &index;
    const MAX_RECORDS: usize = 8_192;
    const TARGET_BYTES: u64 = 8 * 1024 * 1024;
    let mut c = 0usize;
    while c < n {
        let base = index.get_position(c as u64).unwrap();
        let mut low = c + 1;
        let mut high = c.saturating_add(MAX_RECORDS).min(n);
        while low < high {
            let mid = (low + high).div_ceil(2);
            let boundary = if mid < n {
                index.get_position(mid as u64).unwrap()
            } else {
                file_size
            };
            if boundary.saturating_sub(base) <= TARGET_BYTES {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        let end = low;
        let chunk_end = if end < n {
            index.get_position(end as u64).unwrap()
        } else {
            file_size
        };
        file.seek(io::SeekFrom::Start(base))?;
        let mut raw = vec![0u8; (chunk_end - base) as usize];
        file.read_exact(&mut raw)?;
        let raw = &raw;
        let bufs: Vec<Vec<u8>> = (c..end)
            .into_par_iter()
            .map(|j| -> io::Result<Vec<u8>> {
                let rel = (index.get_position(j as u64).unwrap() - base) as usize;
                let mut cur = io::Cursor::new(&raw[rel..]);
                let record = read_record(
                    &mut cur,
                    first_strategy,
                    second_strategy,
                    tp_type,
                    fl,
                    sl,
                    tables,
                )?;
                let mut b = Vec::with_capacity(4096);
                write_paf_line_with_tracepoints(&mut b, &record, st)?;
                Ok(b)
            })
            .collect::<io::Result<Vec<_>>>()?;
        for b in &bufs {
            writer.write_all(b)?;
        }
        c = end;
    }
    writer.flush()?;
    info!(
        "Decompressed {} records [per-record mode, parallel]",
        header.num_records
    );
    Ok(true)
}

fn decompress_per_record(input_path: &str, output_path: &str) -> io::Result<()> {
    let mut file = File::open(input_path)?;
    let (header, tables) = read_header_and_tables(&mut file)?;

    let (first_strategy, second_strategy) = header.strategies()?;
    info!(
        "Reading {} records ({} unique sequence names) [{} / {}] [per-record mode]",
        header.num_records, header.num_strings, first_strategy, second_strategy
    );

    // Fast path: seekable file + per-record index => parallel chunked decode.
    if decompress_per_record_parallel(input_path, output_path, &header, &tables, &mut file)? {
        return Ok(());
    }

    let reader = BufReader::new(file);
    decompress_varint(
        reader,
        output_path,
        &header,
        first_strategy,
        second_strategy,
        &tables,
    )
}

fn decompress_all_records(input_path: &str, output_path: &str) -> io::Result<()> {
    use std::io::Write;

    // Use TpaReader which handles BGZF mode
    let mut reader = TpaReader::new(input_path)?;
    reader.load_string_table()?; // Lazy load string table

    // Extract header info before mutably borrowing reader
    let num_records = reader.header().num_records();
    let num_strings = reader.header().num_strings();
    let (first_strategy, second_strategy) = reader.header().strategies()?;
    info!(
        "Reading {} records ({} unique sequence names) [{} / {}] [all-records mode]",
        num_records, num_strings, first_strategy, second_strategy
    );

    let string_table = reader.string_table_ref().clone();

    // Open output file
    let output: Box<dyn Write> = if output_path == "-" {
        Box::new(std::io::stdout())
    } else {
        Box::new(File::create(output_path)?)
    };
    let mut writer = BufWriter::new(output);

    // Write record
    for record_id in 0..num_records {
        let record = reader.get_compact_record(record_id)?;
        write_paf_line_with_tracepoints(&mut writer, &record, &string_table)?;
    }

    writer.flush()?;
    info!("Decompressed {} records", num_records);
    Ok(())
}

fn parse_paf(
    line: &str,
    string_table: &mut StringTable,
    tp_type: TracepointType,
    max_complexity: u32,
    complexity_metric: ComplexityMetric,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
) -> io::Result<Vec<CompactRecord>> {
    parse_paf_with(
        line,
        tp_type,
        max_complexity,
        complexity_metric,
        contig_table,
        |s, len| Ok(string_table.get_or_insert_id(s, len)),
    )
}

// Read-only parse (names already interned in pass 1). Takes &StringTable, so it runs on many threads.
fn parse_paf_readonly(
    line: &str,
    string_table: &StringTable,
    tp_type: TracepointType,
    max_complexity: u32,
    complexity_metric: ComplexityMetric,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
) -> io::Result<Vec<CompactRecord>> {
    parse_paf_with(
        line,
        tp_type,
        max_complexity,
        complexity_metric,
        contig_table,
        |s, _len| {
            string_table.get_id(s).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("name '{}' not interned in pass 1", s),
                )
            })
        },
    )
}

// Shared parse body. `resolve(name, len) -> id` either interns (pass 1) or looks up (pass 2).
// Called query first, then target. Returns one record per input line, except contig-aware fastga
// (contig_table set), which emits one record per contig/overflow segment.
fn parse_paf_with<F: FnMut(&str, u64) -> io::Result<u64>>(
    line: &str,
    tp_type: TracepointType,
    max_complexity: u32,
    complexity_metric: ComplexityMetric,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
    mut resolve: F,
) -> io::Result<Vec<CompactRecord>> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 12 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("PAF line has {} fields, expected at least 12", fields.len()),
        ));
    }

    // Parse standard PAF fields
    let query_name = fields[0];
    let query_len = parse_usize(fields[1], "query_len")?;
    let query_start = parse_usize(fields[2], "query_start")?;
    let query_end = parse_usize(fields[3], "query_end")?;
    validate_interval(query_start, query_end, query_len, "query")?;
    let strand = match fields[4] {
        "+" => '+',
        "-" => '-',
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Strand must be '+' or '-'",
            ))
        }
    };
    let target_name = fields[5];
    let target_len = parse_usize(fields[6], "target_len")?;
    let target_start = parse_usize(fields[7], "target_start")?;
    let target_end = parse_usize(fields[8], "target_end")?;
    validate_interval(target_start, target_end, target_len, "target")?;
    let residue_matches = parse_usize(fields[9], "residue_matches")?;
    let alignment_block_len = parse_usize(fields[10], "alignment_block_len")?;
    let mapping_quality = parse_u8(fields[11], "mapping_quality")?;

    // Parse tags and extract alignment data (auto-detect CIGAR or tracepoints per row)
    let mut cigar = None;
    let mut tp_str = None;
    let mut tags = Vec::new();

    for field in fields.iter().skip(12) {
        if let Some(cg) = field.strip_prefix("cg:Z:") {
            cigar = Some(cg);
            continue;
        }
        if let Some(tp) = field.strip_prefix("tp:Z:") {
            tp_str = Some(tp);
            continue;
        }
        if let Some(tag) = parse_tag(field) {
            tags.push(tag);
        }
    }

    // Prefer tracepoints if present, otherwise convert from CIGAR
    let tracepoints = if let Some(tp_data) = tp_str {
        parse_tracepoints(tp_data, tp_type)?
    } else if let Some(cigar_str) = cigar {
        validate_cigar(cigar_str, tp_type)?;
        // Convert CIGAR to tracepoints
        match tp_type {
            TracepointType::Standard => {
                let tps = cigar_to_tracepoints(cigar_str, max_complexity, complexity_metric);
                TracepointData::Standard(tps)
            }
            TracepointType::Mixed => {
                let tps = cigar_to_mixed_tracepoints(cigar_str, max_complexity, complexity_metric);
                TracepointData::Mixed(tps)
            }
            TracepointType::Variable => {
                let tps =
                    cigar_to_variable_tracepoints(cigar_str, max_complexity, complexity_metric);
                TracepointData::Variable(tps)
            }
            TracepointType::Fastga => {
                let complement = strand == '-';
                let qcontigs = contig_table.and_then(|t| t.get(query_name));
                let tcontigs = contig_table.and_then(|t| t.get(target_name));
                if qcontigs.is_some() || tcontigs.is_some() {
                    // Contig-aware: emit ONE record per contig/overflow segment (same structure as
                    // `encode --fastga-contigs`), since the segments have distinct coordinate frames
                    // (contig-local grids, RC target on '-') and cannot be flattened into one record.
                    let empty: Vec<(usize, usize)> = Vec::new();
                    let tlen = target_len as usize;
                    let segments = cigar_to_tracepoints_fastga_with_contigs(
                        cigar_str,
                        max_complexity,
                        query_start as usize,
                        query_end as usize,
                        query_len as usize,
                        target_start as usize,
                        target_end as usize,
                        tlen,
                        complement,
                        qcontigs.unwrap_or(&empty),
                        tcontigs.unwrap_or(&empty),
                    );
                    let query_name_id = resolve(query_name, query_len as u64)?;
                    let target_name_id = resolve(target_name, target_len as u64)?;
                    let mut out = Vec::with_capacity(segments.len());
                    for (i, (seg_tps, (sqs, sqe, sts, ste))) in segments.into_iter().enumerate() {
                        // _with_contigs returns RC target coords on '-'; store forward (like encode).
                        let (out_ts, out_te) = if complement {
                            (tlen - ste, tlen - sts)
                        } else {
                            (sts, ste)
                        };
                        let seg_diff: usize = seg_tps.iter().map(|(d, _)| *d).sum();
                        let aln_len = sqe.saturating_sub(sqs);
                        out.push(CompactRecord {
                            query_name_id,
                            query_start: sqs as u64,
                            query_end: sqe as u64,
                            strand,
                            target_name_id,
                            target_start: out_ts as u64,
                            target_end: out_te as u64,
                            residue_matches: aln_len.saturating_sub(seg_diff) as u64,
                            alignment_block_len: aln_len as u64,
                            mapping_quality,
                            tracepoints: TracepointData::Fastga(seg_tps),
                            // Line-level tags belong to the alignment, not each segment: hand them to the
                            // first segment (move, no clone) and leave the rest empty. Avoids a per-segment
                            // Vec alloc + tag copy, and keeps Tag/TagValue Clone-free.
                            tags: if i == 0 {
                                std::mem::take(&mut tags)
                            } else {
                                Vec::new()
                            },
                        });
                    }
                    return Ok(out);
                }
                let segments = cigar_to_tracepoints_fastga(
                    cigar_str,
                    max_complexity,
                    query_start as usize,
                    query_end as usize,
                    query_len as usize,
                    target_start as usize,
                    target_end as usize,
                    target_len as usize,
                    complement,
                );
                // Flatten all segments into single tracepoint list
                let tps: Vec<(usize, usize)> =
                    segments.into_iter().flat_map(|(tps, _)| tps).collect();
                TracepointData::Fastga(tps)
            }
            TracepointType::FastgaNoDiff => {
                let complement = strand == '-';
                let segments = cigar_to_tracepoints_fastga_nodiff(
                    cigar_str,
                    max_complexity,
                    query_start as usize,
                    query_end as usize,
                    query_len as usize,
                    target_start as usize,
                    target_end as usize,
                    target_len as usize,
                    complement,
                );
                let tps: Vec<usize> = segments.into_iter().flat_map(|(tps, _)| tps).collect();
                TracepointData::FastgaNoDiff(tps)
            }
        }
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PAF line has neither tp:Z: (tracepoints) nor cg:Z: (CIGAR) tag",
        ));
    };

    let query_name_id = resolve(query_name, query_len as u64)?;
    let target_name_id = resolve(target_name, target_len as u64)?;

    Ok(vec![CompactRecord {
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
    }])
}

fn parse_tracepoint_pair(part: &str, strict: bool) -> io::Result<(usize, usize)> {
    let mut coords = part.split(',');
    let first = coords
        .next()
        .ok_or_else(|| invalid_tracepoint_format(part))?;
    let second = coords
        .next()
        .ok_or_else(|| invalid_tracepoint_format(part))?;
    if strict && coords.next().is_some() {
        return Err(invalid_tracepoint_format(part));
    }

    Ok((
        parse_usize_value(first, "Invalid first value")?,
        parse_usize_value(second, "Invalid second value")?,
    ))
}

fn parse_usize_value(value: &str, err_msg: &'static str) -> io::Result<usize> {
    value
        .parse::<usize>()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, err_msg))
}

fn invalid_tracepoint_format(part: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("Invalid mixed tracepoint format: '{}'", part),
    )
}
