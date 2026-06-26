//! TracePoint Alignment (TPA) format for efficient storage of sequence alignments with tracepoints
//!
//! Format: [Header] → [StringTable] → [Records] → [Footer]
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

use crate::binary::read_header;
use crate::format::{parse_tag, CompactRecord, TpaFooter};
use crate::utils::{parse_u8, parse_usize};

pub use index::{build_index_all_records, build_index_per_record, TpaIndex};
pub use reader::{
    read_mixed_tracepoints_at_offset, read_mixed_tracepoints_at_vpos,
    read_standard_tracepoints_at_offset_with_strategies, read_standard_tracepoints_at_vpos,
    read_variable_tracepoints_at_offset, read_variable_tracepoints_at_vpos, AlignmentRecord,
    AlignmentRecordIterator, RecordIterator, StreamingMetadataIterator, TpaReader,
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

use crate::binary::{
    automatic_select, decompress_varint, read_record, write_paf_line_with_tracepoints,
    StrategyAnalyzer,
};

use crate::utils::open_paf_reader;

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

// Pass-1 fast path (no analyzer): intern just the two names, skip the CIGAR->tracepoint
// conversion. Same field order as parse_paf. Returns false on a malformed line.
fn intern_names_only(line: &str, string_table: &mut StringTable) -> bool {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 12 {
        return false;
    }
    let (Ok(query_len), Ok(target_len)) = (
        parse_usize(fields[1], "query_len"),
        parse_usize(fields[6], "target_len"),
    ) else {
        return false;
    };
    string_table.get_or_insert_id(fields[0], query_len);
    string_table.get_or_insert_id(fields[5], target_len);
    true
}

/// Intern query+target names reading ONLY the first 7 PAF fields (via splitn), so the (potentially
/// megabyte-long) CIGAR/tag tail is never scanned. Used by the fast plain path's name pass.
fn intern_names_prefix(line: &str, table: &mut StringTable) -> bool {
    let mut it = line.splitn(8, '\t');
    let q = match it.next() {
        Some(s) => s,
        None => return false,
    };
    let qlen = match it.next() {
        Some(s) => s,
        None => return false,
    };
    if it.next().is_none() || it.next().is_none() || it.next().is_none() {
        return false; // fields 2,3,4
    }
    let t = match it.next() {
        Some(s) => s,
        None => return false,
    };
    let tlen = match it.next() {
        Some(s) => s,
        None => return false,
    };
    let (Ok(ql), Ok(tl)) = (
        parse_usize(qlen, "query_len"),
        parse_usize(tlen, "target_len"),
    ) else {
        return false;
    };
    table.get_or_insert_id(q, ql);
    table.get_or_insert_id(t, tl);
    true
}

/// Parallel pass 1 for a plain (seekable, uncompressed) PAF file. Splits the file into byte ranges,
/// one per rayon worker, and parses each range with a thread-local StringTable + record counter,
/// then merges in file order. Merging earliest-range-first reproduces the exact first-seen id
/// assignment of the sequential scan, so pass 2 output is byte-identical. Each range resyncs from
/// `beg-1` to a line boundary so a line starting exactly at `beg` is owned by exactly one range.
#[allow(clippy::too_many_arguments)]
fn parallel_pass1_plain(
    input_path: &str,
    tp_type: TracepointType,
    max_complexity: u32,
    complexity_metric: ComplexityMetric,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
    multi_record: bool,
) -> io::Result<(u64, StringTable)> {
    let size = std::fs::metadata(input_path)?.len();
    if size == 0 {
        return Ok((0, StringTable::new()));
    }
    let nthreads = (rayon::current_num_threads() as u64).max(1);
    let ranges: Vec<(u64, u64)> = (0..nthreads)
        .map(|p| (size * p / nthreads, size * (p + 1) / nthreads))
        .filter(|(b, e)| b < e)
        .collect();

    let partials: Vec<io::Result<(u64, StringTable)>> = ranges
        .par_iter()
        .map(|&(beg, end)| -> io::Result<(u64, StringTable)> {
            let mut f = File::open(input_path)?;
            let start = if beg == 0 { 0 } else { beg - 1 };
            f.seek(io::SeekFrom::Start(start))?;
            let mut rdr = BufReader::new(f);
            let mut buf: Vec<u8> = Vec::new();
            let mut pos = start;
            if beg != 0 {
                // discard bytes up to and including the next newline (resync to a line start)
                let n = rdr.read_until(b'\n', &mut buf)?;
                pos += n as u64;
                buf.clear();
            }
            let mut local = StringTable::new();
            let mut count = 0u64;
            // own every line whose start byte is < end (read it fully, even past end)
            while pos < end {
                buf.clear();
                let n = rdr.read_until(b'\n', &mut buf)?;
                if n == 0 {
                    break;
                }
                pos += n as u64;
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
                if multi_record {
                    match parse_paf(line, &mut local, tp_type, max_complexity, complexity_metric, contig_table) {
                        Ok(recs) => count += recs.len() as u64,
                        Err(_) => continue,
                    }
                } else if intern_names_only(line, &mut local) {
                    count += 1;
                }
            }
            Ok((count, local))
        })
        .collect();

    let mut total = 0u64;
    let mut global = StringTable::new();
    for r in partials {
        let (c, local) = r?;
        total += c;
        for id in 0..local.len() as u64 {
            if let Some((name, len)) = local.get_name_and_len(id) {
                global.get_or_insert_id(name, len);
            }
        }
    }
    Ok((total, global))
}

/// Convert a PAF file to TPA format
pub fn paf_to_tpa(
    input_path: &str,
    output_path: &str,
    config: CompressionConfig,
) -> io::Result<()> {
    info!(
        "Converting PAF to TPA using strategies {} / {}...",
        config.first_strategy, config.second_strategy
    );

    // Pass 1: Build string table + collect sample for analysis
    let mut string_table = StringTable::new();
    let mut record_count = 0u64;

    // Check if either strategy uses Automatic (lookup) mode
    let uses_automatic = matches!(
        (&config.first_strategy, &config.second_strategy),
        (CompressionStrategy::Automatic(_), _) | (_, CompressionStrategy::Automatic(_))
    );

    // For all-records mode, we skip layer testing since layers are not used
    let test_layers = !config.all_records;
    let mut analyzer = match (&config.first_strategy, &config.second_strategy) {
        (CompressionStrategy::Benchmark(level, sample_size, forced_layer), _)
        | (_, CompressionStrategy::Benchmark(level, sample_size, forced_layer)) => {
            Some(StrategyAnalyzer::new(
                *level,
                *sample_size,
                test_layers,
                *forced_layer,
            ))
        }
        _ => None,
    };

    let contig_table = config.contig_table.as_ref();
    // Contig-aware fastga emits multiple records per input line, so pass 1 can't use the name-only
    // fast path (it must convert to count the segments for the header record_count).
    let multi_record = contig_table.is_some() && matches!(config.tp_type, TracepointType::Fastga);

    // Fast path: known strategy (no benchmark analyzer) + plain seekable input + per-record mode.
    // Two byte-range-parallel passes: pass 1 interns names only (cheap, no tracepoint compute),
    // pass 2 parses+encodes once into per-range buffers (tracepoints computed exactly once).
    let plain_seekable =
        input_path != "-" && !input_path.ends_with(".gz") && !input_path.ends_with(".bgz");
    if analyzer.is_none() && plain_seekable && !config.all_records {
        let (cf, cs, fl, sl) = if uses_automatic {
            let level = config.first_strategy.zstd_level();
            let r = automatic_select(config.tp_type, config.complexity_metric, level);
            info!(
                "Automatic: {} [{}] → {} [{}]",
                r.0,
                r.2.as_str(),
                r.1,
                r.3.as_str()
            );
            r
        } else {
            (
                config.first_strategy,
                config.second_strategy,
                config.first_layer,
                config.second_layer,
            )
        };
        return compress_fast_plain(
            input_path,
            output_path,
            cf,
            cs,
            fl,
            sl,
            config.tp_type,
            config.complexity_metric,
            config.max_complexity,
            config.distance,
            contig_table,
        );
    }

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
    let plain_seekable =
        input_path != "-" && !input_path.ends_with(".gz") && !input_path.ends_with(".bgz");
    if analyzer.is_none() && plain_seekable {
        let (rc, st) = parallel_pass1_plain(
            input_path,
            config.tp_type,
            config.max_complexity,
            config.complexity_metric,
            contig_table,
            multi_record,
        )?;
        record_count = rc;
        string_table = st;
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

            if let Some(an) = analyzer.as_mut() {
                // The analyzer needs the full record (its tracepoints), so do the full parse.
                let Some(records) = parse_record(line, cur, &mut string_table) else {
                    continue;
                };
                for record in &records {
                    an.analyze_record(record)?;
                }
                record_count += records.len() as u64;
            } else if multi_record {
                // Contig-aware fastga: full parse to count the per-segment records (also interns names).
                let Some(records) = parse_record(line, cur, &mut string_table) else {
                    continue;
                };
                record_count += records.len() as u64;
            } else {
                // No analyzer: pass 1 needs only names + count, so skip the conversion (done in pass 2).
                if !intern_names_only(line, &mut string_table) {
                    warn!("Skipping malformed line {}", cur + 1);
                    continue;
                }
                record_count += 1;
            }
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

    // Pass 2: Write Header → StringTable → Records → Footer
    if config.all_records {
        // All-records mode: header/string table plain, records BGZIP-compressed all together
        compress_all_records(
            input_path,
            output_path,
            record_count,
            &mut string_table,
            chosen_first,
            chosen_second,
            config.tp_type,
            config.complexity_metric,
            config.max_complexity,
            config.distance,
            config.all_records_level,
            config.contig_table.as_ref(),
        )
    } else {
        // Per-record compression with layers
        compress_per_record(
            input_path,
            output_path,
            record_count,
            &mut string_table,
            chosen_first,
            chosen_second,
            first_layer,
            second_layer,
            config.tp_type,
            config.complexity_metric,
            config.max_complexity,
            config.distance,
            config.contig_table.as_ref(),
        )
    }
}

pub fn tpa_to_paf(input_path: &str, output_path: &str) -> io::Result<()> {
    info!("Converting {} to PAF format...", input_path);

    let mut file = File::open(input_path)?;
    let header = read_header(&mut file)?;
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

/// Per-record compression mode: each record uses strategy encoding + optional compression layers
///
/// Format: [Header] [StringTable] [Record1] [Record2] ... [RecordN] [Footer]
///
/// Each record is independently compressed with configurable layers (Zstd/Bgzip/Nocomp).
/// Random access via byte offset index (.tpa.idx file, rebuilt if missing).
/// Offsets can also be stored externally and used with standalone seek functions.
/// Two byte-range-parallel passes for plain input with a known strategy.
/// Pass 1 interns names only (no tracepoint compute). Pass 2 parses + encodes each range ONCE into
/// a local byte buffer (+ relative offsets). The buffers are concatenated in file order, so the
/// string-table ids and record order match the sequential path exactly (byte-identical output).
/// Tracepoints are computed exactly once; there is no serial chunked read and no per-record lseek.
#[allow(clippy::too_many_arguments)]
fn compress_fast_plain(
    input_path: &str,
    output_path: &str,
    chosen_first: CompressionStrategy,
    chosen_second: CompressionStrategy,
    first_layer: format::CompressionLayer,
    second_layer: format::CompressionLayer,
    tp_type: TracepointType,
    complexity_metric: ComplexityMetric,
    max_complexity: u32,
    distance: Distance,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
) -> io::Result<()> {
    let size = std::fs::metadata(input_path)?.len();
    if size == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No records found in PAF file",
        ));
    }
    let nthreads = (rayon::current_num_threads() as u64).max(1);
    let ranges: Vec<(u64, u64)> = (0..nthreads)
        .map(|p| (size * p / nthreads, size * (p + 1) / nthreads))
        .filter(|(b, e)| b < e)
        .collect();

    // Visit every line whose start byte is in [beg, end); resync from beg-1 so a line starting
    // exactly at beg is owned by exactly one range. `f` is called with each line.
    fn for_each_line_in_range<F: FnMut(&str)>(
        input_path: &str,
        beg: u64,
        end: u64,
        mut f: F,
    ) -> io::Result<()> {
        let mut file = File::open(input_path)?;
        let start = if beg == 0 { 0 } else { beg - 1 };
        file.seek(io::SeekFrom::Start(start))?;
        let mut rdr = BufReader::new(file);
        let mut buf: Vec<u8> = Vec::new();
        let mut pos = start;
        if beg != 0 {
            let n = rdr.read_until(b'\n', &mut buf)?;
            pos += n as u64;
            buf.clear();
        }
        while pos < end {
            buf.clear();
            let n = rdr.read_until(b'\n', &mut buf)?;
            if n == 0 {
                break;
            }
            pos += n as u64;
            while matches!(buf.last(), Some(b'\n') | Some(b'\r')) {
                buf.pop();
            }
            if let Ok(line) = std::str::from_utf8(&buf) {
                let t = line.trim();
                if !t.is_empty() && !t.starts_with('#') {
                    f(line);
                }
            }
        }
        Ok(())
    }

    // Pass 1: intern names only (cheap; no tracepoint computation).
    let name_tables: Vec<StringTable> = ranges
        .par_iter()
        .map(|&(beg, end)| -> io::Result<StringTable> {
            let mut local = StringTable::new();
            for_each_line_in_range(input_path, beg, end, |line| {
                intern_names_prefix(line, &mut local);
            })?;
            Ok(local)
        })
        .collect::<io::Result<Vec<_>>>()?;
    let mut global = StringTable::new();
    for local in &name_tables {
        for id in 0..local.len() as u64 {
            if let Some((name, len)) = local.get_name_and_len(id) {
                global.get_or_insert_id(name, len);
            }
        }
    }
    let global_ref = &global;

    // Pass 2: parse + encode each range ONCE into a local buffer with relative offsets.
    let range_out: Vec<(Vec<u8>, Vec<u64>)> = ranges
        .par_iter()
        .map(|&(beg, end)| -> io::Result<(Vec<u8>, Vec<u64>)> {
            // Pre-size the output buffer (~1/3 of the input range; TPA is smaller than the PAF) to
            // avoid repeated Vec doubling+copy as it grows to tens of MB.
            let mut buf: Vec<u8> = Vec::with_capacity(((end - beg) / 3) as usize);
            let mut offs: Vec<u64> = Vec::new();
            for_each_line_in_range(input_path, beg, end, |line| {
                match parse_paf_readonly(
                    line,
                    global_ref,
                    tp_type,
                    max_complexity,
                    complexity_metric,
                    contig_table,
                ) {
                    Ok(records) => {
                        for r in &records {
                            offs.push(buf.len() as u64);
                            r.write(&mut buf, chosen_first, chosen_second, first_layer, second_layer)
                                .expect("writing a record to an in-memory buffer is infallible");
                        }
                    }
                    Err(e) => warn!("Skipping malformed line: {}", e),
                }
            })?;
            Ok((buf, offs))
        })
        .collect::<io::Result<Vec<_>>>()?;

    let record_count: u64 = range_out.iter().map(|(_, o)| o.len() as u64).sum();
    if record_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No records found in PAF file",
        ));
    }

    // Assemble: header -> string table -> concatenated range buffers; index from offsets.
    let mut output = File::create(output_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{}': {}", output_path, e),
        )
    })?;
    let header = TpaHeader::new(
        record_count,
        global.len() as u64,
        chosen_first,
        chosen_second,
        first_layer,
        second_layer,
        tp_type,
        complexity_metric,
        max_complexity,
        distance,
        false,
    )?;
    header.write(&mut output)?;
    global.write(&mut output)?;
    let mut base = output.stream_position()?;
    let mut byte_offsets: Vec<u64> = Vec::with_capacity(record_count as usize);
    let mut writer = BufWriter::new(&mut output);
    for (buf, offs) in &range_out {
        for &o in offs {
            byte_offsets.push(base + o);
        }
        writer.write_all(buf)?;
        base += buf.len() as u64;
    }
    writer.flush()?;
    drop(writer);
    let footer = TpaFooter::new(record_count, global.len() as u64);
    footer.write(&mut output)?;
    output.sync_all()?;
    let idx_path = format!("{}.idx", output_path);
    index::TpaIndex::new_raw(byte_offsets).save(&idx_path)?;
    info!(
        "Converted {} records ({} unique sequence names) with strategies {} / {} [per-record mode, fast plain]",
        record_count,
        global.len(),
        chosen_first,
        chosen_second
    );
    Ok(())
}

fn compress_per_record(
    input_path: &str,
    output_path: &str,
    record_count: u64,
    string_table: &mut StringTable,
    chosen_first: CompressionStrategy,
    chosen_second: CompressionStrategy,
    first_layer: format::CompressionLayer,
    second_layer: format::CompressionLayer,
    tp_type: TracepointType,
    complexity_metric: ComplexityMetric,
    max_complexity: u32,
    distance: Distance,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
) -> io::Result<()> {
    let mut output = File::create(output_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{}': {}", output_path, e),
        )
    })?;

    // Write header with chosen strategy
    let header = TpaHeader::new(
        record_count,
        string_table.len() as u64,
        chosen_first,
        chosen_second,
        first_layer,
        second_layer,
        tp_type,
        complexity_metric,
        max_complexity,
        distance,
        false, // per-record mode
    )?;
    header.write(&mut output)?;

    // Write string table
    string_table.write(&mut output)?;

    // Track byte offsets for index
    let mut byte_offsets = Vec::with_capacity(record_count as usize);

    // Write records with chosen strategy
    let mut cur_off = output.stream_position()?; // base offset (after header + string table)
    let mut writer = BufWriter::new(&mut output);
    let mut input = open_paf_reader(input_path)?;
    // read_until into a reused buffer instead of lines() (a String alloc per line); records are
    // whole-chromosome-sized, so reuse avoids a big per-record allocation.
    let mut buf: Vec<u8> = Vec::new();
    let mut line_num = 0usize;
    // Pass 1 filled the string table, so pass 2 only reads it. Per chunk: read raw lines serially,
    // parse + pack each in PARALLEL (parse_paf_readonly takes &table), then write the buffers
    // serially in order so the byte-offset index stays exact.
    const CHUNK: usize = 1024;
    let table: &StringTable = string_table;
    let mut chunk: Vec<(usize, String)> = Vec::with_capacity(CHUNK);
    let mut eof = false;
    while !eof {
        chunk.clear();
        while chunk.len() < CHUNK {
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
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            chunk.push((cur, line.to_string()));
        }
        if chunk.is_empty() {
            continue;
        }
        // Each line yields one or more records (contig-aware fastga splits into several). Pack each
        // into its own buffer; the serial write below flattens them in order so the index stays exact.
        let packed: Vec<Vec<Vec<u8>>> = chunk
            .par_iter()
            .map(|(cur, line)| {
                match parse_paf_readonly(
                    line,
                    table,
                    tp_type,
                    max_complexity,
                    complexity_metric,
                    contig_table,
                ) {
                    Ok(records) => records
                        .iter()
                        .map(|r| {
                            let mut b = Vec::new();
                            r.write(
                                &mut b,
                                chosen_first,
                                chosen_second,
                                first_layer,
                                second_layer,
                            )
                            .expect("writing a record to an in-memory buffer is infallible");
                            b
                        })
                        .collect(),
                    Err(e) => {
                        warn!("Skipping malformed line {}: {}", cur + 1, e);
                        Vec::new()
                    }
                }
            })
            .collect();
        for b in packed.into_iter().flatten() {
            byte_offsets.push(cur_off);
            cur_off += b.len() as u64;
            writer.write_all(&b)?;
        }
    }
    writer.flush()?;
    drop(writer);

    // Append footer to mark the file as complete
    let footer = TpaFooter::new(record_count, string_table.len() as u64);
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
        chosen_first,
        chosen_second
    );
    Ok(())
}

/// All-records compression mode: BGZIP all records together (header/string table/footer as plain bytes)
///
/// Format: [Header (plain)] [StringTable (plain)] [BGZF: Records...] [BGZF EOF] [Footer (plain)]
///
/// This enables fast file open since header can be read directly without decompressing
/// a BGZF block. Random access via virtual position index (.tpa.idx file, rebuilt if missing).
/// Virtual positions can also be stored externally and used with standalone seek functions.
fn compress_all_records(
    input_path: &str,
    output_path: &str,
    record_count: u64,
    string_table: &mut StringTable,
    chosen_first: CompressionStrategy,
    chosen_second: CompressionStrategy,
    tp_type: TracepointType,
    complexity_metric: ComplexityMetric,
    max_complexity: u32,
    distance: Distance,
    compression_level: u32,
    contig_table: Option<&HashMap<String, Vec<(usize, usize)>>>,
) -> io::Result<()> {
    use noodles::bgzf;

    let mut output_file = File::create(output_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{}': {}", output_path, e),
        )
    })?;

    // Write header as plain bytes
    let header = TpaHeader::new(
        record_count,
        string_table.len() as u64,
        chosen_first,
        chosen_second,
        format::CompressionLayer::Nocomp, // Per-record layer ignored in this mode
        format::CompressionLayer::Nocomp, // Per-record layer ignored in this mode
        tp_type,
        complexity_metric,
        max_complexity,
        distance,
        true, // all-records mode
    )?;
    header.write(&mut output_file)?;

    // Write string table as plain bytes
    string_table.write(&mut output_file)?;

    // Record BGZF section start for virtual position adjustment
    let bgzf_section_start = output_file.stream_position()?;

    // Create BGZF writer starting at current position (records section)
    let level =
        bgzf::io::writer::CompressionLevel::new(compression_level as u8).unwrap_or_default();
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

        let records = match parse_paf(
            &line,
            string_table,
            tp_type,
            max_complexity,
            complexity_metric,
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
                chosen_first,
                chosen_second,
                format::CompressionLayer::Nocomp,
                format::CompressionLayer::Nocomp,
            )?;
        }
    }

    // Finalize BGZF (flushes and writes EOF marker)
    let mut output_file = bgzf_writer.finish()?;

    // Write footer as plain bytes (after BGZF EOF)
    let footer = TpaFooter::new(record_count, string_table.len() as u64);
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
        chosen_first,
        chosen_second,
        compression_level
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

    // Read the string table (right after the header), then seek per chunk via the index.
    let mut f = File::open(input_path)?;
    read_header(&mut f)?;
    let string_table = StringTable::read(&mut f, header.num_strings())?;
    let file_size = std::fs::metadata(input_path)?.len();

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
    // CHUNK bounds peak memory to ~CHUNK output lines; records decode independently (read_record
    // self-delimits, so each record's slice may run past its end into the next record/footer).
    const CHUNK: usize = 65536;
    let mut c = 0usize;
    while c < n {
        let end = (c + CHUNK).min(n);
        let base = index.get_position(c as u64).unwrap();
        let chunk_end = if end < n {
            index.get_position(end as u64).unwrap()
        } else {
            file_size
        };
        f.seek(io::SeekFrom::Start(base))?;
        let mut raw = vec![0u8; (chunk_end - base) as usize];
        f.read_exact(&mut raw)?;
        let raw = &raw;
        let bufs: Vec<Vec<u8>> = (c..end)
            .into_par_iter()
            .map(|j| -> io::Result<Vec<u8>> {
                let rel = (index.get_position(j as u64).unwrap() - base) as usize;
                let mut cur = io::Cursor::new(&raw[rel..]);
                let record =
                    read_record(&mut cur, first_strategy, second_strategy, tp_type, fl, sl)?;
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
    info!("Decompressed {} records [per-record mode, parallel]", header.num_records);
    Ok(true)
}

fn decompress_per_record(input_path: &str, output_path: &str) -> io::Result<()> {
    let mut file = File::open(input_path)?;
    let header = read_header(&mut file)?;

    let (first_strategy, second_strategy) = header.strategies()?;
    info!(
        "Reading {} records ({} unique sequence names) [{} / {}] [per-record mode]",
        header.num_records, header.num_strings, first_strategy, second_strategy
    );

    // Fast path: seekable file + per-record index => parallel chunked decode.
    if decompress_per_record_parallel(input_path, output_path, &header)? {
        return Ok(());
    }

    let reader = BufReader::new(file);
    decompress_varint(
        reader,
        output_path,
        &header,
        first_strategy,
        second_strategy,
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
    let strand = fields[4]
        .chars()
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Empty strand field"))?;
    let target_name = fields[5];
    let target_len = parse_usize(fields[6], "target_len")?;
    let target_start = parse_usize(fields[7], "target_start")?;
    let target_end = parse_usize(fields[8], "target_end")?;
    let residue_matches = parse_usize(fields[9], "residue_matches")?;
    let alignment_block_len = parse_usize(fields[10], "alignment_block_len")?;
    let mapping_quality = parse_u8(fields[11], "mapping_quality")?;

    // Resolve string IDs (insert in pass 1, read-only lookup in parallel pass 2)
    let query_name_id = resolve(query_name, query_len as u64)?;
    let target_name_id = resolve(target_name, target_len as u64)?;

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
