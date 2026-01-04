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

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, Write};

// Re-export public types
pub use lib_wfa2::affine_wavefront::Distance;
use tracepoints::{
    cigar_to_mixed_tracepoints, cigar_to_tracepoints, cigar_to_tracepoints_fastga,
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
    AlignmentRecordIterator, RecordIterator, TpaReader,
};

// Re-export cigar module types and functions
pub use cigar::{
    calculate_alignment_score, reconstruct_cigar, reconstruct_cigar_with_aligner,
    reconstruct_cigar_with_heuristic, CigarReconstructError, CigarStats,
};

// Re-export aligner type for reconstruct_cigar_with_aligner API
pub use lib_wfa2::affine_wavefront::AffineWavefronts;

// Re-export noodles bgzf for BGZF mode standalone access
pub use noodles::bgzf;

// Re-export utility functions for external tools
pub use utils::{read_varint, varint_size};

use crate::binary::{decompress_varint, write_paf_line_with_tracepoints, StrategyAnalyzer};

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

    // For all-records mode, we skip layer testing since layers are not used
    let test_layers = !config.all_records;
    let mut analyzer = match (&config.first_strategy, &config.second_strategy) {
        (CompressionStrategy::Automatic(level, sample_size), _)
        | (_, CompressionStrategy::Automatic(level, sample_size)) => {
            Some(StrategyAnalyzer::new(*level, *sample_size, test_layers))
        }
        _ => None,
    };

    let parse_record =
        |line: &str, line_num: usize, table: &mut StringTable| -> Option<CompactRecord> {
            match parse_paf(
                line,
                table,
                config.tp_type,
                config.max_complexity,
                config.complexity_metric,
            ) {
                Ok(r) => Some(r),
                Err(e) => {
                    warn!("Skipping malformed line {}: {}", line_num + 1, e);
                    None
                }
            }
        };

    let input = open_paf_reader(input_path)?;
    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        let Some(record) = parse_record(&line, line_num, &mut string_table) else {
            continue;
        };

        // Analyze record if automatic strategy is enabled
        if let Some(an) = analyzer.as_mut() {
            an.analyze_record(&record)?;
        }
        record_count += 1;
    }

    // Verify file is not empty
    if record_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "No records found in PAF file",
        ));
    }

    // Choose strategy (automatic or explicit)
    let (chosen_first, chosen_second, first_layer, second_layer) = match analyzer {
        Some(analyzer) => analyzer.select_best(),
        None => (
            config.first_strategy,
            config.second_strategy,
            config.first_layer,
            config.second_layer,
        ),
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
pub fn format_tracepoints(data: &TracepointData) -> String {
    match data {
        TracepointData::Standard(tps) | TracepointData::Fastga(tps) => tps
            .iter()
            .map(|(a, b)| format!("{},{}", a, b))
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
    let mut writer = BufWriter::new(&mut output);
    let input = open_paf_reader(input_path)?;
    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        let record = match parse_paf(
            &line,
            string_table,
            tp_type,
            max_complexity,
            complexity_metric,
        ) {
            Ok(r) => r,
            Err(e) => {
                warn!("Skipping malformed line {}: {}", line_num + 1, e);
                continue;
            }
        };

        // Capture byte offset before writing record
        byte_offsets.push(writer.stream_position()?);

        // Write with chosen strategy and layer
        record.write(
            &mut writer,
            chosen_first,
            chosen_second,
            first_layer,
            second_layer,
        )?;
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

        let record = match parse_paf(
            &line,
            string_table,
            tp_type,
            max_complexity,
            complexity_metric,
        ) {
            Ok(r) => r,
            Err(e) => {
                warn!("Skipping malformed line {}: {}", line_num + 1, e);
                continue;
            }
        };

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

fn decompress_per_record(input_path: &str, output_path: &str) -> io::Result<()> {
    let mut file = File::open(input_path)?;
    let header = read_header(&mut file)?;
    let reader = BufReader::new(file);

    let (first_strategy, second_strategy) = header.strategies()?;
    info!(
        "Reading {} records ({} unique sequence names) [{} / {}] [per-record mode]",
        header.num_records, header.num_strings, first_strategy, second_strategy
    );

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
) -> io::Result<CompactRecord> {
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

    // Get or create string IDs
    let query_name_id = string_table.get_or_insert_id(query_name, query_len);
    let target_name_id = string_table.get_or_insert_id(target_name, target_len);

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
                let tps =
                    cigar_to_mixed_tracepoints(cigar_str, max_complexity, complexity_metric);
                TracepointData::Mixed(tps)
            }
            TracepointType::Variable => {
                let tps =
                    cigar_to_variable_tracepoints(cigar_str, max_complexity, complexity_metric);
                TracepointData::Variable(tps)
            }
            TracepointType::Fastga => {
                let complement = strand == '-';
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
        }
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PAF line has neither tp:Z: (tracepoints) nor cg:Z: (CIGAR) tag",
        ));
    };

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
