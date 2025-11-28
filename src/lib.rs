//! Binary PAF format for efficient storage of sequence alignments with tracepoints
//!
//! Format: [Header] → [StringTable] → [Records] → [Footer]
//! - Header: Magic "BPAF" + version + metadata
//! - StringTable: Deduplicated sequence names with lengths
//! - Records: Core PAF fields + compressed tracepoints
//! - Footer: Crash-safety marker with record/string counts

mod binary;
mod format;
mod hybrids;
mod index;
mod reader;
mod utils;

use log::{info, warn};

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};

// Re-export public types
use lib_tracepoints::{
    cigar_to_mixed_tracepoints, cigar_to_tracepoints, cigar_to_variable_tracepoints,
};
pub use lib_wfa2::affine_wavefront::Distance;

pub use format::{
    AlignmentRecord, BinaryPafHeader, CompressionConfig, CompressionLayer, CompressionStrategy,
    StringTable, Tag, TagValue, BPAF_MAGIC,
};
pub use lib_tracepoints::{ComplexityMetric, MixedRepresentation, TracepointData, TracepointType};

use crate::format::{parse_tag, open_with_footer, BinaryPafFooter};
use crate::utils::{parse_u8, parse_usize};

pub use index::{build_index, BpafIndex};
pub use reader::{
    read_mixed_tracepoints_at_offset, read_standard_tracepoints_at_offset,
    read_standard_tracepoints_at_offset_with_strategies, read_variable_tracepoints_at_offset,
    BpafReader, RecordIterator,
};

// Re-export utility functions for external tools
pub use utils::{read_varint, varint_size};

use crate::binary::{decompress_varint, SmartDualAnalyzer};

use crate::utils::open_paf_reader;

pub(crate) fn ensure_tracepoints(data: &TracepointData) -> io::Result<()> {
    let has = match data {
        TracepointData::Standard(tps) | TracepointData::Fastga(tps) => !tps.is_empty(),
        TracepointData::Variable(tps) => !tps.is_empty(),
        TracepointData::Mixed(items) => !items.is_empty(),
    };
    if !has {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Encountered record without tracepoints",
        ));
    }
    Ok(())
}

// ============================================================================
// PUBLIC API
// ============================================================================

pub fn is_binary_paf(path: &str) -> io::Result<bool> {
    if path == "-" {
        return Ok(false);
    }
    let mut file = File::open(path)?;
    let mut magic = [0u8; 4];
    match file.read_exact(&mut magic) {
        Ok(()) => Ok(&magic == BPAF_MAGIC),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e),
    }
}

/// Compress a PAF file to BPAF format
///
/// # Example
/// ```no_run
/// use lib_bpaf::{compress_paf_to_bpaf, CompressionConfig, CompressionStrategy, CompressionLayer};
///
/// // Simple usage with defaults (Automatic, Zstd, Standard tracepoints)
/// compress_paf_to_bpaf("input.paf", "output.bpaf", CompressionConfig::new())?;
///
/// // With specific strategy
/// compress_paf_to_bpaf(
///     "input.paf",
///     "output.bpaf",
///     CompressionConfig::new()
///         .strategy(CompressionStrategy::ZigzagDelta(3))
///         .layer(CompressionLayer::Zstd),
/// )?;
///
/// // Dual strategy with CIGAR input
/// compress_paf_to_bpaf(
///     "input.paf",
///     "output.bpaf",
///     CompressionConfig::new()
///         .dual_strategy(
///             CompressionStrategy::Raw(3),
///             CompressionStrategy::ZigzagDelta(3),
///         )
///         .from_cigar(),
/// )?;
/// # Ok::<(), std::io::Error>(())
/// ```
pub fn compress_paf_to_bpaf(
    input_path: &str,
    output_path: &str,
    config: CompressionConfig,
) -> io::Result<()> {
    let second = config.effective_second_strategy();
    compress_paf_internal(
        input_path,
        output_path,
        config.first_strategy,
        second,
        config.layer,
        config.tp_type,
        config.max_complexity,
        config.complexity_metric,
        config.distance,
        config.use_cigar,
    )
}

pub fn decompress_bpaf(input_path: &str, output_path: &str) -> io::Result<()> {
    info!("Decompressing {} to text format...", input_path);

    let (input, header, _after_header_pos) = open_with_footer(input_path)?;
    let reader = BufReader::new(input);

    let (first_strategy, second_strategy) = header.strategies()?;
    info!(
        "Reading {} records ({} unique sequence names) [{} / {}]",
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

// ============================================================================
// Helpers
// ============================================================================

fn compress_paf_internal(
    input_path: &str,
    output_path: &str,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    user_specified_layer: format::CompressionLayer,
    tp_type: TracepointType,
    max_complexity: u64,
    complexity_metric: ComplexityMetric,
    distance: Distance,
    use_cigar: bool,
) -> io::Result<()> {
    info!(
        "Compressing PAF with {} using strategies {} / {}...",
        if use_cigar { "CIGAR" } else { "TRACEPOINTS" },
        first_strategy,
        second_strategy
    );

    // Pass 1: Build string table + collect sample for analysis
    let mut string_table = StringTable::new();
    let mut record_count = 0u64;

    let mut analyzer = match (&first_strategy, &second_strategy) {
        (CompressionStrategy::Automatic(level, sample_size), _)
        | (_, CompressionStrategy::Automatic(level, sample_size)) => {
            let limit = if *sample_size == 0 {
                None // 0 = entire file
            } else {
                Some(*sample_size)
            };
            let parallel = *sample_size == 0; // Use parallel only for full-file analysis
            Some(SmartDualAnalyzer::new(*level, limit, parallel))
        }
        _ => None,
    };

    let parse_record =
        |line: &str, line_num: usize, table: &mut StringTable| -> Option<AlignmentRecord> {
            match parse_paf(
                line,
                table,
                use_cigar,
                tp_type,
                max_complexity as usize,
                complexity_metric,
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

        if let Some(an) = analyzer.as_mut() {
            an.ingest(&record)?;
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

    // Choose strategy based on user's preference
    let (chosen_first, chosen_second, first_layer, second_layer) = match analyzer {
        Some(analyzer) => {
            let (first_strat, first_layer, second_strat, second_layer) = analyzer.finalize_pair()?;
            (first_strat, second_strat, first_layer, second_layer)
        }
        None => (
            first_strategy,
            second_strategy,
            user_specified_layer,
            user_specified_layer,
        ),
    };

    // Pass 2: Write Header → StringTable → Records → Footer
    let mut output = File::create(output_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{}': {}", output_path, e),
        )
    })?;

    // Write header with chosen strategy
    let header = BinaryPafHeader::new(
        record_count,
        string_table.len() as u64,
        chosen_first.clone(),
        chosen_second.clone(),
        first_layer,
        second_layer,
        tp_type,
        complexity_metric,
        max_complexity,
        distance,
    )?;
    header.write(&mut output)?;

    // Write string table
    string_table.write(&mut output)?;

    // Write records with chosen strategy
    let mut writer = BufWriter::new(&mut output);
    let input = open_paf_reader(input_path)?;
    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        let Some(record) = parse_record(&line, line_num, &mut string_table) else {
            continue;
        };

        // Write with chosen strategy and layer
        record.write(
            &mut writer,
            chosen_first.clone(),
            chosen_second.clone(),
            first_layer,
            second_layer,
        )?;
    }
    writer.flush()?;
    drop(writer);

    // Append footer to mark the file as complete
    let footer = BinaryPafFooter::new(record_count, string_table.len() as u64);
    footer.write(&mut output)?;
    output.sync_all()?;

    info!(
        "Compressed {} records ({} unique sequence names) with strategies {} / {}",
        record_count,
        string_table.len(),
        chosen_first,
        chosen_second
    );
    Ok(())
}

fn parse_paf(
    line: &str,
    string_table: &mut StringTable,
    use_cigar: bool,
    tp_type: TracepointType,
    max_complexity: usize,
    complexity_metric: ComplexityMetric,
) -> io::Result<AlignmentRecord> {
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

    // Intern strings
    let query_name_id = string_table.intern(query_name, query_len);
    let target_name_id = string_table.intern(target_name, target_len);

    // Parse tags and extract alignment data (CIGAR or tracepoints)
    let mut cigar = None;
    let mut tp_str = None;
    let mut tags = Vec::new();

    for field in fields.iter().skip(12) {
        if use_cigar {
            if let Some(cg) = field.strip_prefix("cg:Z:") {
                cigar = Some(cg);
                continue;
            }
        } else if let Some(tp) = field.strip_prefix("tp:Z:") {
            tp_str = Some(tp);
            continue;
        }
        if let Some(tag) = parse_tag(field) {
            tags.push(tag);
        }
    }

    let tracepoints = if use_cigar {
        // CIGAR mode: convert CIGAR to tracepoints
        let cigar_str = cigar.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "PAF line missing CIGAR string (cg:Z: tag)",
            )
        })?;

        match tp_type {
            TracepointType::Standard => {
                let tps = cigar_to_tracepoints(cigar_str, max_complexity, complexity_metric);
                TracepointData::Standard(tps)
            }
            TracepointType::Mixed => {
                let mixed =
                    cigar_to_mixed_tracepoints(cigar_str, max_complexity, complexity_metric);
                TracepointData::Mixed(mixed)
            }
            TracepointType::Variable => {
                let tps =
                    cigar_to_variable_tracepoints(cigar_str, max_complexity, complexity_metric);
                TracepointData::Variable(tps)
            }
            TracepointType::Fastga => {
                // FASTGA requires sequence information which we don't have here
                // Fall back to Standard tracepoints
                let tps = cigar_to_tracepoints(cigar_str, max_complexity, complexity_metric);
                TracepointData::Fastga(tps)
            }
        }
    } else {
        // Tracepoint mode: parse tracepoints directly
        let tp_data = tp_str.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "PAF line missing tracepoints (tp:Z: tag)",
            )
        })?;

        parse_tracepoints(tp_data, tp_type)?
    };

    ensure_tracepoints(&tracepoints)?;

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

fn parse_tracepoint_pair(part: &str, strict: bool) -> io::Result<(usize, usize)> {
    let mut coords = part.split(',');
    let first = coords.next().ok_or_else(|| invalid_tracepoint_format(part))?;
    let second = coords.next().ok_or_else(|| invalid_tracepoint_format(part))?;
    if strict && coords.next().is_some() {
        return Err(invalid_tracepoint_format(part));
    }

    Ok((
        parse_usize_value(first, "Invalid first value")?,
        parse_usize_value(second, "Invalid second value")?,
    ))
}

fn parse_tracepoints(tp_str: &str, tp_type: TracepointType) -> io::Result<TracepointData> {
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
