//! TracePoint Alignment (TPA) format for efficient storage of sequence alignments with tracepoints
//!
//! Format: [Header] → [StringTable] → [Records] → [Footer]
//! - Header: Magic "TPA\0" + version + metadata
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
pub use lib_wfa2::affine_wavefront::Distance;
use tracepoints::{
    cigar_to_mixed_tracepoints, cigar_to_tracepoints, cigar_to_tracepoints_fastga,
    cigar_to_variable_tracepoints,
};

pub use format::{
    detect_bgzf, AlignmentRecord, CompressionConfig, CompressionLayer, CompressionStrategy,
    StringTable, Tag, TagValue, TpaHeader, TPA_MAGIC,
};
pub use tracepoints::{ComplexityMetric, MixedRepresentation, TracepointData, TracepointType};

use crate::binary::read_header;
use crate::format::{parse_tag, TpaFooter};
use crate::utils::{parse_u8, parse_usize};

pub use index::{build_index, TpaIndex};
pub use reader::{
    read_mixed_tracepoints_at_offset, read_mixed_tracepoints_at_vpos,
    read_standard_tracepoints_at_offset_with_strategies, read_standard_tracepoints_at_vpos,
    read_variable_tracepoints_at_offset, read_variable_tracepoints_at_vpos, RecordIterator,
    TpaReader,
};

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
    fn check_magic<R: Read>(reader: &mut R) -> io::Result<bool> {
        let mut magic = [0u8; 4];
        match reader.read_exact(&mut magic) {
            Ok(()) => Ok(&magic == TPA_MAGIC),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
            Err(e) => Err(e),
        }
    }

    if path == "-" {
        return Ok(false);
    }

    if detect_bgzf(path)? {
        let file = File::open(path)?;
        check_magic(&mut noodles::bgzf::io::Reader::new(BufReader::new(file)))
    } else {
        check_magic(&mut File::open(path)?)
    }
}

/// Convert a PAF file to TPA format
pub fn paf_to_tpa(
    input_path: &str,
    output_path: &str,
    config: CompressionConfig,
) -> io::Result<()> {
    info!(
        "Converting PAF (with {}) to TPA using strategies {} / {}...",
        if config.from_cigar {
            "CIGAR"
        } else {
            "TRACEPOINTS"
        },
        config.first_strategy,
        config.second_strategy
    );

    // Pass 1: Build string table + collect sample for analysis
    let mut string_table = StringTable::new();
    let mut record_count = 0u64;

    let mut analyzer = match (&config.first_strategy, &config.second_strategy) {
        (CompressionStrategy::Automatic(level, sample_size), _)
        | (_, CompressionStrategy::Automatic(level, sample_size)) => Some(
            StrategyAnalyzer::new(*level, *sample_size, !config.bgzip_all_records),
        ),
        _ => None,
    };

    let parse_record =
        |line: &str, line_num: usize, table: &mut StringTable| -> Option<AlignmentRecord> {
            match parse_paf(
                line,
                table,
                config.from_cigar,
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
        Some(analyzer) => analyzer.select_best()?,
        None => (
            config.first_strategy,
            config.second_strategy,
            config.first_layer,
            config.second_layer,
        ),
    };

    // Pass 2: Write Header → StringTable → Records → Footer
    if config.bgzip_all_records {
        // Compress all records with BGZIP, without layers
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
            config.bgzip_level,
            config.from_cigar,
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
            config.from_cigar,
        )
    }
}

pub fn tpa_to_paf(input_path: &str, output_path: &str) -> io::Result<()> {
    info!("Converting {} to PAF format...", input_path);

    if detect_bgzf(input_path)? {
        decompress_all_records(input_path, output_path)
    } else {
        decompress_per_record(input_path, output_path)
    }
}

fn decompress_per_record(input_path: &str, output_path: &str) -> io::Result<()> {
    let mut file = File::open(input_path)?;
    let header = read_header(&mut file)?;
    let reader = BufReader::new(file);

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
        "Reading {} records ({} unique sequence names) [{} / {}] [BGZIP mode]",
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
        let record = reader.get_alignment_record(record_id)?;
        write_paf_line_with_tracepoints(&mut writer, &record, &string_table)?;
    }

    writer.flush()?;
    info!("Converted {} records", num_records);
    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

pub(crate) fn check_not_empty_tps(data: &TracepointData) -> io::Result<()> {
    if data.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Encountered record without tracepoints",
        ));
    }
    Ok(())
}

/// Per-record compression mode: each record compressed with layers
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
    from_cigar: bool,
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
        chosen_first.clone(),
        chosen_second.clone(),
        first_layer,
        second_layer,
        tp_type,
        complexity_metric,
        max_complexity,
        distance,
        false, // bgzip_all_records = false
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

        let record = match parse_paf(
            &line,
            string_table,
            from_cigar,
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
    let footer = TpaFooter::new(record_count, string_table.len() as u64);
    footer.write(&mut output)?;
    output.sync_all()?;

    info!(
        "Converted {} records ({} unique sequence names) with strategies {} / {}",
        record_count,
        string_table.len(),
        chosen_first,
        chosen_second
    );
    Ok(())
}

/// All-records compression mode: wrap entire file in BGZF
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
    bgzip_level: u32,
    from_cigar: bool,
) -> io::Result<()> {
    use noodles::bgzf;

    let output_file = File::create(output_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{}': {}", output_path, e),
        )
    })?;

    // Create BGZF writer with specified compression level
    let mut bgzf_writer = bgzf::io::Writer::new(output_file);

    // Track virtual positions for index
    let mut virtual_positions = Vec::with_capacity(record_count as usize);

    // Write header (bgzip_all_records = true, layers set to Nocomp since BGZF handles compression)
    let header = TpaHeader::new(
        record_count,
        string_table.len() as u64,
        chosen_first.clone(),
        chosen_second.clone(),
        format::CompressionLayer::Nocomp, // Per-record layer ignored in this mode
        format::CompressionLayer::Nocomp,
        tp_type,
        complexity_metric,
        max_complexity,
        distance,
        true, // bgzip_all_records = true
    )?;
    header.write(&mut bgzf_writer)?;

    // Write string table
    string_table.write(&mut bgzf_writer)?;

    // Write records WITHOUT per-record compression
    let input = open_paf_reader(input_path)?;
    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        let record = match parse_paf(
            &line,
            string_table,
            from_cigar,
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

        // Capture virtual position BEFORE writing record
        virtual_positions.push(u64::from(bgzf_writer.virtual_position()));

        // Write with strategy encoding but NO per-record compression layer
        // Use Nocomp layer since BGZF handles compression
        record.write(
            &mut bgzf_writer,
            chosen_first.clone(),
            chosen_second.clone(),
            format::CompressionLayer::Nocomp,
            format::CompressionLayer::Nocomp,
        )?;
    }

    // Write footer
    let footer = TpaFooter::new(record_count, string_table.len() as u64);
    footer.write(&mut bgzf_writer)?;

    // Finalize BGZF (flushes and writes EOF marker)
    bgzf_writer.finish()?;

    // Save index with virtual positions
    let idx_path = format!("{}.idx", output_path);
    let index = index::TpaIndex::new_virtual(virtual_positions);
    index.save(&idx_path)?;

    // Log compression level (default for noodles is 6)
    info!(
        "Converted {} records ({} unique sequence names) with strategies {} / {} [BGZIP whole-file mode, level={}]",
        record_count,
        string_table.len(),
        chosen_first,
        chosen_second,
        bgzip_level
    );
    Ok(())
}

fn parse_paf(
    line: &str,
    string_table: &mut StringTable,
    from_cigar: bool,
    tp_type: TracepointType,
    max_complexity: u32,
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
        if from_cigar {
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

    let tracepoints = if from_cigar {
        // Convert CIGAR to tracepoints
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
        // Tracepoint mode: parse tracepoints directly
        let tp_data = tp_str.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "PAF line missing tracepoints (tp:Z: tag)",
            )
        })?;

        parse_tracepoints(tp_data, tp_type)?
    };

    check_not_empty_tps(&tracepoints)?;

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
