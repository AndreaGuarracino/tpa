mod binary;
mod format;
mod hybrids;
/// Binary PAF format for efficient storage of sequence alignments with tracepoints
///
/// Format: [Header] → [StringTable] → [Records]
/// - Header: Magic "BPAF" + version + metadata
/// - StringTable: Deduplicated sequence names with lengths
/// - Records: Core PAF fields + compressed tracepoints
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
    AlignmentRecord, BinaryPafHeader, CompressionLayer, CompressionStrategy, StringTable, Tag,
    TagValue,
};
pub use lib_tracepoints::{ComplexityMetric, MixedRepresentation, TracepointData, TracepointType};

use crate::format::parse_tag;
use crate::utils::{parse_u8, parse_usize};

pub use binary::{
    build_index,
    read_mixed_tracepoints_at_offset,
    // Standalone functions for ultimate performance
    read_standard_tracepoints_at_offset,
    read_variable_tracepoints_at_offset,
    BpafIndex,
    BpafReader,
    RecordIterator,
    BINARY_MAGIC,
};

// Re-export utility functions for external tools
pub use utils::{read_varint, varint_size};

use crate::binary::{analyze_correlation, analyze_smart_dual_compression, decompress_varint};

use crate::utils::open_paf_reader;

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
        Ok(()) => Ok(&magic == BINARY_MAGIC),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e),
    }
}

pub fn compress_paf_with_cigar(
    input_path: &str,
    output_path: &str,
    strategy: CompressionStrategy,
    layer: format::CompressionLayer,
    tp_type: TracepointType,
    max_complexity: u64,
    complexity_metric: ComplexityMetric,
    distance: Distance,
) -> io::Result<()> {
    compress_paf(
        input_path,
        output_path,
        strategy,
        layer,
        tp_type,
        max_complexity,
        complexity_metric,
        distance,
        true, // use_cigar
    )
}

pub fn compress_paf_with_cigar_dual(
    input_path: &str,
    output_path: &str,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    layer: format::CompressionLayer,
    tp_type: TracepointType,
    max_complexity: u64,
    complexity_metric: ComplexityMetric,
    distance: Distance,
) -> io::Result<()> {
    // Extract zstd level from first strategy
    let zstd_level = match &first_strategy {
        CompressionStrategy::Raw(lvl)
        | CompressionStrategy::ZigzagDelta(lvl)
        | CompressionStrategy::TwoDimDelta(lvl)
        | CompressionStrategy::RunLength(lvl)
        | CompressionStrategy::BitPacked(lvl)
        | CompressionStrategy::DeltaOfDelta(lvl)
        | CompressionStrategy::FrameOfReference(lvl)
        | CompressionStrategy::HybridRLE(lvl)
        | CompressionStrategy::OffsetJoint(lvl)
        | CompressionStrategy::XORDelta(lvl)
        | CompressionStrategy::Dictionary(lvl)
        | CompressionStrategy::Simple8(lvl)
        | CompressionStrategy::StreamVByte(lvl)
        | CompressionStrategy::FastPFOR(lvl)
        | CompressionStrategy::Cascaded(lvl)
        | CompressionStrategy::Simple8bFull(lvl)
        | CompressionStrategy::SelectiveRLE(lvl) => *lvl,
        CompressionStrategy::Dual(_, _, lvl) => *lvl,
        CompressionStrategy::Automatic(lvl) => *lvl,
        CompressionStrategy::AdaptiveCorrelation(lvl) => *lvl,
    };

    // Create Dual strategy
    let dual_strategy = CompressionStrategy::Dual(
        Box::new(first_strategy),
        Box::new(second_strategy),
        zstd_level,
    );

    compress_paf(
        input_path,
        output_path,
        dual_strategy,
        layer,
        tp_type,
        max_complexity,
        complexity_metric,
        distance,
        true, // use_cigar
    )
}

pub fn compress_paf_with_tracepoints(
    input_path: &str,
    output_path: &str,
    strategy: CompressionStrategy,
    layer: format::CompressionLayer,
    tp_type: TracepointType,
    max_complexity: u64,
    complexity_metric: ComplexityMetric,
    distance: Distance,
) -> io::Result<()> {
    compress_paf(
        input_path,
        output_path,
        strategy,
        layer,
        tp_type,
        max_complexity,
        complexity_metric,
        distance,
        false, // use_cigar
    )
}

/// Compress a PAF file with tracepoints using separate strategies for first and second values
///
/// This allows explicit control over which compression strategy is used for each value
/// in tracepoint pairs, enabling testing of all 17×17=289 strategy combinations.
///
/// # Arguments
/// * `input_path` - Path to input PAF file with tracepoints
/// * `output_path` - Path to output BPAF file
/// * `first_strategy` - Strategy for compressing first values in tracepoint pairs
/// * `second_strategy` - Strategy for compressing second values in tracepoint pairs
/// * `tp_type` - Type of tracepoint representation
/// * `max_complexity` - Maximum complexity/spacing parameter
/// * `complexity_metric` - Metric used for complexity calculation
/// * `distance` - Distance parameters for alignment
///
/// # Example
/// ```no_run
/// use lib_bpaf::{compress_paf_with_tracepoints_dual, CompressionStrategy, TracepointType, ComplexityMetric, Distance};
///
/// compress_paf_with_tracepoints_dual(
///     "input.paf",
///     "output.bpaf",
///     CompressionStrategy::Raw(3),
///     CompressionStrategy::ZigzagDelta(3),
///     TracepointType::Standard,
///     32,
///     ComplexityMetric::EditDistance,
///     Distance::Edit,
/// ).unwrap();
/// ```
pub fn compress_paf_with_tracepoints_dual(
    input_path: &str,
    output_path: &str,
    first_strategy: CompressionStrategy,
    second_strategy: CompressionStrategy,
    layer: format::CompressionLayer,
    tp_type: TracepointType,
    max_complexity: u64,
    complexity_metric: ComplexityMetric,
    distance: Distance,
) -> io::Result<()> {
    // Extract zstd level from first strategy (both should have same level in practice)
    let zstd_level = match &first_strategy {
        CompressionStrategy::Raw(lvl)
        | CompressionStrategy::ZigzagDelta(lvl)
        | CompressionStrategy::TwoDimDelta(lvl)
        | CompressionStrategy::RunLength(lvl)
        | CompressionStrategy::BitPacked(lvl)
        | CompressionStrategy::DeltaOfDelta(lvl)
        | CompressionStrategy::FrameOfReference(lvl)
        | CompressionStrategy::HybridRLE(lvl)
        | CompressionStrategy::OffsetJoint(lvl)
        | CompressionStrategy::XORDelta(lvl)
        | CompressionStrategy::Dictionary(lvl)
        | CompressionStrategy::Simple8(lvl)
        | CompressionStrategy::StreamVByte(lvl)
        | CompressionStrategy::FastPFOR(lvl)
        | CompressionStrategy::Cascaded(lvl)
        | CompressionStrategy::Simple8bFull(lvl)
        | CompressionStrategy::SelectiveRLE(lvl) => *lvl,
        CompressionStrategy::Dual(_, _, lvl) => *lvl,
        CompressionStrategy::Automatic(lvl) => *lvl,
        CompressionStrategy::AdaptiveCorrelation(lvl) => *lvl,
    };

    // Create Dual strategy
    let dual_strategy = CompressionStrategy::Dual(
        Box::new(first_strategy),
        Box::new(second_strategy),
        zstd_level,
    );

    compress_paf(
        input_path,
        output_path,
        dual_strategy,
        layer,
        tp_type,
        max_complexity,
        complexity_metric,
        distance,
        false, // use_cigar
    )
}

pub fn decompress_bpaf(input_path: &str, output_path: &str) -> io::Result<()> {
    info!("Decompressing {} to text format...", input_path);

    let input = File::open(input_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to open input file '{}': {}", input_path, e),
        )
    })?;
    let mut reader = BufReader::new(input);

    let header = BinaryPafHeader::read(&mut reader)?;

    if header.version != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported format version: {}", header.version),
        ));
    }

    let strategy = header.strategy()?;
    info!(
        "Reading {} records ({} unique sequence names) [{}]",
        header.num_records, header.num_strings, strategy
    );

    decompress_varint(reader, output_path, &header, strategy)
}

// ============================================================================
// Helpers
// ============================================================================

fn compress_paf(
    input_path: &str,
    output_path: &str,
    strategy: CompressionStrategy,
    user_specified_layer: format::CompressionLayer,
    tp_type: TracepointType,
    max_complexity: u64,
    complexity_metric: ComplexityMetric,
    distance: Distance,
    use_cigar: bool,
) -> io::Result<()> {
    info!(
        "Compressing PAF with {} using {} strategy...",
        if use_cigar { "CIGAR" } else { "TRACEPOINTS" },
        strategy
    );

    const SAMPLE_SIZE: usize = 1000; // Sample size for empirical compression test to decide strategy

    // Pass 1: Build string table + collect sample for analysis
    let mut string_table = StringTable::new();
    let mut sample = Vec::new();
    let mut record_count = 0u64;

    let input = open_paf_reader(input_path)?;
    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        let record = if use_cigar {
            parse_paf_with_cigar(
                &line,
                &mut string_table,
                tp_type,
                max_complexity as usize,
                &complexity_metric,
            )
        } else {
            parse_paf_with_tracepoints(
                &line,
                &mut string_table,
                tp_type,
                max_complexity as usize,
                &complexity_metric,
            )
        };

        let record = match record {
            Ok(r) => r,
            Err(e) => {
                warn!("Skipping malformed line {}: {}", line_num + 1, e);
                continue;
            }
        };

        if sample.len() < SAMPLE_SIZE {
            sample.push(record);
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
    let (chosen_strategy, chosen_layer) = match strategy {
        CompressionStrategy::Automatic(level) => {
            // Run empirical compression test for DUAL strategies (test all 17×17×3 = 867 combinations)
            let (best_first, best_second, best_layer) =
                analyze_smart_dual_compression(&sample, level);
            info!(
                "Automatic: Selected {} → {} with layer {:?}",
                best_first, best_second, best_layer
            );
            // Wrap in Dual strategy
            let dual_strategy =
                CompressionStrategy::Dual(Box::new(best_first), Box::new(best_second), level);
            (dual_strategy, best_layer)
        }
        CompressionStrategy::AdaptiveCorrelation(level) => {
            // Analyze correlation and choose optimal strategy
            let correlation = analyze_correlation(&sample);
            let chosen = if correlation > 0.95 {
                info!(
                    "Adaptive: High correlation ({:.4}) → OffsetJoint",
                    correlation
                );
                CompressionStrategy::OffsetJoint(level)
            } else if correlation > 0.80 {
                info!(
                    "Adaptive: Medium correlation ({:.4}) → 2D-Delta",
                    correlation
                );
                CompressionStrategy::TwoDimDelta(level)
            } else if correlation > 0.50 {
                info!(
                    "Adaptive: Low correlation ({:.4}) → ZigzagDelta",
                    correlation
                );
                CompressionStrategy::ZigzagDelta(level)
            } else {
                info!(
                    "Adaptive: Very low correlation ({:.4}) → FrameOfReference",
                    correlation
                );
                CompressionStrategy::FrameOfReference(level)
            };
            // Use user-specified layer or default
            (chosen, user_specified_layer)
        }
        // Respect explicit user choices - use user-specified layer
        strategy => (strategy, user_specified_layer),
    };

    // Pass 2: Stream write - Header → StringTable → Records
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
        chosen_strategy.clone(),
        chosen_layer,
        tp_type,
        complexity_metric,
        max_complexity,
        distance,
    );
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

        let record = if use_cigar {
            parse_paf_with_cigar(
                &line,
                &mut string_table,
                tp_type,
                max_complexity as usize,
                &complexity_metric,
            )
        } else {
            parse_paf_with_tracepoints(
                &line,
                &mut string_table,
                tp_type,
                max_complexity as usize,
                &complexity_metric,
            )
        };

        let record = match record {
            Ok(r) => r,
            Err(e) => {
                warn!("Skipping malformed line {}: {}", line_num + 1, e);
                continue;
            }
        };

        // Write with chosen strategy and layer
        record.write(&mut writer, chosen_strategy.clone(), chosen_layer)?;
    }
    writer.flush()?;

    info!(
        "Compressed {} records ({} unique sequence names) with {} strategy",
        record_count,
        string_table.len(),
        chosen_strategy
    );
    Ok(())
}

fn parse_paf_with_cigar(
    line: &str,
    string_table: &mut StringTable,
    tp_type: TracepointType,
    max_complexity: usize,
    complexity_metric: &ComplexityMetric,
) -> io::Result<AlignmentRecord> {
    parse_paf(
        line,
        string_table,
        true,
        tp_type,
        max_complexity,
        *complexity_metric,
    )
}

fn parse_paf_with_tracepoints(
    line: &str,
    string_table: &mut StringTable,
    tp_type: TracepointType,
    max_complexity: usize,
    complexity_metric: &ComplexityMetric,
) -> io::Result<AlignmentRecord> {
    parse_paf(
        line,
        string_table,
        false,
        tp_type,
        max_complexity,
        *complexity_metric,
    )
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
        if use_cigar && field.starts_with("cg:Z:") {
            cigar = Some(field.strip_prefix("cg:Z:").unwrap());
        } else if !use_cigar && field.starts_with("tp:Z:") {
            tp_str = Some(field.strip_prefix("tp:Z:").unwrap());
        } else if let Some(tag) = parse_tag(field) {
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

fn parse_tracepoints(tp_str: &str, tp_type: TracepointType) -> io::Result<TracepointData> {
    let segments: Vec<&str> = tp_str.split(';').collect();

    match tp_type {
        TracepointType::Standard | TracepointType::Fastga => {
            // Parse as Standard/FASTGA tracepoints
            let mut tps = Vec::new();
            for part in segments {
                let coords: Vec<&str> = part.split(',').collect();
                if coords.len() != 2 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Invalid mixed tracepoint format: '{}'", part),
                    ));
                }
                let first = coords[0].parse::<usize>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid first value")
                })?;
                let second = coords[1].parse::<usize>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid second value")
                })?;
                tps.push((first, second));
            }

            match tp_type {
                TracepointType::Standard => Ok(TracepointData::Standard(tps)),
                TracepointType::Fastga => Ok(TracepointData::Fastga(tps)),
                _ => unreachable!(),
            }
        }
        TracepointType::Mixed => {
            // Parse as Mixed tracepoints
            let mut items = Vec::new();

            for part in segments {
                if part.contains(',') {
                    // Standard tracepoint
                    let coords: Vec<&str> = part.split(',').collect();
                    if coords.len() != 2 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("Invalid mixed tracepoint format: '{}'", part),
                        ));
                    }
                    let first = coords[0].parse::<usize>().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid first value")
                    })?;
                    let second = coords[1].parse::<usize>().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid second value")
                    })?;
                    items.push(MixedRepresentation::Tracepoint(first, second));
                } else {
                    // CIGAR operation
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
                    let len = len_str.parse::<usize>().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid CIGAR length")
                    })?;
                    items.push(MixedRepresentation::CigarOp(len, op));
                }
            }

            Ok(TracepointData::Mixed(items))
        }
        TracepointType::Variable => {
            // Parse as Variable tracepoints
            let mut tps = Vec::new();
            for part in segments {
                if part.contains(',') {
                    let coords: Vec<&str> = part.split(',').collect();
                    let first = coords[0].parse::<usize>().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid first value")
                    })?;
                    let second = coords[1].parse::<usize>().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid second value")
                    })?;
                    tps.push((first, Some(second)));
                } else {
                    let first = part.parse::<usize>().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid first value")
                    })?;
                    tps.push((first, None));
                }
            }
            Ok(TracepointData::Variable(tps))
        }
    }
}
