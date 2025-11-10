//! Binary I/O operations for BPAF format

use lib_tracepoints::{
    cigar_to_mixed_tracepoints, cigar_to_tracepoints,
    cigar_to_variable_tracepoints, ComplexityMetric, TracepointType,
};
use lib_wfa2::affine_wavefront::Distance;
use log::{debug, error, info};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::format::*;
use crate::utils::*;

// ============================================================================
// CONSTANTS
// ============================================================================

pub const BINARY_MAGIC: &[u8; 4] = b"BPAF";

// ============================================================================
// COMPLEXITY METRIC SERIALIZATION
// ============================================================================

pub fn complexity_metric_to_u8(metric: &ComplexityMetric) -> u8 {
    match metric {
        ComplexityMetric::EditDistance => 0,
        ComplexityMetric::DiagonalDistance => 1,
    }
}

pub fn complexity_metric_from_u8(byte: u8) -> io::Result<ComplexityMetric> {
    match byte {
        0 => Ok(ComplexityMetric::EditDistance),
        1 => Ok(ComplexityMetric::DiagonalDistance),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid complexity metric code: {}", byte),
        )),
    }
}

// ============================================================================
// DETECTION
// ============================================================================

/// Detect if file is binary PAF
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

/// Heuristic to decide if delta encoding is beneficial
fn should_use_delta(values: &[u64]) -> bool {
    if values.len() < 2 {
        return false;
    }

    // Check monotonicity and delta statistics
    let mut monotonic = true;
    let mut total_delta: u64 = 0;
    let mut max_delta: i64 = 0;
    let mut negative_count = 0;

    for i in 1..values.len() {
        let delta = values[i] as i64 - values[i - 1] as i64;

        if delta < 0 {
            monotonic = false;
            negative_count += 1;
        }

        total_delta += delta.abs() as u64;
        max_delta = max_delta.max(delta.abs());
    }

    // Average delta magnitude
    let avg_delta = total_delta / (values.len() - 1) as u64;

    // Average raw value
    let avg_value = values.iter().sum::<u64>() / values.len() as u64;

    // Heuristic: Use delta if:
    // 1. Values are mostly monotonic (< 10% negative deltas)
    // 2. Average delta is significantly smaller than average value (< 50%)
    // 3. Max delta is not too large (< 10x average)

    let negative_ratio = negative_count as f64 / (values.len() - 1) as f64;
    let delta_ratio = avg_delta as f64 / avg_value.max(1) as f64;
    let max_delta_ratio = max_delta as f64 / avg_delta.max(1) as f64;

    let use_delta = monotonic
        || (negative_ratio < 0.1 && delta_ratio < 0.5 && max_delta_ratio < 10.0);

    debug!(
        "Delta heuristic: mono={}, neg_ratio={:.2}, delta_ratio={:.2}, max_ratio={:.2} -> {}",
        monotonic, negative_ratio, delta_ratio, max_delta_ratio, use_delta
    );

    use_delta
}

fn analyze_smart_light_compression(records: &[AlignmentRecord]) -> (bool, bool) {
    const SAMPLE_SIZE: usize = 1000;

    let mut all_first_vals = Vec::new();
    let mut all_second_vals = Vec::new();

    // Sample first N records (or all if fewer)
    let sample_count = records.len().min(SAMPLE_SIZE);
    for record in records.iter().take(sample_count) {
        if let TracepointData::Standard(tps) | TracepointData::Fastga(tps) = &record.tracepoints {
            for (first, second) in tps {
                all_first_vals.push(*first);
                all_second_vals.push(*second);
            }
        }
    }

    if all_first_vals.is_empty() {
        return (false, false);
    }

    // Analyze first_values
    let use_delta_first = should_use_delta(&all_first_vals);
    let use_delta_second = should_use_delta(&all_second_vals);

    info!(
        "Automatic analysis: sampled {} records, {} tracepoints - first_values: use_delta={}, second_values: use_delta={}",
        sample_count,
        all_first_vals.len(),
        use_delta_first,
        use_delta_second
    );

    (use_delta_first, use_delta_second)
}

// ============================================================================
// PAF PARSING
// ============================================================================

fn parse_paf_with_cigar(
    line: &str,
    string_table: &mut StringTable,
    tp_type: &TracepointType,
    max_complexity: usize,
    complexity_metric: &ComplexityMetric,
) -> io::Result<AlignmentRecord> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 12 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("PAF line has {} fields, expected at least 12", fields.len()),
        ));
    }

    let query_name = fields[0];
    let query_len = parse_usize(fields[1], "query_len")?;
    let query_start = parse_usize(fields[2], "query_start")?;
    let query_end = parse_usize(fields[3], "query_end")?;
    let strand = fields[4].chars().next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "Empty strand field")
    })?;
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

    // Extract CIGAR and other tags
    let mut cigar = None;
    let mut tags = Vec::new();

    for field in fields.iter().skip(12) {
        if field.starts_with("cg:Z:") {
            cigar = Some(field.strip_prefix("cg:Z:").unwrap());
        } else if let Some(tag) = parse_tag(field) {
            tags.push(tag);
        }
    }

    let cigar_str = cigar.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "PAF line missing CIGAR string (cg:Z: tag)",
        )
    })?;

    // Convert CIGAR to tracepoints based on type
    let tracepoints = match tp_type {
        TracepointType::Standard => {
            let tps = cigar_to_tracepoints(cigar_str, max_complexity, *complexity_metric)
                .into_iter()
                .map(|(a, b)| (a as u64, b as u64))
                .collect();
            TracepointData::Standard(tps)
        }
        TracepointType::Mixed => {
            let mixed = cigar_to_mixed_tracepoints(cigar_str, max_complexity, *complexity_metric);
            TracepointData::Mixed(
                mixed
                    .into_iter()
                    .map(|item| MixedTracepointItem::from(&item))
                    .collect(),
            )
        }
        TracepointType::Variable => {
            let tps = cigar_to_variable_tracepoints(cigar_str, max_complexity, *complexity_metric)
                .into_iter()
                .map(|(a, b)| (a as u64, b.map(|v| v as u64)))
                .collect();
            TracepointData::Variable(tps)
        }
        TracepointType::Fastga => {
            // FASTGA requires sequence information which we don't have here
            // Fall back to Standard tracepoints
            let tps = cigar_to_tracepoints(cigar_str, max_complexity, *complexity_metric)
                .into_iter()
                .map(|(a, b)| (a as u64, b as u64))
                .collect();
            TracepointData::Fastga(tps)
        }
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
        tp_type: *tp_type,
        complexity_metric: *complexity_metric,
        max_complexity: max_complexity as u64,
        tracepoints,
        tags,
    })
}

pub fn parse_paf_with_tracepoints(
    line: &str,
    string_table: &mut StringTable,
) -> io::Result<AlignmentRecord> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() < 12 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("PAF line has {} fields, expected at least 12", fields.len()),
        ));
    }

    let query_name = fields[0];
    let query_len = parse_usize(fields[1], "query_len")?;
    let query_start = parse_usize(fields[2], "query_start")?;
    let query_end = parse_usize(fields[3], "query_end")?;
    let strand = fields[4].chars().next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "Empty strand field")
    })?;
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

    // Extract tp:Z: and other tags
    let mut tp_str = None;
    let mut tags = Vec::new();

    for field in fields.iter().skip(12) {
        if field.starts_with("tp:Z:") {
            tp_str = Some(field.strip_prefix("tp:Z:").unwrap());
        } else if let Some(tag) = parse_tag(field) {
            tags.push(tag);
        }
    }

    let tp_data = tp_str.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "PAF line missing tracepoints (tp:Z: tag)",
        )
    })?;

    // Parse tracepoints string
    let (tracepoints, tp_type) = parse_tracepoints_auto(tp_data)?;

    // Infer complexity metric and max value from tracepoints
    let (complexity_metric, max_complexity) = match &tracepoints {
        TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
            let max_second = tps.iter().map(|(_, s)| *s).max().unwrap_or(0);
            (ComplexityMetric::EditDistance, max_second)
        }
        TracepointData::Variable(tps) => {
            let max_second = tps.iter().filter_map(|(_, s)| *s).max().unwrap_or(0);
            (ComplexityMetric::EditDistance, max_second)
        }
        TracepointData::Mixed(_) => {
            // For Mixed, we don't have a clear max_complexity
            (ComplexityMetric::EditDistance, 0)
        }
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
        tp_type,
        complexity_metric,
        max_complexity,
        tracepoints,
        tags,
    })
}

fn parse_tracepoints_auto(tp_str: &str) -> io::Result<(TracepointData, TracepointType)> {
    // Check if it's mixed format (contains letters for CIGAR ops)
    let has_letters = tp_str.contains(|c: char| c.is_alphabetic());

    if has_letters {
        // Parse as Mixed tracepoints
        let parts: Vec<&str> = tp_str.split(';').collect();
        let mut items = Vec::new();

        for part in parts {
            if part.contains(',') {
                // Standard tracepoint
                let coords: Vec<&str> = part.split(',').collect();
                if coords.len() != 2 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Invalid mixed tracepoint format: {}", part),
                    ));
                }
                let first = coords[0].parse::<u64>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid first value")
                })?;
                let second = coords[1].parse::<u64>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid second value")
                })?;
                items.push(MixedTracepointItem::Tracepoint(first, second));
            } else {
                // CIGAR operation
                let len = part[..part.len() - 1].parse::<u64>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid CIGAR length")
                })?;
                let op = part.chars().last().unwrap();
                items.push(MixedTracepointItem::CigarOp(len, op));
            }
        }

        Ok((TracepointData::Mixed(items), TracepointType::Mixed))
    } else {
        // Check if it's variable format (some second values missing)
        let parts: Vec<&str> = tp_str.split(';').collect();
        let has_single_values = parts.iter().any(|p| !p.contains(','));

        if has_single_values {
            // Variable tracepoints
            let mut tps = Vec::new();
            for part in parts {
                if part.contains(',') {
                    let coords: Vec<&str> = part.split(',').collect();
                    let first = coords[0].parse::<u64>().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid first value")
                    })?;
                    let second = coords[1].parse::<u64>().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid second value")
                    })?;
                    tps.push((first, Some(second)));
                } else {
                    let first = part.parse::<u64>().map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "Invalid first value")
                    })?;
                    tps.push((first, None));
                }
            }
            Ok((TracepointData::Variable(tps), TracepointType::Variable))
        } else {
            // Standard or FASTGA tracepoints
            let mut tps = Vec::new();
            for part in parts {
                let coords: Vec<&str> = part.split(',').collect();
                if coords.len() != 2 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Invalid tracepoint format: {}", part),
                    ));
                }
                let first = coords[0].parse::<u64>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid first value")
                })?;
                let second = coords[1].parse::<u64>().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid second value")
                })?;
                tps.push((first, second));
            }

            // Assume Standard for now (FASTGA detection would require spacing info)
            Ok((TracepointData::Standard(tps), TracepointType::Standard))
        }
    }
}

// ============================================================================
// PAF WRITING
// ============================================================================

pub fn write_paf_line<W: Write>(
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
    let tp_str = format_tracepoints(&record.tracepoints);
    write!(writer, "\ttp:Z:{}", tp_str)?;

    writeln!(writer)?;
    Ok(())
}

pub fn format_tracepoints(tps: &TracepointData) -> String {
    match tps {
        TracepointData::Standard(tps) | TracepointData::Fastga(tps) => tps
            .iter()
            .map(|(a, b)| format!("{},{}", a, b))
            .collect::<Vec<_>>()
            .join(";"),
        TracepointData::Variable(tps) => tps
            .iter()
            .map(|(a, b)| {
                if let Some(b_val) = b {
                    format!("{},{}", a, b_val)
                } else {
                    format!("{}", a)
                }
            })
            .collect::<Vec<_>>()
            .join(";"),
        TracepointData::Mixed(items) => items
            .iter()
            .map(|item| match item {
                MixedTracepointItem::Tracepoint(a, b) => format!("{},{}", a, b),
                MixedTracepointItem::CigarOp(len, op) => format!("{}{}", len, op),
            })
            .collect::<Vec<_>>()
            .join(";"),
    }
}

// ============================================================================
// COMPRESSION
// ============================================================================

/// Encode PAF with CIGAR to binary with tracepoints
pub fn encode_cigar_to_binary(
    input_path: &str,
    output_path: &str,
    tp_type: &TracepointType,
    max_complexity: usize,
    complexity_metric: &ComplexityMetric,
    distance: Distance,
    strategy: CompressionStrategy,
) -> io::Result<()> {
    info!("Encoding CIGAR with {} strategy...", strategy);

    let input = open_paf_reader(input_path)?;
    let mut string_table = StringTable::new();
    let mut records = Vec::new();

    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        match parse_paf_with_cigar(
            &line,
            &mut string_table,
            tp_type,
            max_complexity,
            complexity_metric,
        ) {
            Ok(record) => records.push(record),
            Err(e) => {
                error!("Line {}: {}", line_num + 1, e);
                return Err(e);
            }
        }
    }

    write_binary(output_path, &records, &string_table, strategy, *tp_type, max_complexity as u64, *complexity_metric, distance)?;

    info!(
        "Encoded {} records ({} unique names) with {} strategy",
        records.len(),
        string_table.len(),
        strategy
    );
    Ok(())
}

pub fn compress_paf(
    input_path: &str,
    output_path: &str,
    strategy: CompressionStrategy,
    tp_type: TracepointType,
    max_complexity: u64,
    complexity_metric: ComplexityMetric,
    distance: Distance,
) -> io::Result<()> {
    info!("Compressing PAF with {} strategy...", strategy);

    // Pass 1: Build string table + collect sample for analysis
    let mut string_table = StringTable::new();
    let mut sample = Vec::new();
    let mut record_count = 0u64;

    let input = open_paf_reader(input_path)?;
    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') { continue; }

        let record = parse_paf_with_tracepoints(&line, &mut string_table)
            .map_err(|e| { error!("Line {}: {}", line_num + 1, e); e })?;

        if sample.len() < 1000 { sample.push(record); }
        record_count += 1;
    }

    // Verify file is not empty
    if sample.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "No records found in PAF file"));
    }

    // Analyze sample for Automatic mode
    let (use_delta_first, use_delta_second) = match strategy {
        CompressionStrategy::Automatic(_) => analyze_smart_light_compression(&sample),
        _ => (false, false),
    };

    // Pass 2: Stream write - Header → StringTable → Records
    let mut output = File::create(output_path).map_err(|e| {
        io::Error::new(e.kind(), format!("Failed to create output file '{}': {}", output_path, e))
    })?;

    // Write header
    let header = BinaryPafHeader::new(record_count, string_table.len() as u64, strategy, use_delta_first, use_delta_second, tp_type, complexity_metric, max_complexity, distance);
    header.write(&mut output)?;

    // Write string table
    string_table.write(&mut output)?;

    // Write records
    let mut writer = BufWriter::new(&mut output);
    let input = open_paf_reader(input_path)?;
    let mut temp_table = string_table.clone();
    for line_result in input.lines() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') { continue; }

        let record = parse_paf_with_tracepoints(&line, &mut temp_table)?;
        match strategy {
            CompressionStrategy::Automatic(_) => record.write_automatic(&mut writer, use_delta_first, use_delta_second, strategy)?,
            CompressionStrategy::VarintZstd(_) => record.write(&mut writer, false, strategy)?,
            CompressionStrategy::DeltaVarintZstd(_) => record.write(&mut writer, true, strategy)?,
        }
    }
    writer.flush()?;

    info!("Compressed {} records ({} unique names) with {} strategy", record_count, string_table.len(), strategy);
    Ok(())
}

pub fn compress_paf_with_cigar(
    input_path: &str,
    output_path: &str,
    strategy: CompressionStrategy,
    tp_type: TracepointType,
    max_complexity: u64,
    complexity_metric: ComplexityMetric,
    distance: Distance,
) -> io::Result<()> {
    info!("Compressing PAF with CIGAR using {} strategy...", strategy);

    // Pass 1: Build string table + collect sample for analysis
    let mut string_table = StringTable::new();
    let mut sample = Vec::new();
    let mut record_count = 0u64;

    let input = open_paf_reader(input_path)?;
    for (line_num, line_result) in input.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') { continue; }

        let record = parse_paf_with_cigar(&line, &mut string_table, &tp_type, max_complexity as usize, &complexity_metric)
            .map_err(|e| { error!("Line {}: {}", line_num + 1, e); e })?;

        if sample.len() < 1000 { sample.push(record); }
        record_count += 1;
    }

    // Verify file is not empty
    if sample.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "No records found in PAF file"));
    }

    // Analyze sample for Automatic mode
    let (use_delta_first, use_delta_second) = match strategy {
        CompressionStrategy::Automatic(_) => analyze_smart_light_compression(&sample),
        _ => (false, false),
    };

    // Pass 2: Stream write - Header → StringTable → Records
    let mut output = File::create(output_path).map_err(|e| {
        io::Error::new(e.kind(), format!("Failed to create output file '{}': {}", output_path, e))
    })?;

    // Write header
    let header = BinaryPafHeader::new(record_count, string_table.len() as u64, strategy, use_delta_first, use_delta_second, tp_type, complexity_metric, max_complexity, distance);
    header.write(&mut output)?;

    // Write string table
    string_table.write(&mut output)?;

    // Write records
    let mut writer = BufWriter::new(&mut output);
    let input = open_paf_reader(input_path)?;
    let mut temp_table = string_table.clone();
    for line_result in input.lines() {
        let line = line_result?;
        if line.trim().is_empty() || line.starts_with('#') { continue; }

        let record = parse_paf_with_cigar(&line, &mut temp_table, &tp_type, max_complexity as usize, &complexity_metric)?;
        match strategy {
            CompressionStrategy::Automatic(_) => record.write_automatic(&mut writer, use_delta_first, use_delta_second, strategy)?,
            CompressionStrategy::VarintZstd(_) => record.write(&mut writer, false, strategy)?,
            CompressionStrategy::DeltaVarintZstd(_) => record.write(&mut writer, true, strategy)?,
        }
    }
    writer.flush()?;

    info!("Compressed {} records ({} unique names) with {} strategy", record_count, string_table.len(), strategy);
    Ok(())
}

fn write_binary(
    output_path: &str,
    records: &[AlignmentRecord],
    string_table: &StringTable,
    strategy: CompressionStrategy,
    tp_type: TracepointType,
    max_complexity: u64,
    complexity_metric: ComplexityMetric,
    distance: Distance,
) -> io::Result<()> {
    let mut output = File::create(output_path).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to create output file '{}': {}", output_path, e),
        )
    })?;

    // Verify records is not empty
    if records.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "No records provided"));
    }

    // Analyze data for Automatic mode
    let (use_delta_first, use_delta_second) = match strategy {
        CompressionStrategy::Automatic(_) => analyze_smart_light_compression(records),
        _ => (false, false),
    };

    // Write header
    let header = BinaryPafHeader::new(
        records.len() as u64,
        string_table.len() as u64,
        strategy,
        use_delta_first,
        use_delta_second,
        tp_type,
        complexity_metric,
        max_complexity,
        distance,
    );
    header.write(&mut output)?;

    // Write string table
    string_table.write(&mut output)?;

    // Write records
    let mut writer = BufWriter::new(&mut output);
    match strategy {
        CompressionStrategy::Automatic(_) => {
            for record in records {
                record.write_automatic(&mut writer, use_delta_first, use_delta_second, strategy)?;
            }
        }
        CompressionStrategy::VarintZstd(_) => {
            for record in records {
                record.write(&mut writer, false, strategy)?;
            }
        }
        CompressionStrategy::DeltaVarintZstd(_) => {
            for record in records {
                record.write(&mut writer, true, strategy)?;
            }
        }
    }
    writer.flush()?;

    Ok(())
}

fn write_paf_output(
    output_path: &str,
    records: &[AlignmentRecord],
    string_table: &StringTable,
) -> io::Result<()> {
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

    for record in records {
        write_paf_line(&mut writer, record, string_table)?;
    }
    writer.flush()?;
    Ok(())
}

// ============================================================================
// DECOMPRESSION
// ============================================================================

pub fn decompress_paf(input_path: &str, output_path: &str) -> io::Result<()> {
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
        "Reading {} records ({} unique names) [{}]",
        header.num_records, header.num_strings, strategy
    );

    decompress_varint(reader, output_path, &header, strategy)
}

fn decompress_varint<R: Read>(
    mut reader: R,
    output_path: &str,
    header: &BinaryPafHeader,
    strategy: CompressionStrategy,
) -> io::Result<()> {
    // Read string table
    let string_table = StringTable::read(&mut reader)?;

    // Stream write records
    let output: Box<dyn Write> = if output_path == "-" {
        Box::new(io::stdout())
    } else {
        Box::new(File::create(output_path).map_err(|e| {
            io::Error::new(e.kind(), format!("Failed to create output file '{}': {}", output_path, e))
        })?)
    };
    let mut writer = BufWriter::new(output);

    match strategy {
        CompressionStrategy::Automatic(_) => {
            let use_delta_first = header.use_delta_first();
            let use_delta_second = header.use_delta_second();
            for _ in 0..header.num_records {
                let record = read_record_automatic(
                    &mut reader,
                    use_delta_first,
                    use_delta_second,
                    header.tracepoint_type,
                    header.complexity_metric,
                    header.max_complexity,
                )?;
                write_paf_line(&mut writer, &record, &string_table)?;
            }
        }
        CompressionStrategy::VarintZstd(_) => {
            for _ in 0..header.num_records {
                let record = read_record_varint(
                    &mut reader,
                    false,
                    header.tracepoint_type,
                    header.complexity_metric,
                    header.max_complexity,
                )?;
                write_paf_line(&mut writer, &record, &string_table)?;
            }
        }
        CompressionStrategy::DeltaVarintZstd(_) => {
            for _ in 0..header.num_records {
                let record = AlignmentRecord::read(
                    &mut reader,
                    header.tracepoint_type,
                    header.complexity_metric,
                    header.max_complexity,
                )?;
                write_paf_line(&mut writer, &record, &string_table)?;
            }
        }
    }
    writer.flush()?;

    info!("Decompressed {} records", header.num_records);
    Ok(())
}

fn read_record_varint<R: Read>(
    reader: &mut R,
    _use_delta: bool,
    tp_type: TracepointType,
    complexity_metric: ComplexityMetric,
    max_complexity: u64,
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

    let tracepoints = read_tracepoints_raw(reader, tp_type)?;

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
        tp_type,
        complexity_metric,
        max_complexity,
        tracepoints,
        tags,
    })
}

fn read_record_automatic<R: Read>(
    reader: &mut R,
    use_delta_first: bool,
    use_delta_second: bool,
    tp_type: TracepointType,
    complexity_metric: ComplexityMetric,
    max_complexity: u64,
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

    let tracepoints = read_tracepoints_automatic(reader, tp_type, use_delta_first, use_delta_second)?;

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
        tp_type,
        complexity_metric,
        max_complexity,
        tracepoints,
        tags,
    })
}

fn read_tracepoints_raw<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
) -> io::Result<TracepointData> {
    let num_items = read_varint(reader)? as usize;
    if num_items == 0 {
        return Ok(match tp_type {
            TracepointType::Standard => TracepointData::Standard(Vec::new()),
            _ => TracepointData::Fastga(Vec::new()),
        });
    }

    let pos_len = read_varint(reader)? as usize;
    let mut pos_compressed = vec![0u8; pos_len];
    reader.read_exact(&mut pos_compressed)?;
    let score_len = read_varint(reader)? as usize;
    let mut score_compressed = vec![0u8; score_len];
    reader.read_exact(&mut score_compressed)?;

    let pos_buf = zstd::decode_all(&pos_compressed[..])?;
    let score_buf = zstd::decode_all(&score_compressed[..])?;

    let mut pos_reader = &pos_buf[..];
    let mut positions = Vec::with_capacity(num_items);
    for _ in 0..num_items {
        positions.push(read_varint(&mut pos_reader)?);
    }

    let mut score_reader = &score_buf[..];
    let mut scores = Vec::with_capacity(num_items);
    for _ in 0..num_items {
        scores.push(read_varint(&mut score_reader)?);
    }

    let tps: Vec<(u64, u64)> = positions.into_iter().zip(scores).collect();
    Ok(match tp_type {
        TracepointType::Standard => TracepointData::Standard(tps),
        _ => TracepointData::Fastga(tps),
    })
}

fn read_tracepoints_automatic<R: Read>(
    reader: &mut R,
    tp_type: TracepointType,
    use_delta_first: bool,
    use_delta_second: bool,
) -> io::Result<TracepointData> {
    let num_items = read_varint(reader)? as usize;
    if num_items == 0 {
        return Ok(match tp_type {
            TracepointType::Standard => TracepointData::Standard(Vec::new()),
            _ => TracepointData::Fastga(Vec::new()),
        });
    }

    let first_len = read_varint(reader)? as usize;
    let mut first_compressed = vec![0u8; first_len];
    reader.read_exact(&mut first_compressed)?;
    let second_len = read_varint(reader)? as usize;
    let mut second_compressed = vec![0u8; second_len];
    reader.read_exact(&mut second_compressed)?;

    let first_buf = zstd::decode_all(&first_compressed[..])?;
    let second_buf = zstd::decode_all(&second_compressed[..])?;

    // Decode first values (delta or raw)
    let mut first_reader = &first_buf[..];
    let first_vals = if use_delta_first {
        let mut deltas = Vec::with_capacity(num_items);
        for _ in 0..num_items {
            let zigzag = read_varint(&mut first_reader)?;
            let val = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
            deltas.push(val);
        }
        delta_decode(&deltas)
    } else {
        let mut vals = Vec::with_capacity(num_items);
        for _ in 0..num_items {
            vals.push(read_varint(&mut first_reader)?);
        }
        vals
    };

    // Decode second values (delta or raw)
    let mut second_reader = &second_buf[..];
    let second_vals = if use_delta_second {
        let mut deltas = Vec::with_capacity(num_items);
        for _ in 0..num_items {
            let zigzag = read_varint(&mut second_reader)?;
            let val = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
            deltas.push(val);
        }
        delta_decode(&deltas)
    } else {
        let mut vals = Vec::with_capacity(num_items);
        for _ in 0..num_items {
            vals.push(read_varint(&mut second_reader)?);
        }
        vals
    };

    let tps: Vec<(u64, u64)> = first_vals.into_iter().zip(second_vals).collect();
    Ok(match tp_type {
        TracepointType::Standard => TracepointData::Standard(tps),
        _ => TracepointData::Fastga(tps),
    })
}

// ============================================================================
// ALIGNMENT RECORD I/O
// ============================================================================

impl AlignmentRecord {
    pub(crate) fn write<W: Write>(&self, writer: &mut W, use_delta: bool, strategy: CompressionStrategy) -> io::Result<()> {
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
        if use_delta {
            self.write_tracepoints(writer, strategy)?;
        } else {
            self.write_tracepoints_raw(writer, strategy)?;
        }
        write_varint(writer, self.tags.len() as u64)?;
        for tag in &self.tags {
            tag.write(writer)?;
        }
        Ok(())
    }

    pub(crate) fn write_automatic<W: Write>(
        &self,
        writer: &mut W,
        use_delta_first: bool,
        use_delta_second: bool,
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
        self.write_tracepoints_automatic(writer, use_delta_first, use_delta_second, strategy)?;
        write_varint(writer, self.tags.len() as u64)?;
        for tag in &self.tags {
            tag.write(writer)?;
        }
        Ok(())
    }

    fn write_tracepoints<W: Write>(&self, writer: &mut W, strategy: CompressionStrategy) -> io::Result<()> {
        match &self.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                write_varint(writer, tps.len() as u64)?;
                if tps.is_empty() {
                    return Ok(());
                }
                let (first_vals, second_vals): (Vec<u64>, Vec<u64>) = tps.iter().copied().unzip();

                let use_delta = !matches!(self.tp_type, TracepointType::Fastga);
                let first_vals_encoded = if use_delta {
                    delta_encode(&first_vals)
                } else {
                    first_vals.iter().map(|&v| v as i64).collect()
                };

                let mut first_val_buf = Vec::with_capacity(first_vals_encoded.len() * 2);
                let mut second_val_buf = Vec::with_capacity(second_vals.len() * 2);

                for &val in &first_vals_encoded {
                    let zigzag = ((val << 1) ^ (val >> 63)) as u64;
                    write_varint(&mut first_val_buf, zigzag)?;
                }
                for &val in &second_vals {
                    write_varint(&mut second_val_buf, val)?;
                }

                let first_compressed = zstd::encode_all(&first_val_buf[..], strategy.zstd_level())?;
                let second_compressed = zstd::encode_all(&second_val_buf[..], strategy.zstd_level())?;

                write_varint(writer, first_compressed.len() as u64)?;
                writer.write_all(&first_compressed)?;
                write_varint(writer, second_compressed.len() as u64)?;
                writer.write_all(&second_compressed)?;
            }
            TracepointData::Variable(tps) => {
                write_varint(writer, tps.len() as u64)?;
                for (a, b_opt) in tps {
                    write_varint(writer, *a)?;
                    if let Some(b) = b_opt {
                        writer.write_all(&[1])?;
                        write_varint(writer, *b)?;
                    } else {
                        writer.write_all(&[0])?;
                    }
                }
            }
            TracepointData::Mixed(items) => {
                write_varint(writer, items.len() as u64)?;
                for item in items {
                    match item {
                        MixedTracepointItem::Tracepoint(a, b) => {
                            writer.write_all(&[0])?;
                            write_varint(writer, *a)?;
                            write_varint(writer, *b)?;
                        }
                        MixedTracepointItem::CigarOp(len, op) => {
                            writer.write_all(&[1])?;
                            write_varint(writer, *len)?;
                            writer.write_all(&[*op as u8])?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn write_tracepoints_raw<W: Write>(&self, writer: &mut W, strategy: CompressionStrategy) -> io::Result<()> {
        match &self.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                write_varint(writer, tps.len() as u64)?;
                if tps.is_empty() {
                    return Ok(());
                }
                let (first_vals, second_vals): (Vec<u64>, Vec<u64>) = tps.iter().copied().unzip();

                let mut first_val_buf = Vec::with_capacity(first_vals.len() * 2);
                let mut second_val_buf = Vec::with_capacity(second_vals.len() * 2);

                for &val in &first_vals {
                    write_varint(&mut first_val_buf, val)?;
                }
                for &val in &second_vals {
                    write_varint(&mut second_val_buf, val)?;
                }

                let first_compressed = zstd::encode_all(&first_val_buf[..], strategy.zstd_level())?;
                let second_compressed = zstd::encode_all(&second_val_buf[..], strategy.zstd_level())?;

                write_varint(writer, first_compressed.len() as u64)?;
                writer.write_all(&first_compressed)?;
                write_varint(writer, second_compressed.len() as u64)?;
                writer.write_all(&second_compressed)?;
            }
            TracepointData::Variable(tps) => {
                write_varint(writer, tps.len() as u64)?;
                for (a, b_opt) in tps {
                    write_varint(writer, *a)?;
                    if let Some(b) = b_opt {
                        writer.write_all(&[1])?;
                        write_varint(writer, *b)?;
                    } else {
                        writer.write_all(&[0])?;
                    }
                }
            }
            TracepointData::Mixed(items) => {
                write_varint(writer, items.len() as u64)?;
                for item in items {
                    match item {
                        MixedTracepointItem::Tracepoint(a, b) => {
                            writer.write_all(&[0])?;
                            write_varint(writer, *a)?;
                            write_varint(writer, *b)?;
                        }
                        MixedTracepointItem::CigarOp(len, op) => {
                            writer.write_all(&[1])?;
                            write_varint(writer, *len)?;
                            writer.write_all(&[*op as u8])?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn write_tracepoints_automatic<W: Write>(
        &self,
        writer: &mut W,
        use_delta_first: bool,
        use_delta_second: bool,
        strategy: CompressionStrategy,
    ) -> io::Result<()> {
        match &self.tracepoints {
            TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                write_varint(writer, tps.len() as u64)?;
                if tps.is_empty() {
                    return Ok(());
                }
                let (first_vals, second_vals): (Vec<u64>, Vec<u64>) = tps.iter().copied().unzip();

                let mut first_val_buf = Vec::with_capacity(first_vals.len() * 2);
                if use_delta_first {
                    let first_vals_encoded = delta_encode(&first_vals);
                    for &val in &first_vals_encoded {
                        let zigzag = ((val << 1) ^ (val >> 63)) as u64;
                        write_varint(&mut first_val_buf, zigzag)?;
                    }
                } else {
                    for &val in &first_vals {
                        write_varint(&mut first_val_buf, val)?;
                    }
                }

                let mut second_val_buf = Vec::with_capacity(second_vals.len() * 2);
                if use_delta_second {
                    let second_vals_encoded = delta_encode(&second_vals);
                    for &val in &second_vals_encoded {
                        let zigzag = ((val << 1) ^ (val >> 63)) as u64;
                        write_varint(&mut second_val_buf, zigzag)?;
                    }
                } else {
                    for &val in &second_vals {
                        write_varint(&mut second_val_buf, val)?;
                    }
                }

                let first_compressed = zstd::encode_all(&first_val_buf[..], strategy.zstd_level())?;
                let second_compressed = zstd::encode_all(&second_val_buf[..], strategy.zstd_level())?;

                write_varint(writer, first_compressed.len() as u64)?;
                writer.write_all(&first_compressed)?;
                write_varint(writer, second_compressed.len() as u64)?;
                writer.write_all(&second_compressed)?;
            }
            TracepointData::Variable(tps) => {
                write_varint(writer, tps.len() as u64)?;
                for (a, b_opt) in tps {
                    write_varint(writer, *a)?;
                    if let Some(b) = b_opt {
                        writer.write_all(&[1])?;
                        write_varint(writer, *b)?;
                    } else {
                        writer.write_all(&[0])?;
                    }
                }
            }
            TracepointData::Mixed(items) => {
                write_varint(writer, items.len() as u64)?;
                for item in items {
                    match item {
                        MixedTracepointItem::Tracepoint(a, b) => {
                            writer.write_all(&[0])?;
                            write_varint(writer, *a)?;
                            write_varint(writer, *b)?;
                        }
                        MixedTracepointItem::CigarOp(len, op) => {
                            writer.write_all(&[1])?;
                            write_varint(writer, *len)?;
                            writer.write_all(&[*op as u8])?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn read<R: Read>(
        reader: &mut R,
        tp_type: TracepointType,
        complexity_metric: ComplexityMetric,
        max_complexity: u64,
    ) -> io::Result<Self> {
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
        let tracepoints = Self::read_tracepoints(reader, tp_type)?;
        let num_tags = read_varint(reader)? as usize;
        let mut tags = Vec::with_capacity(num_tags);
        for _ in 0..num_tags {
            tags.push(Tag::read(reader)?);
        }
        Ok(Self {
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
            tp_type,
            complexity_metric,
            max_complexity,
            tracepoints,
            tags,
        })
    }

    pub(crate) fn read_tracepoints<R: Read>(
        reader: &mut R,
        tp_type: TracepointType,
    ) -> io::Result<TracepointData> {
        let num_items = read_varint(reader)? as usize;
        match tp_type {
            TracepointType::Standard | TracepointType::Fastga => {
                if num_items == 0 {
                    return Ok(match tp_type {
                        TracepointType::Standard => TracepointData::Standard(Vec::new()),
                        _ => TracepointData::Fastga(Vec::new()),
                    });
                }
                let pos_len = read_varint(reader)? as usize;
                let mut pos_compressed = vec![0u8; pos_len];
                reader.read_exact(&mut pos_compressed)?;
                let score_len = read_varint(reader)? as usize;
                let mut score_compressed = vec![0u8; score_len];
                reader.read_exact(&mut score_compressed)?;

                let pos_buf = zstd::decode_all(&pos_compressed[..])?;
                let score_buf = zstd::decode_all(&score_compressed[..])?;

                let mut pos_reader = &pos_buf[..];
                let mut pos_values = Vec::with_capacity(num_items);
                for _ in 0..num_items {
                    let zigzag = read_varint(&mut pos_reader)?;
                    let val = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
                    pos_values.push(val);
                }

                let positions: Vec<u64> = if matches!(tp_type, TracepointType::Fastga) {
                    pos_values.iter().map(|&v| v as u64).collect()
                } else {
                    delta_decode(&pos_values)
                };

                let mut score_reader = &score_buf[..];
                let mut scores = Vec::with_capacity(num_items);
                for _ in 0..num_items {
                    scores.push(read_varint(&mut score_reader)?);
                }

                let tps: Vec<(u64, u64)> = positions.into_iter().zip(scores).collect();
                Ok(match tp_type {
                    TracepointType::Standard => TracepointData::Standard(tps),
                    _ => TracepointData::Fastga(tps),
                })
            }
            TracepointType::Variable => {
                let mut tps = Vec::with_capacity(num_items);
                for _ in 0..num_items {
                    let a = read_varint(reader)?;
                    let mut flag = [0u8; 1];
                    reader.read_exact(&mut flag)?;
                    let b_opt = if flag[0] == 1 {
                        Some(read_varint(reader)?)
                    } else {
                        None
                    };
                    tps.push((a, b_opt));
                }
                Ok(TracepointData::Variable(tps))
            }
            TracepointType::Mixed => {
                let mut items = Vec::with_capacity(num_items);
                for _ in 0..num_items {
                    let mut item_type = [0u8; 1];
                    reader.read_exact(&mut item_type)?;
                    match item_type[0] {
                        0 => {
                            let a = read_varint(reader)?;
                            let b = read_varint(reader)?;
                            items.push(MixedTracepointItem::Tracepoint(a, b));
                        }
                        1 => {
                            let len = read_varint(reader)?;
                            let mut op = [0u8; 1];
                            reader.read_exact(&mut op)?;
                            items.push(MixedTracepointItem::CigarOp(len, op[0] as char));
                        }
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "Invalid mixed item type",
                            ))
                        }
                    }
                }
                Ok(TracepointData::Mixed(items))
            }
        }
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
    StringTable::read(&mut reader)?;

    let mut offsets = Vec::with_capacity(header.num_records as usize);
    for _ in 0..header.num_records {
        offsets.push(reader.stream_position()?);
        skip_record(&mut reader, false)?;
    }

    info!("Index built: {} records", offsets.len());
    Ok(BpafIndex { offsets })
}

#[inline]
fn skip_record<R: Read + Seek>(reader: &mut R, _is_adaptive: bool) -> io::Result<()> {
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

    // Skip tracepoints
    let num_items = read_varint(reader)? as usize;
    if num_items > 0 {
        let pos_len = read_varint(reader)? as usize;
        reader.seek(SeekFrom::Current(pos_len as i64))?;

        let score_len = read_varint(reader)? as usize;
        reader.seek(SeekFrom::Current(score_len as i64))?;
    }

    // Skip tags
    let num_tags = read_varint(reader)? as usize;
    for _ in 0..num_tags {
        reader.seek(SeekFrom::Current(2))?; // key
        let mut tag_type = [0u8; 1];
        reader.read_exact(&mut tag_type)?;
        match tag_type[0] {
            b'i' => reader.seek(SeekFrom::Current(8))?,
            b'f' => reader.seek(SeekFrom::Current(4))?,
            b'Z' => {
                let len = read_varint(reader)? as usize;
                reader.seek(SeekFrom::Current(len as i64))?
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Invalid tag type",
                ))
            }
        };
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
            info!("Loading existing index: {}", idx_path);
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

        let string_table = StringTable::new();

        Ok(Self {
            file,
            index,
            header,
            string_table,
        })
    }

    /// Open a BPAF file without index (for offset-based access only)
    pub fn open_without_index(bpaf_path: &str) -> io::Result<Self> {
        let mut file = File::open(bpaf_path)?;
        let header = BinaryPafHeader::read(&mut file)?;

        let index = BpafIndex {
            offsets: Vec::new(),
        };
        let string_table = StringTable::new();

        Ok(Self {
            file,
            index,
            header,
            string_table,
        })
    }

    /// Load string table (call this if you need sequence names)
    pub fn load_string_table(&mut self) -> io::Result<()> {
        if !self.string_table.is_empty() {
            return Ok(());
        }

        self.file.seek(SeekFrom::Start(0))?;
        BinaryPafHeader::read(&mut self.file)?;
        self.string_table = StringTable::read(&mut self.file)?;
        Ok(())
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

    /// Get immutable reference to string table (must be loaded first with load_string_table)
    pub fn string_table_ref(&self) -> &StringTable {
        &self.string_table
    }

    /// Get full alignment record by ID - O(1) random access
    pub fn get_alignment_record(&mut self, record_id: u64) -> io::Result<AlignmentRecord> {
        if record_id >= self.index.len() as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Record ID {} out of range (max: {})",
                    record_id,
                    self.index.len() - 1
                ),
            ));
        }

        let offset = self.index.offsets[record_id as usize];
        self.get_alignment_record_at_offset(offset)
    }

    /// Get alignment record by file offset (for impg compatibility)
    pub fn get_alignment_record_at_offset(&mut self, offset: u64) -> io::Result<AlignmentRecord> {
        self.file.seek(SeekFrom::Start(offset))?;

        let strategy = self.header.strategy()?;
        match strategy {
            CompressionStrategy::Automatic(_) => {
                let use_delta_first = self.header.use_delta_first();
                let use_delta_second = self.header.use_delta_second();
                read_record_automatic(
                    &mut self.file,
                    use_delta_first,
                    use_delta_second,
                    self.header.tracepoint_type,
                    self.header.complexity_metric,
                    self.header.max_complexity,
                )
            }
            CompressionStrategy::VarintZstd(_) => read_record_varint(
                &mut self.file,
                false,
                self.header.tracepoint_type,
                self.header.complexity_metric,
                self.header.max_complexity,
            ),
            CompressionStrategy::DeltaVarintZstd(_) => AlignmentRecord::read(
                &mut self.file,
                self.header.tracepoint_type,
                self.header.complexity_metric,
                self.header.max_complexity,
            ),
        }
    }

    /// Get tracepoints only (optimized) - O(1) random access by record ID
    pub fn get_tracepoints(
        &mut self,
        record_id: u64,
    ) -> io::Result<(TracepointData, TracepointType, ComplexityMetric, u64)> {
        if record_id >= self.index.len() as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Record ID {} out of range (max: {})",
                    record_id,
                    self.index.len() - 1
                ),
            ));
        }

        let tracepoint_offset = self.get_tracepoint_offset(record_id)?;
        self.get_tracepoints_at_offset(tracepoint_offset)
    }

    /// Get tracepoint offset for a specific record ID
    /// Returns the byte offset where tracepoint data starts within the record
    /// Use this to build external indexes that store tracepoint offsets for fastest access
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

    /// Get tracepoints by tracepoint offset (fastest access)
    /// Seeks directly to tracepoint data within a record
    /// Use this when your external index stores tracepoint offsets for O(1) access
    pub fn get_tracepoints_at_offset(
        &mut self,
        tracepoint_offset: u64,
    ) -> io::Result<(TracepointData, TracepointType, ComplexityMetric, u64)> {
        self.file.seek(SeekFrom::Start(tracepoint_offset))?;

        let tp_type = self.header.tracepoint_type;
        let complexity_metric = self.header.complexity_metric;
        let max_complexity = self.header.max_complexity;

        let tracepoints = match self.header.strategy()? {
            CompressionStrategy::Automatic(_) => {
                read_tracepoints_automatic(
                    &mut self.file,
                    tp_type,
                    self.header.use_delta_first(),
                    self.header.use_delta_second(),
                )?
            }
            CompressionStrategy::VarintZstd(_) => {
                read_tracepoints_raw(&mut self.file, tp_type)?
            }
            CompressionStrategy::DeltaVarintZstd(_) => {
                AlignmentRecord::read_tracepoints(&mut self.file, tp_type)?
            }
        };

        Ok((tracepoints, tp_type, complexity_metric, max_complexity))
    }

    /// Iterator over all records (sequential access)
    pub fn iter_records(&mut self) -> RecordIterator<'_> {
        RecordIterator {
            reader: self,
            current_id: 0,
        }
    }
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
