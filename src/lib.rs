/// Binary PAF format for efficient storage of sequence alignments with tracepoints
///
/// Format: [Header] → [StringTable] → [Records]
/// - Header: Magic "BPAF" + version + metadata
/// - StringTable: Deduplicated sequence names with lengths
/// - Records: Core PAF fields + compressed tracepoints

mod utils;
mod format;
mod binary;

use std::io;

// Re-export public types
pub use lib_wfa2::affine_wavefront::Distance;
pub use lib_tracepoints::{ComplexityMetric, TracepointType};

pub use format::{
    AlignmentRecord, BinaryPafHeader, CompressionStrategy, MixedTracepointItem, StringTable, Tag,
    TagValue, TracepointData,
};

pub use binary::{
    build_index, is_binary_paf, write_paf_line, BpafIndex, BpafReader, RecordIterator,
};

// ============================================================================
// PUBLIC API
// ============================================================================

/// Detect if file is binary PAF
pub fn detect_binary(path: &str) -> io::Result<bool> {
    binary::is_binary_paf(path)
}

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
    binary::encode_cigar_to_binary(
        input_path,
        output_path,
        tp_type,
        max_complexity,
        complexity_metric,
        distance,
        strategy,
    )
}

/// Compress PAF with tracepoints to binary format
pub fn compress_paf(
    input_path: &str,
    output_path: &str,
    strategy: CompressionStrategy,
    tp_type: TracepointType,
    max_complexity: u64,
    complexity_metric: ComplexityMetric,
    distance: Distance,
) -> io::Result<()> {
    binary::compress_paf(
        input_path,
        output_path,
        strategy,
        tp_type,
        max_complexity,
        complexity_metric,
        distance,
    )
}

/// Compress PAF with CIGAR to binary format
pub fn compress_paf_with_cigar(
    input_path: &str,
    output_path: &str,
    strategy: CompressionStrategy,
    tp_type: TracepointType,
    max_complexity: u64,
    complexity_metric: ComplexityMetric,
    distance: Distance,
) -> io::Result<()> {
    binary::compress_paf_with_cigar(
        input_path,
        output_path,
        strategy,
        tp_type,
        max_complexity,
        complexity_metric,
        distance,
    )
}

/// Convert binary PAF to text format
pub fn decompress_paf(input_path: &str, output_path: &str) -> io::Result<()> {
    binary::decompress_paf(input_path, output_path)
}

/// Format tracepoints as string (for tp:Z: tag)
pub fn format_tracepoints(tps: &TracepointData) -> String {
    binary::format_tracepoints(tps)
}

/// Parse PAF line with tracepoints
pub fn parse_paf_with_tracepoints(
    line: &str,
    string_table: &mut StringTable,
) -> io::Result<AlignmentRecord> {
    binary::parse_paf_with_tracepoints(line, string_table)
}
