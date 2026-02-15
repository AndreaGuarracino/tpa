//! CIGAR string analysis and reconstruction utilities

use crate::Distance;
use lib_wfa2::affine_wavefront::AffineWavefronts;
use tracepoints::{
    mixed_tracepoints_to_cigar_with_aligner, tracepoints_to_cigar_fastga_with_aligner,
    tracepoints_to_cigar_with_aligner, variable_tracepoints_to_cigar_with_aligner,
    ComplexityMetric, TracepointData,
};

/// Statistics extracted from a CIGAR string
///
/// Provides methods to calculate various identity metrics and alignment scores
/// compatible with WFA2-lib scoring behavior.
#[derive(Debug, Clone, Default)]
pub struct CigarStats {
    /// Number of matching bases (M or = operations)
    pub matches: usize,
    /// Number of mismatching bases (X operations)
    pub mismatches: usize,
    /// Length of each insertion run
    insertion_runs: Vec<usize>,
    /// Length of each deletion run
    deletion_runs: Vec<usize>,
}

impl CigarStats {
    /// Parse a CIGAR string and extract all statistics
    pub fn from_cigar(cigar: &str) -> Self {
        let mut stats = CigarStats::default();
        let mut num_buffer = String::new();

        for c in cigar.chars() {
            if c.is_ascii_digit() {
                num_buffer.push(c);
            } else {
                let len = num_buffer.parse::<usize>().unwrap_or(0);
                num_buffer.clear();

                match c {
                    'M' | '=' => stats.matches += len,
                    'X' => stats.mismatches += len,
                    'I' => stats.insertion_runs.push(len),
                    'D' => stats.deletion_runs.push(len),
                    'S' | 'H' | 'P' | 'N' => {}
                    _ => {}
                }
            }
        }

        stats
    }

    /// Number of insertion events (runs)
    pub fn insertions(&self) -> usize {
        self.insertion_runs.len()
    }

    /// Total inserted base pairs
    pub fn inserted_bp(&self) -> usize {
        self.insertion_runs.iter().sum()
    }

    /// Number of deletion events (runs)
    pub fn deletions(&self) -> usize {
        self.deletion_runs.len()
    }

    /// Total deleted base pairs
    pub fn deleted_bp(&self) -> usize {
        self.deletion_runs.iter().sum()
    }

    /// Calculate gap-compressed identity: matches / (matches + mismatches + gap_events)
    pub fn gap_compressed_identity(&self) -> f32 {
        let denom = self.matches + self.mismatches + self.insertions() + self.deletions();
        if denom > 0 {
            (self.matches as f32) / (denom as f32)
        } else {
            0.0
        }
    }

    /// Calculate block identity: matches / (matches + edit_distance)
    pub fn block_identity(&self) -> f32 {
        let edit_distance = self.mismatches + self.inserted_bp() + self.deleted_bp();
        let denom = self.matches + edit_distance;
        if denom > 0 {
            (self.matches as f32) / (denom as f32)
        } else {
            0.0
        }
    }

    /// Calculate alignment score using the specified distance model
    ///
    /// Score calculation matches WFA2-lib behavior:
    /// - Edit: -(mismatches + inserted_bp + deleted_bp)
    /// - GapAffine: -(mismatch * X + gap_open * gaps + gap_ext * gap_bp)
    /// - GapAffine2p: uses dual gap model, choosing the better penalty for each gap
    pub fn alignment_score(&self, distance: &Distance) -> i32 {
        let mismatches = self.mismatches as i32;

        match distance {
            Distance::Edit => {
                let inserted_bp = self.inserted_bp() as i32;
                let deleted_bp = self.deleted_bp() as i32;
                -(mismatches + inserted_bp + deleted_bp)
            }
            Distance::GapAffine {
                mismatch,
                gap_opening,
                gap_extension,
            } => {
                let mismatch_penalty = mismatch * mismatches;

                let insertion_penalty: i32 = self
                    .insertion_runs
                    .iter()
                    .map(|&len| gap_opening + gap_extension * (len as i32))
                    .sum();

                let deletion_penalty: i32 = self
                    .deletion_runs
                    .iter()
                    .map(|&len| gap_opening + gap_extension * (len as i32))
                    .sum();

                -(mismatch_penalty + insertion_penalty + deletion_penalty)
            }
            Distance::GapAffine2p {
                mismatch,
                gap_opening1,
                gap_extension1,
                gap_opening2,
                gap_extension2,
            } => {
                let mismatch_penalty = mismatch * mismatches;

                let insertion_penalty: i32 = self
                    .insertion_runs
                    .iter()
                    .map(|&len| {
                        let len = len as i32;
                        let score1 = gap_opening1 + gap_extension1 * len;
                        let score2 = gap_opening2 + gap_extension2 * len;
                        std::cmp::min(score1, score2)
                    })
                    .sum();

                let deletion_penalty: i32 = self
                    .deletion_runs
                    .iter()
                    .map(|&len| {
                        let len = len as i32;
                        let score1 = gap_opening1 + gap_extension1 * len;
                        let score2 = gap_opening2 + gap_extension2 * len;
                        std::cmp::min(score1, score2)
                    })
                    .sum();

                -(mismatch_penalty + insertion_penalty + deletion_penalty)
            }
        }
    }
}

/// Calculate alignment score from a CIGAR string using the specified distance model
pub fn calculate_alignment_score(cigar: &str, distance: &Distance) -> i32 {
    CigarStats::from_cigar(cigar).alignment_score(distance)
}

/// Reconstruct a CIGAR string from tracepoint data
///
/// This is a unified dispatch function that handles all TracepointData variants,
/// delegating to the appropriate tracepoints crate function.
///
/// # Arguments
/// * `tp` - The tracepoint data to reconstruct from
/// * `query_seq` - Query sequence bytes
/// * `target_seq` - Target sequence bytes
/// * `query_offset` - Offset in query sequence (for FastGA)
/// * `target_offset` - Offset in target sequence (for FastGA)
/// * `distance` - Distance model for alignment (Edit, GapAffine, GapAffine2p)
/// * `metric` - Complexity metric used during tracepoint generation
/// * `spacing` - Trace spacing (only used for FastGA tracepoints)
/// * `complement` - Whether the alignment is on reverse complement strand (for FastGA)
pub fn reconstruct_cigar(
    tp: &TracepointData,
    query_seq: &[u8],
    target_seq: &[u8],
    query_offset: usize,
    target_offset: usize,
    distance: &Distance,
    metric: ComplexityMetric,
    spacing: u32,
    complement: bool,
) -> String {
    let mut aligner = distance.create_aligner(None, None);
    reconstruct_cigar_with_aligner_impl(
        tp,
        query_seq,
        target_seq,
        query_offset,
        target_offset,
        metric,
        spacing,
        complement,
        &mut aligner,
        None,
    )
}

/// Reconstruct a CIGAR string using heuristic mode (WFA2 realignment with band)
///
/// Creates an aligner internally for each call. For better performance when processing
/// many records, use `reconstruct_cigar_with_aligner()` to reuse an aligner.
///
/// # Arguments
/// * `tp` - The tracepoint data to reconstruct from
/// * `query_seq` - Query sequence bytes
/// * `target_seq` - Target sequence bytes
/// * `query_offset` - Offset in query sequence (for FastGA)
/// * `target_offset` - Offset in target sequence (for FastGA)
/// * `distance` - Distance model for alignment
/// * `metric` - Complexity metric (EditDistance or DiagonalDistance)
/// * `spacing` - Trace spacing (only used for FastGA tracepoints)
/// * `complement` - Whether the alignment is on reverse complement strand
/// * `max_complexity` - Maximum complexity value for band heuristic
pub fn reconstruct_cigar_with_heuristic(
    tp: &TracepointData,
    query_seq: &[u8],
    target_seq: &[u8],
    query_offset: usize,
    target_offset: usize,
    distance: &Distance,
    metric: ComplexityMetric,
    spacing: u32,
    complement: bool,
    max_complexity: u32,
) -> String {
    let mut aligner = distance.create_aligner(None, None);
    reconstruct_cigar_with_aligner_impl(
        tp,
        query_seq,
        target_seq,
        query_offset,
        target_offset,
        metric,
        spacing,
        complement,
        &mut aligner,
        Some(max_complexity),
    )
}

/// Reconstruct a CIGAR string using a caller-provided aligner
///
/// Useful when processing many records with the same distance model - reusing
/// the aligner avoids repeated allocation overhead.
///
/// # Arguments
/// * `tp` - The tracepoint data to reconstruct from
/// * `query_seq` - Query sequence bytes
/// * `target_seq` - Target sequence bytes
/// * `query_offset` - Offset in query sequence (for FastGA)
/// * `target_offset` - Offset in target sequence (for FastGA)
/// * `metric` - Complexity metric (EditDistance or DiagonalDistance)
/// * `spacing` - Trace spacing (only used for FastGA tracepoints)
/// * `complement` - Whether the alignment is on reverse complement strand
/// * `aligner` - Mutable reference to a WFA2 aligner
/// * `max_value` - Optional max complexity value for banded alignment heuristic
///
pub fn reconstruct_cigar_with_aligner(
    tp: &TracepointData,
    query_seq: &[u8],
    target_seq: &[u8],
    query_offset: usize,
    target_offset: usize,
    metric: ComplexityMetric,
    spacing: u32,
    complement: bool,
    aligner: &mut AffineWavefronts,
    max_value: Option<u32>,
) -> String {
    reconstruct_cigar_with_aligner_impl(
        tp,
        query_seq,
        target_seq,
        query_offset,
        target_offset,
        metric,
        spacing,
        complement,
        aligner,
        max_value,
    )
}

/// Internal implementation for aligner-based reconstruction
fn reconstruct_cigar_with_aligner_impl(
    tp: &TracepointData,
    query_seq: &[u8],
    target_seq: &[u8],
    query_offset: usize,
    target_offset: usize,
    metric: ComplexityMetric,
    spacing: u32,
    complement: bool,
    aligner: &mut AffineWavefronts,
    max_value: Option<u32>,
) -> String {
    match tp {
        TracepointData::Standard(tps) => tracepoints_to_cigar_with_aligner(
            tps, query_seq, target_seq, 0, 0, metric, aligner, max_value,
        ),
        TracepointData::Mixed(items) => mixed_tracepoints_to_cigar_with_aligner(
            items, query_seq, target_seq, 0, 0, metric, aligner, max_value,
        ),
        TracepointData::Variable(tps) => variable_tracepoints_to_cigar_with_aligner(
            tps, query_seq, target_seq, 0, 0, metric, aligner, max_value,
        ),
        TracepointData::Fastga(tps) => tracepoints_to_cigar_fastga_with_aligner(
            tps,
            spacing,
            query_seq,
            target_seq,
            query_offset,
            target_offset,
            complement,
            aligner,
            max_value.is_some(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cigar_stats_basic() {
        let stats = CigarStats::from_cigar("10=5X3I2D");
        assert_eq!(stats.matches, 10);
        assert_eq!(stats.mismatches, 5);
        assert_eq!(stats.insertions(), 1);
        assert_eq!(stats.inserted_bp(), 3);
        assert_eq!(stats.deletions(), 1);
        assert_eq!(stats.deleted_bp(), 2);
    }

    #[test]
    fn test_gap_compressed_identity() {
        // 10 matches, 0 mismatches, 1 ins, 1 del
        // Identity = 10 / (10 + 0 + 1 + 1) = 10/12 = 0.833...
        let stats = CigarStats::from_cigar("10=3I2D");
        let identity = stats.gap_compressed_identity();
        assert!((identity - 0.833).abs() < 0.01);
    }

    #[test]
    fn test_block_identity() {
        // 10 matches, edit_distance = 0 + 3 + 2 = 5
        // Identity = 10 / (10 + 5) = 10/15 = 0.666...
        let stats = CigarStats::from_cigar("10=3I2D");
        let identity = stats.block_identity();
        assert!((identity - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_alignment_score_edit() {
        let stats = CigarStats::from_cigar("10=2X3I2D");
        let score = stats.alignment_score(&Distance::Edit);
        // Score = -(2 + 3 + 2) = -7
        assert_eq!(score, -7);
    }
}
