//! Shared utilities for TPA benchmark examples
//!
//! This module contains common functionality used across multiple benchmark examples
//! to avoid code duplication.

use std::fs::File;
use std::io::{BufRead, BufReader};
use tpa::{MixedRepresentation, TracepointData};

/// Reference tracepoints parsed from PAF file for validation
pub enum Reference {
    Standard(Vec<Vec<(usize, usize)>>),
    Variable(Vec<Vec<(usize, Option<usize>)>>),
    Mixed(Vec<Vec<MixedRepresentation>>),
}

/// Parse tracepoints from a PAF file for validation
///
/// # Arguments
/// * `path` - Path to the PAF file
/// * `limit` - Maximum number of records to parse
/// * `tp_type` - Tracepoint type: "standard", "fastga", "variable", or "mixed"
pub fn parse_reference(path: &str, limit: usize, tp_type: &str) -> Reference {
    let file = File::open(path).expect("reference PAF open failed");
    let reader = BufReader::new(file);

    match tp_type {
        "variable" => {
            let mut refs = Vec::new();
            for line in reader.lines().take(limit) {
                let line = line.expect("line read");
                if let Some(tp_idx) = line.find("tp:Z:") {
                    let tp_str_full = &line[tp_idx + 5..];
                    let tp_str = tp_str_full.split('\t').next().unwrap_or(tp_str_full);
                    let tps: Vec<(usize, Option<usize>)> = tp_str
                        .split(';')
                        .filter(|s| !s.is_empty())
                        .map(|pair| {
                            let mut it = pair.split(',');
                            let a: usize = it.next().unwrap().parse().unwrap();
                            let b: Option<usize> = it.next().map(|s| s.parse().unwrap());
                            (a, b)
                        })
                        .collect();
                    refs.push(tps);
                }
            }
            Reference::Variable(refs)
        }
        "mixed" => {
            let mut refs = Vec::new();
            for line in reader.lines().take(limit) {
                let line = line.expect("line read");
                if let Some(tp_idx) = line.find("tp:Z:") {
                    let tp_str_full = &line[tp_idx + 5..];
                    let tp_str = tp_str_full.split('\t').next().unwrap_or(tp_str_full);
                    let items: Vec<MixedRepresentation> = tp_str
                        .split(';')
                        .filter(|s| !s.is_empty())
                        .map(|part| {
                            if part.contains(',') {
                                let mut it = part.split(',');
                                let a: usize = it.next().unwrap().parse().unwrap();
                                let b: usize = it.next().unwrap().parse().unwrap();
                                MixedRepresentation::Tracepoint(a, b)
                            } else {
                                let op = part.chars().last().unwrap();
                                let len: usize = part[..part.len() - 1].parse().unwrap();
                                MixedRepresentation::CigarOp(len, op)
                            }
                        })
                        .collect();
                    refs.push(items);
                }
            }
            Reference::Mixed(refs)
        }
        _ => {
            // standard - parse as pairs
            let mut refs = Vec::new();
            for line in reader.lines().take(limit) {
                let line = line.expect("line read");
                if let Some(tp_idx) = line.find("tp:Z:") {
                    let tp_str_full = &line[tp_idx + 5..];
                    let tp_str = tp_str_full.split('\t').next().unwrap_or(tp_str_full);
                    let tps: Vec<(usize, usize)> = tp_str
                        .split(';')
                        .filter(|s| !s.is_empty())
                        .map(|pair| {
                            let mut it = pair.split(',');
                            let a: usize = it.next().unwrap().parse().unwrap();
                            let b: usize = it.next().unwrap().parse().unwrap();
                            (a, b)
                        })
                        .collect();
                    refs.push(tps);
                }
            }
            Reference::Standard(refs)
        }
    }
}

/// Validate decoded tracepoints against reference data
///
/// Returns true if the decoded tracepoints match the reference at the given position
pub fn validate_tracepoints(tp: &TracepointData, reference: &Reference, pos: usize) -> bool {
    match (tp, reference) {
        (TracepointData::Standard(tps), Reference::Standard(refs))
        | (TracepointData::Fastga(tps), Reference::Standard(refs)) => refs
            .get(pos)
            .map(|expected| expected.as_slice() == tps.as_slice())
            .unwrap_or(false),
        (TracepointData::Variable(tps), Reference::Variable(refs)) => refs
            .get(pos)
            .map(|expected| expected.as_slice() == tps.as_slice())
            .unwrap_or(false),
        (TracepointData::Mixed(items), Reference::Mixed(refs)) => refs
            .get(pos)
            .map(|expected| expected.as_slice() == items.as_slice())
            .unwrap_or(false),
        _ => false, // Type mismatch
    }
}
