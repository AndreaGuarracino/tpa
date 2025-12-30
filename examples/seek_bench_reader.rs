//! Seek benchmark using TpaReader with index (high-level API)
//!
//! Measures random access performance using TpaReader::get_tracepoints().
//! Validates decoded tracepoints against a reference PAF file.
//!
//! Usage: seek_bench_reader <file.tpa> <num_records> <num_positions> <iterations> <tp_type> <reference.paf>
//!
//! Output: avg_us stddev_us decode_ratio valid_ratio
//!   - decode_ratio: fraction of get_tracepoints() calls that succeeded
//!   - valid_ratio: fraction of decoded tracepoints that matched reference

use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use tpa::{MixedRepresentation, TpaReader, TracepointData};

/// Reference tracepoints parsed from PAF file
enum Reference {
    Standard(Vec<Vec<(usize, usize)>>),
    Variable(Vec<Vec<(usize, Option<usize>)>>),
    Mixed(Vec<Vec<MixedRepresentation>>),
}

/// Parse tracepoints from a PAF file for validation
fn parse_reference(path: &str, limit: usize, tp_type: &str) -> Reference {
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
                                // Tracepoint: "first,second"
                                let mut it = part.split(',');
                                let a: usize = it.next().unwrap().parse().unwrap();
                                let b: usize = it.next().unwrap().parse().unwrap();
                                MixedRepresentation::Tracepoint(a, b)
                            } else {
                                // CIGAR op: "lenOP" like "10M"
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

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 {
        eprintln!(
            "Usage: {} <file.tpa> <num_records> <num_positions> <iterations> <tp_type> <reference.paf>",
            args[0]
        );
        eprintln!("\ntp_type: standard, fastga, variable, or mixed");
        eprintln!("Output:  avg_us stddev_us decode_ratio valid_ratio");
        std::process::exit(1);
    }

    let tpa_path = &args[1];
    let num_records: u64 = args[2].parse().unwrap();
    let num_positions: usize = args[3].parse().unwrap();
    let iterations_per_pos: usize = args[4].parse().unwrap();
    let tp_type = &args[5];
    let reference_paf = &args[6];

    let mut reader = TpaReader::new(tpa_path).unwrap();
    let reference = parse_reference(reference_paf, num_records as usize, tp_type);

    // Generate deterministic pseudo-random positions
    let mut rng = 12345u64;
    let mut positions = HashSet::new();
    while positions.len() < num_positions {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        positions.insert((rng % num_records) as u64);
    }
    let positions: Vec<u64> = positions.into_iter().collect();

    let mut sum_us = 0u128;
    let mut sum_sq_us = 0u128;
    let mut decode_count = 0usize;
    let mut valid_count = 0usize;
    let total_tests = num_positions * iterations_per_pos;

    for &pos in &positions {
        // Warmup (3 iterations)
        for _ in 0..3 {
            let _ = reader.get_tracepoints(pos);
        }

        // Benchmark
        for _ in 0..iterations_per_pos {
            let start = Instant::now();
            match reader.get_tracepoints(pos) {
                Ok((tp, _, _)) => {
                    let time_us = start.elapsed().as_micros();
                    sum_us += time_us;
                    sum_sq_us += time_us * time_us;
                    decode_count += 1;

                    // Validate against reference
                    let is_valid = match (&tp, &reference) {
                        (TracepointData::Standard(tps), Reference::Standard(refs))
                        | (TracepointData::Fastga(tps), Reference::Standard(refs)) => refs
                            .get(pos as usize)
                            .map(|expected| expected.as_slice() == tps.as_slice())
                            .unwrap_or(false),
                        (TracepointData::Variable(tps), Reference::Variable(refs)) => refs
                            .get(pos as usize)
                            .map(|expected| expected.as_slice() == tps.as_slice())
                            .unwrap_or(false),
                        (TracepointData::Mixed(items), Reference::Mixed(refs)) => refs
                            .get(pos as usize)
                            .map(|expected| expected.as_slice() == items.as_slice())
                            .unwrap_or(false),
                        _ => false, // Type mismatch
                    };

                    if is_valid {
                        valid_count += 1;
                    }
                }
                Err(_) => {}
            }
        }
    }

    let avg_us = sum_us as f64 / total_tests as f64;
    let variance = (sum_sq_us as f64 / total_tests as f64) - (avg_us * avg_us);
    let stddev_us = variance.sqrt();
    let decode_ratio = decode_count as f64 / total_tests as f64;
    let valid_ratio = valid_count as f64 / total_tests as f64;

    println!(
        "{:.2} {:.2} {:.4} {:.4}",
        avg_us, stddev_us, decode_ratio, valid_ratio
    );
}
