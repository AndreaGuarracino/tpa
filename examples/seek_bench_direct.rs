//! Seek benchmark using standalone functions (low-level API, maximum performance)
//!
//! Measures random access performance using read_*_tracepoints_at_offset() functions.
//! Pre-computes offsets from index, then seeks directly to tracepoint data.
//! This is the fastest access method when you only need tracepoints.
//!
//! Supports both per-record mode (raw offsets) and all-records mode (virtual positions).
//!
//! Usage: seek_bench_direct <file.tpa> <num_records> <num_positions> <iterations> <tp_type> <reference.paf>
//!
//! Output: avg_us stddev_us decode_ratio valid_ratio
//!   - decode_ratio: fraction of read_*_tracepoints_at_offset() calls that succeeded
//!   - valid_ratio: fraction of decoded tracepoints that matched reference

use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use tpa::{
    bgzf, read_mixed_tracepoints_at_offset, read_mixed_tracepoints_at_vpos,
    read_standard_tracepoints_at_offset_with_strategies, read_standard_tracepoints_at_vpos,
    read_variable_tracepoints_at_offset, read_variable_tracepoints_at_vpos, MixedRepresentation,
    TpaReader,
};

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
        eprintln!("\ntp_type: standard, variable, or mixed");
        eprintln!("Output:  avg_us stddev_us decode_ratio valid_ratio");
        std::process::exit(1);
    }

    let tpa_path = &args[1];
    let num_records: u64 = args[2].parse().unwrap();
    let num_positions: usize = args[3].parse().unwrap();
    let iterations_per_pos: usize = args[4].parse().unwrap();
    let tp_type = &args[5];
    let reference_paf = &args[6];

    // Open TpaReader to get strategies, layers, offsets, and detect mode
    let mut reader = TpaReader::new(tpa_path).unwrap();
    let (first_strategy, second_strategy) = reader.header().strategies().unwrap();
    let first_layer = reader.header().first_layer();
    let second_layer = reader.header().second_layer();
    let is_bgzip = reader.is_bgzip_mode();
    let _bgzf_section_start = reader.bgzf_section_start();
    let reference = parse_reference(reference_paf, num_records as usize, tp_type);

    // Generate deterministic pseudo-random positions
    let mut rng = 12345u64;
    let mut positions = HashSet::new();
    while positions.len() < num_positions {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        positions.insert((rng % num_records) as u64);
    }

    // Pre-compute byte offsets from index (one-time cost)
    // For all-records mode these are virtual positions, for per-record mode these are raw offsets
    let offsets: Vec<u64> = positions
        .iter()
        .map(|&pos| reader.get_tracepoint_offset(pos).unwrap())
        .collect();

    // Drop reader - from here we use only standalone functions
    drop(reader);

    let mut sum_us = 0u128;
    let mut sum_sq_us = 0u128;
    let mut decode_count = 0usize;
    let mut valid_count = 0usize;
    let total_tests = num_positions * iterations_per_pos;

    if is_bgzip {
        // All-records mode: use bgzf reader and _at_vpos functions
        // We use absolute virtual positions, so open file from start
        let file = File::open(tpa_path).unwrap();
        let mut bgzf_reader = bgzf::io::Reader::new(file);

        for (&vpos, &record_id) in offsets.iter().zip(positions.iter()) {
            // Warmup (3 iterations)
            for _ in 0..3 {
                match tp_type.as_str() {
                    "standard" => {
                        let _ = read_standard_tracepoints_at_vpos(
                            &mut bgzf_reader,
                            vpos,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        );
                    }
                    "variable" => {
                        let _ = read_variable_tracepoints_at_vpos(
                            &mut bgzf_reader,
                            vpos,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        );
                    }
                    "mixed" => {
                        let _ = read_mixed_tracepoints_at_vpos(
                            &mut bgzf_reader,
                            vpos,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        );
                    }
                    _ => panic!("Invalid tp_type: {}", tp_type),
                }
            }

            // Benchmark
            for _ in 0..iterations_per_pos {
                let start = Instant::now();
                let (decoded, is_valid) = match tp_type.as_str() {
                    "standard" => {
                        match read_standard_tracepoints_at_vpos(
                            &mut bgzf_reader,
                            vpos,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        ) {
                            Ok(tps) => {
                                let valid = if let Reference::Standard(refs) = &reference {
                                    refs.get(record_id as usize)
                                        .map(|expected| expected.as_slice() == tps.as_slice())
                                        .unwrap_or(false)
                                } else {
                                    false
                                };
                                (true, valid)
                            }
                            Err(_) => (false, false),
                        }
                    }
                    "variable" => {
                        match read_variable_tracepoints_at_vpos(
                            &mut bgzf_reader,
                            vpos,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        ) {
                            Ok(tps) => {
                                let valid = if let Reference::Variable(refs) = &reference {
                                    refs.get(record_id as usize)
                                        .map(|expected| expected.as_slice() == tps.as_slice())
                                        .unwrap_or(false)
                                } else {
                                    false
                                };
                                (true, valid)
                            }
                            Err(_) => (false, false),
                        }
                    }
                    "mixed" => {
                        match read_mixed_tracepoints_at_vpos(
                            &mut bgzf_reader,
                            vpos,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        ) {
                            Ok(items) => {
                                let valid = if let Reference::Mixed(refs) = &reference {
                                    refs.get(record_id as usize)
                                        .map(|expected| expected.as_slice() == items.as_slice())
                                        .unwrap_or(false)
                                } else {
                                    false
                                };
                                (true, valid)
                            }
                            Err(_) => (false, false),
                        }
                    }
                    _ => panic!("Invalid tp_type: {}", tp_type),
                };

                if decoded {
                    let time_us = start.elapsed().as_micros();
                    sum_us += time_us;
                    sum_sq_us += time_us * time_us;
                    decode_count += 1;

                    if is_valid {
                        valid_count += 1;
                    }
                }
            }
        }
    } else {
        // Classic mode: use raw file handle and _at_offset functions
        let mut file = File::open(tpa_path).unwrap();

        for (&offset, &record_id) in offsets.iter().zip(positions.iter()) {
            // Warmup (3 iterations)
            for _ in 0..3 {
                match tp_type.as_str() {
                    "standard" => {
                        let _ = read_standard_tracepoints_at_offset_with_strategies(
                            &mut file,
                            offset,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        );
                    }
                    "variable" => {
                        let _ = read_variable_tracepoints_at_offset(
                            &mut file,
                            offset,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        );
                    }
                    "mixed" => {
                        let _ = read_mixed_tracepoints_at_offset(
                            &mut file,
                            offset,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        );
                    }
                    _ => panic!("Invalid tp_type: {}", tp_type),
                }
            }

            // Benchmark
            for _ in 0..iterations_per_pos {
                let start = Instant::now();
                let (decoded, is_valid) = match tp_type.as_str() {
                    "standard" => {
                        match read_standard_tracepoints_at_offset_with_strategies(
                            &mut file,
                            offset,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        ) {
                            Ok(tps) => {
                                let valid = if let Reference::Standard(refs) = &reference {
                                    refs.get(record_id as usize)
                                        .map(|expected| expected.as_slice() == tps.as_slice())
                                        .unwrap_or(false)
                                } else {
                                    false
                                };
                                (true, valid)
                            }
                            Err(_) => (false, false),
                        }
                    }
                    "variable" => {
                        match read_variable_tracepoints_at_offset(
                            &mut file,
                            offset,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        ) {
                            Ok(tps) => {
                                let valid = if let Reference::Variable(refs) = &reference {
                                    refs.get(record_id as usize)
                                        .map(|expected| expected.as_slice() == tps.as_slice())
                                        .unwrap_or(false)
                                } else {
                                    false
                                };
                                (true, valid)
                            }
                            Err(_) => (false, false),
                        }
                    }
                    "mixed" => {
                        match read_mixed_tracepoints_at_offset(
                            &mut file,
                            offset,
                            first_strategy.clone(),
                            second_strategy.clone(),
                            first_layer,
                            second_layer,
                        ) {
                            Ok(items) => {
                                let valid = if let Reference::Mixed(refs) = &reference {
                                    refs.get(record_id as usize)
                                        .map(|expected| expected.as_slice() == items.as_slice())
                                        .unwrap_or(false)
                                } else {
                                    false
                                };
                                (true, valid)
                            }
                            Err(_) => (false, false),
                        }
                    }
                    _ => panic!("Invalid tp_type: {}", tp_type),
                };

                if decoded {
                    let time_us = start.elapsed().as_micros();
                    sum_us += time_us;
                    sum_sq_us += time_us * time_us;
                    decode_count += 1;

                    if is_valid {
                        valid_count += 1;
                    }
                }
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
