//! Seek benchmark for BGZIP-compressed tracepoint PAF files
//!
//! Measures random access performance on bgzipped PAF files using BGZF virtual positions.
//! This provides a baseline comparison against TPA format seek performance.
//!
//! How it works:
//! 1. Parses the bgzipped PAF file once, capturing virtual positions for tp:Z: fields
//! 2. Randomly seeks to those positions and reads/parses tracepoints
//! 3. Validates decoded tracepoints against reference
//!
//! Usage: seek_bench_bgzip_paf <file.tp.paf.gz> <num_records> <num_positions> <iterations> <tp_type> <reference.paf>
//!
//! Output: avg_us stddev_us decode_ratio valid_ratio
//!   - decode_ratio: fraction of seeks that successfully parsed tracepoints
//!   - valid_ratio: fraction of decoded tracepoints that matched reference

use noodles::bgzf;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::time::Instant;

/// Mixed representation item (tracepoint pair or CIGAR op)
#[derive(Debug, Clone, PartialEq)]
enum MixedItem {
    Tracepoint(usize, usize),
    CigarOp(usize, char),
}

/// Reference tracepoints parsed from PAF file
enum Reference {
    Standard(Vec<Vec<(usize, usize)>>),
    Variable(Vec<Vec<(usize, Option<usize>)>>),
    Mixed(Vec<Vec<MixedItem>>),
}

/// Parsed tracepoints with virtual position for seeking
struct TracepointRecord {
    /// BGZF virtual position of the tp:Z: field value (after "tp:Z:")
    virtual_position: u64,
    /// Length of the tracepoint string (for reading)
    tp_length: usize,
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
                    let tp_str = &line[tp_idx + 5..];
                    // Find end of tp:Z: field (next tab or end of line)
                    let tp_end = tp_str.find('\t').unwrap_or(tp_str.len());
                    let tp_str = &tp_str[..tp_end];
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
                    let tp_str = &line[tp_idx + 5..];
                    // Find end of tp:Z: field (next tab or end of line)
                    let tp_end = tp_str.find('\t').unwrap_or(tp_str.len());
                    let tp_str = &tp_str[..tp_end];
                    let items: Vec<MixedItem> = tp_str
                        .split(';')
                        .filter(|s| !s.is_empty())
                        .map(|part| {
                            if part.contains(',') {
                                // Tracepoint: "first,second"
                                let mut it = part.split(',');
                                let a: usize = it.next().unwrap().parse().unwrap();
                                let b: usize = it.next().unwrap().parse().unwrap();
                                MixedItem::Tracepoint(a, b)
                            } else {
                                // CIGAR op: "lenOP" like "10M"
                                let op = part.chars().last().unwrap();
                                let len: usize = part[..part.len() - 1].parse().unwrap();
                                MixedItem::CigarOp(len, op)
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
                    let tp_str = &line[tp_idx + 5..];
                    // Find end of tp:Z: field (next tab or end of line)
                    let tp_end = tp_str.find('\t').unwrap_or(tp_str.len());
                    let tp_str = &tp_str[..tp_end];
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

/// Parse bgzipped PAF to extract virtual positions for tp:Z: fields
fn build_virtual_position_index(bgzip_path: &str, limit: usize) -> Vec<TracepointRecord> {
    let file = File::open(bgzip_path).expect("Failed to open bgzip file");
    let mut reader = bgzf::io::Reader::new(file);

    let mut records = Vec::new();
    let mut line_bytes = Vec::new();

    for _ in 0..limit {
        // Get virtual position at start of line
        let line_start_vpos = reader.virtual_position();
        line_bytes.clear();

        let bytes_read = reader.read_until(b'\n', &mut line_bytes).expect("read line");
        if bytes_read == 0 {
            break; // EOF
        }

        // Convert to string for parsing
        let line_len = if line_bytes.ends_with(b"\n") {
            line_bytes.len() - 1
        } else {
            line_bytes.len()
        };
        let line = std::str::from_utf8(&line_bytes[..line_len]).expect("valid UTF-8");

        if line.is_empty() {
            continue;
        }

        // Find tp:Z: field position within the line
        if let Some(tp_idx) = line.find("tp:Z:") {
            let tp_start_in_line = tp_idx + 5; // Skip "tp:Z:"
            let tp_str = &line[tp_start_in_line..];
            let tp_end = tp_str.find('\t').unwrap_or(tp_str.len());

            // Seek back to line start, then advance to tp:Z: value position
            reader.seek(line_start_vpos).expect("seek to line start");

            if tp_start_in_line > 0 {
                std::io::copy(
                    &mut reader.by_ref().take(tp_start_in_line as u64),
                    &mut std::io::sink(),
                )
                .expect("advance to tp:Z:");
            }

            let tp_vpos = reader.virtual_position();

            records.push(TracepointRecord {
                virtual_position: u64::from(tp_vpos),
                tp_length: tp_end,
            });

            // Skip to end of line for next iteration
            let remaining = line_bytes.len() - tp_start_in_line;
            if remaining > 0 {
                std::io::copy(
                    &mut reader.by_ref().take(remaining as u64),
                    &mut std::io::sink(),
                )
                .expect("skip to line end");
            }
        }
    }

    records
}

/// Parse tracepoints from a string (tp:Z: field value)
fn parse_tracepoints_standard(tp_str: &str) -> Vec<(usize, usize)> {
    tp_str
        .split(';')
        .filter(|s| !s.is_empty())
        .filter_map(|pair| {
            let mut it = pair.split(',');
            let a: usize = it.next()?.parse().ok()?;
            let b: usize = it.next()?.parse().ok()?;
            Some((a, b))
        })
        .collect()
}

fn parse_tracepoints_variable(tp_str: &str) -> Vec<(usize, Option<usize>)> {
    tp_str
        .split(';')
        .filter(|s| !s.is_empty())
        .filter_map(|pair| {
            let mut it = pair.split(',');
            let a: usize = it.next()?.parse().ok()?;
            let b: Option<usize> = it.next().and_then(|s| s.parse().ok());
            Some((a, b))
        })
        .collect()
}

fn parse_tracepoints_mixed(tp_str: &str) -> Vec<MixedItem> {
    tp_str
        .split(';')
        .filter(|s| !s.is_empty())
        .filter_map(|part| {
            if part.contains(',') {
                // Tracepoint: "first,second"
                let mut it = part.split(',');
                let a: usize = it.next()?.parse().ok()?;
                let b: usize = it.next()?.parse().ok()?;
                Some(MixedItem::Tracepoint(a, b))
            } else {
                // CIGAR op: "lenOP" like "10M"
                let op = part.chars().last()?;
                let len: usize = part[..part.len() - 1].parse().ok()?;
                Some(MixedItem::CigarOp(len, op))
            }
        })
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 7 {
        eprintln!(
            "Usage: {} <file.tp.paf.gz> <num_records> <num_positions> <iterations> <tp_type> <reference.paf>",
            args[0]
        );
        eprintln!("\ntp_type: standard, variable, or mixed");
        eprintln!("Output:  avg_us stddev_us decode_ratio valid_ratio");
        std::process::exit(1);
    }

    let bgzip_path = &args[1];
    let num_records: usize = args[2].parse().unwrap();
    let num_positions: usize = args[3].parse().unwrap();
    let iterations_per_pos: usize = args[4].parse().unwrap();
    let tp_type = &args[5];
    let reference_paf = &args[6];

    // Build index of virtual positions
    let vpos_index = build_virtual_position_index(bgzip_path, num_records);
    let actual_records = vpos_index.len();

    if actual_records == 0 {
        eprintln!("No tracepoint records found in {}", bgzip_path);
        println!("0 0 0 0");
        return;
    }

    // Parse reference for validation
    let reference = parse_reference(reference_paf, num_records, tp_type);

    // Generate deterministic pseudo-random positions
    let mut rng = 12345u64;
    let mut positions = HashSet::new();
    let target_positions = num_positions.min(actual_records);
    while positions.len() < target_positions {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        positions.insert((rng as usize % actual_records) as usize);
    }
    let positions: Vec<usize> = positions.into_iter().collect();

    // Open bgzip reader for seeking
    let file = File::open(bgzip_path).expect("open bgzip file");
    let mut reader = bgzf::io::Reader::new(file);

    let mut sum_us = 0u128;
    let mut sum_sq_us = 0u128;
    let mut decode_count = 0usize;
    let mut valid_count = 0usize;
    let total_tests = positions.len() * iterations_per_pos;

    let mut buffer = Vec::new();

    for &pos in &positions {
        let record = &vpos_index[pos];
        let vpos = bgzf::VirtualPosition::from(record.virtual_position);

        // Warmup (3 iterations)
        for _ in 0..3 {
            buffer.clear();
            buffer.resize(record.tp_length, 0u8);
            let _ = reader.seek(vpos);
            let _ = reader.read_exact(&mut buffer);
        }

        // Benchmark
        for _ in 0..iterations_per_pos {
            buffer.clear();
            buffer.resize(record.tp_length, 0u8);

            let start = Instant::now();
            if reader.seek(vpos).is_ok() && reader.read_exact(&mut buffer).is_ok() {
                let time_us = start.elapsed().as_micros();
                sum_us += time_us;
                sum_sq_us += time_us * time_us;
                decode_count += 1;

                // Parse and validate
                if let Ok(tp_str) = std::str::from_utf8(&buffer) {
                    let is_valid = match tp_type.as_str() {
                        "variable" => {
                            let parsed = parse_tracepoints_variable(tp_str);
                            if let Reference::Variable(refs) = &reference {
                                refs.get(pos)
                                    .map(|expected| expected.as_slice() == parsed.as_slice())
                                    .unwrap_or(false)
                            } else {
                                false
                            }
                        }
                        "mixed" => {
                            let parsed = parse_tracepoints_mixed(tp_str);
                            if let Reference::Mixed(refs) = &reference {
                                refs.get(pos)
                                    .map(|expected| expected.as_slice() == parsed.as_slice())
                                    .unwrap_or(false)
                            } else {
                                false
                            }
                        }
                        _ => {
                            let parsed = parse_tracepoints_standard(tp_str);
                            if let Reference::Standard(refs) = &reference {
                                refs.get(pos)
                                    .map(|expected| expected.as_slice() == parsed.as_slice())
                                    .unwrap_or(false)
                            } else {
                                false
                            }
                        }
                    };

                    if is_valid {
                        valid_count += 1;
                    }
                }
            }
        }
    }

    let avg_us = if total_tests > 0 {
        sum_us as f64 / total_tests as f64
    } else {
        0.0
    };
    let variance = if total_tests > 0 {
        (sum_sq_us as f64 / total_tests as f64) - (avg_us * avg_us)
    } else {
        0.0
    };
    let stddev_us = variance.max(0.0).sqrt();
    let decode_ratio = if total_tests > 0 {
        decode_count as f64 / total_tests as f64
    } else {
        0.0
    };
    let valid_ratio = if total_tests > 0 {
        valid_count as f64 / total_tests as f64
    } else {
        0.0
    };

    println!(
        "{:.2} {:.2} {:.4} {:.4}",
        avg_us, stddev_us, decode_ratio, valid_ratio
    );
}
