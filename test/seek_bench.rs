/// Unified seek benchmark tool for PAF files
///
/// Auto-detects and benchmarks:
/// - CIGAR PAF files (.paf.gz with cg:Z: tags)
/// - Tracepoint PAF files (.paf.gz with tp:Z: tags)
/// - BPAF files (.bpaf)
///
/// Usage: seek_bench <file> <num_records> <num_positions> <iterations>
///
/// Examples:
///   seek_bench alignments.paf.gz 10000 100 100     # Auto-detect CIGAR or tracepoint
///   seek_bench alignments.bpaf 10000 100 100       # BPAF format
use std::env;
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::time::Instant;

// For bgzipped PAF files
use noodles::bgzf::io::Reader as BgzfReader;
use noodles::bgzf::VirtualPosition;

// For BPAF files
use lib_bpaf::BpafReader;

enum FileFormat {
    CigarPafBgz,
    TracepointPafBgz,
    CigarPafPlain,
    TracepointPafPlain,
    Bpaf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 5 {
        eprintln!(
            "Usage: {} <file> <num_records> <num_positions> <iterations>",
            args[0]
        );
        eprintln!();
        eprintln!("Auto-detects format:");
        eprintln!("  *.bpaf           - BPAF binary format");
        eprintln!("  *.paf.gz cg:Z:   - Bgzipped CIGAR PAF");
        eprintln!("  *.paf.gz tp:Z:   - Bgzipped tracepoint PAF");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} alignments.bpaf 10000 100 100", args[0]);
        std::process::exit(1);
    }

    let file_path = &args[1];
    let num_records: usize = args[2]
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid num_records"))?;
    let num_positions: usize = args[3]
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid num_positions"))?;
    let iterations: usize = args[4]
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid iterations"))?;

    let format = detect_format(file_path)?;

    match format {
        FileFormat::Bpaf => {
            eprintln!("Format: BPAF");
            benchmark_bpaf(file_path, num_records, num_positions, iterations)?;
        }
        FileFormat::CigarPafBgz => {
            eprintln!("Format: CIGAR PAF (bgzipped)");
            benchmark_cigar_paf_bgz(file_path, num_records, num_positions, iterations)?;
        }
        FileFormat::TracepointPafBgz => {
            eprintln!("Format: Tracepoint PAF (bgzipped)");
            benchmark_tracepoint_paf_bgz(file_path, num_records, num_positions, iterations)?;
        }
        FileFormat::CigarPafPlain => {
            eprintln!("Format: CIGAR PAF (plain)");
            benchmark_cigar_paf_plain(file_path, num_records, num_positions, iterations)?;
        }
        FileFormat::TracepointPafPlain => {
            eprintln!("Format: Tracepoint PAF (plain)");
            benchmark_tracepoint_paf_plain(file_path, num_records, num_positions, iterations)?;
        }
    }

    Ok(())
}

fn detect_format(path: &str) -> Result<FileFormat, Box<dyn std::error::Error>> {
    // Check extension
    if path.ends_with(".bpaf") {
        return Ok(FileFormat::Bpaf);
    }

    if path.ends_with(".paf.gz") || path.ends_with(".paf") {
        let mut line = Vec::new();
        let is_bgz = path.ends_with(".paf.gz");

        if is_bgz {
            // Read first line to detect tags
            let file = File::open(path)?;
            let mut reader = BgzfReader::new(file);
            reader.read_until(b'\n', &mut line)?;
        } else {
            let file = File::open(path)?;
            let mut reader = io::BufReader::new(file);
            reader.read_until(b'\n', &mut line)?;
        }

        if line.windows(5).any(|w| w == b"cg:Z:") {
            return Ok(if is_bgz {
                FileFormat::CigarPafBgz
            } else {
                FileFormat::CigarPafPlain
            });
        } else if line.windows(5).any(|w| w == b"tp:Z:") {
            return Ok(if is_bgz {
                FileFormat::TracepointPafBgz
            } else {
                FileFormat::TracepointPafPlain
            });
        }

        return Err("PAF file has neither cg:Z: nor tp:Z: tags".into());
    }

    Err(format!("Unknown file format: {}", path).into())
}

fn benchmark_bpaf(
    path: &str,
    num_records: usize,
    num_positions: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = BpafReader::open(path)?;
    let total_records = reader.header().num_records() as usize;
    let test_records = if num_records == 0 {
        total_records
    } else {
        num_records.min(total_records)
    };

    if test_records == 0 {
        println!("No records to benchmark");
        return Ok(());
    }

    // Generate random positions
    use std::collections::HashSet;
    let mut rng = 12345u64;
    let mut positions = HashSet::new();
    while positions.len() < num_positions.min(test_records) {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        positions.insert((rng % test_records as u64) as u64);
    }
    let positions: Vec<u64> = positions.into_iter().collect();

    // Benchmark
    let mut times = Vec::new();
    for &pos in &positions {
        // Warmup
        for _ in 0..3 {
            let _ = reader.get_tracepoints(pos);
        }

        // Measure
        for _ in 0..iterations {
            let start = Instant::now();
            match reader.get_tracepoints(pos) {
                Ok(_) => times.push(start.elapsed().as_micros() as f64),
                Err(e) => eprintln!("Seek error at position {}: {}", pos, e),
            }
        }
    }

    print_stats("BPAF", &times);
    Ok(())
}

fn benchmark_cigar_paf_bgz(
    path: &str,
    num_records: usize,
    num_positions: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build index
    let entries = build_cigar_index_bgz(path, num_records)?;
    if entries.is_empty() {
        println!("No CIGAR entries to benchmark");
        return Ok(());
    }

    eprintln!("Indexed {} CIGAR entries", entries.len());

    // Select positions to test
    let test_positions: Vec<usize> = (0..entries.len())
        .step_by(entries.len() / num_positions.min(entries.len()).max(1))
        .take(num_positions)
        .collect();

    // Benchmark
    let file = File::open(path)?;
    let mut reader = BgzfReader::new(file);
    let mut times = Vec::new();

    for &idx in &test_positions {
        let entry = &entries[idx];
        let mut buffer = vec![0u8; entry.len];

        // Warmup
        for _ in 0..3 {
            reader.seek(entry.pos)?;
            reader.read_exact(&mut buffer)?;
        }

        // Measure
        for _ in 0..iterations {
            let start = Instant::now();
            reader.seek(entry.pos)?;
            reader.read_exact(&mut buffer)?;
            times.push(start.elapsed().as_micros() as f64);
        }
    }

    print_stats("CIGAR PAF", &times);
    Ok(())
}

fn benchmark_tracepoint_paf_bgz(
    path: &str,
    num_records: usize,
    num_positions: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build index
    let entries = build_tracepoint_index_bgz(path, num_records)?;
    if entries.is_empty() {
        println!("No tracepoint entries to benchmark");
        return Ok(());
    }

    eprintln!("Indexed {} tracepoint entries", entries.len());

    // Select positions to test
    let test_positions: Vec<usize> = (0..entries.len())
        .step_by(entries.len() / num_positions.min(entries.len()).max(1))
        .take(num_positions)
        .collect();

    // Benchmark
    let file = File::open(path)?;
    let mut reader = BgzfReader::new(file);
    let mut times = Vec::new();

    for &idx in &test_positions {
        let entry = &entries[idx];
        let mut buffer = vec![0u8; entry.len];

        // Warmup
        for _ in 0..3 {
            reader.seek(entry.pos)?;
            reader.read_exact(&mut buffer)?;
        }

        // Measure
        for _ in 0..iterations {
            let start = Instant::now();
            reader.seek(entry.pos)?;
            reader.read_exact(&mut buffer)?;
            times.push(start.elapsed().as_micros() as f64);
        }
    }

    print_stats("Tracepoint PAF", &times);
    Ok(())
}

fn benchmark_cigar_paf_plain(
    path: &str,
    num_records: usize,
    num_positions: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = build_cigar_index_plain(path, num_records)?;
    if entries.is_empty() {
        println!("No CIGAR entries to benchmark");
        return Ok(());
    }

    eprintln!("Indexed {} CIGAR entries (plain)", entries.len());

    let test_positions: Vec<usize> = (0..entries.len())
        .step_by(entries.len() / num_positions.min(entries.len()).max(1))
        .take(num_positions)
        .collect();

    let mut file = File::open(path)?;
    let mut reader = io::BufReader::new(file);
    let mut times = Vec::new();

    for &idx in &test_positions {
        let entry = &entries[idx];
        let mut buffer = vec![0u8; entry.len];

        // Warmup
        for _ in 0..3 {
            reader.seek(io::SeekFrom::Start(entry.pos))?;
            reader.read_exact(&mut buffer)?;
        }

        // Measure
        for _ in 0..iterations {
            let start = Instant::now();
            reader.seek(io::SeekFrom::Start(entry.pos))?;
            reader.read_exact(&mut buffer)?;
            times.push(start.elapsed().as_micros() as f64);
        }
    }

    print_stats("CIGAR PAF (plain)", &times);
    Ok(())
}

fn benchmark_tracepoint_paf_plain(
    path: &str,
    num_records: usize,
    num_positions: usize,
    iterations: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let entries = build_tracepoint_index_plain(path, num_records)?;
    if entries.is_empty() {
        println!("No tracepoint entries to benchmark");
        return Ok(());
    }

    eprintln!("Indexed {} tracepoint entries (plain)", entries.len());

    let test_positions: Vec<usize> = (0..entries.len())
        .step_by(entries.len() / num_positions.min(entries.len()).max(1))
        .take(num_positions)
        .collect();

    let mut reader = io::BufReader::new(File::open(path)?);
    let mut times = Vec::new();
    let mut buffer = Vec::new();

    for &idx in &test_positions {
        let entry = &entries[idx];
        buffer.resize(entry.len, 0);

        // Warmup
        for _ in 0..3 {
            reader.seek(io::SeekFrom::Start(entry.pos))?;
            reader.read_exact(&mut buffer)?;
        }

        // Measure
        for _ in 0..iterations {
            let start = Instant::now();
            reader.seek(io::SeekFrom::Start(entry.pos))?;
            reader.read_exact(&mut buffer)?;
            times.push(start.elapsed().as_micros() as f64);
        }
    }

    print_stats("Tracepoint PAF (plain)", &times);
    Ok(())
}

// Helper structs and functions

#[derive(Clone, Copy)]
struct CigarEntryBgz {
    pos: VirtualPosition,
    len: usize,
}

#[derive(Clone, Copy)]
struct TracepointEntryBgz {
    pos: VirtualPosition,
    len: usize,
}

#[derive(Clone, Copy)]
struct PlainEntry {
    pos: u64,
    len: usize,
}

fn build_cigar_index_bgz(
    path: &str,
    limit: usize,
) -> Result<Vec<CigarEntryBgz>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BgzfReader::new(file);
    let mut line = Vec::new();
    let mut entries = Vec::new();

    while entries.len() < limit {
        let line_start = reader.virtual_position();
        line.clear();

        let bytes_read = reader.read_until(b'\n', &mut line)?;
        if bytes_read == 0 {
            break;
        }

        if let Some((offset, len)) = find_tag_offset_and_len(&line, b"cg:Z:") {
            reader.seek(line_start)?;
            if offset > 0 {
                io::copy(&mut reader.by_ref().take(offset as u64), &mut io::sink())?;
            }
            let cigar_pos = reader.virtual_position();
            let remaining = line.len() as u64 - offset as u64;
            if remaining > 0 {
                io::copy(&mut reader.by_ref().take(remaining), &mut io::sink())?;
            }

            if len > 0 {
                entries.push(CigarEntryBgz {
                    pos: cigar_pos,
                    len,
                });
            }
        }
    }

    Ok(entries)
}

fn build_tracepoint_index_bgz(
    path: &str,
    limit: usize,
) -> Result<Vec<TracepointEntryBgz>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BgzfReader::new(file);
    let mut line = Vec::new();
    let mut entries = Vec::new();

    while entries.len() < limit {
        let line_start = reader.virtual_position();
        line.clear();

        let bytes_read = reader.read_until(b'\n', &mut line)?;
        if bytes_read == 0 {
            break;
        }

        if let Some((offset, len)) = find_tag_offset_and_len(&line, b"tp:Z:") {
            reader.seek(line_start)?;
            if offset > 0 {
                io::copy(&mut reader.by_ref().take(offset as u64), &mut io::sink())?;
            }
            let tp_pos = reader.virtual_position();
            let remaining = line.len() as u64 - offset as u64;
            if remaining > 0 {
                io::copy(&mut reader.by_ref().take(remaining), &mut io::sink())?;
            }

            if len > 0 {
                entries.push(TracepointEntryBgz { pos: tp_pos, len });
            }
        }
    }

    Ok(entries)
}

fn build_cigar_index_plain(
    path: &str,
    limit: usize,
) -> Result<Vec<PlainEntry>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = io::BufReader::new(file);
    let mut line = Vec::new();
    let mut entries = Vec::new();

    loop {
        let pos = reader.stream_position()?;
        line.clear();
        let bytes_read = reader.read_until(b'\n', &mut line)?;
        if bytes_read == 0 || entries.len() >= limit {
            break;
        }

        if let Some((offset, len)) = find_tag_offset_and_len(&line, b"cg:Z:") {
            if len > 0 {
                entries.push(PlainEntry {
                    pos: pos + offset as u64,
                    len,
                });
            }
        }
    }

    Ok(entries)
}

fn build_tracepoint_index_plain(
    path: &str,
    limit: usize,
) -> Result<Vec<PlainEntry>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = io::BufReader::new(file);
    let mut line = Vec::new();
    let mut entries = Vec::new();

    loop {
        let pos = reader.stream_position()?;
        line.clear();
        let bytes_read = reader.read_until(b'\n', &mut line)?;
        if bytes_read == 0 || entries.len() >= limit {
            break;
        }

        if let Some((offset, len)) = find_tag_offset_and_len(&line, b"tp:Z:") {
            if len > 0 {
                entries.push(PlainEntry {
                    pos: pos + offset as u64,
                    len,
                });
            }
        }
    }

    Ok(entries)
}

fn find_tag_offset_and_len(line: &[u8], tag: &[u8]) -> Option<(usize, usize)> {
    let pos = line.windows(tag.len()).position(|window| window == tag)?;
    let start = pos + tag.len();
    if start >= line.len() {
        return None;
    }

    let len = line[start..]
        .iter()
        .take_while(|&&b| b != b'\t' && b != b'\n' && b != b'\r')
        .count();

    if len == 0 {
        None
    } else {
        Some((start, len))
    }
}

fn print_stats(format: &str, times: &[f64]) {
    if times.is_empty() {
        println!("{} avg_us=N/A stddev_us=N/A min_us=N/A max_us=N/A", format);
        return;
    }

    let sum: f64 = times.iter().sum();
    let avg = sum / times.len() as f64;

    let variance: f64 = times.iter().map(|&t| (t - avg).powi(2)).sum::<f64>() / times.len() as f64;
    let stddev = variance.sqrt();

    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "{} avg_us={:.2} stddev_us={:.2} min_us={:.2} max_us={:.2} samples={}",
        format,
        avg,
        stddev,
        min,
        max,
        times.len()
    );
}
