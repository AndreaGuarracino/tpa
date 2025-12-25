/// Benchmark for measuring TPA file open time across different compression modes.
///
/// Usage: open_bench <file.tpa> [iterations]
///
/// The benchmark:
/// 1. Opens the TPA file multiple times (default: 1000 iterations)
/// 2. Measures average, min, max open time
/// 3. Reports mode detection (per-record or all-records)
use std::time::Instant;
use tpa::TpaReader;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file.tpa> [iterations] [--simple]", args[0]);
        eprintln!();
        eprintln!("Benchmarks file open time for TPA files.");
        eprintln!("Default: 1000 iterations");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --simple  Output only the average time in μs (for scripting)");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} alignments.tpa", args[0]);
        eprintln!("  {} alignments.tpa 5000", args[0]);
        eprintln!(
            "  {} alignments.tpa 100 --simple  # outputs just the average μs",
            args[0]
        );
        std::process::exit(1);
    }

    let tpa_path = &args[1];
    let simple_mode = args.iter().any(|a| a == "--simple");
    let iterations: usize = args
        .iter()
        .skip(2)
        .filter(|s| *s != "--simple")
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    if !simple_mode {
        println!("=== TPA File Open Benchmark ===");
        println!("File: {}", tpa_path);
        println!("Iterations: {}", iterations);
        println!();
    }

    // First, open once to get file info and detect mode
    let reader = TpaReader::new(tpa_path)?;
    let num_records = reader.header().num_records();
    let num_strings = reader.header().num_strings();
    let is_all_records_mode = reader.is_all_records_mode();
    let (first_strategy, second_strategy) = reader.header().strategies()?;

    // Determine mode description
    let mode_desc = if is_all_records_mode {
        "All-records mode"
    } else {
        "Per-record mode"
    };

    if !simple_mode {
        println!("File info:");
        println!("  Mode: {}", mode_desc);
        println!("  Records: {}", num_records);
        println!("  Strings: {}", num_strings);
        println!("  Strategies: {} / {}", first_strategy, second_strategy);
        println!();
    }
    drop(reader);

    // Benchmark: measure open time
    if !simple_mode {
        println!("Benchmarking {} opens...", iterations);
    }

    let mut open_times = Vec::with_capacity(iterations);

    // Warmup (10% of iterations or at least 10)
    let warmup_count = (iterations / 10).max(10);
    for _ in 0..warmup_count {
        let _ = TpaReader::new(tpa_path)?;
    }

    // Actual benchmark
    for _ in 0..iterations {
        let start = Instant::now();
        let _reader = TpaReader::new(tpa_path)?;
        let elapsed = start.elapsed();
        open_times.push(elapsed.as_micros() as f64);
    }

    // Calculate statistics
    let sum: f64 = open_times.iter().sum();
    let avg = sum / iterations as f64;

    // Simple mode: just output the average
    if simple_mode {
        println!("{:.2}", avg);
        return Ok(());
    }

    let min = open_times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = open_times.iter().copied().fold(0.0, f64::max);

    // Standard deviation
    let variance: f64 =
        open_times.iter().map(|t| (t - avg).powi(2)).sum::<f64>() / iterations as f64;
    let stddev = variance.sqrt();

    // Percentiles
    open_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = open_times[iterations / 2];
    let p95 = open_times[(iterations as f64 * 0.95) as usize];
    let p99 = open_times[(iterations as f64 * 0.99) as usize];

    println!();
    println!("=== Results ===");
    println!("Mode: {}", mode_desc);
    println!();
    println!("Open time statistics (μs):");
    println!("  Average: {:.2} μs", avg);
    println!("  Std dev: {:.2} μs", stddev);
    println!("  Min:     {:.2} μs", min);
    println!("  Max:     {:.2} μs", max);
    println!();
    println!("Percentiles:");
    println!("  P50:     {:.2} μs", p50);
    println!("  P95:     {:.2} μs", p95);
    println!("  P99:     {:.2} μs", p99);
    println!();

    // Throughput
    let opens_per_sec = 1_000_000.0 / avg;
    println!("Throughput: {:.0} opens/sec", opens_per_sec);

    Ok(())
}
