/// Seek Performance Benchmark
/// Tests O(1) random access performance with statistics
use lib_bpaf::{BpafReader, Result};
use std::env;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <bpaf_file> [num_seeks]", args[0]);
        eprintln!("Example: {} test.bpaf 1000", args[0]);
        std::process::exit(1);
    }

    let bpaf_path = &args[1];
    let num_seeks: usize = args.get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    println!("═══════════════════════════════════════════════");
    println!("Seek Performance Benchmark");
    println!("═══════════════════════════════════════════════");
    println!("File: {}", bpaf_path);
    println!("Seeks: {}", num_seeks);
    println!();

    // Open BPAF file
    let reader = BpafReader::from_path(bpaf_path)?;
    let total_records = reader.num_records();

    println!("Total records: {}", total_records);

    if total_records == 0 {
        println!("No records to seek!");
        return Ok(());
    }

    // Generate random seek positions
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let positions: Vec<usize> = (0..num_seeks)
        .map(|_| rng.gen_range(0..total_records))
        .collect();

    println!("Testing {} random seeks...", num_seeks);
    println!();

    // Perform seeks and measure time
    let mut times_ns: Vec<u64> = Vec::with_capacity(num_seeks);
    let mut success_count = 0;

    for &pos in &positions {
        let start = Instant::now();
        match reader.get_record(pos) {
            Ok(_record) => {
                let duration = start.elapsed();
                times_ns.push(duration.as_nanos() as u64);
                success_count += 1;
            }
            Err(e) => {
                eprintln!("Seek to {} failed: {}", pos, e);
            }
        }
    }

    if success_count == 0 {
        println!("All seeks failed!");
        return Ok(());
    }

    // Calculate statistics
    let times_us: Vec<f64> = times_ns.iter()
        .map(|&ns| ns as f64 / 1000.0)
        .collect();

    let sum: f64 = times_us.iter().sum();
    let mean = sum / times_us.len() as f64;

    let variance: f64 = times_us.iter()
        .map(|&t| (t - mean).powi(2))
        .sum::<f64>() / times_us.len() as f64;
    let stddev = variance.sqrt();

    let mut sorted_times = times_us.clone();
    sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = sorted_times[0];
    let max = sorted_times[sorted_times.len() - 1];
    let median = sorted_times[sorted_times.len() / 2];
    let p95 = sorted_times[(sorted_times.len() as f64 * 0.95) as usize];
    let p99 = sorted_times[(sorted_times.len() as f64 * 0.99) as usize];

    // Print results
    println!("───────────────────────────────────────────────");
    println!("RESULTS");
    println!("───────────────────────────────────────────────");
    println!("Successful seeks: {} / {}", success_count, num_seeks);
    println!();
    println!("Seek time statistics (microseconds):");
    println!("  Mean:     {:>10.2} μs", mean);
    println!("  Std Dev:  {:>10.2} μs", stddev);
    println!("  Median:   {:>10.2} μs", median);
    println!("  Min:      {:>10.2} μs", min);
    println!("  Max:      {:>10.2} μs", max);
    println!("  P95:      {:>10.2} μs", p95);
    println!("  P99:      {:>10.2} μs", p99);
    println!();

    // Check if O(1) is achieved (< 1ms for in-memory access)
    if mean < 1000.0 {
        println!("✓ O(1) performance achieved (mean < 1ms)");
    } else {
        println!("⚠ Slow seeks detected (mean >= 1ms)");
    }

    // Check consistency (low variance means predictable O(1))
    let cv = (stddev / mean) * 100.0; // Coefficient of variation
    println!("  Coefficient of variation: {:.1}%", cv);

    if cv < 20.0 {
        println!("✓ Consistent O(1) performance (CV < 20%)");
    } else if cv < 50.0 {
        println!("⚠ Moderate variance in seek times (CV < 50%)");
    } else {
        println!("⚠ High variance in seek times (CV >= 50%)");
    }

    println!("═══════════════════════════════════════════════");

    Ok(())
}
