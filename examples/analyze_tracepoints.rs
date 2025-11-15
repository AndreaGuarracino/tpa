use lib_bpaf::BpafReader;
use lib_tracepoints::TracepointData;
use std::collections::HashMap;
use std::env;
use std::io;

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <file.bpaf>", args[0]);
        eprintln!("");
        eprintln!("Analyzes tracepoint statistics for compression research");
        std::process::exit(1);
    }

    let bpaf_path = &args[1];

    println!("======================================");
    println!("Tracepoint Analysis Report");
    println!("======================================");
    println!("File: {}", bpaf_path);
    println!();

    let mut reader = BpafReader::open(bpaf_path)?;

    // Get header info
    let header = reader.header();
    let strategy = header.strategy().expect("Failed to get strategy");
    let record_count = header.num_records();

    println!("Records: {}", record_count);
    println!("Strategy: {:?}", strategy);
    println!();

    // Collect all tracepoints (only Standard and Fastga types for now)
    println!("Reading all tracepoints...");
    let mut all_tracepoints: Vec<Vec<(usize, usize)>> = Vec::new();

    for record_id in 0..record_count {
        if let Ok((tp_data, _, _)) = reader.get_tracepoints(record_id) {
            match tp_data {
                TracepointData::Standard(tps) | TracepointData::Fastga(tps) => {
                    all_tracepoints.push(tps);
                }
                TracepointData::Mixed(_) | TracepointData::Variable(_) => {
                    eprintln!("Warning: Mixed and Variable tracepoint types not yet supported for analysis");
                    break;
                }
            }
        }
    }

    let total_records = all_tracepoints.len();
    println!("Successfully read {} records", total_records);
    println!();

    // Analyze tracepoints
    analyze_tracepoints(&all_tracepoints);

    Ok(())
}

fn analyze_tracepoints(all_tracepoints: &[Vec<(usize, usize)>]) {
    println!("======================================");
    println!("Delta Analysis");
    println!("======================================");

    let mut all_query_deltas = Vec::new();
    let mut all_target_deltas = Vec::new();
    let mut query_target_diffs = Vec::new();

    // Compute deltas for each record
    for tps in all_tracepoints {
        if tps.is_empty() {
            continue;
        }

        // Query deltas
        for i in 1..tps.len() {
            let delta = tps[i].0 as i64 - tps[i - 1].0 as i64;
            all_query_deltas.push(delta);
        }

        // Target deltas
        for i in 1..tps.len() {
            let delta = tps[i].1 as i64 - tps[i - 1].1 as i64;
            all_target_deltas.push(delta);
        }

        // Query-target differences (for each tracepoint pair)
        for &(q, t) in tps {
            let diff = t as i64 - q as i64;
            query_target_diffs.push(diff);
        }
    }

    if all_query_deltas.is_empty() {
        println!("No deltas to analyze");
        return;
    }

    // Delta statistics
    println!("\n--- Query Deltas ---");
    print_delta_stats(&all_query_deltas, "Query");

    println!("\n--- Target Deltas ---");
    print_delta_stats(&all_target_deltas, "Target");

    println!("\n--- Query-Target Differences ---");
    print_delta_stats(&query_target_diffs, "Q-T Diff");

    // Small deltas percentage
    let query_small = all_query_deltas.iter().filter(|&&d| d.abs() < 256).count();
    let target_small = all_target_deltas.iter().filter(|&&d| d.abs() < 256).count();

    println!("\n======================================");
    println!("Small Deltas (<256) Analysis");
    println!("======================================");
    println!("Query deltas <256:  {}/{} ({:.2}%)",
             query_small, all_query_deltas.len(),
             100.0 * query_small as f64 / all_query_deltas.len() as f64);
    println!("Target deltas <256: {}/{} ({:.2}%)",
             target_small, all_target_deltas.len(),
             100.0 * target_small as f64 / all_target_deltas.len() as f64);

    // Delta value distribution
    println!("\n======================================");
    println!("Delta Value Distribution");
    println!("======================================");
    print_distribution(&all_query_deltas, "Query");
    print_distribution(&all_target_deltas, "Target");

    // Run length analysis
    println!("\n======================================");
    println!("Run Length Analysis");
    println!("======================================");
    let (q_max_run, q_avg_run, q_runs) = analyze_runs(&all_query_deltas);
    let (t_max_run, t_avg_run, t_runs) = analyze_runs(&all_target_deltas);

    println!("Query deltas:");
    println!("  Max run length: {}", q_max_run);
    println!("  Avg run length: {:.2}", q_avg_run);
    println!("  Total runs: {}", q_runs);

    println!("Target deltas:");
    println!("  Max run length: {}", t_max_run);
    println!("  Avg run length: {:.2}", t_avg_run);
    println!("  Total runs: {}", t_runs);

    // Correlation analysis
    println!("\n======================================");
    println!("Correlation Analysis");
    println!("======================================");

    let correlation = compute_correlation(&all_query_deltas, &all_target_deltas);
    println!("Query-Target delta correlation: {:.4}", correlation);

    // Query-Target difference statistics
    let diff_mean = query_target_diffs.iter().map(|&x| x as f64).sum::<f64>()
                    / query_target_diffs.len() as f64;
    let diff_variance = query_target_diffs.iter()
        .map(|&x| (x as f64 - diff_mean).powi(2))
        .sum::<f64>() / query_target_diffs.len() as f64;
    let diff_std_dev = diff_variance.sqrt();

    println!("Query-Target differences:");
    println!("  Mean: {:.2}", diff_mean);
    println!("  Std Dev: {:.2}", diff_std_dev);
    println!("  Coefficient of Variation: {:.4}", diff_std_dev / diff_mean.abs());

    // Additional statistics
    println!("\n======================================");
    println!("Additional Statistics");
    println!("======================================");

    // Zero deltas
    let query_zeros = all_query_deltas.iter().filter(|&&d| d == 0).count();
    let target_zeros = all_target_deltas.iter().filter(|&&d| d == 0).count();

    println!("Zero deltas:");
    println!("  Query: {}/{} ({:.2}%)",
             query_zeros, all_query_deltas.len(),
             100.0 * query_zeros as f64 / all_query_deltas.len() as f64);
    println!("  Target: {}/{} ({:.2}%)",
             target_zeros, all_target_deltas.len(),
             100.0 * target_zeros as f64 / all_target_deltas.len() as f64);

    // Positive vs negative deltas
    let query_pos = all_query_deltas.iter().filter(|&&d| d > 0).count();
    let query_neg = all_query_deltas.iter().filter(|&&d| d < 0).count();
    let target_pos = all_target_deltas.iter().filter(|&&d| d > 0).count();
    let target_neg = all_target_deltas.iter().filter(|&&d| d < 0).count();

    println!("\nDelta directions:");
    println!("  Query: +{} ({}%) / -{} ({}%)",
             query_pos, 100 * query_pos / all_query_deltas.len(),
             query_neg, 100 * query_neg / all_query_deltas.len());
    println!("  Target: +{} ({}%) / -{} ({}%)",
             target_pos, 100 * target_pos / all_target_deltas.len(),
             target_neg, 100 * target_neg / all_target_deltas.len());
}

fn print_delta_stats(deltas: &[i64], name: &str) {
    if deltas.is_empty() {
        return;
    }

    let min = *deltas.iter().min().unwrap();
    let max = *deltas.iter().max().unwrap();
    let mean = deltas.iter().map(|&x| x as f64).sum::<f64>() / deltas.len() as f64;

    let variance = deltas.iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>() / deltas.len() as f64;
    let std_dev = variance.sqrt();

    // Median
    let mut sorted = deltas.to_vec();
    sorted.sort_unstable();
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) as f64 / 2.0
    } else {
        sorted[sorted.len() / 2] as f64
    };

    println!("{} deltas: n={}", name, deltas.len());
    println!("  Min: {}", min);
    println!("  Max: {}", max);
    println!("  Mean: {:.2}", mean);
    println!("  Median: {:.2}", median);
    println!("  Std Dev: {:.2}", std_dev);
    println!("  Range: {}", max - min);
}

fn print_distribution(deltas: &[i64], name: &str) {
    let mut buckets: HashMap<&str, usize> = HashMap::new();

    for &delta in deltas {
        let abs_delta = delta.abs();
        let bucket = if abs_delta == 0 {
            "0"
        } else if abs_delta < 8 {
            "1-7"
        } else if abs_delta < 16 {
            "8-15"
        } else if abs_delta < 32 {
            "16-31"
        } else if abs_delta < 64 {
            "32-63"
        } else if abs_delta < 128 {
            "64-127"
        } else if abs_delta < 256 {
            "128-255"
        } else if abs_delta < 512 {
            "256-511"
        } else if abs_delta < 1024 {
            "512-1023"
        } else if abs_delta < 2048 {
            "1024-2047"
        } else {
            "2048+"
        };

        *buckets.entry(bucket).or_insert(0) += 1;
    }

    println!("{} distribution:", name);
    let order = ["0", "1-7", "8-15", "16-31", "32-63", "64-127",
                 "128-255", "256-511", "512-1023", "1024-2047", "2048+"];

    for bucket_name in &order {
        if let Some(&count) = buckets.get(bucket_name) {
            let pct = 100.0 * count as f64 / deltas.len() as f64;
            println!("  {:>10}: {:>8} ({:>5.2}%)", bucket_name, count, pct);
        }
    }
}

fn analyze_runs(deltas: &[i64]) -> (usize, f64, usize) {
    if deltas.is_empty() {
        return (0, 0.0, 0);
    }

    let mut max_run = 1;
    let mut current_run = 1;
    let mut total_runs = 1;
    let mut prev = deltas[0];

    for &delta in &deltas[1..] {
        if delta == prev {
            current_run += 1;
        } else {
            max_run = max_run.max(current_run);
            current_run = 1;
            total_runs += 1;
        }
        prev = delta;
    }
    max_run = max_run.max(current_run);

    let avg_run = deltas.len() as f64 / total_runs as f64;

    (max_run, avg_run, total_runs)
}

fn compute_correlation(x: &[i64], y: &[i64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    let x = &x[..n];
    let y = &y[..n];

    let mean_x = x.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let mean_y = y.iter().map(|&v| v as f64).sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..n {
        let dx = x[i] as f64 - mean_x;
        let dy = y[i] as f64 - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    if sum_sq_x == 0.0 || sum_sq_y == 0.0 {
        return 0.0;
    }

    numerator / (sum_sq_x * sum_sq_y).sqrt()
}
