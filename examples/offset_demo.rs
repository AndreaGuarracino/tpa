use std::time::Instant;
/// Demo of offset-based access using direct file offsets
use tpa::{TpaReader, TracepointData};

/// Helper to display ComplexityMetric (doesn't implement Debug)
fn complexity_metric_str(metric: tracepoints::ComplexityMetric) -> &'static str {
    match metric {
        tracepoints::ComplexityMetric::EditDistance => "EditDistance",
        tracepoints::ComplexityMetric::DiagonalDistance => "DiagonalDistance",
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <file.tpa> <offset>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} alignments.tpa 123456", args[0]);
        std::process::exit(1);
    }

    let tpa_path = &args[1];
    let offset: u64 = args[2].parse().expect("Invalid offset");

    println!("Opening {}...", tpa_path);
    let start = Instant::now();
    let mut reader = TpaReader::new(tpa_path)?;
    let open_time = start.elapsed();

    println!(
        "✓ Opened in {:.3}ms (index-backed)",
        open_time.as_secs_f64() * 1000.0
    );
    println!();

    // Access by offset
    println!("Reading tracepoints at offset {}...", offset);
    let start = Instant::now();
    let (tracepoints, complexity_metric, max_complexity) =
        reader.get_tracepoints_at_offset(offset)?;
    let read_time = start.elapsed();

    let tp_count = tracepoint_len(&tracepoints);

    println!("✓ Read in {:.3}ms", read_time.as_secs_f64() * 1000.0);
    println!("  Type: {:?}", tracepoints.tp_type());
    println!(
        "  Complexity: {} (max: {})",
        complexity_metric_str(complexity_metric),
        max_complexity
    );
    println!("  Tracepoints: {} items", tp_count);

    Ok(())
}

fn tracepoint_len(tp: &TracepointData) -> usize {
    match tp {
        TracepointData::Standard(tps) | TracepointData::Fastga(tps) => tps.len(),
        TracepointData::Variable(tps) => tps.len(),
        TracepointData::Mixed(items) => items.len(),
    }
}
