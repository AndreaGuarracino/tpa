/// Demo of offset-based access without index loading
use lib_bpaf::{BpafReader, TracepointData};
use std::time::Instant;

/// Helper to display ComplexityMetric (doesn't implement Debug)
fn complexity_metric_str(metric: lib_tracepoints::ComplexityMetric) -> &'static str {
    match metric {
        lib_tracepoints::ComplexityMetric::EditDistance => "EditDistance",
        lib_tracepoints::ComplexityMetric::DiagonalDistance => "DiagonalDistance",
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <file.bpaf> <offset>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  {} alignments.bpaf 123456", args[0]);
        std::process::exit(1);
    }

    let bpaf_path = &args[1];
    let offset: u64 = args[2].parse().expect("Invalid offset");

    println!("Opening {} WITHOUT index...", bpaf_path);
    let start = Instant::now();
    let mut reader = BpafReader::open_without_index(bpaf_path)?;
    let open_time = start.elapsed();

    println!(
        "✓ Opened in {:.3}ms (no index loaded)",
        open_time.as_secs_f64() * 1000.0
    );
    println!();

    // Access by offset
    println!("Reading tracepoints at offset {}...", offset);
    let start = Instant::now();
    let (tracepoints, tp_type, complexity_metric, max_complexity) =
        reader.get_tracepoints_at_offset(offset)?;
    let read_time = start.elapsed();

    let tp_count = tracepoint_len(&tracepoints);

    println!("✓ Read in {:.3}ms", read_time.as_secs_f64() * 1000.0);
    println!("  Type: {:?}", tp_type);
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
