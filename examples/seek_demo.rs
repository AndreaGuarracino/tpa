/// Demo of seekable BPAF reader with O(1) random access and performance profiling
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
    if args.len() < 2 {
        eprintln!("Usage: {} <file.bpaf> [mode] [record_ids...]", args[0]);
        eprintln!("\nModes:");
        eprintln!("  full       - Full alignment record access (default)");
        eprintln!("  tracepoint - Tracepoint-only access (optimized)");
        eprintln!("  profile    - Profile both access methods");
        eprintln!("\nExamples:");
        eprintln!("  {} alignments.bpaf full 0 100 500 1000", args[0]);
        eprintln!("  {} alignments.bpaf tracepoint 0 100 500 1000", args[0]);
        eprintln!("  {} alignments.bpaf profile", args[0]);
        std::process::exit(1);
    }

    let bpaf_path = &args[1];

    // Determine mode and where record IDs start
    let (mode, record_ids_start) = if args.len() > 2 {
        match args[2].as_str() {
            "profile" | "full" | "tracepoint" | "fast" => (args[2].as_str(), 3),
            _ => ("full", 2), // If arg[2] is not a known mode, treat it as a record ID
        }
    } else {
        ("full", 2)
    };

    // Open BPAF file - builds index if it doesn't exist
    println!("Opening {}...", bpaf_path);
    let mut reader = BpafReader::open(bpaf_path)?;

    println!("File info:");
    println!("  Total records: {}", reader.len());
    println!();

    match mode {
        "profile" => profile_methods(&mut reader)?,
        "full" => demo_full_access(&mut reader, &args[record_ids_start..])?,
        "tracepoint" | "fast" => demo_tracepoint_access(&mut reader, &args[record_ids_start..])?,
        _ => {
            eprintln!("Unknown mode: {}", mode);
            eprintln!("Valid modes: full, tracepoint, profile");
            std::process::exit(1);
        }
    }

    Ok(())
}

fn demo_full_access(reader: &mut BpafReader, args: &[String]) -> std::io::Result<()> {
    let record_ids: Vec<u64> = if args.is_empty() {
        (0..5.min(reader.len() as u64)).collect()
    } else {
        args.iter().filter_map(|s| s.parse().ok()).collect()
    };

    println!("=== Full Record Access ===");
    println!("Getting {} records...\n", record_ids.len());

    // Load string table once for name lookups
    reader.load_string_table()?;

    let start = Instant::now();
    for &record_id in &record_ids {
        match reader.get_alignment_record(record_id) {
            Ok(record) => {
                let query_name = reader.string_table_ref().get(record.query_name_id).unwrap();
                let target_name = reader
                    .string_table_ref()
                    .get(record.target_name_id)
                    .unwrap();

                let tp_count = tracepoint_len(&record.tracepoints);

                println!("Record {}:", record_id);
                println!(
                    "  Query:  {} ({}..{})",
                    query_name, record.query_start, record.query_end
                );
                println!(
                    "  Target: {} ({}..{})",
                    target_name, record.target_start, record.target_end
                );
                println!("  Strand: {}", record.strand);
                println!("  Tracepoints: {} items", tp_count);
                println!("  Match quality: {}", record.mapping_quality);
                println!();
            }
            Err(e) => eprintln!("Error seeking to record {}: {}", record_id, e),
        }
    }
    let elapsed = start.elapsed();

    println!(
        "✓ Full access complete: {} records in {:.3}ms ({:.3}ms/record)",
        record_ids.len(),
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / record_ids.len() as f64
    );

    Ok(())
}

fn demo_tracepoint_access(reader: &mut BpafReader, args: &[String]) -> std::io::Result<()> {
    let record_ids: Vec<u64> = if args.is_empty() {
        (0..5.min(reader.len() as u64)).collect()
    } else {
        args.iter().filter_map(|s| s.parse().ok()).collect()
    };

    println!("=== Tracepoint-Only Access (Optimized) ===");
    println!("Getting {} records...\n", record_ids.len());

    let start = Instant::now();
    for &record_id in &record_ids {
        match reader.get_tracepoints(record_id) {
            Ok((tracepoints, complexity_metric, max_complexity)) => {
                let tp_count = tracepoint_len(&tracepoints);

                println!("Record {}:", record_id);
                println!("  Type: {:?}", tracepoints.tp_type());
                println!(
                    "  Complexity: {} (max: {})",
                    complexity_metric_str(complexity_metric),
                    max_complexity
                );
                println!("  Tracepoints: {} items", tp_count);
                println!();
            }
            Err(e) => eprintln!("Error seeking to record {}: {}", record_id, e),
        }
    }
    let elapsed = start.elapsed();

    println!(
        "✓ Tracepoint access complete: {} records in {:.3}ms ({:.3}ms/record)",
        record_ids.len(),
        elapsed.as_secs_f64() * 1000.0,
        elapsed.as_secs_f64() * 1000.0 / record_ids.len() as f64
    );

    Ok(())
}

fn profile_methods(reader: &mut BpafReader) -> std::io::Result<()> {
    println!("=== Performance Profiling ===\n");

    // Generate test record IDs: random access pattern
    let total_records = reader.len() as u64;
    let test_sizes = vec![10, 100, 1000];

    for &size in &test_sizes {
        if size > total_records {
            continue;
        }

        // Generate random-ish record IDs (pseudo-random for determinism)
        let record_ids: Vec<u64> = (0..size)
            .map(|i| (i * 997 + 13) % total_records) // Simple pseudo-random
            .collect();

        println!(
            "--- Testing with {} records (random access pattern) ---",
            size
        );

        // Test 1: Full record access
        let start = Instant::now();
        for &id in &record_ids {
            let _ = reader.get_alignment_record(id)?;
        }
        let full_elapsed = start.elapsed();

        // Test 2: Tracepoint-only access (optimized)
        let start = Instant::now();
        for &id in &record_ids {
            let _ = reader.get_tracepoints(id)?;
        }
        let tracepoint_elapsed = start.elapsed();

        // Report results
        println!(
            "  Full record access:     {:.3}ms total, {:.3}ms/record",
            full_elapsed.as_secs_f64() * 1000.0,
            full_elapsed.as_secs_f64() * 1000.0 / size as f64
        );
        println!(
            "  Tracepoint-only access: {:.3}ms total, {:.3}ms/record ({:.1}x speedup)",
            tracepoint_elapsed.as_secs_f64() * 1000.0,
            tracepoint_elapsed.as_secs_f64() * 1000.0 / size as f64,
            full_elapsed.as_secs_f64() / tracepoint_elapsed.as_secs_f64()
        );
        println!();
    }

    println!("✓ Profiling complete");
    Ok(())
}

fn tracepoint_len(tp: &TracepointData) -> usize {
    match tp {
        TracepointData::Standard(tps) | TracepointData::Fastga(tps) => tps.len(),
        TracepointData::Variable(tps) => tps.len(),
        TracepointData::Mixed(items) => items.len(),
    }
}
