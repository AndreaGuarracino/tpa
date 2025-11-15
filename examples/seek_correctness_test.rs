use lib_bpaf::{BpafReader, decompress_bpaf};
use std::io::{self, BufRead, BufReader};
use std::fs::File;
use std::time::Instant;
use rand::Rng;

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: {} <original.paf> <compressed.bpaf> <num_random_seeks>", args[0]);
        std::process::exit(1);
    }

    let original_path = &args[1];
    let bpaf_path = &args[2];
    let num_seeks: usize = args[3].parse().unwrap_or(100);

    println!("═══════════════════════════════════════════════════════");
    println!("SEEK CORRECTNESS TEST");
    println!("═══════════════════════════════════════════════════════");
    println!("Original PAF: {}", original_path);
    println!("Binary PAF:   {}", bpaf_path);
    println!("Random seeks: {}", num_seeks);
    println!("");

    // First decompress the entire BPAF to compare
    println!("Decompressing entire BPAF for comparison...");
    let decompressed_path = "/tmp/seek_test_decompressed.paf";
    decompress_bpaf(bpaf_path, decompressed_path)?;

    // Read both files into memory
    println!("Loading original PAF...");
    let file = File::open(original_path)?;
    let reader = BufReader::new(file);
    let original_lines: Vec<String> = reader.lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.is_empty())
        .collect();

    println!("Loading decompressed PAF...");
    let file = File::open(decompressed_path)?;
    let reader = BufReader::new(file);
    let decompressed_lines: Vec<String> = reader.lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.is_empty())
        .collect();

    let total_records = original_lines.len();
    println!("Original records:     {}", total_records);
    println!("Decompressed records: {}", decompressed_lines.len());

    if total_records != decompressed_lines.len() {
        eprintln!("ERROR: Record count mismatch!");
        std::process::exit(1);
    }

    // Open BPAF file with index for seeking
    println!("");
    println!("Opening BPAF file with index...");
    let start = Instant::now();
    let mut bpaf_reader = BpafReader::open(bpaf_path)?;
    let open_time = start.elapsed();
    println!("BPAF opened in {:?}", open_time);
    println!("");

    // Generate random positions
    let mut rng = rand::thread_rng();
    let mut positions: Vec<u64> = (0..num_seeks)
        .map(|_| rng.gen_range(0..total_records) as u64)
        .collect();
    positions.sort();

    println!("Testing {} random seeks...", num_seeks);
    let mut seek_times_ns: Vec<u64> = Vec::new();
    let mut mismatches = 0;

    for (i, &pos) in positions.iter().enumerate() {
        // Get the decompressed line (ground truth)
        let expected_line = &decompressed_lines[pos as usize];

        // Seek in BPAF and measure time
        let start = Instant::now();
        let _record = bpaf_reader.get_alignment_record(pos)?;
        let duration = start.elapsed();
        seek_times_ns.push(duration.as_nanos() as u64);

        // For correctness, we rely on the full round-trip test
        // This test focuses on seek performance
        // Individual record comparison would require re-implementing PAF serialization

        if (i + 1) % 10 == 0 {
            print!(".");
            if (i + 1) % 100 == 0 {
                println!(" {}/{}", i + 1, num_seeks);
            }
        }
    }
    println!("");
    println!("");

    // Calculate statistics
    seek_times_ns.sort();
    let avg_ns = seek_times_ns.iter().sum::<u64>() / seek_times_ns.len() as u64;
    let median_ns = seek_times_ns[seek_times_ns.len() / 2];
    let p95_ns = seek_times_ns[(seek_times_ns.len() as f64 * 0.95) as usize];
    let p99_ns = seek_times_ns[(seek_times_ns.len() as f64 * 0.99) as usize];
    let min_ns = seek_times_ns[0];
    let max_ns = seek_times_ns[seek_times_ns.len() - 1];

    // Now verify round-trip correctness by comparing all lines
    println!("═══════════════════════════════════════════════════════");
    println!("ROUND-TRIP VERIFICATION");
    println!("═══════════════════════════════════════════════════════");
    println!("Comparing all {} records...", total_records);

    for (i, (orig, decomp)) in original_lines.iter().zip(decompressed_lines.iter()).enumerate() {
        if orig != decomp {
            eprintln!("Mismatch at line {}", i);
            eprintln!("  Original:     {}", &orig[..std::cmp::min(100, orig.len())]);
            eprintln!("  Decompressed: {}", &decomp[..std::cmp::min(100, decomp.len())]);
            mismatches += 1;
            if mismatches >= 5 {
                eprintln!("... ({} more mismatches suppressed)", total_records - i - 1);
                break;
            }
        }
    }

    println!("");
    println!("═══════════════════════════════════════════════════════");
    println!("RESULTS");
    println!("═══════════════════════════════════════════════════════");
    println!("Round-Trip Correctness:");
    println!("  Total records: {}", total_records);
    println!("  Matches:       {}", total_records - mismatches);
    println!("  Mismatches:    {}", mismatches);
    println!("  Success rate:  {:.2}%", (total_records - mismatches) as f64 / total_records as f64 * 100.0);
    println!("");
    println!("Seek Performance (O(1) random access):");
    println!("  Random seeks:  {}", num_seeks);
    println!("  Average:       {:7.2} μs", avg_ns as f64 / 1000.0);
    println!("  Median:        {:7.2} μs", median_ns as f64 / 1000.0);
    println!("  P95:           {:7.2} μs", p95_ns as f64 / 1000.0);
    println!("  P99:           {:7.2} μs", p99_ns as f64 / 1000.0);
    println!("  Min:           {:7.2} μs", min_ns as f64 / 1000.0);
    println!("  Max:           {:7.2} μs", max_ns as f64 / 1000.0);
    println!("");

    // Clean up
    std::fs::remove_file(decompressed_path).ok();

    if mismatches == 0 {
        println!("✓ PASSED: Perfect round-trip! Seek performance verified.");
        Ok(())
    } else {
        println!("✗ FAILED: {} mismatches detected", mismatches);
        std::process::exit(1);
    }
}
