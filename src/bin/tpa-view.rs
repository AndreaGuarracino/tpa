use std::env;
use std::io::{self, Write};
use std::process;
use tpa::TpaReader;

fn print_usage() {
    eprintln!("Usage: tpa-view [OPTIONS] <tpa-file>");
    eprintln!();
    eprintln!("Display TPA file contents:");
    eprintln!("  - Header information");
    eprintln!("  - String table");
    eprintln!("  - Alignment records in PAF format with tp:Z: tags");
    eprintln!();
    eprintln!("Options:");
    eprintln!(
        "  --strategies    Output first_strategy<TAB>second_strategy<TAB>first_layer<TAB>second_layer"
    );
}

fn strategy_to_name(strat: &str) -> String {
    // Convert Debug format to hyphenated lowercase
    // E.g., "ZigzagDelta" -> "zigzag-delta", "TwoDimDelta" -> "2d-delta"
    let name = strat.split('(').next().unwrap_or("unknown");

    match name {
        "Raw" => "raw",
        "ZigzagDelta" => "zigzag-delta",
        "TwoDimDelta" => "2d-delta",
        "RunLength" => "rle",
        "BitPacked" => "bit-packed",
        "DeltaOfDelta" => "delta-of-delta",
        "FrameOfReference" => "frame-of-reference",
        "HybridRLE" => "hybrid-rle",
        "XORDelta" => "xor-delta",
        "Dictionary" => "dictionary",
        "StreamVByte" => "stream-vbyte",
        "FastPFOR" => "fastpfor",
        "Cascaded" => "cascaded",
        "Simple8bFull" => "simple8b-full",
        "SelectiveRLE" => "selective-rle",
        "Rice" => "rice",
        "Huffman" => "huffman",
        "LZ77" => "lz77",
        _ => "unknown",
    }
    .to_string()
}

fn main() -> io::Result<()> {
    // Initialize logger
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    // Parse arguments
    let mut strategies_only = false;
    let mut tpa_path = None;

    for arg in args.iter().skip(1) {
        if arg == "--strategies" {
            strategies_only = true;
        } else if !arg.starts_with("--") {
            tpa_path = Some(arg.as_str());
        }
    }

    let Some(tpa_path) = tpa_path else {
        print_usage();
        process::exit(1);
    };

    // Open TPA file
    let reader = TpaReader::open(tpa_path)?;

    // If --strategies flag, output strategies and exit
    if strategies_only {
        let header = reader.header();
        let first_strat = header.first_strategy()?;
        let second_strat = header.second_strategy()?;

        let first_name = strategy_to_name(&format!("{:?}", first_strat));
        let second_name = strategy_to_name(&format!("{:?}", second_strat));
        let first_layer = header.first_layer().as_str();
        let second_layer = header.second_layer().as_str();

        println!(
            "{}\t{}\t{}\t{}",
            first_name, second_name, first_layer, second_layer
        );
        return Ok(());
    }

    let mut reader = TpaReader::open(tpa_path)?;

    // Load string table
    reader.load_string_table()?;

    let header = reader.header();
    let string_table = reader.string_table_ref().clone();

    // Print header
    println!("=== TPA Header ===");
    println!("Format version: {}", header.version());
    println!("Number of records: {}", header.num_records());
    println!("Number of unique strings: {}", header.num_strings());

    let (first_strategy, second_strategy) = header.strategies()?;
    println!(
        "Compression strategies: {} / {}",
        first_strategy, second_strategy
    );
    println!("First stream layer: {}", header.first_layer().as_str());
    println!("Second stream layer: {}", header.second_layer().as_str());

    println!("Tracepoint type: {:?}", header.tp_type());
    println!("Complexity metric: {:?}", header.complexity_metric());
    println!("Max complexity: {}", header.max_complexity());
    println!("Distance: {:?}", header.distance());
    println!();

    // Print string table
    println!("=== String Table ===");
    for i in 0..string_table.len() {
        if let Some(name) = string_table.get(i as u64) {
            if let Some(length) = string_table.get_length(i as u64) {
                println!("{}: {} (length: {})", i, name, length);
            }
        }
    }
    println!();

    // Print alignment records
    println!("=== Alignment Records ===");

    let stdout = io::stdout();
    let mut writer = io::BufWriter::new(stdout.lock());

    for record_id in 0..header.num_records() {
        let record = reader.get_alignment_record(record_id)?;

        let query_name = string_table
            .get(record.query_name_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid query name ID"))?;
        let query_len = string_table
            .get_length(record.query_name_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid query length"))?;
        let target_name = string_table
            .get(record.target_name_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid target name ID"))?;
        let target_len = string_table
            .get_length(record.target_name_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid target length"))?;

        // Write standard PAF fields
        write!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            query_name,
            query_len,
            record.query_start,
            record.query_end,
            record.strand,
            target_name,
            target_len,
            record.target_start,
            record.target_end,
            record.residue_matches,
            record.alignment_block_len,
            record.mapping_quality
        )?;

        // Write other tags
        for tag in &record.tags {
            let key = String::from_utf8_lossy(&tag.key);
            match &tag.value {
                tpa::TagValue::Int(v) => write!(writer, "\t{}:i:{}", key, v)?,
                tpa::TagValue::Float(v) => write!(writer, "\t{}:f:{}", key, v)?,
                tpa::TagValue::String(s) => write!(writer, "\t{}:Z:{}", key, s)?,
            }
        }

        // Write tracepoints as tp:Z: tag
        let tp_str = record.tracepoints.to_tp_tag();
        writeln!(writer, "\ttp:Z:{}", tp_str)?;
    }

    writer.flush()?;

    Ok(())
}
