use lib_bpaf::{BpafReader, MixedRepresentation, TracepointData};
use std::env;
use std::io::{self, Write};
use std::process;

fn print_usage() {
    eprintln!("Usage: bpaf-view <bpaf-file>");
    eprintln!();
    eprintln!("Display BPAF file contents:");
    eprintln!("  - Header information");
    eprintln!("  - String table");
    eprintln!("  - Alignment records in PAF format with tp:Z: tags");
}

fn format_tracepoints(tp_data: &TracepointData) -> String {
    match tp_data {
        TracepointData::Standard(tps) | TracepointData::Fastga(tps) => tps
            .iter()
            .map(|(a, b)| format!("{},{}", a, b))
            .collect::<Vec<_>>()
            .join(";"),
        TracepointData::Variable(tps) => tps
            .iter()
            .map(|(a, b_opt)| {
                if let Some(b) = b_opt {
                    format!("{},{}", a, b)
                } else {
                    format!("{}", a)
                }
            })
            .collect::<Vec<_>>()
            .join(";"),
        TracepointData::Mixed(items) => items
            .iter()
            .map(|item| match item {
                MixedRepresentation::Tracepoint(a, b) => format!("{},{}", a, b),
                MixedRepresentation::CigarOp(len, op) => format!("{}{}", len, op),
            })
            .collect::<Vec<_>>()
            .join(";"),
    }
}

fn main() -> io::Result<()> {
    // Initialize logger
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        print_usage();
        process::exit(1);
    }

    let bpaf_path = &args[1];

    // Open BPAF file
    let mut reader = BpafReader::open(bpaf_path)?;

    // Load string table
    reader.load_string_table()?;

    let header = reader.header();
    let string_table = reader.string_table_ref().clone();

    // Print header
    println!("=== BPAF Header ===");
    println!("Format version: {}", header.version());
    println!("Number of records: {}", header.num_records());
    println!("Number of unique strings: {}", header.num_strings());

    let strategy = header.strategy()?;
    println!("Compression strategy: {}", strategy);

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
                lib_bpaf::TagValue::Int(v) => write!(writer, "\t{}:i:{}", key, v)?,
                lib_bpaf::TagValue::Float(v) => write!(writer, "\t{}:f:{}", key, v)?,
                lib_bpaf::TagValue::String(s) => write!(writer, "\t{}:Z:{}", key, s)?,
            }
        }

        // Write tracepoints as tp:Z: tag
        let tp_str = format_tracepoints(&record.tracepoints);
        writeln!(writer, "\ttp:Z:{}", tp_str)?;
    }

    writer.flush()?;

    Ok(())
}
