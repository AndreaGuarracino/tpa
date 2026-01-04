use std::env;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use tpa::{is_tpa_file, varint_size, StringTable, TpaHeader};

/// Structure to hold detailed size analysis of a TPA file
#[derive(Debug)]
struct TpaSizeAnalysis {
    // File sections in bytes (as stored in the file)
    total_file_size: u64,
    header_size: u64,
    string_table_size: u64,
    records_section_size: u64,

    // Metadata
    num_records: u64,
    num_strings: u64,
    all_records_mode: bool,

    // String table breakdown
    string_name_length_varints: u64,
    string_name_bytes: u64,
    sequence_length_varints: u64,
}

impl TpaSizeAnalysis {
    fn format_percentage(pct: f64) -> String {
        if pct < 0.0001 && pct > 0.0 {
            "< 0.0001%".to_string()
        } else {
            format!("{:>7.4}%", pct)
        }
    }

    fn print_report(&self, header: &TpaHeader, magic: &[u8; 4]) {
        // Calculate header field sizes
        let magic_size = 4u64; // "TPA\0" magic bytes
        let version_size = 1u64; // version byte
        let layer_size = 0u64; // packed into strategy bytes (bits 7-6)
        let strategy_size = 2u64; // first + second strategy codes (with layer bits)
        let num_records_varint = varint_size(self.num_records);
        let num_strings_varint = varint_size(self.num_strings);
        let tp_type_size = 1u64; // TracepointType byte
        let complexity_metric_size = 1u64; // ComplexityMetric byte
        let max_complexity_varint = varint_size(header.max_complexity() as u64);
        let distance_size = match header.distance() {
            tpa::Distance::Edit => 1u64,
            tpa::Distance::GapAffine { .. } => 1 + 3 * 4, // code + 3 i32 values
            tpa::Distance::GapAffine2p { .. } => 1 + 5 * 4, // code + 5 i32 values
        };

        let magic_str = String::from_utf8_lossy(magic);
        println!("=== TPA Header ==============");
        println!(
            "  Magic bytes:                {:>12} bytes - value: \"{}\"",
            magic_size, magic_str
        );
        println!(
            "  Version:                    {:>12} bytes - value: {}",
            version_size,
            header.version()
        );
        println!(
            "  All-records mode:           {:>12} - value: {}",
            "", // no size, just metadata
            if self.all_records_mode { "YES" } else { "NO" }
        );
        println!(
            "  First layer (packed):       {:>12} bytes - value: {}",
            layer_size,
            header.first_layer().as_str()
        );
        println!(
            "  Second layer (packed):      {:>12} bytes - value: {}",
            layer_size,
            header.second_layer().as_str()
        );
        println!(
            "  Strategy codes:             {:>12} bytes - value: {:?}",
            strategy_size,
            header.strategies().unwrap_or_else(|e| {
                eprintln!("Warning: failed to decode strategies: {}", e);
                (
                    tpa::CompressionStrategy::Raw(3),
                    tpa::CompressionStrategy::Raw(3),
                )
            })
        );
        println!(
            "  Num records (varint):       {:>12} bytes - value: {}",
            num_records_varint, self.num_records
        );
        println!(
            "  Num strings (varint):       {:>12} bytes - value: {}",
            num_strings_varint, self.num_strings
        );
        println!(
            "  Tracepoint type:            {:>12} bytes - value: {:?}",
            tp_type_size,
            header.tp_type()
        );
        println!(
            "  Complexity metric:          {:>12} bytes - value: {:?}",
            complexity_metric_size,
            header.complexity_metric()
        );
        println!(
            "  Max complexity (varint):    {:>12} bytes - value: {}",
            max_complexity_varint,
            header.max_complexity()
        );
        println!(
            "  Distance mode:              {:>12} bytes - value: {:?}",
            distance_size,
            header.distance()
        );
        let header_pct = self.header_size as f64 / self.total_file_size as f64 * 100.0;
        println!(
            "  Total:                      {:>12} bytes ({} of the file)",
            self.header_size,
            Self::format_percentage(header_pct)
        );

        println!("\n=== String Table ==============");
        println!(
            "  String name len varints:    {:>12} bytes  (avg: {:.2})",
            self.string_name_length_varints,
            self.string_name_length_varints as f64 / self.num_strings as f64
        );
        println!(
            "  String name bytes:          {:>12} bytes  (avg: {:.2})",
            self.string_name_bytes,
            self.string_name_bytes as f64 / self.num_strings as f64
        );
        println!(
            "  Sequence len varints:       {:>12} bytes  (avg: {:.2})",
            self.sequence_length_varints,
            self.sequence_length_varints as f64 / self.num_strings as f64
        );
        let strtab_pct = self.string_table_size as f64 / self.total_file_size as f64 * 100.0;
        println!(
            "  Total:                      {:>12} bytes  ({} of the file)",
            self.string_table_size,
            Self::format_percentage(strtab_pct)
        );

        println!("\n=== Alignment Records ==============");
        let records_pct = self.records_section_size as f64 / self.total_file_size as f64 * 100.0;
        println!(
            "  Total size:                 {:>12} bytes  ({} of the file)",
            self.records_section_size,
            Self::format_percentage(records_pct)
        );
        println!(
            "  Avg bytes per record:       {:>12.2}",
            self.records_section_size as f64 / self.num_records as f64
        );

        println!("\n=== TOTAL ===");
        println!(
            "Total file size:              {:>12} bytes  (100.00%)",
            self.total_file_size
        );
    }
}

fn analyze_tpa_size(path: &str) -> io::Result<(TpaSizeAnalysis, TpaHeader, [u8; 4])> {
    let file = File::open(path)?;
    let total_file_size = file.metadata()?.len();
    analyze_tpa(path, total_file_size)
}

fn analyze_tpa(
    path: &str,
    total_file_size: u64,
) -> io::Result<(TpaSizeAnalysis, TpaHeader, [u8; 4])> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read magic bytes first
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;

    // Rewind to start
    reader.seek(io::SeekFrom::Start(0))?;

    // Read header and track position
    let start_pos = reader.stream_position()?;
    let header = TpaHeader::read(&mut reader)?;
    let header_end_pos = reader.stream_position()?;
    let header_size = header_end_pos - start_pos;

    // Read string table and track position
    let string_table_start = reader.stream_position()?;
    let num_strings = header.num_strings();
    let string_table = StringTable::read(&mut reader, num_strings)?;
    let string_table_end = reader.stream_position()?;
    let string_table_size = string_table_end - string_table_start;

    // Calculate string table breakdown
    let num_strings = header.num_strings();

    let mut string_name_length_varints = 0u64;
    let mut string_name_bytes = 0u64;
    let mut sequence_length_varints = 0u64;

    for i in 0..string_table.len() {
        let Some(name) = string_table.get(i as u64) else {
            continue;
        };
        let Some(seq_len) = string_table.get_length(i as u64) else {
            continue;
        };

        string_name_length_varints += varint_size(name.len() as u64);
        string_name_bytes += name.len() as u64;
        sequence_length_varints += varint_size(seq_len);
    }

    // Calculate records section by subtraction
    let records_section_size = total_file_size - header_size - string_table_size;

    let analysis = TpaSizeAnalysis {
        total_file_size,
        header_size,
        string_table_size,
        records_section_size,
        num_records: header.num_records(),
        num_strings,
        all_records_mode: header.all_records(),
        string_name_length_varints,
        string_name_bytes,
        sequence_length_varints,
    };

    Ok((analysis, header, magic))
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <file.tpa>", args[0]);
        eprintln!("\nAnalyzes a TPA file and reports detailed size breakdown.");
        std::process::exit(1);
    }

    let path = &args[1];

    if !is_tpa_file(path)? {
        eprintln!("Error: '{}' is not a valid TPA file", path);
        std::process::exit(1);
    }

    let (analysis, header, magic) = analyze_tpa_size(path)?;
    analysis.print_report(&header, &magic);

    Ok(())
}
