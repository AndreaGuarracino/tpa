//! Integration tests for the TPA format: the `huffman` strategy (one code table per file, format
//! version 2) across every read path and layout, then malformed input and file lifecycle.

use flate2::write::GzEncoder;
use std::fs;
use std::io::{Cursor, Seek, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tpa::{
    bgzf, build_index_per_record, paf_to_tpa, read_standard_tracepoints_at_offset_with_strategies,
    read_standard_tracepoints_at_offset_with_tables, read_standard_tracepoints_at_vpos,
    read_standard_tracepoints_at_vpos_with_tables, read_varint, tpa_to_paf, ComplexityMetric,
    CompressionConfig, CompressionLayer, CompressionStrategy, Distance, HuffmanTables, TpaHeader,
    TpaIndex, TpaReader, TracepointData, TracepointType,
};

/// A fresh per-test directory, wiped if a previous run left one behind.
fn tmpdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("tpa_{}_{}", tag, std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}
/// Config with `strategy` on both streams; layer comes from the strategy string.
fn config(strategy: &str, tp_type: TracepointType) -> CompressionConfig {
    let (s, layer) = CompressionStrategy::from_str_with_layer(strategy).unwrap();
    CompressionConfig {
        first_strategy: s,
        second_strategy: s,
        first_layer: layer,
        second_layer: layer,
        tp_type,
        max_complexity: 100,
        complexity_metric: ComplexityMetric::EditDistance,
        distance: Distance::Edit,
        all_records: false,
        all_records_level: 6,
        contig_table: None,
    }
}

/// The tp:Z: tag of every line, in order: what must survive a round trip.
fn tp_tags(paf: &str) -> Vec<String> {
    paf.lines()
        .filter_map(|l| {
            l.split('\t')
                .find(|f| f.starts_with("tp:Z:"))
                .map(String::from)
        })
        .collect()
}

/// FASTGA-like advances: near-constant, which is where a shared table pays off.
/// `pairs` writes "diff,advance" items (Standard), otherwise single values (no-diff).
fn make_paf(path: &Path, records: usize, pairs: bool) {
    let mut out = String::new();
    let sequence_len = records * 5000;
    for i in 0..records {
        let tp: Vec<String> = (0..50)
            .map(|j| {
                let advance = match (i + j) % 7 {
                    0 => 97,
                    1 => 109,
                    2 => 92,
                    _ => 100,
                };
                if pairs {
                    format!("{},{}", (i + j) % 5, advance)
                } else {
                    advance.to_string()
                }
            })
            .collect();
        let start = i * 5000;
        out.push_str(&format!(
            "q{}\t{sequence_len}\t{}\t{}\t+\tt{}\t{sequence_len}\t{}\t{}\t4900\t5000\t60\ttp:Z:{}\n",
            i % 3,
            start,
            start + 5000,
            i % 2,
            start,
            start + 5000,
            tp.join(";")
        ));
    }
    fs::write(path, out).unwrap();
}

fn assert_rejected_without_truncation(input: &str, output: &Path, cfg: CompressionConfig) {
    fs::write(output, b"keep").unwrap();
    assert!(paf_to_tpa(input, output.to_str().unwrap(), cfg).is_err());
    assert_eq!(fs::read(output).unwrap(), b"keep");
}

#[test]
fn plain_parallel_output_is_thread_stable() {
    let dir = tmpdir("thread_stable");
    let paf = dir.join("in.paf");
    make_paf(&paf, 1000, true);

    for (tag, first, second) in [
        (
            "huffman",
            CompressionStrategy::Huffman(3),
            CompressionStrategy::Huffman(3),
        ),
        (
            "dual",
            CompressionStrategy::Rice(3),
            CompressionStrategy::TwoDimDelta(3),
        ),
    ] {
        let mut outputs = Vec::new();
        for threads in [1, 4] {
            let output = dir.join(format!("{tag}-{threads}.tpa"));
            let mut cfg = config("raw-nocomp", TracepointType::Standard);
            cfg.first_strategy = first;
            cfg.second_strategy = second;
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| paf_to_tpa(paf.to_str().unwrap(), output.to_str().unwrap(), cfg))
                .unwrap();
            TpaReader::new(output.to_str().unwrap()).unwrap();
            outputs.push((
                fs::read(&output).unwrap(),
                fs::read(output.with_extension("tpa.idx")).unwrap(),
            ));
        }
        assert_eq!(outputs[0], outputs[1]);
    }
    fs::remove_dir_all(dir).ok();
}

#[test]
fn benchmark_can_select_huffman_from_first_n_rows() {
    let dir = tmpdir("benchmark_huffman");
    let paf = dir.join("in.paf");
    make_paf(&paf, 400, true);

    let mut lines: Vec<_> = fs::read_to_string(&paf)
        .unwrap()
        .lines()
        .map(String::from)
        .collect();
    lines[200] = lines[200].replace("0,100", "0,777");
    fs::write(&paf, lines.join("\n") + "\n").unwrap();

    for all_records in [false, true] {
        let tpa = dir.join(format!("out-{all_records}.tpa"));
        let rt = dir.join(format!("roundtrip-{all_records}.paf"));
        let mut cfg = config("benchmark-nocomp,3,20", TracepointType::Standard);
        cfg.all_records = all_records;
        paf_to_tpa(paf.to_str().unwrap(), tpa.to_str().unwrap(), cfg).unwrap();

        let reader = TpaReader::new(tpa.to_str().unwrap()).unwrap();
        let strategies = reader.header().strategies().unwrap();
        assert!(
            matches!(strategies.1, CompressionStrategy::Huffman(_)),
            "selected {strategies:?}"
        );
        tpa_to_paf(tpa.to_str().unwrap(), rt.to_str().unwrap()).unwrap();
        assert_eq!(
            tp_tags(&fs::read_to_string(rt).unwrap()),
            tp_tags(&fs::read_to_string(&paf).unwrap())
        );
    }
}

#[test]
fn valid_cigar_converts_for_every_tracepoint_type() {
    let dir = tmpdir("cigar");
    let paf = dir.join("in.paf");
    fs::write(
        &paf,
        "q\t200\t0\t200\t+\tt\t200\t0\t198\t198\t200\t60\tcg:Z:100M2I98M\n",
    )
    .unwrap();

    for tp_type in [
        TracepointType::Standard,
        TracepointType::Fastga,
        TracepointType::FastgaNoDiff,
        TracepointType::Variable,
        TracepointType::Mixed,
    ] {
        let tpa = dir.join(format!("{tp_type:?}.tpa"));
        paf_to_tpa(
            paf.to_str().unwrap(),
            tpa.to_str().unwrap(),
            config("raw-nocomp", tp_type),
        )
        .unwrap();
        let reader = TpaReader::new(tpa.to_str().unwrap()).unwrap();
        assert_eq!(reader.header().num_records(), 1);
    }
    fs::remove_dir_all(dir).ok();
}

#[test]
fn invalid_huffman_modes_preserve_the_destination() {
    let dir = tmpdir("invalid");
    let paf = dir.join("in.paf");
    let output = dir.join("out.tpa");
    make_paf(&paf, 1, true);

    for tp_type in [TracepointType::Variable, TracepointType::Mixed] {
        assert_rejected_without_truncation(
            paf.to_str().unwrap(),
            &output,
            config("huffman-nocomp", tp_type),
        );
    }
    let mut zero_spacing = config("raw-nocomp", TracepointType::Fastga);
    zero_spacing.max_complexity = 0;
    assert_rejected_without_truncation(paf.to_str().unwrap(), &output, zero_spacing);

    let mut invalid_bgzip = config("raw-nocomp", TracepointType::Standard);
    invalid_bgzip.all_records = true;
    invalid_bgzip.all_records_level = 10;
    assert_rejected_without_truncation(paf.to_str().unwrap(), &output, invalid_bgzip);

    let mut mixed_mode = config("huffman-nocomp", TracepointType::Standard);
    mixed_mode.first_strategy = CompressionStrategy::Automatic(3);
    assert_rejected_without_truncation(paf.to_str().unwrap(), &output, mixed_mode);
    let mut mixed_mode = config("huffman-nocomp", TracepointType::Standard);
    mixed_mode.first_strategy = CompressionStrategy::Benchmark(3, 10, None);
    assert_rejected_without_truncation(paf.to_str().unwrap(), &output, mixed_mode);

    let mut invalid_benchmark = config("raw-nocomp", TracepointType::Standard);
    let benchmark = CompressionStrategy::Benchmark(100, 1, Some(CompressionLayer::Bgzip));
    invalid_benchmark.first_strategy = benchmark;
    invalid_benchmark.second_strategy = benchmark;
    assert_rejected_without_truncation(paf.to_str().unwrap(), &output, invalid_benchmark);

    let malformed = dir.join("malformed.paf");
    fs::write(
        &malformed,
        "q\t100\t0\t10\t+\tt\t100\t0\t10\t10\t10\t60\ttp:Z:\n",
    )
    .unwrap();
    assert_rejected_without_truncation(
        malformed.to_str().unwrap(),
        &output,
        config("huffman-nocomp", TracepointType::Standard),
    );
    fs::remove_dir_all(dir).ok();
}

#[test]
fn version2_no_diff_survives_every_read_path() {
    let dir = tmpdir("nodiff");
    let paf = dir.join("in.paf");
    let tpa = dir.join("huffman.tpa");
    let plain = dir.join("plain.tpa");
    make_paf(&paf, 400, false);
    let cfg = |s| config(s, TracepointType::FastgaNoDiff);

    paf_to_tpa(
        paf.to_str().unwrap(),
        tpa.to_str().unwrap(),
        cfg("huffman-nocomp"),
    )
    .unwrap();
    paf_to_tpa(
        paf.to_str().unwrap(),
        plain.to_str().unwrap(),
        cfg("local-huffman-nocomp"),
    )
    .unwrap();

    let (g, p) = (
        fs::metadata(&tpa).unwrap().len(),
        fs::metadata(&plain).unwrap().len(),
    );
    assert!(
        g < p,
        "file table {g} should be smaller than per-record {p}"
    );

    // 1. sequential decode reproduces the SOURCE tracepoints
    let rt = dir.join("huffman.rt.paf");
    tpa_to_paf(tpa.to_str().unwrap(), rt.to_str().unwrap()).unwrap();
    let source = fs::read_to_string(&paf).unwrap();
    assert_eq!(
        tp_tags(&fs::read_to_string(&rt).unwrap()),
        tp_tags(&source),
        "version-2 decode must reproduce the source tracepoints"
    );

    // 2. index rebuild must not misparse the table section as the string table
    let _ = fs::remove_file(dir.join("huffman.tpa.idx"));
    let index = build_index_per_record(tpa.to_str().unwrap()).unwrap();
    assert_eq!(index.len(), 400);

    // 3. random access through TpaReader
    let mut reader = TpaReader::new(tpa.to_str().unwrap()).unwrap();
    match &reader.get_compact_record(7).unwrap().tracepoints {
        TracepointData::FastgaNoDiff(v) => assert_eq!(v.len(), 50),
        other => panic!("unexpected payload: {other:?}"),
    }

    // 4. tables load standalone, for the seek APIs
    let tables = HuffmanTables::load(tpa.to_str().unwrap()).unwrap();
    assert!(tables.symbol_counts().0 > 0, "first table should exist");
    assert_eq!(tables.symbol_counts().1, 0, "no-diff has no second stream");

    // 5. the analyzer must parse the layout of a version-2 file
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_tpa-analyze"))
        .arg(tpa.to_str().unwrap())
        .output()
        .expect("tpa-analyze must run");
    assert!(out.status.success(), "tpa-analyze failed: {:?}", out.status);
    assert!(
        String::from_utf8_lossy(&out.stdout).contains(&g.to_string()),
        "analyzer should report the real file size {g}"
    );

    // 6. gzipped input: byte ranges are meaningless there, so it must take the sequential path
    let gz_path = dir.join("in.paf.gz");
    {
        use flate2::write::GzEncoder;
        use std::io::Write;
        let mut enc = GzEncoder::new(
            fs::File::create(&gz_path).unwrap(),
            flate2::Compression::fast(),
        );
        enc.write_all(source.as_bytes()).unwrap();
        enc.finish().unwrap();
    }
    let from_gz = dir.join("from_gz.tpa");
    paf_to_tpa(
        gz_path.to_str().unwrap(),
        from_gz.to_str().unwrap(),
        cfg("huffman-nocomp"),
    )
    .expect("huffman must work for gzipped input");
    assert_eq!(fs::read(&tpa).unwrap(), fs::read(&from_gz).unwrap());
    let rt_gz = dir.join("from_gz.rt.paf");
    tpa_to_paf(from_gz.to_str().unwrap(), rt_gz.to_str().unwrap()).unwrap();
    assert_eq!(
        tp_tags(&fs::read_to_string(&rt_gz).unwrap()),
        tp_tags(&source),
        "gzipped input must round-trip too"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn version2_standard_uses_both_tables() {
    let dir = tmpdir("standard");
    let paf = dir.join("in.paf");
    let tpa = dir.join("huffman.tpa");
    make_paf(&paf, 300, true);

    paf_to_tpa(
        paf.to_str().unwrap(),
        tpa.to_str().unwrap(),
        config("huffman-nocomp", TracepointType::Standard),
    )
    .unwrap();

    let tables = HuffmanTables::load(tpa.to_str().unwrap()).unwrap();
    let (first, second) = tables.symbol_counts();
    assert!(
        first > 0 && second > 0,
        "both tables expected, got {first}/{second}"
    );

    let rt = dir.join("rt.paf");
    tpa_to_paf(tpa.to_str().unwrap(), rt.to_str().unwrap()).unwrap();
    assert_eq!(
        tp_tags(&fs::read_to_string(&rt).unwrap()),
        tp_tags(&fs::read_to_string(&paf).unwrap()),
        "both streams must round-trip"
    );

    // standalone seek must decode the same values as TpaReader when given the tables
    let mut reader = TpaReader::new(tpa.to_str().unwrap()).unwrap();
    let expected = match &reader.get_compact_record(5).unwrap().tracepoints {
        TracepointData::Standard(v) | TracepointData::Fastga(v) => v.clone(),
        other => panic!("unexpected payload: {other:?}"),
    };
    let tp_offset = reader.get_tracepoint_offset(5).unwrap();
    let strategy = CompressionStrategy::from_str_with_layer("huffman-nocomp")
        .unwrap()
        .0;
    let mut file = fs::File::open(&tpa).unwrap();
    let via_seek = read_standard_tracepoints_at_offset_with_tables(
        &mut file,
        tp_offset,
        strategy,
        strategy,
        &tables,
        CompressionLayer::Nocomp,
        CompressionLayer::Nocomp,
    )
    .expect("standalone seek with tables must decode");
    assert_eq!(via_seek, expected, "seek API must match TpaReader");

    let mut file = fs::File::open(&tpa).unwrap();
    let before = file.stream_position().unwrap();
    assert!(read_standard_tracepoints_at_offset_with_strategies(
        &mut file,
        tp_offset,
        strategy,
        strategy,
        CompressionLayer::Nocomp,
        CompressionLayer::Nocomp,
    )
    .is_err());
    assert_eq!(file.stream_position().unwrap(), before);

    // A table may belong to only one stream; the absent marker must preserve alignment.
    let second_only = dir.join("second-only.tpa");
    let mut second_only_cfg = config("raw-nocomp", TracepointType::Standard);
    second_only_cfg.second_strategy = CompressionStrategy::from_str_with_layer("huffman-nocomp")
        .unwrap()
        .0;
    paf_to_tpa(
        paf.to_str().unwrap(),
        second_only.to_str().unwrap(),
        second_only_cfg,
    )
    .unwrap();
    let (first, second) = HuffmanTables::load(second_only.to_str().unwrap())
        .unwrap()
        .symbol_counts();
    assert_eq!(first, 0);
    assert!(second > 0);
    let second_only_rt = dir.join("second-only.rt.paf");
    tpa_to_paf(
        second_only.to_str().unwrap(),
        second_only_rt.to_str().unwrap(),
    )
    .unwrap();
    assert_eq!(
        tp_tags(&fs::read_to_string(second_only_rt).unwrap()),
        tp_tags(&fs::read_to_string(&paf).unwrap())
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn vpos_seek_needs_the_file_tables() {
    let dir = tmpdir("vpos");
    let paf = dir.join("in.paf");
    let tpa = dir.join("huffman.tpa");
    make_paf(&paf, 200, true);

    let mut cfg = config("huffman-nocomp", TracepointType::Standard);
    cfg.all_records = true;
    paf_to_tpa(paf.to_str().unwrap(), tpa.to_str().unwrap(), cfg).unwrap();

    let (first_strategy, second_strategy, first_layer, second_layer, vpos) = {
        let mut reader = TpaReader::new(tpa.to_str().unwrap()).unwrap();
        let vpos = reader.get_tracepoint_offset(7).unwrap();
        let h = reader.header();
        let (f, s) = h.strategies().unwrap();
        (f, s, h.first_layer(), h.second_layer(), vpos)
    };
    let tables = HuffmanTables::load(tpa.to_str().unwrap()).unwrap();

    let mut bgzf_reader = bgzf::io::Reader::new(fs::File::open(&tpa).unwrap());
    let tps = read_standard_tracepoints_at_vpos_with_tables(
        &mut bgzf_reader,
        vpos,
        first_strategy,
        second_strategy,
        &tables,
        first_layer,
        second_layer,
    )
    .expect("with tables the seek must decode");
    assert_eq!(tps.len(), 50);

    let mut bgzf_reader = bgzf::io::Reader::new(fs::File::open(&tpa).unwrap());
    assert!(
        read_standard_tracepoints_at_vpos(
            &mut bgzf_reader,
            vpos,
            first_strategy,
            second_strategy,
            first_layer,
            second_layer,
        )
        .is_err(),
        "without tables the seek must fail loudly"
    );

    fs::remove_dir_all(&dir).ok();
}

/// Every layout that supports the strategy, in both storage modes and every layer: the tables must
/// be written with the right shape, the file must round-trip, the reader must hand back the right
/// payload variant, and the file table must never be bigger than the per-record one.
/// Replaces four near-identical single-case tests.
#[test]
fn version2_round_trips_every_layout() {
    let dir = tmpdir("layouts");
    for (tp_type, pairs, layer, all_records, second_table) in [
        (TracepointType::Fastga, true, "nocomp", false, true),
        (TracepointType::Standard, true, "zstd", false, true),
        (TracepointType::Standard, true, "bgzip", false, true),
        (TracepointType::Standard, true, "nocomp", true, true),
        (TracepointType::FastgaNoDiff, false, "nocomp", true, false),
    ] {
        let tag = format!(
            "{tp_type:?}-{layer}-{}",
            if all_records { "all" } else { "per" }
        );
        let paf = dir.join(format!("{tag}.paf"));
        let tpa = dir.join(format!("{tag}.tpa"));
        let plain = dir.join(format!("{tag}.plain.tpa"));
        let rt = dir.join(format!("{tag}.rt.paf"));
        make_paf(&paf, 120, pairs);

        let mut cfg = config(&format!("huffman-{layer}"), tp_type);
        cfg.all_records = all_records;
        paf_to_tpa(paf.to_str().unwrap(), tpa.to_str().unwrap(), cfg).unwrap();
        let mut cfg_plain = config(&format!("local-huffman-{layer}"), tp_type);
        cfg_plain.all_records = all_records;
        paf_to_tpa(paf.to_str().unwrap(), plain.to_str().unwrap(), cfg_plain).unwrap();

        let (first, second) = HuffmanTables::load(tpa.to_str().unwrap())
            .unwrap()
            .symbol_counts();
        assert!(first > 0, "{tag}: first table missing");
        assert_eq!(second > 0, second_table, "{tag}: second table presence");

        tpa_to_paf(tpa.to_str().unwrap(), rt.to_str().unwrap()).unwrap();
        assert_eq!(
            tp_tags(&fs::read_to_string(&rt).unwrap()),
            tp_tags(&fs::read_to_string(&paf).unwrap()),
            "{tag}: round trip"
        );

        let mut reader = TpaReader::new(tpa.to_str().unwrap()).unwrap();
        let got = reader.get_compact_record(3).unwrap().tracepoints;
        match (tp_type, &got) {
            (TracepointType::Fastga, TracepointData::Fastga(_))
            | (TracepointType::Standard, TracepointData::Standard(_))
            | (TracepointType::FastgaNoDiff, TracepointData::FastgaNoDiff(_)) => {}
            other => panic!("{tag}: unexpected payload {other:?}"),
        }

        assert!(
            fs::metadata(&tpa).unwrap().len() <= fs::metadata(&plain).unwrap().len(),
            "{tag}: the file table must not make the file bigger"
        );
    }
    fs::remove_dir_all(dir).ok();
}

/// Two good records, then lines that must each be rejected: a non-numeric coordinate, a broken
/// tracepoint list, two malformed CIGARs, an out-of-range interval and an invalid strand.
fn write_input(path: &PathBuf) {
    let mut f = fs::File::create(path).unwrap();
    for (i, q) in [(0usize, "q1"), (1, "q2")] {
        let tp = (0..40)
            .map(|j| format!("{},{}", (i + j) % 4, 100))
            .collect::<Vec<_>>()
            .join(";");
        let s = i * 5000;
        writeln!(
            f,
            "{q}\t100000\t{s}\t{}\t+\tt1\t100000\t{s}\t{}\t3900\t4000\t60\ttp:Z:{tp}",
            s + 4000,
            s + 4000
        )
        .unwrap();
    }
    writeln!(
        f,
        "qBAD\t100000\tNOTANUMBER\t4000\t+\ttBAD\t100000\t0\t4000\t3900\t4000\t60\ttp:Z:1,100;1,100"
    )
    .unwrap();
    writeln!(
        f,
        "qBAD2\t100000\t0\t4000\t+\ttBAD2\t100000\t0\t4000\t3900\t4000\t60\ttp:Z:not-a-tracepoint"
    )
    .unwrap();
    for (name, cigar) in [
        ("qBAD3", "10M2Q"),
        ("qBAD4", "999999999999999999999999999999M"),
    ] {
        writeln!(
            f,
            "{name}\t100000\t0\t4000\t+\tt{name}\t100000\t0\t4000\t3900\t4000\t60\tcg:Z:{cigar}"
        )
        .unwrap();
    }
    writeln!(
        f,
        "qBAD5\t100\t0\t101\t+\ttBAD5\t100\t0\t10\t10\t10\t60\ttp:Z:1,10"
    )
    .unwrap();
    writeln!(
        f,
        "qBAD6\t100\t0\t10\t?\ttBAD6\t100\t0\t10\t10\t10\t60\ttp:Z:1,10"
    )
    .unwrap();
}

#[test]
fn a_skipped_line_leaves_no_record() {
    let dir = tmpdir("skip");
    let paf = dir.join("in.paf");
    write_input(&paf);

    // same input, gzipped
    let gz = dir.join("in.paf.gz");
    let mut encoder = GzEncoder::new(fs::File::create(&gz).unwrap(), flate2::Compression::fast());
    encoder.write_all(&fs::read(&paf).unwrap()).unwrap();
    encoder.finish().unwrap();

    for (tag, input, all_records) in [
        ("plain", &paf, false),
        ("gz", &gz, false),
        ("plain_ar", &paf, true),
        ("gz_ar", &gz, true),
    ] {
        for strategy in ["huffman-nocomp", "rice-nocomp"] {
            let out = dir.join(format!("{tag}_{strategy}.tpa"));
            let mut cfg = config(strategy, TracepointType::Standard);
            cfg.all_records = all_records;
            paf_to_tpa(input.to_str().unwrap(), out.to_str().unwrap(), cfg).unwrap();

            let reader = TpaReader::new(out.to_str().unwrap()).unwrap();
            let header = reader.header();
            assert_eq!(
                header.num_records(),
                2,
                "{tag}/{strategy}: the malformed line must not be counted"
            );
            assert_eq!(
                header.num_strings(),
                3,
                "{tag}/{strategy}: malformed-line names must not be interned"
            );
            drop(reader);

            // and the file must decode to exactly the two good records
            let rt = dir.join(format!("{tag}_{strategy}.rt.paf"));
            tpa_to_paf(out.to_str().unwrap(), rt.to_str().unwrap()).unwrap();
            let text = fs::read_to_string(&rt).unwrap();
            assert_eq!(text.lines().count(), 2, "{tag}/{strategy}");
            assert!(!text.contains("BAD"), "{tag}/{strategy}");
        }
    }

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn source_and_incomplete_files_are_preserved_or_rejected() {
    let dir = tmpdir("lifecycle");
    let paf = dir.join("in.paf");
    write_input(&paf);
    let source = fs::read(&paf).unwrap();
    assert!(paf_to_tpa(
        paf.to_str().unwrap(),
        paf.to_str().unwrap(),
        config("raw-nocomp", TracepointType::Standard)
    )
    .is_err());
    assert_eq!(fs::read(&paf).unwrap(), source);

    let tpa = dir.join("out.tpa");
    paf_to_tpa(
        paf.to_str().unwrap(),
        tpa.to_str().unwrap(),
        config("raw-nocomp", TracepointType::Standard),
    )
    .unwrap();
    let encoded = fs::read(&tpa).unwrap();
    assert!(tpa_to_paf(tpa.to_str().unwrap(), tpa.to_str().unwrap()).is_err());
    assert_eq!(fs::read(&tpa).unwrap(), encoded);

    let incomplete = dir.join("incomplete.tpa");
    fs::write(&incomplete, &encoded[..encoded.len() - 4]).unwrap();
    TpaReader::new_shallow(incomplete.to_str().unwrap()).unwrap();
    assert!(TpaReader::new(incomplete.to_str().unwrap()).is_err());
    assert!(TpaReader::new_streaming(incomplete.to_str().unwrap()).is_err());

    let mut trailing = encoded.clone();
    let trailer = trailing.len() - 4;
    let footer_len = u32::from_le_bytes(trailing[trailer..].try_into().unwrap());
    trailing.insert(trailer, 0);
    let trailer = trailing.len() - 4;
    trailing[trailer..].copy_from_slice(&(footer_len + 1).to_le_bytes());
    let trailing_footer = dir.join("trailing-footer.tpa");
    fs::write(&trailing_footer, trailing).unwrap();
    assert!(TpaReader::new(trailing_footer.to_str().unwrap()).is_err());

    fs::remove_dir_all(dir).ok();
}

#[test]
fn malformed_index_falls_back_to_sequential_decode() {
    let dir = tmpdir("index");
    let paf = dir.join("in.paf");
    let tpa = dir.join("in.tpa");
    let decoded = dir.join("decoded.paf");
    write_input(&paf);
    paf_to_tpa(
        paf.to_str().unwrap(),
        tpa.to_str().unwrap(),
        config("raw-nocomp", TracepointType::Standard),
    )
    .unwrap();
    let index = tpa.with_extension("tpa.idx");
    TpaIndex::new_raw(vec![0, 0])
        .save(index.to_str().unwrap())
        .unwrap();

    tpa_to_paf(tpa.to_str().unwrap(), decoded.to_str().unwrap()).unwrap();
    assert_eq!(fs::read_to_string(decoded).unwrap().lines().count(), 2);
    TpaReader::new(tpa.to_str().unwrap())
        .unwrap()
        .get_compact_record(1)
        .unwrap();

    TpaIndex::new_virtual_with_section_start(vec![0, 0], 0)
        .save(index.to_str().unwrap())
        .unwrap();
    TpaReader::new(tpa.to_str().unwrap())
        .unwrap()
        .get_compact_record(1)
        .unwrap();

    let mut oversized = b"TPAI\x02\x00".to_vec();
    oversized.extend_from_slice(&[0; 8]);
    oversized.extend_from_slice(&[0xff; 9]);
    oversized.push(1);
    fs::write(&index, oversized).unwrap();
    assert!(TpaIndex::load(index.to_str().unwrap()).is_err());
    TpaReader::new(tpa.to_str().unwrap()).unwrap();
    fs::remove_dir_all(dir).ok();
}

#[test]
fn stdin_compression_child() {
    let Ok(output) = std::env::var("TPA_STDIN_TEST_OUTPUT") else {
        return;
    };
    let strategy = std::env::var("TPA_STDIN_TEST_STRATEGY").unwrap();
    paf_to_tpa("-", &output, config(&strategy, TracepointType::Standard)).unwrap();
}

#[test]
fn stdin_is_spooled_for_two_pass_compression() {
    let dir = tmpdir("stdin");
    let paf = dir.join("in.paf");
    write_input(&paf);

    for strategy in ["raw-nocomp", "huffman-nocomp"] {
        let output = dir.join(format!("stdin-{strategy}.tpa"));
        let expected = dir.join(format!("file-{strategy}.tpa"));
        paf_to_tpa(
            paf.to_str().unwrap(),
            expected.to_str().unwrap(),
            config(strategy, TracepointType::Standard),
        )
        .unwrap();

        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "stdin_compression_child"])
            .env("TPA_STDIN_TEST_OUTPUT", &output)
            .env("TPA_STDIN_TEST_STRATEGY", strategy)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(&fs::read(&paf).unwrap())
            .unwrap();
        assert!(child.wait().unwrap().success());
        assert_eq!(fs::read(&output).unwrap(), fs::read(expected).unwrap());

        let reader = TpaReader::new(output.to_str().unwrap()).unwrap();
        assert_eq!(reader.header().num_records(), 2);
        assert_eq!(reader.header().num_strings(), 3);
    }
    fs::remove_dir_all(dir).ok();
}

fn strategy_offset(bytes: &[u8]) -> usize {
    let mut cursor = Cursor::new(bytes);
    cursor.set_position(5); // magic + version
    read_varint(&mut cursor).unwrap();
    read_varint(&mut cursor).unwrap();
    cursor.position() as usize
}

#[test]
fn malformed_table_layouts_are_rejected_early() {
    let dir = tmpdir("layout");
    let paf = dir.join("in.paf");
    write_input(&paf);

    let plain = dir.join("plain.tpa");
    paf_to_tpa(
        paf.to_str().unwrap(),
        plain.to_str().unwrap(),
        config("raw-nocomp", TracepointType::Standard),
    )
    .unwrap();
    let mut bytes = fs::read(&plain).unwrap();
    let strategy = strategy_offset(&bytes);
    bytes[strategy] = (bytes[strategy] & 0xc0) | 19;
    let v1_with_tables = dir.join("v1-with-tables.tpa");
    fs::write(&v1_with_tables, bytes).unwrap();
    assert!(TpaReader::new_shallow(v1_with_tables.to_str().unwrap()).is_err());

    let mut bytes = fs::read(&plain).unwrap();
    let strategy = strategy_offset(&bytes);
    bytes[strategy] |= 63;
    let unknown_strategy = dir.join("unknown-strategy.tpa");
    fs::write(&unknown_strategy, bytes).unwrap();
    assert!(TpaReader::new_shallow(unknown_strategy.to_str().unwrap()).is_err());

    let mut bytes = fs::read(&plain).unwrap();
    let mut cursor = Cursor::new(bytes.as_slice());
    TpaHeader::read(&mut cursor).unwrap();
    let flag = cursor.position() as usize - 1;
    bytes[flag] = 2;
    let invalid_flag = dir.join("invalid-flag.tpa");
    fs::write(&invalid_flag, bytes).unwrap();
    assert!(TpaReader::new_shallow(invalid_flag.to_str().unwrap()).is_err());

    let mut bytes = fs::read(&plain).unwrap();
    let max_complexity = strategy_offset(&bytes) + 4;
    bytes.splice(
        max_complexity..max_complexity + 1,
        [0xff; 9].into_iter().chain([1]),
    );
    let oversized_max = dir.join("oversized-max.tpa");
    fs::write(&oversized_max, bytes).unwrap();
    assert!(TpaReader::new_shallow(oversized_max.to_str().unwrap()).is_err());

    let tpa = dir.join("huffman.tpa");
    paf_to_tpa(
        paf.to_str().unwrap(),
        tpa.to_str().unwrap(),
        config("huffman-nocomp", TracepointType::Standard),
    )
    .unwrap();
    let mut bytes = fs::read(&tpa).unwrap();
    let strategy = strategy_offset(&bytes);
    bytes[strategy] &= 0xc0;
    bytes[strategy + 1] &= 0xc0;
    let v2_plain = dir.join("v2-plain.tpa");
    fs::write(&v2_plain, bytes).unwrap();
    assert!(TpaReader::new_shallow(v2_plain.to_str().unwrap()).is_err());

    let mut bytes = fs::read(&tpa).unwrap();
    let mut cursor = Cursor::new(bytes.as_slice());
    TpaHeader::read(&mut cursor).unwrap();
    let table_start = cursor.position() as usize;
    let table_len = read_varint(&mut cursor).unwrap() as usize;
    let table_end = cursor.position() as usize + table_len;
    bytes.splice(table_start..table_end, [0]);
    let missing = dir.join("missing-table.tpa");
    fs::write(&missing, bytes).unwrap();

    // Header-only open remains shallow; every table-aware open rejects the malformed section.
    TpaReader::new_shallow(missing.to_str().unwrap()).unwrap();
    assert!(HuffmanTables::load(missing.to_str().unwrap()).is_err());
    assert!(TpaReader::new(missing.to_str().unwrap()).is_err());

    fs::remove_dir_all(dir).ok();
}

#[test]
fn bgzip_level_is_rejected_outside_zero_to_nine() {
    let dir = tmpdir("level");
    let paf = dir.join("in.paf");
    let mut f = fs::File::create(&paf).unwrap();
    for i in 0..50u64 {
        let tp: Vec<String> = (0..20).map(|j| format!("{},{}", j % 3, 100)).collect();
        writeln!(
            f,
            "q{}\t100000\t{}\t{}\t+\tt0\t100000\t{}\t{}\t1900\t2000\t60\ttp:Z:{}",
            i % 4,
            i * 2000,
            i * 2000 + 2000,
            i * 2000,
            i * 2000 + 2000,
            tp.join(";")
        )
        .unwrap();
    }
    drop(f);
    for (level, want_ok) in [
        (0i32, true),
        (9, true),
        (10, false),
        (-1, false),
        (i32::MIN, false),
    ] {
        let out = dir.join(format!("l{level}.tpa"));
        let mut cfg = config("raw-nocomp", TracepointType::Standard);
        cfg.first_layer = CompressionLayer::Bgzip;
        cfg.second_layer = CompressionLayer::Bgzip;
        cfg.first_strategy = CompressionStrategy::Raw(level);
        cfg.second_strategy = CompressionStrategy::Raw(level);
        let got = paf_to_tpa(paf.to_str().unwrap(), out.to_str().unwrap(), cfg);
        assert_eq!(got.is_ok(), want_ok, "level {level}: {got:?}");
        if !want_ok {
            let msg = got.unwrap_err().to_string();
            assert!(msg.contains("between 0 and 9"), "level {level}: {msg}");
        }
    }
    fs::remove_dir_all(dir).ok();
}
