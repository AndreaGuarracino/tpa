use lib_bpaf::{
    compress_paf_with_tracepoints, ComplexityMetric, CompressionLayer, CompressionStrategy,
    TracepointType,
};
use lib_wfa2::affine_wavefront::Distance;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: auto_check <input.paf> <output.bpaf>");
        std::process::exit(1);
    }

    compress_paf_with_tracepoints(
        &args[1],
        &args[2],
        CompressionStrategy::AutomaticSlow(3),
        CompressionLayer::Zstd,
        TracepointType::Standard,
        32,
        ComplexityMetric::EditDistance,
        Distance::GapAffine {
            mismatch: 8,
            gap_opening: 5,
            gap_extension: 2,
        },
    )?;

    Ok(())
}
