//! Real-weight parity of the FLUX.1 VAE against `diffusers`.
//!
//! `vae_flux_real_reference.safetensors` holds `AutoencoderKL`'s encoding of a
//! 64x64 test image and its decoding of a seeded 8x8x16 latent, computed in
//! float32 over the released weights. The released tensors are bfloat16, so
//! this loads them, widens them to float32 and compares at float32
//! accumulation order; a bfloat16 run is held to bf16 noise.
//!
//! Needs the VAE (~170 MB): set `PMETAL_FLUX_VAE_DIR` to the `vae` directory of
//! a FLUX.1 snapshot. Skipped, loudly, without it.

mod common;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance, print_report_table};
use pmetal_models::architectures::vae::{AutoencoderKL, VAEConfig};
use pmetal_models::loader::{load_vae_weights, load_weights};
use pmetal_runtime::compat::{Array, Dtype};

fn load(dtype: Dtype) -> Option<AutoencoderKL> {
    let Ok(dir) = std::env::var("PMETAL_FLUX_VAE_DIR") else {
        eprintln!("SKIPPED: set PMETAL_FLUX_VAE_DIR to a FLUX.1 `vae` directory");
        return None;
    };
    let raw = std::fs::read_to_string(std::path::Path::new(&dir).join("config.json"))
        .expect("the VAE has a config.json");
    let config: VAEConfig = serde_json::from_str(&raw).expect("config.json parses");
    let mut vae = AutoencoderKL::new(config).expect("the VAE builds");
    let weights = load_weights(&dir).expect("the VAE's weights read");
    let weights = weights
        .into_iter()
        .map(|(k, v)| (k, v.as_dtype(dtype.as_i32())))
        .collect();
    load_vae_weights(&mut vae, weights).expect("the released VAE loads strictly");
    Some(vae)
}

/// `tolerance` bounds the moments, `decoded_tolerance` the image.
fn check(dtype: Dtype, tolerance: Tolerance, decoded_tolerance: Tolerance) {
    let Some(mut vae) = load(dtype) else { return };
    let reference = load_shard(&fixture_path("vae_flux_real_reference.safetensors"));
    let cast = |name: &str| -> Array { ref_tensor(&reference, name).as_dtype(dtype.as_i32()) };

    let distribution = vae.encode_distribution(&cast("image")).expect("encode");
    let decoded = vae.decode_raw(&cast("z")).expect("decode");
    pmetal_runtime::check_unobserved_error().expect("no runtime op failed");

    let reports = [
        ParityReport::compute("mean", &distribution.mean, ref_tensor(&reference, "mean"), tolerance),
        ParityReport::compute("logvar", &distribution.logvar, ref_tensor(&reference, "logvar"), tolerance),
        ParityReport::compute(
            "decoded",
            &decoded,
            ref_tensor(&reference, "decoded"),
            decoded_tolerance,
        ),
    ];
    print_report_table(&reports);
    for report in &reports {
        assert!(report.passed(), "{dtype:?} {}: {report:?}", report.name);
        assert!(
            report.cosine_similarity > 0.998,
            "{dtype:?} {} points elsewhere: {report:?}",
            report.name
        );
    }
}

#[test]
fn the_released_flux_vae_matches_diffusers_in_float32() {
    check(
        Dtype::Float32,
        Tolerance::new(2e-3, 1e-4),
        Tolerance::new(2e-5, 2e-5),
    );
}

/// bfloat16 carries 8 bits of mantissa through 30-odd convolutions, so the
/// encoder's moments (magnitude 5, log-variance 29) move by tenths. The
/// reference run in bfloat16 lands 0.46 and 0.28 from its float32 one on this
/// image (0.011 on the decoded image); the bounds are about twice that.
#[test]
fn the_released_flux_vae_matches_diffusers_in_bfloat16() {
    check(
        Dtype::Bfloat16,
        Tolerance::new(1.0, 0.0),
        Tolerance::new(0.05, 0.0),
    );
}
