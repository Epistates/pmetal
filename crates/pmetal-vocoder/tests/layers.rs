//! Convolution layers and AMP blocks against the reference.

mod common;

use common::{Fixture, assert_close, assert_runtime_ok, max_abs_diff, to_vec};
use pmetal_runtime::compat::{Array, ModuleParameters};
use pmetal_vocoder::nn::{AMPBlock, Conv1d, Conv1dConfig, ConvTranspose1d, StateDict};
use pmetal_vocoder::{ActivationType, ResBlockKind};

fn nlc(x: &Array) -> Array {
    x.transpose_axes(&[0, 2, 1])
}

fn conv(name: &str, config: Conv1dConfig, fx: &Fixture) -> Conv1d {
    let mut layer = Conv1d::new(config).unwrap();
    layer.load_state(&mut fx.state_dict(name), "").unwrap();
    layer
}

#[test]
fn conv1d_matches_reference_including_dilation_stride_and_groups() {
    let fx = Fixture::load("conv_layers");
    let x = nlc(&fx.array("conv_x"));
    let cases = [
        ("conv_a", Conv1dConfig::new(3, 4, 5).dilation(2).padding(4)),
        ("conv_b", Conv1dConfig::new(3, 4, 3).stride(2).padding(1)),
        ("conv_c", Conv1dConfig::new(3, 6, 7).padding(3).bias(false)),
        ("conv_d", Conv1dConfig::new(3, 6, 3).groups(3).padding(1)),
    ];
    for (name, config) in cases {
        let layer = conv(name, config, &fx);
        let y = nlc(&layer.forward(&x).unwrap());
        let key = format!("{name}.y");
        assert_close(&y, fx.shape(&key), fx.values(&key), 1e-4, 1e-5, name);
    }
}

#[test]
fn conv1d_fused_weight_norm_is_the_same_function() {
    let fx = Fixture::load("conv_layers");
    let x = nlc(&fx.array("conv_x"));
    let mut layer = conv(
        "conv_a",
        Conv1dConfig::new(3, 4, 5).dilation(2).padding(4),
        &fx,
    );
    assert!(layer.weight.is_normalized());
    layer.fuse();
    assert!(!layer.weight.is_normalized());
    let y = nlc(&layer.forward(&x).unwrap());
    assert_close(
        &y,
        fx.shape("conv_a.y"),
        fx.values("conv_a.y"),
        1e-4,
        1e-5,
        "fused conv_a",
    );
}

#[test]
fn conv_transpose1d_matches_reference_for_every_stride_and_kernel_shape() {
    let fx = Fixture::load("conv_layers");
    let x = nlc(&fx.array("convt_x"));
    // Includes kernels that are not a multiple of the stride (7/3, 10/4).
    for name in [
        "convt_a", "convt_b", "convt_c", "convt_d", "convt_e", "convt_f",
    ] {
        let meta: Vec<i32> = serde_json::from_str(fx.meta(name)).unwrap();
        let (cin, cout, k, s, p) = (meta[0], meta[1], meta[2], meta[3], meta[4]);
        let mut layer = ConvTranspose1d::new(cin, cout, k, s, p, true).unwrap();
        layer.load_state(&mut fx.state_dict(name), "").unwrap();
        let y = nlc(&layer.forward(&x).unwrap());
        let key = format!("{name}.y");
        assert_eq!(y.dim(2), layer.output_len(x.dim(1)), "{name}: output_len");
        assert_close(&y, fx.shape(&key), fx.values(&key), 1e-4, 1e-5, name);
    }
}

/// `padding = (kernel - stride) / 2` makes the output exactly `stride` times
/// as long, which is what ties the mel hop size to the upsampling rates.
#[test]
fn conv_transpose1d_with_bigvgan_geometry_scales_length_exactly() {
    for (k, s) in [(16, 8), (8, 4), (4, 2), (16, 4)] {
        let layer = ConvTranspose1d::new(4, 2, k, s, (k - s) / 2, true).unwrap();
        let x = Array::random_normal(&[1, 7, 4], pmetal_runtime::compat::Dtype::Float32);
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 7 * s, 2], "kernel {k} stride {s}");
    }
    assert_runtime_ok();
}

#[test]
fn state_round_trips_through_the_reference_layout() {
    let fx = Fixture::load("conv_layers");
    let layer = conv(
        "conv_a",
        Conv1dConfig::new(3, 4, 5).dilation(2).padding(4),
        &fx,
    );
    let mut saved = Vec::new();
    layer.save_state("conv_a", &mut saved);
    for (name, tensor) in &saved {
        let expected_shape = fx.shape(name);
        assert_eq!(tensor.shape(), expected_shape, "{name}");
        assert_eq!(
            max_abs_diff(&to_vec(tensor), fx.values(name)),
            0.0,
            "{name}"
        );
    }
    assert_eq!(saved.len(), 3, "weight_v, weight_g, bias");
}

#[test]
fn strict_loading_rejects_missing_and_misshapen_tensors() {
    let fx = Fixture::load("conv_layers");
    // Wrong channel count for the checkpoint.
    let mut wrong = Conv1d::new(Conv1dConfig::new(3, 5, 5).dilation(2).padding(4)).unwrap();
    assert!(wrong.load_state(&mut fx.state_dict("conv_a"), "").is_err());
    // An empty dict is missing everything.
    let mut layer = Conv1d::new(Conv1dConfig::new(3, 4, 5)).unwrap();
    assert!(layer.load_state(&mut StateDict::default(), "x").is_err());
}

#[test]
fn parameters_use_flat_reference_names() {
    let layer = Conv1d::new(Conv1dConfig::new(2, 3, 3)).unwrap();
    let mut names: Vec<String> = layer.parameters().keys().map(|k| k.to_string()).collect();
    names.sort();
    assert_eq!(names, ["bias", "weight_g", "weight_v"]);
    let mut fused = Conv1d::new(Conv1dConfig::new(2, 3, 3)).unwrap();
    fused.fuse();
    let mut names: Vec<String> = fused.parameters().keys().map(|k| k.to_string()).collect();
    names.sort();
    assert_eq!(names, ["bias", "weight"]);
}

fn amp_block(name: &str, kind: ResBlockKind) -> (AMPBlock, Array, Array, Fixture) {
    let fx = Fixture::load(name);
    let activation: ActivationType = serde_json::from_str(fx.meta("activation")).unwrap();
    let logscale: bool = serde_json::from_str(fx.meta("logscale")).unwrap();
    let mut block = AMPBlock::new(kind, 6, 5, &[1, 2, 3], activation, logscale).unwrap();
    block.load_state(&mut fx.state_dict("block"), "").unwrap();
    let x = nlc(&fx.array("x"));
    let y = nlc(&block.forward(&x).unwrap());
    (block, x, y, fx)
}

#[test]
fn amp_block1_matches_reference() {
    let (_, _, y, fx) = amp_block("amp_block1", ResBlockKind::One);
    assert_close(&y, fx.shape("y"), fx.values("y"), 1e-4, 1e-4, "AMPBlock1");
}

#[test]
fn amp_block2_matches_reference() {
    let (_, _, y, fx) = amp_block("amp_block2", ResBlockKind::Two);
    assert_close(&y, fx.shape("y"), fx.values("y"), 1e-4, 1e-4, "AMPBlock2");
}

#[test]
fn amp_block_layout_follows_the_kind() {
    let one = AMPBlock::new(
        ResBlockKind::One,
        4,
        3,
        &[1, 3, 5],
        ActivationType::SnakeBeta,
        true,
    )
    .unwrap();
    assert_eq!(
        (one.convs1.len(), one.convs2.len(), one.activations.len()),
        (3, 3, 6)
    );
    let two = AMPBlock::new(
        ResBlockKind::Two,
        4,
        3,
        &[1, 3, 5],
        ActivationType::Snake,
        true,
    )
    .unwrap();
    assert_eq!(
        (two.convs1.len(), two.convs2.len(), two.activations.len()),
        (3, 0, 3)
    );
}
