//! A released NVIDIA ModelOpt NVFP4 checkpoint loads with its projections
//! packed.
//!
//! Run with `PMETAL_MODELOPT_DIR` set to the directory of a ModelOpt NVFP4
//! release (for example `nvidia/Qwen3.8-27B-NVFP4`). It reports how many
//! projections stayed packed, what the model holds in memory against what the
//! same projections would hold as dense bf16, and runs a forward pass.

use pmetal_models::DynamicModel;
use pmetal_runtime::compat::{Array, Dtype, VisitLinears};

#[test]
#[ignore = "needs a ModelOpt NVFP4 checkpoint on disk and a lot of memory"]
fn a_released_modelopt_checkpoint_stays_packed() {
    let dir = std::env::var("PMETAL_MODELOPT_DIR").expect("PMETAL_MODELOPT_DIR");
    let before = pmetal_runtime::inline_array::get_active_memory();
    let mut model = DynamicModel::load(std::path::Path::new(&dir)).expect("load");
    let after = pmetal_runtime::inline_array::get_active_memory();

    let (mut packed, mut dense_layers) = (0usize, 0usize);
    let (mut packed_bytes, mut as_bf16_bytes) = (0usize, 0usize);
    model.visit_linears_mut("", &mut |_, linear| {
        let (out, input) = linear.shape();
        if let Some(quant) = &linear.quant {
            packed += 1;
            packed_bytes += linear.weight.value.nbytes()
                + quant.scales.nbytes()
                + quant.tensor_scale.as_ref().map_or(0, |s| s.nbytes());
            as_bf16_bytes += (out as usize) * (input as usize) * 2;
        } else {
            dense_layers += 1;
        }
    });
    let gb = |bytes: usize| bytes as f64 / 1e9;
    println!(
        "{packed} projections packed, {dense_layers} dense; the packed ones hold {:.2} GB \
         (as dense bf16 they would hold {:.2} GB); the model holds {:.2} GB",
        gb(packed_bytes),
        gb(as_bf16_bytes),
        gb(after.saturating_sub(before)),
    );
    assert!(packed > 0, "nothing stayed packed");
    // Some projections of a release are per-tensor FP8 (one byte a weight)
    // beside the NVFP4 ones (half a byte).
    assert!(packed_bytes * 2 < as_bf16_bytes);

    let ids: Vec<i32> = (0..16).map(|i| (i * 131 + 7) % 1000).collect();
    let logits = model
        .forward(&Array::from_slice(&ids, &[1, 16]), None)
        .expect("forward");
    let last = logits
        .as_dtype(Dtype::Float32.as_i32())
        .abs()
        .max(None)
        .item_f32();
    assert!(last.is_finite() && last > 0.0, "logits are {last}");
    pmetal_runtime::check_unobserved_error().expect("no runtime error");
    println!(
        "forward ok: largest |logit| {last:.2}; peak {:.2} GB",
        gb(pmetal_runtime::inline_array::get_peak_memory())
    );

    // Against the same bytes unpacked to dense (it needs the memory of the
    // dense model, so it is asked for).
    if std::env::var("PMETAL_MODELOPT_COMPARE").is_ok() {
        use pmetal_models::architectures::qwen3_next::{
            Qwen3NextForCausalLM, Qwen3NextRoutedExpertMode,
        };
        use pmetal_models::loader::{Qwen3NextLoadOptions, load_qwen3_next_weights_with_options};
        let DynamicModel::Qwen3Next(inner) = &model else {
            panic!("the comparison is for a Qwen3.5-family model");
        };
        let config = inner.config().clone();
        let packed_logits = logits.as_dtype(Dtype::Float32.as_i32());
        packed_logits.eval();
        drop(model);
        let mut dense = Qwen3NextForCausalLM::new_with_routed_expert_mode(
            config.clone(),
            Qwen3NextRoutedExpertMode::Resident,
        )
        .expect("model");
        load_qwen3_next_weights_with_options(
            &mut dense,
            std::path::Path::new(&dir),
            &config,
            Qwen3NextLoadOptions {
                unpack_quantized_experts: true,
                ..Default::default()
            },
        )
        .expect("dense load");
        let want = dense
            .forward(&Array::from_slice(&ids, &[1, 16]), None)
            .expect("dense forward")
            .as_dtype(Dtype::Float32.as_i32());
        let worst = packed_logits.subtract(&want).abs().max(None).item_f32();
        let same = packed_logits
            .argmax(-1)
            .equal(&want.argmax(-1))
            .as_dtype(Dtype::Float32.as_i32())
            .min(None)
            .item_f32()
            == 1.0;
        println!(
            "against the dense unpacking: max |logit difference| {worst:.3e}, \
             same argmax at every position: {same}"
        );
        assert!(worst < 0.5, "packed and dense logits differ by {worst}");
    }
}
