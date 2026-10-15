//! SSD-offloaded routed experts compute what resident ones do.
//!
//! The `qwen3_5_moe` parity fixture (four sparse layers of eight experts, two
//! per token, plus a shared expert) has its experts quantized to 4-bit affine
//! groups, packed into the per-layer files `pmetal pack-experts` writes, and
//! run two ways through `DynamicModel`:
//!
//! * resident: the stacked expert arrays hold the *dequantized* quantized
//!   weights, so the model computes with exactly the numbers the packed files
//!   hold;
//! * offloaded: the routed experts are read from the packed files into aligned
//!   Metal buffers and run by the fused expert kernels.
//!
//! Prefill (the windowed path), single-token decode and a prefill continued
//! after decode (a cache the offloaded layers must not disturb) are each
//! compared on logits. The two differ only in the order of float32 sums, so
//! the gate is tight; a wrong record offset, a swapped gate and up, a scale
//! applied to the wrong group or an expert read from the wrong slot moves the
//! logits by O(1).

// `set_var` is `unsafe`; the tests that set the slab size run one at a time.
#![allow(unsafe_code)]

mod common;

use std::collections::HashMap;
use std::path::Path;

use common::{fixture_path, load_shard, ref_tensor};
use pmetal_cache::test_utils::{ParityReport, Tolerance};
use pmetal_models::architectures::qwen3_next::Qwen3NextFeedForward;
use pmetal_models::dispatcher::{DynamicModel, DynamicModelLoadOptions};
use pmetal_models::expert_layout::{ExpertPackLayout, PackedBits};
use pmetal_models::expert_offload::SLAB_BYTES_ENV_VAR;
use pmetal_runtime::compat::{Array, Dtype, ops, ops::slice_axis};
use serial_test::serial;

const GROUP_SIZE: i32 = 32;
const BITS: i32 = 4;

/// float32 sums in a different order, through a model that is otherwise the
/// same. Wrong weights move the logits by O(1) (the logits' own magnitude is
/// about 3).
const TOL: Tolerance = Tolerance::new(5e-5, 0.0);

fn drain(op: &str) {
    pmetal_runtime::check_unobserved_error().unwrap_or_else(|e| panic!("{op}: runtime error: {e}"));
}

/// bf16 bits of an array whose elements are all exactly representable in bf16.
fn bf16_bits(array: &Array) -> Vec<u16> {
    let f32s = array.as_dtype(Dtype::Float32.as_i32());
    f32s.as_slice::<f32>()
        .iter()
        .map(|v| {
            assert_eq!(v.to_bits() & 0xFFFF, 0, "{v} is not a bf16 value");
            (v.to_bits() >> 16) as u16
        })
        .collect()
}

fn le_bytes_u32(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn le_bytes_u16(halves: &[u16]) -> Vec<u8> {
    halves.iter().flat_map(|h| h.to_le_bytes()).collect()
}

/// One quantized projection: the bytes of its three components and its
/// dequantized weight as float32.
struct Quantized {
    weight: Vec<u8>,
    scales: Vec<u8>,
    biases: Vec<u8>,
    dequantized: Array,
}

fn quantize(weight: &Array) -> Quantized {
    let bf16 = weight.as_dtype(Dtype::Bfloat16.as_i32());
    let (w, s, b) = bf16.quantize_weights(GROUP_SIZE, BITS);
    // Dequantized in float32 from the bf16 scales and biases the files hold:
    // the kernels compute `scale * q + bias` in float32, where a dequantize
    // in bf16 would round every weight to 8 bits of mantissa.
    let dequantized = w.dequantize(
        &s.as_dtype(Dtype::Float32.as_i32()),
        &b.as_dtype(Dtype::Float32.as_i32()),
        GROUP_SIZE,
        BITS,
    );
    Quantized {
        weight: le_bytes_u32(w.as_slice::<u32>()),
        scales: le_bytes_u16(&bf16_bits(&s)),
        biases: le_bytes_u16(&bf16_bits(&b)),
        dequantized,
    }
}

/// The dequantized stacks `[E, out, in]` of every layer's gate, up and down
/// projections, after packing the experts into `dir`.
struct PackedLayer {
    gate: Array,
    up: Array,
    down: Array,
}

/// Write `dir/layout.json` and one `layer_XX.bin` per sparse layer from the
/// fixture's fused `gate_up_proj` / `down_proj` stacks.
fn pack(weights: &HashMap<String, Array>, dir: &Path) -> Vec<PackedLayer> {
    let any = ref_tensor(
        weights,
        "model.language_model.layers.0.mlp.experts.down_proj",
    );
    let (experts, hidden, intermediate) = (any.dim(0), any.dim(1), any.dim(2));
    let layers = (0..)
        .take_while(|l| {
            weights.contains_key(&format!(
                "model.language_model.layers.{l}.mlp.experts.down_proj"
            ))
        })
        .count();

    let layout = ExpertPackLayout::new(
        "qwen3_5_moe_fixture".into(),
        layers,
        experts as usize,
        hidden as usize,
        intermediate as usize,
        GROUP_SIZE as usize,
        PackedBits::Four,
        (0..layers).collect(),
    );
    layout.save(dir).expect("layout.json");
    let record = &layout.record;

    let mut packed = Vec::new();
    for layer in 0..layers {
        let prefix = format!("model.language_model.layers.{layer}.mlp.experts");
        let gate_up = ref_tensor(weights, &format!("{prefix}.gate_up_proj"));
        let down = ref_tensor(weights, &format!("{prefix}.down_proj"));

        let mut file = vec![0u8; layout.expert_size * experts as usize];
        let (mut gates, mut ups, mut downs) = (Vec::new(), Vec::new(), Vec::new());
        for e in 0..experts {
            let expert = slice_axis(gate_up, 0, e, e + 1).reshape(&[2 * intermediate, hidden]);
            let parts = [
                quantize(&slice_axis(&expert, 0, 0, intermediate)),
                quantize(&slice_axis(&expert, 0, intermediate, 2 * intermediate)),
                quantize(&slice_axis(down, 0, e, e + 1).reshape(&[hidden, intermediate])),
            ];
            let base = e as usize * layout.expert_size;
            let components = [
                (&record.gate_weight, &parts[0].weight),
                (&record.gate_scales, &parts[0].scales),
                (&record.gate_biases, &parts[0].biases),
                (&record.up_weight, &parts[1].weight),
                (&record.up_scales, &parts[1].scales),
                (&record.up_biases, &parts[1].biases),
                (&record.down_weight, &parts[2].weight),
                (&record.down_scales, &parts[2].scales),
                (&record.down_biases, &parts[2].biases),
            ];
            for (component, bytes) in components {
                assert_eq!(bytes.len(), component.size, "component size");
                let at = base + component.offset;
                file[at..at + bytes.len()].copy_from_slice(bytes);
            }
            let [g, u, d] = parts;
            gates.push(g.dequantized);
            ups.push(u.dequantized);
            downs.push(d.dequantized);
        }
        std::fs::write(layout.layer_file_path(dir, layer), file).expect("layer file");

        let stack = |parts: Vec<Array>| ops::stack_axis(&parts, 0);
        packed.push(PackedLayer {
            gate: stack(gates),
            up: stack(ups),
            down: stack(downs),
        });
    }
    packed
}

struct Models {
    resident: DynamicModel,
    offloaded: DynamicModel,
    ids: Array,
    _checkpoint: tempfile::TempDir,
    _experts: tempfile::TempDir,
}

fn models() -> Models {
    let checkpoint = tempfile::tempdir().expect("tempdir");
    std::fs::copy(
        fixture_path("qwen3_5_moe_config.json"),
        checkpoint.path().join("config.json"),
    )
    .expect("copy config");
    std::fs::copy(
        fixture_path("qwen3_5_moe_weights.safetensors"),
        checkpoint.path().join("model.safetensors"),
    )
    .expect("copy weights");
    let weights = load_shard(&fixture_path("qwen3_5_moe_weights.safetensors"));
    let reference = load_shard(&fixture_path("qwen3_5_moe_reference.safetensors"));
    let ids = ref_tensor(&reference, "input_ids").clone();

    let experts = tempfile::tempdir().expect("tempdir");
    let packed = pack(&weights, experts.path());

    // Resident: the stacks hold what the packed files dequantize to.
    let mut resident = DynamicModel::load(checkpoint.path()).expect("resident loads");
    {
        let qwen = resident.as_qwen3_next_mut().expect("Qwen3Next");
        let mut sparse = 0;
        for (layer, packed) in qwen.model.layers.iter_mut().zip(&packed) {
            if let Qwen3NextFeedForward::MoE(block) = &mut layer.mlp {
                *block.switch_mlp_gate_proj = packed.gate.clone();
                *block.switch_mlp_up_proj = packed.up.clone();
                *block.switch_mlp_down_proj = packed.down.clone();
                sparse += 1;
            }
        }
        assert_eq!(sparse, packed.len(), "every packed layer is a sparse layer");
    }

    let mut offloaded = DynamicModel::load_with_options(
        checkpoint.path(),
        DynamicModelLoadOptions {
            prefer_expert_offload: true,
        },
    )
    .expect("offloaded loads");
    assert!(offloaded.requires_expert_offloading());
    offloaded
        .enable_expert_offloading(experts.path())
        .expect("offloading enables");
    assert!(!offloaded.requires_expert_offloading());
    drain("load");

    Models {
        resident,
        offloaded,
        ids,
        _checkpoint: checkpoint,
        _experts: experts,
    }
}

fn step(
    model: &mut DynamicModel,
    ids: &Array,
    kv: &mut pmetal_cache::kv_cache::KVCache,
    mamba: &mut pmetal_cache::kv_cache::MambaCache,
) -> Array {
    let logits = model
        .forward_with_hybrid_cache(ids, None, Some(kv), Some(mamba))
        .expect("forward");
    drain("forward");
    logits
}

/// Prefill, decode and a continuation of the offloaded model against the
/// resident one, with the slab at `slab_bytes` (the default when `None`).
fn compare(slab_bytes: Option<&str>) {
    // SAFETY: serial tests; nothing else reads the environment meanwhile.
    unsafe {
        match slab_bytes {
            Some(bytes) => std::env::set_var(SLAB_BYTES_ENV_VAR, bytes),
            None => std::env::remove_var(SLAB_BYTES_ENV_VAR),
        }
    }
    let Models {
        mut resident,
        mut offloaded,
        ids,
        ..
    } = models();
    // SAFETY: as above; the reader has read it by now.
    unsafe { std::env::remove_var(SLAB_BYTES_ENV_VAR) };
    let t = ids.dim(1);
    let caches = |model: &DynamicModel| {
        (
            model.create_cache(t as usize + 1),
            model.create_mamba_cache().expect("GDN cache"),
        )
    };
    let (mut kv_r, mut mamba_r) = caches(&resident);
    let (mut kv_o, mut mamba_o) = caches(&offloaded);

    let mut reports = Vec::new();
    // Prefill (more tokens than one slab of a few experts reaches), then
    // single tokens, then a multi-token continuation.
    let schedule: [(i32, i32); 4] = [(0, 21), (21, 22), (22, 23), (23, t)];
    for (start, end) in schedule {
        let chunk = slice_axis(&ids, 1, start, end);
        let want = step(&mut resident, &chunk, &mut kv_r, &mut mamba_r);
        let got = step(&mut offloaded, &chunk, &mut kv_o, &mut mamba_o);
        reports.push(ParityReport::compute(
            &format!("rows_{start}_{end}"),
            &got,
            &want,
            TOL,
        ));
    }
    println!();
    pmetal_cache::test_utils::print_report_table(&reports);
    let failed: Vec<&str> = reports
        .iter()
        .filter(|r| !r.passed())
        .map(|r| r.name.as_str())
        .collect();
    assert!(failed.is_empty(), "offloaded logits differ: {failed:?}");
}

#[test]
#[serial]
fn offloaded_experts_match_resident_experts() {
    compare(None);
}

/// A slab that holds a token's experts and little more: every pass runs in
/// small token windows, each reading its own experts, and the windows' outputs
/// join to the same result.
#[test]
#[serial]
fn offloaded_experts_in_small_slabs_match_resident_experts() {
    compare(Some("1"));
}
