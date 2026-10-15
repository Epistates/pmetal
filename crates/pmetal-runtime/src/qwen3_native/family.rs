//! What a Qwen3.5-family `config.json` means: Qwen3-Next, Qwen3.5, Qwen3.6
//! and Qwen3.8, dense and MoE (`qwen3_next`, `qwen3_5*`, `qwen3_6*`).
//!
//! Both of pmetal's engines run this family, the native bridge here and the
//! `DynamicModel` path in `pmetal-models`, and each used to read the config
//! its own way: different defaults for absent keys, `layer_types` honoured by
//! one and ignored by the other, the head tie taken from a different object
//! than Hugging Face `transformers` uses. [`normalize_text_config`] is the one
//! reading both now parse, resolved the way the reference implementation
//! resolves it, with every value the engines cannot run refused by name
//! instead of defaulted.

use serde_json::{Map, Value};

use crate::rope::{RopeConfig, RopeScaling, Rotary};

/// Activation on the gated-delta-net output gate, `rms_norm(o) * act(z)`.
///
/// `output_gate_type` in the text config names it. transformers' `qwen3_5`
/// hard-codes SiLU there and does not read the key; its `qwen4_exp` reads it
/// as `output_gate_type or hidden_act`. `"swish"` (what Qwen3.6 and 3.8 ship)
/// is SiLU under another name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GdnGateActivation {
    /// `z * sigmoid(z)`.
    #[default]
    Silu,
    /// `sigmoid(z)`.
    Sigmoid,
}

impl GdnGateActivation {
    /// Resolve the gate from `output_gate_type`, falling back to `hidden_act`
    /// when the key is absent or `null`.
    pub fn resolve(
        output_gate_type: Option<&str>,
        hidden_act: Option<&str>,
    ) -> Result<Self, String> {
        let name = output_gate_type.or(hidden_act).unwrap_or("silu");
        match name.to_ascii_lowercase().as_str() {
            "silu" | "swish" => Ok(Self::Silu),
            "sigmoid" => Ok(Self::Sigmoid),
            other => Err(format!(
                "unsupported gated-delta-net output gate activation {other:?} \
                 (output_gate_type); supported: \"silu\"/\"swish\", \"sigmoid\""
            )),
        }
    }

    /// The canonical config spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Silu => "silu",
            Self::Sigmoid => "sigmoid",
        }
    }
}

/// The L2-norm epsilon transformers applies to gated-delta-net queries and
/// keys (`l2norm(x, eps=1e-6)`: `x * rsqrt(sum(x²) + eps)`).
pub const GDN_QK_L2NORM_EPS: f32 = 1e-6;

/// The `rms_norm` epsilon that makes pmetal's fused query/key normalization
/// equal transformers' L2 norm.
///
/// pmetal folds the L2 norm and the `1/sqrt(Dk)` query scale into one
/// `rms_norm` with a constant weight. `rms_norm` divides by
/// `sqrt(mean(x²) + eps)`, which is `sqrt((sum(x²) + Dk·eps) / Dk)`, so the
/// same `eps` as the L2 norm lands `Dk` times too large inside the root.
/// Passing `eps / Dk` makes the two identical.
pub fn gdn_qk_rms_norm_eps(head_k_dim: i32) -> f32 {
    GDN_QK_L2NORM_EPS / head_k_dim.max(1) as f32
}

/// The width the gated-delta-net Metal kernel needs the key head dimension to
/// be a multiple of: each of its 32 lanes owns `Dk / 32` state columns, so a
/// smaller `Dk` compiles to a zero-length array and the kernel never builds.
pub const GDN_KERNEL_DK_MULTIPLE: i64 = 32;

/// Key/tensor name a checkpoint uses, reduced to the canonical text-model
/// layout: the vision-language wrappers (`model.language_model.`,
/// `language_model.model.`, `language_model.`) to `model.`/bare, and `A_log`
/// to `a_log`.
pub fn canonical_checkpoint_key(key: &str) -> String {
    let mut key = if let Some(rest) = key.strip_prefix("language_model.model.") {
        format!("model.{rest}")
    } else if let Some(rest) = key.strip_prefix("language_model.") {
        rest.to_string()
    } else if let Some(rest) = key.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else {
        key.to_string()
    };
    if key.contains(".A_log") {
        key = key.replace(".A_log", ".a_log");
    }
    key
}

/// Tensors a Qwen3.5-family checkpoint carries that the text model does not
/// consume: the vision tower (`model.visual.*`, `visual.*`) and the bundled
/// multi-token predictor (`mtp.*`), which `--mtp` loads on its own.
pub fn is_unused_by_text_model(key: &str) -> bool {
    let key = canonical_checkpoint_key(key);
    key.starts_with("model.visual.")
        || key.starts_with("visual.")
        || key.starts_with("mtp.")
        || key.starts_with("model.mtp.")
}

/// The rotary embedding of a text config [`normalize_text_config`] returned:
/// partial, at `rope_theta`, and YaRN-scaled when `rope_parameters` says so.
pub fn rotary(text: &Value) -> Result<Rotary, String> {
    let head_dim = text
        .get("head_dim")
        .and_then(Value::as_i64)
        .ok_or("config key `head_dim` must be an integer")?;
    Rotary::from_config(head_dim as i32, RopeConfig::from_json(text), 10_000.0, 0.25)
}

/// The `rope_parameters` of a text config, resolved into the canonical object
/// both engines read: `rope_type` (`"default"` or `"yarn"`), `rope_theta`,
/// `partial_rotary_factor`, the mRoPE keys, and for YaRN every parameter
/// [`crate::rope`] resolved (defaults filled in, `attention_factor`
/// computed). A legacy `rope_scaling` stands in when `rope_parameters` is
/// absent.
fn resolve_rope_parameters(
    text: &Map<String, Value>,
    rope_theta: f64,
    partial: f64,
) -> Result<Map<String, Value>, String> {
    let source = text
        .get("rope_parameters")
        .filter(|v| !v.is_null())
        .or_else(|| text.get("rope_scaling").filter(|v| !v.is_null()));
    let mut rope = match source {
        Some(Value::Object(map)) => map.clone(),
        Some(other) => return Err(format!("rope_parameters must be an object, got {other}")),
        None => Map::new(),
    };
    let max_pos = text.get("max_position_embeddings").and_then(Value::as_f64);
    let scaling = RopeScaling::from_params(
        Some(&rope),
        &RopeConfig {
            max_position_embeddings: max_pos,
            original_max_position_embeddings: text
                .get("original_max_position_embeddings")
                .and_then(Value::as_f64),
            ..RopeConfig::default()
        },
    )?;
    rope.remove("type");
    rope.insert("rope_theta".into(), rope_theta.into());
    rope.insert("partial_rotary_factor".into(), partial.into());
    match scaling {
        RopeScaling::Default => {
            rope.insert("rope_type".into(), "default".into());
        }
        RopeScaling::Yarn(yarn) => {
            rope.insert("rope_type".into(), "yarn".into());
            rope.insert("factor".into(), yarn.factor.into());
            rope.insert(
                "original_max_position_embeddings".into(),
                yarn.original_max_position_embeddings.into(),
            );
            rope.insert("attention_factor".into(), yarn.attention_factor.into());
            rope.insert("beta_fast".into(), yarn.beta_fast.into());
            rope.insert("beta_slow".into(), yarn.beta_slow.into());
            rope.insert("truncate".into(), yarn.truncate.into());
        }
        other => {
            return Err(format!(
                "rope_parameters.rope_type {:?} is not supported; this family runs the \
                 default (partial, multimodal-sectioned) rotary embedding and static \"yarn\"",
                other.rope_type()
            ));
        }
    }
    Ok(rope)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    /// `Qwen3NextConfig`.
    Next,
    /// `Qwen3_5TextConfig` (dense).
    Dense,
    /// `Qwen3_5MoeTextConfig`.
    Moe,
}

impl Family {
    fn of(model_type: &str) -> Option<Self> {
        let mt = model_type.to_ascii_lowercase();
        let mt = mt.strip_suffix("_mtp").unwrap_or(&mt);
        match mt {
            "qwen3_next" => Some(Self::Next),
            "qwen3_5" | "qwen3.5" | "qwen3_5_text" | "qwen3_6" | "qwen3.6" | "qwen3_6_text" => {
                Some(Self::Dense)
            }
            "qwen3_5_moe" | "qwen3_5_moe_text" | "qwen3_6_moe" | "qwen3_6_moe_text" => {
                Some(Self::Moe)
            }
            _ => None,
        }
    }

    /// transformers' defaults for the keys pmetal reads, per config class.
    fn defaults(self) -> Vec<(&'static str, Value)> {
        let mut d: Vec<(&'static str, Value)> = vec![
            ("hidden_act", "silu".into()),
            ("rms_norm_eps", 1e-6.into()),
            ("attention_bias", false.into()),
            ("head_dim", 256.into()),
            ("linear_conv_kernel_dim", 4.into()),
            ("linear_key_head_dim", 128.into()),
            ("linear_value_head_dim", 128.into()),
            ("linear_num_key_heads", 16.into()),
            ("linear_num_value_heads", 32.into()),
            ("num_attention_heads", 16.into()),
            ("max_position_embeddings", 32768.into()),
        ];
        match self {
            Self::Next => d.extend([
                ("vocab_size", 151_936.into()),
                ("hidden_size", 2048.into()),
                ("intermediate_size", 5632.into()),
                ("num_hidden_layers", 48.into()),
                ("num_key_value_heads", 2.into()),
                ("decoder_sparse_step", 1.into()),
                ("moe_intermediate_size", 512.into()),
                ("shared_expert_intermediate_size", 512.into()),
                ("num_experts_per_tok", 10.into()),
                ("num_experts", 512.into()),
                ("norm_topk_prob", true.into()),
                ("mlp_only_layers", Value::Array(Vec::new())),
            ]),
            Self::Dense => d.extend([
                ("vocab_size", 248_320.into()),
                ("hidden_size", 4096.into()),
                ("intermediate_size", 12_288.into()),
                ("num_hidden_layers", 32.into()),
                ("num_key_value_heads", 4.into()),
            ]),
            Self::Moe => d.extend([
                ("vocab_size", 248_320.into()),
                ("hidden_size", 2048.into()),
                ("num_hidden_layers", 40.into()),
                ("num_key_value_heads", 2.into()),
                ("moe_intermediate_size", 512.into()),
                ("shared_expert_intermediate_size", 512.into()),
                ("num_experts_per_tok", 8.into()),
                ("num_experts", 256.into()),
                ("norm_topk_prob", true.into()),
            ]),
        }
        d
    }
}

/// `true` for every `model_type` this module normalizes.
pub fn is_qwen35_family(model_type: &str) -> bool {
    Family::of(model_type).is_some()
}

fn int(obj: &Map<String, Value>, key: &str) -> Result<i64, String> {
    obj.get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("config key `{key}` must be an integer"))
}

/// The text config of a Qwen3.5-family `config.json`, resolved the way
/// transformers resolves it and checked against what pmetal can run.
///
/// `config` is the whole file, flat (`*ForCausalLM`) or nested under
/// `text_config` (`*ForConditionalGeneration`, what Qwen ships). The returned
/// object is the text config with:
///
/// * every key pmetal reads present, absent ones filled with the reference
///   class's default rather than each engine's own,
/// * `tie_word_embeddings` from the outer config when nested: that is the
///   flag transformers ties the head by, and the text one is ignored,
/// * `layer_types` expanded from `full_attention_interval` when absent,
/// * `rope_theta` and `partial_rotary_factor` promoted from `rope_parameters`,
///   which wins over the legacy top-level keys as it does in transformers,
/// * `rope_parameters` itself resolved (a legacy `rope_scaling` standing in
///   when it is absent): `rope_type` `"default"` or `"yarn"`, and for YaRN
///   every parameter filled in, `attention_factor` computed (see [`Rotary`]),
/// * `output_gate_type` reduced to `"silu"` or `"sigmoid"`.
///
/// Refused, by name: a `hidden_act` other than SiLU, an unknown
/// `output_gate_type`, `attn_output_gate: false`, a RoPE type other than the
/// default or YaRN, unknown or miscounted `layer_types`, head counts that do not
/// divide, and a gated-delta-net key head dimension the Metal kernel cannot
/// build for.
pub fn normalize_text_config(config: &Value) -> Result<Value, String> {
    let outer = config
        .as_object()
        .ok_or("config.json must be a JSON object")?;
    let nested = outer.get("text_config").filter(|v| v.is_object());
    let mut text = nested.and_then(Value::as_object).unwrap_or(outer).clone();
    if !text.contains_key("model_type")
        && let Some(mt) = outer.get("model_type")
    {
        text.insert("model_type".into(), mt.clone());
    }
    let model_type = text
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("qwen3_5_text")
        .to_string();
    let family = Family::of(&model_type)
        .ok_or_else(|| format!("{model_type:?} is not a Qwen3.5-family model_type"))?;

    for (key, default) in family.defaults() {
        if text.get(key).is_none_or(Value::is_null) {
            text.insert(key.into(), default);
        }
    }

    // Head tie: the outer flag governs a nested config, defaulting to false.
    let tie = nested
        .and(outer.get("tie_word_embeddings"))
        .or_else(|| text.get("tie_word_embeddings"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    text.insert("tie_word_embeddings".into(), tie.into());

    let hidden_act = text
        .get("hidden_act")
        .and_then(Value::as_str)
        .unwrap_or("silu")
        .to_string();
    if !matches!(hidden_act.to_ascii_lowercase().as_str(), "silu" | "swish") {
        return Err(format!(
            "unsupported hidden_act {hidden_act:?}: the MLP and the gated-delta-net convolution \
             are SiLU"
        ));
    }

    let gate = GdnGateActivation::resolve(
        text.get("output_gate_type").and_then(Value::as_str),
        Some(&hidden_act),
    )?;
    text.insert("output_gate_type".into(), gate.as_str().into());

    match text.get("attn_output_gate") {
        None | Some(Value::Null) | Some(Value::Bool(true)) => {
            text.insert("attn_output_gate".into(), true.into());
        }
        Some(other) => {
            return Err(format!(
                "attn_output_gate = {other} is not supported: every Qwen3.5-family full-attention \
                 layer gates its output (q_proj is twice the head width)"
            ));
        }
    }

    // RoPE: `rope_parameters` (or a legacy `rope_scaling`) wins over the
    // legacy top-level keys.
    let rope = text
        .get("rope_parameters")
        .filter(|v| !v.is_null())
        .or_else(|| text.get("rope_scaling").filter(|v| !v.is_null()))
        .and_then(Value::as_object);
    let pick = |key: &str| {
        rope.and_then(|r| r.get(key))
            .filter(|v| !v.is_null())
            .or_else(|| text.get(key).filter(|v| !v.is_null()))
            .and_then(Value::as_f64)
    };
    let rope_theta = pick("rope_theta").unwrap_or(10_000.0);
    let partial = pick("partial_rotary_factor").unwrap_or(0.25);
    if !(partial > 0.0 && partial <= 1.0) {
        return Err(format!("partial_rotary_factor {partial} is outside (0, 1]"));
    }
    text.insert("rope_theta".into(), rope_theta.into());
    text.insert("partial_rotary_factor".into(), partial.into());
    let rope_parameters = resolve_rope_parameters(&text, rope_theta, partial)?;
    text.insert("rope_parameters".into(), Value::Object(rope_parameters));
    // Read once, from `rope_parameters`; a stale legacy copy would only
    // mislead an engine that looked at it.
    text.remove("rope_scaling");

    let n_layers = int(&text, "num_hidden_layers")?;
    let layer_types: Vec<Value> = match text.get("layer_types").filter(|v| !v.is_null()) {
        Some(Value::Array(types)) => {
            if types.len() as i64 != n_layers {
                return Err(format!(
                    "layer_types has {} entries for num_hidden_layers = {n_layers}",
                    types.len()
                ));
            }
            for (i, t) in types.iter().enumerate() {
                match t.as_str() {
                    Some("linear_attention" | "full_attention") => {}
                    _ => {
                        return Err(format!(
                            "layer_types[{i}] = {t} is not supported; expected \
                             \"linear_attention\" or \"full_attention\""
                        ));
                    }
                }
            }
            types.clone()
        }
        Some(other) => return Err(format!("layer_types must be a list, got {other}")),
        None => {
            let interval = text
                .get("full_attention_interval")
                .and_then(Value::as_i64)
                .unwrap_or(4);
            if interval < 1 {
                return Err(format!("full_attention_interval {interval} must be >= 1"));
            }
            (0..n_layers)
                .map(|i| {
                    if (i + 1) % interval == 0 {
                        "full_attention".into()
                    } else {
                        "linear_attention".into()
                    }
                })
                .collect()
        }
    };
    text.insert("layer_types".into(), Value::Array(layer_types));

    let heads = int(&text, "num_attention_heads")?;
    let kv_heads = int(&text, "num_key_value_heads")?;
    if kv_heads <= 0 || heads % kv_heads != 0 {
        return Err(format!(
            "num_attention_heads {heads} is not a multiple of num_key_value_heads {kv_heads}"
        ));
    }
    let lk = int(&text, "linear_num_key_heads")?;
    let lv = int(&text, "linear_num_value_heads")?;
    if lk <= 0 || lv % lk != 0 {
        return Err(format!(
            "linear_num_value_heads {lv} is not a multiple of linear_num_key_heads {lk}"
        ));
    }
    let dk = int(&text, "linear_key_head_dim")?;
    if dk <= 0 || dk % GDN_KERNEL_DK_MULTIPLE != 0 {
        return Err(format!(
            "linear_key_head_dim {dk} is not a multiple of {GDN_KERNEL_DK_MULTIPLE}, which the \
             gated-delta-net Metal kernel requires"
        ));
    }
    let head_dim = int(&text, "head_dim")?;
    let rotary = (head_dim as f64 * partial) as i64;
    if rotary < 2 || rotary % 2 != 0 {
        return Err(format!(
            "head_dim {head_dim} x partial_rotary_factor {partial} = {rotary} rotary dims; \
             expected a positive even number"
        ));
    }

    Ok(Value::Object(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn qwen38_27b() -> Value {
        json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "tie_word_embeddings": false,
            "text_config": {
                "attention_bias": false, "attn_output_gate": true, "bos_token_id": 248044,
                "eos_token_id": 248044, "full_attention_interval": 4, "head_dim": 256,
                "hidden_act": "silu", "hidden_size": 5120, "intermediate_size": 17408,
                "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_value_head_dim": 128, "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144, "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 24, "num_hidden_layers": 64, "num_key_value_heads": 4,
                "output_gate_type": "swish", "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-6,
                "rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"},
                "tie_word_embeddings": false, "vocab_size": 248320
            }
        })
    }

    #[test]
    fn qwen38_resolves_to_its_reference_reading() {
        let t = normalize_text_config(&qwen38_27b()).expect("Qwen3.8-27B is supported");
        assert_eq!(t["output_gate_type"], "silu");
        assert_eq!(t["rope_theta"], 10_000_000.0);
        assert_eq!(t["partial_rotary_factor"], 0.25);
        assert_eq!(t["tie_word_embeddings"], false);
        let types = t["layer_types"].as_array().unwrap();
        assert_eq!(types.len(), 64);
        assert_eq!(types[3], "full_attention");
        assert_eq!(types[4], "linear_attention");
        assert_eq!(types[63], "full_attention");
    }

    #[test]
    fn the_outer_tie_flag_governs_a_nested_config() {
        let mut c = qwen38_27b();
        c["tie_word_embeddings"] = true.into();
        assert_eq!(
            normalize_text_config(&c).unwrap()["tie_word_embeddings"],
            true
        );
        c["tie_word_embeddings"] = false.into();
        c["text_config"]["tie_word_embeddings"] = true.into();
        assert_eq!(
            normalize_text_config(&c).unwrap()["tie_word_embeddings"],
            false
        );
    }

    #[test]
    fn rope_parameters_win_over_legacy_keys_and_absent_theta_is_ten_thousand() {
        let mut c = qwen38_27b();
        c["text_config"]["rope_theta"] = 5e6.into();
        c["text_config"]["partial_rotary_factor"] = 0.5.into();
        let t = normalize_text_config(&c).unwrap();
        assert_eq!(t["rope_theta"], 10_000_000.0);
        assert_eq!(t["partial_rotary_factor"], 0.25);

        let tc = c["text_config"].as_object_mut().unwrap();
        tc.remove("rope_parameters");
        tc.remove("rope_theta");
        tc.remove("partial_rotary_factor");
        let t = normalize_text_config(&c).unwrap();
        assert_eq!(t["rope_theta"], 10_000.0);
        assert_eq!(t["partial_rotary_factor"], 0.25);
    }

    /// The YaRN block the Qwen3.8 card gives for contexts past 262,144 tokens.
    fn qwen38_card_yarn() -> Value {
        let mut c = qwen38_27b();
        c["text_config"]["rope_parameters"] = json!({
            "mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn",
            "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0,
            "original_max_position_embeddings": 262144
        });
        c
    }

    #[test]
    fn card_yarn_resolves_like_transformers() {
        let t = normalize_text_config(&qwen38_card_yarn()).expect("yarn is supported");
        let rope = &t["rope_parameters"];
        assert_eq!(rope["rope_type"], "yarn");
        assert_eq!(rope["mrope_section"], json!([11, 11, 10]));
        let af = rope["attention_factor"].as_f64().unwrap();
        assert!((af - (0.1 * 4f64.ln() + 1.0)).abs() < 1e-12, "{af}");
        assert_eq!(rope["beta_fast"], 32.0);
        assert_eq!(rope["beta_slow"], 1.0);
        assert_eq!(rope["truncate"], true);

        let rotary = rotary(&t).unwrap();
        assert_eq!(rotary.dims, 64);
        let inv = rotary.inverse_frequencies(0);
        let plain = Rotary {
            scaling: RopeScaling::Default,
            ..rotary.clone()
        }
        .inverse_frequencies(0);
        // The highest frequencies stay as trained, the lowest are divided by
        // the factor, and the ones between are blended.
        assert_eq!(inv[0], plain[0]);
        assert!((inv[31] - plain[31] / 4.0).abs() <= 1e-6 * plain[31]);
        assert!(inv.iter().zip(&plain).all(|(y, p)| y <= p));
        assert!((rotary.attention_factor() - af as f32).abs() < 1e-7);
    }

    #[test]
    fn yarn_defaults_and_overrides() {
        // No factor: max_position_embeddings / original. mscale pair: their
        // ratio. A legacy `rope_scaling` stands in for `rope_parameters`.
        let mut c = qwen38_27b();
        let tc = c["text_config"].as_object_mut().unwrap();
        tc.remove("rope_parameters");
        tc.insert(
            "rope_scaling".into(),
            json!({"type": "yarn", "original_max_position_embeddings": 65536,
                   "mscale": 1.0, "mscale_all_dim": 0.5}),
        );
        tc.insert("rope_theta".into(), 1e6.into());
        let t = normalize_text_config(&c).unwrap();
        let rope = &t["rope_parameters"];
        assert_eq!(rope["factor"], 4.0);
        let want = (0.1 * 4f64.ln() + 1.0) / (0.05 * 4f64.ln() + 1.0);
        assert!((rope["attention_factor"].as_f64().unwrap() - want).abs() < 1e-12);
        assert_eq!(rope["rope_theta"], 1e6);
        assert!(t.get("rope_scaling").is_none());

        // An explicit attention_factor wins; a default rope_type is plain.
        let mut c = qwen38_card_yarn();
        c["text_config"]["rope_parameters"]["attention_factor"] = 1.0.into();
        let t = normalize_text_config(&c).unwrap();
        assert_eq!(rotary(&t).unwrap().attention_factor(), 1.0);
        let plain = normalize_text_config(&qwen38_27b()).unwrap();
        assert_eq!(rotary(&plain).unwrap().scaling, RopeScaling::Default);
    }

    #[test]
    fn output_gate_types() {
        let mut c = qwen38_27b();
        c["text_config"]["output_gate_type"] = "sigmoid".into();
        assert_eq!(
            normalize_text_config(&c).unwrap()["output_gate_type"],
            "sigmoid"
        );
        c["text_config"]["output_gate_type"] = Value::Null;
        assert_eq!(
            normalize_text_config(&c).unwrap()["output_gate_type"],
            "silu"
        );
        c["text_config"]["output_gate_type"] = "gelu".into();
        let err = normalize_text_config(&c).unwrap_err();
        assert!(err.contains("output_gate_type"), "{err}");
    }

    #[test]
    fn unsupported_values_are_refused_by_name() {
        let cases: [(&str, Value, &str); 6] = [
            ("attn_output_gate", false.into(), "attn_output_gate"),
            ("hidden_act", "gelu".into(), "hidden_act"),
            ("layer_types", json!(["linear_attention"]), "layer_types"),
            ("linear_key_head_dim", 48.into(), "linear_key_head_dim"),
            ("num_key_value_heads", 5.into(), "num_key_value_heads"),
            (
                "rope_parameters",
                json!({"rope_type": "linear", "factor": 4.0, "rope_theta": 1e7}),
                "rope_type",
            ),
        ];
        for (key, value, needle) in cases {
            let mut c = qwen38_27b();
            c["text_config"][key] = value;
            let err = normalize_text_config(&c).expect_err(key);
            assert!(err.contains(needle), "{key}: {err}");
        }
        let mut c = qwen38_27b();
        let mut types = vec![json!("linear_attention"); 64];
        types[5] = json!("sliding_attention");
        c["text_config"]["layer_types"] = Value::Array(types);
        assert!(
            normalize_text_config(&c)
                .unwrap_err()
                .contains("layer_types[5]")
        );
    }

    #[test]
    fn absent_keys_take_the_reference_class_defaults() {
        let c = json!({"model_type": "qwen3_5_text", "hidden_size": 64,
                       "num_hidden_layers": 4, "num_attention_heads": 4});
        let t = normalize_text_config(&c).unwrap();
        assert_eq!(t["linear_num_key_heads"], 16);
        assert_eq!(t["linear_num_value_heads"], 32);
        assert_eq!(t["num_key_value_heads"], 4);
        assert_eq!(t["head_dim"], 256);
        assert_eq!(t["tie_word_embeddings"], false);
        let moe = json!({"model_type": "qwen3_5_moe_text", "num_hidden_layers": 4});
        let t = normalize_text_config(&moe).unwrap();
        assert_eq!(t["num_experts"], 256);
        assert_eq!(t["num_key_value_heads"], 2);
        assert!(t.get("intermediate_size").is_none());
    }

    #[test]
    fn canonical_keys() {
        assert_eq!(
            canonical_checkpoint_key("model.language_model.layers.0.linear_attn.A_log"),
            "model.layers.0.linear_attn.a_log"
        );
        assert_eq!(
            canonical_checkpoint_key("language_model.model.embed_tokens.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            canonical_checkpoint_key("language_model.lm_head.weight"),
            "lm_head.weight"
        );
        assert!(is_unused_by_text_model(
            "model.visual.blocks.0.attn.qkv.weight"
        ));
        assert!(is_unused_by_text_model("mtp.fc.weight"));
        assert!(!is_unused_by_text_model("model.language_model.norm.weight"));
    }
}
