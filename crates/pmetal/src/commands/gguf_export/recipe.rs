//! The type each tensor of a GGUF file is stored in.
//!
//! A K-quant file is not one type: the name of the method (`q3_k_m`) is the
//! type of most tensors, and the ones that matter most are kept at more bits.
//! Which they are, and for which layers, is a recipe that has become part of
//! the format's meaning: a `q3_k_m` file is "Q3_K with the sensitive tensors
//! promoted the way every other `q3_k_m` file is", and a recipe that promotes
//! fewer of them is a smaller, worse file under the same name.
//!
//! The recipe here is the reference quantizer's, tensor class by tensor class
//! and layer by layer, for the methods `pmetal quantize` offers. It depends on
//! the model (layer count, the grouped-query ratio, the experts, whether the
//! head is tied to the embedding) and on whether an importance matrix is used,
//! so those are what [`Recipe`] is built from.

use super::{ExportArch, json_usize};
use crate::cli::quantize::QuantizeMethod;
use pmetal_gguf::GgmlType;

/// Everything about a model and an export that decides a tensor's type.
#[derive(Debug, Clone)]
pub(crate) struct Recipe {
    method: QuantizeMethod,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    n_experts: usize,
    /// The architecture is `llama`, where 80 layers with grouped-query
    /// attention is the 70B model.
    llama: bool,
    /// An importance matrix is applied, which changes what a few legacy types
    /// can be trusted with.
    has_imatrix: bool,
    /// The file has an `output.weight` of its own. Without one the head is the
    /// embedding, which is then quantized as a head.
    has_output: bool,
}

impl Recipe {
    pub(crate) fn new(
        method: QuantizeMethod,
        arch: &ExportArch,
        config_json: Option<&serde_json::Value>,
        has_imatrix: bool,
        has_output: bool,
    ) -> Self {
        let count = |keys: &[&str]| {
            config_json
                .and_then(|json| keys.iter().find_map(|key| json_usize(json, key)))
                .unwrap_or(0)
        };
        let n_heads = count(&["num_attention_heads"]);
        let n_kv_heads = config_json
            .and_then(|json| json_usize(json, "num_key_value_heads"))
            .unwrap_or(n_heads.max(1));
        Self {
            method,
            n_layers: count(&["num_hidden_layers"]),
            n_heads,
            n_kv_heads,
            n_experts: count(&["num_local_experts", "num_experts"]),
            llama: arch.name == "llama",
            has_imatrix,
            has_output,
        }
    }

    /// Query heads per key/value head.
    fn n_gqa(&self) -> usize {
        self.n_heads.checked_div(self.n_kv_heads).unwrap_or(1)
    }

    /// The type of tensor `gguf_name`, of row-major `shape`.
    ///
    /// `proposed` is what the dynamic quantizer chose, which only the
    /// `dynamic` method uses.
    pub(crate) fn tensor_type(
        &self,
        gguf_name: &str,
        shape: &[u64],
        proposed: GgmlType,
    ) -> GgmlType {
        use GgmlType as T;
        use QuantizeMethod as M;
        let method = self.method;

        // A vector (a norm weight, a bias) is kept in f32 whatever the type of
        // the file. An engine multiplies activations by it element-wise, and the
        // ops that do so take it as f32 (the reference engine aborts on a bf16
        // one), so a half-precision norm is a file no engine will run. A vector
        // is a negligible share of the bytes.
        if shape.len() <= 1 {
            return T::F32;
        }
        if matches!(method, M::F32 | M::F16 | M::Bf16) {
            return default_type(method);
        }
        if !is_quantizable(gguf_name, shape) {
            return T::F16;
        }

        let dynamic = matches!(method, M::Dynamic);
        let mut dtype = if dynamic {
            proposed
        } else {
            default_type(method)
        };
        let layer = layer_of(gguf_name);
        let n_layers = self.n_layers;

        // The head, and the embedding when it is the head. A file with a head
        // of its own keeps the embedding at the type of the file.
        let is_head =
            gguf_name == "output.weight" || (!self.has_output && gguf_name == "token_embd.weight");
        if is_head || gguf_name == "token_embd.weight" && dynamic {
            return match method {
                M::Q1_0 | M::Tq1_0 | M::Tq2_0 | M::Mxfp4 | M::Nvfp4 => T::F16,
                _ if dtype == T::Q8_0 => T::Q8_0,
                M::Q8_1 => dtype,
                _ => T::Q6K,
            };
        }
        if gguf_name == "token_embd.weight"
            && matches!(method, M::Q1_0 | M::Tq1_0 | M::Tq2_0 | M::Mxfp4 | M::Nvfp4)
        {
            return T::F16;
        }

        if gguf_name.contains("attn_v.weight") {
            dtype = match method {
                M::Q2K if self.n_gqa() >= 4 => T::Q4K,
                M::Q2K => T::Q3K,
                M::Q3KM if layer < 2 => T::Q5K,
                M::Q3KM => T::Q4K,
                M::Q3KL => T::Q5K,
                M::Q4KM | M::Q5KM if use_more_bits(layer, n_layers) => T::Q6K,
                M::Q4KS if layer < 4 => T::Q5K,
                M::Iq4Nl | M::Iq4Xs if self.n_gqa() >= 4 => T::Q5K,
                _ => dtype,
            };
            if !dynamic {
                // The 70B model has eight query heads per value head, so this
                // tensor is an eighth of the size of the query projection and a
                // few more bits on it cost almost nothing.
                if self.llama
                    && n_layers == 80
                    && self.n_heads != self.n_kv_heads
                    && matches!(dtype, T::Q3K | T::Q4K)
                {
                    dtype = T::Q5K;
                }
                // The eight-expert model's value projection is small, and
                // eight bits on it add little.
                if self.n_experts == 8 {
                    dtype = T::Q8_0;
                }
            }
        } else if gguf_name.contains("attn_k.weight") {
            if self.n_experts == 8 && !dynamic {
                dtype = T::Q8_0;
            }
        } else if gguf_name.contains("ffn_down") {
            dtype = match method {
                M::Q2K => T::Q3K,
                M::Q3KM if layer < n_layers / 16 => T::Q5K,
                M::Q3KM => T::Q4K,
                M::Q3KL => T::Q5K,
                M::Q4KM | M::Q5KM if use_more_bits(layer, n_layers) => T::Q6K,
                M::Iq4Nl | M::Iq4Xs if !self.has_imatrix && layer < n_layers / 8 => T::Q5K,
                M::Q4KS if layer < n_layers / 8 => T::Q5K,
                // Without an importance matrix the first layers of a 4- or
                // 5-bit legacy file go wrong in this tensor; with one they do
                // not need the extra bit.
                M::Q4_0 if self.has_imatrix && layer < n_layers / 8 => T::Q4_1,
                M::Q5_0 if self.has_imatrix && layer < n_layers / 8 => T::Q5_1,
                _ => dtype,
            };
        } else if gguf_name.contains("attn_output.weight") {
            dtype = if self.n_experts == 8 && !dynamic {
                match method {
                    M::Q2K | M::Q3KS | M::Q3KM | M::Q4KS | M::Q4KM | M::Iq4Nl | M::Iq4Xs => T::Q5K,
                    _ => dtype,
                }
            } else {
                match method {
                    M::Q2K => T::Q3K,
                    M::Q3KM => T::Q4K,
                    M::Q3KL => T::Q5K,
                    _ => dtype,
                }
            };
        } else if gguf_name.contains("attn_qkv.weight") {
            dtype = match method {
                M::Q3KM | M::Q3KL => T::Q4K,
                M::Q4KM => T::Q5K,
                M::Q5KM => T::Q6K,
                _ => dtype,
            };
        }

        dtype
    }
}

/// The type a method stores its bulk tensors in.
fn default_type(method: QuantizeMethod) -> GgmlType {
    method.to_ggml_type().unwrap_or(GgmlType::Q4K)
}

/// The layer a block tensor belongs to (`blk.12.attn_q.weight` is 12).
fn layer_of(gguf_name: &str) -> usize {
    gguf_name
        .strip_prefix("blk.")
        .and_then(|rest| rest.split_once('.'))
        .and_then(|(layer, _)| layer.parse().ok())
        .unwrap_or(0)
}

/// The first and last eighth of the layers, and every third one between, are
/// kept at more bits: they are where a model is most sensitive to rounding.
fn use_more_bits(layer: usize, n_layers: usize) -> bool {
    n_layers != 0
        && (layer < n_layers / 8 || layer >= 7 * n_layers / 8 || (layer - n_layers / 8) % 3 == 2)
}

/// Whether a tensor is quantized at all: a matrix, but not a norm, a router,
/// a convolution or a position table, which cost little and cannot lose bits.
fn is_quantizable(gguf_name: &str, shape: &[u64]) -> bool {
    if !(shape.len() == 2 || shape.len() == 3) || !gguf_name.ends_with(".weight") {
        return false;
    }
    let name = gguf_name.to_ascii_lowercase();
    !(name.contains("norm.weight")
        || name.contains("ffn_gate_inp.weight")
        || name.contains("router")
        || name.contains("altup")
        || name.contains("laurel")
        || name.contains("per_layer_model_proj")
        || name.contains("pos_embd")
        || name.contains("token_types")
        || name.contains("ssm_conv1d")
        || name.contains("shortconv.conv.weight")
        || name.contains("attn_rel_b.weight")
        || name.contains(".position_embd")
        || name.contains(".rel_pos")
        || name.contains(".patch_embd")
        || name.contains(".patch_merger"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A model as `(layers, heads, kv heads, experts, a head of its own)`.
    type Model = (usize, usize, usize, usize, bool);

    fn recipe(method: QuantizeMethod, arch: &str, model: Model, has_imatrix: bool) -> Recipe {
        let (layers, heads, kv_heads, experts, has_output) = model;
        let mut config = json!({
            "architectures": [if arch == "llama" { "LlamaForCausalLM" } else { "Qwen3ForCausalLM" }],
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
        });
        if experts > 0 {
            config["num_local_experts"] = json!(experts);
        }
        let arch = ExportArch::detect(Some(&config)).unwrap();
        Recipe::new(method, &arch, Some(&config), has_imatrix, has_output)
    }

    /// The models `recipe_reference.txt` is about.
    fn reference_model(name: &str) -> Model {
        match name {
            // Qwen3-0.6B's layer count, with a tied head.
            "tied28" => (28, 8, 4, 0, false),
            "dense32" => (32, 8, 2, 0, true),
            "dense12" => (12, 8, 2, 0, true),
            // Eighty layers: the 70B model with grouped-query attention, the
            // 65B without.
            "llama70b" => (80, 8, 2, 0, true),
            "llama65b" => (80, 8, 8, 0, true),
            "moe8" => (16, 8, 2, 8, true),
            "moe4" => (16, 8, 8, 4, true),
            other => panic!("no model {other}"),
        }
    }

    fn reference_method(name: &str) -> QuantizeMethod {
        use QuantizeMethod as M;
        match name {
            "Q2_K" => M::Q2K,
            "Q3_K_S" => M::Q3KS,
            "Q3_K_M" => M::Q3KM,
            "Q3_K_L" => M::Q3KL,
            "Q4_0" => M::Q4_0,
            "Q4_1" => M::Q4_1,
            "Q5_0" => M::Q5_0,
            "Q5_1" => M::Q5_1,
            "Q4_K_S" => M::Q4KS,
            "Q4_K_M" => M::Q4KM,
            "Q5_K_S" => M::Q5KS,
            "Q5_K_M" => M::Q5KM,
            "Q6_K" => M::Q6K,
            "Q8_0" => M::Q8_0,
            "IQ4_NL" => M::Iq4Nl,
            "IQ4_XS" => M::Iq4Xs,
            other => panic!("no method {other}"),
        }
    }

    /// The type of every tensor of a class, layer by layer, as the run-length
    /// list `recipe_reference.txt` uses.
    fn runs(types: &[GgmlType]) -> String {
        let mut out: Vec<String> = Vec::new();
        let mut i = 0;
        while i < types.len() {
            let n = types[i..].iter().take_while(|t| **t == types[i]).count();
            out.push(format!("{:?}:{n}", types[i]));
            i += n;
        }
        out.join(",")
    }

    /// Every method on every kind of model gives every tensor class the type the
    /// reference quantizer gives it, at every layer.
    ///
    /// The table is what the reference quantizer wrote for synthetic models of
    /// these shapes. A `q3_k_m` that promotes `ffn_down` in the layers the
    /// reference does not (or fails to in those it does) is a different, and
    /// here a worse, file under the same name: Qwen3-0.6B's perplexity went from
    /// 94.7 against 92.1 to within noise of it with this recipe.
    #[test]
    fn every_method_gives_every_tensor_the_type_the_reference_quantizer_gives_it() {
        use std::collections::HashMap;
        let mut expected: HashMap<(&str, &str, bool, &str), &str> = HashMap::new();
        let mut cells: Vec<(&str, &str)> = Vec::new();
        for line in include_str!("recipe_reference.txt")
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        {
            let p: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(p.len(), 5, "{line}");
            expected.insert((p[0], p[1], p[2] == "1", p[3]), p[4]);
            if p[2] == "0" && p[3] == "base" {
                cells.push((p[0], p[1]));
            }
        }
        assert!(cells.len() > 60, "{} model and method pairs", cells.len());

        let mut checked = 0;
        for (model, method_name) in cells {
            let (layers, heads, kv_heads, experts, has_output) = reference_model(model);
            for has_imatrix in [false, true] {
                let recipe = recipe(
                    reference_method(method_name),
                    "llama",
                    (layers, heads, kv_heads, experts, has_output),
                    has_imatrix,
                );
                let want = |class: &str| {
                    expected
                        .get(&(model, method_name, has_imatrix, class))
                        .or_else(|| expected.get(&(model, method_name, false, class)))
                        .copied()
                };
                let matrix = [512u64, 256];
                let stacked = [experts.max(1) as u64, 256, 256];
                let type_of =
                    |name: &str, shape: &[u64]| recipe.tensor_type(name, shape, GgmlType::Q4K);
                let per_layer = |class: &str, shape: &[u64]| -> String {
                    let types: Vec<GgmlType> = (0..layers)
                        .map(|l| type_of(&format!("blk.{l}.{class}.weight"), shape))
                        .collect();
                    runs(&types)
                };
                let ctx = format!("{model} {method_name} importance={has_imatrix}");

                for class in ["attn_v", "attn_k", "attn_output"] {
                    assert_eq!(
                        per_layer(class, &matrix),
                        want(class).unwrap(),
                        "{ctx} {class}"
                    );
                    checked += 1;
                }
                let (down, down_shape): (&str, &[u64]) = if experts > 0 {
                    ("ffn_down_exps", &stacked)
                } else {
                    ("ffn_down", &matrix)
                };
                assert_eq!(
                    per_layer(down, down_shape),
                    want(down).unwrap(),
                    "{ctx} {down}"
                );
                // The tensors no method promotes are the method's own type.
                let base = want("base").unwrap();
                for class in ["attn_q", "ffn_gate", "ffn_up"] {
                    let (class, shape) = if experts > 0 && class != "attn_q" {
                        (format!("{class}_exps"), &stacked[..])
                    } else {
                        (class.to_string(), &matrix[..])
                    };
                    assert_eq!(
                        per_layer(&class, shape),
                        format!("{base}:{layers}"),
                        "{ctx} {class}"
                    );
                }
                // The head, and the embedding: which is quantized as a head
                // depends on whether the file has one of its own.
                if has_output {
                    assert_eq!(
                        runs(&[type_of("output.weight", &matrix)]),
                        want("output").unwrap(),
                        "{ctx} output"
                    );
                } else {
                    assert!(want("output").is_none(), "{ctx}");
                }
                assert_eq!(
                    runs(&[type_of("token_embd.weight", &matrix)]),
                    want("token_embd").unwrap(),
                    "{ctx} token_embd"
                );
                checked += 4;
            }
        }
        assert!(checked > 800, "{checked}");
    }

    /// The pieces of the recipe the table above cannot show, because they are
    /// the ones the reference does not decide.
    #[test]
    fn vectors_stay_f32_and_unquantized_matrices_f16_whatever_the_method() {
        for method in [
            QuantizeMethod::Q4KM,
            QuantizeMethod::Q8_0,
            QuantizeMethod::Dynamic,
        ] {
            let r = recipe(method, "llama", (28, 8, 4, 0, false), false);
            assert_eq!(
                r.tensor_type("blk.0.attn_norm.weight", &[1024], GgmlType::Q4K),
                GgmlType::F32
            );
            assert_eq!(
                r.tensor_type("blk.0.attn_q.bias", &[1024], GgmlType::Q4K),
                GgmlType::F32
            );
            assert_eq!(
                r.tensor_type("blk.0.ffn_gate_inp.weight", &[8, 1024], GgmlType::Q4K),
                GgmlType::F16
            );
        }
        for method in [
            QuantizeMethod::F32,
            QuantizeMethod::F16,
            QuantizeMethod::Bf16,
        ] {
            let r = recipe(method, "llama", (28, 8, 4, 0, false), false);
            let want = method.to_ggml_type().unwrap();
            assert_eq!(
                r.tensor_type("blk.0.ffn_up.weight", &[512, 256], GgmlType::Q4K),
                want
            );
            assert_eq!(
                r.tensor_type("blk.0.attn_norm.weight", &[256], GgmlType::Q4K),
                GgmlType::F32
            );
        }
    }

    /// The `dynamic` method takes its types from the quantizer and keeps the
    /// head and the embedding at six bits; the exotic types keep both at f16.
    #[test]
    fn the_dynamic_and_exotic_methods_keep_their_head_and_embedding_out_of_the_recipe() {
        let r = recipe(QuantizeMethod::Dynamic, "llama", (28, 8, 4, 0, true), false);
        assert_eq!(
            r.tensor_type("blk.3.ffn_down.weight", &[512, 256], GgmlType::Q3K),
            GgmlType::Q3K
        );
        assert_eq!(
            r.tensor_type("output.weight", &[512, 256], GgmlType::Q3K),
            GgmlType::Q6K
        );
        assert_eq!(
            r.tensor_type("token_embd.weight", &[512, 256], GgmlType::Q3K),
            GgmlType::Q6K
        );
        let r = recipe(QuantizeMethod::Tq1_0, "llama", (28, 8, 4, 0, true), false);
        assert_eq!(
            r.tensor_type("output.weight", &[512, 256], GgmlType::Tq1_0),
            GgmlType::F16
        );
        assert_eq!(
            r.tensor_type("token_embd.weight", &[512, 256], GgmlType::Tq1_0),
            GgmlType::F16
        );
    }

    #[test]
    fn the_more_bits_layers_are_the_first_and_last_eighth_and_every_third_between() {
        let chosen: Vec<usize> = (0..32).filter(|&l| use_more_bits(l, 32)).collect();
        assert_eq!(
            chosen,
            [0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 29, 30, 31]
        );
        assert!(!use_more_bits(0, 0));
    }
}
