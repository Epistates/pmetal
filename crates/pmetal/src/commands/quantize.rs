use crate::commands::quantize_source::TensorMeta;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;
const TOKEN_TYPE_UNUSED: i32 = 5;

const LLAMA_PRETOKENIZER_CHECK: &str = "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙 3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български ''''''```````\"\"\"\"......!!!!!!?????? I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL";

fn lower_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn quantize_method_file_type(
    method: crate::cli::quantize::QuantizeMethod,
) -> pmetal_gguf::FileType {
    use pmetal_gguf::FileType;
    match method {
        crate::cli::quantize::QuantizeMethod::Dynamic => FileType::MostlyQ4KM,
        crate::cli::quantize::QuantizeMethod::Q8_0 => FileType::MostlyQ8_0,
        crate::cli::quantize::QuantizeMethod::Q8_1 => FileType::MostlyQ8_0,
        crate::cli::quantize::QuantizeMethod::Q6K => FileType::MostlyQ6K,
        crate::cli::quantize::QuantizeMethod::Q5KM => FileType::MostlyQ5KM,
        crate::cli::quantize::QuantizeMethod::Q5KS => FileType::MostlyQ5KS,
        crate::cli::quantize::QuantizeMethod::Q5_0 => FileType::MostlyQ5_0,
        crate::cli::quantize::QuantizeMethod::Q5_1 => FileType::MostlyQ5_1,
        crate::cli::quantize::QuantizeMethod::Q4KM => FileType::MostlyQ4KM,
        crate::cli::quantize::QuantizeMethod::Q4KS => FileType::MostlyQ4KS,
        crate::cli::quantize::QuantizeMethod::Q4_0 => FileType::MostlyQ4_0,
        crate::cli::quantize::QuantizeMethod::Q4_1 => FileType::MostlyQ4_1,
        crate::cli::quantize::QuantizeMethod::Iq4Nl => FileType::MostlyIq4Nl,
        crate::cli::quantize::QuantizeMethod::Iq4Xs => FileType::MostlyIq4Xs,
        crate::cli::quantize::QuantizeMethod::Q3KM => FileType::MostlyQ3KM,
        crate::cli::quantize::QuantizeMethod::Q3KS => FileType::MostlyQ3KS,
        crate::cli::quantize::QuantizeMethod::Q3KL => FileType::MostlyQ3KL,
        crate::cli::quantize::QuantizeMethod::Q2K => FileType::MostlyQ2K,
        crate::cli::quantize::QuantizeMethod::Q1_0 => FileType::MostlyQ1_0,
        crate::cli::quantize::QuantizeMethod::Tq1_0 => FileType::MostlyTq1_0,
        crate::cli::quantize::QuantizeMethod::Tq2_0 => FileType::MostlyTq2_0,
        crate::cli::quantize::QuantizeMethod::Mxfp4 => FileType::MostlyMxfp4Moe,
        crate::cli::quantize::QuantizeMethod::Nvfp4 => FileType::MostlyNvfp4,
        crate::cli::quantize::QuantizeMethod::Bf16 => FileType::MostlyBf16,
        crate::cli::quantize::QuantizeMethod::F16 => FileType::MostlyF16,
        crate::cli::quantize::QuantizeMethod::F32 => FileType::AllF32,
    }
}

fn json_i64(json: &serde_json::Value, key: &str) -> Option<i64> {
    json.get(key).and_then(|value| value.as_i64())
}

fn json_u32(json: &serde_json::Value, key: &str) -> Option<u32> {
    json_i64(json, key).and_then(|value| u32::try_from(value).ok())
}

fn json_usize(json: &serde_json::Value, key: &str) -> Option<usize> {
    json_i64(json, key).and_then(|value| usize::try_from(value).ok())
}

fn json_f64(json: &serde_json::Value, key: &str) -> Option<f64> {
    json.get(key).and_then(|value| value.as_f64())
}

fn json_bool(json: &serde_json::Value, key: &str) -> Option<bool> {
    json.get(key).and_then(|value| value.as_bool())
}

fn json_str<'a>(json: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    json.get(key).and_then(|value| value.as_str())
}

fn add_arch_u32(
    builder: &mut pmetal_gguf::GgufBuilder,
    architecture: &str,
    suffix: &str,
    value: u32,
) {
    builder.add_u32(format!("{architecture}.{suffix}"), value);
}

fn add_arch_f32(
    builder: &mut pmetal_gguf::GgufBuilder,
    architecture: &str,
    suffix: &str,
    value: f32,
) {
    builder.add_f32(format!("{architecture}.{suffix}"), value);
}

fn add_arch_bool(
    builder: &mut pmetal_gguf::GgufBuilder,
    architecture: &str,
    suffix: &str,
    value: bool,
) {
    builder.add_bool(format!("{architecture}.{suffix}"), value);
}

fn add_hf_config_metadata(
    builder: &mut pmetal_gguf::GgufBuilder,
    architecture: &str,
    config_json: Option<&serde_json::Value>,
    method: crate::cli::quantize::QuantizeMethod,
) {
    builder.add_string("general.type", "model");
    builder.add_u32(
        pmetal_gguf::keys::GENERAL_FILE_TYPE,
        quantize_method_file_type(method) as u32,
    );

    let Some(json) = config_json else {
        return;
    };

    let mut add_u32 = |hf_keys: &[&str], gguf_suffix: &str| {
        for hf_key in hf_keys {
            if let Some(value) = json_u32(json, hf_key) {
                add_arch_u32(builder, architecture, gguf_suffix, value);
                break;
            }
        }
    };

    add_u32(&["hidden_size", "n_embd", "dim"], "embedding_length");
    add_u32(&["num_hidden_layers", "n_layer", "n_layers"], "block_count");
    add_u32(
        &["num_attention_heads", "n_head", "n_heads"],
        "attention.head_count",
    );
    add_u32(
        &["num_key_value_heads", "n_kv_heads"],
        "attention.head_count_kv",
    );
    add_u32(
        &["intermediate_size", "n_inner", "hidden_dim"],
        "feed_forward_length",
    );
    add_u32(
        &[
            "max_position_embeddings",
            "n_ctx",
            "n_positions",
            "max_length",
            "max_sequence_length",
            "model_max_length",
        ],
        "context_length",
    );
    add_u32(&["vocab_size"], "vocab_size");
    add_u32(&["num_local_experts", "num_experts"], "expert_count");
    add_u32(
        &[
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
        ],
        "expert_used_count",
    );
    add_u32(&["n_group"], "expert_group_count");
    add_u32(&["topk_group"], "expert_group_used_count");

    if let Some(false) = json_bool(json, "is_causal") {
        add_arch_bool(builder, architecture, "attention.causal", false);
    }

    if let Some(value) = json_u32(json, "sliding_window") {
        add_arch_u32(builder, architecture, "attention.sliding_window", value);
    }

    let head_dim = json_u32(json, "head_dim").or_else(|| {
        let hidden = json_u32(json, "hidden_size")?;
        let heads = json_u32(json, "num_attention_heads")?;
        (heads != 0).then_some(hidden / heads)
    });
    if let Some(value) = head_dim {
        // The GGUF convention is key_length/value_length. Keep the older pmetal
        // compatibility key until all internal loaders consume the standard one.
        add_arch_u32(builder, architecture, "attention.head_dim", value);
        add_arch_u32(builder, architecture, "attention.key_length", value);
        add_arch_u32(builder, architecture, "attention.value_length", value);
        add_arch_u32(builder, architecture, "rope.dimension_count", value);
    }

    if let Some(value) = json_f64(json, "rope_theta") {
        add_arch_f32(builder, architecture, "rope.freq_base", value as f32);
    }
    if let Some(value) = json_f64(json, "rms_norm_eps") {
        add_arch_f32(
            builder,
            architecture,
            "attention.layer_norm_rms_epsilon",
            value as f32,
        );
    }
    if let Some(value) = json_f64(json, "layer_norm_eps")
        .or_else(|| json_f64(json, "layer_norm_epsilon"))
        .or_else(|| json_f64(json, "norm_epsilon"))
    {
        add_arch_f32(
            builder,
            architecture,
            "attention.layer_norm_epsilon",
            value as f32,
        );
    }

    if let Some(rope_scaling) = json.get("rope_scaling").and_then(|value| value.as_object()) {
        let rope_type = rope_scaling
            .get("rope_type")
            .or_else(|| rope_scaling.get("type"))
            .and_then(|value| value.as_str());
        if let Some(rope_type) = rope_type {
            let gguf_type = match rope_type.to_ascii_lowercase().as_str() {
                "linear" => Some("linear"),
                "yarn" => Some("yarn"),
                "su" | "longrope" => Some("longrope"),
                "none" => Some("none"),
                _ => None,
            };
            if let Some(gguf_type) = gguf_type {
                builder.add_string(format!("{architecture}.rope.scaling.type"), gguf_type);
            }
        }
        if let Some(value) = rope_scaling.get("factor").and_then(|value| value.as_f64()) {
            add_arch_f32(builder, architecture, "rope.scaling.factor", value as f32);
        }
        if let Some(value) = rope_scaling
            .get("original_max_position_embeddings")
            .or_else(|| rope_scaling.get("original_context_length"))
            .and_then(|value| value.as_u64())
            .and_then(|value| u32::try_from(value).ok())
        {
            add_arch_u32(
                builder,
                architecture,
                "rope.scaling.original_context_length",
                value,
            );
        }
        if let Some(value) = rope_scaling
            .get("attention_factor")
            .or_else(|| rope_scaling.get("attn_factor"))
            .and_then(|value| value.as_f64())
        {
            add_arch_f32(
                builder,
                architecture,
                "rope.scaling.yarn_attn_factor",
                value as f32,
            );
        }
        if let Some(value) = rope_scaling
            .get("beta_fast")
            .and_then(|value| value.as_f64())
        {
            add_arch_f32(
                builder,
                architecture,
                "rope.scaling.yarn_beta_fast",
                value as f32,
            );
        }
        if let Some(value) = rope_scaling
            .get("beta_slow")
            .and_then(|value| value.as_f64())
        {
            add_arch_f32(
                builder,
                architecture,
                "rope.scaling.yarn_beta_slow",
                value as f32,
            );
        }
    }
}

fn extract_token_string(value: &serde_json::Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(|value| value.as_str()))
}

fn read_json_file(path: &Path) -> anyhow::Result<Option<serde_json::Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(path)?;
    Ok(Some(serde_json::from_str(&content)?))
}

fn tokenizer_model_kind(
    tokenizer_json: &serde_json::Value,
    config_json: Option<&serde_json::Value>,
) -> String {
    let model_type = tokenizer_json
        .pointer("/model/type")
        .and_then(|value| value.as_str())
        .unwrap_or_default();
    match model_type {
        "BPE" => {
            let model_type = config_json
                .and_then(|json| json_str(json, "model_type"))
                .unwrap_or_default()
                .to_ascii_lowercase();
            if model_type.contains("gemma4") {
                "gemma4".to_string()
            } else {
                "gpt2".to_string()
            }
        }
        "WordPiece" => "bert".to_string(),
        "Unigram" => {
            let model_type = config_json
                .and_then(|json| json_str(json, "model_type"))
                .unwrap_or_default()
                .to_ascii_lowercase();
            if model_type.contains("t5") {
                "t5".to_string()
            } else {
                "llama".to_string()
            }
        }
        _ => "gpt2".to_string(),
    }
}

fn known_pre_tokenizer(hash: &str) -> Option<&'static str> {
    match hash {
        "b6e8e1518dc4305be2fe39c313ed643381c4da5db34a98f6a04c093f8afbe99b" => Some("chatglm-bpe"),
        "81d72c7348a9f0ebe86f23298d37debe0a5e71149e29bd283904c02262b27516" => Some("chatglm-bpe"),
        "a1336059768a55c99a734006ffb02203cd450fed003e9a71886c88acf24fdbc2" => Some("glm4"),
        "9ca2dd618e8afaf09731a7cf6e2105b373ba6a1821559f258b272fe83e6eb902" => Some("glm4"),
        "cdf5f35325780597efd76153d4d1c16778f766173908894c04afc20108536267" => Some("glm4"),
        "1431a23e583c97432bc230bff598d103ddb5a1f89960c8f1d1051aaa944d0b35" => Some("minerva-7b"),
        "7e57df22b1fe23a7b1e1c7f3dc4e3f96d43a4eb0836d0c6bdc3436d7b2f1c664" => Some("hunyuan"),
        "bba3b3366b646dbdded5dbc42d59598b849371afc42f7beafa914afaa5b70aa6" => Some("hunyuan-dense"),
        "a6b57017d60e6edb4d88ecc2845188e0eb333a70357e45dcc9b53964a73bbae6" => Some("falcon-h1"),
        "60476e1243776c4fb1b993dbd7a5f15ac22f83c80afdf425fa5ae01c8d44ef86" => Some("falcon-h1"),
        "3eda48b4c4dc7de733d1a8b3e3b4a85243dbbf704da2ee9d42c6beced8897896" => Some("falcon-h1"),
        "48f8e02c0359c0bbdd82f26909171fac1c18a457bb47573ed1fe3bbb2c1cfd4b" => Some("falcon-h1"),
        "81212dc7cdb7e0c1074ca62c5aeab0d43c9f52b8a737be7b12a777c953027890" => Some("kimi-k2"),
        "d4540891389ea895b53b399da6ac824becc30f2fba0e9ddbb98f92e55ca0e97c" => Some("qwen2"),
        "66b8d4e19ab16c3bfd89bce5d785fb7e0155e8648708a1f42077cb9fe002c273" => Some("grok-2"),
        "b3d1dd861f1d4c5c0d2569ce36baf3f90fe8a102db3de50dd71ff860d91be3df" => Some("jina-v2-de"),
        "0fe1cf6eda062318a1af7270f3331a85c539a01778ff948e24388e949c5282f4" => Some("gpt-2"),
        "0ef9807a4087ebef797fc749390439009c3b9eda9ad1a097abbe738f486c01e5" => Some("llama-bpe"),
        "049ecf7629871e3041641907f3de7c733e4dbfdc736f57d882ba0b0845599754" => Some("deepseek-llm"),
        "347715f544604f9118bb75ed199f68779f423cabb20db6de6f31b908d04d7821" => {
            Some("deepseek-coder")
        }
        "8aeee3860c56296a157a1fe2fad249ec40aa59b1bb5709f4ade11c4e6fe652ed" => Some("falcon"),
        "9d032fcbd5501f4a38150912590928bfb36091efb5df11b8e2124b0390e3fb1e" => Some("falcon3"),
        "8e62295832751ca1e8f92f2226f403dea30dc5165e448b5bfa05af5340c64ec7" => {
            Some("bert-bge-large")
        }
        "35d91631860c815f952d711435f48d356ebac988362536bed955d43bfa436e34" => Some("starcoder"),
        "3ce83efda5659b07b1ad37ca97ca5797ea4285d9b9ab0dc679e4a720c9da7454" => Some("gpt-2"),
        "32d85c31273f8019248f2559fed492d929ea28b17e51d81d3bb36fff23ca72b3" => Some("stablelm2"),
        "6221ad2852e85ce96f791f476e0b390cf9b474c9e3d1362f53a24a06dc8220ff" => Some("refact"),
        "9c2227e4dd922002fb81bde4fc02b0483ca4f12911410dee2255e4987644e3f8" => Some("command-r"),
        "d772b220ace2baec124bed8cfafce0ead7d6c38a4b65ef11261cf9d5d62246d1" => Some("tiny_aya"),
        "e636dc30a262dcc0d8c323492e32ae2b70728f4df7dfe9737d9f920a282b8aea" => Some("qwen2"),
        "b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166" => Some("olmo"),
        "a8594e3edff7c29c003940395316294b2c623e09894deebbc65f33f1515df79e" => Some("dbrx"),
        "c7699093ba4255a91e702aa38a596aa81669f3525dae06c2953267dde580f448" => Some("jina-v1-en"),
        "0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f" => Some("jina-v2-en"),
        "171aeeedd6fb548d418a7461d053f11b6f1f1fc9b387bd66640d28a4b9f5c643" => Some("jina-v2-es"),
        "27949a2493fc4a9f53f5b9b029c82689cfbe5d3a1929bb25e043089e28466de6" => Some("jina-v2-de"),
        "a023e9fdc5a11f034d3ef515b92350e56fb2af1f66c6b6811a4444ea9bf8763d" => Some("jina-v5-nano"),
        "c136ed14d01c2745d4f60a9596ae66800e2b61fa45643e72436041855ad4089d" => Some("smaug-bpe"),
        "c7ea5862a53e4272c035c8238367063e2b270d51faa48c0f09e9d5b54746c360" => Some("poro-chat"),
        "7967bfa498ade6b757b064f31e964dddbb80f8f9a4d68d4ba7998fcf281c531a" => Some("jina-v2-code"),
        "7fc505bd3104ca1083b150b17d088b59534ede9bde81f0dd2090967d7fe52cee" => Some("viking"),
        "b53802fb28e26d645c3a310b34bfe07da813026ec7c7716883404d5e0f8b1901" => Some("jais"),
        "bc5108ee1eb6a3d600cadd065f63190fbd0554dbc9e4bbd6a0d977970afc8d2a" => Some("jais-2"),
        "7b3e7548e4308f52a76e8229e4e6cc831195d0d1df43aed21ac6c93da05fec5f" => Some("codeshell"),
        "63b97e4253352e6f357cc59ea5b583e3a680eaeaf2632188c2b952de2588485e" => Some("tekken"),
        "855059429035d75a914d1eda9f10a876752e281a054a7a3d421ef0533e5b6249" => Some("smollm"),
        "3c30d3ad1d6b64202cd222813e7736c2db6e1bd6d67197090fc1211fbc612ae7" => Some("bloom"),
        "bc01ce58980e1db43859146dc51b1758b3b88729b217a74792e9f8d43e479d21" => Some("gpt3-finnish"),
        "4e2b24cc4770243d65a2c9ec19770a72f08cffc161adbb73fcbb6b7dd45a0aae" => Some("exaone"),
        "fcace8b9cac38ce847670c970cd5892031a753a1ef381abd1d9af00f713da085" => Some("phi-2"),
        "60824e3c0d9401f89943cbb2fff727f0e2d4c545ba4df2d6e4f09a6db0f5b450" => Some("chameleon"),
        "8b5a93ed704057481f240da0be7e7dca721d7f8f4755263b6807227a2cbeae65" => Some("roberta-bpe"),
        "ad851be1dba641f2e3711822f816db2c265f788b37c63b4e1aeacb9ee92de8eb" => Some("gigachat"),
        "d4c8f286ea6b520b3d495c4455483cfa2302c0cfcd4be05d781b6a8a0a7cdaf1" => Some("megrez"),
        "877081d19cf6996e2c4ff0e1236341e9b7bde288f5311a56a937f0afbbb3aeb5" => Some("deepseek-v3"),
        "b3f499bb4255f8ca19fccd664443283318f2fd2414d5e0b040fbdd0cc195d6c5" => {
            Some("deepseek-r1-qwen")
        }
        "ccc2ef013c104be7bae2965776d611e1d7a8a2a9c547dd93a682c9a9fc80352e" => Some("gpt-4o"),
        "7dec86086fcc38b66b7bc1575a160ae21cf705be7718b9d5598190d7c12db76f" => Some("superbpe"),
        "1994ffd01900cfb37395608534236ecd63f2bd5995d6cb1004dda1af50240f15" => Some("trillion"),
        "96a5f08be6259352137b512d4157e333e21df7edd3fcd152990608735a65b224" => Some("bailingmoe"),
        "d353350c764d8c3b39c763113960e4fb4919bea5fbf208a0e3b22e8469dc7406" => Some("llama4"),
        "0e9433cbbb161f89e264eb32e8e64bfe69e834973ffca5d41d3948a604a3e2a3" => Some("pixtral"),
        "d5f1dd6f980fec569fb218a81a7658ac45fc56b38c5a0adeb1c232fbe04ef5ec" => Some("seed-coder"),
        "b0a6b1c0bd5998ebd9df08611efde34a4ff03faed45ae09c43e6b31ebd4b94cf" => Some("a.x-4.0"),
        "f6791d196f87ce6b56a7d234be618e0d58f8cda3549416635b2bebcd22cd95c4" => Some("midm-2.0"),
        "169bf0296a13c4d9b7672313f749eb36501d931022de052aad6e36f2bf34dd51" => Some("lfm2"),
        "2085e1638f6c377a0aa4ead21b27bb4cb941bf800df86ed391011769c1758dfb" => Some("exaone4"),
        "a1e163ecab2e718a4c829d1148b6e86824ec36163bb71941c3dca9cd5ac25756" => Some("mellum"),
        "a0b64b4385f123663873756336c085744376d015ff328bb1d901598f63c44152" => Some("modern-bert"),
        "49fc0303c9e0d2c2c565c510f64b2d9b271276acdcdadff733249eda9f7d59df" => Some("afmoe"),
        "9b1be57e70d20d9501b2b3186e792d81181ae36ada3903c26f9fea418cf87206" => Some("bailingmoe2"),
        "53e325976a6e142379c19b09afcae354f2f496f147afa8f9e189a33fe4e3024e" => {
            Some("granite-docling")
        }
        "f4f37b6c8eb9ea29b3eac6bb8c8487c5ab7885f8d8022e67edc1c68ce8403e95" => Some("minimax-m2"),
        "4a2e2abae11ca2b86d570fc5b44be4d5eb5e72cc8f22dd136a94b37da83ab665" => Some("kormo"),
        "9d70134b369a70e5735009b6de918f7581b5211f7c074d1f89f753aea8248af1" => Some("youtu"),
        "16389f0a1f51ee53e562ffd51c371dc508639ab0e4261502071836e50e223e91" => Some("solar-open"),
        "6c81ce329e0802883b22eabab0d3fa48357337ef1ecb45443828bf1f6254833f" => Some("exaone-moe"),
        "d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4" => Some("qwen35"),
        "b4b8ca1f9769494fbd956ebc4c249de6131fb277a4a3345a7a92c7dd7a55808d" => Some("joyai-llm"),
        "e4d54df1ebc1f2b91acd986c5b51aa50837d5faf7c7398e73c1f9e9ee5d19869" => Some("kanana2"),
        "862f827721df956049dff5ca81a57f29e575280bc622e290d3bf4e35eca29015" => Some("f2llmv2"),
        _ => None,
    }
}

fn infer_pre_tokenizer(
    tokenizer: &tokenizers::Tokenizer,
    config_json: Option<&serde_json::Value>,
    tokenizer_config: Option<&serde_json::Value>,
    tokenizer_json: &serde_json::Value,
) -> Option<String> {
    if let Some(value) = tokenizer_config.and_then(|json| json_str(json, "tokenizer_pre")) {
        return Some(value.to_string());
    }

    let ids = tokenizer.encode(LLAMA_PRETOKENIZER_CHECK, false).ok()?;
    let token_ids = ids.get_ids();
    let digest = Sha256::digest(format!("{token_ids:?}").as_bytes());
    let hash = lower_hex(digest.as_ref());
    if let Some(pre) = known_pre_tokenizer(&hash) {
        return Some(pre.to_string());
    }

    let model_type = config_json
        .and_then(|json| json_str(json, "model_type"))
        .unwrap_or_default()
        .to_ascii_lowercase();
    let architecture = config_json
        .and_then(|json| json.get("architectures"))
        .and_then(|value| value.as_array())
        .and_then(|array| array.first())
        .and_then(|value| value.as_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    if model_type.contains("qwen3_5")
        || architecture.contains("qwen3_5")
        || model_type.contains("qwen3_6")
        || architecture.contains("qwen3_6")
    {
        return Some("qwen35".to_string());
    }
    if model_type.contains("qwen") || architecture.contains("qwen") {
        return Some("qwen2".to_string());
    }
    if model_type.contains("deepseek") || architecture.contains("deepseek") {
        return Some("deepseek-v3".to_string());
    }
    if model_type.contains("llama") || architecture.contains("llama") {
        let has_begin_of_text = tokenizer_json
            .pointer("/model/vocab/<|begin_of_text|>")
            .is_some()
            || tokenizer.token_to_id("<|begin_of_text|>").is_some();
        if has_begin_of_text {
            return Some("llama-bpe".to_string());
        }
    }
    if model_type.contains("phi") || architecture.contains("phi") {
        return Some("phi-2".to_string());
    }
    Some("gpt-2".to_string())
}

fn tokenizer_config_bool(tokenizer_config: Option<&serde_json::Value>, key: &str) -> Option<bool> {
    tokenizer_config.and_then(|json| json_bool(json, key))
}

fn add_chat_template_metadata(
    builder: &mut pmetal_gguf::GgufBuilder,
    tokenizer_config: Option<&serde_json::Value>,
) {
    let Some(field) = tokenizer_config.and_then(|json| json.get("chat_template")) else {
        return;
    };

    if let Some(template) = field.as_str() {
        builder.add_string(pmetal_gguf::keys::TOKENIZER_CHAT_TEMPLATE, template);
        return;
    }

    let Some(choices) = field.as_array() else {
        return;
    };

    let mut names = Vec::new();
    let mut default_template = None;
    for choice in choices {
        let Some(name) = choice.get("name").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(template) = choice.get("template").and_then(|value| value.as_str()) else {
            continue;
        };
        let sanitized: String = name
            .chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
            .collect();
        if sanitized == "default" {
            default_template = Some(template.to_string());
        } else if !sanitized.is_empty() {
            names.push(sanitized.clone());
            builder.add_string(
                format!(
                    "{}{}",
                    pmetal_gguf::keys::TOKENIZER_CHAT_TEMPLATE_N_PREFIX,
                    sanitized
                ),
                template,
            );
        }
    }
    if !names.is_empty() {
        builder.add_string_array(pmetal_gguf::keys::TOKENIZER_CHAT_TEMPLATES, names);
    }
    if let Some(template) = default_template {
        builder.add_string(pmetal_gguf::keys::TOKENIZER_CHAT_TEMPLATE, template);
    }
}

fn parse_merges(
    tokenizer_json: &serde_json::Value,
    model_dir: &Path,
) -> anyhow::Result<Vec<String>> {
    if let Some(merges) = tokenizer_json
        .pointer("/model/merges")
        .and_then(|value| value.as_array())
    {
        let mut out = Vec::with_capacity(merges.len());
        for merge in merges {
            if let Some(s) = merge.as_str() {
                out.push(s.to_string());
            } else if let Some(pair) = merge.as_array() {
                if pair.len() == 2 {
                    if let (Some(a), Some(b)) = (pair[0].as_str(), pair[1].as_str()) {
                        out.push(format!("{a} {b}"));
                    }
                }
            }
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }

    let merges_path = model_dir.join("merges.txt");
    if !merges_path.exists() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(merges_path)?;
    Ok(content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect())
}

fn parse_unigram_scores(tokenizer_json: &serde_json::Value, vocab_size: usize) -> Option<Vec<f32>> {
    let vocab = tokenizer_json
        .pointer("/model/vocab")
        .and_then(|value| value.as_array())?;
    let mut scores = vec![0.0f32; vocab_size];
    for (idx, entry) in vocab.iter().enumerate() {
        let Some(score) = entry
            .as_array()
            .and_then(|parts| parts.get(1))
            .and_then(|value| value.as_f64())
        else {
            continue;
        };
        if idx < scores.len() {
            scores[idx] = score as f32;
        }
    }
    Some(scores)
}

fn add_tokenizer_metadata(
    builder: &mut pmetal_gguf::GgufBuilder,
    model_dir: &Path,
    config_json: Option<&serde_json::Value>,
) -> anyhow::Result<()> {
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        tracing::warn!(
            "tokenizer.json not found; GGUF output will contain weights/config but no tokenizer"
        );
        return Ok(());
    }

    let tokenizer_json_text = std::fs::read_to_string(&tokenizer_path)?;
    let tokenizer_json: serde_json::Value = serde_json::from_str(&tokenizer_json_text)?;
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|err| anyhow::anyhow!("Failed to load tokenizer.json: {err}"))?;
    let tokenizer_config = read_json_file(&model_dir.join("tokenizer_config.json"))?;
    let special_tokens_map = read_json_file(&model_dir.join("special_tokens_map.json"))?;

    let vocab = tokenizer.get_vocab(true);
    let vocab_size = config_json
        .and_then(|json| json_usize(json, "vocab_size"))
        .unwrap_or_else(|| tokenizer.get_vocab_size(true));
    let mut tokens = vec![String::new(); vocab_size];
    let mut token_types = vec![TOKEN_TYPE_UNUSED; vocab_size];
    let added = tokenizer.get_added_tokens_decoder();

    for (token, id) in vocab {
        let idx = id as usize;
        if idx >= vocab_size {
            continue;
        }
        let token_type = match added.get(&id) {
            Some(added_token)
                if added_token.special || looks_special_token(&added_token.content) =>
            {
                TOKEN_TYPE_CONTROL
            }
            Some(_) => TOKEN_TYPE_USER_DEFINED,
            None => TOKEN_TYPE_NORMAL,
        };
        tokens[idx] = token;
        token_types[idx] = token_type;
    }

    for (idx, token) in tokens.iter_mut().enumerate() {
        if token.is_empty() {
            *token = format!("[PAD{idx}]");
        }
    }

    let tokenizer_model = tokenizer_model_kind(&tokenizer_json, config_json);
    builder.add_string(pmetal_gguf::keys::TOKENIZER_MODEL, tokenizer_model.as_str());
    if tokenizer_model == "gpt2" || tokenizer_model == "gemma4" {
        if let Some(pre) = infer_pre_tokenizer(
            &tokenizer,
            config_json,
            tokenizer_config.as_ref(),
            &tokenizer_json,
        ) {
            builder.add_string(pmetal_gguf::keys::TOKENIZER_PRE, pre);
        }
    } else if tokenizer_model == "bert" {
        builder.add_string(pmetal_gguf::keys::TOKENIZER_PRE, "default");
        if let Some(type_vocab_size) =
            config_json.and_then(|json| json_u32(json, "type_vocab_size"))
        {
            builder.add_u32(
                pmetal_gguf::keys::TOKENIZER_TOKEN_TYPE_COUNT,
                type_vocab_size,
            );
        }
    } else if tokenizer_model == "llama" || tokenizer_model == "t5" {
        builder.add_string(pmetal_gguf::keys::TOKENIZER_PRE, "default");
    }

    builder.add_string_array(pmetal_gguf::keys::TOKENIZER_TOKENS, tokens);
    builder.add_i32_array(pmetal_gguf::keys::TOKENIZER_TOKEN_TYPE, token_types);

    let merges = parse_merges(&tokenizer_json, model_dir)?;
    if !merges.is_empty() {
        builder.add_string_array(pmetal_gguf::keys::TOKENIZER_MERGES, merges);
    }
    if let Some(scores) = parse_unigram_scores(&tokenizer_json, vocab_size) {
        builder.add_f32_array(pmetal_gguf::keys::TOKENIZER_SCORES, scores);
    }

    if let Some(id) = resolve_special_id(
        "bos_token",
        &["<s>", "<|begin_of_text|>", "<bos>"],
        &tokenizer,
        tokenizer_config.as_ref(),
        special_tokens_map.as_ref(),
    ) {
        builder.add_u32(pmetal_gguf::keys::TOKENIZER_BOS_TOKEN_ID, id);
    }
    if let Some(id) = resolve_special_id(
        "eos_token",
        &["</s>", "<|endoftext|>", "<|end_of_text|>", "<eos>"],
        &tokenizer,
        tokenizer_config.as_ref(),
        special_tokens_map.as_ref(),
    ) {
        builder.add_u32(pmetal_gguf::keys::TOKENIZER_EOS_TOKEN_ID, id);
    }
    if let Some(id) = resolve_special_id(
        "unk_token",
        &["<unk>", "[UNK]"],
        &tokenizer,
        tokenizer_config.as_ref(),
        special_tokens_map.as_ref(),
    ) {
        builder.add_u32(pmetal_gguf::keys::TOKENIZER_UNK_TOKEN_ID, id);
    }
    if let Some(id) = resolve_special_id(
        "pad_token",
        &["<pad>", "[PAD]", "<|pad|>", "<|finetune_right_pad_id|>"],
        &tokenizer,
        tokenizer_config.as_ref(),
        special_tokens_map.as_ref(),
    ) {
        builder.add_u32(pmetal_gguf::keys::TOKENIZER_PAD_TOKEN_ID, id);
    }
    add_special_id_from_configs(
        builder,
        "sep_token",
        pmetal_gguf::keys::TOKENIZER_SEP_TOKEN_ID,
        &tokenizer,
        tokenizer_config.as_ref(),
        special_tokens_map.as_ref(),
    );
    add_special_id_from_configs(
        builder,
        "mask_token",
        pmetal_gguf::keys::TOKENIZER_MASK_TOKEN_ID,
        &tokenizer,
        tokenizer_config.as_ref(),
        special_tokens_map.as_ref(),
    );
    add_special_id_from_configs(
        builder,
        "eot_token",
        pmetal_gguf::keys::TOKENIZER_EOT_TOKEN_ID,
        &tokenizer,
        tokenizer_config.as_ref(),
        special_tokens_map.as_ref(),
    );
    add_special_id_from_configs(
        builder,
        "eom_token",
        pmetal_gguf::keys::TOKENIZER_EOM_TOKEN_ID,
        &tokenizer,
        tokenizer_config.as_ref(),
        special_tokens_map.as_ref(),
    );

    if let Some(value) = tokenizer_config_bool(tokenizer_config.as_ref(), "add_bos_token") {
        builder.add_bool(pmetal_gguf::keys::TOKENIZER_ADD_BOS, value);
    }
    if let Some(value) = tokenizer_config_bool(tokenizer_config.as_ref(), "add_eos_token") {
        builder.add_bool(pmetal_gguf::keys::TOKENIZER_ADD_EOS, value);
    }

    add_chat_template_metadata(builder, tokenizer_config.as_ref());

    Ok(())
}

fn looks_special_token(token: &str) -> bool {
    (token.starts_with('<') && token.ends_with('>'))
        || (token.starts_with('[') && token.ends_with(']'))
}

fn add_special_id_from_configs(
    builder: &mut pmetal_gguf::GgufBuilder,
    token_key: &str,
    gguf_key: &str,
    tokenizer: &tokenizers::Tokenizer,
    tokenizer_config: Option<&serde_json::Value>,
    special_tokens_map: Option<&serde_json::Value>,
) {
    let token = special_tokens_map
        .and_then(|json| json.get(token_key))
        .and_then(extract_token_string)
        .or_else(|| {
            tokenizer_config
                .and_then(|json| json.get(token_key))
                .and_then(extract_token_string)
        });
    if let Some(token) = token {
        if let Some(id) = tokenizer.token_to_id(token) {
            builder.add_u32(gguf_key, id);
        }
    }
}

fn resolve_special_id(
    token_key: &str,
    fallback_tokens: &[&str],
    tokenizer: &tokenizers::Tokenizer,
    tokenizer_config: Option<&serde_json::Value>,
    special_tokens_map: Option<&serde_json::Value>,
) -> Option<u32> {
    let configured = special_tokens_map
        .and_then(|json| json.get(token_key))
        .and_then(extract_token_string)
        .or_else(|| {
            tokenizer_config
                .and_then(|json| json.get(token_key))
                .and_then(extract_token_string)
        });
    configured
        .and_then(|token| tokenizer.token_to_id(token))
        .or_else(|| {
            fallback_tokens
                .iter()
                .find_map(|token| tokenizer.token_to_id(token))
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ExpertProjection {
    Gate,
    Up,
    Down,
}

impl ExpertProjection {
    fn gguf_suffix(self) -> &'static str {
        match self {
            Self::Gate => "ffn_gate_exps.weight",
            Self::Up => "ffn_up_exps.weight",
            Self::Down => "ffn_down_exps.weight",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ExpertGroupKey {
    layer: usize,
    projection: ExpertProjection,
}

#[derive(Debug, Clone)]
struct ExpertTensor {
    expert: usize,
    name: String,
}

fn parse_expert_tensor_name(name: &str) -> Option<(ExpertGroupKey, usize)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, suffix) = rest.split_once('.')?;
    let layer = layer_str.parse::<usize>().ok()?;

    let patterns = [
        (
            "mlp.experts.",
            [
                ("gate_proj.weight", ExpertProjection::Gate),
                ("up_proj.weight", ExpertProjection::Up),
                ("down_proj.weight", ExpertProjection::Down),
            ],
        ),
        (
            "block_sparse_moe.experts.",
            [
                ("w1.weight", ExpertProjection::Gate),
                ("w3.weight", ExpertProjection::Up),
                ("w2.weight", ExpertProjection::Down),
            ],
        ),
    ];

    for (prefix, projections) in patterns {
        let Some(expert_rest) = suffix.strip_prefix(prefix) else {
            continue;
        };
        let (expert_str, projection_suffix) = expert_rest.split_once('.')?;
        let expert = expert_str.parse::<usize>().ok()?;
        for (candidate, projection) in projections {
            if projection_suffix == candidate {
                return Some((ExpertGroupKey { layer, projection }, expert));
            }
        }
    }

    None
}

fn collect_expert_groups<'a>(
    names: impl Iterator<Item = &'a str>,
) -> HashMap<ExpertGroupKey, Vec<ExpertTensor>> {
    let mut groups: HashMap<ExpertGroupKey, Vec<ExpertTensor>> = HashMap::new();
    for name in names {
        if let Some((key, expert)) = parse_expert_tensor_name(name) {
            groups.entry(key).or_default().push(ExpertTensor {
                expert,
                name: name.to_string(),
            });
        }
    }
    groups.retain(|_, tensors| tensors.len() > 1);
    groups
}

/// The experts of a group in expert order, and the shape of their stack
/// `[experts, out, in]`; the experts must be the run `0..n` and agree in shape.
fn expert_stack_layout(
    group: &[ExpertTensor],
    shapes: &HashMap<&str, &Vec<usize>>,
) -> anyhow::Result<(Vec<ExpertTensor>, Vec<u64>)> {
    let mut sorted = group.to_vec();
    sorted.sort_by_key(|tensor| tensor.expert);
    for (expected, tensor) in sorted.iter().enumerate() {
        if tensor.expert != expected {
            anyhow::bail!(
                "Expert tensors for {} are not contiguous: expected expert {}, found {}",
                tensor.name,
                expected,
                tensor.expert
            );
        }
    }
    let shape_of = |tensor: &ExpertTensor| {
        shapes
            .get(tensor.name.as_str())
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Expert tensor {} is not a float tensor", tensor.name))
    };
    let base_shape = shape_of(&sorted[0])?;
    if base_shape.len() != 2 {
        anyhow::bail!(
            "Expert tensor {} must be 2D for GGUF expert stacking, got {:?}",
            sorted[0].name,
            base_shape
        );
    }
    for tensor in &sorted {
        let shape = shape_of(tensor)?;
        if shape != base_shape {
            anyhow::bail!(
                "Expert tensor {} shape {:?} does not match {:?}",
                tensor.name,
                shape,
                base_shape
            );
        }
    }
    let mut shape = Vec::with_capacity(base_shape.len() + 1);
    shape.push(sorted.len() as u64);
    shape.extend(base_shape.iter().map(|&dim| dim as u64));
    Ok((sorted, shape))
}

fn is_quantizable_gguf_tensor(gguf_name: &str, shape: &[u64]) -> bool {
    if !(shape.len() == 2 || shape.len() == 3) || !gguf_name.ends_with(".weight") {
        return false;
    }
    let name = gguf_name.to_ascii_lowercase();
    if name.contains("_norm.weight")
        || name.contains("norm.weight")
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
        || name.contains(".patch_merger")
    {
        return false;
    }
    true
}

fn parse_layer_from_gguf_name(gguf_name: &str) -> Option<usize> {
    let rest = gguf_name.strip_prefix("blk.")?;
    let (layer, _) = rest.split_once('.')?;
    layer.parse::<usize>().ok()
}

fn use_more_bits(layer: usize, n_layers: usize) -> bool {
    if n_layers == 0 {
        return false;
    }
    layer < n_layers / 8 || layer >= 7 * n_layers / 8 || layer.saturating_sub(n_layers / 8) % 3 == 2
}

fn llama_default_type(method: crate::cli::quantize::QuantizeMethod) -> pmetal_gguf::GgmlType {
    method.to_ggml_type().unwrap_or(pmetal_gguf::GgmlType::Q4K)
}

fn llama_style_tensor_type(
    method: crate::cli::quantize::QuantizeMethod,
    gguf_name: &str,
    shape: &[u64],
    config_json: Option<&serde_json::Value>,
    proposed: pmetal_gguf::GgmlType,
) -> pmetal_gguf::GgmlType {
    use pmetal_gguf::GgmlType;

    if matches!(
        method,
        crate::cli::quantize::QuantizeMethod::F32
            | crate::cli::quantize::QuantizeMethod::F16
            | crate::cli::quantize::QuantizeMethod::Bf16
    ) {
        return llama_default_type(method);
    }

    if !is_quantizable_gguf_tensor(gguf_name, shape) {
        return GgmlType::F16;
    }

    let mut dtype = if matches!(method, crate::cli::quantize::QuantizeMethod::Dynamic) {
        proposed
    } else {
        llama_default_type(method)
    };
    let n_layers = config_json
        .and_then(|json| json_usize(json, "num_hidden_layers"))
        .unwrap_or(0);
    let n_heads = config_json
        .and_then(|json| json_usize(json, "num_attention_heads"))
        .unwrap_or(0);
    let n_kv_heads = config_json
        .and_then(|json| json_usize(json, "num_key_value_heads"))
        .unwrap_or(n_heads.max(1));
    let n_gqa = n_heads.checked_div(n_kv_heads).unwrap_or(1);
    let layer = parse_layer_from_gguf_name(gguf_name).unwrap_or(0);

    if gguf_name == "output.weight" || gguf_name == "token_embd.weight" {
        return match method {
            crate::cli::quantize::QuantizeMethod::Q1_0
            | crate::cli::quantize::QuantizeMethod::Tq1_0
            | crate::cli::quantize::QuantizeMethod::Tq2_0
            | crate::cli::quantize::QuantizeMethod::Mxfp4
            | crate::cli::quantize::QuantizeMethod::Nvfp4 => GgmlType::F16,
            _ if dtype == GgmlType::Q8_0 => GgmlType::Q8_0,
            crate::cli::quantize::QuantizeMethod::Dynamic
            | crate::cli::quantize::QuantizeMethod::Q2K
            | crate::cli::quantize::QuantizeMethod::Q3KM
            | crate::cli::quantize::QuantizeMethod::Q3KS
            | crate::cli::quantize::QuantizeMethod::Q3KL
            | crate::cli::quantize::QuantizeMethod::Q4KM
            | crate::cli::quantize::QuantizeMethod::Q4KS
            | crate::cli::quantize::QuantizeMethod::Q5KM
            | crate::cli::quantize::QuantizeMethod::Q5KS
            | crate::cli::quantize::QuantizeMethod::Iq4Nl
            | crate::cli::quantize::QuantizeMethod::Iq4Xs
            | crate::cli::quantize::QuantizeMethod::Q6K => GgmlType::Q6K,
            _ => dtype,
        };
    }

    if gguf_name.contains("attn_v.weight") {
        dtype = match method {
            crate::cli::quantize::QuantizeMethod::Q2K => {
                if n_gqa >= 4 {
                    GgmlType::Q4K
                } else {
                    GgmlType::Q3K
                }
            }
            crate::cli::quantize::QuantizeMethod::Q3KM => {
                if layer < 2 {
                    GgmlType::Q5K
                } else {
                    GgmlType::Q4K
                }
            }
            crate::cli::quantize::QuantizeMethod::Q3KL => GgmlType::Q5K,
            crate::cli::quantize::QuantizeMethod::Q4KM
            | crate::cli::quantize::QuantizeMethod::Q5KM => {
                if use_more_bits(layer, n_layers) {
                    GgmlType::Q6K
                } else {
                    dtype
                }
            }
            crate::cli::quantize::QuantizeMethod::Q4KS => {
                if layer < 4 {
                    GgmlType::Q5K
                } else {
                    dtype
                }
            }
            _ => dtype,
        };
    } else if gguf_name.contains("ffn_down.weight") || gguf_name.contains("ffn_down_exps.weight") {
        dtype = match method {
            crate::cli::quantize::QuantizeMethod::Q2K => GgmlType::Q3K,
            crate::cli::quantize::QuantizeMethod::Q3KM => {
                if layer < n_layers / 16 {
                    GgmlType::Q5K
                } else if use_more_bits(layer, n_layers) {
                    GgmlType::Q4K
                } else {
                    GgmlType::Q3K
                }
            }
            crate::cli::quantize::QuantizeMethod::Q3KL => GgmlType::Q5K,
            crate::cli::quantize::QuantizeMethod::Q4KM
            | crate::cli::quantize::QuantizeMethod::Q5KM => {
                if use_more_bits(layer, n_layers) {
                    GgmlType::Q6K
                } else {
                    dtype
                }
            }
            crate::cli::quantize::QuantizeMethod::Q4KS => {
                if layer < n_layers / 8 {
                    GgmlType::Q5K
                } else {
                    dtype
                }
            }
            _ => dtype,
        };
    } else if gguf_name.contains("attn_output.weight") {
        dtype = match method {
            crate::cli::quantize::QuantizeMethod::Q2K => GgmlType::Q3K,
            crate::cli::quantize::QuantizeMethod::Q3KM => GgmlType::Q4K,
            crate::cli::quantize::QuantizeMethod::Q3KL => GgmlType::Q5K,
            _ => dtype,
        };
    } else if gguf_name.contains("attn_qkv.weight") {
        dtype = match method {
            crate::cli::quantize::QuantizeMethod::Q3KM
            | crate::cli::quantize::QuantizeMethod::Q3KL => GgmlType::Q4K,
            crate::cli::quantize::QuantizeMethod::Q4KM => GgmlType::Q5K,
            crate::cli::quantize::QuantizeMethod::Q5KM => GgmlType::Q6K,
            _ => dtype,
        };
    }

    dtype
}

/// Run `pmetal quantize`: GGUF (the default) or MLX safetensors, from the base
/// model, or from the base model with a LoRA adapter fused in first.
pub(crate) async fn run_quantize(spec: pmetal_core::jobs::QuantizeSpec) -> anyhow::Result<()> {
    use clap::ValueEnum;

    if spec.format == "mlx" {
        // The MLX path takes no adapter: fuse it first with `pmetal fuse`.
        anyhow::ensure!(
            spec.lora.is_none(),
            "--lora is not supported with --format mlx; fuse the adapter first with `pmetal fuse`, \
             then quantize the fused model"
        );
        anyhow::ensure!(
            spec.imatrix.is_none() && spec.calibration_dataset.is_none(),
            "--imatrix and --calibration-dataset weight the GGUF quantizers and have no effect \
             on --format mlx output"
        );
        return run_quantization_mlx(
            &spec.model,
            &spec.output,
            spec.bits,
            spec.group_size,
            spec.target_bpw,
        )
        .await;
    }

    let method = crate::cli::quantize::QuantizeMethod::from_str(&spec.method, true)
        .map_err(|e| anyhow::anyhow!("--method: {e}"))?;
    let importance = if let Some(dataset) = spec.calibration_dataset.as_deref() {
        ImportanceSource::Measure {
            dataset,
            ctx_size: spec.calibration_ctx_size,
            chunks: spec.calibration_chunks,
        }
    } else if let Some(path) = spec.imatrix.as_deref() {
        ImportanceSource::File(path)
    } else {
        ImportanceSource::None
    };
    let Some(lora) = spec.lora.as_deref() else {
        return run_quantization(
            &spec.model,
            &spec.output,
            importance,
            method,
            spec.kl_calibrate,
            spec.target_bpw,
            spec.kl_threshold,
        )
        .await;
    };

    // Fuse the adapter into a scratch copy beside the output, quantize that,
    // and remove it however the run ends: it is as large as the model.
    #[cfg(not(feature = "lora"))]
    {
        let _ = lora;
        anyhow::bail!("fusing a LoRA adapter while quantizing needs the `lora` feature");
    }
    #[cfg(feature = "lora")]
    {
        let output = crate::validate_output_path(&spec.output, "quantization output")?;
        let scratch_parent = output
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let scratch = tempfile::Builder::new()
            .prefix(".pmetal-fused-")
            .tempdir_in(&scratch_parent)
            .map_err(|e| {
                anyhow::anyhow!(
                    "could not make a scratch directory in '{}': {e}",
                    scratch_parent.display()
                )
            })?;
        let fused = scratch.path().join("model");
        crate::commands::fuse::run_fuse(pmetal_core::jobs::FuseSpec {
            model: spec.model.clone(),
            lora: lora.to_string(),
            output: fused.to_string_lossy().into_owned(),
            ..Default::default()
        })
        .await?;
        run_quantization(
            &fused.to_string_lossy(),
            &spec.output,
            importance,
            method,
            spec.kl_calibrate,
            spec.target_bpw,
            spec.kl_threshold,
        )
        .await
    }
}

/// Where the quantization's per-channel importance comes from.
#[derive(Clone, Copy)]
pub(crate) enum ImportanceSource<'a> {
    /// None: every weight counts the same.
    None,
    /// An importance matrix file, measured earlier (`pmetal imatrix`).
    File(&'a str),
    /// Measured now, on this calibration data, from the model being quantized.
    Measure {
        dataset: &'a str,
        ctx_size: usize,
        /// 0: all the data makes.
        chunks: usize,
    },
}

/// Run model quantization.
pub(crate) async fn run_quantization(
    model_path: &str,
    output_path: &str,
    importance: ImportanceSource<'_>,
    method: crate::cli::quantize::QuantizeMethod,
    kl_calibrate: bool,
    target_bpw: Option<f32>,
    kl_threshold: f64,
) -> anyhow::Result<()> {
    use pmetal_gguf::{
        GgufBuilder, TensorInfo,
        dynamic::{
            CalibrationMap, DynamicQuantizationConfig, DynamicQuantizer, KlCalibrationConfig,
        },
        imatrix::IMatrix,
        quantize::{fit_to_row_length, quantize_tensor},
    };

    println!("========================================");
    println!("  PMetal GGUF Quantization");
    println!("========================================");
    println!("Model:    {}", model_path);
    println!("Output:   {}", output_path);
    println!("Method:   {}", method.as_str());
    match importance {
        ImportanceSource::None => {}
        ImportanceSource::File(path) => println!("IMatrix:  {path}"),
        ImportanceSource::Measure { dataset, .. } => {
            println!("IMatrix:  measured on {dataset}")
        }
    }
    if kl_calibrate {
        println!("KL Calib: enabled (threshold={:.4})", kl_threshold);
        if let Some(bpw) = target_bpw {
            println!("Target BPW: {:.2}", bpw);
        }
    }
    println!("========================================\n");

    // Resolve HuggingFace model ID to local path
    tracing::info!("Resolving model: {}", model_path);
    let resolved_model_path: PathBuf =
        pmetal_hub::resolve_model_path(model_path, None, None).await?;

    // 1. The importance matrix: read, or measured on the model being quantized
    // (which is loaded for that and dropped before any weight is quantized).
    let imatrix = match importance {
        ImportanceSource::None => None,
        ImportanceSource::File(path) => {
            tracing::info!("Loading IMatrix from {}", path);
            Some(IMatrix::load(Path::new(path))?)
        }
        ImportanceSource::Measure {
            dataset,
            ctx_size,
            chunks,
        } => {
            println!("Measuring the importance matrix...");
            let run = crate::commands::imatrix::measure_imatrix(
                &resolved_model_path,
                model_path,
                dataset,
                ctx_size,
                (chunks > 0).then_some(chunks),
                pmetal_core::jobs::ImatrixSpec::default().batch_size,
            )
            .await?;
            println!(
                "Measured {} matrices over {} chunks ({} tokens)\n",
                run.imatrix.len(),
                run.chunks,
                run.tokens
            );
            Some(run.imatrix)
        }
    };

    // 2. Initialize quantizer
    let quantizer = if let Some(base_type) = method.to_ggml_type() {
        let config = DynamicQuantizationConfig {
            base_type,
            high_precision_type: base_type,
            fallback_type: base_type,
            ..Default::default()
        };
        // The imatrix still weights the quantization error of a fixed-type
        // method; every tier is the same type, so it does not change the type.
        DynamicQuantizer::new(config, imatrix)
    } else {
        let config = DynamicQuantizationConfig::default();
        DynamicQuantizer::new(config, imatrix)
    };

    // 3. Open the weights. They are read one tensor at a time, here and below:
    // the model is never held in memory (only one tensor, its f32 copy and its
    // quantized bytes are, and the experts of one layer's projection).
    let validated_output = crate::validate_output_path(output_path, "quantization output")?;
    let mut source = crate::commands::quantize_source::open(&resolved_model_path)?;
    let metas: Vec<TensorMeta> = source.tensors().to_vec();
    tracing::info!("Found {} float tensors", metas.len());

    // 4. Detect Architecture
    let config_path = resolved_model_path.join("config.json");
    let mut architecture = "llama".to_string();
    let mut config_json: Option<serde_json::Value> = None;

    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(archs) = json.get("architectures").and_then(|v| v.as_array()) {
                    if let Some(arch_str) = archs.first().and_then(|v| v.as_str()) {
                        architecture = match arch_str {
                            "LlamaForCausalLM" | "LlamaModel" => "llama".to_string(),
                            "MistralForCausalLM" | "MixtralForCausalLM" => "mistral".to_string(),
                            "Qwen2ForCausalLM" | "Qwen2Model" => "qwen2".to_string(),
                            "Qwen2MoeForCausalLM" => "qwen2moe".to_string(),
                            "Qwen3ForCausalLM" | "Qwen3Model" => "qwen3".to_string(),
                            "Qwen3MoeForCausalLM" => "qwen3moe".to_string(),
                            "Qwen3NextForCausalLM" => "qwen3next".to_string(),
                            "Qwen3_5ForCausalLM" | "Qwen3_5ForConditionalGeneration" => {
                                "qwen35".to_string()
                            }
                            "Qwen3_5MoeForCausalLM" | "Qwen3_5MoeForConditionalGeneration" => {
                                "qwen35moe".to_string()
                            }
                            "GemmaForCausalLM" | "Gemma2ForCausalLM" => "gemma".to_string(),
                            "PhiForCausalLM" | "Phi3ForCausalLM" => "phi".to_string(),
                            "DeepseekV2ForCausalLM" | "DeepseekV3ForCausalLM" => {
                                "deepseek2".to_string()
                            }
                            "GptOssForCausalLM" => "gpt-oss".to_string(),
                            _ => {
                                tracing::warn!(
                                    "Unknown architecture '{}', defaulting to 'llama'",
                                    arch_str
                                );
                                "llama".to_string()
                            }
                        };
                        tracing::info!(
                            "Detected architecture: {} (from {})",
                            architecture,
                            arch_str
                        );
                    }
                }
                config_json = Some(json);
            }
        }
    } else {
        tracing::warn!("config.json not found, defaulting architecture to 'llama'");
    }

    // 5. Plan: which tensors the file holds and in what order, under the names
    // it gives them (the names the imatrix and the calibration are keyed by).
    // The routed experts of a layer's projection are one stacked tensor, which
    // is how they are scored and weighted below as well as how they are
    // written.
    tracing::info!("Planning the quantization...");
    let shapes: HashMap<&str, &Vec<usize>> = metas
        .iter()
        .map(|meta| (meta.name.as_str(), &meta.shape))
        .collect();
    let expert_groups = collect_expert_groups(metas.iter().map(|meta| meta.name.as_str()));
    let expert_source_keys: HashSet<&str> = expert_groups
        .values()
        .flat_map(|group| group.iter().map(|tensor| tensor.name.as_str()))
        .collect();

    let mut entries: Vec<PlanEntry> = Vec::new();
    // Stacked experts first, in layer order.
    let mut group_keys: Vec<ExpertGroupKey> = expert_groups.keys().copied().collect();
    group_keys.sort_by_key(|key| (key.layer, key.projection.gguf_suffix()));
    for key in group_keys {
        let (experts, shape) = expert_stack_layout(&expert_groups[&key], &shapes)?;
        entries.push(PlanEntry {
            gguf_name: format!("blk.{}.{}", key.layer, key.projection.gguf_suffix()),
            shape,
            from: Origin::Experts(experts),
        });
    }
    for meta in &metas {
        if expert_source_keys.contains(meta.name.as_str()) {
            continue;
        }
        entries.push(PlanEntry {
            gguf_name: pmetal_models::weight_format::WeightLoader::hf_to_gguf_name(&meta.name),
            shape: meta.shape.iter().map(|&d| d as u64).collect(),
            from: Origin::Tensor(meta.name.clone()),
        });
    }
    let read_entry = |source: &mut dyn crate::commands::quantize_source::TensorSource,
                      from: &Origin|
     -> anyhow::Result<Vec<f32>> {
        match from {
            Origin::Tensor(name) => source.read_f32(name),
            Origin::Experts(experts) => {
                let mut stacked = Vec::new();
                for expert in experts {
                    stacked.append(&mut source.read_f32(&expert.name)?);
                }
                Ok(stacked)
            }
        }
    };

    // 6. KL-divergence calibration pass (optional): each tensor is read, scored
    // and let go of; only the choices are kept.
    let calibration_map: CalibrationMap = if kl_calibrate {
        println!(
            "Running KL calibration pass over {} tensors...",
            entries.len()
        );
        let listing: Vec<(String, Vec<i32>)> = entries
            .iter()
            .map(|e| {
                (
                    e.gguf_name.clone(),
                    e.shape.iter().map(|&d| d as i32).collect(),
                )
            })
            .collect();
        let by_name: HashMap<&str, &Origin> = entries
            .iter()
            .map(|e| (e.gguf_name.as_str(), &e.from))
            .collect();
        let kl_config = KlCalibrationConfig {
            kl_threshold,
            target_bpw,
            ..Default::default()
        };
        let map = quantizer.calibrate_streaming(
            &listing,
            |gguf_name| read_entry(&mut *source, by_name[gguf_name]),
            &kl_config,
        )?;

        let tensor_sizes: Vec<(String, usize)> = entries
            .iter()
            .map(|e| {
                (
                    e.gguf_name.clone(),
                    e.shape.iter().product::<u64>() as usize,
                )
            })
            .collect();
        let summary = quantizer.summarize_calibration(&map, &tensor_sizes);
        println!(
            "Calibration complete: {} tensors, avg KL={:.6}, worst={} ({:.6}), est. BPW={:.2}",
            summary.total_tensors,
            summary.avg_kl_score,
            summary.worst_tensor,
            summary.max_kl_score,
            summary.estimated_bpw,
        );
        let mut type_vec: Vec<_> = summary.type_counts.iter().collect();
        type_vec.sort_by_key(|(t, _)| format!("{:?}", t));
        for (dtype, count) in type_vec {
            println!("  {:?}: {} tensors", dtype, count);
        }
        println!();
        map
    } else {
        CalibrationMap::new()
    };

    // 7. Metadata
    let mut builder = GgufBuilder::with_model(&architecture, "quantized-model");
    add_hf_config_metadata(&mut builder, &architecture, config_json.as_ref(), method);
    add_tokenizer_metadata(&mut builder, &resolved_model_path, config_json.as_ref())?;

    // Each tensor's type. The header of a GGUF file comes first and names every
    // tensor's type and size, which depend on the shapes alone.
    let choose_type = |gguf_name: &str, shape_u64: &[u64]| {
        let proposed = if calibration_map.is_empty() {
            quantizer.get_tensor_type(gguf_name, shape_u64)
        } else {
            quantizer.get_tensor_type_calibrated(gguf_name, shape_u64, &calibration_map)
        };
        let chosen =
            llama_style_tensor_type(method, gguf_name, shape_u64, config_json.as_ref(), proposed);
        // A block never straddles a row: fall back to a type whose block
        // divides the row, as the reference does.
        let n_per_row = shape_u64.last().copied().unwrap_or(1) as usize;
        fit_to_row_length(chosen, n_per_row)
    };
    let plan: Vec<PlannedTensor> = entries
        .into_iter()
        .map(|e| PlannedTensor {
            dtype: choose_type(&e.gguf_name, &e.shape),
            gguf_name: e.gguf_name,
            shape: e.shape,
            from: e.from,
        })
        .collect();

    // 8. Quantize and write, one tensor at a time. The file is built beside the
    // destination and moved into place when complete.
    let infos: Vec<TensorInfo> = plan
        .iter()
        .map(|t| TensorInfo::new(t.gguf_name.clone(), t.shape.clone(), t.dtype))
        .collect();
    let mut writer = builder
        .file_writer(&validated_output, infos)
        .map_err(|e| anyhow::anyhow!("could not start '{}': {e}", validated_output.display()))?;
    let total = plan.len();
    for (done, planned) in plan.iter().enumerate() {
        let data_f32 = read_entry(&mut *source, &planned.from)?;
        let n_per_row = planned.shape.last().copied().unwrap_or(1) as usize;
        match &planned.from {
            Origin::Tensor(name) => tracing::info!(
                "[{}/{total}] Quantizing {} ({}) to {:?}",
                done + 1,
                name,
                planned.gguf_name,
                planned.dtype
            ),
            Origin::Experts(experts) => tracing::info!(
                "[{}/{total}] Quantizing stacked expert tensor {} ({} experts) to {:?}",
                done + 1,
                planned.gguf_name,
                experts.len(),
                planned.dtype
            ),
        }
        let importance = quantizer.importance_for(&planned.gguf_name);
        let quantized = quantize_tensor(&data_f32, planned.dtype, n_per_row, importance.as_deref())
            .map_err(|e| {
                anyhow::anyhow!("Quantization error for {}: {:?}", planned.gguf_name, e)
            })?;
        drop(data_f32);
        match &planned.from {
            Origin::Tensor(name) => source.done_with(name),
            Origin::Experts(experts) => {
                for expert in experts {
                    source.done_with(&expert.name);
                }
            }
        }
        writer
            .write_tensor(&planned.gguf_name, &quantized)
            .map_err(|e| anyhow::anyhow!("could not write {}: {e}", planned.gguf_name))?;
    }
    writer
        .finish()
        .map_err(|e| anyhow::anyhow!("could not finish '{}': {e}", validated_output.display()))?;

    println!("Quantization complete!");
    Ok(())
}

/// A tensor of the output file before its type is chosen.
struct PlanEntry {
    gguf_name: String,
    /// Row-major, as the builder takes it (the writer reverses).
    shape: Vec<u64>,
    from: Origin,
}

/// One tensor of the output file, in file order.
struct PlannedTensor {
    gguf_name: String,
    /// Row-major, as the builder takes it (the writer reverses).
    shape: Vec<u64>,
    dtype: pmetal_gguf::GgmlType,
    from: Origin,
}

/// Where a planned tensor's values come from.
enum Origin {
    /// One tensor of the model, by its name there.
    Tensor(String),
    /// The experts of a layer's projection, stacked, in expert order.
    Experts(Vec<ExpertTensor>),
}

// ── MLX safetensors path ──────────────────────────────────────────────────────

/// Run MLX-format safetensors quantization with per-tensor quality-based bit allocation.
///
/// `output_path` is treated as a directory.  The directory is created if it
/// does not exist.  Inside it the function writes:
/// - `model.safetensors`   — quantized weights in MLX affine format
/// - `config.json`         — source config + injected `quantization_config`
/// - `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`,
///   `merges.txt`, `vocab.json`, `tokenizer.model` (copied if present)
pub(crate) async fn run_quantization_mlx(
    model_path: &str,
    output_path: &str,
    default_bits: i32,
    group_size: i32,
    target_bpw: Option<f32>,
) -> anyhow::Result<()> {
    use pmetal_runtime::mlx_quant;

    println!("========================================");
    println!("  PMetal MLX Safetensors Quantization");
    println!("========================================");
    println!("Model:      {}", model_path);
    println!("Output:     {}", output_path);
    println!("Bits:       {}", default_bits);
    println!("Group size: {}", group_size);
    if let Some(bpw) = target_bpw {
        println!("Target BPW: {:.2}", bpw);
    }
    println!("========================================\n");

    // 1. Resolve HuggingFace model ID to local path.
    println!("Resolving model: {}", model_path);
    let resolved_model_path: std::path::PathBuf =
        pmetal_hub::resolve_model_path(model_path, None, None).await?;

    // 2. Load all weights as InlineArray (stays on GPU, avoids f32 copy).
    println!("Loading weights...");
    let weights = pmetal_models::loader::load_weights(&resolved_model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;
    println!("Loaded {} tensors", weights.len());

    // 3. Run the full pipeline: evaluate quality → allocate bits → quantize → save.
    let output_dir = std::path::PathBuf::from(output_path);
    let effective_bpw = target_bpw.unwrap_or(default_bits as f32);
    let source_config = resolved_model_path.join("config.json");

    println!(
        "Evaluating tensor quality and allocating bits (target BPW={:.2})...",
        effective_bpw
    );

    let assignments = mlx_quant::quantize_model(
        &weights,
        &source_config,
        &output_dir,
        effective_bpw,
        group_size,
        mlx_quant::DEFAULT_BITS_CANDIDATES,
        &[], // no extra critical tensor patterns
    )
    .map_err(|e| anyhow::anyhow!("Quantization failed: {}", e))?;

    // 4. Print allocation summary.
    let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    let mut total_params: usize = 0;
    let mut total_weighted_bits: f64 = 0.0;
    for a in &assignments {
        *counts.entry(a.bits).or_insert(0) += 1;
        total_params += a.param_count;
        let bits = if a.bits == 0 { 16 } else { a.bits };
        total_weighted_bits += a.param_count as f64 * bits as f64;
    }
    let final_bpw = if total_params > 0 {
        total_weighted_bits / total_params as f64
    } else {
        0.0
    };

    println!("\nBit allocation summary:");
    let mut bit_keys: Vec<_> = counts.keys().collect();
    bit_keys.sort();
    for &bits in &bit_keys {
        let count = counts[bits];
        if *bits == 0 {
            println!("  bf16 (passthrough): {} tensors", count);
        } else {
            println!("  Q{}: {} tensors", *bits, count);
        }
    }
    println!("Effective BPW: {:.3}", final_bpw);
    println!("Total tensors: {}", assignments.len());

    // 5. Copy tokenizer files from source to output.
    let tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "tokenizer.model",
    ];
    for fname in &tokenizer_files {
        let src = resolved_model_path.join(fname);
        if src.exists() {
            let dst = output_dir.join(fname);
            std::fs::copy(&src, &dst).ok();
        }
    }

    println!("\nMLX quantization complete!");
    println!("Output: {}", output_dir.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmetal_gguf::{GgmlType, GgufBuilder, GgufContent, MetadataValue};
    use std::io::Cursor;

    #[test]
    fn llama_style_recipe_keeps_norms_dense_and_bumps_q4km_sensitive_tensors() {
        let config = serde_json::json!({
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8
        });

        assert_eq!(
            llama_style_tensor_type(
                crate::cli::quantize::QuantizeMethod::Q4KM,
                "blk.0.ffn_down.weight",
                &[4096, 14336],
                Some(&config),
                GgmlType::Q4K,
            ),
            GgmlType::Q6K
        );
        assert_eq!(
            llama_style_tensor_type(
                crate::cli::quantize::QuantizeMethod::Q4KM,
                "blk.0.attn_norm.weight",
                &[4096],
                Some(&config),
                GgmlType::Q4K,
            ),
            GgmlType::F16
        );
        assert_eq!(
            llama_style_tensor_type(
                crate::cli::quantize::QuantizeMethod::Q4KM,
                "output.weight",
                &[32000, 4096],
                Some(&config),
                GgmlType::Q4K,
            ),
            GgmlType::Q6K
        );
        assert_eq!(
            llama_style_tensor_type(
                crate::cli::quantize::QuantizeMethod::Tq1_0,
                "token_embd.weight",
                &[32000, 4096],
                Some(&config),
                GgmlType::Tq1_0,
            ),
            GgmlType::F16
        );
    }

    #[test]
    fn quantize_methods_have_matching_gguf_file_types() {
        use pmetal_gguf::FileType;

        assert_eq!(
            quantize_method_file_type(crate::cli::quantize::QuantizeMethod::Q4_0),
            FileType::MostlyQ4_0
        );
        assert_eq!(
            quantize_method_file_type(crate::cli::quantize::QuantizeMethod::Q5_1),
            FileType::MostlyQ5_1
        );
        assert_eq!(
            quantize_method_file_type(crate::cli::quantize::QuantizeMethod::Q1_0),
            FileType::MostlyQ1_0
        );
        assert_eq!(
            quantize_method_file_type(crate::cli::quantize::QuantizeMethod::Mxfp4),
            FileType::MostlyMxfp4Moe
        );
        assert_eq!(
            quantize_method_file_type(crate::cli::quantize::QuantizeMethod::Nvfp4),
            FileType::MostlyNvfp4
        );
    }

    #[test]
    fn known_pre_tokenizer_tracks_current_llama_hashes() {
        assert_eq!(
            known_pre_tokenizer("7e57df22b1fe23a7b1e1c7f3dc4e3f96d43a4eb0836d0c6bdc3436d7b2f1c664"),
            Some("hunyuan")
        );
        assert_eq!(
            known_pre_tokenizer("ccc2ef013c104be7bae2965776d611e1d7a8a2a9c547dd93a682c9a9fc80352e"),
            Some("gpt-4o")
        );
        assert_eq!(
            known_pre_tokenizer("b6dc8df998e1cfbdc4eac8243701a65afe638679230920b50d6f17d81c098166"),
            Some("olmo")
        );
        assert_eq!(
            known_pre_tokenizer("0876d13b50744004aa9aeae05e7b0647eac9d801b5ba4668afc01e709c15e19f"),
            Some("jina-v2-en")
        );
    }

    #[test]
    fn expert_tensor_names_are_grouped_by_layer_and_projection() {
        let (key, expert) =
            parse_expert_tensor_name("model.layers.12.mlp.experts.3.down_proj.weight").unwrap();
        assert_eq!(key.layer, 12);
        assert_eq!(key.projection, ExpertProjection::Down);
        assert_eq!(expert, 3);

        let (key, expert) =
            parse_expert_tensor_name("model.layers.2.block_sparse_moe.experts.7.w1.weight")
                .unwrap();
        assert_eq!(key.layer, 2);
        assert_eq!(key.projection, ExpertProjection::Gate);
        assert_eq!(expert, 7);
    }

    #[test]
    fn hf_config_metadata_writes_standard_llama_cpp_head_keys() {
        let config = serde_json::json!({
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 14336,
            "max_position_embeddings": 32768,
            "vocab_size": 128256,
            "head_dim": 128,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 0.000001
        });
        let mut builder = GgufBuilder::with_model("qwen3", "test");
        add_hf_config_metadata(
            &mut builder,
            "qwen3",
            Some(&config),
            crate::cli::quantize::QuantizeMethod::Q4KM,
        );

        let bytes = builder.build_to_bytes().unwrap();
        let content = GgufContent::read(&mut Cursor::new(bytes)).unwrap();
        assert!(matches!(
            content.get_metadata("qwen3.attention.key_length"),
            Some(MetadataValue::Uint32(128))
        ));
        assert!(matches!(
            content.get_metadata("qwen3.attention.value_length"),
            Some(MetadataValue::Uint32(128))
        ));
        assert!(matches!(
            content.get_metadata("qwen3.rope.dimension_count"),
            Some(MetadataValue::Uint32(128))
        ));
        assert!(matches!(
            content.get_metadata("general.file_type"),
            Some(MetadataValue::Uint32(15))
        ));
    }

    // ── streaming quantization, end to end on a tiny model ───────────────────

    /// A float in `[-1, 1]` that depends on where it is: tensors whose values
    /// differ, so a tensor written in the wrong slot or the wrong row shows.
    fn pattern(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * 37 + seed * 11) % 101) as f32 - 50.0) / 50.0)
            .collect()
    }

    /// `bf16` rounding of `values`, as stored in the checkpoint.
    fn as_stored(values: &[f32]) -> Vec<f32> {
        values
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_f32())
            .collect()
    }

    /// A checkpoint of bf16 tensors in two shards, with a config.
    fn write_tiny_model(dir: &Path, experts: bool) -> Vec<(String, Vec<usize>, Vec<f32>)> {
        use safetensors::tensor::{Dtype as StDtype, TensorView, serialize};
        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut add = |name: &str, shape: &[usize]| {
            let n: usize = shape.iter().product();
            let seed = tensors.len();
            tensors.push((
                name.to_string(),
                shape.to_vec(),
                as_stored(&pattern(seed, n)),
            ));
        };
        add("model.embed_tokens.weight", &[32, 64]);
        add("model.norm.weight", &[64]);
        add("lm_head.weight", &[32, 64]);
        add("model.layers.0.input_layernorm.weight", &[64]);
        add("model.layers.0.self_attn.q_proj.weight", &[64, 64]);
        add("model.layers.0.self_attn.k_proj.weight", &[32, 64]);
        add("model.layers.0.self_attn.v_proj.weight", &[32, 64]);
        add("model.layers.0.self_attn.o_proj.weight", &[64, 64]);
        if experts {
            for e in 0..3 {
                for (proj, shape) in [
                    ("gate_proj", [16, 64]),
                    ("up_proj", [16, 64]),
                    ("down_proj", [64, 16]),
                ] {
                    add(
                        &format!("model.layers.0.mlp.experts.{e}.{proj}.weight"),
                        &shape,
                    );
                }
            }
        } else {
            add("model.layers.0.mlp.gate_proj.weight", &[128, 64]);
            add("model.layers.0.mlp.up_proj.weight", &[128, 64]);
            add("model.layers.0.mlp.down_proj.weight", &[64, 128]);
        }

        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["LlamaForCausalLM"],"hidden_size":64,"num_hidden_layers":1,
               "num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":128,
               "vocab_size":32,"max_position_embeddings":128}"#,
        )
        .unwrap();
        // Two shards with an index, the layout of a real download.
        let (first, second) = tensors.split_at(tensors.len() / 2);
        let mut weight_map = serde_json::Map::new();
        for (file, part) in [
            ("model-00001-of-00002.safetensors", first),
            ("model-00002-of-00002.safetensors", second),
        ] {
            let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = part
                .iter()
                .map(|(name, shape, values)| {
                    let data = values
                        .iter()
                        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                        .collect();
                    (name.clone(), shape.clone(), data)
                })
                .collect();
            let views: Vec<(String, TensorView<'_>)> = bytes
                .iter()
                .map(|(name, shape, data)| {
                    weight_map.insert(name.clone(), serde_json::json!(file));
                    (
                        name.clone(),
                        TensorView::new(StDtype::BF16, shape.clone(), data).unwrap(),
                    )
                })
                .collect();
            std::fs::write(dir.join(file), serialize(views, None).unwrap()).unwrap();
        }
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({"weight_map": weight_map}).to_string(),
        )
        .unwrap();
        tensors
    }

    fn dequantized(model: &pmetal_gguf::GgufModel, name: &str) -> (Vec<f32>, Vec<u64>, GgmlType) {
        let info = model
            .tensor_info(name)
            .unwrap_or_else(|| panic!("no tensor {name}"));
        let shape: Vec<i32> = info.dimensions.iter().map(|&d| d as i32).collect();
        let values =
            pmetal_gguf::dequant::dequantize(model.tensor_data(name).unwrap(), info.dtype, &shape)
                .unwrap();
        (values, info.dimensions.clone(), info.dtype)
    }

    fn worst_difference(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    async fn quantize_dir(
        dir: &Path,
        out: &Path,
        method: crate::cli::quantize::QuantizeMethod,
        kl: bool,
        bpw: Option<f32>,
    ) -> anyhow::Result<()> {
        run_quantization(
            dir.to_str().unwrap(),
            out.to_str().unwrap(),
            ImportanceSource::None,
            method,
            kl,
            bpw,
            0.01,
        )
        .await
    }

    #[tokio::test]
    async fn a_model_is_quantized_tensor_by_tensor_into_a_valid_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = write_tiny_model(dir.path(), false);
        let out = dir.path().join("out").join("model.gguf");
        quantize_dir(
            dir.path(),
            &out,
            crate::cli::quantize::QuantizeMethod::Q8_0,
            false,
            None,
        )
        .await
        .unwrap();

        // Only the finished file is there: no temporary left beside it.
        let entries: Vec<_> = std::fs::read_dir(out.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(entries, ["model.gguf"]);

        let model = pmetal_gguf::GgufModel::open(&out).unwrap();
        assert_eq!(model.num_tensors(), tensors.len());
        for (hf_name, shape, values) in &tensors {
            let gguf_name = pmetal_models::weight_format::WeightLoader::hf_to_gguf_name(hf_name);
            let (got, dims, dtype) = dequantized(&model, &gguf_name);
            let want_dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
            assert_eq!(dims, want_dims, "{gguf_name}");
            let worst = worst_difference(&got, values);
            if shape.len() == 2 {
                assert_eq!(dtype, GgmlType::Q8_0, "{gguf_name}");
                assert!(worst < 0.01, "{gguf_name}: q8_0 differs by {worst}");
            } else {
                // A norm weight stays dense.
                assert!(
                    matches!(dtype, GgmlType::F32 | GgmlType::F16),
                    "{gguf_name}: {dtype:?}"
                );
                assert!(worst < 1e-3, "{gguf_name}: differs by {worst}");
            }
        }
    }

    /// The experts of a layer's projection are one stacked tensor
    /// `[experts, out, in]` in expert order, however the checkpoint lists them.
    #[tokio::test]
    async fn per_expert_tensors_are_stacked_in_expert_order() {
        let dir = tempfile::tempdir().unwrap();
        let tensors = write_tiny_model(dir.path(), true);
        let out = dir.path().join("moe.gguf");
        quantize_dir(
            dir.path(),
            &out,
            crate::cli::quantize::QuantizeMethod::Q8_0,
            false,
            None,
        )
        .await
        .unwrap();
        let model = pmetal_gguf::GgufModel::open(&out).unwrap();
        let (stack, dims, _) = dequantized(&model, "blk.0.ffn_up_exps.weight");
        assert_eq!(dims, [3, 16, 64]);
        let mut want = Vec::new();
        for e in 0..3 {
            let name = format!("model.layers.0.mlp.experts.{e}.up_proj.weight");
            want.extend(&tensors.iter().find(|(n, ..)| *n == name).unwrap().2);
        }
        let worst = worst_difference(&stack, &want);
        assert!(worst < 0.01, "the stacked experts differ by {worst}");
        // No per-expert tensor is left beside the stack.
        assert!(model.tensor_names().all(|n| !n.contains("experts.0")));
    }

    #[tokio::test]
    async fn kl_calibration_streams_too_and_picks_types_within_the_budget() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_model(dir.path(), false);
        let out = dir.path().join("kl.gguf");
        quantize_dir(
            dir.path(),
            &out,
            crate::cli::quantize::QuantizeMethod::Dynamic,
            true,
            Some(5.0),
        )
        .await
        .unwrap();
        let model = pmetal_gguf::GgufModel::open(&out).unwrap();
        assert!(model.num_tensors() > 0);
        let (_, _, dtype) = dequantized(&model, "blk.0.ffn_up.weight");
        assert_ne!(dtype, GgmlType::F32);
    }

    /// A shard cut short is an error, and the destination is not touched.
    #[tokio::test]
    async fn a_damaged_shard_fails_without_leaving_an_output() {
        let dir = tempfile::tempdir().unwrap();
        write_tiny_model(dir.path(), false);
        let shard = dir.path().join("model-00002-of-00002.safetensors");
        let bytes = std::fs::read(&shard).unwrap();
        std::fs::write(&shard, &bytes[..bytes.len() - 64]).unwrap();
        let out = dir.path().join("bad.gguf");
        let result = quantize_dir(
            dir.path(),
            &out,
            crate::cli::quantize::QuantizeMethod::Q8_0,
            false,
            None,
        )
        .await;
        assert!(result.is_err());
        assert!(!out.exists());
    }
}
