use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use pmetal_mlx::{Array, Dtype};
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

fn quantize_method_file_type(method: crate::QuantizeMethod) -> pmetal_gguf::FileType {
    use pmetal_gguf::FileType;
    match method {
        crate::QuantizeMethod::Dynamic => FileType::MostlyQ4KM,
        crate::QuantizeMethod::Q8_0 => FileType::MostlyQ8_0,
        crate::QuantizeMethod::Q8_1 => FileType::MostlyQ8_0,
        crate::QuantizeMethod::Q6K => FileType::MostlyQ6K,
        crate::QuantizeMethod::Q5KM => FileType::MostlyQ5KM,
        crate::QuantizeMethod::Q5KS => FileType::MostlyQ5KS,
        crate::QuantizeMethod::Q5_0 => FileType::MostlyQ5_0,
        crate::QuantizeMethod::Q5_1 => FileType::MostlyQ5_1,
        crate::QuantizeMethod::Q4KM => FileType::MostlyQ4KM,
        crate::QuantizeMethod::Q4KS => FileType::MostlyQ4KS,
        crate::QuantizeMethod::Q4_0 => FileType::MostlyQ4_0,
        crate::QuantizeMethod::Q4_1 => FileType::MostlyQ4_1,
        crate::QuantizeMethod::Q3KM => FileType::MostlyQ3KM,
        crate::QuantizeMethod::Q3KS => FileType::MostlyQ3KS,
        crate::QuantizeMethod::Q3KL => FileType::MostlyQ3KL,
        crate::QuantizeMethod::Q2K => FileType::MostlyQ2K,
        crate::QuantizeMethod::Q1_0 => FileType::MostlyQ1_0,
        crate::QuantizeMethod::Tq1_0 => FileType::MostlyTq1_0,
        crate::QuantizeMethod::Tq2_0 => FileType::MostlyTq2_0,
        crate::QuantizeMethod::Mxfp4 => FileType::MostlyMxfp4Moe,
        crate::QuantizeMethod::Nvfp4 => FileType::MostlyNvfp4,
        crate::QuantizeMethod::Bf16 => FileType::MostlyBf16,
        crate::QuantizeMethod::F16 => FileType::MostlyF16,
        crate::QuantizeMethod::F32 => FileType::AllF32,
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
    method: crate::QuantizeMethod,
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
        // llama.cpp uses key_length/value_length. Keep the older pmetal
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

    if model_type.contains("qwen3_5") || architecture.contains("qwen3_5") {
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

fn collect_expert_groups(
    weights: &HashMap<String, Array>,
) -> HashMap<ExpertGroupKey, Vec<ExpertTensor>> {
    let mut groups: HashMap<ExpertGroupKey, Vec<ExpertTensor>> = HashMap::new();
    for name in weights.keys() {
        if let Some((key, expert)) = parse_expert_tensor_name(name) {
            groups.entry(key).or_default().push(ExpertTensor {
                expert,
                name: name.clone(),
            });
        }
    }
    groups.retain(|_, tensors| tensors.len() > 1);
    groups
}

fn stack_expert_group(
    weights: &HashMap<String, Array>,
    group: &[ExpertTensor],
) -> anyhow::Result<(Vec<f32>, Vec<u64>)> {
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

    let first = weights
        .get(&sorted[0].name)
        .ok_or_else(|| anyhow::anyhow!("Missing expert tensor {}", sorted[0].name))?;
    let base_shape = first.shape();
    if base_shape.len() != 2 {
        anyhow::bail!(
            "Expert tensor {} must be 2D for GGUF expert stacking, got {:?}",
            sorted[0].name,
            base_shape
        );
    }

    let mut data = Vec::new();
    for tensor_ref in &sorted {
        let tensor = weights
            .get(&tensor_ref.name)
            .ok_or_else(|| anyhow::anyhow!("Missing expert tensor {}", tensor_ref.name))?;
        if tensor.shape() != base_shape {
            anyhow::bail!(
                "Expert tensor {} shape {:?} does not match {:?}",
                tensor_ref.name,
                tensor.shape(),
                base_shape
            );
        }
        let mut tensor_data = tensor_to_f32_vec(&tensor_ref.name, tensor)?
            .ok_or_else(|| anyhow::anyhow!("Expert tensor {} is not float", tensor_ref.name))?;
        data.append(&mut tensor_data);
    }

    let mut shape = Vec::with_capacity(base_shape.len() + 1);
    shape.push(sorted.len() as u64);
    shape.extend(base_shape.iter().map(|&dim| dim as u64));
    Ok((data, shape))
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

fn llama_default_type(method: crate::QuantizeMethod) -> pmetal_gguf::GgmlType {
    method.to_ggml_type().unwrap_or(pmetal_gguf::GgmlType::Q4K)
}

fn llama_style_tensor_type(
    method: crate::QuantizeMethod,
    gguf_name: &str,
    shape: &[u64],
    config_json: Option<&serde_json::Value>,
    proposed: pmetal_gguf::GgmlType,
) -> pmetal_gguf::GgmlType {
    use pmetal_gguf::GgmlType;

    if matches!(
        method,
        crate::QuantizeMethod::F32 | crate::QuantizeMethod::F16 | crate::QuantizeMethod::Bf16
    ) {
        return llama_default_type(method);
    }

    if !is_quantizable_gguf_tensor(gguf_name, shape) {
        return GgmlType::F16;
    }

    let mut dtype = if matches!(method, crate::QuantizeMethod::Dynamic) {
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
            crate::QuantizeMethod::Q1_0
            | crate::QuantizeMethod::Tq1_0
            | crate::QuantizeMethod::Tq2_0
            | crate::QuantizeMethod::Mxfp4
            | crate::QuantizeMethod::Nvfp4 => GgmlType::F16,
            _ if dtype == GgmlType::Q8_0 => GgmlType::Q8_0,
            crate::QuantizeMethod::Dynamic
            | crate::QuantizeMethod::Q2K
            | crate::QuantizeMethod::Q3KM
            | crate::QuantizeMethod::Q3KS
            | crate::QuantizeMethod::Q3KL
            | crate::QuantizeMethod::Q4KM
            | crate::QuantizeMethod::Q4KS
            | crate::QuantizeMethod::Q5KM
            | crate::QuantizeMethod::Q5KS
            | crate::QuantizeMethod::Q6K => GgmlType::Q6K,
            _ => dtype,
        };
    }

    if gguf_name.contains("attn_v.weight") {
        dtype = match method {
            crate::QuantizeMethod::Q2K => {
                if n_gqa >= 4 {
                    GgmlType::Q4K
                } else {
                    GgmlType::Q3K
                }
            }
            crate::QuantizeMethod::Q3KM => {
                if layer < 2 {
                    GgmlType::Q5K
                } else {
                    GgmlType::Q4K
                }
            }
            crate::QuantizeMethod::Q3KL => GgmlType::Q5K,
            crate::QuantizeMethod::Q4KM | crate::QuantizeMethod::Q5KM => {
                if use_more_bits(layer, n_layers) {
                    GgmlType::Q6K
                } else {
                    dtype
                }
            }
            crate::QuantizeMethod::Q4KS => {
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
            crate::QuantizeMethod::Q2K => GgmlType::Q3K,
            crate::QuantizeMethod::Q3KM => {
                if layer < n_layers / 16 {
                    GgmlType::Q5K
                } else if use_more_bits(layer, n_layers) {
                    GgmlType::Q4K
                } else {
                    GgmlType::Q3K
                }
            }
            crate::QuantizeMethod::Q3KL => GgmlType::Q5K,
            crate::QuantizeMethod::Q4KM | crate::QuantizeMethod::Q5KM => {
                if use_more_bits(layer, n_layers) {
                    GgmlType::Q6K
                } else {
                    dtype
                }
            }
            crate::QuantizeMethod::Q4KS => {
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
            crate::QuantizeMethod::Q2K => GgmlType::Q3K,
            crate::QuantizeMethod::Q3KM => GgmlType::Q4K,
            crate::QuantizeMethod::Q3KL => GgmlType::Q5K,
            _ => dtype,
        };
    } else if gguf_name.contains("attn_qkv.weight") {
        dtype = match method {
            crate::QuantizeMethod::Q3KM | crate::QuantizeMethod::Q3KL => GgmlType::Q4K,
            crate::QuantizeMethod::Q4KM => GgmlType::Q5K,
            crate::QuantizeMethod::Q5KM => GgmlType::Q6K,
            _ => dtype,
        };
    }

    dtype
}

fn tensor_to_f32_vec(name: &str, tensor: &Array) -> anyhow::Result<Option<Vec<f32>>> {
    let materialized = tensor.clone();
    materialized.eval();

    match materialized.dtype() {
        Dtype::Float32 => Ok(Some(materialized.as_slice::<f32>().to_vec())),
        Dtype::Float16 | Dtype::Bfloat16 => {
            let float32 = materialized.as_dtype(Dtype::Float32.as_i32());
            float32.eval();
            Ok(Some(float32.as_slice::<f32>().to_vec()))
        }
        other => {
            tracing::debug!("Skipping non-float tensor {name} with dtype {:?}", other);
            Ok(None)
        }
    }
}

/// Run model quantization.
pub(crate) async fn run_quantization(
    model_path: &str,
    output_path: &str,
    imatrix_path: Option<&str>,
    method: crate::QuantizeMethod,
    kl_calibrate: bool,
    target_bpw: Option<f32>,
    kl_threshold: f64,
) -> anyhow::Result<()> {
    use pmetal_gguf::{
        GgufBuilder,
        dynamic::{
            CalibrationMap, DynamicQuantizationConfig, DynamicQuantizer, KlCalibrationConfig,
        },
        imatrix::IMatrix,
        quantize::quantize,
    };

    println!("========================================");
    println!("  PMetal GGUF Quantization");
    println!("========================================");
    println!("Model:    {}", model_path);
    println!("Output:   {}", output_path);
    println!("Method:   {}", method.as_str());
    if let Some(imp) = imatrix_path {
        println!("IMatrix:  {}", imp);
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

    // 1. Load IMatrix if provided
    let imatrix = if let Some(path) = imatrix_path {
        tracing::info!("Loading IMatrix from {}", path);
        Some(IMatrix::load(Path::new(path))?)
    } else {
        None
    };

    // 2. Initialize quantizer
    let quantizer = if let Some(base_type) = method.to_ggml_type() {
        let config = DynamicQuantizationConfig {
            base_type,
            high_precision_type: base_type,
            fallback_type: base_type,
            ..Default::default()
        };
        DynamicQuantizer::new(config, None)
    } else {
        let config = DynamicQuantizationConfig::default();
        DynamicQuantizer::new(config, imatrix)
    };

    // 3. Load Model Weights
    tracing::info!("Scanning model weights from {:?}...", resolved_model_path);
    let weights = pmetal_models::loader::load_weights(&resolved_model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load weights: {}", e))?;
    tracing::info!("Loaded {} tensors", weights.len());

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

    // 5. KL-divergence calibration pass (optional)
    let mut float_cache: std::collections::HashMap<String, (Vec<f32>, Vec<i32>)> =
        std::collections::HashMap::new();
    let calibration_map: CalibrationMap;

    if kl_calibrate {
        println!(
            "Running KL calibration pass over {} tensors...",
            weights.len()
        );

        let mut tensor_data: Vec<(String, Vec<f32>, Vec<i32>)> = Vec::new();
        let mut sorted_names: Vec<_> = weights.keys().cloned().collect();
        sorted_names.sort();

        for name in &sorted_names {
            let tensor = weights
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Tensor {} missing after key listing", name))?;

            let data_f32 = match tensor_to_f32_vec(name, tensor)? {
                Some(data) => data,
                None => continue,
            };

            let shape_i32: Vec<i32> = tensor.shape().iter().map(|&d| d as i32).collect();
            float_cache.insert(name.clone(), (data_f32.clone(), shape_i32.clone()));
            tensor_data.push((name.clone(), data_f32, shape_i32));
        }

        let kl_config = KlCalibrationConfig {
            kl_threshold,
            target_bpw,
            ..Default::default()
        };

        calibration_map = quantizer.calibrate_all(&tensor_data, &kl_config);

        let tensor_sizes: Vec<(String, usize)> = tensor_data
            .iter()
            .map(|(n, d, _)| (n.clone(), d.len()))
            .collect();
        let summary = quantizer.summarize_calibration(&calibration_map, &tensor_sizes);
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
    } else {
        calibration_map = CalibrationMap::new();
    }

    // 6. Initialize GGUF Builder
    let mut builder = GgufBuilder::with_model(&architecture, "quantized-model");
    add_hf_config_metadata(&mut builder, &architecture, config_json.as_ref(), method);
    add_tokenizer_metadata(&mut builder, &resolved_model_path, config_json.as_ref())?;

    // 7. Quantize and Write
    tracing::info!("Starting quantization...");

    let expert_groups = collect_expert_groups(&weights);
    let expert_source_keys: HashSet<String> = expert_groups
        .values()
        .flat_map(|group| group.iter().map(|tensor| tensor.name.clone()))
        .collect();

    let mut keys: Vec<_> = weights.keys().collect();
    keys.sort();

    for (key, group) in &expert_groups {
        let (data_f32, shape_u64) = stack_expert_group(&weights, group)?;
        let gguf_name = format!("blk.{}.{}", key.layer, key.projection.gguf_suffix());
        let proposed_type = if calibration_map.is_empty() {
            quantizer.get_tensor_type(&gguf_name, &shape_u64)
        } else {
            quantizer.get_tensor_type_calibrated(&gguf_name, &shape_u64, &calibration_map)
        };
        let target_type = llama_style_tensor_type(
            method,
            &gguf_name,
            &shape_u64,
            config_json.as_ref(),
            proposed_type,
        );

        tracing::info!(
            "Quantizing stacked expert tensor {} ({} experts) to {:?}",
            gguf_name,
            group.len(),
            target_type
        );
        let quantized_data = quantize(&data_f32, target_type)
            .map_err(|e| anyhow::anyhow!("Quantization error for {}: {:?}", gguf_name, e))?;
        builder.add_raw_tensor(gguf_name, shape_u64, target_type, quantized_data);
    }

    for name in keys {
        if expert_source_keys.contains(name.as_str()) {
            continue;
        }

        let shape_u64: Vec<u64>;
        let data_f32: Vec<f32>;

        if let Some((cached_data, cached_shape)) = float_cache.remove(name) {
            shape_u64 = cached_shape.iter().map(|&d| d as u64).collect();
            data_f32 = cached_data;
        } else {
            let tensor = weights
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Tensor {} not found in loaded weights", name))?;

            let shape = tensor.shape();
            shape_u64 = shape.iter().map(|&d| d as u64).collect();

            data_f32 = match tensor_to_f32_vec(name, tensor)? {
                Some(data) => data,
                None => continue,
            };
        }

        let gguf_name = pmetal_models::weight_format::WeightLoader::hf_to_gguf_name(name);

        let target_type = if calibration_map.is_empty() {
            quantizer.get_tensor_type(&gguf_name, &shape_u64)
        } else {
            quantizer.get_tensor_type_calibrated(name, &shape_u64, &calibration_map)
        };
        let target_type = llama_style_tensor_type(
            method,
            &gguf_name,
            &shape_u64,
            config_json.as_ref(),
            target_type,
        );

        tracing::info!("Quantizing {} ({}) to {:?}", name, gguf_name, target_type);
        let quantized_data = quantize(&data_f32, target_type)
            .map_err(|e| anyhow::anyhow!("Quantization error for {}: {:?}", name, e))?;

        builder.add_raw_tensor(gguf_name, shape_u64, target_type, quantized_data);
    }

    // 8. Write GGUF output
    let validated_output = crate::validate_output_path(output_path, "quantization output")?;
    let mut file = std::fs::File::create(&validated_output)?;
    builder.write(&mut file)?;

    println!("Quantization complete!");
    Ok(())
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
    use pmetal_bridge::mlx_quant;

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
                crate::QuantizeMethod::Q4KM,
                "blk.0.ffn_down.weight",
                &[4096, 14336],
                Some(&config),
                GgmlType::Q4K,
            ),
            GgmlType::Q6K
        );
        assert_eq!(
            llama_style_tensor_type(
                crate::QuantizeMethod::Q4KM,
                "blk.0.attn_norm.weight",
                &[4096],
                Some(&config),
                GgmlType::Q4K,
            ),
            GgmlType::F16
        );
        assert_eq!(
            llama_style_tensor_type(
                crate::QuantizeMethod::Q4KM,
                "output.weight",
                &[32000, 4096],
                Some(&config),
                GgmlType::Q4K,
            ),
            GgmlType::Q6K
        );
        assert_eq!(
            llama_style_tensor_type(
                crate::QuantizeMethod::Tq1_0,
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
            quantize_method_file_type(crate::QuantizeMethod::Q4_0),
            FileType::MostlyQ4_0
        );
        assert_eq!(
            quantize_method_file_type(crate::QuantizeMethod::Q5_1),
            FileType::MostlyQ5_1
        );
        assert_eq!(
            quantize_method_file_type(crate::QuantizeMethod::Q1_0),
            FileType::MostlyQ1_0
        );
        assert_eq!(
            quantize_method_file_type(crate::QuantizeMethod::Mxfp4),
            FileType::MostlyMxfp4Moe
        );
        assert_eq!(
            quantize_method_file_type(crate::QuantizeMethod::Nvfp4),
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
            crate::QuantizeMethod::Q4KM,
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
}
