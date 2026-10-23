//! Tokenizer metadata of a GGUF export.

use super::{json_bool, json_str, json_u32, json_usize};
use sha2::{Digest, Sha256};
use std::path::Path;

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

pub(crate) fn add_tokenizer_metadata(
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
