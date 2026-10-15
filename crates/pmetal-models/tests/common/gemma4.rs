//! The synthetic Gemma 4 configs the committed fixtures were generated with.
//!
//! Changing a field here without regenerating the fixtures breaks every
//! comparison made against them.

#![allow(dead_code)]

/// The synthetic config the fixture was generated with. Changing a field
/// here without regenerating the fixture breaks the comparison.
pub fn synthetic_config_json() -> &'static str {
    r#"{
        "model_type": "gemma4_text",
        "vocab_size": 512,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "global_head_dim": 64,
        "num_global_key_value_heads": 1,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "attention_k_eq_v": true,
        "tie_word_embeddings": true,
        "sliding_window": 8,
        "final_logit_softcapping": 30.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional"
            },
            "sliding_attention": {
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000.0,
                "rope_type": "default"
            }
        },
        "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0,
        "use_double_wide_mlp": false,
        "enable_moe_block": false
    }"#
}

/// MoE synthetic config the fixture was generated with (same geometry as the
/// dense fixture with the routed-experts branch enabled).
pub fn moe_synthetic_config_json() -> &'static str {
    r#"{
        "model_type": "gemma4_text",
        "vocab_size": 512,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "global_head_dim": 64,
        "num_global_key_value_heads": 1,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "attention_k_eq_v": true,
        "tie_word_embeddings": true,
        "sliding_window": 8,
        "final_logit_softcapping": 30.0,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional"
            },
            "sliding_attention": {
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000.0,
                "rope_type": "default"
            }
        },
        "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0,
        "use_double_wide_mlp": false,
        "enable_moe_block": true,
        "num_experts": 4,
        "top_k_experts": 2,
        "moe_intermediate_size": 64
    }"#
}

/// The E2B/E4B shape: six layers, the last two sharing the KV cache of the
/// last layer of their own attention type (3 for sliding, 2 for full), with
/// per-layer input embeddings. The native engine's only fixture for either.
pub fn shared_synthetic_config_json() -> &'static str {
    r#"{
        "model_type": "gemma4_text",
        "vocab_size": 512,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "global_head_dim": 64,
        "num_global_key_value_heads": 1,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "attention_k_eq_v": true,
        "tie_word_embeddings": true,
        "sliding_window": 8,
        "final_logit_softcapping": 30.0,
        "layer_types": [
            "sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "full_attention"
        ],
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional"
            },
            "sliding_attention": {
                "partial_rotary_factor": 1.0,
                "rope_theta": 10000.0,
                "rope_type": "default"
            }
        },
        "hidden_size_per_layer_input": 16,
        "vocab_size_per_layer_input": 512,
        "num_kv_shared_layers": 2,
        "use_double_wide_mlp": false,
        "enable_moe_block": false
    }"#
}
