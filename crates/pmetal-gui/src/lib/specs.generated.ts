// Generated from the job specs in pmetal-core by the `spec_types` test in
// crates/pmetal-gui/src-tauri/src/jobs/typegen.rs. Do not edit by hand:
// change the spec, then run
//   UPDATE_SPEC_TYPES=1 cargo test --manifest-path crates/pmetal-gui/src-tauri/Cargo.toml spec_types

/** The fields each job's form may set, by the spec's own names. */
export interface JobFields {
  train: 'model' | 'dataset' | 'eval_dataset' | 'output_dir' | 'learning_rate' | 'embedding_lr' | 'label_smoothing' | 'z_loss' | 'neftune_alpha' | 'batch_size' | 'epochs' | 'max_seq_len' | 'gradient_accumulation_steps' | 'no_gradient_checkpointing' | 'max_grad_norm' | 'warmup_steps' | 'warmup_ratio' | 'min_lr' | 'wsd_stable_ratio' | 'wsd_decay_style' | 'cosine_num_restarts' | 'polynomial_power' | 'max_steps' | 'logging_steps' | 'eval_steps' | 'save_steps' | 'weight_decay' | 'optimizer' | 'lr_schedule' | 'seed' | 'loss_scale' | 'lora_r' | 'lora_alpha' | 'dora' | 'rslora' | 'lora_dropout' | 'lora_targets' | 'quantization' | 'quant_block_size' | 'double_quant' | 'text_column' | 'text_columns' | 'column_separator' | 'prompt_column' | 'response_column' | 'no_sequence_packing' | 'cut_cross_entropy' | 'no_adaptive_lr' | 'ane' | 'full_finetune' | 'freeze_embeddings' | 'no_memory_check' | 'pack_max_seq_len' | 'distributed_peers' | 'distributed_auto' | 'compression_strategy' | 'resume';
  distill: 'teacher' | 'student' | 'dataset' | 'eval_dataset' | 'output_dir' | 'method' | 'offline' | 'offline_cache' | 'offline_generate' | 'offline_compression' | 'offline_top_k' | 'loss_type' | 'skew_alpha' | 'universal_logit_top_k' | 'temperature' | 'alpha' | 'rationale' | 'rationale_weight' | 'on_policy' | 'on_policy_max_tokens' | 'lora_r' | 'lora_alpha' | 'learning_rate' | 'optimizer' | 'batch_size' | 'epochs' | 'max_seq_len' | 'seed' | 'checkpoint_every' | 'text_column' | 'text_columns' | 'column_separator' | 'prompt_column' | 'response_column' | 'log_metrics' | 'resume';
  grpo: 'model' | 'dataset' | 'output_dir' | 'num_generations' | 'beta' | 'learning_rate' | 'optimizer' | 'epochs' | 'lora_r' | 'lora_alpha' | 'max_seq_len' | 'max_completion_length' | 'seed' | 'checkpoint_every' | 'dapo' | 'loss_type' | 'num_iterations' | 'reasoning_rewards' | 'vlm' | 'max_image_size' | 'reward_model' | 'reward_model_max_length' | 'reward_model_weight' | 'reward_model_template' | 'reward_cmd' | 'reward_weight' | 'reward_timeout' | 'reward_regex' | 'reward_regex_weight' | 'reward_json' | 'reward_json_schema' | 'reward_json_weight' | 'async_rewards' | 'prompt_column' | 'response_column' | 'grpo_kv_bits' | 'log_metrics' | 'resume';
  rlkd: 'model' | 'teacher_model' | 'dataset' | 'output_dir' | 'distill_alpha' | 'final_alpha' | 'anneal_alpha' | 'distill_temperature' | 'num_generations' | 'beta' | 'learning_rate' | 'epochs' | 'lora_r' | 'lora_alpha' | 'max_seq_len' | 'max_completion_length' | 'seed' | 'reasoning_rewards' | 'prompt_column' | 'response_column' | 'log_metrics';
  preference: 'model' | 'dataset' | 'output_dir' | 'loss' | 'beta' | 'simpo_gamma_ratio' | 'label_smoothing' | 'desirable_weight' | 'undesirable_weight' | 'learning_rate' | 'batch_size' | 'gradient_accumulation_steps' | 'epochs' | 'max_steps' | 'warmup_ratio' | 'max_grad_norm' | 'weight_decay' | 'optimizer' | 'lora_r' | 'lora_alpha' | 'max_prompt_length' | 'max_length' | 'seed' | 'log_metrics';
  embed_train: 'model' | 'dataset' | 'output_dir' | 'loss' | 'pooling' | 'temperature' | 'margin' | 'learning_rate' | 'batch_size' | 'epochs' | 'max_seq_len' | 'weight_decay' | 'no_normalize' | 'log_every' | 'seed' | 'lora' | 'lora_r' | 'lora_alpha';
  pretrain: 'arch' | 'shards' | 'eval_shards' | 'seq_len' | 'batch_size' | 'steps' | 'learning_rate' | 'min_lr' | 'warmup_steps' | 'lr_schedule' | 'weight_decay' | 'max_grad_norm' | 'eos_token_id' | 'output_dir' | 'checkpoint_every' | 'resume' | 'model_config' | 'z_loss' | 'document_mask' | 'gradient_accumulation_steps' | 'log_every' | 'eval_every' | 'eval_batches' | 'seed';
  train_mtp: 'model' | 'family' | 'init_mtp' | 'mtp_layers' | 'assistant_layers' | 'num_assistant_tokens' | 'shards' | 'eos_token_id' | 'output' | 'seq_len' | 'batch_size' | 'steps' | 'learning_rate' | 'min_lr' | 'warmup_steps' | 'lr_schedule' | 'weight_decay' | 'max_grad_norm' | 'checkpoint_every' | 'log_every' | 'resume' | 'seed';
  train_draft: 'target' | 'draft' | 'draft_config' | 'shards' | 'eos_token_id' | 'output' | 'seq_len' | 'batch_size' | 'steps' | 'learning_rate' | 'min_lr' | 'warmup_steps' | 'lr_schedule' | 'weight_decay' | 'max_grad_norm' | 'checkpoint_every' | 'log_every' | 'resume' | 'seed';
  train_diffusion: 'model' | 'dataset' | 'output' | 'lora_r' | 'lora_alpha' | 'lora_targets' | 'corrupted_only' | 'quantization' | 'quant_block_size' | 'double_quant' | 'steps' | 'learning_rate' | 'weight_decay' | 'max_grad_norm' | 'max_context_len' | 'checkpoint_every' | 'log_every' | 'seed';
  dflash: 'target' | 'draft' | 'prompt' | 'max_new_tokens' | 'temperature' | 'speculative_tokens' | 'draft_fp8' | 'json' | 'no_chat' | 'tree_budget' | 'compare_greedy';
  serve: 'model' | 'host' | 'port' | 'cors_origin' | 'request_timeout' | 'max_seq_len' | 'experts_dir' | 'fp8' | 'kv_quant' | 'no_kv_quant' | 'kv_group_size' | 'kv_turboquant' | 'kv_turboquant_preset' | 'ane' | 'ane_max_seq_len' | 'draft_model' | 'continuous_batch' | 'cb_max_slots' | 'cb_max_queue_depth' | 'cb_block_size' | 'cb_max_blocks' | 'cb_exact';
  bench: 'model' | 'batch_size' | 'seq_len';
  bench_workload: 'preset' | 'model' | 'dataset' | 'experts_dir' | 'inference_context' | 'prompt_samples' | 'max_prompt_tokens' | 'decode_steps' | 'inference_warmup_passes' | 'inference_session_repeats' | 'inference_repeats' | 'train_samples' | 'train_steps' | 'batch_size' | 'max_seq_len' | 'json' | 'output';
  eval: 'model' | 'dataset' | 'task' | 'fewshot' | 'lora' | 'max_seq_len' | 'num_samples' | 'seed' | 'batch_size' | 'max_gen_tokens' | 'output' | 'log_samples' | 'json';
  merge: 'model_a' | 'model_b' | 'output' | 'method' | 'base' | 't' | 'weight_a' | 'weight_b' | 'density' | 'dtype';
  fuse: 'model' | 'lora' | 'output' | 'alpha' | 'rank' | 'low_memory';
  quantize: 'model' | 'output' | 'imatrix' | 'method' | 'lora' | 'kl_calibrate' | 'target_bpw' | 'kl_threshold' | 'format' | 'bits' | 'group_size';
  pack_experts: 'model' | 'output' | 'bits';
  tokenize: 'input' | 'output' | 'tokenizer' | 'text_column' | 'docs_per_shard';
  vocode: 'model' | 'input' | 'output' | 'chunk_frames';
  decide: 'model' | 'request' | 'max_length' | 'compact';
  ollama_modelfile: 'base' | 'lora' | 'output' | 'template' | 'system' | 'temperature' | 'num_ctx' | 'top_k' | 'top_p' | 'license';
}

export type JobKey = keyof JobFields;

/** `pmetal infer`'s spec, as the chat sends it. Every field may be left out. */
export interface InferSpec {
  ane?: boolean;
  ane_max_seq_len?: number;
  backend?: 'auto' | 'standard' | 'ane';
  benchmark?: boolean;
  benchmark_iters?: number;
  benchmark_prompt_tokens?: number | null;
  chat?: boolean;
  chat_template_kwargs?: string | null;
  detect_repetition?: boolean;
  draft_model?: string | null;
  experts_dir?: string | null;
  fp8?: boolean;
  frequency_penalty?: number | null;
  hide_thinking?: boolean;
  image?: string[];
  json?: boolean;
  kv_group_size?: number;
  kv_k_bits?: number | null;
  kv_qjl?: boolean;
  kv_quant?: number | null;
  kv_quant_preset?: 'q2_5' | 'q3_5' | null;
  kv_turboquant?: boolean;
  kv_turboquant_preset?: 'q2_5' | 'q3_5' | null;
  kv_v_bits?: number | null;
  lora?: string | null;
  max_tokens?: number | null;
  min_p?: number | null;
  mode?: 'auto' | 'thinking-general' | 'thinking-coding' | 'instruct-general' | 'instruct-reasoning';
  model?: string;
  mtp?: boolean;
  mtp_draft_tokens?: number;
  mtp_model?: string | null;
  no_kv_quant?: boolean;
  no_preserve_thinking?: boolean;
  no_thinking?: boolean;
  presence_penalty?: number | null;
  profile_layers?: boolean;
  profile_output?: string | null;
  prompt?: string;
  reasoning_effort?: string | null;
  repetition_penalty?: number | null;
  seed?: number | null;
  system?: string | null;
  temperature?: number | null;
  tools?: string | null;
  top_k?: number | null;
  top_p?: number | null;
  video?: string[];
  video_fps?: number | null;
}
