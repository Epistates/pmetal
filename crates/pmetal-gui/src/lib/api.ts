/**
 * PMetal API - TypeScript wrappers for Tauri commands
 */

import { invoke, Channel } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';

// =============================================================================
// Types
// =============================================================================

export interface SystemInfo {
  version: string;
  platform: string;
  arch: string;
  is_apple_silicon: boolean;
  gpu_name: string;
  chip_tier: string | null;
  total_memory: number;
  available_memory: number;
  total_memory_formatted: string;
  available_memory_formatted: string;
  gpu_cores: number | null;
  ane_cores: number | null;
  memory_bandwidth_gbps: number | null;
  has_ane: boolean;
  has_nax: boolean;
}

export interface AppConfig {
  cache_dir: string;
  hf_token: string | null;
  default_model: string | null;
  theme: string;
}

export type ModelSource = 'hf_cache' | 'trained' | 'custom';

export interface CachedModel {
  id: string;
  path: string;
  size: number;
  size_formatted: string;
  downloaded_at: string;
  model_type: string | null;
  source: ModelSource;
}

export interface ModelInfo {
  id: string;
  path: string;
  size: number;
  size_formatted: string;
  model_type: string | null;
  hidden_size: number | null;
  num_layers: number | null;
  vocab_size: number | null;
  context_length: number | null;
}

export interface ModelFitInfo {
  inference_fit: string; // "fits" | "tight" | "too_large"
  training_fit: string;
  weights_gb: number;
  inference_memory_gb: number;
  training_memory_gb: number;
  available_memory_gb: number;
  estimated_tps: number | null;
  recommended_batch_size: number;
}

export type TrainingStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface TrainingRun {
  id: string;
  status: TrainingStatus;
  model: string;
  method: string;
  dataset: string | null;
  epoch: number;
  total_epochs: number;
  step: number;
  total_steps: number;
  loss: number | null;
  best_loss: number | null;
  learning_rate: number | null;
  grad_norm: number | null;
  tokens_per_second: number | null;
  eta_seconds: number | null;
  started_at: string;
  ended_at: string | null;
  output_dir: string | null;
  error_message: string | null;
  status_message: string | null;
  config_summary: TrainingConfigSummary | null;
}

export interface TrainingConfigSummary {
  learning_rate: number;
  batch_size: number;
  max_seq_len: number;
  lora_rank: number | null;
  lora_alpha: number | null;
  sequence_packing: boolean;
  flash_attention: boolean;
  jit_compilation: boolean;
  gradient_checkpointing: boolean;
}

export interface TrainSpec {
  model: string;
  dataset: string;
  eval_dataset?: string | null;
  output_dir?: string;
  learning_rate?: number;
  embedding_lr?: number | null;
  batch_size?: number;
  epochs?: number;
  max_seq_len?: number;
  gradient_accumulation_steps?: number;
  no_gradient_checkpointing?: boolean;
  gradient_checkpointing_layers?: number;
  max_grad_norm?: number;
  warmup_steps?: number;
  weight_decay?: number;
  lr_schedule?: string;
  seed?: number;
  loss_scale?: number;
  lora_r?: number;
  lora_alpha?: number;
  quantization?: string | null;
  text_column?: string | null;
  text_columns?: string | null;
  column_separator?: string | null;
  prompt_column?: string | null;
  response_column?: string | null;
  no_flash_attention?: boolean;
  no_sequence_packing?: boolean;
  no_jit_compilation?: boolean;
  no_metal_fused_optimizer?: boolean;
  cut_cross_entropy?: boolean;
  no_adaptive_lr?: boolean;
  ane?: boolean;
  pack_max_seq_len?: number | null;
  config_path?: string | null;
  resume?: boolean;
}

export interface GrpoRun {
  id: string;
  status: TrainingStatus;
  model: string;
  dataset: string | null;
  step: number;
  total_steps: number;
  reward_mean: number | null;
  reward_std: number | null;
  kl_div: number | null;
  loss: number | null;
  learning_rate: number | null;
  tokens_per_second: number | null;
  eta_seconds: number | null;
  started_at: string;
  ended_at: string | null;
  output_dir: string | null;
  error_message: string | null;
}

export interface GrpoSpec {
  model: string;
  dataset: string;
  output_dir?: string;
  num_generations?: number;
  beta?: number;
  learning_rate?: number;
  epochs?: number;
  lora_r?: number;
  lora_alpha?: number;
  max_seq_len?: number;
  max_completion_length?: number;
  seed?: number;
  dapo?: boolean;
  reasoning_rewards?: boolean;
  no_flash_attention?: boolean;
  vlm?: boolean;
  max_image_size?: number;
  reward_model?: string | null;
  reward_model_max_length?: number;
  reward_model_weight?: number;
  reward_model_template?: string | null;
  speculative?: boolean;
  speculative_draft_tokens?: number;
  async_rewards?: boolean;
  text_column?: string | null;
  text_columns?: string | null;
  column_separator?: string | null;
  prompt_column?: string | null;
  response_column?: string | null;
  grpo_kv_bits?: number | null;
  log_metrics?: string | null;
}

export type DistillationStatus =
  | 'pending'
  | 'loading_models'
  | 'generating_signals'
  | 'training'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface DistillationRun {
  id: string;
  status: DistillationStatus;
  student_model: string;
  teacher_model: string;
  dataset: string | null;
  loss_type: string;
  epoch: number;
  total_epochs: number;
  step: number;
  total_steps: number;
  loss: number | null;
  best_loss: number | null;
  learning_rate: number | null;
  tokens_per_second: number | null;
  eta_seconds: number | null;
  started_at: string;
  ended_at: string | null;
  output_dir: string | null;
  error_message: string | null;
}

export interface DistillSpec {
  teacher: string;
  student: string;
  dataset: string;
  output_dir?: string;
  method?: string;
  offline?: boolean;
  offline_cache?: string | null;
  offline_generate?: boolean;
  offline_compression?: string;
  offline_top_k?: number;
  loss_type?: string;
  temperature?: number;
  alpha?: number;
  rationale?: boolean;
  rationale_weight?: number;
  lora_r?: number;
  lora_alpha?: number;
  learning_rate?: number;
  batch_size?: number;
  epochs?: number;
  max_seq_len?: number;
  seed?: number;
  text_column?: string | null;
  text_columns?: string | null;
  column_separator?: string | null;
  prompt_column?: string | null;
  response_column?: string | null;
  log_metrics?: string | null;
}

export interface InferSpec {
  model: string;
  lora?: string | null;
  prompt?: string;
  max_tokens?: number;
  temperature?: number | null;
  top_k?: number | null;
  top_p?: number | null;
  min_p?: number | null;
  repetition_penalty?: number | null;
  frequency_penalty?: number | null;
  presence_penalty?: number | null;
  seed?: number | null;
  system?: string | null;
  no_thinking?: boolean;
  fp8?: boolean;
  experts_dir?: string | null;
  kv_quant?: number | null;
  kv_k_bits?: number | null;
  kv_v_bits?: number | null;
  kv_group_size?: number;
  no_kv_quant?: boolean;
  kv_turboquant?: boolean;
  kv_turboquant_preset?: 'q2_5' | 'q3_5' | null;
}

export interface InferenceMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface HubSearchResult {
  id: string;
  author: string | null;
  downloads: number;
  downloads_formatted: string;
  likes: number;
  pipeline_tag: string | null;
  is_gated: boolean;
  library_name: string | null;
  tags: string[];
}

export interface DatasetSearchResult {
  id: string;
  author: string | null;
  downloads: number;
  downloads_formatted: string;
  likes: number;
  tags: string[];
  description: string | null;
}

export interface CachedDatasetInfo {
  name: string;
  path: string;
  size_bytes: number;
  size_formatted: string;
}

export interface MergeSpec {
  model_a: string;
  model_b: string;
  output: string;
  method?: string;
  base?: string | null;
  t?: number;
  weight_a?: number;
  weight_b?: number;
  density?: number;
  dtype?: string;
}

export interface MergeStrategy {
  name: string;
  description: string;
  supports_weights: boolean;
}

export interface DashboardStats {
  models_count: number;
  total_model_size: string;
  active_training_runs: number;
  completed_training_runs: number;
  total_training_runs: number;
  active_grpo_runs: number;
  active_distillation_runs: number;
}

export interface FuseResult {
  output_dir: string;
  model_size_bytes: number;
}

export interface ModelDefaults {
  temperature: number | null;
  top_k: number | null;
  top_p: number | null;
  max_new_tokens: number | null;
  repetition_penalty: number | null;
  max_position_embeddings: number | null;
  hidden_size: number | null;
  num_hidden_layers: number | null;
  vocab_size: number | null;
}

export interface TrainedAdapter {
  path: string;
  name: string;
  base_model: string | null;
  rank: number | null;
  alpha: number | null;
  size_bytes: number;
}

export interface DeviceInfo {
  gpu_name: string;
  arch: string;
  platform: string;
  total_memory: number;
  available_memory: number;
  total_memory_formatted: string;
  available_memory_formatted: string;
  gpu_cores: number | null;
  ane_cores: number | null;
  memory_bandwidth_gbps: number | null;
  is_apple_silicon: boolean;
  has_ane: boolean;
  has_nax: boolean;
  chip_tier: string | null;
}

// =============================================================================
// System API
// =============================================================================

export async function getConfig(): Promise<AppConfig> {
  return await invoke('get_config');
}

export async function setConfig(config: AppConfig): Promise<void> {
  return await invoke('set_config', { config });
}

export async function getSystemInfo(): Promise<SystemInfo> {
  return await invoke('get_system_info');
}

export async function getDeviceInfo(): Promise<DeviceInfo> {
  return await invoke('get_device_info');
}

// =============================================================================
// Model API
// =============================================================================

export async function listModels(): Promise<CachedModel[]> {
  return await invoke('list_models');
}

export async function getModelInfo(modelId: string): Promise<ModelInfo> {
  return await invoke('get_model_info', { modelId });
}

export async function downloadModel(
  modelId: string,
  revision?: string,
  onProgress?: (progress: number) => void
): Promise<string> {
  let unlistenStart: UnlistenFn | undefined;
  let unlistenComplete: UnlistenFn | undefined;

  try {
    if (onProgress) {
      unlistenStart = await listen<string>('download-started', () => {
        onProgress(0);
      });
      unlistenComplete = await listen<string>('download-completed', () => {
        onProgress(100);
      });
    }

    return await invoke('download_model', { modelId, revision });
  } finally {
    unlistenStart?.();
    unlistenComplete?.();
  }
}

export async function deleteModel(modelId: string): Promise<void> {
  return await invoke('delete_model', { modelId });
}

export async function addModelDirectory(path: string): Promise<CachedModel[]> {
  return await invoke('add_model_directory', { path });
}

export async function removeModelDirectory(path: string): Promise<CachedModel[]> {
  return await invoke('remove_model_directory', { path });
}

export async function listModelDirectories(): Promise<string[]> {
  return await invoke('list_model_directories');
}

export async function searchHubModels(query: string, limit?: number): Promise<HubSearchResult[]> {
  return await invoke('search_hub_models', { query, limit });
}

export async function getTrendingModels(limit?: number): Promise<HubSearchResult[]> {
  return await invoke('get_trending_models', { limit });
}

export async function getModelFit(modelId: string): Promise<ModelFitInfo> {
  return await invoke('get_model_fit', { modelId });
}

// =============================================================================
// Dataset API
// =============================================================================

export async function searchHubDatasets(query: string, limit?: number): Promise<DatasetSearchResult[]> {
  return await invoke('search_hub_datasets', { query, limit });
}

export async function getTrendingDatasets(limit?: number): Promise<DatasetSearchResult[]> {
  return await invoke('get_trending_datasets', { limit });
}

export async function listCachedDatasets(): Promise<CachedDatasetInfo[]> {
  return await invoke('list_cached_datasets');
}

export async function downloadDataset(datasetId: string): Promise<string> {
  return await invoke('download_dataset', { datasetId });
}

export interface DatasetPeek {
  columns: string[];
  avg_tokens_estimate: number;
  max_tokens_estimate: number;
  suggested_seq_len: number;
  rows_sampled: number;
}

export async function peekDatasetColumns(path: string, limit?: number): Promise<DatasetPeek> {
  return await invoke('peek_dataset_columns', { path, limit: limit ?? null });
}

// =============================================================================
// Training API
// =============================================================================

export async function startTraining(
  spec: TrainSpec,
  onMetrics?: (data: Record<string, unknown>) => void,
): Promise<string> {
  const channel = new Channel<Record<string, unknown>>();
  if (onMetrics) {
    channel.onmessage = onMetrics;
  }
  return await invoke('start_training', { spec, onMetrics: channel });
}

export async function getTrainingStatus(runId: string): Promise<TrainingRun> {
  return await invoke('get_training_status', { runId });
}

export async function listTrainingRuns(): Promise<TrainingRun[]> {
  return await invoke('list_training_runs');
}

export async function stopTraining(runId: string): Promise<void> {
  return await invoke('stop_training', { runId });
}

export function onTrainingStarted(callback: (run: TrainingRun) => void): Promise<UnlistenFn> {
  return listen<TrainingRun>('training-started', (event) => {
    callback(event.payload);
  });
}

export function onTrainingStopped(callback: (runId: string) => void): Promise<UnlistenFn> {
  return listen<string>('training-stopped', (event) => {
    callback(event.payload);
  });
}

export function onTrainingUpdate(callback: (run: TrainingRun) => void): Promise<UnlistenFn> {
  return listen<TrainingRun>('training-update', (event) => {
    callback(event.payload);
  });
}

// =============================================================================
// Serve API
// =============================================================================

export type ServeStatus = 'starting' | 'running' | 'stopped' | 'failed';

export interface ServeInstance {
  id: string;
  status: ServeStatus;
  model: string;
  host: string;
  port: number;
  bind_url: string;
  max_seq_len: number;
  fp8: boolean;
  kv_cache: string;
  started_at: string;
  ready_at: string | null;
  stopped_at: string | null;
  error_message: string | null;
  status_message: string | null;
  log_tail: string[];
}

export interface ServeSpec {
  model: string;
  host?: string;
  port?: number;
  max_seq_len?: number;
  experts_dir?: string | null;
  lora?: string | null;
  fp8?: boolean;
  kv_quant?: number | null;
  no_kv_quant?: boolean;
  kv_group_size?: number;
  kv_turboquant?: boolean;
  kv_turboquant_preset?: string | null;
  ane?: boolean;
  ane_max_seq_len?: number;
  ane_real_time?: boolean;
  continuous_batch?: boolean;
  cb_max_slots?: number;
  cb_max_queue_depth?: number;
}

export async function startServe(spec: ServeSpec): Promise<string> {
  return await invoke('start_serve', { spec });
}

export async function stopServe(instanceId: string): Promise<void> {
  return await invoke('stop_serve', { instanceId });
}

export async function listServeInstances(): Promise<ServeInstance[]> {
  return await invoke('list_serve_instances');
}

export function onServeStarted(
  callback: (instance: ServeInstance) => void,
): Promise<UnlistenFn> {
  return listen<ServeInstance>('serve-started', (event) => {
    callback(event.payload);
  });
}

export function onServeUpdate(
  callback: (instance: ServeInstance) => void,
): Promise<UnlistenFn> {
  return listen<ServeInstance>('serve-update', (event) => {
    callback(event.payload);
  });
}

export function onServeStopped(callback: (instanceId: string) => void): Promise<UnlistenFn> {
  return listen<string>('serve-stopped', (event) => {
    callback(event.payload);
  });
}

// =============================================================================
// Bench API
// =============================================================================

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface BenchTrial {
  index: number;
  prompt_tps: number;
  generation_tps: number;
  peak_memory_gb: number;
}

export interface BenchRun {
  id: string;
  status: JobStatus;
  mode: string;
  model: string;
  preset: string | null;
  started_at: string;
  ended_at: string | null;
  trials: BenchTrial[];
  error_message: string | null;
  log_tail: string[];
}

export interface BenchSpec {
  model: string;
  batch_size?: number;
  seq_len?: number;
}

export async function startBench(
  spec: BenchSpec,
  onEvent?: (e: Record<string, unknown>) => void,
): Promise<string> {
  const channel = new Channel<Record<string, unknown>>();
  if (onEvent) channel.onmessage = onEvent;
  return await invoke('start_bench', { spec, onEvent: channel });
}

export async function stopBench(runId: string): Promise<void> {
  return await invoke('stop_bench', { runId });
}

export async function listBenchRuns(): Promise<BenchRun[]> {
  return await invoke('list_bench_runs');
}

export function onBenchStarted(callback: (run: BenchRun) => void): Promise<UnlistenFn> {
  return listen<BenchRun>('bench-started', (event) => {
    callback(event.payload);
  });
}

export function onBenchUpdate(callback: (run: BenchRun) => void): Promise<UnlistenFn> {
  return listen<BenchRun>('bench-update', (event) => {
    callback(event.payload);
  });
}

export function onBenchStopped(callback: (runId: string) => void): Promise<UnlistenFn> {
  return listen<string>('bench-stopped', (event) => {
    callback(event.payload);
  });
}

// =============================================================================
// Eval API
// =============================================================================

export interface EvalMetrics {
  samples_done: number;
  samples_total: number;
  perplexity: number | null;
  accuracy: number | null;
  loss: number | null;
}

export interface EvalRun {
  id: string;
  status: JobStatus;
  model: string;
  dataset: string;
  started_at: string;
  ended_at: string | null;
  metrics: EvalMetrics;
  error_message: string | null;
  log_tail: string[];
}

export interface EvalSpec {
  model: string;
  dataset: string;
  lora?: string | null;
  max_seq_len?: number;
  num_samples?: number;
  json?: boolean;
}

export async function startEval(
  spec: EvalSpec,
  onEvent?: (e: Record<string, unknown>) => void,
): Promise<string> {
  const channel = new Channel<Record<string, unknown>>();
  if (onEvent) channel.onmessage = onEvent;
  return await invoke('start_eval', { spec, onEvent: channel });
}

export async function stopEval(runId: string): Promise<void> {
  return await invoke('stop_eval', { runId });
}

export async function listEvalRuns(): Promise<EvalRun[]> {
  return await invoke('list_eval_runs');
}

export function onEvalStarted(callback: (run: EvalRun) => void): Promise<UnlistenFn> {
  return listen<EvalRun>('eval-started', (event) => {
    callback(event.payload);
  });
}

export function onEvalUpdate(callback: (run: EvalRun) => void): Promise<UnlistenFn> {
  return listen<EvalRun>('eval-update', (event) => {
    callback(event.payload);
  });
}

export function onEvalStopped(callback: (runId: string) => void): Promise<UnlistenFn> {
  return listen<string>('eval-stopped', (event) => {
    callback(event.payload);
  });
}

// =============================================================================
// GRPO API
// =============================================================================

export async function startGrpo(
  spec: GrpoSpec,
  onEvent?: (e: Record<string, unknown>) => void,
): Promise<string> {
  const channel = new Channel<Record<string, unknown>>();
  if (onEvent) channel.onmessage = onEvent;
  return await invoke('start_grpo', { spec, onEvent: channel });
}

export async function getGrpoStatus(runId: string): Promise<GrpoRun> {
  return await invoke('get_grpo_status', { runId });
}

export async function listGrpoRuns(): Promise<GrpoRun[]> {
  return await invoke('list_grpo_runs');
}

export async function stopGrpo(runId: string): Promise<void> {
  return await invoke('stop_grpo', { runId });
}

export function onGrpoStarted(callback: (run: GrpoRun) => void): Promise<UnlistenFn> {
  return listen<GrpoRun>('grpo-started', (event) => {
    callback(event.payload);
  });
}

export function onGrpoStopped(callback: (runId: string) => void): Promise<UnlistenFn> {
  return listen<string>('grpo-stopped', (event) => {
    callback(event.payload);
  });
}

export function onGrpoUpdate(callback: (run: GrpoRun) => void): Promise<UnlistenFn> {
  return listen<GrpoRun>('grpo-update', (event) => {
    callback(event.payload);
  });
}

// =============================================================================
// Distillation API
// =============================================================================

export async function startDistillation(
  spec: DistillSpec,
  onEvent?: (e: Record<string, unknown>) => void,
): Promise<string> {
  const channel = new Channel<Record<string, unknown>>();
  if (onEvent) channel.onmessage = onEvent;
  return await invoke('start_distillation', { spec, onEvent: channel });
}

export async function getDistillationStatus(runId: string): Promise<DistillationRun> {
  return await invoke('get_distillation_status', { runId });
}

export async function listDistillationRuns(): Promise<DistillationRun[]> {
  return await invoke('list_distillation_runs');
}

export async function stopDistillation(runId: string): Promise<void> {
  return await invoke('stop_distillation', { runId });
}

export function onDistillationStarted(callback: (run: DistillationRun) => void): Promise<UnlistenFn> {
  return listen<DistillationRun>('distillation-started', (event) => {
    callback(event.payload);
  });
}

export function onDistillationStopped(callback: (runId: string) => void): Promise<UnlistenFn> {
  return listen<string>('distillation-stopped', (event) => {
    callback(event.payload);
  });
}

export function onDistillationUpdate(callback: (run: DistillationRun) => void): Promise<UnlistenFn> {
  return listen<DistillationRun>('distillation-update', (event) => {
    callback(event.payload);
  });
}

// =============================================================================
// Inference API
// =============================================================================

export async function startInference(spec: InferSpec, messages?: InferenceMessage[] | null): Promise<void> {
  return await invoke('start_inference', { spec, messages: messages ?? null });
}

export async function stopInference(): Promise<void> {
  return await invoke('stop_inference');
}

export function onInferenceToken(callback: (token: string) => void): Promise<UnlistenFn> {
  return listen<string>('inference-token', (event) => {
    callback(event.payload);
  });
}

export interface InferenceMetrics {
  prompt_tokens: number;
  generated_tokens: number;
  total_ms: number;
  ttft_ms: number | null;
  tok_per_sec: number | null;
  response_text?: string | null;
  thinking?: string | null;
  truncated_thinking?: boolean | null;
}

export function onInferenceDone(callback: (metrics: InferenceMetrics | null) => void): Promise<UnlistenFn> {
  return listen<InferenceMetrics | null>('inference-done', (event) => {
    callback(event.payload);
  });
}

export function onInferenceError(callback: (message: string) => void): Promise<UnlistenFn> {
  return listen<{ session_id?: string; error?: string }>('inference-error', (event) => {
    callback(event.payload?.error ?? 'Inference failed');
  });
}

// =============================================================================
// Merge API
// =============================================================================

export async function mergeModels(spec: MergeSpec): Promise<string> {
  return await invoke('merge_models', { spec });
}

export async function getMergeStrategies(): Promise<MergeStrategy[]> {
  return await invoke('get_merge_strategies');
}

// =============================================================================
// Model Defaults API
// =============================================================================

export async function getModelDefaults(modelId: string): Promise<ModelDefaults> {
  return await invoke('get_model_defaults', { modelId });
}

// =============================================================================
// Adapters API
// =============================================================================

export async function listTrainedAdapters(): Promise<TrainedAdapter[]> {
  return await invoke('list_trained_adapters');
}

// =============================================================================
// Fuse / Quantize API
// =============================================================================

export async function fuseLora(
  baseModel: string,
  loraPath: string,
  outputDir: string
): Promise<FuseResult> {
  return await invoke('fuse_lora', { baseModel, loraPath, outputDir });
}

export async function quantizeModel(
  modelId: string,
  quantType: string,
  outputDir: string
): Promise<string> {
  return await invoke('quantize_model', { modelId, quantType, outputDir });
}

// =============================================================================
// Dashboard API
// =============================================================================

export async function getDashboardStats(): Promise<DashboardStats> {
  return await invoke('get_dashboard_stats');
}

// =============================================================================
// Pretrain API
// =============================================================================

export interface PretrainMetrics {
  step: number;
  total_steps: number;
  loss: number | null;
  best_loss: number | null;
  tokens_per_second: number | null;
  learning_rate: number | null;
  eta_seconds: number | null;
}

export interface PretrainRun {
  id: string;
  status: JobStatus;
  arch: string;
  output_dir: string;
  started_at: string;
  ended_at: string | null;
  metrics: PretrainMetrics;
  error_message: string | null;
  log_tail: string[];
}

export interface PretrainSpec {
  arch: string;
  shards?: string | null;
  seq_len?: number;
  batch_size?: number;
  steps?: number;
  learning_rate?: number;
  min_lr?: number;
  warmup_steps?: number;
  lr_schedule?: string;
  weight_decay?: number;
  max_grad_norm?: number;
  eos_token_id?: number;
  output_dir?: string;
  checkpoint_every?: number;
  resume?: string | null;
  model_config?: string | null;
  z_loss?: number;
  gradient_accumulation_steps?: number;
  log_every?: number;
  eval_every?: number;
  eval_batches?: number;
  seed?: number;
}

export async function startPretrain(
  spec: PretrainSpec,
  onEvent?: (e: Record<string, unknown>) => void,
): Promise<string> {
  const channel = new Channel<Record<string, unknown>>();
  if (onEvent) channel.onmessage = onEvent;
  return await invoke('start_pretrain', { spec, onEvent: channel });
}

export async function listPretrainRuns(): Promise<PretrainRun[]> {
  return await invoke('list_pretrain_runs');
}

export async function stopPretrain(runId: string): Promise<void> {
  return await invoke('stop_pretrain', { runId });
}

export function onPretrainStarted(callback: (run: PretrainRun) => void): Promise<UnlistenFn> {
  return listen<PretrainRun>('pretrain-started', (event) => {
    callback(event.payload);
  });
}

export function onPretrainUpdate(callback: (run: PretrainRun) => void): Promise<UnlistenFn> {
  return listen<PretrainRun>('pretrain-update', (event) => {
    callback(event.payload);
  });
}

export function onPretrainStopped(callback: (runId: string) => void): Promise<UnlistenFn> {
  return listen<string>('pretrain-stopped', (event) => {
    callback(event.payload);
  });
}

// =============================================================================
// Embed-Train API
// =============================================================================

export interface EmbedTrainConfig {
  model: string;
  dataset: string;
  output_dir?: string | null;
  loss?: string | null;
  pooling?: string | null;
  temperature?: number | null;
  margin?: number | null;
  learning_rate?: number | null;
  batch_size?: number | null;
  epochs?: number | null;
  max_seq_len?: number | null;
  weight_decay?: number | null;
  no_normalize?: boolean | null;
  log_every?: number | null;
  seed?: number | null;
}

export async function startEmbedTrain(
  config: EmbedTrainConfig,
  onEvent?: (e: Record<string, unknown>) => void,
): Promise<string> {
  const channel = new Channel<Record<string, unknown>>();
  if (onEvent) channel.onmessage = onEvent;
  return await invoke('start_embed_train', { config, onEvent: channel });
}

// =============================================================================
// RLKD API
// =============================================================================

export interface RlkdConfig {
  model: string;
  teacher_model: string;
  dataset: string;
  output_dir?: string | null;
  distill_alpha?: number | null;
  final_alpha?: number | null;
  anneal_alpha?: boolean | null;
  distill_temperature?: number | null;
  num_generations?: number | null;
  beta?: number | null;
  learning_rate?: number | null;
  epochs?: number | null;
  lora_r?: number | null;
  lora_alpha?: number | null;
  max_seq_len?: number | null;
  max_completion_length?: number | null;
  seed?: number | null;
  reasoning_rewards?: boolean | null;
  no_flash_attention?: boolean | null;
  text_column?: string | null;
  prompt_column?: string | null;
  response_column?: string | null;
}

export async function startRlkd(
  config: RlkdConfig,
  onEvent?: (e: Record<string, unknown>) => void,
): Promise<string> {
  const channel = new Channel<Record<string, unknown>>();
  if (onEvent) channel.onmessage = onEvent;
  return await invoke('start_rlkd', { config, onEvent: channel });
}

// =============================================================================
// Ollama API
// =============================================================================

export type OllamaAction = 'install' | 'run' | 'list' | 'pull';

export async function startOllama(
  action: OllamaAction,
  model: string,
  onEvent?: (e: Record<string, unknown>) => void,
): Promise<string> {
  const channel = new Channel<Record<string, unknown>>();
  if (onEvent) channel.onmessage = onEvent;
  return await invoke('start_ollama', { action, model, onEvent: channel });
}
