/**
 * PMetal API - TypeScript wrappers for Tauri commands
 */

import { invoke } from '@tauri-apps/api/core';
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

export interface CachedModel {
  id: string;
  path: string;
  size: number;
  size_formatted: string;
  downloaded_at: string;
  model_type: string | null;
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
}

export interface TrainingConfig {
  model: string;
  dataset: string | null;
  method: string;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  lora_rank: number | null;
  lora_alpha: number | null;
  lora_dropout: number | null;
  use_rslora: boolean | null;
  use_dora: boolean | null;
  output_dir: string | null;
  load_in_4bit: boolean | null;
  gradient_accumulation_steps: number | null;
  max_seq_len: number | null;
  text_column: string | null;
  dataset_format: string | null;
  embedding_lr: number | null;
  jit_compilation: boolean | null;
  gradient_checkpointing: boolean | null;
  flash_attention: boolean | null;
  fused_optimizer: boolean | null;
  warmup_steps: number | null;
  weight_decay: number | null;
  max_grad_norm: number | null;
  save_steps: number | null;
  logging_steps: number | null;
  lr_scheduler: string | null;
  sequence_packing: boolean | null;
  resume_from: string | null;
  // DPO
  dpo_beta?: number | null;
  dpo_loss_type?: string | null;
  ref_model?: string | null;
  // SimPO
  simpo_beta?: number | null;
  simpo_gamma?: number | null;
  // ORPO
  orpo_lambda?: number | null;
  // KTO
  kto_desirable_weight?: number | null;
  kto_undesirable_weight?: number | null;
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

export interface GrpoConfig {
  model: string;
  dataset: string | null;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  group_size: number;
  beta: number;
  lora_rank: number | null;
  lora_alpha: number | null;
  max_seq_len: number | null;
  output_dir: string | null;
  use_reasoning_rewards: boolean | null;
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

export interface DistillationConfig {
  student_model: string;
  teacher_model: string;
  dataset: string | null;
  loss_type: string;
  temperature: number;
  alpha: number;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  lora_rank: number | null;
  lora_alpha: number | null;
  max_seq_len: number | null;
  output_dir: string | null;
}

export interface InferenceConfig {
  model: string;
  lora_path: string | null;
  prompt: string;
  system_message: string | null;
  temperature: number;
  top_k: number | null;
  top_p: number | null;
  max_tokens: number;
  repetition_penalty: number | null;
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

export interface MergeConfig {
  base_model: string;
  models: MergeModelEntry[];
  strategy: string;
  output: string;
}

export interface MergeModelEntry {
  model: string;
  weight: number;
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

// =============================================================================
// Training API
// =============================================================================

export async function startTraining(config: TrainingConfig): Promise<string> {
  return await invoke('start_training', { config });
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
// GRPO API
// =============================================================================

export async function startGrpo(config: GrpoConfig): Promise<string> {
  return await invoke('start_grpo', { config });
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

export async function startDistillation(config: DistillationConfig): Promise<string> {
  return await invoke('start_distillation', { config });
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

export async function startInference(config: InferenceConfig): Promise<void> {
  return await invoke('start_inference', { config });
}

export async function stopInference(): Promise<void> {
  return await invoke('stop_inference');
}

export function onInferenceToken(callback: (token: string) => void): Promise<UnlistenFn> {
  return listen<string>('inference-token', (event) => {
    callback(event.payload);
  });
}

export function onInferenceDone(callback: () => void): Promise<UnlistenFn> {
  return listen<null>('inference-done', () => {
    callback();
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

export async function mergeModels(config: MergeConfig): Promise<string> {
  return await invoke('merge_models', { config });
}

export async function getMergeStrategies(): Promise<MergeStrategy[]> {
  return await invoke('get_merge_strategies');
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
