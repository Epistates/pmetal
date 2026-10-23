/**
 * The app's backend, as TypeScript.
 *
 * Job forms, their defaults and their ranges are not here: the backend reads
 * them from the job specs and sends them (`listJobForms`). What this file
 * types is the contract of the commands and events; the names of the fields a
 * job may be given come from `specs.generated.ts`, which a test keeps equal to
 * the specs.
 */

import { invoke } from '@tauri-apps/api/core';
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type { InferSpec, JobFields, JobKey } from './specs.generated';

export type { InferSpec, JobFields, JobKey } from './specs.generated';

// =============================================================================
// Errors
// =============================================================================

/** Why a job did not start; `issues` name the fields of the form that are wrong. */
export interface StartFailure {
  message: string;
  issues: FieldIssue[];
}

export interface FieldIssue {
  field: string;
  message: string;
}

/** The text of whatever a command rejected with. */
export function errorMessage(e: unknown): string {
  if (typeof e === 'string') return e;
  if (e instanceof Error) return e.message;
  if (e && typeof e === 'object' && 'message' in e && typeof e.message === 'string') {
    return e.message;
  }
  return String(e);
}

/** The field-level problems in a rejection, if it carries any. */
export function issuesOf(e: unknown): Record<string, string> {
  const out: Record<string, string> = {};
  if (e && typeof e === 'object' && 'issues' in e && Array.isArray(e.issues)) {
    for (const issue of e.issues as FieldIssue[]) {
      if (issue.field && !(issue.field in out)) out[issue.field] = issue.message;
    }
  }
  return out;
}

// =============================================================================
// System and settings
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

export type Theme = 'system' | 'light' | 'dark';

/** The settings. The Hugging Face token is not in it: only whether one is saved. */
export interface AppConfig {
  default_model: string | null;
  theme: Theme;
  custom_model_dirs: string[];
  /** The Hub cache in use; set by HF_HOME or HF_HUB_CACHE. */
  hf_cache_dir: string;
  /** Where jobs write what they produce. */
  output_dir: string;
  hf_token_set: boolean;
  notes: string[];
  pmetal_binary: string | null;
  pmetal_binary_error: string | null;
}

export const getSystemInfo = () => invoke<SystemInfo>('get_system_info');
export const getConfig = () => invoke<AppConfig>('get_config');
export const setConfig = (patch: { default_model?: string; theme?: Theme }) =>
  invoke<AppConfig>('set_config', { patch });
export const setHfToken = (token: string) => invoke<AppConfig>('set_hf_token', { token });
export const clearHfToken = () => invoke<AppConfig>('clear_hf_token');

/** Let the person choose a folder; `null` if they cancel. */
export const pickDirectory = (title?: string) =>
  invoke<string | null>('pick_directory', { title: title ?? null });
export const pickFile = (title?: string) =>
  invoke<string | null>('pick_file', { title: title ?? null });

// =============================================================================
// Models
// =============================================================================

export type ModelSource = 'hf_cache' | 'trained' | 'custom';

export interface CachedModel {
  id: string;
  path: string;
  /** What to give a job to use it: the Hub id, or a full path. */
  reference: string;
  size: number;
  size_formatted: string;
  downloaded_at: string | null;
  model_type: string;
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
  inference_fit: 'fits' | 'tight' | 'too_large';
  training_fit: 'fits' | 'tight' | 'too_large';
  weights_gb: number;
  inference_memory_gb: number;
  training_memory_gb: number;
  available_memory_gb: number;
  estimated_tps: number | null;
  recommended_batch_size: number;
  notes: string[];
  /** Why pmetal cannot run this model, when its architecture is one pmetal declines. */
  unsupported: string | null;
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
  params_b: number | null;
  size_bytes: number | null;
  fit: 'fits' | 'tight' | 'too_large' | null;
}

export interface TrainedAdapter {
  path: string;
  name: string;
  base_model: string | null;
  rank: number | null;
  alpha: number | null;
  size_bytes: number;
}

export const listModels = () => invoke<CachedModel[]>('list_models');
export const refreshModels = () => invoke<CachedModel[]>('refresh_models');
export const getModelInfo = (modelId: string) =>
  invoke<ModelInfo | null>('get_model_info', { modelId });
export const getModelDefaults = (modelId: string) =>
  invoke<ModelDefaults>('get_model_defaults', { modelId });
export const getModelFit = (modelId: string) => invoke<ModelFitInfo>('get_model_fit', { modelId });
export const deleteModel = (modelId: string) => invoke<void>('delete_model', { modelId });
export const addModelDirectory = (path: string) =>
  invoke<CachedModel[]>('add_model_directory', { path });
export const removeModelDirectory = (path: string) =>
  invoke<CachedModel[]>('remove_model_directory', { path });
export const searchHubModels = (query: string, limit?: number) =>
  invoke<HubSearchResult[]>('search_hub_models', { query, limit });
export const getTrendingModels = (limit?: number) =>
  invoke<HubSearchResult[]>('get_trending_models', { limit });
export const listTrainedAdapters = () => invoke<TrainedAdapter[]>('list_trained_adapters');

// =============================================================================
// Datasets
// =============================================================================

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

export interface DatasetPeek {
  columns: string[];
  avg_tokens_estimate: number;
  max_tokens_estimate: number;
  suggested_seq_len: number;
  rows_sampled: number;
}

export const searchHubDatasets = (query: string, limit?: number) =>
  invoke<DatasetSearchResult[]>('search_hub_datasets', { query, limit });
export const getTrendingDatasets = (limit?: number) =>
  invoke<DatasetSearchResult[]>('get_trending_datasets', { limit });
export const listCachedDatasets = () => invoke<CachedDatasetInfo[]>('list_cached_datasets');
export const peekDatasetColumns = (path: string, limit?: number) =>
  invoke<DatasetPeek>('peek_dataset_columns', { path, limit: limit ?? null });

// =============================================================================
// Downloads
// =============================================================================

export type DownloadState = 'queued' | 'downloading' | 'completed' | 'failed' | 'cancelled';

export interface DownloadView {
  id: string;
  kind: 'model' | 'dataset';
  model_id: string;
  state: DownloadState;
  files_done: number;
  total_files: number;
  bytes_done: number;
  total_bytes: number;
  bytes_per_sec: number | null;
  eta_seconds: number | null;
  current_file: string | null;
  path: string | null;
  error: string | null;
  started_at: string;
}

export const downloadModel = (modelId: string, revision?: string) =>
  invoke<DownloadView>('download_model', { modelId, revision: revision || null });
export const downloadDataset = (datasetId: string) =>
  invoke<DownloadView>('download_dataset', { datasetId });
export const listDownloads = () => invoke<DownloadView[]>('list_downloads');
export const cancelDownload = (downloadId: string) =>
  invoke<void>('cancel_download', { downloadId });
export const clearFinishedDownloads = () => invoke<void>('clear_finished_downloads');

// =============================================================================
// Jobs
// =============================================================================

export type FieldInput =
  | { type: 'text' }
  | { type: 'number'; min: number | null; max: number | null }
  | { type: 'integer'; min: number | null; max: number | null }
  | { type: 'choice'; options: string[] }
  | { type: 'toggle' }
  | { type: 'model' }
  | { type: 'dataset' }
  | { type: 'path' }
  | { type: 'readonly' };

export interface FieldInfo {
  name: string;
  label: string;
  help: string | null;
  group: string;
  input: FieldInput;
  required: boolean;
  /** Entered as comma-separated values. */
  list: boolean;
  /** May be left empty. */
  optional: boolean;
  /** The text the form starts with: the command line's own default. */
  default: string;
}

export interface FormInfo {
  key: JobKey;
  label: string;
  description: string;
  subcommand: string;
  fields: FieldInfo[];
}

export type JobState = 'running' | 'stopping' | 'completed' | 'failed' | 'stopped';

export interface JobView {
  id: string;
  kind: JobKey;
  label: string;
  title: string;
  command: string;
  state: JobState;
  error: string | null;
  exit_code: number | null;
  phase: string | null;
  step: number | null;
  total_steps: number | null;
  epoch: number | null;
  total_epochs: number | null;
  loss: number | null;
  best_loss: number | null;
  eval_loss: number | null;
  learning_rate: number | null;
  grad_norm: number | null;
  tokens_per_second: number | null;
  progress: number | null;
  eta_seconds: number | null;
  started_at: string;
  finished_at: string | null;
  elapsed_seconds: number;
  output: string | null;
  serve: { url: string; reachable: boolean } | null;
  result: {
    output_path: string | null;
    final_loss: number | null;
    total_steps: number | null;
    extra: Record<string, string>;
  } | null;
}

export interface OutputLine {
  seq: number;
  stream: 'stdout' | 'stderr';
  text: string;
}

export interface OutputPage {
  lines: OutputLine[];
  next_since: number;
  dropped_lines: number;
}

export interface StepPoint {
  step: number;
  loss: number;
  learning_rate: number;
  tokens_per_second: number;
}

export interface StepPage {
  steps: StepPoint[];
  next_since: number;
}

/** The text entered for fields of job `K`, by the spec's own field names. */
export type JobValues<K extends JobKey> = Partial<Record<JobFields[K], string>>;

export const listJobForms = () => invoke<FormInfo[]>('list_job_forms');
export const listJobs = () => invoke<JobView[]>('list_jobs');
export const stopJob = (jobId: string) => invoke<void>('stop_job', { jobId });
export const forgetJob = (jobId: string) => invoke<void>('forget_job', { jobId });
export const jobOutput = (jobId: string, tail?: number, since?: number) =>
  invoke<OutputPage>('job_output', { jobId, tail: tail ?? null, since: since ?? null });
export const jobSteps = (jobId: string, since?: number) =>
  invoke<StepPage>('job_steps', { jobId, since: since ?? null });

/**
 * Start job `kind` from the text entered in its form. Rejects with a
 * {@link StartFailure} whose `issues` name the fields that are wrong.
 */
export function startJob<K extends JobKey>(kind: K, values: JobValues<K>): Promise<JobView> {
  return invoke<JobView>('start_job', { kind, values });
}

// =============================================================================
// Chat
// =============================================================================

export interface ChatTurn {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatMetrics {
  session_id: string;
  /** 0 when the reply was stopped before the command reported its statistics. */
  generated_tokens: number;
  total_ms: number;
  ttft_ms: number | null;
  tok_per_sec: number | null;
  response_text: string;
  thinking: string | null;
  truncated_thinking: boolean;
  stopped: boolean;
}

/**
 * Start a reply. The events it produces carry `sessionId`, which the page
 * chooses so it can recognise them before this resolves.
 */
export const startChat = (spec: InferSpec, messages: ChatTurn[] | null, sessionId: string) =>
  invoke<void>('start_chat', { spec, messages, sessionId });
export const stopChat = (sessionId?: string) =>
  invoke<void>('stop_chat', { sessionId: sessionId ?? null });

// =============================================================================
// Events
// =============================================================================

const on = <T>(event: string, callback: (payload: T) => void): Promise<UnlistenFn> =>
  listen<T>(event, (e) => callback(e.payload));

export const onJobUpdate = (cb: (job: JobView) => void) => on<JobView>('job-update', cb);
export const onJobOutput = (cb: (e: { job_id: string; lines: OutputLine[] }) => void) =>
  on('job-output', cb);
export const onJobsResync = (cb: () => void) => on<null>('jobs-resync', () => cb());
export const onDownloadUpdate = (cb: (d: DownloadView) => void) =>
  on<DownloadView>('download-update', cb);
export const onModelsUpdated = (cb: () => void) => on<null>('models-updated', () => cb());
export const onAppQuitting = (cb: () => void) => on<null>('app-quitting', () => cb());
export const onChatToken = (
  cb: (e: { session_id: string; text: string; thinking: string }) => void,
) =>
  on('chat-token', cb);
export const onChatDone = (cb: (m: ChatMetrics) => void) => on<ChatMetrics>('chat-done', cb);
export const onChatError = (cb: (e: { session_id: string; error: string }) => void) =>
  on('chat-error', cb);
