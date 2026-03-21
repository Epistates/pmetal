<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { modelsStore, trainingStore } from '$lib/stores.svelte';
  import type { TrainingConfig, TrainingRun, CachedDatasetInfo, TrainedAdapter } from '$lib/api';
  import { fuseLora, listCachedDatasets, listTrainedAdapters, peekDatasetColumns, getModelDefaults } from '$lib/api';
  import { formatEta, runProgress, getStatusBadgeClass } from '$lib/utils';

  let cachedDatasets = $state<CachedDatasetInfo[]>([]);
  let datasetColumns = $state<string[]>([]);
  let columnsLoading = $state(false);
  let lastPeekedPath = $state('');
  let datasetAvgTokens = $state(0);
  let datasetMaxTokens = $state(0);
  let datasetSuggestedSeqLen = $state(0);
  let datasetRowsSampled = $state(0);
  let datasetFullScan = $state(false);
  let scanningAll = $state(false);

  // Training methods
  const trainingMethods = [
    { value: 'sft', label: 'SFT', description: 'Supervised Fine-Tuning on instruction/chat data' },
    { value: 'lora', label: 'LoRA', description: 'Low-Rank Adaptation — parameter-efficient SFT' },
    { value: 'qlora', label: 'QLoRA', description: 'Quantized LoRA for reduced memory usage (4-bit)' },
  ];

  const lrSchedulers = ['cosine', 'linear', 'constant', 'cosine_with_restarts', 'polynomial'];
  const datasetFormats = ['auto', 'alpaca', 'sharegpt', 'hf', 'jsonl'];

  // Form state
  let selectedModel = $state('');
  let selectedMethod = $state('lora');
  let datasetPath = $state('');
  let datasetFormat = $state('auto');
  let textColumn = $state('text');
  let selectedTextColumns = $state<string[]>([]);
  let columnToAdd = $state('');
  let epochs = $state(3);
  let learningRate = $state(0.0002);
  let batchSize = $state(1);
  let gradAccumSteps = $state(1);
  let loraRank = $state(16);
  let loraAlpha = $state(32);
  let loraDropout = $state(0.0);
  let useRslora = $state(false);
  let useDora = $state(false);
  let maxSeqLen = $state(2048);
  let warmupSteps = $state(10);
  let weightDecay = $state(0.01);
  let maxGradNorm = $state(1.0);
  let lrScheduler = $state('cosine');
  let saveSteps = $state(500);
  let loggingSteps = $state(10);
  let outputDir = $state('');
  let resumeFrom = $state('');
  let loadIn4bit = $state(true);

  // PMetal optimizations
  let jitCompilation = $state(false);
  let gradientCheckpointing = $state(true);
  let sequencePacking = $state(true);
  let flashAttention = $state(true);
  let fusedOptimizer = $state(false);
  let embeddingLr = $state(0.0);

  // UI state
  let showAdvanced = $state(false);
  let showPmetalOpts = $state(true);
  let isSubmitting = $state(false);
  let formError = $state<string | null>(null);
  let formSuccess = $state<string | null>(null);
  let selectedRunId = $state<string | null>(null);
  let showFuseModal = $state(false);
  let fuseBaseModel = $state('');
  let fuseLoraPath = $state('');
  let fuseOutputDir = $state('');
  let isFusing = $state(false);
  let fuseError = $state<string | null>(null);
  let fuseSuccess = $state<string | null>(null);
  let trainedAdapters = $state<TrainedAdapter[]>([]);
  let fuseCustomPath = $state(false);

  // Derived state
  let models = $derived(modelsStore.models);
  let runs = $derived(trainingStore.runs);
  let activeRuns = $derived(trainingStore.activeRuns);
  let selectedRun = $derived(runs.find(r => r.id === selectedRunId) ?? null);
  // PMetal always uses LoRA; show config for all methods except bare SFT
  let isLoraMethod = $derived(selectedMethod !== 'sft');

  // Auto-fill max seq len from model's max_position_embeddings
  let trainingDefaultsModel = $state('');
  $effect(() => {
    const model = selectedModel;
    if (!model || model === trainingDefaultsModel) return;
    trainingDefaultsModel = model;
    getModelDefaults(model).then((d) => {
      if (d.max_position_embeddings != null) {
        // Cap at model's context length, but don't exceed 4096 for training memory
        maxSeqLen = Math.min(d.max_position_embeddings, 4096);
      }
    }).catch(() => {});
  });

  // ── Live training metrics history (for charts) ──
  interface MetricPoint { step: number; loss: number; lr: number; tokSec: number; }
  let liveHistory = $state<MetricPoint[]>([]);
  let liveRunId = $state<string | null>(null);

  // Track the active run's metrics over time
  $effect(() => {
    if (activeRuns.length === 0) return;
    const run = activeRuns[0];
    // Reset history when a new run starts
    if (run.id !== liveRunId) {
      liveHistory = [];
      liveRunId = run.id;
    }
    if (run.step > 0 && run.loss !== null) {
      const last = liveHistory[liveHistory.length - 1];
      if (!last || last.step !== run.step) {
        liveHistory = [...liveHistory, {
          step: run.step,
          loss: run.loss,
          lr: run.learning_rate ?? 0,
          tokSec: run.tokens_per_second ?? 0,
        }];
      }
    }
  });

  // Fetch dataset columns when path changes, but only when the resolved path
  // is different from the last successfully peeked path to avoid resetting the
  // column selection on every keystroke while the user is still typing.
  $effect(() => {
    const path = datasetPath;
    if (!path || path.length < 3) {
      datasetColumns = [];
      return;
    }
    if (path === lastPeekedPath) return;
    lastPeekedPath = path;
    columnsLoading = true;
    peekDatasetColumns(path)
      .then(peek => {
        datasetColumns = peek.columns;
        datasetAvgTokens = peek.avg_tokens_estimate;
        datasetMaxTokens = peek.max_tokens_estimate;
        datasetSuggestedSeqLen = peek.suggested_seq_len;
        datasetRowsSampled = peek.rows_sampled;
        datasetFullScan = false;
        selectedTextColumns = [];
        // Auto-select: prefer 'text', else first column
        const cols = peek.columns;
        if (cols.length > 0) {
          if (cols.includes('text')) {
            selectedTextColumns = ['text'];
          } else if (cols.includes('content')) {
            selectedTextColumns = ['content'];
          } else if (cols.includes('instruction')) {
            selectedTextColumns = ['instruction'];
          } else if (cols.length === 1) {
            selectedTextColumns = [cols[0]];
          }
        }
        textColumn = selectedTextColumns[0] ?? 'text';
      })
      .catch(() => { datasetColumns = []; lastPeekedPath = ''; datasetSuggestedSeqLen = 0; })
      .finally(() => { columnsLoading = false; });
  });

  // SVG chart helpers
  function lossSvgPath(points: MetricPoint[], width: number, height: number): string {
    if (points.length < 2) return '';
    const losses = points.map(p => p.loss);
    const minL = Math.min(...losses);
    const maxL = Math.max(...losses);
    const range = maxL - minL || 0.001;
    const pad = 4;
    return points.map((p, i) => {
      const x = pad + (i / (points.length - 1)) * (width - 2 * pad);
      const y = pad + (1 - (p.loss - minL) / range) * (height - 2 * pad);
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
  }

  onMount(async () => {
    // Deep-link: pre-select model from query param
    const modelParam = $page.url.searchParams.get('model');
    if (modelParam) selectedModel = modelParam;

    // Load cached datasets for the dropdown
    try {
      cachedDatasets = await listCachedDatasets();
    } catch (e) {
      console.error('Failed to load cached datasets:', e);
    }
  });

  // Close fuse modal on Escape
  function handleFuseKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') showFuseModal = false;
  }

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    formSuccess = null;

    if (!selectedModel) { formError = 'Please select a model'; return; }
    if (!datasetPath.trim()) { formError = 'Please provide a dataset path'; return; }

    isSubmitting = true;
    try {
      const config: TrainingConfig = {
        model: selectedModel,
        dataset: datasetPath || null,
        method: selectedMethod,
        epochs,
        learning_rate: learningRate,
        batch_size: batchSize,
        lora_rank: isLoraMethod ? loraRank : null,
        lora_alpha: isLoraMethod ? loraAlpha : null,
        lora_dropout: isLoraMethod ? loraDropout : null,
        use_rslora: isLoraMethod ? useRslora : null,
        use_dora: isLoraMethod ? useDora : null,
        output_dir: outputDir || null,
        load_in_4bit: selectedMethod === 'qlora' ? loadIn4bit : null,
        gradient_accumulation_steps: gradAccumSteps,
        max_seq_len: maxSeqLen,
        text_column: selectedTextColumns.length > 1
          ? selectedTextColumns.join('+')
          : selectedTextColumns.length === 1
            ? selectedTextColumns[0]
            : textColumn || null,
        dataset_format: datasetFormat,
        embedding_lr: embeddingLr > 0 ? embeddingLr : null,
        jit_compilation: jitCompilation,
        gradient_checkpointing: gradientCheckpointing,
        flash_attention: flashAttention,
        fused_optimizer: fusedOptimizer,
        warmup_steps: warmupSteps,
        weight_decay: weightDecay,
        max_grad_norm: maxGradNorm,
        save_steps: saveSteps,
        logging_steps: loggingSteps,
        lr_scheduler: lrScheduler,
        sequence_packing: sequencePacking,
        resume_from: resumeFrom || null,
        prompt_column: null,
        response_column: null,
      };

      const runId = await trainingStore.start(config);
      formSuccess = `Training started (run ID: ${runId})`;
      selectedRunId = runId;
    } catch (e) {
      formError = e instanceof Error ? e.message : String(e);
    } finally {
      isSubmitting = false;
    }
  }

  async function handleFuse(e: Event) {
    e.preventDefault();
    if (!fuseBaseModel || !fuseLoraPath) return;
    fuseError = null;
    fuseSuccess = null;
    isFusing = true;
    try {
      const result = await fuseLora(fuseBaseModel, fuseLoraPath, fuseOutputDir || `${fuseLoraPath}-fused`);
      fuseSuccess = `LoRA fused. Output: ${result.output_dir}`;
    } catch (e) {
      fuseError = e instanceof Error ? e.message : String(e);
    } finally {
      isFusing = false;
    }
  }
</script>

<div class="space-y-6">
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div>
      <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Training</h1>
      <p class="text-surface-500 dark:text-surface-400 mt-1">Fine-tune models with LoRA, QLoRA, DPO, and more</p>
    </div>
    <div class="flex gap-2">
      <button class="btn-secondary btn-sm" aria-label="Fuse LoRA adapter into base model" onclick={() => { showFuseModal = true; fuseCustomPath = false; fuseError = null; fuseSuccess = null; listTrainedAdapters().then(a => trainedAdapters = a).catch(() => {}); }}>
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 14v6m-3-3h6M6 10h2a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v2a2 2 0 002 2zm10 0h2a2 2 0 002-2V6a2 2 0 00-2-2h-2a2 2 0 00-2 2v2a2 2 0 002 2zM6 20h2a2 2 0 002-2v-2a2 2 0 00-2-2H6a2 2 0 00-2 2v2a2 2 0 002 2z" />
        </svg>
        Fuse LoRA
      </button>
    </div>
  </div>

  <!-- Failed runs alert -->
  {#each trainingStore.runs.filter(r => r.status === 'failed') as run}
    <div class="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
      <div class="flex items-center justify-between">
        <div>
          <span class="font-semibold text-red-800 dark:text-red-200 text-sm">{run.model.split('/').pop()}</span>
          <span class="ml-2 text-xs text-red-600 dark:text-red-400">Failed</span>
        </div>
      </div>
      {#if run.error_message}
        <p class="mt-1 text-xs text-red-700 dark:text-red-300 font-mono break-all">{run.error_message}</p>
      {/if}
    </div>
  {/each}

  <!-- ════════════════════════════════════════════════════════════════
       LIVE TRAINING DASHBOARD — replaces config form when training
       ════════════════════════════════════════════════════════════════ -->
  {#if activeRuns.length > 0}
    {@const run = activeRuns[0]}
    <div class="space-y-4">
      <!-- Header bar: model name, status, progress, stop button -->
      <div class="card">
        <div class="card-body">
          <div class="flex items-center justify-between mb-3">
            <div>
              <span class="font-bold text-lg text-surface-900 dark:text-surface-100">{run.model.split('/').pop()}</span>
              <span class="ml-2 text-sm text-surface-500">{run.method.toUpperCase()}</span>
              {#if run.dataset}
                <span class="ml-2 text-xs text-surface-400">on {run.dataset.split('/').pop()}</span>
              {/if}
            </div>
            <div class="flex items-center gap-3">
              {#if run.status_message && run.step === 0}
                <span class="text-sm text-amber-600 dark:text-amber-400 animate-pulse">{run.status_message}</span>
              {:else}
                <span class="text-sm font-mono text-surface-600 dark:text-surface-300">
                  Epoch {Math.floor(run.epoch)}/{run.total_epochs} · Step {run.step}/{run.total_steps}
                </span>
              {/if}
              <span class="text-sm text-surface-500">ETA {formatEta(run.eta_seconds)}</span>
              <button class="btn-danger btn-sm" onclick={() => trainingStore.stop(run.id)}>Stop</button>
            </div>
          </div>
          {#if run.step > 0 || !run.status_message}
            <div class="progress-bar h-2">
              <div class="progress-bar-fill" style="width: {runProgress(run.step, run.total_steps)}%"></div>
            </div>
          {:else}
            <div class="progress-bar h-2 overflow-hidden">
              <div class="progress-bar-fill animate-pulse" style="width: 100%; opacity: 0.3"></div>
            </div>
          {/if}
        </div>
      </div>

      <!-- Metric cards row -->
      <div class="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3">
        <div class="card">
          <div class="card-body p-3 text-center">
            <p class="text-xs text-surface-500 mb-1">Loss</p>
            <p class="text-xl font-mono font-bold text-surface-900 dark:text-surface-100">
              {run.loss !== null ? run.loss.toFixed(4) : '--'}
            </p>
          </div>
        </div>
        <div class="card">
          <div class="card-body p-3 text-center">
            <p class="text-xs text-surface-500 mb-1">Best Loss</p>
            <p class="text-xl font-mono font-bold text-green-600 dark:text-green-400">
              {run.best_loss !== null ? run.best_loss.toFixed(4) : '--'}
            </p>
          </div>
        </div>
        <div class="card">
          <div class="card-body p-3 text-center">
            <p class="text-xs text-surface-500 mb-1">Tok/s</p>
            <p class="text-xl font-mono font-bold text-surface-900 dark:text-surface-100">
              {run.tokens_per_second !== null ? run.tokens_per_second.toFixed(0) : '--'}
            </p>
          </div>
        </div>
        <div class="card">
          <div class="card-body p-3 text-center">
            <p class="text-xs text-surface-500 mb-1">Learning Rate</p>
            <p class="text-xl font-mono font-bold text-surface-900 dark:text-surface-100">
              {run.learning_rate !== null ? run.learning_rate.toExponential(1) : '--'}
            </p>
          </div>
        </div>
        <div class="card">
          <div class="card-body p-3 text-center">
            <p class="text-xs text-surface-500 mb-1">Grad Norm</p>
            <p class="text-xl font-mono font-bold text-surface-900 dark:text-surface-100">
              {run.grad_norm !== null ? run.grad_norm.toFixed(3) : '--'}
            </p>
          </div>
        </div>
        <div class="card">
          <div class="card-body p-3 text-center">
            <p class="text-xs text-surface-500 mb-1">Progress</p>
            <p class="text-xl font-mono font-bold text-surface-900 dark:text-surface-100">
              {run.total_steps > 0 ? Math.round(run.step / run.total_steps * 100) : 0}%
            </p>
          </div>
        </div>
      </div>

      <!-- Loss curve chart + config details -->
      <div class="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <!-- Loss Curve (SVG) -->
        <div class="xl:col-span-2 card">
          <div class="card-body">
            <p class="text-sm font-semibold text-surface-700 dark:text-surface-300 mb-2">Loss Curve</p>
            {#if liveHistory.length >= 2}
              <svg viewBox="0 0 600 200" class="w-full h-48" preserveAspectRatio="none">
                <!-- Grid lines -->
                <line x1="4" y1="4" x2="4" y2="196" stroke="currentColor" stroke-opacity="0.1" />
                <line x1="4" y1="196" x2="596" y2="196" stroke="currentColor" stroke-opacity="0.1" />
                <line x1="4" y1="100" x2="596" y2="100" stroke="currentColor" stroke-opacity="0.05" stroke-dasharray="4" />
                <!-- Loss line -->
                <path d={lossSvgPath(liveHistory, 600, 200)} fill="none" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
              </svg>
              <div class="flex justify-between text-xs text-surface-400 mt-1">
                <span>Step {liveHistory[0].step}</span>
                <span>Step {liveHistory[liveHistory.length - 1].step}</span>
              </div>
            {:else}
              <div class="h-48 flex items-center justify-center text-surface-400 text-sm">
                Waiting for training data...
              </div>
            {/if}
          </div>
        </div>

        <!-- Config & run details -->
        <div class="card">
          <div class="card-body">
            <p class="text-sm font-semibold text-surface-700 dark:text-surface-300 mb-3">Run Details</p>
            <div class="space-y-2 text-xs">
              <div class="flex justify-between">
                <span class="text-surface-500">Model</span>
                <span class="font-mono text-surface-700 dark:text-surface-300 truncate ml-2">{run.model}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-surface-500">Method</span>
                <span class="font-mono">{run.method.toUpperCase()}</span>
              </div>
              {#if run.dataset}
                <div class="flex justify-between">
                  <span class="text-surface-500">Dataset</span>
                  <span class="font-mono truncate ml-2">{run.dataset.split('/').pop()}</span>
                </div>
              {/if}
              {#if run.output_dir}
                <div class="flex justify-between">
                  <span class="text-surface-500">Output</span>
                  <span class="font-mono truncate ml-2">{run.output_dir}</span>
                </div>
              {/if}
              {#if run.config_summary}
                <hr class="border-surface-200 dark:border-surface-600" />
                <div class="flex justify-between">
                  <span class="text-surface-500">Learning Rate</span>
                  <span class="font-mono">{run.config_summary.learning_rate}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-surface-500">Batch Size</span>
                  <span class="font-mono">{run.config_summary.batch_size}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-surface-500">Seq Length</span>
                  <span class="font-mono">{run.config_summary.max_seq_len}</span>
                </div>
                {#if run.config_summary.lora_rank}
                  <div class="flex justify-between">
                    <span class="text-surface-500">LoRA</span>
                    <span class="font-mono">r={run.config_summary.lora_rank} a={run.config_summary.lora_alpha}</span>
                  </div>
                {/if}
                <div class="flex flex-wrap gap-1 mt-1">
                  {#if run.config_summary.sequence_packing}<span class="badge badge-xs">Packing</span>{/if}
                  {#if run.config_summary.flash_attention}<span class="badge badge-xs">FlashAttn</span>{/if}
                  {#if run.config_summary.gradient_checkpointing}<span class="badge badge-xs">GradCkpt</span>{/if}
                  {#if run.config_summary.jit_compilation}<span class="badge badge-xs">JIT</span>{/if}
                </div>
              {/if}
            </div>
          </div>
        </div>
      </div>
    </div>

  {:else}
  <!-- ════════════════════════════════════════════════════════════════
       CONFIGURATION FORM — shown when no training is active
       ════════════════════════════════════════════════════════════════ -->

  <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
    <!-- Training Form -->
    <div class="xl:col-span-2 space-y-4">
      <!-- Method Selector -->
      <div class="card">
        <div class="card-body">
          <p class="section-title">Training Method</p>
          <div class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
            {#each trainingMethods as method}
              <button
                type="button"
                class="p-2 rounded-lg border text-center transition-all {selectedMethod === method.value
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                  : 'border-surface-200 dark:border-surface-700 hover:border-surface-300 dark:hover:border-surface-600'}"
                title={method.description}
                onclick={() => (selectedMethod = method.value)}
              >
                <span class="text-sm font-semibold">{method.label}</span>
              </button>
            {/each}
          </div>
          <p class="text-xs text-surface-500 dark:text-surface-400 mt-2">
            {trainingMethods.find(m => m.value === selectedMethod)?.description ?? ''}
          </p>
        </div>
      </div>

      <form onsubmit={handleSubmit} class="space-y-4">
        <!-- Model & Data -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Model & Data</h3>
          </div>
          <div class="card-body space-y-4">
            <div>
              <label class="label" for="model">Base Model</label>
              <select id="model" class="input" bind:value={selectedModel}>
                <option value="">Select a cached model...</option>
                {#each models as model}
                  <option value={model.id}>{model.id} ({model.size_formatted})</option>
                {/each}
              </select>
            </div>
            <div>
              <label class="label" for="dataset">Dataset</label>
              <select id="dataset" class="input" bind:value={datasetPath}>
                <option value="">Select a cached dataset...</option>
                {#each cachedDatasets as ds}
                  <option value={ds.path}>{ds.name} ({ds.size_formatted})</option>
                {/each}
              </select>
              <input
                type="text"
                class="input mt-2"
                placeholder="Or enter a path / HuggingFace dataset ID"
                bind:value={datasetPath}
              />
            </div>
            <!-- Column detection & selection -->
            {#if datasetColumns.length > 0}
              <div class="p-3 rounded-lg bg-surface-50 dark:bg-surface-800/50 border border-surface-200 dark:border-surface-700">
                <p class="text-xs font-semibold text-surface-600 dark:text-surface-400 mb-2">
                  Detected columns: <span class="font-mono">{datasetColumns.join(', ')}</span>
                </p>

                <!-- Ordered text column builder -->
                <div class="mb-3">
                  <label class="label text-xs mb-1" for="add-column">Text Columns <span class="text-surface-400">(in order)</span></label>
                  <!-- Selected columns as reorderable pills -->
                  {#if selectedTextColumns.length > 0}
                    <div class="flex flex-wrap gap-1.5 mb-2">
                      {#each selectedTextColumns as col, i}
                        <span class="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-primary-100 dark:bg-primary-900/40 text-primary-800 dark:text-primary-200 text-xs font-mono border border-primary-200 dark:border-primary-700">
                          {#if i > 0}
                            <button
                              type="button"
                              class="hover:text-primary-500 -ml-0.5"
                              title="Move left"
                              onclick={() => {
                                const arr = [...selectedTextColumns];
                                [arr[i - 1], arr[i]] = [arr[i], arr[i - 1]];
                                selectedTextColumns = arr;
                                textColumn = arr[0];
                              }}
                            >&larr;</button>
                          {/if}
                          {col}
                          {#if i < selectedTextColumns.length - 1}
                            <button
                              type="button"
                              class="hover:text-primary-500"
                              title="Move right"
                              onclick={() => {
                                const arr = [...selectedTextColumns];
                                [arr[i], arr[i + 1]] = [arr[i + 1], arr[i]];
                                selectedTextColumns = arr;
                                textColumn = arr[0];
                              }}
                            >&rarr;</button>
                          {/if}
                          <button
                            type="button"
                            class="hover:text-red-500 ml-0.5"
                            title="Remove"
                            onclick={() => {
                              selectedTextColumns = selectedTextColumns.filter((_, idx) => idx !== i);
                              textColumn = selectedTextColumns[0] ?? 'text';
                            }}
                          >&times;</button>
                        </span>
                        {#if i < selectedTextColumns.length - 1}
                          <span class="text-xs text-surface-400 self-center">+</span>
                        {/if}
                      {/each}
                    </div>
                  {/if}
                  <!-- Add column selector (adds on select) -->
                  {#if datasetColumns.filter(c => !selectedTextColumns.includes(c)).length > 0}
                    <select
                      id="add-column"
                      class="input text-sm"
                      value=""
                      onchange={(e) => {
                        const val = (e.target as HTMLSelectElement).value;
                        if (val && !selectedTextColumns.includes(val)) {
                          selectedTextColumns = [...selectedTextColumns, val];
                          textColumn = selectedTextColumns[0];
                        }
                        (e.target as HTMLSelectElement).value = '';
                      }}
                    >
                      <option value="">Add a column...</option>
                      {#each datasetColumns.filter(c => !selectedTextColumns.includes(c)) as col}
                        <option value={col}>{col}</option>
                      {/each}
                    </select>
                  {/if}
                </div>

                <div class="grid grid-cols-2 gap-3">
                  <div>
                    <label class="label text-xs" for="prompt-column">Prompt Column <span class="text-surface-400">(loss-masked)</span></label>
                    <select id="prompt-column" class="input text-sm">
                      <option value="">None (all tokens train)</option>
                      {#each datasetColumns as col}
                        <option value={col}>{col}</option>
                      {/each}
                    </select>
                  </div>
                  <div>
                    <label class="label text-xs" for="dataset-format">Format</label>
                    <select id="dataset-format" class="input text-sm" bind:value={datasetFormat}>
                      {#each datasetFormats as fmt}
                        <option value={fmt}>{fmt}</option>
                      {/each}
                    </select>
                  </div>
                </div>
              </div>
            {:else}
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="label" for="dataset-format">Dataset Format</label>
                  <select id="dataset-format" class="input" bind:value={datasetFormat}>
                    {#each datasetFormats as fmt}
                      <option value={fmt}>{fmt}</option>
                    {/each}
                  </select>
                </div>
                <div>
                  <label class="label" for="text-column">Text Column</label>
                  <input id="text-column" type="text" class="input" placeholder="text" bind:value={textColumn} />
                </div>
              </div>
            {/if}
          </div>
        </div>

        <!-- LoRA Configuration (shown for all methods except bare SFT) -->
        {#if isLoraMethod}
          <div class="card">
            <div class="card-header">
              <h3 class="font-semibold text-surface-900 dark:text-surface-100">LoRA Configuration</h3>
            </div>
            <div class="card-body space-y-4">
              <div class="grid grid-cols-3 gap-4">
                <div>
                  <label class="label" for="lora-rank">Rank</label>
                  <input id="lora-rank" type="number" class="input" min="4" max="256" step="4" bind:value={loraRank} />
                </div>
                <div>
                  <label class="label" for="lora-alpha">Alpha</label>
                  <input id="lora-alpha" type="number" class="input" min="1" bind:value={loraAlpha} />
                </div>
                <div>
                  <label class="label" for="lora-dropout">Dropout</label>
                  <input id="lora-dropout" type="number" class="input" min="0" max="1" step="0.01" bind:value={loraDropout} />
                </div>
              </div>
              <div class="flex gap-6">
                <label class="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" class="rounded border-surface-300" bind:checked={useRslora} />
                  <span class="text-sm text-surface-700 dark:text-surface-300">RSLoRA</span>
                </label>
                <label class="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" class="rounded border-surface-300" bind:checked={useDora} />
                  <span class="text-sm text-surface-700 dark:text-surface-300">DoRA</span>
                </label>
                {#if selectedMethod === 'qlora'}
                  <label class="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" class="rounded border-surface-300" bind:checked={loadIn4bit} />
                    <span class="text-sm text-surface-700 dark:text-surface-300">4-bit quantization</span>
                  </label>
                {/if}
              </div>
            </div>
          </div>
        {/if}

        <!-- Training Hyperparameters -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Training Hyperparameters</h3>
          </div>
          <div class="card-body space-y-4">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label class="label" for="epochs">Epochs</label>
                <input id="epochs" type="number" class="input" min="1" bind:value={epochs} />
              </div>
              <div>
                <label class="label" for="lr">Learning Rate</label>
                <input id="lr" type="number" class="input" step="0.00001" bind:value={learningRate} />
              </div>
              <div>
                <label class="label" for="batch-size">Batch Size</label>
                <input id="batch-size" type="number" class="input" min="1" bind:value={batchSize} />
              </div>
              <div>
                <label class="label" for="grad-accum">Grad Accumulation</label>
                <input id="grad-accum" type="number" class="input" min="1" bind:value={gradAccumSteps} />
              </div>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <label class="label" for="max-seq-len">Max Seq Length</label>
                <input id="max-seq-len" type="number" class="input" step="1" min="1" bind:value={maxSeqLen} />
                {#if datasetSuggestedSeqLen > 0}
                  {#if maxSeqLen < datasetAvgTokens}
                    <p class="text-xs text-red-500 mt-1">
                      Most samples will be truncated (avg ~{datasetAvgTokens} tokens, max ~{datasetMaxTokens}). Suggest {datasetSuggestedSeqLen}.
                    </p>
                  {:else if maxSeqLen < datasetSuggestedSeqLen && datasetMaxTokens > maxSeqLen}
                    <p class="text-xs text-amber-500 mt-1">
                      Some samples may be truncated (max ~{datasetMaxTokens} tokens). Suggest {datasetSuggestedSeqLen}.
                    </p>
                  {:else if maxSeqLen > datasetMaxTokens * 2 && datasetMaxTokens > 0}
                    <p class="text-xs text-blue-500 mt-1">
                      Seq len is much larger than data (max ~{datasetMaxTokens} tokens). Could reduce to {datasetSuggestedSeqLen}.
                    </p>
                  {/if}
                  <p class="text-xs text-surface-400 mt-0.5">
                    {#if scanningAll}
                      Scanning full dataset...
                    {:else}
                      Based on first {datasetRowsSampled} rows{#if !datasetFullScan}
                        — <button
                          type="button"
                          class="text-primary-500 hover:text-primary-400 underline"
                          onclick={async () => {
                            scanningAll = true;
                            try {
                              const peek = await peekDatasetColumns(datasetPath, 0);
                              datasetAvgTokens = peek.avg_tokens_estimate;
                              datasetMaxTokens = peek.max_tokens_estimate;
                              datasetSuggestedSeqLen = peek.suggested_seq_len;
                              datasetRowsSampled = peek.rows_sampled;
                              datasetFullScan = true;
                            } catch { /* ignore */ }
                            scanningAll = false;
                          }}
                        >check all rows</button>
                      {/if}
                    {/if}
                  </p>
                {/if}
              </div>
              <div>
                <label class="label" for="lr-scheduler">LR Scheduler</label>
                <select id="lr-scheduler" class="input" bind:value={lrScheduler}>
                  {#each lrSchedulers as sched}
                    <option value={sched}>{sched}</option>
                  {/each}
                </select>
              </div>
              <div>
                <label class="label" for="output-dir">Output Directory</label>
                <input id="output-dir" type="text" class="input" placeholder="Auto: ~/pmetal-output/model-method-YYYYMMDD" bind:value={outputDir} />
              </div>
            </div>

            <!-- Advanced hyperparams toggle -->
            <button
              type="button"
              class="flex items-center gap-2 text-sm text-primary-600 dark:text-primary-400 hover:underline"
              aria-expanded={showAdvanced}
              onclick={() => (showAdvanced = !showAdvanced)}
            >
              <svg class="w-4 h-4 transition-transform {showAdvanced ? 'rotate-90' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
              </svg>
              {showAdvanced ? 'Hide' : 'Show'} advanced hyperparameters
            </button>

            {#if showAdvanced}
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2 border-t border-surface-200 dark:border-surface-700">
                <div>
                  <label class="label" for="warmup-steps">Warmup Steps</label>
                  <input id="warmup-steps" type="number" class="input" min="0" bind:value={warmupSteps} />
                </div>
                <div>
                  <label class="label" for="weight-decay">Weight Decay</label>
                  <input id="weight-decay" type="number" class="input" step="0.001" bind:value={weightDecay} />
                </div>
                <div>
                  <label class="label" for="max-grad-norm">Max Grad Norm</label>
                  <input id="max-grad-norm" type="number" class="input" step="0.1" bind:value={maxGradNorm} />
                </div>
                <div>
                  <label class="label" for="save-steps">Save Steps</label>
                  <input id="save-steps" type="number" class="input" min="1" bind:value={saveSteps} />
                </div>
                <div>
                  <label class="label" for="logging-steps">Logging Steps</label>
                  <input id="logging-steps" type="number" class="input" min="1" bind:value={loggingSteps} />
                </div>
                <div>
                  <label class="label" for="resume-from">Resume From</label>
                  <input id="resume-from" type="text" class="input" placeholder="/path/to/checkpoint" bind:value={resumeFrom} />
                </div>
              </div>
            {/if}
          </div>
        </div>

        <!-- PMetal Optimizations -->
        <div class="card">
          <div
            class="card-header cursor-pointer flex items-center justify-between"
            role="button"
            tabindex="0"
            aria-expanded={showPmetalOpts}
            onclick={() => (showPmetalOpts = !showPmetalOpts)}
            onkeydown={(e) => e.key === 'Enter' && (showPmetalOpts = !showPmetalOpts)}
          >
            <h3 class="font-semibold text-surface-900 dark:text-surface-100 flex items-center gap-2">
              <div class="w-2 h-2 rounded-full bg-accent-500" aria-hidden="true"></div>
              PMetal Optimizations
            </h3>
            <svg class="w-4 h-4 text-surface-500 transition-transform {showPmetalOpts ? 'rotate-180' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </div>
          {#if showPmetalOpts}
            <div class="card-body space-y-4">
              <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
                <label class="flex items-start gap-3 cursor-pointer">
                  <input type="checkbox" class="mt-0.5 rounded border-surface-300" bind:checked={jitCompilation} />
                  <div>
                    <span class="text-sm font-medium text-surface-700 dark:text-surface-300">JIT Compilation</span>
                    <p class="text-xs text-surface-500 mt-0.5">Compile the training graph</p>
                  </div>
                </label>
                <label class="flex items-start gap-3 cursor-pointer">
                  <input type="checkbox" class="mt-0.5 rounded border-surface-300" bind:checked={gradientCheckpointing} />
                  <div>
                    <span class="text-sm font-medium text-surface-700 dark:text-surface-300">Gradient Checkpointing</span>
                    <p class="text-xs text-surface-500 mt-0.5">Reduce memory usage</p>
                  </div>
                </label>
                <label class="flex items-start gap-3 cursor-pointer">
                  <input type="checkbox" class="mt-0.5 rounded border-surface-300" bind:checked={sequencePacking} />
                  <div>
                    <span class="text-sm font-medium text-surface-700 dark:text-surface-300">Sequence Packing</span>
                    <p class="text-xs text-surface-500 mt-0.5">Pack short sequences</p>
                  </div>
                </label>
                <label class="flex items-start gap-3 cursor-pointer">
                  <input type="checkbox" class="mt-0.5 rounded border-surface-300" bind:checked={flashAttention} />
                  <div>
                    <span class="text-sm font-medium text-surface-700 dark:text-surface-300">Flash Attention</span>
                    <p class="text-xs text-surface-500 mt-0.5">Fused SDPA kernel</p>
                  </div>
                </label>
                <label class="flex items-start gap-3 cursor-pointer">
                  <input type="checkbox" class="mt-0.5 rounded border-surface-300" bind:checked={fusedOptimizer} />
                  <div>
                    <span class="text-sm font-medium text-surface-700 dark:text-surface-300">Fused Optimizer</span>
                    <p class="text-xs text-surface-500 mt-0.5">Metal-fused Adam step</p>
                  </div>
                </label>
              </div>
              <div>
                <label class="label" for="embedding-lr">Embedding LR (0 = same as base)</label>
                <input id="embedding-lr" type="number" class="input max-w-[200px]" step="0.00001" min="0" bind:value={embeddingLr} />
              </div>
            </div>
          {/if}
        </div>

        <!-- Error / Success -->
        {#if formError}
          <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
            {formError}
          </div>
        {/if}
        {#if formSuccess}
          <div class="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-sm" role="status">
            {formSuccess}
          </div>
        {/if}

        <button type="submit" class="btn-primary w-full" disabled={isSubmitting}>
          {#if isSubmitting}
            <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
            Starting...
          {:else}
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Start Training
          {/if}
        </button>
      </form>
    </div>

    <!-- Run History -->
    <div class="xl:col-span-1">
      <div class="card sticky top-0">
        <div class="card-header flex items-center justify-between">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Run History</h3>
          <span class="badge-neutral">{runs.length}</span>
        </div>
        <div class="divide-y divide-surface-200 dark:divide-surface-700 max-h-[600px] overflow-y-auto scrollbar-thin">
          {#if runs.length === 0}
            <div class="p-6 text-center text-surface-500 dark:text-surface-400 text-sm">
              No training runs yet
            </div>
          {:else}
            {#each runs as run}
              <button
                class="w-full text-left p-4 hover:bg-surface-50 dark:hover:bg-surface-700/50 transition-colors {selectedRunId === run.id ? 'bg-primary-50 dark:bg-primary-900/20' : ''}"
                onclick={() => (selectedRunId = selectedRunId === run.id ? null : run.id)}
              >
                <div class="flex items-center justify-between mb-1">
                  <span class="text-sm font-medium text-surface-900 dark:text-surface-100 truncate max-w-[140px]">
                    {run.model.split('/').pop()}
                  </span>
                  <span class={getStatusBadgeClass(run.status)}>{run.status}</span>
                </div>
                <div class="text-xs text-surface-500 dark:text-surface-400 mb-2">
                  {run.method.toUpperCase()} · {run.step}/{run.total_steps} steps
                </div>
                {#if run.status === 'running'}
                  <div class="progress-bar">
                    <div class="progress-bar-fill" style="width: {runProgress(run.step, run.total_steps)}%"></div>
                  </div>
                  <div class="mt-1 text-xs text-surface-500 flex gap-2">
                    {#if run.loss !== null}<span>Loss: {run.loss.toFixed(4)}</span>{/if}
                    <span>ETA: {formatEta(run.eta_seconds)}</span>
                  </div>
                {:else if run.status === 'completed'}
                  <div class="text-xs text-surface-500">
                    Best loss: {run.best_loss?.toFixed(4) ?? '--'}
                  </div>
                {:else if run.status === 'failed' && run.error_message}
                  <div class="text-xs text-red-500 truncate">{run.error_message}</div>
                {/if}
              </button>
            {/each}
          {/if}
        </div>

        <!-- Selected run detail -->
        {#if selectedRun}
          <div class="card-footer">
            <div class="space-y-3 text-sm">
              <!-- Status message for setup phases -->
              {#if selectedRun.status_message && selectedRun.step === 0}
                <div class="text-amber-600 dark:text-amber-400 text-xs font-medium">
                  {selectedRun.status_message}
                </div>
              {/if}

              <!-- Error message -->
              {#if selectedRun.error_message}
                <div class="p-2 rounded bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-xs font-mono break-all">
                  {selectedRun.error_message}
                </div>
              {/if}

              <!-- Live metrics -->
              {#if selectedRun.status === 'running' || selectedRun.status === 'completed'}
                <div class="grid grid-cols-2 gap-2">
                  {#if selectedRun.loss !== null}
                    <div>
                      <span class="text-surface-500">Loss</span>
                      <span class="font-mono ml-2">{selectedRun.loss.toFixed(4)}</span>
                    </div>
                  {/if}
                  {#if selectedRun.best_loss !== null}
                    <div>
                      <span class="text-surface-500">Best</span>
                      <span class="font-mono ml-2 text-green-600 dark:text-green-400">{selectedRun.best_loss.toFixed(4)}</span>
                    </div>
                  {/if}
                  {#if selectedRun.learning_rate !== null}
                    <div>
                      <span class="text-surface-500">LR</span>
                      <span class="font-mono ml-2">{selectedRun.learning_rate.toExponential(2)}</span>
                    </div>
                  {/if}
                  {#if selectedRun.tokens_per_second !== null}
                    <div>
                      <span class="text-surface-500">Tok/s</span>
                      <span class="font-mono ml-2">{selectedRun.tokens_per_second.toFixed(0)}</span>
                    </div>
                  {/if}
                  {#if selectedRun.grad_norm !== null}
                    <div>
                      <span class="text-surface-500">Grad</span>
                      <span class="font-mono ml-2">{selectedRun.grad_norm.toFixed(3)}</span>
                    </div>
                  {/if}
                </div>
              {/if}

              <!-- Config summary -->
              {#if selectedRun.config_summary}
                <div class="border-t border-surface-200 dark:border-surface-600 pt-2">
                  <p class="text-xs font-semibold text-surface-500 mb-1">Configuration</p>
                  <div class="grid grid-cols-2 gap-1 text-xs text-surface-600 dark:text-surface-400">
                    <div>Model: <span class="font-mono">{selectedRun.model.split('/').pop()}</span></div>
                    <div>Method: <span class="font-mono">{selectedRun.method.toUpperCase()}</span></div>
                    <div>LR: <span class="font-mono">{selectedRun.config_summary.learning_rate}</span></div>
                    <div>Batch: <span class="font-mono">{selectedRun.config_summary.batch_size}</span></div>
                    <div>Seq len: <span class="font-mono">{selectedRun.config_summary.max_seq_len}</span></div>
                    {#if selectedRun.config_summary.lora_rank}
                      <div>LoRA: <span class="font-mono">r={selectedRun.config_summary.lora_rank} a={selectedRun.config_summary.lora_alpha}</span></div>
                    {/if}
                    <div>Packing: <span class="font-mono">{selectedRun.config_summary.sequence_packing ? 'on' : 'off'}</span></div>
                    <div>FlashAttn: <span class="font-mono">{selectedRun.config_summary.flash_attention ? 'on' : 'off'}</span></div>
                    {#if selectedRun.config_summary.gradient_checkpointing}
                      <div>GradCkpt: <span class="font-mono">on</span></div>
                    {/if}
                    {#if selectedRun.config_summary.jit_compilation}
                      <div>JIT: <span class="font-mono">on</span></div>
                    {/if}
                  </div>
                </div>
              {/if}

              {#if selectedRun.dataset}
                <p class="text-xs text-surface-500 truncate">Dataset: {selectedRun.dataset}</p>
              {/if}
              {#if selectedRun.output_dir}
                <p class="text-xs text-surface-500 truncate">Output: {selectedRun.output_dir}</p>
              {/if}

              {#if selectedRun.status === 'running'}
                <button
                  class="btn-danger btn-sm w-full"
                  onclick={() => trainingStore.stop(selectedRun!.id)}
                  aria-label="Stop this training run"
                >
                  Stop Training
                </button>
              {:else if selectedRun.config_summary}
                <button
                  class="btn-primary btn-sm w-full"
                  onclick={() => {
                    const r = selectedRun!;
                    const c = r.config_summary!;
                    // Load config back into form
                    selectedModel = r.model;
                    selectedMethod = r.method;
                    if (r.dataset) datasetPath = r.dataset;
                    learningRate = c.learning_rate;
                    batchSize = c.batch_size;
                    maxSeqLen = c.max_seq_len;
                    if (c.lora_rank) { loraRank = c.lora_rank; loraAlpha = c.lora_alpha ?? 32; }
                    sequencePacking = c.sequence_packing;
                    flashAttention = c.flash_attention;
                    jitCompilation = c.jit_compilation;
                    gradientCheckpointing = c.gradient_checkpointing;
                    // Clear selection to show the form
                    selectedRunId = null;
                  }}
                  aria-label="Load this run's settings into the form"
                >
                  Retry with these settings
                </button>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>
  {/if}
</div>

<!-- Fuse LoRA Modal -->
{#if showFuseModal}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" role="presentation" onkeydown={handleFuseKeydown}>
    <div
      class="card w-full max-w-md"
      role="dialog"
      aria-modal="true"
      aria-labelledby="fuse-modal-title"
      tabindex="-1"
    >
      <div class="card-header flex items-center justify-between">
        <h3 id="fuse-modal-title" class="font-semibold text-surface-900 dark:text-surface-100">Fuse LoRA Adapter</h3>
        <button class="btn-ghost btn-sm" aria-label="Close dialog" onclick={() => (showFuseModal = false)}>
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <form onsubmit={handleFuse} class="card-body space-y-4">
        <div>
          <label class="label" for="fuse-lora">LoRA Adapter</label>
          {#if trainedAdapters.length > 0 && !fuseCustomPath}
            <select id="fuse-lora" class="input" bind:value={fuseLoraPath} onchange={(e) => {
              const val = (e.target as HTMLSelectElement).value;
              if (val === '__custom__') { fuseCustomPath = true; fuseLoraPath = ''; return; }
              const adapter = trainedAdapters.find(a => a.path === val);
              if (adapter?.base_model) {
                const match = models.find(m => m.id === adapter.base_model || m.id.endsWith('/' + adapter.base_model));
                if (match) fuseBaseModel = match.id;
              }
            }}>
              <option value="">Select trained adapter...</option>
              {#each trainedAdapters as adapter}
                <option value={adapter.path}>
                  {adapter.name}{adapter.rank ? ` (r=${adapter.rank})` : ''}{adapter.base_model ? ` — ${adapter.base_model.split('/').pop()}` : ''} — {(adapter.size_bytes / 1048576).toFixed(0)} MB
                </option>
              {/each}
              <option value="__custom__">Custom path...</option>
            </select>
          {:else}
            <div class="flex gap-2">
              <input id="fuse-lora" type="text" class="input flex-1" placeholder="/path/to/lora/adapter" bind:value={fuseLoraPath} />
              {#if trainedAdapters.length > 0}
                <button type="button" class="btn-ghost btn-sm" onclick={() => { fuseCustomPath = false; fuseLoraPath = ''; }}>List</button>
              {/if}
            </div>
          {/if}
        </div>
        <div>
          <label class="label" for="fuse-base">Base Model</label>
          <input id="fuse-base" type="text" class="input bg-surface-100 dark:bg-surface-700 cursor-not-allowed" value={fuseBaseModel || '(auto-detected from adapter)'} readonly />
        </div>
        <div>
          <label class="label" for="fuse-output">Output Directory</label>
          <input id="fuse-output" type="text" class="input" placeholder="/path/to/fused-model" bind:value={fuseOutputDir} />
        </div>
        {#if fuseError}
          <div class="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-sm" role="alert">{fuseError}</div>
        {/if}
        {#if fuseSuccess}
          <div class="p-3 rounded-lg bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 text-sm" role="status">{fuseSuccess}</div>
        {/if}
        <div class="flex gap-3">
          {#if fuseSuccess}
            <button type="submit" class="btn-primary flex-1" disabled={isFusing || !fuseBaseModel || !fuseLoraPath}>Fuse Another</button>
            <button type="button" class="btn-secondary" onclick={() => (showFuseModal = false)}>Done</button>
          {:else}
            <button type="submit" class="btn-primary flex-1" disabled={isFusing || !fuseBaseModel || !fuseLoraPath}>
              {#if isFusing}
                <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
                Fusing...
              {:else}
                Fuse
              {/if}
            </button>
            <button type="button" class="btn-secondary" onclick={() => (showFuseModal = false)}>Cancel</button>
          {/if}
        </div>
      </form>
    </div>
  </div>
{/if}
