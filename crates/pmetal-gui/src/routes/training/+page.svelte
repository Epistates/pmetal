<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { modelsStore, trainingStore } from '$lib/stores.svelte';
  import type { TrainingConfig, TrainingRun } from '$lib/api';
  import { fuseLora } from '$lib/api';
  import { formatEta, runProgress, getStatusBadgeClass } from '$lib/utils';

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
  let epochs = $state(3);
  let learningRate = $state(0.0002);
  let batchSize = $state(4);
  let gradAccumSteps = $state(4);
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
  let gradientCheckpointing = $state(false);
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

  // Derived state
  let models = $derived(modelsStore.models);
  let runs = $derived(trainingStore.runs);
  let activeRuns = $derived(trainingStore.activeRuns);
  let selectedRun = $derived(runs.find(r => r.id === selectedRunId) ?? null);
  // PMetal always uses LoRA; show config for all methods except bare SFT
  let isLoraMethod = $derived(selectedMethod !== 'sft');

  onMount(() => {
    // Deep-link: pre-select model from query param
    const modelParam = $page.url.searchParams.get('model');
    if (modelParam) selectedModel = modelParam;
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
        text_column: textColumn || null,
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
      <button class="btn-secondary btn-sm" aria-label="Fuse LoRA adapter into base model" onclick={() => (showFuseModal = true)}>
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 14v6m-3-3h6M6 10h2a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v2a2 2 0 002 2zm10 0h2a2 2 0 002-2V6a2 2 0 00-2-2h-2a2 2 0 00-2 2v2a2 2 0 002 2zM6 20h2a2 2 0 002-2v-2a2 2 0 00-2-2H6a2 2 0 00-2 2v2a2 2 0 002 2z" />
        </svg>
        Fuse LoRA
      </button>
    </div>
  </div>

  <!-- Active training banner -->
  {#if activeRuns.length > 0}
    <div class="space-y-3">
      {#each activeRuns as run}
        <div class="p-4 rounded-xl bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800">
          <div class="flex items-center justify-between mb-2">
            <div>
              <span class="font-semibold text-primary-900 dark:text-primary-100 text-sm">{run.model.split('/').pop()}</span>
              <span class="ml-2 text-xs text-primary-700 dark:text-primary-300">{run.method.toUpperCase()} · Epoch {run.epoch}/{run.total_epochs} · Step {run.step}/{run.total_steps}</span>
            </div>
            <div class="flex items-center gap-3">
              {#if run.loss !== null}
                <span class="text-xs font-mono text-primary-800 dark:text-primary-200">Loss: {run.loss.toFixed(4)}</span>
              {/if}
              {#if run.tokens_per_second !== null}
                <span class="text-xs font-mono text-primary-800 dark:text-primary-200">{run.tokens_per_second.toFixed(0)} tok/s</span>
              {/if}
              <span class="text-xs text-primary-700 dark:text-primary-300">ETA {formatEta(run.eta_seconds)}</span>
              <button
                class="btn-danger btn-sm"
                onclick={() => trainingStore.stop(run.id)}
                aria-label="Stop training run"
              >Stop</button>
            </div>
          </div>
          <div class="progress-bar">
            <div class="progress-bar-fill" style="width: {runProgress(run.step, run.total_steps)}%"></div>
          </div>
        </div>
      {/each}
    </div>
  {/if}

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
              <label class="label" for="dataset">Dataset Path</label>
              <input
                id="dataset"
                type="text"
                class="input"
                placeholder="/path/to/dataset or HuggingFace dataset ID"
                bind:value={datasetPath}
              />
            </div>
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
                <input id="max-seq-len" type="number" class="input" step="64" bind:value={maxSeqLen} />
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
                <input id="output-dir" type="text" class="input" placeholder="./output" bind:value={outputDir} />
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
        {#if selectedRun && (selectedRun.status === 'running' || selectedRun.status === 'completed')}
          <div class="card-footer">
            <div class="space-y-2 text-sm">
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
                    <span class="font-mono ml-2">{selectedRun.best_loss.toFixed(4)}</span>
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
              </div>
              {#if selectedRun.status === 'running'}
                <button
                  class="btn-danger btn-sm w-full"
                  onclick={() => trainingStore.stop(selectedRun!.id)}
                  aria-label="Stop this training run"
                >
                  Stop Training
                </button>
              {/if}
              {#if selectedRun.output_dir}
                <p class="text-xs text-surface-500 truncate">Output: {selectedRun.output_dir}</p>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>
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
          <label class="label" for="fuse-base">Base Model</label>
          <select id="fuse-base" class="input" bind:value={fuseBaseModel}>
            <option value="">Select base model...</option>
            {#each models as model}
              <option value={model.id}>{model.id}</option>
            {/each}
          </select>
        </div>
        <div>
          <label class="label" for="fuse-lora">LoRA Adapter Path</label>
          <input id="fuse-lora" type="text" class="input" placeholder="/path/to/lora/adapter" bind:value={fuseLoraPath} />
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
          <button type="submit" class="btn-primary flex-1" disabled={isFusing || !fuseBaseModel || !fuseLoraPath}>
            {#if isFusing}
              <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
              Fusing...
            {:else}
              Fuse
            {/if}
          </button>
          <button type="button" class="btn-secondary" onclick={() => (showFuseModal = false)}>Cancel</button>
        </div>
      </form>
    </div>
  </div>
{/if}
