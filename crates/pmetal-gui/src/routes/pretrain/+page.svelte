<script lang="ts">
  import { pretrainStore } from '$lib/stores.svelte';
  import { formatEta, runProgress, getStatusBadgeClass } from '$lib/utils';

  const architectures = [
    'llama',
    'qwen2',
    'qwen3',
    'qwen3.5',
    'qwen3_moe',
    'gemma',
    'mistral',
    'phi',
    'gpt-oss',
  ];

  const lrSchedules = [
    { value: 'cosine', label: 'Cosine' },
    { value: 'linear', label: 'Linear' },
    { value: 'constant', label: 'Constant' },
  ];

  // Form state
  let arch = $state('llama');
  let modelConfig = $state('');
  let shardPaths = $state('');
  let seqLen = $state(2048);
  let batchSize = $state(4);
  let gradAccum = $state(1);
  let steps = $state(10000);
  let learningRate = $state(3e-4);
  let minLr = $state(1e-5);
  let warmupSteps = $state(1000);
  let lrSchedule = $state('cosine');
  let weightDecay = $state(0.1);
  let maxGradNorm = $state(1.0);
  let zLoss = $state(0.0);
  let eosTokenId = $state(0);
  let checkpointEvery = $state(1000);
  let outputDir = $state('./pretrain-output');
  let seed = $state(42);

  let isSubmitting = $state(false);
  let formError = $state<string | null>(null);
  let formSuccess = $state<string | null>(null);
  let selectedRunId = $state<string | null>(null);

  // Derived state
  let runs = $derived(pretrainStore.runs);
  let activeRuns = $derived(pretrainStore.activeRuns);
  let selectedRun = $derived(runs.find(r => r.id === selectedRunId) ?? null);

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    formSuccess = null;

    if (!arch) {
      formError = 'Please select an architecture';
      return;
    }

    isSubmitting = true;
    try {
      const runId = await pretrainStore.start({
        arch,
        model_config: modelConfig.trim() || undefined,
        shards: shardPaths.trim() || undefined,
        seq_len: seqLen,
        batch_size: batchSize,
        gradient_accumulation_steps: gradAccum,
        steps,
        learning_rate: learningRate,
        min_lr: minLr,
        warmup_steps: warmupSteps,
        lr_schedule: lrSchedule,
        weight_decay: weightDecay,
        max_grad_norm: maxGradNorm,
        z_loss: zLoss,
        eos_token_id: eosTokenId,
        checkpoint_every: checkpointEvery,
        output_dir: outputDir.trim() || undefined,
        seed,
      });
      formSuccess = `Pretraining started (run ID: ${runId})`;
      selectedRunId = runId;
    } catch (e) {
      formError = e instanceof Error ? e.message : String(e);
    } finally {
      isSubmitting = false;
    }
  }
</script>

<div class="space-y-6">
  <!-- Header -->
  <div>
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Pretrain</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Train a model from scratch on raw token shards using the pretraining pipeline</p>
  </div>

  <!-- Active runs -->
  {#if activeRuns.length > 0}
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      {#each activeRuns as run}
        <div class="card p-4">
          <div class="flex items-center justify-between mb-2">
            <div>
              <p class="font-semibold text-surface-900 dark:text-surface-100 text-sm">
                {run.arch} &mdash; step {run.metrics.step}{run.metrics.total_steps > 0 ? `/${run.metrics.total_steps}` : ''}
              </p>
              <p class="text-xs text-surface-500">
                {run.output_dir}
                {#if run.metrics.eta_seconds !== null}
                  &middot; ETA {formatEta(run.metrics.eta_seconds)}
                {/if}
              </p>
            </div>
            <span class={getStatusBadgeClass(run.status)}>{run.status}</span>
          </div>
          {#if run.metrics.total_steps > 0}
            <div class="progress-bar mb-2">
              <div class="progress-bar-fill" style="width: {runProgress(run.metrics.step, run.metrics.total_steps)}%"></div>
            </div>
          {/if}
          {#if run.metrics.loss !== null}
            <div class="flex gap-4 text-xs text-surface-500">
              <span>Loss: {run.metrics.loss.toFixed(4)}</span>
              {#if run.metrics.best_loss !== null}<span>Best: {run.metrics.best_loss.toFixed(4)}</span>{/if}
              {#if run.metrics.tokens_per_second !== null}<span>{run.metrics.tokens_per_second.toFixed(0)} tok/s</span>{/if}
            </div>
          {/if}
          <button
            class="btn-danger btn-sm w-full mt-3"
            aria-label="Stop pretrain run"
            onclick={() => pretrainStore.stop(run.id)}
          >
            Stop
          </button>
        </div>
      {/each}
    </div>
  {/if}

  <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
    <!-- Config Form -->
    <div class="xl:col-span-2">
      <form onsubmit={handleSubmit} class="space-y-4">

        <!-- Model -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Model</h3>
          </div>
          <div class="card-body space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label class="label" for="arch">Architecture</label>
                <select id="arch" class="input" bind:value={arch}>
                  {#each architectures as a}
                    <option value={a}>{a}</option>
                  {/each}
                </select>
              </div>
              <div>
                <label class="label" for="model-config">Model Config Path <span class="text-surface-400 font-normal">(optional)</span></label>
                <input
                  id="model-config"
                  type="text"
                  class="input"
                  placeholder="/path/to/config.json"
                  bind:value={modelConfig}
                />
              </div>
            </div>
            <div>
              <label class="label" for="shard-paths">Shard Paths</label>
              <input
                id="shard-paths"
                type="text"
                class="input"
                placeholder="/data/shards/*.bin or comma-separated paths"
                bind:value={shardPaths}
              />
              <p class="text-xs text-surface-500 mt-1">Glob pattern or comma-separated list of pre-tokenized shard files</p>
            </div>
            <div>
              <label class="label" for="output-dir">Output Directory</label>
              <input
                id="output-dir"
                type="text"
                class="input"
                placeholder="./pretrain-output"
                bind:value={outputDir}
              />
            </div>
          </div>
        </div>

        <!-- Training Hyperparameters -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Training Hyperparameters</h3>
          </div>
          <div class="card-body">
            <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <label class="label" for="steps">Training Steps</label>
                <input id="steps" type="number" class="input" min="1" bind:value={steps} />
              </div>
              <div>
                <label class="label" for="batch-size">Batch Size</label>
                <input id="batch-size" type="number" class="input" min="1" bind:value={batchSize} />
              </div>
              <div>
                <label class="label" for="grad-accum">Gradient Accumulation</label>
                <input id="grad-accum" type="number" class="input" min="1" bind:value={gradAccum} />
              </div>
              <div>
                <label class="label" for="seq-len">Sequence Length</label>
                <input id="seq-len" type="number" class="input" step="64" min="64" bind:value={seqLen} />
              </div>
              <div>
                <label class="label" for="seed">Seed</label>
                <input id="seed" type="number" class="input" min="0" bind:value={seed} />
              </div>
              <div>
                <label class="label" for="checkpoint-every">Checkpoint Every</label>
                <input id="checkpoint-every" type="number" class="input" min="1" bind:value={checkpointEvery} />
                <p class="text-xs text-surface-500 mt-1">Steps between checkpoints</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Learning Rate Schedule -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Learning Rate Schedule</h3>
          </div>
          <div class="card-body space-y-4">
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-2">
              {#each lrSchedules as sched}
                <button
                  type="button"
                  class="p-3 rounded-lg border text-left transition-all {lrSchedule === sched.value
                    ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30'
                    : 'border-surface-200 dark:border-surface-700 hover:border-surface-300 dark:hover:border-surface-600'}"
                  onclick={() => (lrSchedule = sched.value)}
                >
                  <p class="text-sm font-semibold text-surface-900 dark:text-surface-100">{sched.label}</p>
                </button>
              {/each}
            </div>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <label class="label" for="lr">Learning Rate</label>
                <input id="lr" type="number" class="input" step="0.00001" bind:value={learningRate} />
              </div>
              <div>
                <label class="label" for="min-lr">Min LR</label>
                <input id="min-lr" type="number" class="input" step="0.000001" bind:value={minLr} />
                <p class="text-xs text-surface-500 mt-1">Floor for cosine decay</p>
              </div>
              <div>
                <label class="label" for="warmup-steps">Warmup Steps</label>
                <input id="warmup-steps" type="number" class="input" min="0" bind:value={warmupSteps} />
              </div>
            </div>
          </div>
        </div>

        <!-- Regularization -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Regularization</h3>
          </div>
          <div class="card-body">
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <label class="label" for="weight-decay">Weight Decay</label>
                <input id="weight-decay" type="number" class="input" step="0.01" min="0" bind:value={weightDecay} />
              </div>
              <div>
                <label class="label" for="max-grad-norm">Max Grad Norm</label>
                <input id="max-grad-norm" type="number" class="input" step="0.1" min="0" bind:value={maxGradNorm} />
                <p class="text-xs text-surface-500 mt-1">0 = disabled</p>
              </div>
              <div>
                <label class="label" for="z-loss">Z-Loss Coefficient</label>
                <input id="z-loss" type="number" class="input" step="0.0001" min="0" bind:value={zLoss} />
                <p class="text-xs text-surface-500 mt-1">0 = disabled</p>
              </div>
              <div>
                <label class="label" for="eos-token">EOS Token ID</label>
                <input id="eos-token" type="number" class="input" min="0" bind:value={eosTokenId} />
                <p class="text-xs text-surface-500 mt-1">0 = none</p>
              </div>
            </div>
          </div>
        </div>

        {#if formError}
          <div
            class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm"
            role="alert"
          >
            {formError}
          </div>
        {/if}
        {#if formSuccess}
          <div
            class="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-sm"
            role="status"
          >
            {formSuccess}
          </div>
        {/if}

        <button type="submit" class="btn-primary w-full" disabled={isSubmitting}>
          {#if isSubmitting}
            <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
            Starting...
          {:else}
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Start Pretraining
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
        <div class="divide-y divide-surface-200 dark:divide-surface-700 max-h-[500px] overflow-y-auto scrollbar-thin">
          {#if runs.length === 0}
            <div class="p-6 text-center text-surface-500 dark:text-surface-400 text-sm">
              No pretrain runs yet
            </div>
          {:else}
            {#each runs as run}
              <button
                class="w-full text-left p-4 hover:bg-surface-50 dark:hover:bg-surface-700/50 transition-colors {selectedRunId === run.id ? 'bg-primary-50 dark:bg-primary-900/20' : ''}"
                onclick={() => (selectedRunId = selectedRunId === run.id ? null : run.id)}
              >
                <div class="flex items-center justify-between mb-1">
                  <span class="text-sm font-medium truncate max-w-[140px]">{run.arch}</span>
                  <span class={getStatusBadgeClass(run.status)}>{run.status}</span>
                </div>
                <div class="text-xs text-surface-500 mb-1 truncate">{run.output_dir}</div>
                <div class="text-xs text-surface-500 mb-2">
                  Step {run.metrics.step}{run.metrics.total_steps > 0 ? `/${run.metrics.total_steps}` : ''}
                </div>
                {#if run.metrics.total_steps > 0 && !['pending', 'completed'].includes(run.status)}
                  <div class="progress-bar">
                    <div class="progress-bar-fill" style="width: {runProgress(run.metrics.step, run.metrics.total_steps)}%"></div>
                  </div>
                {/if}
              </button>
            {/each}
          {/if}
        </div>

        {#if selectedRun}
          <div class="card-footer">
            <div class="space-y-2 text-sm">
              {#if selectedRun.metrics.loss !== null}
                <div class="flex justify-between">
                  <span class="text-surface-500">Loss</span>
                  <span class="font-mono">{selectedRun.metrics.loss.toFixed(4)}</span>
                </div>
              {/if}
              {#if selectedRun.metrics.best_loss !== null}
                <div class="flex justify-between">
                  <span class="text-surface-500">Best Loss</span>
                  <span class="font-mono">{selectedRun.metrics.best_loss.toFixed(4)}</span>
                </div>
              {/if}
              {#if selectedRun.metrics.tokens_per_second !== null}
                <div class="flex justify-between">
                  <span class="text-surface-500">Throughput</span>
                  <span>{selectedRun.metrics.tokens_per_second.toFixed(0)} tok/s</span>
                </div>
              {/if}
              {#if selectedRun.metrics.learning_rate !== null}
                <div class="flex justify-between">
                  <span class="text-surface-500">LR</span>
                  <span class="font-mono">{selectedRun.metrics.learning_rate.toExponential(2)}</span>
                </div>
              {/if}
              {#if selectedRun.metrics.eta_seconds !== null}
                <div class="flex justify-between">
                  <span class="text-surface-500">ETA</span>
                  <span>{formatEta(selectedRun.metrics.eta_seconds)}</span>
                </div>
              {/if}
              {#if selectedRun.error_message}
                <p class="text-xs text-red-500 dark:text-red-400 break-words">{selectedRun.error_message}</p>
              {/if}
              {#if selectedRun.status === 'running' || selectedRun.status === 'pending'}
                <button
                  class="btn-danger btn-sm w-full"
                  aria-label="Stop pretrain run"
                  onclick={() => pretrainStore.stop(selectedRun!.id)}
                >
                  Stop Pretraining
                </button>
              {/if}
            </div>

            <!-- Log tail -->
            {#if selectedRun.log_tail.length > 0}
              <div class="mt-3">
                <p class="text-xs font-medium text-surface-500 mb-1">Output</p>
                <div class="bg-surface-950 dark:bg-black rounded p-2 max-h-40 overflow-y-auto scrollbar-thin">
                  {#each selectedRun.log_tail.slice(-20) as line}
                    <p class="text-xs font-mono text-surface-300 leading-relaxed whitespace-pre-wrap break-all">{line}</p>
                  {/each}
                </div>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>
