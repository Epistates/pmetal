<script lang="ts">
  import { modelsStore } from '#lib/stores.svelte.js';
  import { startEmbedTrain, type EmbedTrainConfig } from '#lib/api.js';

  const lossOptions = [
    { value: 'info_nce', label: 'InfoNCE', description: 'Contrastive loss with in-batch negatives — best general-purpose choice' },
    { value: 'mnrl', label: 'MNRL', description: 'Multiple negatives ranking loss — good for retrieval tasks' },
    { value: 'triplet', label: 'Triplet', description: 'Margin-based triplet loss — requires margin tuning' },
    { value: 'cosent', label: 'CoSENT', description: 'Cosine sentence similarity loss — good for semantic textual similarity' },
    { value: 'cosine_similarity', label: 'Cosine Similarity', description: 'Direct cosine similarity regression' },
  ];

  const poolingOptions = [
    { value: 'mean', label: 'Mean Pooling' },
    { value: 'cls', label: 'CLS Token' },
    { value: 'max', label: 'Max Pooling' },
    { value: 'last_token', label: 'Last Token' },
    { value: 'weighted_mean', label: 'Weighted Mean' },
  ];

  // Form state
  let model = $state('');
  let dataset = $state('');
  let outputDir = $state('./output-embed');
  let loss = $state('info_nce');
  let pooling = $state('mean');
  let temperature = $state(0.05);
  let margin = $state(0.3);
  let learningRate = $state(2e-5);
  let batchSize = $state(32);
  let epochs = $state(3);
  let maxSeqLen = $state(512);
  let weightDecay = $state(0.01);
  let noNormalize = $state(false);
  let logEvery = $state(10);
  let seed = $state(42);

  // UI state
  let isRunning = $state(false);
  let runId = $state<string | null>(null);
  let status = $state<'idle' | 'running' | 'done' | 'failed'>('idle');
  let formError = $state<string | null>(null);
  let logs = $state<string[]>([]);

  let models = $derived(modelsStore.models);

  function appendLog(line: string) {
    logs = [...logs.slice(-499), line];
  }

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    logs = [];
    status = 'idle';

    if (!model) { formError = 'Please select a model'; return; }
    if (!dataset.trim()) { formError = 'Please enter a dataset'; return; }

    isRunning = true;
    status = 'running';

    const config: EmbedTrainConfig = {
      model,
      dataset: dataset.trim(),
      output_dir: outputDir.trim() || null,
      loss,
      pooling,
      temperature,
      margin,
      learning_rate: learningRate,
      batch_size: batchSize,
      epochs,
      max_seq_len: maxSeqLen,
      weight_decay: weightDecay,
      no_normalize: noNormalize,
      log_every: logEvery,
      seed,
    };

    try {
      runId = await startEmbedTrain(config, (e: Record<string, unknown>) => {
        const evt = e as { event?: string; line?: string };
        if (evt.event === 'log' && typeof evt.line === 'string') {
          appendLog(evt.line);
        } else if (evt.event === 'done') {
          status = 'done';
          isRunning = false;
        } else if (evt.event === 'failed') {
          status = 'failed';
          isRunning = false;
        }
      });
    } catch (err) {
      formError = err instanceof Error ? err.message : String(err);
      status = 'failed';
      isRunning = false;
    }
  }
</script>

<div class="space-y-6 max-w-3xl">
  <!-- Header -->
  <div>
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Embed Train</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Train sentence embedding models with contrastive losses</p>
  </div>

  <form onsubmit={handleSubmit} class="space-y-4">
    <!-- Model + Dataset -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Model &amp; Data</h3>
      </div>
      <div class="card-body space-y-4">
        <div>
          <label class="label" for="et-model">Base Model</label>
          <select id="et-model" class="input" bind:value={model}>
            <option value="">Select a model...</option>
            {#each models as m}
              <option value={m.id}>{m.id} ({m.size_formatted})</option>
            {/each}
          </select>
        </div>
        <div>
          <label class="label" for="et-dataset">Dataset (HF ID or local path)</label>
          <input id="et-dataset" type="text" class="input" placeholder="e.g. sentence-transformers/all-nli" bind:value={dataset} />
        </div>
        <div>
          <label class="label" for="et-output">Output Directory</label>
          <input id="et-output" type="text" class="input" bind:value={outputDir} />
        </div>
      </div>
    </div>

    <!-- Loss -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Loss Function</h3>
      </div>
      <div class="card-body space-y-3">
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {#each lossOptions as opt}
            <button
              type="button"
              class="p-3 rounded-lg border text-left transition-all {loss === opt.value
                ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30'
                : 'border-surface-200 dark:border-surface-700 hover:border-surface-300 dark:hover:border-surface-600'}"
              onclick={() => (loss = opt.value)}
            >
              <p class="text-sm font-semibold text-surface-900 dark:text-surface-100">{opt.label}</p>
              <p class="text-xs text-surface-500 mt-0.5">{opt.description}</p>
            </button>
          {/each}
        </div>
      </div>
    </div>

    <!-- Hyperparameters -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Hyperparameters</h3>
      </div>
      <div class="card-body grid grid-cols-2 gap-4">
        <div>
          <label class="label" for="et-pooling">Pooling</label>
          <select id="et-pooling" class="input" bind:value={pooling}>
            {#each poolingOptions as p}
              <option value={p.value}>{p.label}</option>
            {/each}
          </select>
        </div>
        <div>
          <label class="label" for="et-lr">Learning Rate</label>
          <input id="et-lr" type="number" class="input" step="1e-6" min="1e-8" max="1" bind:value={learningRate} />
        </div>
        <div>
          <label class="label" for="et-bsz">Batch Size</label>
          <input id="et-bsz" type="number" class="input" min="1" max="4096" bind:value={batchSize} />
        </div>
        <div>
          <label class="label" for="et-epochs">Epochs</label>
          <input id="et-epochs" type="number" class="input" min="1" max="1000" bind:value={epochs} />
        </div>
        <div>
          <label class="label" for="et-temp">Temperature</label>
          <input id="et-temp" type="number" class="input" step="0.001" min="0.001" max="10" bind:value={temperature} />
        </div>
        <div>
          <label class="label" for="et-margin">Margin</label>
          <input id="et-margin" type="number" class="input" step="0.01" min="0" max="10" bind:value={margin} />
        </div>
        <div>
          <label class="label" for="et-seqlen">Max Seq Len</label>
          <input id="et-seqlen" type="number" class="input" min="32" max="8192" bind:value={maxSeqLen} />
        </div>
        <div>
          <label class="label" for="et-wd">Weight Decay</label>
          <input id="et-wd" type="number" class="input" step="0.001" min="0" max="1" bind:value={weightDecay} />
        </div>
        <div>
          <label class="label" for="et-seed">Seed</label>
          <input id="et-seed" type="number" class="input" min="0" bind:value={seed} />
        </div>
        <div>
          <label class="label" for="et-logevery">Log Every (steps)</label>
          <input id="et-logevery" type="number" class="input" min="1" bind:value={logEvery} />
        </div>
        <div class="col-span-2">
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" class="rounded border-surface-300" bind:checked={noNormalize} />
            <span class="text-sm font-medium text-surface-700 dark:text-surface-300">Disable L2 normalization of embeddings</span>
          </label>
        </div>
      </div>
    </div>

    <!-- Status -->
    {#if status === 'running'}
      <div class="p-4 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 text-primary-700 dark:text-primary-300 text-sm flex items-center gap-2" role="status">
        <div class="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin flex-shrink-0" aria-hidden="true"></div>
        Training in progress… Run ID: {runId}
      </div>
    {/if}
    {#if status === 'done'}
      <div class="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-sm" role="status">
        Training complete. Adapter saved to {outputDir}
      </div>
    {/if}
    {#if status === 'failed'}
      <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
        Training failed. Check the log below for details.
      </div>
    {/if}
    {#if formError}
      <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
        {formError}
      </div>
    {/if}

    <button type="submit" class="btn-primary w-full" disabled={isRunning || !model || !dataset.trim()}>
      {#if isRunning}
        <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
        Training...
      {:else}
        Start Embed Train
      {/if}
    </button>
  </form>

  <!-- Log -->
  {#if logs.length > 0}
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Output Log</h3>
      </div>
      <div class="card-body">
        <pre class="text-xs font-mono text-surface-700 dark:text-surface-300 bg-surface-50 dark:bg-surface-900 rounded p-3 max-h-64 overflow-y-auto whitespace-pre-wrap">{logs.join('\n')}</pre>
      </div>
    </div>
  {/if}
</div>
