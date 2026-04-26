<script lang="ts">
  import { modelsStore, evalStore } from '$lib/stores.svelte';

  // Form state
  let selectedModel = $state('');
  let lora = $state('');
  let dataset = $state('');
  let maxSeqLen = $state(1024);
  let numSamples = $state(0);
  let jsonReport = $state(false);

  let formError = $state<string | null>(null);
  let isStarting = $state(false);

  let models = $derived(modelsStore.models);
  let activeRun = $derived(evalStore.activeRuns[0] ?? null);
  let lastCompleted = $derived(
    evalStore.runs
      .filter(r => r.status === 'completed' || r.status === 'failed')
      .slice(-1)[0] ?? null,
  );
  let display = $derived(activeRun ?? lastCompleted);

  let progressPct = $derived(
    display && display.metrics.samples_total > 0
      ? (display.metrics.samples_done / display.metrics.samples_total) * 100
      : 0,
  );

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    if (!selectedModel) {
      formError = 'Please select a model';
      return;
    }
    if (!dataset.trim()) {
      formError = 'Please specify a dataset path or HF id';
      return;
    }

    isStarting = true;
    try {
      await evalStore.start({
        model: selectedModel,
        dataset: dataset.trim(),
        lora: lora.trim() || undefined,
        max_seq_len: maxSeqLen,
        num_samples: numSamples,
        json: jsonReport,
      });
    } catch (e) {
      formError = e instanceof Error ? e.message : String(e);
    } finally {
      isStarting = false;
    }
  }

  async function handleStop() {
    if (!activeRun) return;
    try {
      await evalStore.stop(activeRun.id);
    } catch (e) {
      formError = e instanceof Error ? e.message : String(e);
    }
  }
</script>

<div class="space-y-6 max-w-4xl">
  <!-- Header -->
  <div class="flex items-start justify-between">
    <div>
      <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Evaluate</h1>
      <p class="text-surface-500 dark:text-surface-400 mt-1">
        Measure perplexity, accuracy, and loss on a held-out dataset
      </p>
    </div>
    {#if activeRun}
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800">
        <span class="w-2 h-2 rounded-full bg-primary-500 animate-pulse"></span>
        <span class="text-sm font-medium text-primary-700 dark:text-primary-300">Running…</span>
      </div>
    {/if}
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
    <!-- Config form -->
    <form onsubmit={handleSubmit} class="lg:col-span-2 space-y-4">
      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Configuration</h3>
        </div>
        <div class="card-body space-y-4">
          <div>
            <label class="label" for="eval-model">Model</label>
            <select id="eval-model" class="input" bind:value={selectedModel}>
              <option value="">Select a model to evaluate…</option>
              {#each models as model}
                <option value={model.id}>{model.id} ({model.size_formatted})</option>
              {/each}
            </select>
          </div>

          <div>
            <label class="label" for="eval-lora">LoRA Adapter (optional)</label>
            <input id="eval-lora" type="text" class="input" placeholder="/path/to/lora/adapter" bind:value={lora} />
          </div>

          <div>
            <label class="label" for="eval-dataset">Dataset</label>
            <input id="eval-dataset" type="text" class="input" placeholder="/path/to/eval.jsonl or owner/dataset" bind:value={dataset} />
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="label" for="eval-max-seq">Max Sequence Length</label>
              <input id="eval-max-seq" type="number" min="64" max="131072" step="64" class="input" bind:value={maxSeqLen} />
            </div>
            <div>
              <label class="label" for="eval-samples">Num Samples</label>
              <input id="eval-samples" type="number" min="0" max="1000000" class="input" bind:value={numSamples} />
              <p class="text-xs text-surface-500 mt-1">0 = evaluate the entire dataset.</p>
            </div>
          </div>

          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" class="rounded border-surface-300" bind:checked={jsonReport} />
            <span class="text-sm font-medium text-surface-700 dark:text-surface-300">
              Persist a JSON report next to the dataset
            </span>
          </label>
        </div>
      </div>

      {#if formError}
        <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
          {formError}
        </div>
      {/if}

      <div class="flex gap-2">
        <button type="submit" class="btn-primary flex-1" disabled={isStarting || activeRun !== null || !selectedModel || !dataset.trim()}>
          {#if isStarting}
            <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
            Starting…
          {:else}
            Run Evaluation
          {/if}
        </button>
        {#if activeRun}
          <button type="button" class="btn-danger" onclick={handleStop}>Stop</button>
        {/if}
      </div>
    </form>

    <!-- Metrics -->
    <div class="space-y-4">
      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Metrics</h3>
        </div>
        <div class="card-body space-y-3">
          {#if !display}
            <p class="text-sm text-surface-500 italic">Run an eval to see results.</p>
          {:else}
            <div class="flex items-center justify-between text-sm">
              <span class="text-surface-500">Status</span>
              <span class="font-mono text-surface-900 dark:text-surface-100">{display.status}</span>
            </div>
            {#if display.metrics.samples_total > 0}
              <div>
                <div class="flex items-center justify-between text-sm mb-1">
                  <span class="text-surface-500">Samples</span>
                  <span class="font-mono text-surface-900 dark:text-surface-100">
                    {display.metrics.samples_done}/{display.metrics.samples_total}
                  </span>
                </div>
                <div class="progress-bar">
                  <div class="progress-bar-fill" style="width: {progressPct}%"></div>
                </div>
              </div>
            {/if}
            {#if display.metrics.perplexity !== null}
              <div class="flex items-center justify-between text-sm">
                <span class="text-surface-500">Perplexity</span>
                <span class="font-mono font-semibold text-surface-900 dark:text-surface-100">{display.metrics.perplexity.toFixed(3)}</span>
              </div>
            {/if}
            {#if display.metrics.accuracy !== null}
              <div class="flex items-center justify-between text-sm">
                <span class="text-surface-500">Accuracy</span>
                <span class="font-mono font-semibold text-surface-900 dark:text-surface-100">{(display.metrics.accuracy * 100).toFixed(2)}%</span>
              </div>
            {/if}
            {#if display.metrics.loss !== null}
              <div class="flex items-center justify-between text-sm">
                <span class="text-surface-500">Loss</span>
                <span class="font-mono font-semibold text-surface-900 dark:text-surface-100">{display.metrics.loss.toFixed(4)}</span>
              </div>
            {/if}
            {#if display.error_message}
              <div class="text-xs text-red-600 dark:text-red-400">{display.error_message}</div>
            {/if}
          {/if}
        </div>
      </div>

      {#if display && display.log_tail.length > 0}
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Log</h3>
          </div>
          <div class="card-body">
            <pre class="text-xs font-mono text-surface-700 dark:text-surface-300 max-h-48 overflow-y-auto whitespace-pre-wrap">{display.log_tail.join('\n')}</pre>
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>
