<script lang="ts">
  import { modelsStore, benchStore } from '$lib/stores.svelte';

  const presets = [
    'dense-qwen3',
    'hybrid-qwen3next',
    'hybrid-qwen35-steady',
    'moe-nemotronh',
    'custom',
  ];
  const contexts = ['auto', 'prompt', 'text-prefix'];

  // Form state
  let mode = $state('workload');
  let selectedModel = $state('');
  let preset = $state('dense-qwen3');
  let inferenceContext = $state('auto');
  let promptSamples = $state(8);
  let maxPromptTokens = $state(0);
  let decodeSteps = $state(32);
  let inferenceWarmup = $state(2);
  let inferenceRepeats = $state(1);
  let batchSize = $state(1);
  let seqLen = $state(512);
  let jsonOutput = $state('');

  let formError = $state<string | null>(null);
  let isStarting = $state(false);

  let models = $derived(modelsStore.models);
  let activeRun = $derived(benchStore.activeRuns[0] ?? null);
  let lastCompleted = $derived(
    benchStore.runs
      .filter(r => r.status === 'completed' || r.status === 'failed')
      .slice(-1)[0] ?? null,
  );
  let display = $derived(activeRun ?? lastCompleted);

  let avgPromptTps = $derived(
    display && display.trials.length > 0
      ? display.trials.reduce((s, t) => s + t.prompt_tps, 0) / display.trials.length
      : null,
  );
  let avgDecodeTps = $derived(
    display && display.trials.length > 0
      ? display.trials.reduce((s, t) => s + t.generation_tps, 0) / display.trials.length
      : null,
  );
  let peakMemGb = $derived(
    display && display.trials.length > 0
      ? Math.max(...display.trials.map(t => t.peak_memory_gb))
      : null,
  );

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    if (!selectedModel) {
      formError = 'Please select a model';
      return;
    }
    if (jsonOutput && !jsonOutput.endsWith('.json')) {
      formError = 'JSON Output, if set, must end in .json';
      return;
    }

    isStarting = true;
    try {
      await benchStore.start({
        model: selectedModel,
        batch_size: batchSize,
        seq_len: seqLen,
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
      await benchStore.stop(activeRun.id);
    } catch (e) {
      formError = e instanceof Error ? e.message : String(e);
    }
  }
</script>

<div class="space-y-6 max-w-5xl">
  <!-- Header -->
  <div class="flex items-start justify-between">
    <div>
      <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Benchmark</h1>
      <p class="text-surface-500 dark:text-surface-400 mt-1">
        Measure prompt and decode throughput on a cached model
      </p>
    </div>
    {#if activeRun}
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800">
        <span class="w-2 h-2 rounded-full bg-primary-500 animate-pulse"></span>
        <span class="text-sm font-medium text-primary-700 dark:text-primary-300">Running…</span>
      </div>
    {/if}
  </div>

  <div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
    <!-- Config form -->
    <form onsubmit={handleSubmit} class="space-y-4">
      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Configuration</h3>
        </div>
        <div class="card-body space-y-4">
          <!-- Mode toggle -->
          <div>
            <label class="label" for="bench-mode">Mode</label>
            <div class="grid grid-cols-2 gap-2">
              <button
                type="button"
                class="p-3 rounded-lg border text-center text-sm transition-all {mode === 'workload'
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30 font-medium'
                  : 'border-surface-200 dark:border-surface-700 hover:border-surface-300 dark:hover:border-surface-600'}"
                onclick={() => (mode = 'workload')}
              >
                Workload
                <p class="text-xs text-surface-500 mt-0.5">preset-driven, real dataset</p>
              </button>
              <button
                type="button"
                class="p-3 rounded-lg border text-center text-sm transition-all {mode === 'basic'
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30 font-medium'
                  : 'border-surface-200 dark:border-surface-700 hover:border-surface-300 dark:hover:border-surface-600'}"
                onclick={() => (mode = 'basic')}
              >
                Basic
                <p class="text-xs text-surface-500 mt-0.5">synthetic batch × seq_len</p>
              </button>
            </div>
          </div>

          <!-- Model -->
          <div>
            <label class="label" for="bench-model">Model</label>
            <select id="bench-model" class="input" bind:value={selectedModel}>
              <option value="">Select a model to benchmark…</option>
              {#each models as model}
                <option value={model.id}>{model.id} ({model.size_formatted})</option>
              {/each}
            </select>
          </div>

          {#if mode === 'workload'}
            <!-- Workload knobs -->
            <div>
              <label class="label" for="bench-preset">Preset</label>
              <select id="bench-preset" class="input" bind:value={preset}>
                {#each presets as p}
                  <option value={p}>{p}</option>
                {/each}
              </select>
            </div>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="label" for="bench-ctx">Inference Context</label>
                <select id="bench-ctx" class="input" bind:value={inferenceContext}>
                  {#each contexts as c}
                    <option value={c}>{c}</option>
                  {/each}
                </select>
              </div>
              <div>
                <label class="label" for="bench-samples">Prompt Samples</label>
                <input id="bench-samples" type="number" min="1" max="128" class="input" bind:value={promptSamples} />
              </div>
              <div>
                <label class="label" for="bench-max-prompt">Max Prompt Tokens</label>
                <input id="bench-max-prompt" type="number" min="0" max="16384" class="input" bind:value={maxPromptTokens} />
              </div>
              <div>
                <label class="label" for="bench-decode">Decode Steps</label>
                <input id="bench-decode" type="number" min="1" max="4096" class="input" bind:value={decodeSteps} />
              </div>
              <div>
                <label class="label" for="bench-warmup">Warmup Passes</label>
                <input id="bench-warmup" type="number" min="0" max="32" class="input" bind:value={inferenceWarmup} />
              </div>
              <div>
                <label class="label" for="bench-repeats">Inference Repeats</label>
                <input id="bench-repeats" type="number" min="1" max="64" class="input" bind:value={inferenceRepeats} />
              </div>
            </div>
          {:else}
            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="label" for="bench-batch">Batch Size</label>
                <input id="bench-batch" type="number" min="1" max="128" class="input" bind:value={batchSize} />
              </div>
              <div>
                <label class="label" for="bench-seq">Seq Len</label>
                <input id="bench-seq" type="number" min="32" max="32768" class="input" bind:value={seqLen} />
              </div>
            </div>
          {/if}

          <div>
            <label class="label" for="bench-json">JSON Output Path (optional)</label>
            <input id="bench-json" type="text" class="input" placeholder="/path/to/report.json" bind:value={jsonOutput} />
          </div>
        </div>
      </div>

      {#if formError}
        <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
          {formError}
        </div>
      {/if}

      <div class="flex gap-2">
        <button type="submit" class="btn-primary flex-1" disabled={isStarting || activeRun !== null || !selectedModel}>
          {#if isStarting}
            <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
            Starting…
          {:else}
            Run Benchmark
          {/if}
        </button>
        {#if activeRun}
          <button type="button" class="btn-danger" onclick={handleStop}>Stop</button>
        {/if}
      </div>
    </form>

    <!-- Trials + log -->
    <div class="space-y-4">
      <div class="card">
        <div class="card-header flex items-center justify-between">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Trials</h3>
          {#if display}
            <span class="text-xs text-surface-500">{display.mode} · {display.status}</span>
          {/if}
        </div>
        <div class="card-body">
          {#if !display || display.trials.length === 0}
            <p class="text-sm text-surface-500 italic">
              {display ? 'Waiting for first trial…' : 'Run a benchmark to see results.'}
            </p>
          {:else}
            <table class="w-full text-sm">
              <thead>
                <tr class="text-xs uppercase text-surface-500 border-b border-surface-200 dark:border-surface-700">
                  <th class="text-left py-2">Trial</th>
                  <th class="text-right py-2">Prompt tok/s</th>
                  <th class="text-right py-2">Decode tok/s</th>
                  <th class="text-right py-2">Peak GB</th>
                </tr>
              </thead>
              <tbody>
                {#each display.trials as trial}
                  <tr class="border-b border-surface-100 dark:border-surface-800">
                    <td class="py-2 font-mono">{trial.index}</td>
                    <td class="py-2 text-right font-mono">{trial.prompt_tps.toFixed(1)}</td>
                    <td class="py-2 text-right font-mono">{trial.generation_tps.toFixed(1)}</td>
                    <td class="py-2 text-right font-mono">{trial.peak_memory_gb.toFixed(2)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
            {#if avgPromptTps !== null && avgDecodeTps !== null}
              <div class="mt-3 pt-3 border-t border-surface-200 dark:border-surface-700 text-xs text-surface-600 dark:text-surface-400">
                Avg <strong class="font-mono text-surface-900 dark:text-surface-100">prompt {avgPromptTps.toFixed(1)}</strong>
                · <strong class="font-mono text-surface-900 dark:text-surface-100">decode {avgDecodeTps.toFixed(1)}</strong>
                {#if peakMemGb !== null}
                  · peak <strong class="font-mono text-surface-900 dark:text-surface-100">{peakMemGb.toFixed(2)} GB</strong>
                {/if}
              </div>
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
            <pre class="text-xs font-mono text-surface-700 dark:text-surface-300 max-h-64 overflow-y-auto whitespace-pre-wrap">{display.log_tail.join('\n')}</pre>
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>
