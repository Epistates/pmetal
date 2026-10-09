<script lang="ts">
  import { modelsStore } from '#lib/stores.svelte.js';
  import { exportModelfile } from '#lib/api.js';

  const templates = ['auto', 'llama3', 'qwen3', 'gemma', 'mistral', 'phi3', 'deep-seek'];

  let base = $state('');
  let lora = $state('');
  let output = $state('Modelfile');
  let template = $state('auto');
  let systemPrompt = $state('');
  let temperature = $state('');
  let numCtx = $state('');
  let topK = $state('');
  let topP = $state('');
  let license = $state('');

  let isRunning = $state(false);
  let runId = $state<string | null>(null);
  let status = $state<'idle' | 'running' | 'done' | 'failed'>('idle');
  let formError = $state<string | null>(null);
  let logs = $state<string[]>([]);

  let models = $derived(modelsStore.models);

  function appendLog(line: string) {
    logs = [...logs.slice(-499), line];
  }

  function num(value: string): number | null {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const parsed = Number(trimmed);
    return Number.isFinite(parsed) ? parsed : null;
  }

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    logs = [];
    status = 'idle';

    if (!base.trim()) {
      formError = 'Please select or enter a base model';
      return;
    }
    if (!output.trim()) {
      formError = 'Please specify an output path';
      return;
    }

    isRunning = true;
    status = 'running';

    try {
      runId = await exportModelfile({
        base: base.trim(),
        lora: lora.trim() || null,
        output: output.trim(),
        template,
        system: systemPrompt.trim() || null,
        temperature: num(temperature),
        num_ctx: num(numCtx),
        top_k: num(topK),
        top_p: num(topP),
        license: license.trim() || null,
      }, (e: Record<string, unknown>) => {
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
  <div>
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Modelfile Export</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Generate a portable Modelfile for external registration</p>
  </div>

  <form onsubmit={handleSubmit} class="space-y-4">
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Model</h3>
      </div>
      <div class="card-body space-y-4">
        <div>
          <label class="label" for="export-base">Base Model</label>
          <select id="export-base" class="input" bind:value={base}>
            <option value="">Select a cached model...</option>
            {#each models as model}
              <option value={model.id}>{model.id} ({model.size_formatted})</option>
            {/each}
          </select>
          <input class="input mt-2" placeholder="Or enter a GGUF path / model name" bind:value={base} />
        </div>
        <div>
          <label class="label" for="export-lora">LoRA Adapter (optional)</label>
          <input id="export-lora" class="input" placeholder="/path/to/adapter" bind:value={lora} />
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Modelfile</h3>
      </div>
      <div class="card-body grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label class="label" for="export-output">Output Path</label>
          <input id="export-output" class="input" bind:value={output} />
        </div>
        <div>
          <label class="label" for="export-template">Template</label>
          <select id="export-template" class="input" bind:value={template}>
            {#each templates as t}
              <option value={t}>{t}</option>
            {/each}
          </select>
        </div>
        <div>
          <label class="label" for="export-temp">Temperature</label>
          <input id="export-temp" class="input" placeholder="0.7" bind:value={temperature} />
        </div>
        <div>
          <label class="label" for="export-ctx">Context Window</label>
          <input id="export-ctx" class="input" placeholder="4096" bind:value={numCtx} />
        </div>
        <div>
          <label class="label" for="export-top-k">Top K</label>
          <input id="export-top-k" class="input" placeholder="40" bind:value={topK} />
        </div>
        <div>
          <label class="label" for="export-top-p">Top P</label>
          <input id="export-top-p" class="input" placeholder="0.9" bind:value={topP} />
        </div>
        <div class="sm:col-span-2">
          <label class="label" for="export-system">System Prompt</label>
          <textarea id="export-system" class="input min-h-24" bind:value={systemPrompt}></textarea>
        </div>
        <div class="sm:col-span-2">
          <label class="label" for="export-license">License Text</label>
          <textarea id="export-license" class="input min-h-20" bind:value={license}></textarea>
        </div>
      </div>
    </div>

    {#if status === 'running'}
      <div class="p-4 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 text-primary-700 dark:text-primary-300 text-sm flex items-center gap-2" role="status">
        <div class="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin flex-shrink-0" aria-hidden="true"></div>
        Exporting… Run ID: {runId}
      </div>
    {/if}
    {#if status === 'done'}
      <div class="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-sm" role="status">
        Modelfile generated successfully.
      </div>
    {/if}
    {#if status === 'failed'}
      <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
        Export failed. Check the output log below.
      </div>
    {/if}
    {#if formError}
      <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
        {formError}
      </div>
    {/if}

    <button type="submit" class="btn-primary w-full" disabled={isRunning || !base.trim() || !output.trim()}>
      {#if isRunning}
        <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
        Exporting...
      {:else}
        Generate Modelfile
      {/if}
    </button>
  </form>

  {#if logs.length > 0}
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Output</h3>
      </div>
      <div class="card-body">
        <pre class="text-xs font-mono text-surface-700 dark:text-surface-300 bg-surface-50 dark:bg-surface-900 rounded p-3 max-h-80 overflow-y-auto whitespace-pre-wrap">{logs.join('\n')}</pre>
      </div>
    </div>
  {/if}
</div>
