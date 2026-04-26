<script lang="ts">
  import { modelsStore, serveStore } from '$lib/stores.svelte';

  const kvPresets = [
    { value: 'auto', label: 'Auto' },
    { value: 'fp16', label: 'FP16' },
    { value: 'q8', label: 'Q8_0' },
    { value: 'q4', label: 'Q4_0' },
    { value: 'tq8', label: 'TurboQuant 8' },
    { value: 'tq4', label: 'TurboQuant 4' },
    { value: 'tq2_5', label: 'TurboQuant 2.5' },
    { value: 'tq3_5', label: 'TurboQuant 3.5' },
  ];

  // Form state
  let selectedModel = $state('');
  let host = $state('0.0.0.0');
  let port = $state(8080);
  let maxSeqLen = $state(4096);
  let fp8 = $state(false);
  let kvCache = $state('auto');
  let kvGroupSize = $state(64);
  let loraAdapter = $state('');
  let expertsDir = $state('');

  // UI state
  let formError = $state<string | null>(null);
  let formSuccess = $state<string | null>(null);
  let isStarting = $state(false);

  let models = $derived(modelsStore.models);
  let activeInstance = $derived(serveStore.running[0] ?? null);
  let isRunning = $derived(activeInstance !== null);

  function bindUrlPreview(): string {
    const display = host === '0.0.0.0' ? 'localhost' : host;
    return `http://${display}:${port}`;
  }

  function formatUptime(startedAt: string | null): string {
    if (!startedAt) return '--';
    const delta = Math.max(0, Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000));
    if (delta < 60) return `${delta}s`;
    if (delta < 3600) return `${Math.floor(delta / 60)}m ${delta % 60}s`;
    return `${Math.floor(delta / 3600)}h ${Math.floor((delta % 3600) / 60)}m`;
  }

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    formSuccess = null;

    if (!selectedModel) {
      formError = 'Please select a model';
      return;
    }
    if (!host.trim()) {
      formError = 'Host is required (e.g. 0.0.0.0 or 127.0.0.1)';
      return;
    }
    if (port < 1 || port > 65535) {
      formError = 'Port must be between 1 and 65535';
      return;
    }

    isStarting = true;
    try {
      const spec: import('$lib/api').ServeSpec = {
        model: selectedModel,
        host,
        port,
        max_seq_len: maxSeqLen,
        fp8,
        kv_quant:
          kvCache === 'q8' ? 8 :
          kvCache === 'q4' ? 4 :
          kvCache === 'fp16' ? 0 :
          undefined,
        no_kv_quant: kvCache === 'fp16' ? true : undefined,
        kv_turboquant: kvCache === 'tq8' || kvCache === 'tq4' ? true : undefined,
        kv_turboquant_preset:
          kvCache === 'tq2_5' ? 'q2_5' :
          kvCache === 'tq3_5' ? 'q3_5' :
          undefined,
        kv_group_size: kvGroupSize,
        lora: loraAdapter.trim() || undefined,
        experts_dir: expertsDir.trim() || undefined,
      };
      await serveStore.start(spec);
      formSuccess = `Server starting on ${bindUrlPreview()}`;
    } catch (e) {
      formError = e instanceof Error ? e.message : String(e);
    } finally {
      isStarting = false;
    }
  }

  async function handleStop() {
    if (!activeInstance) return;
    try {
      await serveStore.stop(activeInstance.id);
      formSuccess = 'Server stopped';
    } catch (e) {
      formError = e instanceof Error ? e.message : String(e);
    }
  }

  function copyCurl(url: string) {
    const snippet =
      `curl ${url}/v1/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'`;
    void navigator.clipboard?.writeText(snippet);
  }
</script>

<div class="space-y-6 max-w-4xl">
  <!-- Header -->
  <div class="flex items-start justify-between">
    <div>
      <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Serve</h1>
      <p class="text-surface-500 dark:text-surface-400 mt-1">
        Start an OpenAI-compatible HTTP inference server backed by PMetal
      </p>
    </div>
    {#if activeInstance}
      <div class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800">
        <span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
        <span class="text-sm font-medium text-green-700 dark:text-green-300">
          {activeInstance.status === 'running' ? 'Serving' : 'Starting…'}
        </span>
      </div>
    {/if}
  </div>

  {#if activeInstance}
    <!-- LIVE STATUS PANEL -->
    <div class="card">
      <div class="card-header flex items-center justify-between">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Active Server</h3>
        <button type="button" class="btn-danger btn-sm" onclick={handleStop}>
          Stop Server
        </button>
      </div>
      <div class="card-body space-y-4">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <div class="text-xs text-surface-500 uppercase tracking-wide">Model</div>
            <div class="text-sm font-mono text-surface-900 dark:text-surface-100 truncate" title={activeInstance.model}>
              {activeInstance.model}
            </div>
          </div>
          <div>
            <div class="text-xs text-surface-500 uppercase tracking-wide">Bind</div>
            <div class="text-sm font-mono text-surface-900 dark:text-surface-100">
              {activeInstance.bind_url}
            </div>
          </div>
          <div>
            <div class="text-xs text-surface-500 uppercase tracking-wide">Uptime</div>
            <div class="text-sm font-mono text-surface-900 dark:text-surface-100">
              {formatUptime(activeInstance.ready_at ?? activeInstance.started_at)}
            </div>
          </div>
          <div>
            <div class="text-xs text-surface-500 uppercase tracking-wide">Max Seq Len</div>
            <div class="text-sm font-mono text-surface-900 dark:text-surface-100">
              {activeInstance.max_seq_len}
            </div>
          </div>
        </div>

        {#if activeInstance.status_message}
          <div class="text-sm text-surface-600 dark:text-surface-400 italic">
            {activeInstance.status_message}
          </div>
        {/if}

        <!-- Quick curl example -->
        <div class="p-3 rounded-lg bg-surface-50 dark:bg-surface-700/50 border border-surface-200 dark:border-surface-700">
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs font-medium text-surface-600 dark:text-surface-400">Test with curl</span>
            <button
              type="button"
              class="text-xs text-primary-600 dark:text-primary-400 hover:underline"
              onclick={() => copyCurl(activeInstance.bind_url)}
            >
              Copy
            </button>
          </div>
          <pre class="text-xs font-mono text-surface-700 dark:text-surface-300 whitespace-pre-wrap">{`curl ${activeInstance.bind_url}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"default","messages":[{"role":"user","content":"Hello"}]}'`}</pre>
        </div>
      </div>
    </div>

    <!-- LOG PANEL -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Server Log</h3>
      </div>
      <div class="card-body">
        {#if activeInstance.log_tail.length === 0}
          <div class="text-sm text-surface-500 dark:text-surface-400 italic">Waiting for output…</div>
        {:else}
          <pre class="text-xs font-mono text-surface-700 dark:text-surface-300 max-h-80 overflow-y-auto whitespace-pre-wrap">{activeInstance.log_tail.join('\n')}</pre>
        {/if}
      </div>
    </div>
  {:else}
    <!-- CONFIG FORM -->
    <form onsubmit={handleSubmit} class="space-y-4">
      <!-- Model -->
      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Model</h3>
        </div>
        <div class="card-body space-y-4">
          <div>
            <label class="label" for="serve-model">Cached Model</label>
            <select id="serve-model" class="input" bind:value={selectedModel}>
              <option value="">Select a model to serve…</option>
              {#each models as model}
                <option value={model.id}>{model.id} ({model.size_formatted})</option>
              {/each}
            </select>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="label" for="serve-lora">LoRA Adapter (optional)</label>
              <input id="serve-lora" type="text" class="input" placeholder="/path/to/lora/adapter" bind:value={loraAdapter} />
            </div>
            <div>
              <label class="label" for="serve-experts">Experts Dir (optional)</label>
              <input id="serve-experts" type="text" class="input" placeholder="/path/to/packed-experts" bind:value={expertsDir} />
            </div>
          </div>
        </div>
      </div>

      <!-- Network -->
      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Network</h3>
        </div>
        <div class="card-body">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="label" for="serve-host">Host</label>
              <input id="serve-host" type="text" class="input" bind:value={host} />
              <p class="text-xs text-surface-500 mt-1">Use <code class="font-mono">0.0.0.0</code> to accept external connections.</p>
            </div>
            <div>
              <label class="label" for="serve-port">Port</label>
              <input id="serve-port" type="number" min="1" max="65535" class="input" bind:value={port} />
              <p class="text-xs text-surface-500 mt-1">Will bind to <span class="font-mono">{bindUrlPreview()}</span>.</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Runtime -->
      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Runtime</h3>
        </div>
        <div class="card-body space-y-4">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label class="label" for="serve-max-seq-len">Max Sequence Length</label>
              <input
                id="serve-max-seq-len"
                type="number"
                min="256"
                max="131072"
                step="256"
                class="input"
                bind:value={maxSeqLen}
              />
            </div>
            <div>
              <label class="label" for="serve-kv-group">KV Group Size</label>
              <input
                id="serve-kv-group"
                type="number"
                min="8"
                max="256"
                step="8"
                class="input"
                bind:value={kvGroupSize}
              />
            </div>
          </div>

          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" class="rounded border-surface-300" bind:checked={fp8} />
            <span class="text-sm font-medium text-surface-700 dark:text-surface-300">
              FP8 weights (~2× memory reduction)
            </span>
          </label>
        </div>
      </div>

      <!-- KV Cache preset -->
      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">KV Cache Quantization</h3>
        </div>
        <div class="card-body">
          <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
            {#each kvPresets as preset}
              <button
                type="button"
                class="p-2 rounded-lg border text-center text-sm transition-all {kvCache === preset.value
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30 font-medium'
                  : 'border-surface-200 dark:border-surface-700 hover:border-surface-300 dark:hover:border-surface-600'}"
                onclick={() => (kvCache = preset.value)}
              >
                {preset.label}
              </button>
            {/each}
          </div>
          <p class="text-xs text-surface-500 mt-2">
            Auto lets PMetal pick the best mode for this device and model. TurboQuant presets enable outlier-aware compression
            at 2.5–3.5 effective bits.
          </p>
        </div>
      </div>

      <!-- Status -->
      {#if formError}
        <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
          {formError}
        </div>
      {/if}
      {#if formSuccess && !formError}
        <div class="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-sm" role="status">
          {formSuccess}
        </div>
      {/if}

      <button
        type="submit"
        class="btn-primary w-full"
        disabled={isStarting || isRunning || !selectedModel}
      >
        {#if isStarting}
          <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
          Starting server…
        {:else}
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          Start Server
        {/if}
      </button>
    </form>
  {/if}

  <!-- Guide -->
  <div class="card">
    <div class="card-header">
      <h3 class="font-semibold text-surface-900 dark:text-surface-100">API Reference</h3>
    </div>
    <div class="card-body space-y-2 text-sm text-surface-600 dark:text-surface-400">
      <p>The server exposes an OpenAI-compatible API once it's running:</p>
      <ul class="list-disc list-inside space-y-1 font-mono text-xs">
        <li><code>GET /health</code> — liveness probe</li>
        <li><code>GET /v1/models</code> — list loaded models</li>
        <li><code>GET /v1/metrics</code> — request metrics (tokens/sec, latency, …)</li>
        <li><code>POST /v1/chat/completions</code> — OpenAI chat API</li>
        <li><code>POST /v1/completions</code> — legacy completions</li>
      </ul>
    </div>
  </div>
</div>
