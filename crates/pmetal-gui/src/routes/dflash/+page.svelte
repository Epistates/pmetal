<script lang="ts">
  import { modelsStore } from '#lib/stores.svelte.js';
  import { startDflash } from '#lib/api.js';

  let target = $state('');
  let draft = $state('');
  let prompt = $state('Write a concise explanation of speculative decoding.');
  let maxNewTokens = $state(128);
  let temperature = $state(0);
  let speculativeTokens = $state('');
  let treeBudget = $state(0);
  let draftFp8 = $state(false);
  let json = $state(false);
  let noChat = $state(false);

  let isRunning = $state(false);
  let status = $state<'idle' | 'running' | 'done' | 'failed'>('idle');
  let runId = $state<string | null>(null);
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

    if (!target) { formError = 'Please select a target model'; return; }
    if (!draft) { formError = 'Please select a draft model'; return; }
    if (!prompt.trim()) { formError = 'Please enter a prompt'; return; }

    isRunning = true;
    status = 'running';

    try {
      runId = await startDflash({
        target,
        draft,
        prompt,
        max_new_tokens: maxNewTokens,
        temperature,
        speculative_tokens: speculativeTokens.trim() ? Number(speculativeTokens) : null,
        draft_fp8: draftFp8,
        json,
        no_chat: noChat,
        tree_budget: treeBudget,
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
    } catch (e) {
      formError = e instanceof Error ? e.message : String(e);
      status = 'failed';
      isRunning = false;
    }
  }
</script>

<div class="space-y-6 max-w-5xl">
  <div>
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">DFlash</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Run block-diffusion speculative decoding with target and draft models</p>
  </div>

  <div class="grid grid-cols-1 xl:grid-cols-2 gap-6">
    <form onsubmit={handleSubmit} class="space-y-4">
      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Models</h3>
        </div>
        <div class="card-body space-y-4">
          <div>
            <label class="label" for="dflash-target">Target Model</label>
            <select id="dflash-target" class="input" bind:value={target}>
              <option value="">Select target...</option>
              {#each models as model}
                <option value={model.id}>{model.id} ({model.size_formatted})</option>
              {/each}
            </select>
          </div>
          <div>
            <label class="label" for="dflash-draft">Draft Model</label>
            <select id="dflash-draft" class="input" bind:value={draft}>
              <option value="">Select draft...</option>
              {#each models as model}
                <option value={model.id}>{model.id} ({model.size_formatted})</option>
              {/each}
            </select>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Sampling</h3>
        </div>
        <div class="card-body space-y-4">
          <div>
            <label class="label" for="dflash-prompt">Prompt</label>
            <textarea id="dflash-prompt" class="input min-h-28" bind:value={prompt}></textarea>
          </div>
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label class="label" for="dflash-max">Max New Tokens</label>
              <input id="dflash-max" type="number" min="1" class="input" bind:value={maxNewTokens} />
            </div>
            <div>
              <label class="label" for="dflash-temp">Temperature</label>
              <input id="dflash-temp" type="number" min="0" max="5" step="0.1" class="input" bind:value={temperature} />
            </div>
            <div>
              <label class="label" for="dflash-spec">Speculative Tokens</label>
              <input id="dflash-spec" class="input" placeholder="auto" bind:value={speculativeTokens} />
            </div>
            <div>
              <label class="label" for="dflash-tree">Tree Budget</label>
              <input id="dflash-tree" type="number" min="0" max="256" class="input" bind:value={treeBudget} />
            </div>
          </div>
          <div class="flex flex-wrap gap-4 text-sm">
            <label class="flex items-center gap-2">
              <input type="checkbox" bind:checked={draftFp8} />
              <span>Draft FP8</span>
            </label>
            <label class="flex items-center gap-2">
              <input type="checkbox" bind:checked={noChat} />
              <span>No chat template</span>
            </label>
            <label class="flex items-center gap-2">
              <input type="checkbox" bind:checked={json} />
              <span>JSON output</span>
            </label>
          </div>
        </div>
      </div>

      {#if formError}
        <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
          {formError}
        </div>
      {/if}

      <button type="submit" class="btn-primary w-full" disabled={isRunning || !target || !draft || !prompt.trim()}>
        {isRunning ? 'Running...' : 'Run DFlash'}
      </button>
    </form>

    <div class="space-y-4">
      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Status</h3>
        </div>
        <div class="card-body text-sm">
          {#if status === 'idle'}
            <p class="text-surface-500">Idle</p>
          {:else if status === 'running'}
            <p class="text-primary-600 dark:text-primary-300">Running {runId}</p>
          {:else if status === 'done'}
            <p class="text-green-600 dark:text-green-300">Completed</p>
          {:else}
            <p class="text-red-600 dark:text-red-300">Failed</p>
          {/if}
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Output</h3>
        </div>
        <div class="card-body">
          <pre class="text-xs font-mono text-surface-700 dark:text-surface-300 max-h-96 overflow-y-auto whitespace-pre-wrap">{logs.length ? logs.join('\n') : 'No output yet.'}</pre>
        </div>
      </div>
    </div>
  </div>
</div>
