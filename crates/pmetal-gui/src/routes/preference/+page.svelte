<script lang="ts">
  import { modelsStore } from '$lib/stores.svelte';
  import { startPreference, type PreferenceSpec } from '$lib/api';

  type Loss = NonNullable<PreferenceSpec['loss']>;

  const LOSSES: { value: Loss; label: string; hint: string }[] = [
    { value: 'dpo', label: 'DPO', hint: 'Pairs, against the starting model. Label smoothing above 0 trains Robust DPO.' },
    { value: 'ipo', label: 'IPO', hint: 'Pairs, against the starting model; regresses the margin to a target.' },
    { value: 'hinge', label: 'Hinge (SLiC)', hint: 'Pairs, against the starting model; stops once the margin clears 1/β.' },
    { value: 'simpo', label: 'SimPO', hint: 'Pairs, no reference; length-normalized reward with a target margin.' },
    { value: 'orpo', label: 'ORPO', hint: 'Pairs, no reference; fine-tunes on the chosen answer while penalizing the rejected one.' },
    { value: 'kto', label: 'KTO', hint: 'Single completions labelled good or bad; use a batch size of 4 or more.' },
  ];

  // Form state (defaults match PreferenceSpec)
  let model = $state('');
  let dataset = $state('');
  let outputDir = $state('./output/preference');
  let loss = $state<Loss>('dpo');
  let beta = $state<number | null>(null);
  let simpoGammaRatio = $state(0.5);
  let labelSmoothing = $state(0.0);
  let desirableWeight = $state(1.0);
  let undesirableWeight = $state(1.0);
  let learningRate = $state(1e-5);
  let batchSize = $state(2);
  let gradAccum = $state(8);
  let epochs = $state(1);
  let warmupRatio = $state(0.1);
  let loraR = $state(16);
  let loraAlpha = $state(32);
  let maxPromptLength = $state(512);
  let maxLength = $state(1024);
  let seed = $state(42);

  // UI state
  let isRunning = $state(false);
  let runId = $state<string | null>(null);
  let status = $state<'idle' | 'running' | 'done' | 'failed'>('idle');
  let formError = $state<string | null>(null);
  let logs = $state<string[]>([]);

  let models = $derived(modelsStore.models);
  let lossHint = $derived(LOSSES.find((l) => l.value === loss)?.hint ?? '');
  let defaultBeta = $derived(loss === 'simpo' ? 2.5 : 0.1);

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

    const spec: PreferenceSpec = {
      model,
      dataset: dataset.trim(),
      output_dir: outputDir.trim() || './output/preference',
      loss,
      beta,
      simpo_gamma_ratio: simpoGammaRatio,
      label_smoothing: labelSmoothing,
      desirable_weight: desirableWeight,
      undesirable_weight: undesirableWeight,
      learning_rate: learningRate,
      batch_size: batchSize,
      gradient_accumulation_steps: gradAccum,
      epochs,
      warmup_ratio: warmupRatio,
      lora_r: loraR,
      lora_alpha: loraAlpha,
      max_prompt_length: maxPromptLength,
      max_length: maxLength,
      seed,
    };

    try {
      runId = await startPreference(spec, (e: Record<string, unknown>) => {
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
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Preference</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Align a model to preference data with a LoRA adapter: DPO, IPO, hinge, SimPO, ORPO or KTO</p>
  </div>

  <form onsubmit={handleSubmit} class="space-y-4">
    <!-- Model and data -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Model &amp; Data</h3>
      </div>
      <div class="card-body space-y-4">
        <div>
          <label class="label" for="pref-model">Model</label>
          <select id="pref-model" class="input" bind:value={model}>
            <option value="">Select model...</option>
            {#each models as m}
              <option value={m.id}>{m.id} ({m.size_formatted})</option>
            {/each}
          </select>
        </div>
        <div>
          <label class="label" for="pref-dataset">Dataset (HF ID or local JSONL / JSON / Parquet)</label>
          <input id="pref-dataset" type="text" class="input" placeholder="prompt / chosen / rejected rows" bind:value={dataset} />
        </div>
        <div>
          <label class="label" for="pref-output">Output Directory</label>
          <input id="pref-output" type="text" class="input" bind:value={outputDir} />
        </div>
      </div>
    </div>

    <!-- Objective -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Objective</h3>
      </div>
      <div class="card-body space-y-4">
        <div>
          <label class="label" for="pref-loss">Loss</label>
          <select id="pref-loss" class="input" bind:value={loss}>
            {#each LOSSES as l}
              <option value={l.value}>{l.label}</option>
            {/each}
          </select>
          <p class="text-xs text-surface-500 dark:text-surface-400 mt-1">{lossHint}</p>
        </div>
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label class="label" for="pref-beta">β (blank for {defaultBeta})</label>
            <input id="pref-beta" type="number" class="input" step="0.01" min="0" max="100" placeholder={String(defaultBeta)} bind:value={beta} />
          </div>
          {#if loss === 'simpo'}
            <div>
              <label class="label" for="pref-gamma">Target margin γ/β</label>
              <input id="pref-gamma" type="number" class="input" step="0.05" min="0" max="10" bind:value={simpoGammaRatio} />
            </div>
          {/if}
          {#if loss === 'dpo'}
            <div>
              <label class="label" for="pref-ls">Label smoothing (Robust DPO)</label>
              <input id="pref-ls" type="number" class="input" step="0.01" min="0" max="0.49" bind:value={labelSmoothing} />
            </div>
          {/if}
          {#if loss === 'kto'}
            <div>
              <label class="label" for="pref-dw">Desirable weight</label>
              <input id="pref-dw" type="number" class="input" step="0.1" min="0" max="100" bind:value={desirableWeight} />
            </div>
            <div>
              <label class="label" for="pref-uw">Undesirable weight</label>
              <input id="pref-uw" type="number" class="input" step="0.1" min="0" max="100" bind:value={undesirableWeight} />
            </div>
          {/if}
        </div>
      </div>
    </div>

    <!-- Training -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Training &amp; LoRA</h3>
      </div>
      <div class="card-body grid grid-cols-2 gap-4">
        <div>
          <label class="label" for="pref-lr">Learning Rate</label>
          <input id="pref-lr" type="number" class="input" step="1e-7" min="1e-9" max="1" bind:value={learningRate} />
        </div>
        <div>
          <label class="label" for="pref-epochs">Epochs</label>
          <input id="pref-epochs" type="number" class="input" min="1" max="1000" bind:value={epochs} />
        </div>
        <div>
          <label class="label" for="pref-bs">Batch Size</label>
          <input id="pref-bs" type="number" class="input" min="1" max="1024" bind:value={batchSize} />
        </div>
        <div>
          <label class="label" for="pref-ga">Gradient Accumulation</label>
          <input id="pref-ga" type="number" class="input" min="1" max="1024" bind:value={gradAccum} />
        </div>
        <div>
          <label class="label" for="pref-warmup">Warmup Ratio</label>
          <input id="pref-warmup" type="number" class="input" step="0.01" min="0" max="1" bind:value={warmupRatio} />
        </div>
        <div>
          <label class="label" for="pref-seed">Seed</label>
          <input id="pref-seed" type="number" class="input" min="0" bind:value={seed} />
        </div>
        <div>
          <label class="label" for="pref-lorar">LoRA r</label>
          <input id="pref-lorar" type="number" class="input" min="1" max="1024" bind:value={loraR} />
        </div>
        <div>
          <label class="label" for="pref-loraalpha">LoRA alpha</label>
          <input id="pref-loraalpha" type="number" class="input" min="1" max="4096" bind:value={loraAlpha} />
        </div>
        <div>
          <label class="label" for="pref-mpl">Max Prompt Length</label>
          <input id="pref-mpl" type="number" class="input" min="1" max="32768" bind:value={maxPromptLength} />
        </div>
        <div>
          <label class="label" for="pref-ml">Max Length</label>
          <input id="pref-ml" type="number" class="input" min="2" max="65536" bind:value={maxLength} />
        </div>
      </div>
    </div>

    <!-- Status -->
    {#if status === 'running'}
      <div class="p-4 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 text-primary-700 dark:text-primary-300 text-sm flex items-center gap-2" role="status">
        <div class="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin flex-shrink-0" aria-hidden="true"></div>
        Training… Run ID: {runId}
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
        Start {LOSSES.find((l) => l.value === loss)?.label}
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
