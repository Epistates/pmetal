<script lang="ts">
  import { modelsStore } from '#lib/stores.svelte.js';
  import { startRlkd, type RlkdConfig } from '#lib/api.js';

  // Form state
  let model = $state('');
  let teacherModel = $state('');
  let dataset = $state('');
  let outputDir = $state('./output/rlkd');
  let distillAlpha = $state(0.3);
  let finalAlpha = $state(0.05);
  let annealAlpha = $state(false);
  let distillTemperature = $state(2.0);
  let numGenerations = $state(8);
  let beta = $state(0.001);
  let learningRate = $state(5e-6);
  let epochs = $state(1);
  let loraR = $state(16);
  let loraAlpha = $state(32.0);
  let maxSeqLen = $state(512);
  let maxCompletionLength = $state(512);
  let seed = $state(42);
  let reasoningRewards = $state(false);
  let textColumn = $state('');
  let promptColumn = $state('');
  let responseColumn = $state('');

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

    if (!model) { formError = 'Please select a policy model'; return; }
    if (!teacherModel) { formError = 'Please select a teacher model'; return; }
    if (!dataset.trim()) { formError = 'Please enter a dataset'; return; }

    isRunning = true;
    status = 'running';

    const config: RlkdConfig = {
      model,
      teacher_model: teacherModel,
      dataset: dataset.trim(),
      output_dir: outputDir.trim() || null,
      distill_alpha: distillAlpha,
      final_alpha: finalAlpha,
      anneal_alpha: annealAlpha,
      distill_temperature: distillTemperature,
      num_generations: numGenerations,
      beta,
      learning_rate: learningRate,
      epochs,
      lora_r: loraR,
      lora_alpha: loraAlpha,
      max_seq_len: maxSeqLen,
      max_completion_length: maxCompletionLength,
      seed,
      reasoning_rewards: reasoningRewards,
      text_column: textColumn.trim() || null,
      prompt_column: promptColumn.trim() || null,
      response_column: responseColumn.trim() || null,
    };

    try {
      runId = await startRlkd(config, (e: Record<string, unknown>) => {
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
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">RLKD</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Reinforcement Learning with Knowledge Distillation — combines GRPO rewards with teacher distillation</p>
  </div>

  <form onsubmit={handleSubmit} class="space-y-4">
    <!-- Models -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Models</h3>
      </div>
      <div class="card-body space-y-4">
        <div>
          <label class="label" for="rlkd-model">Policy Model (student)</label>
          <select id="rlkd-model" class="input" bind:value={model}>
            <option value="">Select policy model...</option>
            {#each models as m}
              <option value={m.id}>{m.id} ({m.size_formatted})</option>
            {/each}
          </select>
        </div>
        <div>
          <label class="label" for="rlkd-teacher">Teacher Model</label>
          <select id="rlkd-teacher" class="input" bind:value={teacherModel}>
            <option value="">Select teacher model...</option>
            {#each models as m}
              <option value={m.id}>{m.id} ({m.size_formatted})</option>
            {/each}
          </select>
        </div>
      </div>
    </div>

    <!-- Data -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Data</h3>
      </div>
      <div class="card-body space-y-4">
        <div>
          <label class="label" for="rlkd-dataset">Dataset (HF ID or local path)</label>
          <input id="rlkd-dataset" type="text" class="input" placeholder="e.g. openai/gsm8k" bind:value={dataset} />
        </div>
        <div>
          <label class="label" for="rlkd-output">Output Directory</label>
          <input id="rlkd-output" type="text" class="input" bind:value={outputDir} />
        </div>
        <div class="grid grid-cols-3 gap-3">
          <div>
            <label class="label" for="rlkd-textcol">Text Column</label>
            <input id="rlkd-textcol" type="text" class="input" placeholder="optional" bind:value={textColumn} />
          </div>
          <div>
            <label class="label" for="rlkd-promptcol">Prompt Column</label>
            <input id="rlkd-promptcol" type="text" class="input" placeholder="optional" bind:value={promptColumn} />
          </div>
          <div>
            <label class="label" for="rlkd-respcol">Response Column</label>
            <input id="rlkd-respcol" type="text" class="input" placeholder="optional" bind:value={responseColumn} />
          </div>
        </div>
      </div>
    </div>

    <!-- Distillation -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Distillation</h3>
      </div>
      <div class="card-body grid grid-cols-2 gap-4">
        <div>
          <label class="label" for="rlkd-da">Distill Alpha (start)</label>
          <input id="rlkd-da" type="number" class="input" step="0.01" min="0" max="1" bind:value={distillAlpha} />
        </div>
        <div>
          <label class="label" for="rlkd-fa">Final Alpha</label>
          <input id="rlkd-fa" type="number" class="input" step="0.01" min="0" max="1" bind:value={finalAlpha} />
        </div>
        <div>
          <label class="label" for="rlkd-dt">Distill Temperature</label>
          <input id="rlkd-dt" type="number" class="input" step="0.1" min="0.1" max="20" bind:value={distillTemperature} />
        </div>
        <div class="flex items-end pb-1">
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" class="rounded border-surface-300" bind:checked={annealAlpha} />
            <span class="text-sm font-medium text-surface-700 dark:text-surface-300">Anneal alpha over training</span>
          </label>
        </div>
      </div>
    </div>

    <!-- GRPO -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">GRPO</h3>
      </div>
      <div class="card-body grid grid-cols-2 gap-4">
        <div>
          <label class="label" for="rlkd-ngen">Num Generations</label>
          <input id="rlkd-ngen" type="number" class="input" min="1" max="1024" bind:value={numGenerations} />
        </div>
        <div>
          <label class="label" for="rlkd-beta">KL Beta</label>
          <input id="rlkd-beta" type="number" class="input" step="0.0001" min="0" max="1" bind:value={beta} />
        </div>
        <div>
          <label class="label" for="rlkd-maxcomp">Max Completion Length</label>
          <input id="rlkd-maxcomp" type="number" class="input" min="32" max="8192" bind:value={maxCompletionLength} />
        </div>
        <div class="flex items-end pb-1">
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" class="rounded border-surface-300" bind:checked={reasoningRewards} />
            <span class="text-sm font-medium text-surface-700 dark:text-surface-300">Enable reasoning rewards</span>
          </label>
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
          <label class="label" for="rlkd-lr">Learning Rate</label>
          <input id="rlkd-lr" type="number" class="input" step="1e-7" min="1e-8" max="1" bind:value={learningRate} />
        </div>
        <div>
          <label class="label" for="rlkd-epochs">Epochs</label>
          <input id="rlkd-epochs" type="number" class="input" min="1" max="1000" bind:value={epochs} />
        </div>
        <div>
          <label class="label" for="rlkd-lorar">LoRA r</label>
          <input id="rlkd-lorar" type="number" class="input" min="1" max="1024" bind:value={loraR} />
        </div>
        <div>
          <label class="label" for="rlkd-loraalpha">LoRA alpha</label>
          <input id="rlkd-loraalpha" type="number" class="input" min="1" max="1024" bind:value={loraAlpha} />
        </div>
        <div>
          <label class="label" for="rlkd-seqlen">Max Seq Len</label>
          <input id="rlkd-seqlen" type="number" class="input" min="32" max="32768" bind:value={maxSeqLen} />
        </div>
        <div>
          <label class="label" for="rlkd-seed">Seed</label>
          <input id="rlkd-seed" type="number" class="input" min="0" bind:value={seed} />
        </div>
      </div>
    </div>

    <!-- Status -->
    {#if status === 'running'}
      <div class="p-4 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 text-primary-700 dark:text-primary-300 text-sm flex items-center gap-2" role="status">
        <div class="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin flex-shrink-0" aria-hidden="true"></div>
        RLKD running… Run ID: {runId}
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

    <button type="submit" class="btn-primary w-full" disabled={isRunning || !model || !teacherModel || !dataset.trim()}>
      {#if isRunning}
        <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
        Running RLKD...
      {:else}
        Start RLKD
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
