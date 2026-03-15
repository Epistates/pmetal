<script lang="ts">
  import { modelsStore, distillationStore } from '$lib/stores.svelte';
  import type { DistillationConfig, DistillationRun } from '$lib/api';
  import { formatEta, runProgress, getStatusBadgeClass } from '$lib/utils';

  const lossTypes = [
    { value: 'kl_divergence', label: 'KL Divergence', description: 'Standard KL(teacher || student) loss' },
    { value: 'reverse_kl', label: 'Reverse KL', description: 'KL(student || teacher) for mode-seeking' },
    { value: 'jensen_shannon', label: 'Jensen-Shannon', description: 'Symmetric JS divergence' },
    { value: 'soft_cross_entropy', label: 'Soft Cross-Entropy', description: 'Cross-entropy with soft teacher labels' },
    { value: 'mse', label: 'MSE', description: 'Mean squared error on logits' },
  ];

  // Form state
  let studentModel = $state('');
  let teacherModel = $state('');
  let datasetPath = $state('');
  let selectedLossType = $state('kl_divergence');
  let temperature = $state(2.0);
  let alpha = $state(0.5);
  let epochs = $state(3);
  let learningRate = $state(0.0001);
  let batchSize = $state(4);
  let loraRank = $state(16);
  let loraAlpha = $state(32);
  let maxSeqLen = $state(2048);
  let outputDir = $state('');
  let isSubmitting = $state(false);
  let formError = $state<string | null>(null);
  let formSuccess = $state<string | null>(null);
  let selectedRunId = $state<string | null>(null);

  // Derived state
  let models = $derived(modelsStore.models);
  let runs = $derived(distillationStore.runs);
  let activeRuns = $derived(distillationStore.activeRuns);
  let selectedRun = $derived(runs.find(r => r.id === selectedRunId) ?? null);

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    formSuccess = null;

    if (!studentModel) { formError = 'Please select a student model'; return; }
    if (!teacherModel) { formError = 'Please select a teacher model'; return; }
    if (studentModel === teacherModel) { formError = 'Student and teacher must be different models'; return; }
    if (!datasetPath.trim()) { formError = 'Please provide a dataset path'; return; }

    isSubmitting = true;
    try {
      const config: DistillationConfig = {
        student_model: studentModel,
        teacher_model: teacherModel,
        dataset: datasetPath || null,
        loss_type: selectedLossType,
        temperature,
        alpha,
        epochs,
        learning_rate: learningRate,
        batch_size: batchSize,
        lora_rank: loraRank,
        lora_alpha: loraAlpha,
        max_seq_len: maxSeqLen,
        output_dir: outputDir || null,
      };

      const runId = await distillationStore.start(config);
      formSuccess = `Distillation started (run ID: ${runId})`;
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
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Knowledge Distillation</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Train a smaller student model to mimic a larger teacher model using soft targets</p>
  </div>

  <!-- Active runs -->
  {#if activeRuns.length > 0}
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      {#each activeRuns as run}
        <div class="card p-4">
          <div class="flex items-center justify-between mb-2">
            <div>
              <p class="font-semibold text-surface-900 dark:text-surface-100 text-sm">
                {run.student_model.split('/').pop()} &larr; {run.teacher_model.split('/').pop()}
              </p>
              <p class="text-xs text-surface-500">Step {run.step}/{run.total_steps} · {run.loss_type} · ETA {formatEta(run.eta_seconds)}</p>
            </div>
            <span class={getStatusBadgeClass(run.status)}>{run.status}</span>
          </div>
          <div class="progress-bar mb-2">
            <div class="progress-bar-fill" style="width: {runProgress(run.step, run.total_steps)}%"></div>
          </div>
          {#if run.loss !== null}
            <div class="flex gap-4 text-xs text-surface-500">
              <span>Loss: {run.loss.toFixed(4)}</span>
              {#if run.best_loss !== null}<span>Best: {run.best_loss.toFixed(4)}</span>{/if}
            </div>
          {/if}
          <button
            class="btn-danger btn-sm w-full mt-3"
            aria-label="Stop distillation run"
            onclick={() => distillationStore.stop(run.id)}
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
        <!-- Models & Data -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Models & Dataset</h3>
          </div>
          <div class="card-body space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label class="label" for="student">Student Model (to train)</label>
                <select id="student" class="input" bind:value={studentModel}>
                  <option value="">Select student model...</option>
                  {#each models as model}
                    <option value={model.id}>{model.id} ({model.size_formatted})</option>
                  {/each}
                </select>
              </div>
              <div>
                <label class="label" for="teacher">Teacher Model (supervisor)</label>
                <select id="teacher" class="input" bind:value={teacherModel}>
                  <option value="">Select teacher model...</option>
                  {#each models as model}
                    <option value={model.id}>{model.id} ({model.size_formatted})</option>
                  {/each}
                </select>
              </div>
            </div>
            <div>
              <label class="label" for="distill-dataset">Dataset Path</label>
              <input
                id="distill-dataset"
                type="text"
                class="input"
                placeholder="/path/to/dataset or HuggingFace dataset ID"
                bind:value={datasetPath}
              />
            </div>
            <div>
              <label class="label" for="distill-output">Output Directory</label>
              <input id="distill-output" type="text" class="input" placeholder="./distilled-model" bind:value={outputDir} />
            </div>
          </div>
        </div>

        <!-- Loss Function -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Loss Function</h3>
          </div>
          <div class="card-body space-y-4">
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
              {#each lossTypes as lossType}
                <button
                  type="button"
                  class="p-3 rounded-lg border text-left transition-all {selectedLossType === lossType.value
                    ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30'
                    : 'border-surface-200 dark:border-surface-700 hover:border-surface-300 dark:hover:border-surface-600'}"
                  onclick={() => (selectedLossType = lossType.value)}
                >
                  <p class="text-sm font-semibold text-surface-900 dark:text-surface-100">{lossType.label}</p>
                  <p class="text-xs text-surface-500 mt-0.5">{lossType.description}</p>
                </button>
              {/each}
            </div>

            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="label" for="temperature">Temperature</label>
                <input id="temperature" type="number" class="input" step="0.1" min="0.1" bind:value={temperature} />
                <p class="text-xs text-surface-500 mt-1">Softens teacher probability distribution</p>
              </div>
              <div>
                <label class="label" for="alpha">Alpha (distillation weight)</label>
                <input id="alpha" type="number" class="input" step="0.05" min="0" max="1" bind:value={alpha} />
                <p class="text-xs text-surface-500 mt-1">Balance: 1.0 = pure distillation, 0.0 = pure SFT</p>
              </div>
            </div>
          </div>
        </div>

        <!-- Training Parameters -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Training Parameters</h3>
          </div>
          <div class="card-body">
            <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <label class="label" for="distill-epochs">Epochs</label>
                <input id="distill-epochs" type="number" class="input" min="1" bind:value={epochs} />
              </div>
              <div>
                <label class="label" for="distill-lr">Learning Rate</label>
                <input id="distill-lr" type="number" class="input" step="0.00001" bind:value={learningRate} />
              </div>
              <div>
                <label class="label" for="distill-batch">Batch Size</label>
                <input id="distill-batch" type="number" class="input" min="1" bind:value={batchSize} />
              </div>
              <div>
                <label class="label" for="distill-lora">LoRA Rank</label>
                <input id="distill-lora" type="number" class="input" min="4" step="4" bind:value={loraRank} />
              </div>
              <div>
                <label class="label" for="distill-lora-alpha">LoRA Alpha</label>
                <input id="distill-lora-alpha" type="number" class="input" min="1" bind:value={loraAlpha} />
              </div>
              <div>
                <label class="label" for="distill-max-seq">Max Seq Length</label>
                <input id="distill-max-seq" type="number" class="input" step="64" bind:value={maxSeqLen} />
              </div>
            </div>
          </div>
        </div>

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
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
            Start Distillation
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
              No distillation runs yet
            </div>
          {:else}
            {#each runs as run}
              <button
                class="w-full text-left p-4 hover:bg-surface-50 dark:hover:bg-surface-700/50 transition-colors {selectedRunId === run.id ? 'bg-primary-50 dark:bg-primary-900/20' : ''}"
                onclick={() => (selectedRunId = selectedRunId === run.id ? null : run.id)}
              >
                <div class="flex items-center justify-between mb-1">
                  <span class="text-sm font-medium truncate max-w-[140px]">{run.student_model.split('/').pop()}</span>
                  <span class={getStatusBadgeClass(run.status)}>{run.status}</span>
                </div>
                <div class="text-xs text-surface-500 mb-1">
                  Teacher: {run.teacher_model.split('/').pop()}
                </div>
                <div class="text-xs text-surface-500 mb-2">
                  {run.loss_type} · Step {run.step}/{run.total_steps}
                </div>
                {#if run.status !== 'pending' && run.status !== 'completed'}
                  <div class="progress-bar">
                    <div class="progress-bar-fill" style="width: {runProgress(run.step, run.total_steps)}%"></div>
                  </div>
                {/if}
              </button>
            {/each}
          {/if}
        </div>

        {#if selectedRun}
          <div class="card-footer">
            <div class="space-y-2 text-sm">
              {#if selectedRun.loss !== null}
                <div class="flex justify-between">
                  <span class="text-surface-500">Loss</span>
                  <span class="font-mono">{selectedRun.loss.toFixed(4)}</span>
                </div>
              {/if}
              {#if selectedRun.best_loss !== null}
                <div class="flex justify-between">
                  <span class="text-surface-500">Best Loss</span>
                  <span class="font-mono">{selectedRun.best_loss.toFixed(4)}</span>
                </div>
              {/if}
              {#if selectedRun.eta_seconds !== null}
                <div class="flex justify-between">
                  <span class="text-surface-500">ETA</span>
                  <span>{formatEta(selectedRun.eta_seconds)}</span>
                </div>
              {/if}
              {#if ['training', 'loading_models', 'generating_signals'].includes(selectedRun.status)}
                <button class="btn-danger btn-sm w-full" aria-label="Stop distillation" onclick={() => distillationStore.stop(selectedRun!.id)}>
                  Stop Distillation
                </button>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>
