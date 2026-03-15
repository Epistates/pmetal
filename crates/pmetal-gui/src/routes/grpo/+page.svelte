<script lang="ts">
  import { modelsStore, grpoStore } from '$lib/stores.svelte';
  import type { GrpoConfig, GrpoRun } from '$lib/api';
  import { formatEta, runProgress, getStatusBadgeClass } from '$lib/utils';

  // Form state
  let selectedModel = $state('');
  let datasetPath = $state('');
  let epochs = $state(1);
  let learningRate = $state(0.00001);
  let batchSize = $state(4);
  let groupSize = $state(8);
  let beta = $state(0.04);
  let loraRank = $state(16);
  let loraAlpha = $state(32);
  let maxSeqLen = $state(2048);
  let outputDir = $state('');
  let useReasoningRewards = $state(true);

  // UI state
  let isSubmitting = $state(false);
  let formError = $state<string | null>(null);
  let formSuccess = $state<string | null>(null);
  let selectedRunId = $state<string | null>(null);

  // Derived state
  let models = $derived(modelsStore.models);
  let runs = $derived(grpoStore.runs);
  let activeRuns = $derived(grpoStore.activeRuns);
  let selectedRun = $derived(runs.find(r => r.id === selectedRunId) ?? null);

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    formSuccess = null;

    if (!selectedModel) { formError = 'Please select a model'; return; }
    if (!datasetPath.trim()) { formError = 'Please provide a dataset path'; return; }

    isSubmitting = true;
    try {
      const config: GrpoConfig = {
        model: selectedModel,
        dataset: datasetPath || null,
        epochs,
        learning_rate: learningRate,
        batch_size: batchSize,
        group_size: groupSize,
        beta,
        lora_rank: loraRank,
        lora_alpha: loraAlpha,
        max_seq_len: maxSeqLen,
        output_dir: outputDir || null,
        use_reasoning_rewards: useReasoningRewards,
      };

      const runId = await grpoStore.start(config);
      formSuccess = `GRPO training started (run ID: ${runId})`;
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
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">GRPO Training</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Group Relative Policy Optimization — reinforcement learning for reasoning and math</p>
  </div>

  <!-- Active runs summary -->
  {#if activeRuns.length > 0}
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      {#each activeRuns as run}
        <div class="card p-4">
          <div class="flex items-center justify-between mb-3">
            <div>
              <p class="font-semibold text-surface-900 dark:text-surface-100">{run.model.split('/').pop()}</p>
              <p class="text-sm text-surface-500">Step {run.step}/{run.total_steps} · ETA {formatEta(run.eta_seconds)}</p>
            </div>
            <span class={getStatusBadgeClass(run.status)}>{run.status}</span>
          </div>
          <div class="progress-bar mb-3">
            <div class="progress-bar-fill-accent" style="width: {runProgress(run.step, run.total_steps)}%"></div>
          </div>
          {#if run.reward_mean !== null || run.kl_div !== null}
            <div class="grid grid-cols-3 gap-2 text-sm">
              {#if run.reward_mean !== null}
                <div class="p-2 rounded bg-surface-50 dark:bg-surface-700/50 text-center">
                  <p class="text-xs text-surface-500 mb-0.5">Reward Mean</p>
                  <p class="font-mono font-semibold text-accent-600 dark:text-accent-400">{run.reward_mean.toFixed(3)}</p>
                </div>
              {/if}
              {#if run.reward_std !== null}
                <div class="p-2 rounded bg-surface-50 dark:bg-surface-700/50 text-center">
                  <p class="text-xs text-surface-500 mb-0.5">Reward Std</p>
                  <p class="font-mono font-semibold">{run.reward_std.toFixed(3)}</p>
                </div>
              {/if}
              {#if run.kl_div !== null}
                <div class="p-2 rounded bg-surface-50 dark:bg-surface-700/50 text-center">
                  <p class="text-xs text-surface-500 mb-0.5">KL Div</p>
                  <p class="font-mono font-semibold">{run.kl_div.toFixed(4)}</p>
                </div>
              {/if}
            </div>
          {/if}
          <button
            class="btn-danger btn-sm w-full mt-3"
            aria-label="Stop GRPO run for {run.model.split('/').pop()}"
            onclick={() => grpoStore.stop(run.id)}
          >
            Stop Run
          </button>
        </div>
      {/each}
    </div>
  {/if}

  <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
    <!-- GRPO Config Form -->
    <div class="xl:col-span-2">
      <form onsubmit={handleSubmit} class="space-y-4">
        <!-- Model & Data -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Model & Dataset</h3>
          </div>
          <div class="card-body space-y-4">
            <div>
              <label class="label" for="grpo-model">Model</label>
              <select id="grpo-model" class="input" bind:value={selectedModel}>
                <option value="">Select a cached model...</option>
                {#each models as model}
                  <option value={model.id}>{model.id} ({model.size_formatted})</option>
                {/each}
              </select>
            </div>
            <div>
              <label class="label" for="grpo-dataset">Dataset Path</label>
              <input
                id="grpo-dataset"
                type="text"
                class="input"
                placeholder="/path/to/dataset or HuggingFace dataset ID"
                bind:value={datasetPath}
              />
              <p class="text-xs text-surface-500 mt-1">Dataset should contain prompts. Responses are generated during training.</p>
            </div>
            <div>
              <label class="label" for="grpo-output">Output Directory</label>
              <input id="grpo-output" type="text" class="input" placeholder="./grpo-output" bind:value={outputDir} />
            </div>
          </div>
        </div>

        <!-- GRPO Parameters -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">GRPO Parameters</h3>
          </div>
          <div class="card-body space-y-4">
            <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
              <div>
                <label class="label" for="grpo-group-size">Group Size</label>
                <input id="grpo-group-size" type="number" class="input" min="2" bind:value={groupSize} />
                <p class="text-xs text-surface-500 mt-1">Completions per prompt</p>
              </div>
              <div>
                <label class="label" for="grpo-beta">Beta (KL Coefficient)</label>
                <input id="grpo-beta" type="number" class="input" step="0.001" min="0" bind:value={beta} />
                <p class="text-xs text-surface-500 mt-1">KL penalty weight</p>
              </div>
              <div>
                <label class="label" for="grpo-lora-rank">LoRA Rank</label>
                <input id="grpo-lora-rank" type="number" class="input" min="4" max="256" step="4" bind:value={loraRank} />
              </div>
              <div>
                <label class="label" for="grpo-lora-alpha">LoRA Alpha</label>
                <input id="grpo-lora-alpha" type="number" class="input" min="1" bind:value={loraAlpha} />
              </div>
              <div>
                <label class="label" for="grpo-lr">Learning Rate</label>
                <input id="grpo-lr" type="number" class="input" step="0.000001" bind:value={learningRate} />
              </div>
              <div>
                <label class="label" for="grpo-batch">Batch Size</label>
                <input id="grpo-batch" type="number" class="input" min="1" bind:value={batchSize} />
              </div>
              <div>
                <label class="label" for="grpo-epochs">Epochs</label>
                <input id="grpo-epochs" type="number" class="input" min="1" bind:value={epochs} />
              </div>
              <div>
                <label class="label" for="grpo-max-seq">Max Seq Length</label>
                <input id="grpo-max-seq" type="number" class="input" step="64" bind:value={maxSeqLen} />
              </div>
            </div>

            <!-- Reasoning Rewards -->
            <div class="p-4 rounded-lg bg-accent-50 dark:bg-accent-900/20 border border-accent-200 dark:border-accent-800">
              <label class="flex items-start gap-3 cursor-pointer">
                <input type="checkbox" class="mt-0.5 rounded border-accent-300" bind:checked={useReasoningRewards} />
                <div>
                  <span class="text-sm font-semibold text-accent-800 dark:text-accent-200">Reasoning Rewards</span>
                  <p class="text-xs text-accent-700 dark:text-accent-300 mt-0.5">
                    Apply format + correctness rewards for math/reasoning tasks. Rewards thinking tokens wrapped in &lt;think&gt; tags and verifiable final answers.
                  </p>
                </div>
              </label>
            </div>
          </div>
        </div>

        <!-- Info box -->
        <div class="p-4 rounded-lg bg-surface-50 dark:bg-surface-800/50 border border-surface-200 dark:border-surface-700 text-sm text-surface-600 dark:text-surface-400">
          <p class="font-medium mb-1">How GRPO works</p>
          <p>For each prompt, PMetal generates <span class="font-mono">{groupSize}</span> completions and scores them with reward functions. The policy is updated to favor higher-reward completions while the KL divergence (beta={beta}) keeps the model close to the reference.</p>
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

        <button type="submit" class="btn-accent w-full" disabled={isSubmitting}>
          {#if isSubmitting}
            <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
            Starting...
          {:else}
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Start GRPO Training
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
              No GRPO runs yet
            </div>
          {:else}
            {#each runs as run}
              <button
                class="w-full text-left p-4 hover:bg-surface-50 dark:hover:bg-surface-700/50 transition-colors {selectedRunId === run.id ? 'bg-primary-50 dark:bg-primary-900/20' : ''}"
                onclick={() => (selectedRunId = selectedRunId === run.id ? null : run.id)}
              >
                <div class="flex items-center justify-between mb-1">
                  <span class="text-sm font-medium truncate max-w-[140px]">{run.model.split('/').pop()}</span>
                  <span class={getStatusBadgeClass(run.status)}>{run.status}</span>
                </div>
                <div class="text-xs text-surface-500 mb-2">
                  Step {run.step}/{run.total_steps}
                </div>
                {#if run.status === 'running'}
                  <div class="progress-bar">
                    <div class="progress-bar-fill-accent" style="width: {runProgress(run.step, run.total_steps)}%"></div>
                  </div>
                {/if}
                {#if run.reward_mean !== null}
                  <div class="mt-1 text-xs text-surface-500">
                    Reward: {run.reward_mean.toFixed(3)} ± {(run.reward_std ?? 0).toFixed(3)}
                  </div>
                {/if}
              </button>
            {/each}
          {/if}
        </div>

        {#if selectedRun}
          <div class="card-footer">
            <div class="space-y-2 text-sm">
              <div class="grid grid-cols-2 gap-2">
                {#if selectedRun.reward_mean !== null}
                  <div>
                    <span class="text-surface-500">Reward</span>
                    <span class="font-mono ml-2 text-accent-600 dark:text-accent-400">{selectedRun.reward_mean.toFixed(3)}</span>
                  </div>
                {/if}
                {#if selectedRun.kl_div !== null}
                  <div>
                    <span class="text-surface-500">KL</span>
                    <span class="font-mono ml-2">{selectedRun.kl_div.toFixed(4)}</span>
                  </div>
                {/if}
                {#if selectedRun.loss !== null}
                  <div>
                    <span class="text-surface-500">Loss</span>
                    <span class="font-mono ml-2">{selectedRun.loss.toFixed(4)}</span>
                  </div>
                {/if}
                {#if selectedRun.tokens_per_second !== null}
                  <div>
                    <span class="text-surface-500">Tok/s</span>
                    <span class="font-mono ml-2">{selectedRun.tokens_per_second.toFixed(0)}</span>
                  </div>
                {/if}
              </div>
              {#if selectedRun.status === 'running'}
                <button class="btn-danger btn-sm w-full" onclick={() => grpoStore.stop(selectedRun!.id)} aria-label="Stop GRPO run">
                  Stop GRPO
                </button>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>
