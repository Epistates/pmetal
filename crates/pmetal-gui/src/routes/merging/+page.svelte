<script lang="ts">
  import { onMount } from 'svelte';
  import { modelsStore } from '$lib/stores.svelte';
  import { getMergeStrategies, mergeModels } from '$lib/api';
  import type { MergeStrategy, MergeSpec } from '$lib/api';

  let models = $derived(modelsStore.models);
  let strategies = $state<MergeStrategy[]>([]);
  let loadingStrategies = $state(true);

  // Form state
  let modelA = $state('');
  let modelB = $state('');
  let baseModel = $state('');
  let selectedStrategy = $state('slerp');
  let outputPath = $state('');
  let weightA = $state(0.5);
  let weightB = $state(0.5);
  let slerpT = $state(0.5);
  let density = $state(0.5);
  let dtype = $state('bfloat16');
  let isSubmitting = $state(false);
  let formError = $state<string | null>(null);
  let mergeSuccess = $state<string | null>(null);

  let currentStrategy = $derived(strategies.find(s => s.name === selectedStrategy));

  onMount(async () => {
    try {
      strategies = await getMergeStrategies();
      if (strategies.length > 0) {
        selectedStrategy = strategies[0].name;
      }
    } catch (e) {
      console.error('Failed to load merge strategies:', e);
    } finally {
      loadingStrategies = false;
    }
  });

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    mergeSuccess = null;

    if (!modelA) { formError = 'Please select Model A'; return; }
    if (!modelB) { formError = 'Please select Model B'; return; }
    if (modelA === modelB) { formError = 'Model A and Model B must be different'; return; }
    if (!selectedStrategy) { formError = 'Please select a merge strategy'; return; }
    if (!outputPath) { formError = 'Please specify an output path'; return; }

    isSubmitting = true;
    try {
      const spec: MergeSpec = {
        model_a: modelA,
        model_b: modelB,
        output: outputPath,
        method: selectedStrategy,
        base: baseModel || undefined,
        t: slerpT,
        weight_a: weightA,
        weight_b: weightB,
        density,
        dtype,
      };
      const result = await mergeModels(spec);
      mergeSuccess = `Merge completed! Output saved to: ${result}`;
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
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Model Merging</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Combine multiple fine-tuned models using SLERP, TIES, DARE, and other strategies</p>
  </div>

  <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
    <!-- Form -->
    <div class="xl:col-span-2">
      <form onsubmit={handleSubmit} class="space-y-4">
        <!-- Strategy Selection -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Merge Strategy</h3>
          </div>
          <div class="card-body space-y-3">
            {#if loadingStrategies}
              <div class="flex justify-center py-4">
                <div class="w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full animate-spin"></div>
              </div>
            {:else}
              <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {#each strategies as strategy}
                  <button
                    type="button"
                    class="p-3 rounded-lg border text-left transition-all {selectedStrategy === strategy.name
                      ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30'
                      : 'border-surface-200 dark:border-surface-700 hover:border-surface-300 dark:hover:border-surface-600'}"
                    onclick={() => (selectedStrategy = strategy.name)}
                  >
                    <p class="text-sm font-semibold text-surface-900 dark:text-surface-100">{strategy.name}</p>
                    <p class="text-xs text-surface-500 mt-0.5">{strategy.description}</p>
                    {#if strategy.supports_weights}
                      <span class="badge-primary text-xs mt-1">Supports weights</span>
                    {/if}
                  </button>
                {/each}
              </div>
            {/if}
          </div>
        </div>

        <!-- Models to merge -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Models</h3>
          </div>
          <div class="card-body space-y-4">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label class="label" for="merge-model-a">Model A</label>
                <select id="merge-model-a" class="input" bind:value={modelA}>
                  <option value="">Select Model A...</option>
                  {#each models as model}
                    <option value={model.id}>{model.id} ({model.size_formatted})</option>
                  {/each}
                </select>
              </div>
              <div>
                <label class="label" for="merge-model-b">Model B</label>
                <select id="merge-model-b" class="input" bind:value={modelB}>
                  <option value="">Select Model B...</option>
                  {#each models as model}
                    <option value={model.id}>{model.id} ({model.size_formatted})</option>
                  {/each}
                </select>
              </div>
            </div>

            <div>
              <label class="label" for="merge-base">Base Model <span class="text-surface-400">(optional, for TIES/DARE)</span></label>
              <select id="merge-base" class="input" bind:value={baseModel}>
                <option value="">None</option>
                {#each models as model}
                  <option value={model.id}>{model.id} ({model.size_formatted})</option>
                {/each}
              </select>
            </div>

            {#if currentStrategy?.supports_weights}
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="label" for="merge-weight-a">Weight A</label>
                  <input id="merge-weight-a" type="number" class="input" step="0.05" min="0" max="10" bind:value={weightA} />
                </div>
                <div>
                  <label class="label" for="merge-weight-b">Weight B</label>
                  <input id="merge-weight-b" type="number" class="input" step="0.05" min="0" max="10" bind:value={weightB} />
                </div>
              </div>
            {/if}

            <div class="grid grid-cols-3 gap-4">
              <div>
                <label class="label" for="merge-t">SLERP t</label>
                <input id="merge-t" type="number" class="input" step="0.05" min="0" max="1" bind:value={slerpT} />
                <p class="text-xs text-surface-400 mt-0.5">0=Model A, 1=Model B</p>
              </div>
              <div>
                <label class="label" for="merge-density">Density</label>
                <input id="merge-density" type="number" class="input" step="0.05" min="0" max="1" bind:value={density} />
                <p class="text-xs text-surface-400 mt-0.5">For DARE/TIES pruning</p>
              </div>
              <div>
                <label class="label" for="merge-dtype">Output Dtype</label>
                <select id="merge-dtype" class="input" bind:value={dtype}>
                  <option value="bfloat16">bfloat16</option>
                  <option value="float16">float16</option>
                  <option value="float32">float32</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <!-- Output -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Output</h3>
          </div>
          <div class="card-body">
            <label class="label" for="merge-output">Output Path</label>
            <input
              id="merge-output"
              type="text"
              class="input"
              placeholder="/path/to/merged-model"
              bind:value={outputPath}
            />
          </div>
        </div>

        {#if formError}
          <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm">
            {formError}
          </div>
        {/if}
        {#if mergeSuccess}
          <div class="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-sm">
            {mergeSuccess}
          </div>
        {/if}

        <button type="submit" class="btn-primary w-full" disabled={isSubmitting}>
          {#if isSubmitting}
            <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            Merging models...
          {:else}
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            Merge Models
          {/if}
        </button>
      </form>
    </div>

    <!-- Strategy Info Panel -->
    <div class="xl:col-span-1">
      <div class="card sticky top-0">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Strategy Guide</h3>
        </div>
        <div class="card-body space-y-4 text-sm">
          <div>
            <p class="font-semibold text-surface-800 dark:text-surface-200 mb-1">SLERP</p>
            <p class="text-surface-500 leading-relaxed">Spherical linear interpolation. Smoothly interpolates between two model parameter spaces. Best for combining two closely-related fine-tunes.</p>
          </div>
          <div>
            <p class="font-semibold text-surface-800 dark:text-surface-200 mb-1">TIES</p>
            <p class="text-surface-500 leading-relaxed">Trim, Elect Sign, and Merge. Resolves parameter conflicts by keeping the sign with the highest total magnitude. Supports merging multiple models.</p>
          </div>
          <div>
            <p class="font-semibold text-surface-800 dark:text-surface-200 mb-1">DARE</p>
            <p class="text-surface-500 leading-relaxed">Drop And REscale. Randomly prunes delta parameters and rescales the survivors. Reduces interference between fine-tunes.</p>
          </div>
          <div>
            <p class="font-semibold text-surface-800 dark:text-surface-200 mb-1">Linear</p>
            <p class="text-surface-500 leading-relaxed">Simple weighted average of model parameters. Fast but can cause interference between models with different training objectives.</p>
          </div>
          <div class="pt-2 border-t border-surface-200 dark:border-surface-700">
            <p class="text-xs text-surface-500">All models must share the same base architecture and tokenizer.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
