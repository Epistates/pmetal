<script lang="ts">
  import { onMount } from 'svelte';
  import { modelsStore } from '$lib/stores.svelte';
  import { getMergeStrategies, mergeModels } from '$lib/api';
  import type { MergeStrategy, MergeModelEntry, MergeConfig } from '$lib/api';

  let models = $derived(modelsStore.models);
  let strategies = $state<MergeStrategy[]>([]);
  let loadingStrategies = $state(true);

  // Form state
  let baseModel = $state('');
  let selectedStrategy = $state('');
  let outputPath = $state('');
  let modelEntries = $state<MergeModelEntry[]>([{ model: '', weight: 1.0 }]);
  let isSubmitting = $state(false);
  let formError = $state<string | null>(null);
  let mergeSuccess = $state<string | null>(null);

  let currentStrategy = $derived(strategies.find(s => s.name === selectedStrategy));
  let canAddMore = $derived(modelEntries.length < 5);

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

  function addModelEntry() {
    if (canAddMore) {
      modelEntries = [...modelEntries, { model: '', weight: 1.0 }];
    }
  }

  function removeModelEntry(index: number) {
    if (modelEntries.length > 1) {
      modelEntries = modelEntries.filter((_, i) => i !== index);
    }
  }

  function updateModelEntry(index: number, field: 'model' | 'weight', value: string | number) {
    modelEntries = modelEntries.map((entry, i) =>
      i === index ? { ...entry, [field]: value } : entry
    );
  }

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    mergeSuccess = null;

    if (!baseModel) { formError = 'Please select a base model'; return; }
    if (!selectedStrategy) { formError = 'Please select a merge strategy'; return; }
    const validEntries = modelEntries.filter(e => e.model);
    if (validEntries.length === 0) { formError = 'Please add at least one model to merge'; return; }
    if (!outputPath) { formError = 'Please specify an output path'; return; }

    isSubmitting = true;
    try {
      const config: MergeConfig = {
        base_model: baseModel,
        models: validEntries,
        strategy: selectedStrategy,
        output: outputPath,
      };
      const result = await mergeModels(config);
      mergeSuccess = `Merge completed! Output saved to: ${result}`;
    } catch (e) {
      formError = e instanceof Error ? e.message : String(e);
    } finally {
      isSubmitting = false;
    }
  }

  function normalizeWeights() {
    const total = modelEntries.reduce((sum, e) => sum + e.weight, 0);
    if (total > 0) {
      modelEntries = modelEntries.map(e => ({
        ...e,
        weight: Math.round((e.weight / total) * 10000) / 10000,
      }));
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

        <!-- Base Model -->
        <div class="card">
          <div class="card-header">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Base Model</h3>
          </div>
          <div class="card-body">
            <select class="input" bind:value={baseModel}>
              <option value="">Select base model...</option>
              {#each models as model}
                <option value={model.id}>{model.id} ({model.size_formatted})</option>
              {/each}
            </select>
            <p class="text-xs text-surface-500 mt-1">The base architecture. Other models are merged relative to this one.</p>
          </div>
        </div>

        <!-- Models to merge -->
        <div class="card">
          <div class="card-header flex items-center justify-between">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Models to Merge</h3>
            <div class="flex gap-2">
              {#if currentStrategy?.supports_weights}
                <button type="button" class="btn-ghost btn-sm" onclick={normalizeWeights} title="Normalize weights to sum to 1.0">
                  Normalize
                </button>
              {/if}
              <button
                type="button"
                class="btn-secondary btn-sm"
                onclick={addModelEntry}
                disabled={!canAddMore}
              >
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
                </svg>
                Add Model
              </button>
            </div>
          </div>
          <div class="card-body space-y-3">
            {#each modelEntries as entry, index}
              <div class="flex items-center gap-3">
                <div class="flex-1">
                  <select
                    class="input"
                    value={entry.model}
                    onchange={(e) => updateModelEntry(index, 'model', (e.target as HTMLSelectElement).value)}
                  >
                    <option value="">Select model {index + 1}...</option>
                    {#each models as model}
                      <option value={model.id}>{model.id}</option>
                    {/each}
                  </select>
                </div>

                {#if currentStrategy?.supports_weights}
                  <div class="w-28 flex-shrink-0">
                    <input
                      type="number"
                      class="input text-sm"
                      placeholder="Weight"
                      step="0.05"
                      min="0"
                      max="1"
                      value={entry.weight}
                      oninput={(e) => updateModelEntry(index, 'weight', parseFloat((e.target as HTMLInputElement).value) || 0)}
                    />
                  </div>
                {/if}

                <button
                  type="button"
                  class="btn-ghost btn-sm text-red-500 hover:text-red-700 flex-shrink-0"
                  onclick={() => removeModelEntry(index)}
                  disabled={modelEntries.length === 1}
                  title="Remove"
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            {/each}

            {#if currentStrategy?.supports_weights}
              <div class="pt-2 text-xs text-surface-500">
                Total weight: {modelEntries.reduce((s, e) => s + e.weight, 0).toFixed(2)}
              </div>
            {/if}
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
