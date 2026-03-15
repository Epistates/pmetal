<script lang="ts">
  import { onMount } from 'svelte';
  import { modelsStore } from '$lib/stores.svelte';
  import { getModelInfo, getModelFit, searchHubModels, getTrendingModels } from '$lib/api';
  import type { CachedModel, ModelInfo, ModelFitInfo, HubSearchResult } from '$lib/api';

  // Cached models
  let models = $derived(modelsStore.models);
  let loading = $derived(modelsStore.loading);
  let modelsError = $derived(modelsStore.error);

  // Search
  let searchQuery = $state('');
  let filteredModels = $derived(
    searchQuery
      ? models.filter(m =>
          m.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
          (m.model_type?.toLowerCase().includes(searchQuery.toLowerCase()) ?? false)
        )
      : models
  );

  // Download modal
  let showDownloadModal = $state(false);
  let modelIdInput = $state('');
  let revisionInput = $state('main');
  let isDownloading = $state(false);
  let downloadError = $state<string | null>(null);

  // Hub search
  let hubQuery = $state('');
  let hubResults = $state<HubSearchResult[]>([]);
  let hubSearching = $state(false);
  let hubSearchError = $state<string | null>(null);
  let hubSearchDone = $state(false); // true once a search has completed
  let trendingModels = $state<HubSearchResult[]>([]);

  // Model details
  let selectedModelId = $state<string | null>(null);
  let selectedModelInfo = $state<ModelInfo | null>(null);
  let loadingModelInfo = $state(false);
  let fitData = $state<Record<string, ModelFitInfo>>({});

  // Delete
  let deleteConfirmModel = $state<CachedModel | null>(null);
  let isDeleting = $state(false);

  onMount(async () => {
    try {
      trendingModels = await getTrendingModels(8);
    } catch (e) {
      console.error('Failed to load trending models:', e);
    }
  });

  async function loadFitData(modelId: string) {
    if (fitData[modelId]) return;
    try {
      const fit = await getModelFit(modelId);
      fitData = { ...fitData, [modelId]: fit };
    } catch (e) {
      console.error(`Failed to get fit data for ${modelId}:`, e);
    }
  }

  async function selectModel(modelId: string) {
    if (selectedModelId === modelId) {
      selectedModelId = null;
      selectedModelInfo = null;
      return;
    }
    selectedModelId = modelId;
    loadingModelInfo = true;
    selectedModelInfo = null;
    try {
      selectedModelInfo = await getModelInfo(modelId);
    } catch (e) {
      console.error('Failed to load model info:', e);
    } finally {
      loadingModelInfo = false;
    }
    loadFitData(modelId);
  }

  async function handleDownload(e: Event) {
    e.preventDefault();
    if (!modelIdInput.trim()) return;
    isDownloading = true;
    downloadError = null;
    try {
      await modelsStore.download(modelIdInput.trim(), revisionInput || undefined);
      showDownloadModal = false;
      modelIdInput = '';
      revisionInput = 'main';
    } catch (e) {
      downloadError = e instanceof Error ? e.message : String(e);
    } finally {
      isDownloading = false;
    }
  }

  async function handleHubSearch() {
    if (!hubQuery.trim()) return;
    hubSearching = true;
    hubSearchError = null;
    hubSearchDone = false;
    try {
      hubResults = await searchHubModels(hubQuery.trim(), 20);
      hubSearchDone = true;
    } catch (e) {
      hubSearchError = e instanceof Error ? e.message : String(e);
    } finally {
      hubSearching = false;
    }
  }

  async function handleDelete() {
    if (!deleteConfirmModel) return;
    // Capture reference before async ops to avoid null race
    const modelToDelete = deleteConfirmModel;
    isDeleting = true;
    try {
      await modelsStore.delete(modelToDelete.id);
      if (selectedModelId === modelToDelete.id) {
        selectedModelId = null;
        selectedModelInfo = null;
      }
      deleteConfirmModel = null;
    } catch (e) {
      console.error('Failed to delete model:', e);
    } finally {
      isDeleting = false;
    }
  }

  function handleDownloadModalKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') showDownloadModal = false;
  }

  function handleDeleteModalKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') deleteConfirmModel = null;
  }

  function getFitColor(fit: string): string {
    switch (fit) {
      case 'fits': return 'text-green-600 dark:text-green-400';
      case 'tight': return 'text-yellow-600 dark:text-yellow-400';
      case 'too_large': return 'text-red-600 dark:text-red-400';
      default: return 'text-surface-500';
    }
  }

  function getFitLabel(fit: string): string {
    switch (fit) {
      case 'fits': return 'Fits';
      case 'tight': return 'Tight';
      case 'too_large': return 'Too large';
      default: return fit;
    }
  }
</script>

<div class="space-y-6">
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div>
      <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Models</h1>
      <p class="text-surface-500 dark:text-surface-400 mt-1">Manage cached models and download from HuggingFace</p>
    </div>
    <button class="btn-primary" onclick={() => (showDownloadModal = true)}>
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
      </svg>
      Download Model
    </button>
  </div>

  <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
    <!-- Left: Cached models list -->
    <div class="xl:col-span-2 space-y-4">
      <!-- Search cached -->
      <div class="card">
        <div class="card-header flex items-center justify-between">
          <h2 class="font-semibold text-surface-900 dark:text-surface-100">Cached Models ({models.length})</h2>
          <button
            class="btn-ghost btn-sm"
            aria-label="Refresh models list"
            onclick={() => modelsStore.refresh()}
          >
            <svg class="w-4 h-4 {loading ? 'animate-spin' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
        <div class="px-6 py-3 border-b border-surface-200 dark:border-surface-700">
          <label for="model-search" class="sr-only">Search cached models</label>
          <input
            id="model-search"
            type="search"
            class="input"
            placeholder="Search cached models..."
            bind:value={searchQuery}
          />
        </div>

        {#if modelsError}
          <div class="p-4 text-sm text-red-600 dark:text-red-400" role="alert">{modelsError}</div>
        {:else if filteredModels.length === 0}
          <div class="p-8 text-center">
            <svg class="w-12 h-12 mx-auto text-surface-300 dark:text-surface-600 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <p class="text-surface-500 dark:text-surface-400">
              {searchQuery ? `No models matching "${searchQuery}"` : 'No models downloaded yet'}
            </p>
            {#if !searchQuery}
              <button class="btn-primary btn-sm mt-3" onclick={() => (showDownloadModal = true)}>
                Download your first model
              </button>
            {/if}
          </div>
        {:else}
          <div class="divide-y divide-surface-200 dark:divide-surface-700">
            {#each filteredModels as model}
              <div
                class="flex items-center gap-4 px-6 py-4 hover:bg-surface-50 dark:hover:bg-surface-700/50 transition-colors cursor-pointer {selectedModelId === model.id ? 'bg-primary-50 dark:bg-primary-900/20' : ''}"
                role="button"
                tabindex="0"
                aria-pressed={selectedModelId === model.id}
                onclick={() => selectModel(model.id)}
                onkeydown={(e) => e.key === 'Enter' && selectModel(model.id)}
              >
                <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-accent-400 to-orange-600 flex items-center justify-center flex-shrink-0" aria-hidden="true">
                  <span class="text-white font-bold text-sm">{model.id.charAt(0).toUpperCase()}</span>
                </div>
                <div class="flex-1 min-w-0">
                  <p class="font-medium text-surface-900 dark:text-surface-100 truncate">{model.id}</p>
                  <div class="flex items-center gap-3 text-xs text-surface-500 mt-0.5">
                    <span>{model.size_formatted}</span>
                    {#if model.model_type}
                      <span class="badge-neutral text-xs">{model.model_type}</span>
                    {/if}
                  </div>
                </div>
                {#if fitData[model.id]}
                  <div class="text-xs {getFitColor(fitData[model.id].inference_fit)} font-medium" aria-label="Memory fit: {getFitLabel(fitData[model.id].inference_fit)}">
                    {getFitLabel(fitData[model.id].inference_fit)}
                  </div>
                {/if}
                <button
                  class="btn-ghost btn-sm text-red-500 hover:text-red-700 dark:hover:text-red-400"
                  aria-label="Delete {model.id}"
                  onclick={(e) => { e.stopPropagation(); deleteConfirmModel = model; }}
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                </button>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <!-- HuggingFace Search -->
      <div class="card">
        <div class="card-header">
          <h2 class="font-semibold text-surface-900 dark:text-surface-100">Search HuggingFace</h2>
        </div>
        <div class="card-body space-y-4">
          <form onsubmit={(e) => { e.preventDefault(); handleHubSearch(); }} class="flex gap-2">
            <label for="hub-search" class="sr-only">Search HuggingFace models</label>
            <input
              id="hub-search"
              type="search"
              class="input flex-1"
              placeholder="Search models (e.g. Qwen, Llama, Mistral)..."
              bind:value={hubQuery}
            />
            <button type="submit" class="btn-secondary" aria-label="Search" disabled={hubSearching}>
              {#if hubSearching}
                <div class="w-4 h-4 border-2 border-surface-500 border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
              {:else}
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              {/if}
            </button>
          </form>

          {#if hubSearchError}
            <p class="text-sm text-red-600 dark:text-red-400" role="alert">{hubSearchError}</p>
          {/if}

          <!-- Hub results -->
          {#if hubResults.length > 0}
            <div class="space-y-2" role="list" aria-label="Search results">
              {#each hubResults as result}
                <div class="flex items-center gap-3 p-3 rounded-lg bg-surface-50 dark:bg-surface-700/50 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors" role="listitem">
                  <div class="flex-1 min-w-0">
                    <p class="font-medium text-surface-900 dark:text-surface-100 text-sm truncate">{result.id}</p>
                    <div class="flex items-center gap-2 text-xs text-surface-500 mt-0.5">
                      <span>{result.downloads_formatted} downloads</span>
                      <span>{result.likes} likes</span>
                      {#if result.pipeline_tag}
                        <span class="badge-neutral">{result.pipeline_tag}</span>
                      {/if}
                      {#if result.is_gated}
                        <span class="badge-warning">Gated</span>
                      {/if}
                    </div>
                  </div>
                  <button
                    class="btn-primary btn-sm flex-shrink-0"
                    onclick={() => { modelIdInput = result.id; showDownloadModal = true; }}
                  >
                    Download
                  </button>
                </div>
              {/each}
            </div>
          {:else if hubSearchDone && !hubSearching && hubQuery.trim()}
            <!-- No results state -->
            <div class="py-8 text-center">
              <svg class="w-10 h-10 mx-auto text-surface-300 dark:text-surface-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <p class="text-surface-500 dark:text-surface-400 text-sm">No models found for "{hubQuery}"</p>
              <p class="text-surface-400 dark:text-surface-500 text-xs mt-1">Try a different search term or browse trending models below</p>
            </div>
          {:else if !hubSearching && !hubSearchDone && trendingModels.length > 0}
            <div>
              <p class="section-title">Trending Models</p>
              <div class="space-y-2" role="list" aria-label="Trending models">
                {#each trendingModels as model}
                  <div class="flex items-center gap-3 p-3 rounded-lg bg-surface-50 dark:bg-surface-700/50" role="listitem">
                    <div class="flex-1 min-w-0">
                      <p class="font-medium text-surface-900 dark:text-surface-100 text-sm truncate">{model.id}</p>
                      <div class="flex items-center gap-2 text-xs text-surface-500 mt-0.5">
                        <span>{model.downloads_formatted} downloads</span>
                        {#if model.pipeline_tag}
                          <span class="badge-neutral">{model.pipeline_tag}</span>
                        {/if}
                      </div>
                    </div>
                    <button
                      class="btn-secondary btn-sm"
                      onclick={() => { modelIdInput = model.id; showDownloadModal = true; }}
                    >
                      Download
                    </button>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      </div>
    </div>

    <!-- Right: Model details -->
    <div class="xl:col-span-1">
      {#if selectedModelId}
        <div class="card sticky top-0">
          <div class="card-header flex items-center justify-between">
            <h3 class="font-semibold text-surface-900 dark:text-surface-100">Model Details</h3>
            <button
              class="btn-ghost btn-sm"
              aria-label="Close model details"
              onclick={() => { selectedModelId = null; selectedModelInfo = null; }}
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div class="card-body space-y-4">
            {#if loadingModelInfo}
              <div class="flex justify-center py-8">
                <div class="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" aria-label="Loading model info..."></div>
              </div>
            {:else if selectedModelInfo}
              <div>
                <p class="text-xs text-surface-500 mb-1">Model ID</p>
                <p class="font-medium text-surface-900 dark:text-surface-100 text-sm break-all">{selectedModelInfo.id}</p>
              </div>
              <div class="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p class="text-xs text-surface-500">Size</p>
                  <p class="font-medium">{selectedModelInfo.size_formatted}</p>
                </div>
                {#if selectedModelInfo.model_type}
                  <div>
                    <p class="text-xs text-surface-500">Architecture</p>
                    <p class="font-medium">{selectedModelInfo.model_type}</p>
                  </div>
                {/if}
                {#if selectedModelInfo.num_layers}
                  <div>
                    <p class="text-xs text-surface-500">Layers</p>
                    <p class="font-medium">{selectedModelInfo.num_layers}</p>
                  </div>
                {/if}
                {#if selectedModelInfo.hidden_size}
                  <div>
                    <p class="text-xs text-surface-500">Hidden Size</p>
                    <p class="font-medium">{selectedModelInfo.hidden_size}</p>
                  </div>
                {/if}
                {#if selectedModelInfo.vocab_size}
                  <div>
                    <p class="text-xs text-surface-500">Vocab Size</p>
                    <p class="font-medium">{selectedModelInfo.vocab_size.toLocaleString()}</p>
                  </div>
                {/if}
                {#if selectedModelInfo.context_length}
                  <div>
                    <p class="text-xs text-surface-500">Context Length</p>
                    <p class="font-medium">{selectedModelInfo.context_length.toLocaleString()}</p>
                  </div>
                {/if}
              </div>

              <!-- Fit info -->
              {#if fitData[selectedModelId]}
                {@const fit = fitData[selectedModelId]}
                <div class="pt-3 border-t border-surface-200 dark:border-surface-700">
                  <p class="section-title">Memory Fit</p>
                  <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                      <span class="text-surface-500">Weights</span>
                      <span class="font-medium">{fit.weights_gb.toFixed(1)} GB</span>
                    </div>
                    <div class="flex justify-between">
                      <span class="text-surface-500">Inference</span>
                      <span class="font-medium {getFitColor(fit.inference_fit)}">
                        {fit.inference_memory_gb.toFixed(1)} GB · {getFitLabel(fit.inference_fit)}
                      </span>
                    </div>
                    <div class="flex justify-between">
                      <span class="text-surface-500">Training</span>
                      <span class="font-medium {getFitColor(fit.training_fit)}">
                        {fit.training_memory_gb.toFixed(1)} GB · {getFitLabel(fit.training_fit)}
                      </span>
                    </div>
                    {#if fit.estimated_tps}
                      <div class="flex justify-between">
                        <span class="text-surface-500">Est. Tok/s</span>
                        <span class="font-medium text-accent-600 dark:text-accent-400">{fit.estimated_tps.toFixed(0)}</span>
                      </div>
                    {/if}
                    <div class="flex justify-between">
                      <span class="text-surface-500">Rec. Batch Size</span>
                      <span class="font-medium">{fit.recommended_batch_size}</span>
                    </div>
                  </div>
                </div>
              {/if}

              <!-- Actions -->
              <div class="pt-3 border-t border-surface-200 dark:border-surface-700 flex flex-col gap-2">
                <a href="/inference?model={encodeURIComponent(selectedModelId)}" class="btn-primary btn-sm w-full text-center">
                  Run Inference
                </a>
                <a href="/training?model={encodeURIComponent(selectedModelId)}" class="btn-secondary btn-sm w-full text-center">
                  Fine-tune
                </a>
              </div>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  </div>
</div>

<!-- Download Modal -->
{#if showDownloadModal}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" role="presentation" onkeydown={handleDownloadModalKeydown}>
    <div
      class="card w-full max-w-md"
      role="dialog"
      aria-modal="true"
      aria-labelledby="download-modal-title"
      tabindex="-1"
    >
      <div class="card-header flex items-center justify-between">
        <h3 id="download-modal-title" class="font-semibold text-surface-900 dark:text-surface-100">Download Model</h3>
        <button class="btn-ghost btn-sm" aria-label="Close dialog" onclick={() => (showDownloadModal = false)}>
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <form onsubmit={handleDownload} class="card-body space-y-4">
        <div>
          <label class="label" for="model-id">HuggingFace Model ID</label>
          <input
            id="model-id"
            type="text"
            class="input"
            placeholder="e.g. Qwen/Qwen3-8B"
            bind:value={modelIdInput}
          />
          <p class="text-xs text-surface-500 mt-1">Format: author/model-name</p>
        </div>
        <div>
          <label class="label" for="revision">Revision / Branch</label>
          <input id="revision" type="text" class="input" placeholder="main" bind:value={revisionInput} />
        </div>

        {#if downloadError}
          <div class="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-sm" role="alert">
            {downloadError}
          </div>
        {/if}

        {#if isDownloading}
          <div class="p-3 rounded-lg bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 text-sm flex items-center gap-2" role="status">
            <div class="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
            Downloading {modelIdInput}...
          </div>
        {/if}

        <div class="flex gap-3">
          <button type="submit" class="btn-primary flex-1" disabled={isDownloading || !modelIdInput.trim()}>
            {isDownloading ? 'Downloading...' : 'Download'}
          </button>
          <button type="button" class="btn-secondary" onclick={() => (showDownloadModal = false)}>Cancel</button>
        </div>
      </form>
    </div>
  </div>
{/if}

<!-- Delete Confirmation Modal -->
{#if deleteConfirmModel}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" role="presentation" onkeydown={handleDeleteModalKeydown}>
    <div
      class="card w-full max-w-sm"
      role="dialog"
      aria-modal="true"
      aria-labelledby="delete-modal-title"
      tabindex="-1"
    >
      <div class="card-body space-y-4">
        <div class="text-center">
          <div class="w-12 h-12 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center mx-auto mb-3" aria-hidden="true">
            <svg class="w-6 h-6 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </div>
          <h3 id="delete-modal-title" class="font-semibold text-surface-900 dark:text-surface-100">Delete Model</h3>
          <p class="text-sm text-surface-500 mt-1">Are you sure you want to delete <strong>{deleteConfirmModel.id}</strong>? This cannot be undone.</p>
        </div>
        <div class="flex gap-3">
          <button
            class="btn-danger flex-1"
            onclick={handleDelete}
            disabled={isDeleting}
          >
            {isDeleting ? 'Deleting...' : 'Delete'}
          </button>
          <button class="btn-secondary flex-1" onclick={() => (deleteConfirmModel = null)}>Cancel</button>
        </div>
      </div>
    </div>
  </div>
{/if}
