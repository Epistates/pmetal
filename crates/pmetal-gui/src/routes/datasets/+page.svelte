<script lang="ts">
  import { onMount } from 'svelte';
  import { datasetsStore } from '$lib/stores.svelte';
  import { downloadDataset } from '$lib/api';
  import type { DatasetSearchResult } from '$lib/api';
  import { formatBytes } from '$lib/utils';

  // Local state for UI
  let hubQuery = $state('');
  let downloadingId = $state<string | null>(null);
  let downloadError = $state<string | null>(null);
  let downloadSuccess = $state<string | null>(null);

  // Stores
  let cached = $derived(datasetsStore.cached);
  let searchResults = $derived(datasetsStore.searchResults);
  let trending = $derived(datasetsStore.trending);
  let searching = $derived(datasetsStore.searching);
  let searchQuery = $derived(datasetsStore.searchQuery);
  let loading = $derived(datasetsStore.loading);

  // Show trending when no active search
  let showTrending = $derived(!searchQuery && searchResults.length === 0);
  let noResults = $derived(!!searchQuery && searchResults.length === 0 && !searching);

  onMount(() => {
    // Refresh cached datasets on page visit
    datasetsStore.refreshCached();
    if (trending.length === 0) {
      datasetsStore.loadTrending();
    }
  });

  async function handleSearch() {
    if (!hubQuery.trim()) return;
    await datasetsStore.search(hubQuery.trim());
  }

  function handleClearSearch() {
    hubQuery = '';
    datasetsStore.clearSearch();
  }

  async function handleDownload(datasetId: string) {
    downloadError = null;
    downloadSuccess = null;
    downloadingId = datasetId;
    try {
      await downloadDataset(datasetId);
      downloadSuccess = `Dataset "${datasetId}" downloaded successfully.`;
      await datasetsStore.refreshCached();
    } catch (e) {
      downloadError = e instanceof Error ? e.message : String(e);
    } finally {
      downloadingId = null;
    }
  }

  function handleSearchKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter') handleSearch();
  }

  function detectFormat(name: string): string {
    const n = name.toLowerCase();
    if (n.includes('alpaca')) return 'Alpaca';
    if (n.includes('sharegpt') || n.includes('chat')) return 'ShareGPT';
    if (n.includes('preference') || n.includes('dpo') || n.includes('rlhf')) return 'Preference';
    if (n.includes('code')) return 'Code';
    if (n.includes('math') || n.includes('reason')) return 'Math/Reasoning';
    return 'General';
  }
</script>

<div class="space-y-6">
  <!-- Header -->
  <div>
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Datasets</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Browse, download, and manage training datasets from HuggingFace</p>
  </div>

  <!-- Status messages -->
  {#if downloadSuccess}
    <div class="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-sm flex items-center justify-between" role="status">
      <span>{downloadSuccess}</span>
      <button class="text-green-600 hover:text-green-800 dark:text-green-400" aria-label="Dismiss" onclick={() => downloadSuccess = null}>
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  {/if}
  {#if downloadError}
    <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm flex items-center justify-between" role="alert">
      <span>{downloadError}</span>
      <button class="text-red-600 hover:text-red-800 dark:text-red-400" aria-label="Dismiss" onclick={() => downloadError = null}>
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  {/if}

  <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
    <!-- Left: Cached datasets -->
    <div class="xl:col-span-1">
      <div class="card">
        <div class="card-header flex items-center justify-between">
          <h2 class="font-semibold text-surface-900 dark:text-surface-100">Cached Datasets ({cached.length})</h2>
          <button
            class="btn-ghost btn-sm"
            aria-label="Refresh cached datasets"
            onclick={() => datasetsStore.refreshCached()}
          >
            <svg class="w-4 h-4 {loading ? 'animate-spin' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>

        {#if cached.length === 0}
          <div class="card-body text-center py-8">
            <svg class="w-12 h-12 mx-auto text-surface-300 dark:text-surface-600 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
            </svg>
            <p class="text-surface-500 dark:text-surface-400 text-sm">No datasets cached yet</p>
            <p class="text-surface-400 dark:text-surface-500 text-xs mt-1">Search and download from HuggingFace</p>
          </div>
        {:else}
          <div class="divide-y divide-surface-200 dark:divide-surface-700 max-h-[500px] overflow-y-auto scrollbar-thin">
            {#each cached as ds}
              <div class="px-6 py-4">
                <div class="flex items-start justify-between gap-2">
                  <div class="min-w-0 flex-1">
                    <p class="font-medium text-surface-900 dark:text-surface-100 text-sm truncate">{ds.name}</p>
                    <p class="text-xs text-surface-500 mt-0.5 truncate" title={ds.path}>{ds.path}</p>
                  </div>
                  <div class="text-right flex-shrink-0">
                    <p class="text-xs font-medium text-surface-700 dark:text-surface-300">{ds.size_formatted}</p>
                    <span class="badge-neutral text-xs mt-0.5">{detectFormat(ds.name)}</span>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <!-- Right: HuggingFace search -->
    <div class="xl:col-span-2 space-y-4">
      <!-- Search bar -->
      <div class="card">
        <div class="card-header">
          <h2 class="font-semibold text-surface-900 dark:text-surface-100">Search HuggingFace Datasets</h2>
        </div>
        <div class="card-body space-y-4">
          <div class="flex gap-2">
            <label for="dataset-search" class="sr-only">Search datasets</label>
            <input
              id="dataset-search"
              type="search"
              class="input flex-1"
              placeholder="Search datasets (e.g. alpaca, openhermes, math, code)..."
              bind:value={hubQuery}
              onkeydown={handleSearchKeydown}
            />
            {#if searchQuery}
              <button class="btn-secondary btn-sm" aria-label="Clear search" onclick={handleClearSearch}>
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            {/if}
            <button class="btn-primary" disabled={searching || !hubQuery.trim()} onclick={handleSearch}>
              {#if searching}
                <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
              {:else}
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              {/if}
              Search
            </button>
          </div>

          <!-- No results state -->
          {#if noResults}
            <div class="py-8 text-center">
              <svg class="w-10 h-10 mx-auto text-surface-300 dark:text-surface-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <p class="text-surface-500 dark:text-surface-400 text-sm">No datasets found for "{searchQuery}"</p>
              <p class="text-surface-400 dark:text-surface-500 text-xs mt-1">Try a broader search term</p>
            </div>
          {/if}

          <!-- Search results -->
          {#if searchResults.length > 0}
            <div class="space-y-2" role="list" aria-label="Dataset search results">
              {#each searchResults as dataset}
                <div class="flex items-center gap-3 p-3 rounded-lg bg-surface-50 dark:bg-surface-700/50 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors" role="listitem">
                  <div class="flex-1 min-w-0">
                    <p class="font-medium text-surface-900 dark:text-surface-100 text-sm truncate">{dataset.id}</p>
                    <div class="flex items-center gap-2 text-xs text-surface-500 mt-0.5 flex-wrap">
                      <span>{dataset.downloads_formatted} downloads</span>
                      <span>{dataset.likes} likes</span>
                      {#each dataset.tags.slice(0, 3) as tag}
                        <span class="badge-neutral">{tag}</span>
                      {/each}
                    </div>
                    {#if dataset.description}
                      <p class="text-xs text-surface-500 mt-1 line-clamp-1">{dataset.description}</p>
                    {/if}
                  </div>
                  <button
                    class="btn-primary btn-sm flex-shrink-0"
                    disabled={downloadingId === dataset.id}
                    aria-label="Download {dataset.id}"
                    onclick={() => handleDownload(dataset.id)}
                  >
                    {#if downloadingId === dataset.id}
                      <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
                    {:else}
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                    {/if}
                  </button>
                </div>
              {/each}
            </div>
          {/if}

          <!-- Trending (shown when no search active) -->
          {#if showTrending && trending.length > 0}
            <div>
              <p class="section-title">Trending Datasets</p>
              <div class="space-y-2" role="list" aria-label="Trending datasets">
                {#each trending as dataset}
                  <div class="flex items-center gap-3 p-3 rounded-lg bg-surface-50 dark:bg-surface-700/50 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors" role="listitem">
                    <div class="flex-1 min-w-0">
                      <p class="font-medium text-surface-900 dark:text-surface-100 text-sm truncate">{dataset.id}</p>
                      <div class="flex items-center gap-2 text-xs text-surface-500 mt-0.5 flex-wrap">
                        <span>{dataset.downloads_formatted} downloads</span>
                        <span>{dataset.likes} likes</span>
                        {#each dataset.tags.slice(0, 2) as tag}
                          <span class="badge-neutral">{tag}</span>
                        {/each}
                      </div>
                      {#if dataset.description}
                        <p class="text-xs text-surface-500 mt-1 line-clamp-1">{dataset.description}</p>
                      {/if}
                    </div>
                    <button
                      class="btn-secondary btn-sm flex-shrink-0"
                      disabled={downloadingId === dataset.id}
                      aria-label="Download {dataset.id}"
                      onclick={() => handleDownload(dataset.id)}
                    >
                      {#if downloadingId === dataset.id}
                        <div class="w-4 h-4 border-2 border-surface-500 border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
                      {:else}
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                      {/if}
                    </button>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      </div>

      <!-- Info box about dataset formats -->
      <div class="card">
        <div class="card-header">
          <h3 class="font-semibold text-surface-900 dark:text-surface-100">Supported Formats</h3>
        </div>
        <div class="card-body">
          <div class="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            {#each [
              { format: 'Alpaca', desc: 'instruction / input / output fields', method: 'SFT, LoRA' },
              { format: 'ShareGPT', desc: 'conversations array with role/value', method: 'SFT, LoRA, DPO' },
              { format: 'HuggingFace', desc: 'Auto-detect from dataset card', method: 'All methods' },
              { format: 'JSONL', desc: 'One JSON object per line with text field', method: 'SFT, LoRA' },
              { format: 'Preference', desc: 'prompt / chosen / rejected fields', method: 'DPO, SimPO, ORPO, KTO' },
              { format: 'GRPO', desc: 'prompt field, rewards generated online', method: 'GRPO' },
            ] as fmt}
              <div class="p-3 rounded-lg bg-surface-50 dark:bg-surface-700/50">
                <p class="font-semibold text-surface-800 dark:text-surface-200">{fmt.format}</p>
                <p class="text-xs text-surface-500 mt-0.5 leading-relaxed">{fmt.desc}</p>
                <span class="badge-neutral text-xs mt-1">{fmt.method}</span>
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
