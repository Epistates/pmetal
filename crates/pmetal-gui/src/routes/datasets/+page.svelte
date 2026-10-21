<script lang="ts">
  import { onMount } from 'svelte';
  import { datasetsStore, downloadsStore } from '#lib/stores.svelte.js';
  import { errorMessage } from '#lib/api.js';
  import { formatBytes, formatEta, getStatusBadgeClass } from '#lib/utils.js';

  let query = $state('');
  let downloadError = $state<string | null>(null);

  const showTrending = $derived(!datasetsStore.searchQuery && datasetsStore.searchResults.length === 0);
  const shown = $derived(showTrending ? datasetsStore.trending : datasetsStore.searchResults);
  const cachedNames = $derived(new Set(datasetsStore.cached.map((d) => d.name)));
  const datasetDownloads = $derived(downloadsStore.items.filter((d) => d.kind === 'dataset'));

  onMount(() => {
    void datasetsStore.refreshCached();
    if (datasetsStore.trending.length === 0) void datasetsStore.loadTrending();
  });

  async function search(e: Event) {
    e.preventDefault();
    await datasetsStore.search(query);
  }

  async function download(id: string) {
    downloadError = null;
    try {
      await downloadsStore.startDataset(id);
    } catch (e) {
      downloadError = errorMessage(e);
    }
  }
</script>

<div class="space-y-6">
  <div>
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Datasets</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">
      Datasets in the Hugging Face cache, and the Hub. To train on a file of your own, give its path in the training form.
    </p>
  </div>

  {#if downloadError}<p role="alert" class="text-sm text-red-600 dark:text-red-400">{downloadError}</p>{/if}

  {#if datasetDownloads.length > 0}
    <section class="card" aria-label="Downloads">
      <ul class="divide-y divide-surface-200 dark:divide-surface-700">
        {#each datasetDownloads as d (d.id)}
          <li class="px-6 py-3">
            <div class="flex items-center justify-between gap-3">
              <span class="truncate text-sm font-medium">{d.model_id}</span>
              <span class="flex items-center gap-2">
                <span class={getStatusBadgeClass(d.state)}>{d.state}</span>
                {#if d.state === 'queued' || d.state === 'downloading'}
                  <button class="btn-ghost btn-sm" onclick={() => downloadsStore.cancel(d.id)}>Cancel</button>
                {/if}
              </span>
            </div>
            {#if d.state === 'downloading' && d.total_bytes > 0}
              <div class="progress-bar mt-2" role="progressbar" aria-label={`Downloading ${d.model_id}`} aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round((d.bytes_done / d.total_bytes) * 100)}>
                <div class="progress-bar-fill" style={`width: ${(d.bytes_done / d.total_bytes) * 100}%`}></div>
              </div>
              <p class="mt-1 text-xs text-surface-500">{formatBytes(d.bytes_done)} of {formatBytes(d.total_bytes)}{#if d.eta_seconds} · {formatEta(d.eta_seconds)} left{/if}</p>
            {:else if d.state === 'failed'}
              <p role="alert" class="mt-1 text-xs text-red-600 dark:text-red-400">{d.error}</p>
            {/if}
          </li>
        {/each}
      </ul>
    </section>
  {/if}

  <div class="grid grid-cols-1 gap-6 xl:grid-cols-2">
    <section class="card self-start" aria-label="Datasets on this Mac">
      <div class="card-header"><h2 class="font-semibold">On this Mac</h2></div>
      {#if datasetsStore.loading && datasetsStore.cached.length === 0}
        <p class="p-6 text-surface-500" role="status">Looking…</p>
      {:else if datasetsStore.cached.length === 0}
        <p class="p-6 text-sm text-surface-500">No datasets downloaded yet. Find one on the right.</p>
      {:else}
        <ul class="divide-y divide-surface-200 dark:divide-surface-700">
          {#each datasetsStore.cached as dataset (dataset.path)}
            <li class="px-6 py-3">
              <p class="text-sm font-medium">{dataset.name}</p>
              <p class="text-xs text-surface-500">{dataset.size_formatted}</p>
              <p class="break-all font-mono text-xs text-surface-400">{dataset.path}</p>
            </li>
          {/each}
        </ul>
      {/if}
    </section>

    <section class="card self-start" aria-label="Hugging Face datasets">
      <div class="card-header"><h2 class="font-semibold">Hugging Face</h2></div>
      <div class="card-body space-y-4">
        <form class="flex gap-2" onsubmit={search}>
          <label for="dataset-query" class="sr-only">Search datasets</label>
          <input id="dataset-query" class="input" type="search" placeholder="Search datasets…" bind:value={query} />
          <button class="btn-primary shrink-0" disabled={datasetsStore.searching}>{datasetsStore.searching ? 'Searching…' : 'Search'}</button>
          {#if datasetsStore.searchQuery}
            <button type="button" class="btn-secondary shrink-0" onclick={() => { query = ''; datasetsStore.clearSearch(); }}>Clear</button>
          {/if}
        </form>
        {#if datasetsStore.error}<p role="alert" class="text-sm text-red-600 dark:text-red-400">{datasetsStore.error}</p>{/if}
        <p class="section-title">{showTrending ? 'Trending' : `Results for “${datasetsStore.searchQuery}”`}</p>
        {#if shown.length === 0 && !datasetsStore.searching}
          <p class="text-sm text-surface-500">{showTrending ? 'Nothing to show; check the internet connection.' : 'Nothing found.'}</p>
        {/if}
        <ul class="space-y-2">
          {#each shown as result (result.id)}
            <li class="flex items-center justify-between gap-2 rounded-lg bg-surface-50 p-3 dark:bg-surface-700/40">
              <div class="min-w-0">
                <p class="truncate text-sm font-medium">{result.id}</p>
                <p class="text-xs text-surface-500">{result.downloads_formatted} downloads · {result.likes} likes</p>
                {#if result.description}<p class="mt-1 line-clamp-2 text-xs text-surface-500">{result.description}</p>{/if}
              </div>
              {#if cachedNames.has(result.id)}
                <span class="badge-success shrink-0">Downloaded</span>
              {:else}
                <button class="btn-secondary btn-sm shrink-0" onclick={() => download(result.id)}>Download</button>
              {/if}
            </li>
          {/each}
        </ul>
      </div>
    </section>
  </div>
</div>
