<script lang="ts">
  import { onMount } from 'svelte';
  import { downloadsStore, modelsStore } from '#lib/stores.svelte.js';
  import {
    errorMessage,
    getModelFit,
    getModelInfo,
    getTrendingModels,
    pickDirectory,
    searchHubModels,
  } from '#lib/api.js';
  import type { CachedModel, HubSearchResult, ModelFitInfo, ModelInfo } from '#lib/api.js';
  import { formatBytes, formatEta, getStatusBadgeClass } from '#lib/utils.js';
  import ConfirmDialog from '#lib/components/ConfirmDialog.svelte';

  let filter = $state('');
  const models = $derived(
    modelsStore.models.filter(
      (m) =>
        !filter.trim() ||
        m.id.toLowerCase().includes(filter.trim().toLowerCase()) ||
        m.model_type.includes(filter.trim().toLowerCase()),
    ),
  );

  // Details of the model selected.
  let selected = $state<CachedModel | null>(null);
  let info = $state<ModelInfo | null>(null);
  let fit = $state<ModelFitInfo | null>(null);
  let detailError = $state<string | null>(null);

  async function select(model: CachedModel) {
    selected = model;
    info = null;
    fit = null;
    detailError = null;
    try {
      [info, fit] = await Promise.all([
        getModelInfo(model.id),
        getModelFit(model.id).catch(() => null),
      ]);
    } catch (e) {
      detailError = errorMessage(e);
    }
  }

  // Hub search.
  let query = $state('');
  let results = $state<HubSearchResult[]>([]);
  let trending = $state<HubSearchResult[]>([]);
  let searching = $state(false);
  let searched = $state(false);
  let searchError = $state<string | null>(null);

  onMount(async () => {
    try {
      trending = await getTrendingModels(8);
    } catch {
      // Being offline is not worth a banner; search says so when it is tried.
    }
  });

  async function search(e: Event) {
    e.preventDefault();
    if (!query.trim()) {
      results = [];
      searched = false;
      return;
    }
    searching = true;
    searchError = null;
    try {
      results = await searchHubModels(query.trim(), 20);
      searched = true;
    } catch (err) {
      searchError = errorMessage(err);
    } finally {
      searching = false;
    }
  }

  const shownHub = $derived(searched ? results : trending);

  // Download.
  let downloadId = $state('');
  let downloadError = $state<string | null>(null);
  const downloaded = $derived(new Set(modelsStore.models.map((m) => m.id)));

  async function download(id: string) {
    downloadError = null;
    try {
      await downloadsStore.startModel(id.trim());
      downloadId = '';
    } catch (e) {
      downloadError = errorMessage(e);
    }
  }

  // Folders.
  let folderError = $state<string | null>(null);
  async function addFolder() {
    folderError = null;
    try {
      const path = await pickDirectory('Choose a folder that holds models');
      if (path) await modelsStore.addDirectory(path);
    } catch (e) {
      folderError = errorMessage(e);
    }
  }

  // Delete.
  let toDelete = $state<CachedModel | null>(null);
  let deleting = $state(false);
  let deleteError = $state<string | null>(null);
  async function confirmDelete() {
    if (!toDelete) return;
    deleting = true;
    deleteError = null;
    try {
      await modelsStore.delete(toDelete.id);
      if (selected?.id === toDelete.id) selected = null;
      toDelete = null;
    } catch (e) {
      deleteError = errorMessage(e);
    } finally {
      deleting = false;
    }
  }

  const SOURCE = { hf_cache: 'Hugging Face', trained: 'Trained', custom: 'Added folder' } as const;
  const FIT = {
    fits: ['badge-success', 'Fits'],
    tight: ['badge-warning', 'Tight'],
    too_large: ['badge-danger', 'Too large'],
  } as const;
</script>

<div class="space-y-6">
  <div class="flex flex-wrap items-center justify-between gap-3">
    <div>
      <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Models</h1>
      <p class="text-surface-500 dark:text-surface-400 mt-1">
        {modelsStore.models.length} on this Mac, from the Hugging Face cache, your training runs and folders you added.
      </p>
    </div>
    <div class="flex gap-2">
      <button class="btn-secondary" onclick={addFolder}>Add a folder…</button>
      <button class="btn-secondary" onclick={() => modelsStore.refresh()} disabled={modelsStore.loading}>
        {modelsStore.loading ? 'Scanning…' : 'Rescan'}
      </button>
    </div>
  </div>

  {#if folderError || modelsStore.error}
    <p role="alert" class="text-sm text-red-600 dark:text-red-400">{folderError ?? modelsStore.error}</p>
  {/if}

  {#if downloadsStore.items.length > 0}
    <section class="card" aria-label="Downloads">
      <div class="card-header flex items-center justify-between">
        <h2 class="font-semibold">Downloads</h2>
        <button class="btn-ghost btn-sm" onclick={() => downloadsStore.clearFinished()}>Clear finished</button>
      </div>
      <ul class="divide-y divide-surface-200 dark:divide-surface-700">
        {#each downloadsStore.items as d (d.id)}
          <li class="px-6 py-3">
            <div class="flex items-center justify-between gap-3">
              <span class="min-w-0 truncate text-sm font-medium">{d.model_id}</span>
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
              <p class="mt-1 text-xs text-surface-500">
                {formatBytes(d.bytes_done)} of {formatBytes(d.total_bytes)}
                {#if d.bytes_per_sec}· {formatBytes(d.bytes_per_sec)}/s{/if}
                {#if d.eta_seconds}· {formatEta(d.eta_seconds)} left{/if}
              </p>
            {:else if d.state === 'failed'}
              <p role="alert" class="mt-1 text-xs text-red-600 dark:text-red-400">{d.error}</p>
            {:else if d.state === 'completed' && d.path}
              <p class="mt-1 text-xs text-surface-500 break-all">Saved to {d.path}</p>
            {/if}
          </li>
        {/each}
      </ul>
    </section>
  {/if}

  <div class="grid grid-cols-1 gap-6 xl:grid-cols-5">
    <section class="card xl:col-span-3 self-start" aria-label="Models on this Mac">
      <div class="card-header">
        <label for="model-filter" class="sr-only">Filter models</label>
        <input id="model-filter" class="input" type="search" placeholder="Filter models…" bind:value={filter} />
      </div>
      {#if modelsStore.loading && modelsStore.models.length === 0}
        <p class="p-6 text-surface-500" role="status">Looking for models…</p>
      {:else if modelsStore.models.length === 0}
        <div class="p-8 text-center text-surface-500">
          <p class="font-medium text-surface-800 dark:text-surface-200">No models yet.</p>
          <p class="mt-1 text-sm">Download one from the right, or add a folder that has some.</p>
        </div>
      {:else if models.length === 0}
        <p class="p-6 text-sm text-surface-500">No model matches “{filter}”.</p>
      {:else}
        <ul class="divide-y divide-surface-200 dark:divide-surface-700">
          {#each models as model (model.path)}
            <li>
              <button
                class="flex w-full items-center justify-between gap-3 px-6 py-3 text-left hover:bg-surface-50 dark:hover:bg-surface-700/40"
                class:bg-primary-50={selected?.path === model.path}
                aria-current={selected?.path === model.path ? 'true' : undefined}
                onclick={() => select(model)}
              >
                <span class="min-w-0">
                  <span class="block truncate text-sm font-medium">{model.id}</span>
                  <span class="block text-xs text-surface-500">{model.size_formatted} · {model.model_type}</span>
                </span>
                <span class="badge-neutral shrink-0">{SOURCE[model.source]}</span>
              </button>
            </li>
          {/each}
        </ul>
      {/if}
    </section>

    <div class="xl:col-span-2 space-y-6">
      {#if selected}
        <section class="card" aria-label="Model details">
          <div class="card-header"><h2 class="font-semibold break-all">{selected.id}</h2></div>
          <div class="card-body space-y-3 text-sm">
            {#if detailError}<p role="alert" class="text-red-600 dark:text-red-400">{detailError}</p>{/if}
            <dl class="grid grid-cols-2 gap-2">
              <dt class="text-surface-500">Size</dt><dd>{selected.size_formatted}</dd>
              {#if info?.model_type}<dt class="text-surface-500">Architecture</dt><dd>{info.model_type}</dd>{/if}
              {#if info?.num_layers}<dt class="text-surface-500">Layers</dt><dd>{info.num_layers}</dd>{/if}
              {#if info?.hidden_size}<dt class="text-surface-500">Hidden size</dt><dd>{info.hidden_size}</dd>{/if}
              {#if info?.context_length}<dt class="text-surface-500">Context</dt><dd>{info.context_length.toLocaleString()}</dd>{/if}
            </dl>
            <p class="break-all font-mono text-xs text-surface-500">{selected.path}</p>
            {#if fit?.unsupported}
              <p class="text-xs text-red-600 dark:text-red-400" role="alert">
                pmetal cannot run this model: {fit.unsupported}
              </p>
            {/if}
            {#if fit}
              <div class="flex flex-wrap gap-2">
                <span class={FIT[fit.training_fit][0]}>Training: {FIT[fit.training_fit][1]}</span>
                <span class={FIT[fit.inference_fit][0]}>Inference: {FIT[fit.inference_fit][1]}</span>
              </div>
              <p class="text-xs text-surface-500">
                Needs about {fit.inference_memory_gb.toFixed(1)} GB to run and {fit.training_memory_gb.toFixed(1)} GB to train
                of {fit.available_memory_gb.toFixed(0)} GB the GPU can use{fit.estimated_tps ? `; roughly ${fit.estimated_tps.toFixed(0)} tokens/s` : ''}.
              </p>
            {/if}
            <div class="flex flex-wrap gap-2 pt-2">
              <a class="btn-primary btn-sm" href={`/run/train?model=${encodeURIComponent(selected.reference)}`}>Fine-tune</a>
              <a class="btn-secondary btn-sm" href={`/inference?model=${encodeURIComponent(selected.reference)}`}>Chat</a>
              <a class="btn-secondary btn-sm" href={`/run/serve?model=${encodeURIComponent(selected.reference)}`}>Serve</a>
              <button class="btn-danger btn-sm ml-auto" onclick={() => (toDelete = selected)}>Delete…</button>
            </div>
          </div>
        </section>
      {/if}

      <section class="card" aria-label="Hugging Face">
        <div class="card-header"><h2 class="font-semibold">Hugging Face</h2></div>
        <div class="card-body space-y-4">
          <form class="flex gap-2" onsubmit={search}>
            <label for="hub-query" class="sr-only">Search models</label>
            <input id="hub-query" class="input" type="search" placeholder="Search models…" bind:value={query} />
            <button class="btn-primary shrink-0" disabled={searching}>{searching ? 'Searching…' : 'Search'}</button>
          </form>
          {#if searchError}<p role="alert" class="text-sm text-red-600 dark:text-red-400">{searchError}</p>{/if}

          <p class="section-title">{searched ? `Results for “${query}”` : 'Trending'}</p>
          {#if shownHub.length === 0 && !searching}
            <p class="text-sm text-surface-500">{searched ? 'Nothing found.' : 'Search to find a model.'}</p>
          {/if}
          <ul class="space-y-2">
            {#each shownHub as result (result.id)}
              <li class="flex items-center justify-between gap-2 rounded-lg bg-surface-50 p-3 dark:bg-surface-700/40">
                <div class="min-w-0">
                  <p class="truncate text-sm font-medium">{result.id}</p>
                  <p class="text-xs text-surface-500">
                    {result.downloads_formatted} downloads{result.params_b ? ` · ${result.params_b.toFixed(1)}B` : ''}{result.size_bytes ? ` · ${formatBytes(result.size_bytes)}` : ''}
                  </p>
                  <p class="mt-1 flex gap-1">
                    {#if result.fit}<span class={FIT[result.fit][0]}>{FIT[result.fit][1]}</span>{/if}
                    {#if result.is_gated}<span class="badge-warning">Gated</span>{/if}
                  </p>
                </div>
                {#if downloaded.has(result.id)}
                  <span class="badge-success shrink-0">Downloaded</span>
                {:else}
                  <button class="btn-secondary btn-sm shrink-0" onclick={() => download(result.id)}>Download</button>
                {/if}
              </li>
            {/each}
          </ul>

          <form class="space-y-1" onsubmit={(e) => { e.preventDefault(); download(downloadId); }}>
            <label for="download-id" class="label">Or download by name</label>
            <div class="flex gap-2">
              <input id="download-id" class="input" placeholder="owner/name" bind:value={downloadId} autocomplete="off" autocapitalize="off" spellcheck="false" />
              <button class="btn-secondary shrink-0" disabled={!downloadId.trim()}>Download</button>
            </div>
          </form>
          {#if downloadError}<p role="alert" class="text-sm text-red-600 dark:text-red-400">{downloadError}</p>{/if}
        </div>
      </section>
    </div>
  </div>
</div>

<ConfirmDialog
  open={toDelete !== null}
  title={`Delete ${toDelete?.id ?? ''}?`}
  confirmLabel="Delete"
  danger
  busy={deleting}
  error={deleteError}
  onconfirm={confirmDelete}
  oncancel={() => { toDelete = null; deleteError = null; }}
>
  <p>
    This removes {toDelete?.size_formatted} from disk and cannot be undone.
    {#if toDelete?.source === 'hf_cache'}It can be downloaded again.{/if}
  </p>
  <p class="break-all font-mono text-xs">{toDelete?.path}</p>
</ConfirmDialog>
