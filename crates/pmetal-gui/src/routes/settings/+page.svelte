<script lang="ts">
  import { onMount } from 'svelte';
  import { configStore, modelsStore, deviceStore } from '$lib/stores.svelte';
  import { getSystemInfo } from '$lib/api';
  import type { SystemInfo, AppConfig } from '$lib/api';

  let config = $derived(configStore.config);
  let models = $derived(modelsStore.models);
  let loading = $derived(configStore.loading);
  let deviceInfo = $derived(deviceStore.info);

  // Local form state
  let cacheDir = $state('');
  let hfToken = $state('');
  let defaultModel = $state('');
  let theme = $state('system');

  // System info
  let systemInfo = $state<SystemInfo | null>(null);
  let loadingSystemInfo = $state(true);

  // UI state
  let saveError = $state<string | null>(null);
  let saveSuccess = $state(false);
  let isSaving = $state(false);
  let showToken = $state(false);

  // Sync local state with store when config loads
  $effect(() => {
    if (config) {
      cacheDir = config.cache_dir;
      hfToken = config.hf_token ?? '';
      defaultModel = config.default_model ?? '';
      theme = config.theme;
    }
  });

  onMount(async () => {
    try {
      systemInfo = await getSystemInfo();
    } catch (e) {
      console.error('Failed to load system info:', e);
    } finally {
      loadingSystemInfo = false;
    }
    // deviceStore is already populated by initializeStores in layout
    // Only refresh if it's somehow empty
    if (!deviceStore.info) {
      deviceStore.refresh();
    }
  });

  async function handleSave(e: Event) {
    e.preventDefault();
    saveError = null;
    saveSuccess = false;
    isSaving = true;

    try {
      const newConfig: AppConfig = {
        cache_dir: cacheDir,
        hf_token: hfToken || null,
        default_model: defaultModel || null,
        theme,
      };
      await configStore.save(newConfig);
      saveSuccess = true;
      setTimeout(() => { saveSuccess = false; }, 3000);
    } catch (e) {
      saveError = e instanceof Error ? e.message : String(e);
    } finally {
      isSaving = false;
    }
  }

  function resetForm() {
    if (config) {
      cacheDir = config.cache_dir;
      hfToken = config.hf_token ?? '';
      defaultModel = config.default_model ?? '';
      theme = config.theme;
    }
    saveError = null;
    saveSuccess = false;
  }

  const themes = [
    { value: 'system', label: 'System', description: 'Follow system preference' },
    { value: 'light', label: 'Light', description: 'Always use light mode' },
    { value: 'dark', label: 'Dark', description: 'Always use dark mode' },
  ];
</script>

<div class="space-y-6 max-w-3xl">
  <!-- Header -->
  <div>
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Settings</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Configure PMetal preferences</p>
  </div>

  <form onsubmit={handleSave} class="space-y-6">
    <!-- Appearance -->
    <div class="card">
      <div class="card-header">
        <h2 class="font-semibold text-surface-900 dark:text-surface-100">Appearance</h2>
      </div>
      <div class="card-body">
        <p class="label">Theme</p>
        <div class="grid grid-cols-3 gap-3">
          {#each themes as t}
            <button
              type="button"
              class="p-3 rounded-lg border text-center transition-all {theme === t.value
                ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30'
                : 'border-surface-200 dark:border-surface-700 hover:border-surface-300 dark:hover:border-surface-600'}"
              onclick={() => (theme = t.value)}
            >
              <p class="text-sm font-semibold text-surface-900 dark:text-surface-100">{t.label}</p>
              <p class="text-xs text-surface-500 mt-0.5">{t.description}</p>
            </button>
          {/each}
        </div>
      </div>
    </div>

    <!-- Storage -->
    <div class="card">
      <div class="card-header">
        <h2 class="font-semibold text-surface-900 dark:text-surface-100">Storage</h2>
      </div>
      <div class="card-body space-y-4">
        <div>
          <label class="label" for="cache-dir">Cache Directory</label>
          <input
            id="cache-dir"
            type="text"
            class="input"
            placeholder="~/.cache/pmetal"
            bind:value={cacheDir}
          />
          <p class="text-xs text-surface-500 mt-1">Where downloaded models and datasets are stored</p>
        </div>
        <div>
          <label class="label" for="default-model">Default Model</label>
          <select id="default-model" class="input" bind:value={defaultModel}>
            <option value="">None</option>
            {#each models as model}
              <option value={model.id}>{model.id}</option>
            {/each}
          </select>
          <p class="text-xs text-surface-500 mt-1">Pre-selected model in training and inference pages</p>
        </div>
      </div>
    </div>

    <!-- HuggingFace -->
    <div class="card">
      <div class="card-header">
        <h2 class="font-semibold text-surface-900 dark:text-surface-100">HuggingFace</h2>
      </div>
      <div class="card-body space-y-4">
        <div>
          <label class="label" for="hf-token">Access Token</label>
          <div class="relative">
            <input
              id="hf-token"
              type={showToken ? 'text' : 'password'}
              class="input pr-12"
              placeholder="hf_..."
              bind:value={hfToken}
              autocomplete="off"
            />
            <button
              type="button"
              class="absolute right-3 top-1/2 -translate-y-1/2 text-surface-400 hover:text-surface-600 dark:hover:text-surface-300"
              onclick={() => (showToken = !showToken)}
            >
              {#if showToken}
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                </svg>
              {:else}
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              {/if}
            </button>
          </div>
          <p class="text-xs text-surface-500 mt-1">
            Required for gated models (Llama, Gemma, etc.). Get your token at
            <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener" class="text-primary-600 dark:text-primary-400 hover:underline">
              huggingface.co/settings/tokens
            </a>
          </p>
        </div>
      </div>
    </div>

    <!-- Error / Success -->
    {#if saveError}
      <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm">
        {saveError}
      </div>
    {/if}
    {#if saveSuccess}
      <div class="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-sm flex items-center gap-2">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
        Settings saved successfully
      </div>
    {/if}

    <div class="flex gap-3">
      <button type="submit" class="btn-primary" disabled={isSaving || loading}>
        {#if isSaving}
          <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          Saving...
        {:else}
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
          Save Settings
        {/if}
      </button>
      <button type="button" class="btn-secondary" onclick={resetForm} disabled={isSaving}>
        Reset
      </button>
    </div>
  </form>

  <!-- About -->
  <div class="card">
    <div class="card-header">
      <h2 class="font-semibold text-surface-900 dark:text-surface-100">About PMetal</h2>
    </div>
    <div class="card-body space-y-4">
      <div class="flex items-center gap-4">
        <div class="w-12 h-12 rounded-xl bg-gradient-to-br from-accent-500 to-orange-600 flex items-center justify-center">
          <span class="text-white font-bold text-xl">P</span>
        </div>
        <div>
          <p class="font-semibold text-surface-900 dark:text-surface-100">PMetal v{systemInfo?.version ?? '...'}</p>
          <p class="text-sm text-surface-500">Apple Silicon ML Training Suite</p>
        </div>
      </div>

      <div class="grid grid-cols-2 gap-4 text-sm pt-2 border-t border-surface-200 dark:border-surface-700">
        <div>
          <p class="text-surface-500">Platform</p>
          <p class="font-medium text-surface-900 dark:text-surface-100">
            {systemInfo?.platform ?? 'Loading...'}
          </p>
        </div>
        <div>
          <p class="text-surface-500">Architecture</p>
          <p class="font-medium text-surface-900 dark:text-surface-100">
            {systemInfo?.arch ?? 'Loading...'}
          </p>
        </div>

        {#if deviceInfo}
          <div>
            <p class="text-surface-500">GPU</p>
            <p class="font-medium text-surface-900 dark:text-surface-100">{deviceInfo.gpu_name}</p>
          </div>
          <div>
            <p class="text-surface-500">Memory</p>
            <p class="font-medium text-surface-900 dark:text-surface-100">{deviceInfo.total_memory_formatted}</p>
          </div>
          {#if deviceInfo.has_ane}
            <div>
              <p class="text-surface-500">ANE</p>
              <p class="font-medium text-green-600 dark:text-green-400">Available</p>
            </div>
          {/if}
          {#if deviceInfo.has_nax}
            <div>
              <p class="text-surface-500">NAX (Apple10+)</p>
              <p class="font-medium text-accent-600 dark:text-accent-400">Available</p>
            </div>
          {/if}
        {/if}
      </div>

      <div class="pt-2 border-t border-surface-200 dark:border-surface-700">
        <p class="text-xs text-surface-500">
          PMetal is licensed under MIT OR Apache-2.0. Built for Apple Silicon — M1 through M5 (including Pro/Max/Ultra/Ultra Max).
        </p>
      </div>
    </div>
  </div>
</div>
