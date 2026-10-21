<script lang="ts">
  import { configStore, deviceStore, modelsStore } from '#lib/stores.svelte.js';
  import { errorMessage } from '#lib/api.js';
  import type { Theme } from '#lib/api.js';
  import ConfirmDialog from '#lib/components/ConfirmDialog.svelte';

  const config = $derived(configStore.config);
  const device = $derived(deviceStore.info);

  let tokenInput = $state('');
  let tokenBusy = $state(false);
  let tokenMessage = $state<{ kind: 'ok' | 'error'; text: string } | null>(null);
  let confirmRemove = $state(false);

  let defaultModel = $state('');
  let defaultsFor: string | null = null;
  let saveMessage = $state<{ kind: 'ok' | 'error'; text: string } | null>(null);

  // Start the default-model box from what is saved, once.
  $effect(() => {
    if (config && defaultsFor === null) {
      defaultsFor = config.default_model ?? '';
      defaultModel = defaultsFor;
    }
  });

  const themes: { value: Theme; label: string; description: string }[] = [
    { value: 'system', label: 'System', description: 'Follow the system setting' },
    { value: 'light', label: 'Light', description: 'Always light' },
    { value: 'dark', label: 'Dark', description: 'Always dark' },
  ];

  async function setTheme(theme: Theme) {
    saveMessage = null;
    try {
      await configStore.save({ theme });
    } catch (e) {
      saveMessage = { kind: 'error', text: errorMessage(e) };
    }
  }

  async function saveDefaultModel(e: Event) {
    e.preventDefault();
    saveMessage = null;
    try {
      await configStore.save({ default_model: defaultModel });
      saveMessage = { kind: 'ok', text: 'Saved.' };
    } catch (err) {
      saveMessage = { kind: 'error', text: errorMessage(err) };
    }
  }

  async function saveToken(e: Event) {
    e.preventDefault();
    tokenBusy = true;
    tokenMessage = null;
    try {
      await configStore.setToken(tokenInput);
      tokenInput = '';
      tokenMessage = { kind: 'ok', text: 'Saved in the macOS Keychain.' };
    } catch (err) {
      tokenMessage = { kind: 'error', text: errorMessage(err) };
    } finally {
      tokenBusy = false;
    }
  }

  async function removeToken() {
    tokenBusy = true;
    try {
      await configStore.clearToken();
      confirmRemove = false;
      tokenMessage = { kind: 'ok', text: 'Removed from the Keychain.' };
    } catch (err) {
      tokenMessage = { kind: 'error', text: errorMessage(err) };
    } finally {
      tokenBusy = false;
    }
  }

  async function removeDirectory(path: string) {
    try {
      await modelsStore.removeDirectory(path);
    } catch (e) {
      saveMessage = { kind: 'error', text: errorMessage(e) };
    }
  }
</script>

<div class="space-y-6 max-w-3xl">
  <div>
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Settings</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Preferences for PMetal on this Mac</p>
  </div>

  {#if !config}
    <p class="text-surface-500" role="status">{configStore.error ?? 'Loading…'}</p>
  {:else}
    {#each config.notes as note}
      <div class="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200" role="alert">
        {note}
      </div>
    {/each}

    {#if saveMessage}
      <p role={saveMessage.kind === 'error' ? 'alert' : 'status'} class={saveMessage.kind === 'error' ? 'text-sm text-red-600 dark:text-red-400' : 'text-sm text-green-700 dark:text-green-400'}>
        {saveMessage.text}
      </p>
    {/if}

    <section class="card" aria-labelledby="appearance">
      <div class="card-header"><h2 id="appearance" class="font-semibold">Appearance</h2></div>
      <div class="card-body">
        <div class="grid grid-cols-3 gap-3" role="radiogroup" aria-label="Theme">
          {#each themes as theme}
            <button
              type="button"
              role="radio"
              aria-checked={config.theme === theme.value}
              class="rounded-lg border p-3 text-left {config.theme === theme.value ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30' : 'border-surface-200 dark:border-surface-700'}"
              onclick={() => setTheme(theme.value)}
            >
              <span class="block text-sm font-medium">{theme.label}</span>
              <span class="block text-xs text-surface-500">{theme.description}</span>
            </button>
          {/each}
        </div>
      </div>
    </section>

    <section class="card" aria-labelledby="hf">
      <div class="card-header"><h2 id="hf" class="font-semibold">Hugging Face</h2></div>
      <form class="card-body space-y-4" onsubmit={saveToken}>
        <div>
          <label for="hf-token" class="label">Access token</label>
          <div class="flex gap-2">
            <input
              id="hf-token"
              class="input"
              type="password"
              autocomplete="off"
              autocapitalize="off"
              spellcheck="false"
              placeholder={config.hf_token_set ? 'A token is saved. Paste a new one to replace it.' : 'hf_…'}
              bind:value={tokenInput}
              aria-describedby="hf-token-help"
            />
            <button type="submit" class="btn-primary shrink-0" disabled={tokenBusy || !tokenInput.trim()}>Save</button>
            {#if config.hf_token_set}
              <button type="button" class="btn-secondary shrink-0" disabled={tokenBusy} onclick={() => (confirmRemove = true)}>Remove</button>
            {/if}
          </div>
          <p id="hf-token-help" class="mt-1 text-xs text-surface-500">
            Needed for private and gated models and datasets. It is kept in the macOS Keychain, never in a file,
            and is not shown again after you save it.
            {#if config.hf_token_set}<span class="badge-success ml-1">Saved</span>{/if}
          </p>
          {#if tokenMessage}
            <p role={tokenMessage.kind === 'error' ? 'alert' : 'status'} class="mt-1 text-xs {tokenMessage.kind === 'error' ? 'text-red-600 dark:text-red-400' : 'text-green-700 dark:text-green-400'}">{tokenMessage.text}</p>
          {/if}
        </div>
        <dl class="text-sm">
          <dt class="text-surface-500">Model cache</dt>
          <dd class="break-all font-mono text-xs">{config.hf_cache_dir}</dd>
          <dd class="mt-1 text-xs text-surface-500">Set with the HF_HOME or HF_HUB_CACHE environment variable, as for every Hugging Face tool.</dd>
        </dl>
      </form>
    </section>

    <section class="card" aria-labelledby="models">
      <div class="card-header"><h2 id="models" class="font-semibold">Models and output</h2></div>
      <div class="card-body space-y-4">
        <form onsubmit={saveDefaultModel}>
          <label for="default-model" class="label">Default model</label>
          <div class="flex gap-2">
            <input id="default-model" class="input" placeholder="Qwen/Qwen3-0.6B" bind:value={defaultModel} />
            <button type="submit" class="btn-secondary shrink-0">Save</button>
          </div>
        </form>
        <dl class="text-sm">
          <dt class="text-surface-500">Jobs write to</dt>
          <dd class="break-all font-mono text-xs">{config.output_dir}</dd>
        </dl>
        <div>
          <p class="text-sm text-surface-500">Extra model folders</p>
          {#if config.custom_model_dirs.length === 0}
            <p class="text-xs text-surface-500">None. Add one on the Models page.</p>
          {:else}
            <ul class="mt-1 space-y-1">
              {#each config.custom_model_dirs as dir}
                <li class="flex items-center justify-between gap-2 text-xs font-mono">
                  <span class="break-all">{dir}</span>
                  <button type="button" class="btn-ghost btn-sm shrink-0" onclick={() => removeDirectory(dir)}>Remove</button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
      </div>
    </section>

    <section class="card" aria-labelledby="cli">
      <div class="card-header"><h2 id="cli" class="font-semibold">Command-line tool</h2></div>
      <div class="card-body text-sm">
        {#if config.pmetal_binary}
          <p>Jobs run <code class="font-mono text-xs break-all">{config.pmetal_binary}</code>.</p>
          <p class="mt-1 text-xs text-surface-500">Set PMETAL_BIN before opening PMetal to use a different one.</p>
        {:else}
          <p class="text-red-600 dark:text-red-400" role="alert">{config.pmetal_binary_error}</p>
        {/if}
      </div>
    </section>

    <section class="card" aria-labelledby="about">
      <div class="card-header"><h2 id="about" class="font-semibold">About this Mac</h2></div>
      <div class="card-body">
        {#if device}
          <dl class="grid grid-cols-2 gap-4 text-sm">
            <div><dt class="text-surface-500">PMetal</dt><dd class="font-medium">v{device.version}</dd></div>
            <div><dt class="text-surface-500">Chip</dt><dd class="font-medium">{device.gpu_name}</dd></div>
            <div><dt class="text-surface-500">Memory</dt><dd class="font-medium">{device.total_memory_formatted} ({device.available_memory_formatted} free)</dd></div>
            <div><dt class="text-surface-500">Neural Engine</dt><dd class="font-medium">{device.has_ane ? 'Yes' : 'No'}</dd></div>
          </dl>
        {:else}
          <p class="text-sm text-surface-500" role="status">{deviceStore.error ?? 'Detecting…'}</p>
        {/if}
      </div>
    </section>
  {/if}
</div>

<ConfirmDialog
  open={confirmRemove}
  title="Remove the saved token?"
  confirmLabel="Remove"
  danger
  busy={tokenBusy}
  onconfirm={removeToken}
  oncancel={() => (confirmRemove = false)}
>
  <p>Private and gated downloads stop working until you save a token again.</p>
</ConfirmDialog>
