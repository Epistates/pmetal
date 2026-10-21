<script lang="ts">
  import '../app.css';
  import { onMount, onDestroy } from 'svelte';
  import { page } from '$app/state';
  import { initializeStores, cleanupStores, configStore, deviceStore, formsStore } from '#lib/stores.svelte.js';
  import { jobsStore } from '#lib/jobs.svelte.js';
  import { onAppQuitting } from '#lib/api.js';
  import { SECTIONS } from '#lib/nav.js';

  let { children } = $props();

  let initialized = $state(false);
  let initError = $state<string | null>(null);
  let quitting = $state(false);
  let stopListening: (() => void) | null = null;

  let theme = $derived(configStore.theme);
  const version = $derived(deviceStore.info?.version ?? '');
  const running = $derived(jobsStore.active.length);

  function applyTheme(t: string) {
    const dark =
      t === 'dark' || (t !== 'light' && window.matchMedia('(prefers-color-scheme: dark)').matches);
    document.documentElement.classList.toggle('dark', dark);
  }

  $effect(() => {
    if (typeof window !== 'undefined') applyTheme(theme);
  });

  onMount(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => {
      if (configStore.theme === 'system') applyTheme('system');
    };
    mq.addEventListener('change', onChange);

    (async () => {
      try {
        stopListening = await onAppQuitting(() => (quitting = true));
        await initializeStores();
      } catch (e) {
        console.error('Failed to initialize stores:', e);
        initError = e instanceof Error ? e.message : String(e);
      }
      initialized = true;
    })();

    return () => mq.removeEventListener('change', onChange);
  });

  onDestroy(() => {
    cleanupStores();
    stopListening?.();
  });

  function isActive(path: string): boolean {
    const pathname = page.url.pathname;
    return path === '/' ? pathname === '/' : pathname === path || pathname.startsWith(`${path}/`);
  }

  const title = $derived.by(() => {
    const items = SECTIONS.flatMap((s) => s.items);
    const item = items.find((i) => isActive(i.path));
    const kind = /^\/run\/(.+)$/.exec(page.url.pathname)?.[1];
    return item?.label ?? (kind ? formsStore.get(kind)?.label : undefined) ?? 'PMetal';
  });

  async function toggleTheme() {
    const dark = document.documentElement.classList.contains('dark');
    try {
      await configStore.save({ theme: dark ? 'light' : 'dark' });
    } catch {
      // The store keeps the error; the theme stays as it was.
    }
  }
</script>

<a
  href="#main-content"
  class="sr-only focus:not-sr-only focus:absolute focus:z-50 focus:m-2 focus:rounded focus:bg-white focus:px-3 focus:py-2 focus:text-sm dark:focus:bg-surface-800"
>
  Skip to content
</a>

<div class="flex h-screen overflow-hidden bg-surface-50 dark:bg-surface-900">
  <aside class="w-64 flex flex-col bg-white dark:bg-surface-800 border-r border-surface-200 dark:border-surface-700">
    <div class="h-16 flex items-center px-6 border-b border-surface-200 dark:border-surface-700">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-500 to-orange-600 flex items-center justify-center shadow-sm" aria-hidden="true">
          <span class="text-white font-bold text-lg">P</span>
        </div>
        <span class="text-xl font-bold text-surface-900 dark:text-surface-100">PMetal</span>
      </div>
    </div>

    <nav class="flex-1 p-4 space-y-4 overflow-y-auto scrollbar-thin" aria-label="Main navigation">
      {#each SECTIONS as section (section.title)}
        <div>
          <p class="px-4 mb-1 text-xs font-semibold uppercase tracking-wider text-surface-400 dark:text-surface-500">
            {section.title}
          </p>
          <div class="space-y-0.5">
            {#each section.items as item (item.path)}
              <a
                href={item.path}
                class="nav-link {isActive(item.path) ? 'nav-link-active' : ''}"
                aria-current={isActive(item.path) ? 'page' : undefined}
              >
                <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d={item.icon} />
                </svg>
                <div class="min-w-0 flex-1">
                  <div class="font-medium leading-tight">{item.label}</div>
                  {#if item.sublabel}
                    <div class="text-xs opacity-60 leading-tight mt-0.5">{item.sublabel}</div>
                  {/if}
                </div>
                {#if item.path === '/jobs' && running > 0}
                  <span class="badge-primary" aria-label={`${running} running`}>{running}</span>
                {/if}
              </a>
            {/each}
          </div>
        </div>
      {/each}
    </nav>

    <div class="p-4 border-t border-surface-200 dark:border-surface-700">
      <div class="text-xs text-surface-400 dark:text-surface-500">
        PMetal{version ? ` v${version}` : ''} · Apple Silicon
      </div>
    </div>
  </aside>

  <main class="flex-1 flex flex-col overflow-hidden">
    <header class="h-16 flex items-center justify-between px-6 bg-white dark:bg-surface-800 border-b border-surface-200 dark:border-surface-700">
      <div class="text-lg font-semibold text-surface-900 dark:text-surface-100">{title}</div>
      <div class="flex items-center gap-4">
        <button class="btn-ghost btn-sm" aria-label="Toggle dark mode" onclick={toggleTheme}>
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
          </svg>
        </button>
      </div>
    </header>

    <div class="flex-1 overflow-y-auto scrollbar-thin p-6" id="main-content" tabindex="-1">
      {#if initialized}
        {#if initError}
          <div class="mb-4 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200" role="alert">
            PMetal could not finish loading: {initError}. Some pages may be empty; restart the app if this keeps happening.
          </div>
        {/if}
        {@render children()}
      {:else}
        <div class="flex items-center justify-center h-full" role="status">
          <div class="text-center">
            <div class="w-12 h-12 border-4 border-accent-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p class="text-surface-500 dark:text-surface-400">Loading PMetal…</p>
          </div>
        </div>
      {/if}
    </div>
  </main>

  {#if quitting}
    <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/60" role="alertdialog" aria-live="assertive" aria-label="Quitting">
      <div class="rounded-xl bg-white p-8 text-center shadow-2xl dark:bg-surface-800">
        <div class="w-10 h-10 border-4 border-accent-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p class="font-medium text-surface-900 dark:text-surface-100">Stopping jobs and quitting…</p>
        <p class="text-sm text-surface-500 mt-1">Training runs save their adapters first.</p>
      </div>
    </div>
  {/if}
</div>
