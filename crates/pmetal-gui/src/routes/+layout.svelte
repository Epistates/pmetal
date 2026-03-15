<script lang="ts">
  import '../app.css';
  import { onMount, onDestroy } from 'svelte';
  import { page } from '$app/stores';
  import { initializeStores, cleanupStores, configStore } from '$lib/stores.svelte';

  let { children } = $props();

  let initialized = $state(false);
  let appVersion = $state('0.3.6');

  // Navigation items
  const navItems = [
    {
      path: '/',
      label: 'Dashboard',
      icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6',
    },
    {
      path: '/training',
      label: 'Training',
      icon: 'M13 10V3L4 14h7v7l9-11h-7z',
      sublabel: 'LoRA / QLoRA / DPO',
    },
    {
      path: '/grpo',
      label: 'GRPO',
      icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
      sublabel: 'Reinforcement Learning',
    },
    {
      path: '/distillation',
      label: 'Distillation',
      icon: 'M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z',
    },
    {
      path: '/inference',
      label: 'Inference',
      icon: 'M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z',
      sublabel: 'Chat with Models',
    },
    {
      path: '/models',
      label: 'Models',
      icon: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10',
      sublabel: 'HuggingFace',
    },
    {
      path: '/datasets',
      label: 'Datasets',
      icon: 'M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4',
      sublabel: 'HuggingFace',
    },
    {
      path: '/merging',
      label: 'Merging',
      icon: 'M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z',
    },
    {
      path: '/quantize',
      label: 'Quantize',
      icon: 'M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z',
    },
    {
      path: '/settings',
      label: 'Settings',
      icon: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z',
    },
  ];

  // Theme handling
  let theme = $derived(configStore.theme);
  let mq: MediaQueryList | null = null;

  function applyTheme(t: string) {
    if (t === 'dark') {
      document.documentElement.classList.add('dark');
    } else if (t === 'light') {
      document.documentElement.classList.remove('dark');
    } else {
      // System preference
      if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    }
  }

  $effect(() => {
    if (typeof window !== 'undefined') {
      applyTheme(theme);
    }
  });

  onMount(() => {
    // Listen to system color scheme changes for 'system' theme
    mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handleMqChange = () => {
      if (configStore.theme === 'system') {
        applyTheme('system');
      }
    };
    mq.addEventListener('change', handleMqChange);

    // Initialize stores and fetch version asynchronously
    (async () => {
      try {
        await initializeStores();
        // Pull version from backend if available
        const { getSystemInfo } = await import('$lib/api');
        try {
          const info = await getSystemInfo();
          appVersion = info.version;
        } catch {
          // keep default
        }
        initialized = true;
      } catch (e) {
        console.error('Failed to initialize stores:', e);
        initialized = true; // Still show UI even if stores fail
      }
    })();

    return () => {
      mq?.removeEventListener('change', handleMqChange);
    };
  });

  onDestroy(() => {
    cleanupStores();
  });

  function isActive(path: string): boolean {
    const pathname = $page.url.pathname;
    if (path === '/') return pathname === '/';
    return pathname.startsWith(path);
  }

  function getPageTitle(): string {
    return navItems.find(i => isActive(i.path))?.label ?? 'PMetal';
  }
</script>

<div class="flex h-screen overflow-hidden bg-surface-50 dark:bg-surface-900">
  <!-- Sidebar -->
  <aside class="w-64 flex flex-col bg-white dark:bg-surface-800 border-r border-surface-200 dark:border-surface-700">
    <!-- Logo -->
    <div class="h-16 flex items-center px-6 border-b border-surface-200 dark:border-surface-700">
      <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-500 to-orange-600 flex items-center justify-center shadow-sm">
          <span class="text-white font-bold text-lg">P</span>
        </div>
        <div>
          <span class="text-xl font-bold text-surface-900 dark:text-surface-100">PMetal</span>
        </div>
      </div>
    </div>

    <!-- Navigation -->
    <nav class="flex-1 p-4 space-y-1 overflow-y-auto scrollbar-thin" aria-label="Main navigation">
      {#each navItems as item}
        <a
          href={item.path}
          class="nav-link {isActive(item.path) ? 'nav-link-active' : ''}"
          aria-current={isActive(item.path) ? 'page' : undefined}
        >
          <svg class="w-5 h-5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d={item.icon} />
          </svg>
          <div class="min-w-0">
            <div class="font-medium leading-tight">{item.label}</div>
            {#if item.sublabel}
              <div class="text-xs opacity-60 leading-tight mt-0.5">{item.sublabel}</div>
            {/if}
          </div>
        </a>
      {/each}
    </nav>

    <!-- Footer -->
    <div class="p-4 border-t border-surface-200 dark:border-surface-700">
      <div class="text-xs text-surface-400 dark:text-surface-500">
        PMetal v{appVersion} · Apple Silicon
      </div>
    </div>
  </aside>

  <!-- Main content -->
  <main class="flex-1 flex flex-col overflow-hidden">
    <!-- Top bar -->
    <header class="h-16 flex items-center justify-between px-6 bg-white dark:bg-surface-800 border-b border-surface-200 dark:border-surface-700">
      <div class="text-lg font-semibold text-surface-900 dark:text-surface-100">
        {getPageTitle()}
      </div>

      <div class="flex items-center gap-4">
        <!-- Theme toggle -->
        <button
          class="btn-ghost btn-sm"
          aria-label="Toggle theme"
          onclick={() => {
            const newTheme = theme === 'dark' ? 'light' : 'dark';
            if (configStore.config) {
              configStore.save({ ...configStore.config, theme: newTheme });
            }
          }}
        >
          {#if theme === 'dark'}
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
          {:else}
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
            </svg>
          {/if}
        </button>
      </div>
    </header>

    <!-- Page content -->
    <div class="flex-1 overflow-y-auto scrollbar-thin p-6" id="main-content">
      {#if initialized}
        {@render children()}
      {:else}
        <div class="flex items-center justify-center h-full">
          <div class="text-center">
            <div class="w-12 h-12 border-4 border-accent-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p class="text-surface-500 dark:text-surface-400">Loading PMetal...</p>
          </div>
        </div>
      {/if}
    </div>
  </main>
</div>
