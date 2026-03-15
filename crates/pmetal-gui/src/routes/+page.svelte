<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { dashboardStore, modelsStore, trainingStore, grpoStore, deviceStore } from '$lib/stores.svelte';
  import { formatEta, runProgress, getStatusBadgeClass } from '$lib/utils';

  let stats = $derived(dashboardStore.stats);
  let recentModels = $derived(modelsStore.models.slice(0, 5));
  let activeTrainingRuns = $derived(trainingStore.activeRuns);
  let activeGrpoRuns = $derived(grpoStore.activeRuns);
  let deviceInfo = $derived(deviceStore.info);

  let refreshInterval: ReturnType<typeof setInterval>;

  onMount(() => {
    // Light periodic refresh for active run stats only — full init already done in layout
    refreshInterval = setInterval(() => {
      if (activeTrainingRuns.length > 0 || activeGrpoRuns.length > 0) {
        dashboardStore.refresh();
      }
    }, 30000);
  });

  onDestroy(() => {
    if (refreshInterval) clearInterval(refreshInterval);
  });
</script>

<div class="space-y-6">
  <!-- Stats Grid -->
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
    <div class="stat-card">
      <div class="flex items-center justify-between">
        <div>
          <p class="stat-label">Cached Models</p>
          <p class="stat-value">{stats?.models_count ?? 0}</p>
        </div>
        <div class="w-12 h-12 rounded-lg bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
          <svg class="w-6 h-6 text-primary-600 dark:text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </div>
      </div>
      <p class="text-sm text-surface-500 mt-2">{stats?.total_model_size ?? '0 B'} total</p>
    </div>

    <div class="stat-card">
      <div class="flex items-center justify-between">
        <div>
          <p class="stat-label">Active Runs</p>
          <p class="stat-value">{(stats?.active_training_runs ?? 0) + (stats?.active_grpo_runs ?? 0) + (stats?.active_distillation_runs ?? 0)}</p>
        </div>
        <div class="w-12 h-12 rounded-lg bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
          <svg class="w-6 h-6 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
      </div>
      <div class="flex gap-3 mt-2 text-xs text-surface-500">
        <span>Train: {stats?.active_training_runs ?? 0}</span>
        <span>GRPO: {stats?.active_grpo_runs ?? 0}</span>
        <span>Distill: {stats?.active_distillation_runs ?? 0}</span>
      </div>
    </div>

    <div class="stat-card">
      <div class="flex items-center justify-between">
        <div>
          <p class="stat-label">Completed</p>
          <p class="stat-value">{stats?.completed_training_runs ?? 0}</p>
        </div>
        <div class="w-12 h-12 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
          <svg class="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
      </div>
      <p class="text-sm text-surface-500 mt-2">{stats?.total_training_runs ?? 0} total training runs</p>
    </div>

    <div class="stat-card">
      <div class="flex items-center justify-between">
        <div>
          <p class="stat-label">Device</p>
          <p class="stat-value text-base truncate max-w-[140px]">{deviceInfo?.gpu_name ?? 'Loading...'}</p>
        </div>
        <div class="w-12 h-12 rounded-lg bg-accent-100 dark:bg-accent-900/30 flex items-center justify-center">
          <svg class="w-6 h-6 text-accent-600 dark:text-accent-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
        </div>
      </div>
      <p class="text-sm text-surface-500 mt-2">
        {#if deviceInfo}
          {deviceInfo.total_memory_formatted}{deviceInfo.memory_bandwidth_gbps ? ` · ${deviceInfo.memory_bandwidth_gbps.toFixed(0)} GB/s` : ''}
        {:else}
          Detecting hardware...
        {/if}
      </p>
    </div>
  </div>

  <!-- Hardware Details (Apple Silicon) -->
  {#if deviceInfo?.is_apple_silicon}
    <div class="card">
      <div class="card-body">
        <div class="flex items-center gap-2 mb-3">
          <div class="w-2 h-2 rounded-full bg-accent-500"></div>
          <h3 class="text-sm font-semibold text-surface-700 dark:text-surface-300">Apple Silicon Hardware</h3>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
          <div>
            <p class="text-surface-500 dark:text-surface-400">Chip</p>
            <p class="font-medium text-surface-900 dark:text-surface-100">{deviceInfo.gpu_name}</p>
          </div>
          <div>
            <p class="text-surface-500 dark:text-surface-400">Tier</p>
            <p class="font-medium text-surface-900 dark:text-surface-100">{deviceInfo.chip_tier ?? 'Unknown'}</p>
          </div>
          <div>
            <p class="text-surface-500 dark:text-surface-400">Total Memory</p>
            <p class="font-medium text-surface-900 dark:text-surface-100">{deviceInfo.total_memory_formatted}</p>
          </div>
          <div>
            <p class="text-surface-500 dark:text-surface-400">Available</p>
            <p class="font-medium text-surface-900 dark:text-surface-100">{deviceInfo.available_memory_formatted}</p>
          </div>
          {#if deviceInfo.gpu_cores}
            <div>
              <p class="text-surface-500 dark:text-surface-400">GPU Cores</p>
              <p class="font-medium text-surface-900 dark:text-surface-100">{deviceInfo.gpu_cores}</p>
            </div>
          {/if}
          {#if deviceInfo.memory_bandwidth_gbps}
            <div>
              <p class="text-surface-500 dark:text-surface-400">Bandwidth</p>
              <p class="font-medium text-surface-900 dark:text-surface-100">{deviceInfo.memory_bandwidth_gbps.toFixed(0)} GB/s</p>
            </div>
          {/if}
          <div>
            <p class="text-surface-500 dark:text-surface-400">ANE</p>
            <p class="font-medium {deviceInfo.has_ane ? 'text-green-600 dark:text-green-400' : 'text-surface-500'}">
              {deviceInfo.has_ane ? `Available${deviceInfo.ane_cores ? ` (${deviceInfo.ane_cores} cores)` : ''}` : 'N/A'}
            </p>
          </div>
          <div>
            <p class="text-surface-500 dark:text-surface-400">NAX</p>
            <p class="font-medium {deviceInfo.has_nax ? 'text-accent-600 dark:text-accent-400' : 'text-surface-500'}">
              {deviceInfo.has_nax ? 'Available' : 'N/A'}
            </p>
          </div>
        </div>
      </div>
    </div>
  {/if}

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <!-- Active Training Runs -->
    <div class="card">
      <div class="card-header flex items-center justify-between">
        <h2 class="text-lg font-semibold text-surface-900 dark:text-surface-100">Active Training</h2>
        <a href="/training" class="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400">View all</a>
      </div>
      <div class="card-body">
        {#if activeTrainingRuns.length === 0}
          <div class="text-center py-8">
            <svg class="w-12 h-12 mx-auto text-surface-300 dark:text-surface-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <p class="mt-2 text-surface-500 dark:text-surface-400">No active training runs</p>
            <a href="/training" class="btn-primary btn-sm mt-4 inline-flex">Start Training</a>
          </div>
        {:else}
          <div class="space-y-4">
            {#each activeTrainingRuns as run}
              <div class="p-4 rounded-lg bg-surface-50 dark:bg-surface-700/50">
                <div class="flex items-center justify-between mb-2">
                  <span class="font-medium text-surface-900 dark:text-surface-100 truncate max-w-[200px]">{run.model.split('/').pop()}</span>
                  <span class={getStatusBadgeClass(run.status)}>{run.status}</span>
                </div>
                <div class="text-sm text-surface-500 dark:text-surface-400 mb-2">
                  {run.method.toUpperCase()} · Step {run.step}/{run.total_steps} · ETA {formatEta(run.eta_seconds)}
                </div>
                <div class="progress-bar">
                  <div class="progress-bar-fill" style="width: {runProgress(run.step, run.total_steps)}%"></div>
                </div>
                {#if run.loss !== null}
                  <div class="mt-2 flex gap-4 text-xs text-surface-500 dark:text-surface-400">
                    <span>Loss: {run.loss.toFixed(4)}</span>
                    {#if run.tokens_per_second !== null}
                      <span>{run.tokens_per_second.toFixed(0)} tok/s</span>
                    {/if}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <!-- Active GRPO Runs -->
    <div class="card">
      <div class="card-header flex items-center justify-between">
        <h2 class="text-lg font-semibold text-surface-900 dark:text-surface-100">Active GRPO</h2>
        <a href="/grpo" class="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400">View all</a>
      </div>
      <div class="card-body">
        {#if activeGrpoRuns.length === 0}
          <div class="text-center py-8">
            <svg class="w-12 h-12 mx-auto text-surface-300 dark:text-surface-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <p class="mt-2 text-surface-500 dark:text-surface-400">No active GRPO runs</p>
            <a href="/grpo" class="btn-accent btn-sm mt-4 inline-flex">Start GRPO</a>
          </div>
        {:else}
          <div class="space-y-4">
            {#each activeGrpoRuns as run}
              <div class="p-4 rounded-lg bg-surface-50 dark:bg-surface-700/50">
                <div class="flex items-center justify-between mb-2">
                  <span class="font-medium text-surface-900 dark:text-surface-100 truncate max-w-[200px]">{run.model.split('/').pop()}</span>
                  <span class={getStatusBadgeClass(run.status)}>{run.status}</span>
                </div>
                <div class="text-sm text-surface-500 dark:text-surface-400 mb-2">
                  Step {run.step}/{run.total_steps} · ETA {formatEta(run.eta_seconds)}
                </div>
                <div class="progress-bar">
                  <div class="progress-bar-fill-accent" style="width: {runProgress(run.step, run.total_steps)}%"></div>
                </div>
                {#if run.reward_mean !== null}
                  <div class="mt-2 flex gap-4 text-xs text-surface-500 dark:text-surface-400">
                    <span>Reward: {run.reward_mean.toFixed(3)}</span>
                    {#if run.kl_div !== null}
                      <span>KL: {run.kl_div.toFixed(4)}</span>
                    {/if}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>
  </div>

  <!-- Cached Models -->
  <div class="card">
    <div class="card-header flex items-center justify-between">
      <h2 class="text-lg font-semibold text-surface-900 dark:text-surface-100">Cached Models</h2>
      <a href="/models" class="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400">View all</a>
    </div>
    <div class="card-body">
      {#if recentModels.length === 0}
        <div class="text-center py-8">
          <svg class="w-12 h-12 mx-auto text-surface-300 dark:text-surface-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
          <p class="mt-2 text-surface-500 dark:text-surface-400">No models downloaded yet</p>
          <a href="/models" class="btn-primary btn-sm mt-4 inline-flex">Download Model</a>
        </div>
      {:else}
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {#each recentModels as model}
            <div class="flex items-center gap-3 p-3 rounded-lg hover:bg-surface-50 dark:hover:bg-surface-700/50 transition-colors">
              <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-accent-400 to-orange-600 flex items-center justify-center flex-shrink-0">
                <span class="text-white font-bold text-sm">{model.id.charAt(0).toUpperCase()}</span>
              </div>
              <div class="min-w-0 flex-1">
                <p class="font-medium text-surface-900 dark:text-surface-100 text-sm truncate">{model.id.split('/').pop()}</p>
                <p class="text-xs text-surface-500 dark:text-surface-400">{model.size_formatted}{model.model_type ? ` · ${model.model_type}` : ''}</p>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </div>

  <!-- Quick Actions -->
  <div class="card">
    <div class="card-header">
      <h2 class="text-lg font-semibold text-surface-900 dark:text-surface-100">Quick Actions</h2>
    </div>
    <div class="card-body">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <a href="/training" class="p-4 rounded-lg bg-surface-50 dark:bg-surface-700/50 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors text-center">
          <svg class="w-8 h-8 mx-auto text-primary-600 dark:text-primary-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <p class="font-medium text-surface-900 dark:text-surface-100">Start Training</p>
          <p class="text-xs text-surface-500 dark:text-surface-400 mt-1">LoRA, QLoRA, DPO</p>
        </a>

        <a href="/grpo" class="p-4 rounded-lg bg-surface-50 dark:bg-surface-700/50 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors text-center">
          <svg class="w-8 h-8 mx-auto text-accent-600 dark:text-accent-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p class="font-medium text-surface-900 dark:text-surface-100">Start GRPO</p>
          <p class="text-xs text-surface-500 dark:text-surface-400 mt-1">RL fine-tuning</p>
        </a>

        <a href="/models" class="p-4 rounded-lg bg-surface-50 dark:bg-surface-700/50 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors text-center">
          <svg class="w-8 h-8 mx-auto text-green-600 dark:text-green-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          <p class="font-medium text-surface-900 dark:text-surface-100">Download Model</p>
          <p class="text-xs text-surface-500 dark:text-surface-400 mt-1">From HuggingFace</p>
        </a>

        <a href="/inference" class="p-4 rounded-lg bg-surface-50 dark:bg-surface-700/50 hover:bg-surface-100 dark:hover:bg-surface-700 transition-colors text-center">
          <svg class="w-8 h-8 mx-auto text-blue-600 dark:text-blue-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          <p class="font-medium text-surface-900 dark:text-surface-100">Chat</p>
          <p class="text-xs text-surface-500 dark:text-surface-400 mt-1">Run inference</p>
        </a>
      </div>
    </div>
  </div>
</div>
