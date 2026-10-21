<script lang="ts">
  import { modelsStore, deviceStore } from '#lib/stores.svelte.js';
  import { jobsStore } from '#lib/jobs.svelte.js';
  import { formatBytes, formatEta, getStatusBadgeClass, stateLabel } from '#lib/utils.js';

  const deviceInfo = $derived(deviceStore.info);
  const recentModels = $derived(modelsStore.models.slice(0, 6));
  const active = $derived(jobsStore.active);
  const finished = $derived(jobsStore.jobs.filter((j) => j.state === 'completed').length);
  const failed = $derived(jobsStore.jobs.filter((j) => j.state === 'failed').length);
  const totalSize = $derived(modelsStore.models.reduce((sum, m) => sum + m.size, 0));

  const actions = [
    { href: '/run/train', title: 'Fine-tune', note: 'LoRA, QLoRA, full' },
    { href: '/run/grpo', title: 'GRPO', note: 'RL fine-tuning' },
    { href: '/models', title: 'Download a model', note: 'From Hugging Face' },
    { href: '/inference', title: 'Chat', note: 'Talk to a model' },
  ];
</script>

<div class="space-y-6">
  <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
    <div class="stat-card">
      <p class="stat-label">Models on this Mac</p>
      <p class="stat-value">{modelsStore.models.length}</p>
      <p class="text-sm text-surface-500 mt-2">{formatBytes(totalSize)} in total</p>
    </div>
    <div class="stat-card">
      <p class="stat-label">Running now</p>
      <p class="stat-value">{active.length}</p>
      <p class="text-sm text-surface-500 mt-2">
        <a href="/jobs" class="underline">All jobs</a>
      </p>
    </div>
    <div class="stat-card">
      <p class="stat-label">Finished</p>
      <p class="stat-value">{finished}</p>
      <p class="text-sm text-surface-500 mt-2">{failed} failed</p>
    </div>
    <div class="stat-card">
      <p class="stat-label">Device</p>
      <p class="stat-value text-base truncate">{deviceInfo?.gpu_name ?? (deviceStore.error ? 'Unknown' : 'Detecting…')}</p>
      <p class="text-sm text-surface-500 mt-2">
        {#if deviceInfo}
          {deviceInfo.total_memory_formatted}{deviceInfo.memory_bandwidth_gbps ? ` · ${deviceInfo.memory_bandwidth_gbps.toFixed(0)} GB/s` : ''}
        {:else if deviceStore.error}
          {deviceStore.error}
        {/if}
      </p>
    </div>
  </div>

  {#if deviceInfo?.is_apple_silicon}
    <div class="card">
      <div class="card-body">
        <h2 class="text-sm font-semibold text-surface-700 dark:text-surface-300 mb-3">Apple Silicon</h2>
        <dl class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
          <div><dt class="text-surface-500">Chip</dt><dd class="font-medium">{deviceInfo.gpu_name}</dd></div>
          <div><dt class="text-surface-500">Tier</dt><dd class="font-medium">{deviceInfo.chip_tier ?? 'Unknown'}</dd></div>
          <div><dt class="text-surface-500">Memory</dt><dd class="font-medium">{deviceInfo.total_memory_formatted}</dd></div>
          <div><dt class="text-surface-500">Available</dt><dd class="font-medium">{deviceInfo.available_memory_formatted}</dd></div>
          {#if deviceInfo.gpu_cores}<div><dt class="text-surface-500">GPU cores</dt><dd class="font-medium">{deviceInfo.gpu_cores}</dd></div>{/if}
          <div>
            <dt class="text-surface-500">Neural Engine</dt>
            <dd class="font-medium">{deviceInfo.has_ane ? `Yes${deviceInfo.ane_cores ? `, ${deviceInfo.ane_cores} cores` : ''}` : 'No'}</dd>
          </div>
        </dl>
      </div>
    </div>
  {/if}

  <div class="card">
    <div class="card-header flex items-center justify-between">
      <h2 class="text-lg font-semibold text-surface-900 dark:text-surface-100">Running</h2>
      <a href="/jobs" class="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400">View all</a>
    </div>
    <div class="card-body">
      {#if !jobsStore.loaded}
        <p class="text-surface-500" role="status">Loading…</p>
      {:else if active.length === 0}
        <p class="text-center py-6 text-surface-500">
          Nothing is running. Start a <a class="underline" href="/run/train">fine-tune</a>, or
          <a class="underline" href="/run/serve">serve a model</a>.
        </p>
      {:else}
        <ul class="space-y-3">
          {#each active as job (job.id)}
            <li class="p-4 rounded-lg bg-surface-50 dark:bg-surface-700/50">
              <div class="flex items-center justify-between mb-2">
                <span class="font-medium truncate">{job.label}{job.title ? ` · ${job.title.split('/').pop()}` : ''}</span>
                <span class={getStatusBadgeClass(job.state)}>{stateLabel(job.state)}</span>
              </div>
              <div class="text-sm text-surface-500 mb-2">
                {job.phase ?? (job.step !== null ? `Step ${job.step}${job.total_steps ? ` of ${job.total_steps}` : ''}` : 'Starting…')}
                {#if job.eta_seconds !== null}· about {formatEta(job.eta_seconds)} left{/if}
              </div>
              {#if job.progress !== null}
                <div class="progress-bar" role="progressbar" aria-label="Progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow={Math.round(job.progress * 100)}>
                  <div class="progress-bar-fill" style={`width: ${job.progress * 100}%`}></div>
                </div>
              {/if}
              {#if job.loss !== null}
                <div class="mt-2 flex gap-4 text-xs text-surface-500">
                  <span>Loss {job.loss.toFixed(4)}</span>
                  {#if job.tokens_per_second !== null}<span>{job.tokens_per_second.toFixed(0)} tok/s</span>{/if}
                </div>
              {/if}
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </div>

  <div class="card">
    <div class="card-header flex items-center justify-between">
      <h2 class="text-lg font-semibold text-surface-900 dark:text-surface-100">Models</h2>
      <a href="/models" class="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400">View all</a>
    </div>
    <div class="card-body">
      {#if recentModels.length === 0}
        <p class="text-center py-6 text-surface-500">
          No models yet. <a class="underline" href="/models">Download one</a> from Hugging Face, or
          add a folder that has some.
        </p>
      {:else}
        <ul class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {#each recentModels as model (model.path)}
            <li class="flex items-center gap-3 p-3 rounded-lg hover:bg-surface-50 dark:hover:bg-surface-700/50">
              <div class="w-10 h-10 rounded-lg bg-gradient-to-br from-accent-400 to-orange-600 flex items-center justify-center flex-shrink-0" aria-hidden="true">
                <span class="text-white font-bold text-sm">{model.id.charAt(0).toUpperCase()}</span>
              </div>
              <div class="min-w-0 flex-1">
                <p class="font-medium text-sm truncate">{model.id.split('/').pop()}</p>
                <p class="text-xs text-surface-500">{model.size_formatted} · {model.model_type}</p>
              </div>
            </li>
          {/each}
        </ul>
      {/if}
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h2 class="text-lg font-semibold">Quick actions</h2></div>
    <div class="card-body grid grid-cols-2 md:grid-cols-4 gap-4">
      {#each actions as action}
        <a href={action.href} class="p-4 rounded-lg bg-surface-50 dark:bg-surface-700/50 hover:bg-surface-100 dark:hover:bg-surface-700 text-center">
          <p class="font-medium">{action.title}</p>
          <p class="text-xs text-surface-500 mt-1">{action.note}</p>
        </a>
      {/each}
    </div>
  </div>
</div>
