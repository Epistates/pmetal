<script lang="ts">
  import { jobsStore } from '#lib/jobs.svelte.js';
  import JobMonitor from '#lib/components/JobMonitor.svelte';
  import { formatDuration, getStatusBadgeClass, stateLabel } from '#lib/utils.js';

  type Filter = 'all' | 'active' | 'finished';
  let filter = $state<Filter>('all');
  let selectedId = $state<string | null>(null);

  const jobs = $derived(
    jobsStore.jobs.filter((j) => {
      const active = j.state === 'running' || j.state === 'stopping';
      return filter === 'all' || (filter === 'active' ? active : !active);
    }),
  );
  const selected = $derived(jobsStore.get(selectedId));

  const filters: { value: Filter; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'active', label: 'Running' },
    { value: 'finished', label: 'Finished' },
  ];
</script>

<div class="space-y-6">
  <div class="flex items-end justify-between">
    <div>
      <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Jobs</h1>
      <p class="text-surface-500 dark:text-surface-400 mt-1">
        Everything PMetal is running or has run since it was opened. Jobs carry on while you use
        other pages, and are stopped when you quit.
      </p>
    </div>
    <div class="flex gap-1" role="group" aria-label="Show">
      {#each filters as f}
        <button
          type="button"
          class={filter === f.value ? 'btn-primary btn-sm' : 'btn-secondary btn-sm'}
          aria-pressed={filter === f.value}
          onclick={() => (filter = f.value)}
        >
          {f.label}
        </button>
      {/each}
    </div>
  </div>

  {#if !jobsStore.loaded}
    <p class="text-surface-500" role="status">Loading…</p>
  {:else if jobsStore.error}
    <div class="card card-body text-red-700 dark:text-red-300" role="alert">
      The job list could not be read: {jobsStore.error}
      <button class="btn-secondary btn-sm mt-3" onclick={() => jobsStore.refresh()}>Try again</button>
    </div>
  {:else if jobsStore.jobs.length === 0}
    <div class="card card-body text-center">
      <p class="font-medium text-surface-800 dark:text-surface-200">No jobs yet.</p>
      <p class="mt-1 text-sm text-surface-500">
        Start one from Train, Run, Tools or Measure in the sidebar; it will appear here.
      </p>
    </div>
  {:else}
    <div class="grid grid-cols-1 gap-6 xl:grid-cols-5">
      <div class="card xl:col-span-2 self-start">
        {#if jobs.length === 0}
          <p class="p-6 text-sm text-surface-500">No {filter === 'active' ? 'running' : 'finished'} jobs.</p>
        {:else}
          <ul class="divide-y divide-surface-200 dark:divide-surface-700" aria-label="Jobs">
            {#each jobs as job (job.id)}
              <li>
                <button
                  type="button"
                  class="flex w-full items-center justify-between gap-3 px-4 py-3 text-left hover:bg-surface-50 dark:hover:bg-surface-700/40"
                  class:bg-primary-50={selected?.id === job.id}
                  aria-current={selected?.id === job.id ? 'true' : undefined}
                  onclick={() => (selectedId = job.id)}
                >
                  <span class="min-w-0">
                    <span class="block truncate text-sm font-medium text-surface-900 dark:text-surface-100">
                      {job.label}{job.title ? ` · ${job.title}` : ''}
                    </span>
                    <span class="block text-xs text-surface-500">
                      {new Date(job.started_at).toLocaleString()} · {formatDuration(job.elapsed_seconds)}
                      {#if job.progress !== null}· {Math.round(job.progress * 100)}%{/if}
                    </span>
                  </span>
                  <span class={getStatusBadgeClass(job.state)}>{stateLabel(job.state)}</span>
                </button>
              </li>
            {/each}
          </ul>
        {/if}
      </div>

      <div class="xl:col-span-3 min-w-0">
        {#if selected}
          {#key selected.id}
            <JobMonitor job={selected} onremoved={() => (selectedId = null)} />
          {/key}
        {:else}
          <div class="card card-body text-center text-surface-500">Choose a job to see its progress and output.</div>
        {/if}
      </div>
    </div>
  {/if}
</div>
