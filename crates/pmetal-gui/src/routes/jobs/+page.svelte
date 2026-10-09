<script lang="ts">
  import {
    trainingStore,
    grpoStore,
    distillationStore,
    benchStore,
    evalStore,
    serveStore,
  } from '#lib/stores.svelte.js';

  /**
   * Unified job row. Every specialized store produces its own run shape; we
   * project them into this common view so the Jobs page doesn't need to
   * know about each domain's type.
   */
  interface JobRow {
    id: string;
    kind: 'training' | 'grpo' | 'distillation' | 'serve' | 'bench' | 'eval';
    label: string;
    model: string;
    status: string;
    progress: number | null; // 0..1, null if not applicable
    detail: string;
    startedAt: string;
    endedAt: string | null;
    canCancel: boolean;
  }

  // Pull reactive slices out of every store. Svelte 5 runes propagate
  // mutations automatically — no manual subscription needed.
  let trainingRuns = $derived(trainingStore.runs);
  let grpoRuns = $derived(grpoStore.runs);
  let distillationRuns = $derived(distillationStore.runs);
  let benchRuns = $derived(benchStore.runs);
  let evalRuns = $derived(evalStore.runs);
  let serveInstances = $derived(serveStore.instances);

  // Filter state
  let statusFilter = $state<'all' | 'running' | 'completed' | 'failed'>('all');
  let kindFilter = $state<'all' | JobRow['kind']>('all');

  function trainingProgress(r: { step: number; total_steps: number }): number | null {
    if (!r.total_steps) return null;
    return Math.min(1, r.step / r.total_steps);
  }

  let allJobs = $derived([
    ...trainingRuns.map<JobRow>(r => ({
      id: r.id,
      kind: 'training',
      label: 'Training',
      model: r.model,
      status: r.status,
      progress: trainingProgress(r),
      detail: r.loss !== null ? `loss ${r.loss.toFixed(4)}` : r.status_message ?? '',
      startedAt: r.started_at,
      endedAt: r.ended_at,
      canCancel: r.status === 'running' || r.status === 'pending',
    })),
    ...grpoRuns.map<JobRow>(r => ({
      id: r.id,
      kind: 'grpo',
      label: 'GRPO',
      model: r.model,
      status: r.status,
      progress: trainingProgress(r),
      detail:
        r.reward_mean !== null
          ? `reward ${r.reward_mean.toFixed(3)}`
          : r.loss !== null
            ? `loss ${r.loss.toFixed(4)}`
            : '',
      startedAt: r.started_at,
      endedAt: r.ended_at,
      canCancel: r.status === 'running' || r.status === 'pending',
    })),
    ...distillationRuns.map<JobRow>(r => ({
      id: r.id,
      kind: 'distillation',
      label: 'Distillation',
      model: `${r.teacher_model} → ${r.student_model}`,
      status: r.status,
      progress: trainingProgress(r),
      detail: r.loss !== null ? `loss ${r.loss.toFixed(4)}` : '',
      startedAt: r.started_at,
      endedAt: r.ended_at,
      canCancel: r.status !== 'completed' && r.status !== 'failed' && r.status !== 'cancelled',
    })),
    ...serveInstances.map<JobRow>(i => ({
      id: i.id,
      kind: 'serve',
      label: 'Serve',
      model: i.model,
      status: i.status,
      progress: null,
      detail: i.bind_url,
      startedAt: i.started_at,
      endedAt: i.stopped_at,
      canCancel: i.status === 'running' || i.status === 'starting',
    })),
    ...benchRuns.map<JobRow>(r => ({
      id: r.id,
      kind: 'bench',
      label: 'Bench',
      model: r.model,
      status: r.status,
      progress: null,
      detail: `${r.trials.length} trial${r.trials.length === 1 ? '' : 's'}${r.preset ? ` · ${r.preset}` : ''}`,
      startedAt: r.started_at,
      endedAt: r.ended_at,
      canCancel: r.status === 'running' || r.status === 'pending',
    })),
    ...evalRuns.map<JobRow>(r => ({
      id: r.id,
      kind: 'eval',
      label: 'Eval',
      model: r.model,
      status: r.status,
      progress:
        r.metrics.samples_total > 0
          ? r.metrics.samples_done / r.metrics.samples_total
          : null,
      detail: r.metrics.perplexity !== null
        ? `ppl ${r.metrics.perplexity.toFixed(2)}`
        : `${r.metrics.samples_done}/${r.metrics.samples_total} samples`,
      startedAt: r.started_at,
      endedAt: r.ended_at,
      canCancel: r.status === 'running' || r.status === 'pending',
    })),
  ]);

  let filteredJobs = $derived(
    allJobs
      .filter(j => {
        if (kindFilter !== 'all' && j.kind !== kindFilter) return false;
        if (statusFilter === 'all') return true;
        if (statusFilter === 'running') return j.status === 'running' || j.status === 'starting' || j.status === 'pending';
        return j.status === statusFilter;
      })
      // Newest first.
      .sort((a, b) => new Date(b.startedAt).getTime() - new Date(a.startedAt).getTime()),
  );

  let runningCount = $derived(
    allJobs.filter(j => j.status === 'running' || j.status === 'starting' || j.status === 'pending').length,
  );
  let completedCount = $derived(allJobs.filter(j => j.status === 'completed' || j.status === 'stopped').length);
  let failedCount = $derived(allJobs.filter(j => j.status === 'failed' || j.status === 'cancelled').length);

  function statusClass(status: string): string {
    if (status === 'running' || status === 'starting' || status === 'pending') {
      return 'bg-primary-50 text-primary-700 dark:bg-primary-900/30 dark:text-primary-300';
    }
    if (status === 'completed' || status === 'stopped') {
      return 'bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-300';
    }
    if (status === 'failed' || status === 'cancelled') {
      return 'bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-300';
    }
    return 'bg-surface-100 text-surface-600 dark:bg-surface-800 dark:text-surface-400';
  }

  function relativeTime(iso: string): string {
    const delta = Math.max(0, Date.now() - new Date(iso).getTime()) / 1000;
    if (delta < 60) return `${Math.round(delta)}s ago`;
    if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
    return `${Math.round(delta / 86400)}d ago`;
  }

  async function cancel(job: JobRow) {
    switch (job.kind) {
      case 'training':
        await trainingStore.stop(job.id);
        break;
      case 'grpo':
        await grpoStore.stop(job.id);
        break;
      case 'distillation':
        await distillationStore.stop(job.id);
        break;
      case 'serve':
        await serveStore.stop(job.id);
        break;
      case 'bench':
        await benchStore.stop(job.id);
        break;
      case 'eval':
        await evalStore.stop(job.id);
        break;
    }
  }

  const kindChoices: { value: 'all' | JobRow['kind']; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'training', label: 'Training' },
    { value: 'grpo', label: 'GRPO' },
    { value: 'distillation', label: 'Distill' },
    { value: 'serve', label: 'Serve' },
    { value: 'bench', label: 'Bench' },
    { value: 'eval', label: 'Eval' },
  ];
</script>

<div class="space-y-6">
  <!-- Header -->
  <div>
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Jobs</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">
      Every running and historical job across training, serving, benchmarking, and evaluation
    </p>
  </div>

  <!-- Stat cards -->
  <div class="grid grid-cols-3 gap-4">
    <div class="card">
      <div class="card-body">
        <div class="text-xs uppercase text-surface-500 tracking-wide">Running</div>
        <div class="text-2xl font-bold text-primary-600 dark:text-primary-400 mt-1">{runningCount}</div>
      </div>
    </div>
    <div class="card">
      <div class="card-body">
        <div class="text-xs uppercase text-surface-500 tracking-wide">Completed</div>
        <div class="text-2xl font-bold text-green-600 dark:text-green-400 mt-1">{completedCount}</div>
      </div>
    </div>
    <div class="card">
      <div class="card-body">
        <div class="text-xs uppercase text-surface-500 tracking-wide">Failed / Cancelled</div>
        <div class="text-2xl font-bold text-red-600 dark:text-red-400 mt-1">{failedCount}</div>
      </div>
    </div>
  </div>

  <!-- Filter bar -->
  <div class="flex flex-wrap items-center gap-4">
    <div class="flex items-center gap-2 text-sm">
      <span class="text-surface-500">Kind</span>
      <div class="flex gap-1">
        {#each kindChoices as choice}
          <button
            type="button"
            class="px-3 py-1 rounded-md text-xs transition-all {kindFilter === choice.value
              ? 'bg-primary-500 text-white'
              : 'bg-surface-100 dark:bg-surface-800 text-surface-600 dark:text-surface-400 hover:bg-surface-200 dark:hover:bg-surface-700'}"
            onclick={() => (kindFilter = choice.value)}
          >
            {choice.label}
          </button>
        {/each}
      </div>
    </div>
    <div class="flex items-center gap-2 text-sm">
      <span class="text-surface-500">Status</span>
      <div class="flex gap-1">
        {#each ['all', 'running', 'completed', 'failed'] as s}
          <button
            type="button"
            class="px-3 py-1 rounded-md text-xs transition-all capitalize {statusFilter === s
              ? 'bg-primary-500 text-white'
              : 'bg-surface-100 dark:bg-surface-800 text-surface-600 dark:text-surface-400 hover:bg-surface-200 dark:hover:bg-surface-700'}"
            onclick={() => (statusFilter = s as typeof statusFilter)}
          >
            {s}
          </button>
        {/each}
      </div>
    </div>
  </div>

  <!-- Jobs table -->
  <div class="card">
    <div class="card-body">
      {#if filteredJobs.length === 0}
        <p class="text-sm text-surface-500 italic text-center py-8">No jobs match the current filter.</p>
      {:else}
        <table class="w-full text-sm">
          <thead>
            <tr class="text-xs uppercase text-surface-500 border-b border-surface-200 dark:border-surface-700">
              <th class="text-left py-2 px-2">Kind</th>
              <th class="text-left py-2 px-2">Model</th>
              <th class="text-left py-2 px-2">Status</th>
              <th class="text-left py-2 px-2">Detail</th>
              <th class="text-left py-2 px-2">Started</th>
              <th class="text-right py-2 px-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {#each filteredJobs as job (job.id)}
              <tr class="border-b border-surface-100 dark:border-surface-800 hover:bg-surface-50 dark:hover:bg-surface-800/50">
                <td class="py-2 px-2">
                  <span class="text-xs font-medium text-surface-700 dark:text-surface-300">{job.label}</span>
                </td>
                <td class="py-2 px-2 font-mono text-xs text-surface-600 dark:text-surface-400 max-w-xs truncate" title={job.model}>
                  {job.model}
                </td>
                <td class="py-2 px-2">
                  <span class="inline-flex px-2 py-0.5 rounded text-xs font-medium {statusClass(job.status)}">
                    {job.status}
                  </span>
                  {#if job.progress !== null && (job.status === 'running' || job.status === 'pending')}
                    <div class="w-24 h-1 mt-1 bg-surface-200 dark:bg-surface-700 rounded overflow-hidden">
                      <div class="h-full bg-primary-500 transition-all" style="width: {(job.progress * 100).toFixed(0)}%"></div>
                    </div>
                  {/if}
                </td>
                <td class="py-2 px-2 text-xs text-surface-600 dark:text-surface-400 font-mono max-w-xs truncate" title={job.detail}>
                  {job.detail || '—'}
                </td>
                <td class="py-2 px-2 text-xs text-surface-500">{relativeTime(job.startedAt)}</td>
                <td class="py-2 px-2 text-right">
                  {#if job.canCancel}
                    <button
                      type="button"
                      class="px-2 py-0.5 text-xs text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded"
                      onclick={() => cancel(job)}
                    >
                      Cancel
                    </button>
                  {/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </div>
  </div>
</div>
