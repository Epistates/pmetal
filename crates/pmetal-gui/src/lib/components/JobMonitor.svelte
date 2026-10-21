<script lang="ts">
  import { tick } from 'svelte';
  import type { JobView } from '#lib/api.js';
  import { errorMessage } from '#lib/api.js';
  import { jobsStore } from '#lib/jobs.svelte.js';
  import { formatEta, formatDuration, formatMetric, getStatusBadgeClass, stateLabel } from '#lib/utils.js';
  import LossChart from './LossChart.svelte';
  import ConfirmDialog from './ConfirmDialog.svelte';

  interface Props {
    job: JobView;
    /** Called after the job is removed from the list. */
    onremoved?: () => void;
  }

  let { job, onremoved }: Props = $props();

  const active = $derived(job.state === 'running' || job.state === 'stopping');
  const lines = $derived(jobsStore.logs[job.id] ?? []);
  const points = $derived(jobsStore.steps[job.id] ?? []);

  let logEl = $state<HTMLElement | null>(null);
  let follow = $state(true);
  let actionError = $state<string | null>(null);
  let copied = $state<string | null>(null);
  let confirmStop = $state(false);
  let stopping = $state(false);

  // Follow this job's output and steps while it is shown.
  $effect(() => {
    const id = job.id;
    void jobsStore.loadLogs(id);
    void jobsStore.loadSteps(id);
    return () => jobsStore.release(id);
  });

  // Keep the end of the log in view while the person has not scrolled away.
  $effect(() => {
    void lines.length;
    if (follow && logEl) {
      void tick().then(() => {
        if (logEl) logEl.scrollTop = logEl.scrollHeight;
      });
    }
  });

  function onScroll() {
    if (!logEl) return;
    follow = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 24;
  }

  async function copy(label: string, text: string) {
    try {
      await navigator.clipboard.writeText(text);
      copied = label;
      setTimeout(() => {
        if (copied === label) copied = null;
      }, 1500);
    } catch (e) {
      actionError = `Could not copy: ${errorMessage(e)}`;
    }
  }

  async function stop() {
    stopping = true;
    actionError = null;
    try {
      await jobsStore.stop(job.id);
      confirmStop = false;
    } catch (e) {
      actionError = errorMessage(e);
    } finally {
      stopping = false;
    }
  }

  async function remove() {
    actionError = null;
    try {
      await jobsStore.forget(job.id);
      onremoved?.();
    } catch (e) {
      actionError = errorMessage(e);
    }
  }

  const metrics = $derived(
    [
      { label: 'Loss', value: formatMetric(job.loss) },
      { label: 'Best loss', value: formatMetric(job.best_loss) },
      { label: 'Eval loss', value: formatMetric(job.eval_loss) },
      { label: 'Learning rate', value: formatMetric(job.learning_rate) },
      { label: 'Grad norm', value: formatMetric(job.grad_norm, 3) },
      { label: 'Tokens/s', value: formatMetric(job.tokens_per_second, 0) },
      {
        label: 'Epoch',
        value: job.epoch !== null ? `${Math.round(job.epoch)}${job.total_epochs ? ` / ${job.total_epochs}` : ''}` : '--',
      },
    ].filter((m) => m.value !== '--'),
  );
</script>

<section class="card" aria-label={`${job.label} ${job.title}`}>
  <div class="card-header flex flex-wrap items-center justify-between gap-3">
    <div class="min-w-0">
      <div class="flex items-center gap-2">
        <h2 class="font-semibold text-surface-900 dark:text-surface-100 truncate">
          {job.label}{job.title ? ` · ${job.title}` : ''}
        </h2>
        <span class={getStatusBadgeClass(job.state)} role="status">{stateLabel(job.state)}</span>
      </div>
      <p class="text-xs text-surface-500 mt-0.5">
        {#if active}Running for {formatDuration(job.elapsed_seconds)}{:else}Took {formatDuration(job.elapsed_seconds)}{/if}
        · started {new Date(job.started_at).toLocaleString()}
      </p>
    </div>
    <div class="flex items-center gap-2">
      <button type="button" class="btn-ghost btn-sm" onclick={() => copy('command', job.command)}>
        {copied === 'command' ? 'Copied' : 'Copy command'}
      </button>
      {#if active}
        <button
          type="button"
          class="btn-danger btn-sm"
          disabled={job.state === 'stopping'}
          onclick={() => (confirmStop = true)}
        >
          {job.state === 'stopping' ? 'Stopping…' : 'Stop'}
        </button>
      {:else}
        <button type="button" class="btn-secondary btn-sm" onclick={remove}>Remove</button>
      {/if}
    </div>
  </div>

  <div class="card-body space-y-5">
    {#if actionError}
      <p role="alert" class="text-sm text-red-600 dark:text-red-400">{actionError}</p>
    {/if}

    {#if job.state === 'failed'}
      <div class="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200" role="alert">
        <p class="font-medium">This job failed{job.exit_code !== null && job.exit_code >= 0 ? ` (exit code ${job.exit_code})` : ''}.</p>
        {#if job.error}<p class="mt-1 break-words">{job.error}</p>{/if}
        <p class="mt-2 text-xs opacity-80">The full output is below.</p>
      </div>
    {:else if job.state === 'stopped'}
      <p class="text-sm text-surface-600 dark:text-surface-400">
        Stopped. A training run saves its adapter before it exits; look in the output folder.
      </p>
    {/if}

    {#if active}
      <div aria-live="polite">
        <p class="text-sm text-surface-700 dark:text-surface-300">
          {#if job.state === 'stopping'}
            Stopping, and saving what it can…
          {:else if job.phase}
            {job.phase}
          {:else if job.step !== null}
            Step {job.step}{job.total_steps ? ` of ${job.total_steps}` : ''}
          {:else}
            Starting…
          {/if}
        </p>
        {#if job.progress !== null}
          <div
            class="progress-bar mt-2"
            role="progressbar"
            aria-label="Progress"
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow={Math.round(job.progress * 100)}
          >
            <div class="progress-bar-fill" style={`width: ${job.progress * 100}%`}></div>
          </div>
          <p class="mt-1 text-xs text-surface-500">
            {Math.round(job.progress * 100)}% · step {job.step} of {job.total_steps}
            {#if job.eta_seconds !== null}· about {formatEta(job.eta_seconds)} left{/if}
          </p>
        {/if}
      </div>
    {/if}

    {#if job.serve}
      <div class="rounded-lg bg-surface-100 p-4 text-sm dark:bg-surface-700/50">
        {#if job.serve.reachable}
          <p class="font-medium text-green-700 dark:text-green-400">Serving at {job.serve.url}</p>
          <pre class="mt-2 overflow-x-auto text-xs"><code>curl {job.serve.url}/v1/chat/completions -H 'Content-Type: application/json' -d '{'{"messages":[{"role":"user","content":"Hello"}]}'}'</code></pre>
        {:else if active}
          <p>Waiting for the server at {job.serve.url} to answer…</p>
        {/if}
      </div>
    {/if}

    {#if metrics.length}
      <dl class="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {#each metrics as metric}
          <div>
            <dt class="stat-label">{metric.label}</dt>
            <dd class="text-lg font-semibold tabular-nums text-surface-900 dark:text-surface-100">{metric.value}</dd>
          </div>
        {/each}
      </dl>
    {/if}

    {#if points.length >= 2}
      <LossChart {points} />
    {/if}

    {#if job.result}
      <div class="text-sm">
        <p class="section-title">Result</p>
        <dl class="space-y-1">
          {#if job.result.output_path}<div class="flex gap-2"><dt class="text-surface-500">Saved to</dt><dd class="break-all">{job.result.output_path}</dd></div>{/if}
          {#if job.result.final_loss !== null}<div class="flex gap-2"><dt class="text-surface-500">Final loss</dt><dd>{formatMetric(job.result.final_loss)}</dd></div>{/if}
          {#each Object.entries(job.result.extra) as [key, value]}
            <div class="flex gap-2"><dt class="text-surface-500">{key}</dt><dd class="break-all">{value}</dd></div>
          {/each}
        </dl>
      </div>
    {/if}

    {#if job.output}
      <div class="flex items-center gap-2 text-sm">
        <span class="text-surface-500">Output folder</span>
        <code class="break-all text-xs">{job.output}</code>
        <button type="button" class="btn-ghost btn-sm" onclick={() => copy('output', job.output ?? '')}>
          {copied === 'output' ? 'Copied' : 'Copy'}
        </button>
      </div>
    {/if}

    <div>
      <div class="flex items-center justify-between">
        <p class="section-title mb-1">Output</p>
        <div class="flex gap-2">
          {#if !follow}
            <button type="button" class="btn-ghost btn-sm" onclick={() => (follow = true)}>Follow</button>
          {/if}
          <button
            type="button"
            class="btn-ghost btn-sm"
            disabled={!lines.length}
            onclick={() => copy('log', lines.map((l) => l.text).join('\n'))}
          >
            {copied === 'log' ? 'Copied' : 'Copy output'}
          </button>
        </div>
      </div>
      <!-- A scrollable region must take focus, or a keyboard cannot scroll it. -->
      <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
      <div
        bind:this={logEl}
        onscroll={onScroll}
        role="log"
        aria-label="Job output"
        tabindex="0"
        class="h-64 overflow-auto rounded-lg bg-surface-900 p-3 font-mono text-xs leading-relaxed text-surface-100 scrollbar-thin"
      >
        {#each lines as line (line.seq)}
          <div class={line.stream === 'stderr' ? 'text-amber-300' : ''}>{line.text}</div>
        {:else}
          <div class="text-surface-500">{active ? 'Waiting for output…' : 'No output.'}</div>
        {/each}
      </div>
    </div>
  </div>
</section>

<ConfirmDialog
  open={confirmStop}
  title="Stop this job?"
  confirmLabel="Stop job"
  danger
  busy={stopping}
  error={actionError}
  onconfirm={stop}
  oncancel={() => (confirmStop = false)}
>
  <p>
    The job is asked to stop and given a few seconds to save. A training run writes its adapter
    before it exits; a server just shuts down.
  </p>
</ConfirmDialog>
