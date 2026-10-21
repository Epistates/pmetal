<script lang="ts">
  import { tick } from 'svelte';
  import { page } from '$app/state';
  import type { JobKey } from '#lib/api.js';
  import { errorMessage, issuesOf } from '#lib/api.js';
  import { configStore, formsStore } from '#lib/stores.svelte.js';
  import { jobsStore } from '#lib/jobs.svelte.js';
  import { defaultsOf } from '#lib/forms.js';
  import { getStatusBadgeClass, stateLabel, formatDuration } from '#lib/utils.js';
  import SpecForm from './SpecForm.svelte';
  import JobMonitor from './JobMonitor.svelte';

  interface Props {
    kind: JobKey;
  }

  let { kind }: Props = $props();

  const form = $derived(formsStore.get(kind));
  const jobs = $derived(jobsStore.of(kind));
  const binaryError = $derived(configStore.config?.pmetal_binary_error ?? null);

  let values = $state<Record<string, string>>({});
  let issues = $state<Record<string, string>>({});
  let formError = $state<string | null>(null);
  let submitting = $state(false);
  let selectedId = $state<string | null>(null);
  let loadedKind = '';

  const selected = $derived(
    jobsStore.get(selectedId) ?? (selectedId === null ? (jobs[0] ?? null) : null),
  );
  const storageKey = $derived(`pmetal.form.${kind}`);

  // Start from the spec's defaults, then what this form was last started with,
  // then what the link that brought the person here asks for.
  $effect(() => {
    const current = form;
    if (!current || loadedKind === current.key) return;
    loadedKind = current.key;
    const next = defaultsOf(current);
    try {
      const saved = JSON.parse(localStorage.getItem(`pmetal.form.${current.key}`) ?? '{}');
      for (const f of current.fields) {
        if (typeof saved[f.name] === 'string') next[f.name] = saved[f.name];
      }
    } catch {
      // A damaged saved form is the defaults.
    }
    for (const [name, text] of page.url.searchParams) {
      if (name in next) next[name] = text;
    }
    values = next;
    issues = {};
    formError = null;
    selectedId = null;
  });

  async function submit(e: Event) {
    e.preventDefault();
    if (!form || submitting) return;
    submitting = true;
    formError = null;
    issues = {};
    try {
      const job = await jobsStore.start(kind, values as never);
      selectedId = job.id;
      try {
        localStorage.setItem(storageKey, JSON.stringify(values));
      } catch {
        // Not being able to remember the form is not worth failing the start.
      }
    } catch (err) {
      formError = errorMessage(err);
      issues = issuesOf(err);
      const first = Object.keys(issues)[0];
      if (first) {
        await tick();
        document.getElementById(`field-${first}`)?.focus();
      }
    } finally {
      submitting = false;
    }
  }
</script>

<div class="space-y-6">
  {#if !formsStore.loaded}
    <div class="flex items-center gap-3 text-surface-500" role="status">
      <div class="w-5 h-5 border-2 border-accent-500 border-t-transparent rounded-full animate-spin"></div>
      Loading…
    </div>
  {:else if !form}
    <div class="card card-body" role="alert">
      <p class="font-medium text-surface-900 dark:text-surface-100">This page is not available.</p>
      <p class="text-sm text-surface-500 mt-1">
        {formsStore.error ?? `This version of PMetal has no job called “${kind}”.`}
      </p>
    </div>
  {:else}
    <div>
      <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">{form.label}</h1>
      <p class="text-surface-500 dark:text-surface-400 mt-1">{form.description}</p>
    </div>

    {#if binaryError}
      <div class="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950/40 dark:text-amber-200" role="alert">
        {binaryError}
      </div>
    {/if}

    <div class="grid grid-cols-1 gap-6 xl:grid-cols-2">
      <form onsubmit={submit} class="card" aria-label={`${form.label} settings`}>
        <div class="card-body space-y-4">
          <SpecForm {form} bind:values {issues} />
        </div>
        <div class="card-footer space-y-3">
          {#if formError}
            <p role="alert" class="text-sm text-red-600 dark:text-red-400">{formError}</p>
          {/if}
          <button type="submit" class="btn-primary w-full" disabled={submitting || !!binaryError}>
            {submitting ? 'Starting…' : `Start ${form.label.toLowerCase()}`}
          </button>
        </div>
      </form>

      <div class="space-y-4 min-w-0">
        {#if jobs.length > 0}
          <div class="card">
            <ul class="divide-y divide-surface-200 dark:divide-surface-700" aria-label="Runs">
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
                        {job.title || job.label}
                      </span>
                      <span class="block text-xs text-surface-500">
                        {new Date(job.started_at).toLocaleTimeString()} · {formatDuration(job.elapsed_seconds)}
                        {#if job.progress !== null}· {Math.round(job.progress * 100)}%{/if}
                      </span>
                    </span>
                    <span class={getStatusBadgeClass(job.state)}>{stateLabel(job.state)}</span>
                  </button>
                </li>
              {/each}
            </ul>
          </div>
        {/if}

        {#if selected}
          {#key selected.id}
            <JobMonitor job={selected} onremoved={() => (selectedId = null)} />
          {/key}
        {:else}
          <div class="card card-body text-center text-surface-500">
            <p class="font-medium text-surface-700 dark:text-surface-300">Nothing has run here yet.</p>
            <p class="text-sm mt-1">
              Fill in the settings and start. Progress, output and the result appear here, and the
              run carries on if you leave this page.
            </p>
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>
