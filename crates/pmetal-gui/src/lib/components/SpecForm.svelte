<script lang="ts">
  import type { DatasetPeek, FormInfo, ModelFitInfo } from '#lib/api.js';
  import { getModelFit, peekDatasetColumns, errorMessage } from '#lib/api.js';
  import { datasetsStore, modelsStore } from '#lib/stores.svelte.js';
  import { defaultsOf, groupsOf, isColumnField, isLocalPath, matchesFilter, seqLenField } from '#lib/forms.js';
  import Field from './Field.svelte';

  interface Props {
    form: FormInfo;
    /** The text of every field, by name. */
    values: Record<string, string>;
    /** What is wrong with a field, by name. */
    issues?: Record<string, string>;
    /** Fields not to show; they keep the spec's own value. */
    hidden?: string[];
  }

  let { form, values = $bindable(), issues = {}, hidden = [] }: Props = $props();

  let filter = $state('');
  const groups = $derived(groupsOf(form, hidden));
  const uid = $derived(`spec-${form.key}`);

  // Fields that name a model or a dataset on this Mac are offered as the
  // person types.
  const modelListId = $derived(`${uid}-models`);
  const datasetListId = $derived(`${uid}-datasets`);

  // ── What is known about the dataset chosen ────────────────────────────────
  const datasetField = $derived(form.fields.find((f) => f.input.type === 'dataset')?.name);
  let peek = $state<DatasetPeek | null>(null);
  let peekError = $state<string | null>(null);
  let peeked = '';
  let peekTimer: ReturnType<typeof setTimeout> | undefined;

  $effect(() => {
    const path = datasetField ? values[datasetField]?.trim() : '';
    clearTimeout(peekTimer);
    if (!path || !isLocalPath(path)) {
      peek = null;
      peekError = null;
      peeked = '';
      return;
    }
    if (path === peeked) return;
    peekTimer = setTimeout(async () => {
      peeked = path;
      try {
        peek = await peekDatasetColumns(path);
        peekError = null;
      } catch (e) {
        peek = null;
        peekError = errorMessage(e);
      }
    }, 400);
    return () => clearTimeout(peekTimer);
  });

  const columnSuggestions = $derived(peek?.columns ?? []);
  const lengthField = $derived(seqLenField(form));

  // ── How the model chosen would run ────────────────────────────────────────
  const modelField = $derived(form.fields.find((f) => f.input.type === 'model')?.name);
  let fit = $state<ModelFitInfo | null>(null);
  let fitFor = '';
  let fitTimer: ReturnType<typeof setTimeout> | undefined;

  $effect(() => {
    const model = modelField ? values[modelField]?.trim() : '';
    clearTimeout(fitTimer);
    if (!model) {
      fit = null;
      fitFor = '';
      return;
    }
    if (model === fitFor) return;
    fitTimer = setTimeout(async () => {
      fitFor = model;
      try {
        fit = await getModelFit(model);
      } catch {
        // A name that is not a model (yet) has no fit to show.
        fit = null;
      }
    }, 600);
    return () => clearTimeout(fitTimer);
  });

  function applyLength() {
    if (peek && lengthField) values[lengthField] = String(peek.suggested_seq_len);
  }

  function reset() {
    Object.assign(values, defaultsOf(form));
  }

  const FIT_TEXT = { fits: 'Fits in memory', tight: 'A tight fit', too_large: 'Too large for this Mac' };
  const FIT_CLASS = { fits: 'badge-success', tight: 'badge-warning', too_large: 'badge-danger' };

  function open(group: { required: boolean; fields: { name: string }[] }): boolean {
    return (
      group.required ||
      filter.trim() !== '' ||
      group.fields.some((f) => f.name in issues)
    );
  }

  const shown = $derived(
    groups
      .map((g) => ({ ...g, fields: g.fields.filter((f) => matchesFilter(f, filter)) }))
      .filter((g) => g.fields.length > 0),
  );
</script>

<datalist id={modelListId}>
  {#each modelsStore.models as model}
    <option value={model.reference}>{model.id} ({model.size_formatted})</option>
  {/each}
</datalist>
<datalist id={datasetListId}>
  {#each datasetsStore.cached as dataset}
    <option value={dataset.name}>{dataset.size_formatted}</option>
  {/each}
</datalist>

<div class="space-y-4">
  <div class="flex items-center gap-2">
    <label for={`${uid}-filter`} class="sr-only">Find an option</label>
    <input
      id={`${uid}-filter`}
      class="input text-sm"
      type="search"
      placeholder="Find an option…"
      bind:value={filter}
    />
    <button type="button" class="btn-ghost btn-sm shrink-0" onclick={reset}>Reset all</button>
  </div>

  {#if fit}
    <div class="flex flex-wrap items-center gap-2 text-sm" aria-live="polite">
      <span class={FIT_CLASS[fit.training_fit]}>Training: {FIT_TEXT[fit.training_fit]}</span>
      <span class={FIT_CLASS[fit.inference_fit]}>Inference: {FIT_TEXT[fit.inference_fit]}</span>
      <span class="text-xs text-surface-500">
        weights {fit.weights_gb.toFixed(1)} GB · training about {fit.training_memory_gb.toFixed(1)} GB of
        {fit.available_memory_gb.toFixed(0)} GB available
      </span>
    </div>
  {/if}

  {#if peek}
    <div
      class="rounded-lg bg-surface-100 dark:bg-surface-700/50 p-3 text-xs text-surface-600 dark:text-surface-300"
      aria-live="polite"
    >
      {peek.rows_sampled} rows sampled · about {peek.avg_tokens_estimate} tokens a row on average, up to
      {peek.max_tokens_estimate}.
      {#if lengthField}
        <button type="button" class="underline ml-1" onclick={applyLength}>
          Use {peek.suggested_seq_len} for {form.fields.find((f) => f.name === lengthField)?.label}
        </button>
      {/if}
    </div>
  {:else if peekError}
    <p role="alert" class="text-xs text-red-600 dark:text-red-400">{peekError}</p>
  {/if}

  {#each shown as group (group.name)}
    <details open={open(group)} class="rounded-lg border border-surface-200 dark:border-surface-700">
      <summary
        class="cursor-pointer select-none px-4 py-2 text-sm font-semibold text-surface-700 dark:text-surface-200"
      >
        {group.name}
        {#if group.fields.some((f) => f.name in issues)}
          <span class="badge-danger ml-2">needs attention</span>
        {/if}
      </summary>
      <div class="grid grid-cols-1 gap-4 p-4 md:grid-cols-2">
        {#each group.fields as field (field.name)}
          <div class={field.input.type === 'model' || field.input.type === 'dataset' || field.input.type === 'path' ? 'md:col-span-2' : ''}>
            <Field
              {field}
              bind:value={values[field.name]}
              error={issues[field.name]}
              suggestions={isColumnField(field.name) ? columnSuggestions : undefined}
              modelList={modelListId}
              datasetList={datasetListId}
            />
          </div>
        {/each}
      </div>
    </details>
  {:else}
    <p class="text-sm text-surface-500">No option matches “{filter}”.</p>
  {/each}
</div>
