<script lang="ts">
  import type { FieldInfo } from '#lib/api.js';
  import { pickDirectory, pickFile, errorMessage } from '#lib/api.js';

  interface Props {
    field: FieldInfo;
    value: string;
    error?: string;
    /** Values to offer as the person types (columns of the dataset, say). */
    suggestions?: string[];
    /** `<datalist>` ids for the models and datasets already on this Mac. */
    modelList?: string;
    datasetList?: string;
    onchange?: () => void;
  }

  let {
    field,
    value = $bindable(),
    error,
    suggestions,
    modelList,
    datasetList,
    onchange,
  }: Props = $props();

  const id = $derived(`field-${field.name}`);
  const helpId = $derived(`${id}-help`);
  const errorId = $derived(`${id}-error`);
  const describedBy = $derived(
    [field.help ? helpId : '', error ? errorId : ''].filter(Boolean).join(' ') || undefined,
  );
  const input = $derived(field.input);
  const canBrowse = $derived(
    input.type === 'path' || input.type === 'dataset' || input.type === 'model',
  );
  const listId = $derived(
    suggestions?.length
      ? `${id}-suggestions`
      : input.type === 'model'
        ? modelList
        : input.type === 'dataset'
          ? datasetList
          : undefined,
  );
  let browseError = $state<string | null>(null);

  function set(next: string) {
    value = next;
    onchange?.();
  }

  async function browse(kind: 'folder' | 'file') {
    browseError = null;
    try {
      const picked =
        kind === 'folder'
          ? await pickDirectory(`Choose ${field.label}`)
          : await pickFile(`Choose ${field.label}`);
      if (picked) set(picked);
    } catch (e) {
      browseError = errorMessage(e);
    }
  }
</script>

<div class="space-y-1">
  {#if input.type === 'toggle'}
    <label class="flex items-start gap-3 cursor-pointer" for={id}>
      <input
        {id}
        type="checkbox"
        class="mt-1 h-4 w-4 rounded border-surface-300 text-primary-600 focus:ring-primary-500"
        checked={value === 'true'}
        aria-describedby={describedBy}
        onchange={(e) => set(e.currentTarget.checked ? 'true' : 'false')}
      />
      <span>
        <span class="text-sm font-medium text-surface-700 dark:text-surface-300">{field.label}</span>
      </span>
    </label>
  {:else}
    <label for={id} class="label">
      {field.label}{#if field.required}<span class="text-red-600" aria-hidden="true"> *</span>{/if}
      {#if field.list}<span class="font-normal text-surface-400"> (comma-separated)</span>{/if}
    </label>

    {#if input.type === 'choice'}
      <select
        {id}
        class="input"
        {value}
        aria-describedby={describedBy}
        aria-invalid={error ? 'true' : undefined}
        aria-required={field.required || undefined}
        onchange={(e) => set(e.currentTarget.value)}
      >
        {#if field.optional || !field.required}
          <option value="">{field.default === '' ? '(default)' : '(none)'}</option>
        {/if}
        {#each input.options as option}
          <option value={option}>{option}</option>
        {/each}
      </select>
    {:else if input.type === 'readonly'}
      <p id={id} class="text-sm text-surface-600 dark:text-surface-400">{value || field.default}</p>
    {:else}
      <div class="flex gap-2">
        <input
          {id}
          class="input"
          class:border-red-500={!!error}
          type={input.type === 'number' || input.type === 'integer' ? 'number' : 'text'}
          step={input.type === 'integer' ? 1 : input.type === 'number' ? 'any' : undefined}
          min={input.type === 'number' || input.type === 'integer' ? (input.min ?? undefined) : undefined}
          max={input.type === 'number' || input.type === 'integer' ? (input.max ?? undefined) : undefined}
          list={listId}
          autocomplete="off"
          autocapitalize="off"
          spellcheck="false"
          placeholder={field.optional && !field.default ? 'Optional' : undefined}
          {value}
          aria-describedby={describedBy}
          aria-invalid={error ? 'true' : undefined}
          aria-required={field.required || undefined}
          oninput={(e) => set(e.currentTarget.value)}
        />
        {#if canBrowse}
          <button
            type="button"
            class="btn-secondary btn-sm shrink-0"
            onclick={() => browse('folder')}
            aria-label={`Choose a folder for ${field.label}`}
          >
            Folder
          </button>
          {#if input.type !== 'model'}
            <button
              type="button"
              class="btn-secondary btn-sm shrink-0"
              onclick={() => browse('file')}
              aria-label={`Choose a file for ${field.label}`}
            >
              File
            </button>
          {/if}
        {/if}
      </div>
      {#if suggestions?.length}
        <datalist id={`${id}-suggestions`}>
          {#each suggestions as suggestion}
            <option value={suggestion}></option>
          {/each}
        </datalist>
      {/if}
    {/if}
  {/if}

  {#if field.help}
    <p id={helpId} class="text-xs text-surface-500 dark:text-surface-400">{field.help}</p>
  {/if}
  {#if error}
    <p id={errorId} role="alert" class="text-xs text-red-600 dark:text-red-400">{error}</p>
  {/if}
  {#if browseError}
    <p role="alert" class="text-xs text-red-600 dark:text-red-400">{browseError}</p>
  {/if}
</div>
