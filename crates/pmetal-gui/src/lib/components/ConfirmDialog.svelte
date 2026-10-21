<script lang="ts">
  import type { Snippet } from 'svelte';

  interface Props {
    open: boolean;
    title: string;
    confirmLabel?: string;
    danger?: boolean;
    busy?: boolean;
    error?: string | null;
    onconfirm: () => void;
    oncancel: () => void;
    children: Snippet;
  }

  let {
    open,
    title,
    confirmLabel = 'Confirm',
    danger = false,
    busy = false,
    error = null,
    onconfirm,
    oncancel,
    children,
  }: Props = $props();

  let dialog = $state<HTMLDialogElement | null>(null);

  // A native modal dialog: focus stays inside it, Escape closes it, and the
  // page behind is inert.
  $effect(() => {
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  });
</script>

<dialog
  bind:this={dialog}
  class="rounded-xl p-0 backdrop:bg-black/50 bg-white dark:bg-surface-800 text-surface-900 dark:text-surface-100 shadow-2xl border border-surface-200 dark:border-surface-700 max-w-md w-full"
  aria-labelledby="confirm-title"
  oncancel={(e) => {
    e.preventDefault();
    if (!busy) oncancel();
  }}
>
  <div class="p-6 space-y-4">
    <h2 id="confirm-title" class="text-lg font-semibold">{title}</h2>
    <div class="text-sm text-surface-600 dark:text-surface-300 space-y-2">
      {@render children()}
    </div>
    {#if error}
      <p role="alert" class="text-sm text-red-600 dark:text-red-400">{error}</p>
    {/if}
    <div class="flex justify-end gap-2">
      <button type="button" class="btn-secondary" onclick={oncancel} disabled={busy}>Cancel</button>
      <button
        type="button"
        class={danger ? 'btn-danger' : 'btn-primary'}
        onclick={onconfirm}
        disabled={busy}
      >
        {busy ? 'Working…' : confirmLabel}
      </button>
    </div>
  </div>
</dialog>
