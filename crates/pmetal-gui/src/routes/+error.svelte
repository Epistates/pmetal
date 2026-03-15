<script lang="ts">
  import { page } from '$app/stores';

  let status = $derived($page.status);
  let message = $derived($page.error?.message ?? 'An unexpected error occurred.');
</script>

<div class="flex items-center justify-center min-h-screen bg-surface-50 dark:bg-surface-900 p-6">
  <div class="text-center max-w-md">
    <!-- Error icon -->
    <div class="w-20 h-20 rounded-2xl bg-red-100 dark:bg-red-900/30 flex items-center justify-center mx-auto mb-6">
      <svg class="w-10 h-10 text-red-600 dark:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    </div>

    <!-- Status code -->
    {#if status && status !== 200}
      <p class="text-6xl font-bold text-surface-300 dark:text-surface-600 mb-2" aria-hidden="true">{status}</p>
    {/if}

    <!-- Title -->
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100 mb-2">
      {#if status === 404}
        Page Not Found
      {:else if status === 500}
        Internal Error
      {:else}
        Something went wrong
      {/if}
    </h1>

    <!-- Message -->
    <p class="text-surface-500 dark:text-surface-400 mb-6 leading-relaxed">{message}</p>

    <!-- Actions -->
    <div class="flex gap-3 justify-center">
      <a href="/" class="btn-primary">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
        </svg>
        Go to Dashboard
      </a>
      <button class="btn-secondary" onclick={() => window.history.back()}>
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        Go Back
      </button>
    </div>
  </div>
</div>
