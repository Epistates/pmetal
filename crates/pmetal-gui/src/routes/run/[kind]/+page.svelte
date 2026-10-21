<script lang="ts">
  import { page } from '$app/state';
  import type { JobKey } from '#lib/api.js';
  import JobPage from '#lib/components/JobPage.svelte';

  const kind = $derived(page.params.kind as JobKey);
</script>

{#key kind}
  <JobPage {kind} />
{/key}
