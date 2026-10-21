<script lang="ts">
  import type { StepPoint } from '#lib/api.js';
  import { formatMetric } from '#lib/utils.js';
  import { chartPath, downsample } from '#lib/chart.js';

  interface Props {
    points: StepPoint[];
    height?: number;
  }

  let { points, height = 140 }: Props = $props();

  const width = 600;
  const pad = 6;
  const shown = $derived(downsample(points, 400));
  const path = $derived(chartPath(shown.map((p) => p.loss), width, height, pad));
  const losses = $derived(points.map((p) => p.loss).filter((l) => Number.isFinite(l)));
  const low = $derived(losses.length ? Math.min(...losses) : null);
  const high = $derived(losses.length ? Math.max(...losses) : null);
  const last = $derived(points.length ? points[points.length - 1] : null);
</script>

{#if points.length >= 2}
  <figure class="m-0">
    <svg
      viewBox={`0 0 ${width} ${height}`}
      class="w-full text-primary-600 dark:text-primary-400"
      role="img"
      aria-label={`Loss over ${points.length} steps, from ${formatMetric(points[0].loss)} to ${formatMetric(last?.loss)}; lowest ${formatMetric(low)}`}
      preserveAspectRatio="none"
      style={`height: ${height}px`}
    >
      <path d={path} fill="none" stroke="currentColor" stroke-width="2" vector-effect="non-scaling-stroke" />
    </svg>
    <figcaption class="mt-1 flex justify-between text-xs text-surface-500">
      <span>step {points[0].step}</span>
      <span>loss {formatMetric(low, 3)} to {formatMetric(high, 3)}</span>
      <span>step {last?.step}</span>
    </figcaption>
  </figure>
{:else}
  <p class="text-sm text-surface-500">The loss chart appears after the second step.</p>
{/if}
