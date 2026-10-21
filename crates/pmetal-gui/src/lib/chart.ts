/** Pure helpers for the loss chart. */

/** At most `limit` points, evenly spread, always keeping the first and last. */
export function downsample<T>(points: readonly T[], limit: number): T[] {
  if (points.length <= limit || limit < 2) return points.slice();
  const out: T[] = [];
  const step = (points.length - 1) / (limit - 1);
  for (let i = 0; i < limit; i++) out.push(points[Math.round(i * step)]);
  return out;
}

/** An SVG path through `values`, scaled to fill `width` x `height` less `pad`. */
export function chartPath(values: readonly number[], width: number, height: number, pad: number): string {
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length < 2) return '';
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  // A flat series is a line across the middle, not a division by zero.
  const range = max - min || 1;
  const n = values.length;
  const commands: string[] = [];
  values.forEach((v, i) => {
    if (!Number.isFinite(v)) return;
    const x = pad + (i / (n - 1)) * (width - 2 * pad);
    const y = max === min ? height / 2 : pad + (1 - (v - min) / range) * (height - 2 * pad);
    commands.push(`${commands.length === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`);
  });
  return commands.join(' ');
}
