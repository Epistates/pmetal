import { describe, expect, test } from 'bun:test';
import { chartPath, downsample } from './chart';

describe('downsample', () => {
  test('returns a short series as it is', () => {
    expect(downsample([1, 2, 3], 10)).toEqual([1, 2, 3]);
  });

  test('keeps the first and last of a long one and the order between', () => {
    const series = Array.from({ length: 1000 }, (_, i) => i);
    const out = downsample(series, 5);
    expect(out).toHaveLength(5);
    expect(out[0]).toBe(0);
    expect(out[4]).toBe(999);
    expect([...out].sort((a, b) => a - b)).toEqual(out);
  });
});

describe('chartPath', () => {
  test('needs two points to draw a line', () => {
    expect(chartPath([1], 100, 50, 2)).toBe('');
    expect(chartPath([], 100, 50, 2)).toBe('');
  });

  test('puts the highest value at the top and the lowest at the bottom', () => {
    expect(chartPath([2, 1], 100, 50, 0)).toBe('M0.0,0.0 L100.0,50.0');
  });

  test('draws a constant series across the middle, not as NaN', () => {
    const path = chartPath([3, 3, 3], 100, 50, 0);
    expect(path).not.toContain('NaN');
    expect(path).toBe('M0.0,25.0 L50.0,25.0 L100.0,25.0');
  });

  test('skips values that are not numbers', () => {
    const path = chartPath([1, Number.NaN, 2], 100, 50, 0);
    expect(path).not.toContain('NaN');
    expect(path.startsWith('M')).toBe(true);
  });
});
