import { describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { navJobKeys, SECTIONS } from './nav';

/** The job keys the specs produced (the generated file lists them). */
function generatedJobKeys(): string[] {
  const text = readFileSync(new URL('./specs.generated.ts', import.meta.url), 'utf8');
  const block = /export interface JobFields \{([\s\S]*?)\n\}/.exec(text)?.[1] ?? '';
  return [...block.matchAll(/^ {2}(\w+):/gm)].map((m) => m[1]);
}

describe('the sidebar', () => {
  test('links to every job the specs have, once', () => {
    const linked = navJobKeys();
    expect(new Set(linked).size).toBe(linked.length);
    expect([...linked].sort()).toEqual(generatedJobKeys().sort());
  });

  test('has no two items for one address', () => {
    const paths = SECTIONS.flatMap((s) => s.items.map((i) => i.path));
    expect(new Set(paths).size).toBe(paths.length);
  });

  test('gives every item an icon', () => {
    for (const item of SECTIONS.flatMap((s) => s.items)) {
      expect(item.icon.length).toBeGreaterThan(10);
    }
  });
});
