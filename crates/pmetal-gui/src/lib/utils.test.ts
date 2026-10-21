import { describe, expect, test } from 'bun:test';
import { formatDuration, formatEta, formatMetric, getStatusBadgeClass, renderMarkdown } from './utils';

describe('renderMarkdown', () => {
  test('cannot be made to run a script or load a resource', () => {
    const attacks = [
      '<script>alert(1)</script>',
      '<img src=x onerror=alert(1)>',
      '[click](javascript:alert(1))',
      '**<b onmouseover=alert(1)>x</b>**',
      '`<svg onload=alert(1)>`',
      '```\n<iframe src="https://example.com"></iframe>\n```',
      '# <style>body{display:none}</style>',
    ];
    for (const attack of attacks) {
      const html = renderMarkdown(attack);
      // Every tag in the output is one the renderer itself writes; whatever the
      // text said is only text.
      expect(html).not.toMatch(/<(?!\/?(p|br|h1|h2|h3|li|strong|em|pre|code)[\s>])/);
    }
  });

  test('renders the basics', () => {
    expect(renderMarkdown('**bold** and `code`')).toContain('<strong>bold</strong>');
    expect(renderMarkdown('**bold** and `code`')).toContain('<code class="inline-code">code</code>');
    expect(renderMarkdown('- one\n- two')).toContain('<li class="md-li">one</li>');
  });
});

describe('formatting', () => {
  test('durations', () => {
    expect(formatDuration(5)).toBe('5s');
    expect(formatDuration(125)).toBe('2m 05s');
    expect(formatDuration(3720)).toBe('1h 02m');
    expect(formatDuration(-3)).toBe('0s');
  });

  test('eta of nothing is a dash', () => {
    expect(formatEta(null)).toBe('--');
    expect(formatEta(0)).toBe('--');
    expect(formatEta(90)).toBe('1m 30s');
  });

  test('metrics', () => {
    expect(formatMetric(null)).toBe('--');
    expect(formatMetric(Number.NaN)).toBe('--');
    expect(formatMetric(2.5)).toBe('2.5000');
    expect(formatMetric(0.0002)).toBe('2.00e-4');
    expect(formatMetric(1234.5, 0)).toBe('1235');
  });

  test('every job and download state has a badge', () => {
    for (const state of ['running', 'stopping', 'completed', 'failed', 'stopped', 'queued', 'downloading', 'cancelled']) {
      expect(getStatusBadgeClass(state)).not.toBe('badge-neutral');
    }
  });
});
