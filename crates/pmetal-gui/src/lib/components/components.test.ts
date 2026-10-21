/**
 * The components, rendered on the server side of Svelte with real data shapes.
 * It does not run effects or events; it catches a template that cannot render
 * at all (a field read off something that is not there, a missing prop).
 */

import { describe, expect, test } from 'bun:test';
import { render } from 'svelte/server';
import type { FieldInfo, FormInfo, JobView } from '../api';
import Field from './Field.svelte';
import SpecForm from './SpecForm.svelte';
import JobMonitor from './JobMonitor.svelte';
import LossChart from './LossChart.svelte';
import ConfirmDialog from './ConfirmDialog.svelte';

function field(name: string, over: Partial<FieldInfo> = {}): FieldInfo {
  return {
    name,
    label: name,
    help: null,
    group: 'General',
    input: { type: 'text' },
    required: false,
    list: false,
    optional: false,
    default: '',
    ...over,
  };
}

const form: FormInfo = {
  key: 'train',
  label: 'Fine-tune',
  description: 'd',
  subcommand: 'train',
  fields: [
    field('model', { input: { type: 'model' }, required: true, label: 'Model' }),
    field('dataset', { input: { type: 'dataset' }, required: true, label: 'Dataset' }),
    field('epochs', { input: { type: 'integer', min: 1, max: 9 }, default: '3', label: 'Epochs', group: 'Training' }),
    field('learning_rate', { input: { type: 'number', min: 0, max: 1 }, default: '0.0002', label: 'Learning rate', group: 'Training' }),
    field('optimizer', { input: { type: 'choice', options: ['adamw', 'lion'] }, default: 'adamw', label: 'Optimizer', group: 'Training' }),
    field('dora', { input: { type: 'toggle' }, default: 'false', label: 'DoRA', group: 'LoRA' }),
    field('lora_targets', { list: true, optional: true, label: 'Targets', group: 'LoRA', help: 'Layer names' }),
  ],
};

const job: JobView = {
  id: 'j1',
  kind: 'train',
  label: 'Fine-tune',
  title: 'Qwen/Qwen3-0.6B',
  command: 'pmetal train --model Qwen/Qwen3-0.6B',
  state: 'running',
  error: null,
  exit_code: null,
  phase: null,
  step: 5,
  total_steps: 10,
  epoch: 1,
  total_epochs: 2,
  loss: 2.5,
  best_loss: 2.0,
  eval_loss: null,
  learning_rate: 0.0002,
  grad_norm: 0.9,
  tokens_per_second: 1234,
  progress: 0.5,
  eta_seconds: 60,
  started_at: '2026-03-18T21:45:00Z',
  finished_at: null,
  elapsed_seconds: 75,
  output: '/out/run',
  serve: null,
  result: null,
};

describe('Field', () => {
  test('labels its control and ties help and errors to it', () => {
    const { body } = render(Field, {
      props: { field: field('epochs', { label: 'Epochs', help: 'How many passes', required: true, input: { type: 'integer', min: 1, max: 9 } }), value: '3', error: 'must be at least 1' },
    });
    expect(body).toContain('for="field-epochs"');
    expect(body).toContain('id="field-epochs"');
    expect(body).toContain('aria-describedby="field-epochs-help field-epochs-error"');
    expect(body).toContain('aria-invalid="true"');
    expect(body).toContain('role="alert"');
    expect(body).toContain('min="1"');
    expect(body).toContain('max="9"');
  });

  test('a toggle is a checkbox that is checked when its text is true', () => {
    const on = render(Field, { props: { field: field('dora', { input: { type: 'toggle' } }), value: 'true' } }).body;
    const off = render(Field, { props: { field: field('dora', { input: { type: 'toggle' } }), value: 'false' } }).body;
    expect(on).toContain('type="checkbox"');
    expect(on).toContain('checked');
    expect(off).not.toContain('checked');
  });

  test('a choice offers exactly the spec options', () => {
    const { body } = render(Field, {
      props: { field: field('optimizer', { input: { type: 'choice', options: ['adamw', 'lion'] }, required: true }), value: 'adamw' },
    });
    expect(body).toContain('<option value="adamw"');
    expect(body).toContain('<option value="lion"');
    expect(body).not.toContain('sgd');
  });
});

describe('SpecForm', () => {
  test('shows every field of the form, grouped, with the groups that need input open', () => {
    const values = Object.fromEntries(form.fields.map((f) => [f.name, f.default]));
    const { body } = render(SpecForm, { props: { form, values } });
    for (const f of form.fields) expect(body).toContain(`id="field-${f.name}"`);
    for (const group of ['General', 'Training', 'LoRA']) expect(body).toContain(group);
    // The group with required fields starts open, the others folded.
    expect(body).toMatch(/<details open[^>]*>\s*<summary[^>]*>\s*General/);
    expect(body).not.toMatch(/<details open[^>]*>\s*<summary[^>]*>\s*Training/);
  });

  test('hides what it is told to, and marks a group with a problem', () => {
    const values = Object.fromEntries(form.fields.map((f) => [f.name, f.default]));
    const { body } = render(SpecForm, { props: { form, values, hidden: ['dora'], issues: { epochs: 'bad' } } });
    expect(body).not.toContain('id="field-dora"');
    expect(body).toContain('needs attention');
    expect(body).toMatch(/<details open[^>]*>\s*<summary[^>]*>\s*Training/);
  });
});

describe('JobMonitor', () => {
  test('shows progress, metrics and the output region of a running job', () => {
    const { body } = render(JobMonitor, { props: { job } });
    expect(body).toContain('role="progressbar"');
    expect(body).toContain('aria-valuenow="50"');
    expect(body).toContain('Stop');
    expect(body).toContain('2.5000');
    expect(body).toContain('role="log"');
    expect(body).toContain('/out/run');
  });

  test('a failed job says why and offers to remove it', () => {
    const { body } = render(JobMonitor, {
      props: { job: { ...job, state: 'failed', error: 'Error: model not found', exit_code: 2, progress: null } },
    });
    expect(body).toContain('This job failed (exit code 2)');
    expect(body).toContain('model not found');
    expect(body).toContain('Remove');
    expect(body).not.toContain('>Stop<');
  });

  test('a server shows where it listens once it answers', () => {
    const waiting = render(JobMonitor, { props: { job: { ...job, kind: 'serve', serve: { url: 'http://localhost:8080', reachable: false } } } }).body;
    const ready = render(JobMonitor, { props: { job: { ...job, kind: 'serve', serve: { url: 'http://localhost:8080', reachable: true } } } }).body;
    expect(waiting).toContain('Waiting for the server');
    expect(ready).toContain('Serving at http://localhost:8080');
  });
});

describe('LossChart and ConfirmDialog', () => {
  test('the chart describes itself for a screen reader', () => {
    const points = [3, 2.5, 2].map((loss, i) => ({ step: i + 1, loss, learning_rate: 0.1, tokens_per_second: 1 }));
    const { body } = render(LossChart, { props: { points } });
    expect(body).toContain('role="img"');
    expect(body).toContain('aria-label="Loss over 3 steps');
  });

  test('the dialog is a labelled modal', () => {
    const { body } = render(ConfirmDialog, {
      props: { open: false, title: 'Stop?', onconfirm() {}, oncancel() {}, children: (() => {}) as never },
    });
    expect(body).toContain('<dialog');
    expect(body).toContain('aria-labelledby="confirm-title"');
  });
});
