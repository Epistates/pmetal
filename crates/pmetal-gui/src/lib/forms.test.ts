import { describe, expect, test } from 'bun:test';
import type { FieldInfo, FormInfo } from './api';
import {
  changedFields,
  defaultsOf,
  groupsOf,
  isColumnField,
  isLocalPath,
  looksLikeHubId,
  matchesFilter,
  seqLenField,
} from './forms';

function field(name: string, over: Partial<FieldInfo> = {}): FieldInfo {
  return {
    name,
    label: name.replace(/_/g, ' '),
    help: null,
    group: 'Training',
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
  description: '',
  subcommand: 'train',
  fields: [
    field('model', { group: 'Model', required: true, input: { type: 'model' } }),
    field('dataset', { group: 'Model', required: true, input: { type: 'dataset' } }),
    field('epochs', { default: '3', input: { type: 'integer', min: 1, max: 100 } }),
    field('learning_rate', { default: '0.0002', group: 'Optimization' }),
    field('max_seq_len', { default: '2048', group: 'Optimization' }),
    field('text_column', { optional: true, group: 'Data' }),
  ],
};

describe('defaultsOf', () => {
  test('is exactly the text the backend sent for each field', () => {
    expect(defaultsOf(form)).toEqual({
      model: '',
      dataset: '',
      epochs: '3',
      learning_rate: '0.0002',
      max_seq_len: '2048',
      text_column: '',
    });
  });
});

describe('groupsOf', () => {
  test('keeps the order groups first appear in, and marks the ones with a required field', () => {
    const groups = groupsOf(form);
    expect(groups.map((g) => g.name)).toEqual(['Model', 'Training', 'Optimization', 'Data']);
    expect(groups.map((g) => g.required)).toEqual([true, false, false, false]);
  });

  test('leaves out hidden fields and a group left empty by it', () => {
    const groups = groupsOf(form, ['text_column']);
    expect(groups.map((g) => g.name)).not.toContain('Data');
  });
});

describe('matchesFilter', () => {
  test('matches the label, the name, the help and the group, ignoring case', () => {
    const f = field('learning_rate', { label: 'Learning Rate', help: 'Peak step size' });
    expect(matchesFilter(f, '')).toBe(true);
    expect(matchesFilter(f, 'LEARNING')).toBe(true);
    expect(matchesFilter(f, 'step size')).toBe(true);
    expect(matchesFilter(f, 'training')).toBe(true);
    expect(matchesFilter(f, 'dropout')).toBe(false);
  });
});

describe('changedFields', () => {
  test('lists only fields whose text is not the default', () => {
    const values = { ...defaultsOf(form), epochs: '5', model: 'Qwen/Qwen3-0.6B' };
    expect(changedFields(form, values).map((f) => f.name)).toEqual(['model', 'epochs']);
  });
});

describe('helpers', () => {
  test('column fields are found by name', () => {
    for (const name of ['text_column', 'text_columns', 'prompt_column', 'response_column']) {
      expect(isColumnField(name)).toBe(true);
    }
    for (const name of ['column_separator', 'columns_total', 'model']) {
      expect(isColumnField(name)).toBe(false);
    }
  });

  test('the longest-sequence field is whichever the form has', () => {
    expect(seqLenField(form)).toBe('max_seq_len');
    expect(seqLenField({ ...form, fields: [field('seq_len')] })).toBe('seq_len');
    expect(seqLenField({ ...form, fields: [field('epochs')] })).toBeNull();
  });

  test('a Hub id is owner/name, optionally with a quantization after a colon', () => {
    expect(looksLikeHubId('Qwen/Qwen3-0.6B')).toBe(true);
    expect(looksLikeHubId('unsloth/gemma-3-GGUF:Q4_K_M')).toBe(true);
    expect(looksLikeHubId('Qwen/')).toBe(false);
    expect(looksLikeHubId('Qwen')).toBe(false);
    expect(looksLikeHubId('/Users/me/model')).toBe(false);
  });

  test('only a full path is a path on disk', () => {
    expect(isLocalPath('/Users/me/data.jsonl')).toBe(true);
    expect(isLocalPath('  /data ')).toBe(true);
    expect(isLocalPath('tatsu-lab/alpaca')).toBe(false);
    expect(isLocalPath('~/data')).toBe(false);
  });
});
