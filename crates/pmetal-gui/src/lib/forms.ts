/**
 * Helpers over a job's form (the backend's description of a spec's fields).
 * Nothing here knows a field name, a default or a range: it all arrives in
 * the {@link FormInfo}.
 */

import type { FieldInfo, FormInfo } from './api';

/** The text each field starts with: the command line's own defaults. */
export function defaultsOf(form: FormInfo): Record<string, string> {
  return Object.fromEntries(form.fields.map((f) => [f.name, f.default]));
}

export interface FieldGroup {
  name: string;
  fields: FieldInfo[];
  /** Has a field that must be filled in. */
  required: boolean;
}

/** The form's fields by group, in the order the groups first appear. */
export function groupsOf(form: FormInfo, hidden: readonly string[] = []): FieldGroup[] {
  const groups: FieldGroup[] = [];
  for (const field of form.fields) {
    if (hidden.includes(field.name)) continue;
    let group = groups.find((g) => g.name === field.group);
    if (!group) {
      group = { name: field.group, fields: [], required: false };
      groups.push(group);
    }
    group.fields.push(field);
    group.required ||= field.required;
  }
  return groups;
}

/** Whether `field` matches what a person typed into the filter box. */
export function matchesFilter(field: FieldInfo, filter: string): boolean {
  const needle = filter.trim().toLowerCase();
  if (!needle) return true;
  return [field.label, field.name, field.help ?? '', field.group].some((text) =>
    text.toLowerCase().includes(needle),
  );
}

/** The fields whose text is not the default, for a summary of what was changed. */
export function changedFields(form: FormInfo, values: Record<string, string>): FieldInfo[] {
  return form.fields.filter((f) => (values[f.name] ?? f.default) !== f.default);
}

/** The field a dataset's columns belong to, by name. */
export function isColumnField(name: string): boolean {
  return /(^|_)column(s)?$/.test(name);
}

/** The field that holds the longest sequence, if the form has one. */
export function seqLenField(form: FormInfo): string | null {
  for (const name of ['max_seq_len', 'seq_len', 'max_length']) {
    if (form.fields.some((f) => f.name === name)) return name;
  }
  return null;
}

/** Whether `text` is a full path on disk (as opposed to a Hub id). */
export function isLocalPath(text: string): boolean {
  return text.trim().startsWith('/');
}
