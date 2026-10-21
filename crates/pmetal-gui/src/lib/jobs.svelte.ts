/**
 * Every job, in one store.
 *
 * The backend owns the jobs (their processes, output and progress) and tells
 * the webview what changed. This store holds what the pages draw; it survives
 * navigation because it lives for the life of the window, and it can always be
 * rebuilt from the backend (`refresh`), which is what a missed event does.
 */

import * as api from './api';
import type { JobKey, JobValues, JobView, OutputLine, StepPoint } from './api';

/** Lines of output kept per job in the window. */
const MAX_LINES = 2000;

class JobsStore {
  /** Newest first. */
  jobs = $state<JobView[]>([]);
  loaded = $state(false);
  error = $state<string | null>(null);
  /** Output of jobs somebody looked at, by job id. */
  logs = $state<Record<string, OutputLine[]>>({});
  /** Steps of jobs somebody looked at, for the chart. */
  steps = $state<Record<string, StepPoint[]>>({});
  #stepsNext: Record<string, number> = {};
  #stepsBusy = new Set<string>();

  get active(): JobView[] {
    return this.jobs.filter((j) => j.state === 'running' || j.state === 'stopping');
  }

  of(kind: JobKey): JobView[] {
    return this.jobs.filter((j) => j.kind === kind);
  }

  get(id: string | null): JobView | null {
    return id ? (this.jobs.find((j) => j.id === id) ?? null) : null;
  }

  async refresh() {
    try {
      const jobs = await api.listJobs();
      this.jobs = jobs;
      this.error = null;
      // What was being looked at is read again, in case events were missed.
      for (const id of Object.keys(this.logs)) void this.loadLogs(id);
      for (const id of Object.keys(this.#stepsNext)) void this.pullSteps(id);
    } catch (e) {
      this.error = api.errorMessage(e);
    } finally {
      this.loaded = true;
    }
  }

  /** A job changed. */
  upsert(job: JobView) {
    const at = this.jobs.findIndex((j) => j.id === job.id);
    const before = at >= 0 ? this.jobs[at] : null;
    if (at >= 0) {
      this.jobs[at] = job;
    } else {
      this.jobs = [job, ...this.jobs];
    }
    if (job.id in this.#stepsNext && before?.step !== job.step) {
      void this.pullSteps(job.id);
    }
  }

  appendOutput(id: string, lines: OutputLine[]) {
    const have = this.logs[id];
    if (!have) return; // not being looked at; read in full when it is
    const last = have.length ? have[have.length - 1].seq : -1;
    const fresh = lines.filter((l) => l.seq > last);
    if (!fresh.length) return;
    const next = have.concat(fresh);
    this.logs[id] = next.length > MAX_LINES ? next.slice(next.length - MAX_LINES) : next;
  }

  /** Read a job's output so it can be shown and followed. */
  async loadLogs(id: string) {
    try {
      const page = await api.jobOutput(id, MAX_LINES);
      this.logs[id] = page.lines;
    } catch {
      // The job was removed; there is nothing to show.
      delete this.logs[id];
    }
  }

  /** Read a job's steps so they can be charted and followed. */
  async loadSteps(id: string) {
    if (!(id in this.#stepsNext)) {
      this.#stepsNext[id] = 0;
      this.steps[id] = [];
    }
    await this.pullSteps(id);
  }

  async pullSteps(id: string) {
    if (this.#stepsBusy.has(id)) return;
    this.#stepsBusy.add(id);
    try {
      const page = await api.jobSteps(id, this.#stepsNext[id] ?? 0);
      this.#stepsNext[id] = page.next_since;
      if (page.steps.length) this.steps[id] = (this.steps[id] ?? []).concat(page.steps);
    } catch {
      // The job was removed.
    } finally {
      this.#stepsBusy.delete(id);
    }
  }

  async start<K extends JobKey>(kind: K, values: JobValues<K>): Promise<JobView> {
    const job = await api.startJob(kind, values);
    this.upsert(job);
    return job;
  }

  async stop(id: string) {
    await api.stopJob(id);
    const job = this.get(id);
    if (job && job.state === 'running') this.upsert({ ...job, state: 'stopping' });
  }

  async forget(id: string) {
    await api.forgetJob(id);
    this.jobs = this.jobs.filter((j) => j.id !== id);
    delete this.logs[id];
    delete this.steps[id];
    delete this.#stepsNext[id];
  }

  /** Stop looking at a job's output and steps (they stop being followed). */
  release(id: string) {
    delete this.logs[id];
    delete this.steps[id];
    delete this.#stepsNext[id];
  }
}

export const jobsStore = new JobsStore();
