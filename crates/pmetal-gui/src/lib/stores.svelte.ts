/**
 * Svelte 5 stores using runes for reactive state management.
 */

import * as api from './api';
import type {
  CachedModel,
  TrainingRun,
  GrpoRun,
  DistillationRun,
  DashboardStats,
  AppConfig,
  DeviceInfo,
  DatasetSearchResult,
  CachedDatasetInfo,
} from './api';

// =============================================================================
// Models Store
// =============================================================================

class ModelsStore {
  models = $state<CachedModel[]>([]);
  customDirs = $state<string[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      const [models, dirs] = await Promise.all([
        api.listModels(),
        api.listModelDirectories(),
      ]);
      this.models = models;
      this.customDirs = dirs;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  async download(modelId: string, revision?: string) {
    this.loading = true;
    this.error = null;
    try {
      await api.downloadModel(modelId, revision);
      await this.refresh();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  async delete(modelId: string) {
    try {
      await api.deleteModel(modelId);
      await this.refresh();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    }
  }

  async addDirectory(path: string) {
    try {
      this.models = await api.addModelDirectory(path);
      this.customDirs = await api.listModelDirectories();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    }
  }

  async removeDirectory(path: string) {
    try {
      this.models = await api.removeModelDirectory(path);
      this.customDirs = await api.listModelDirectories();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    }
  }
}

export const modelsStore = new ModelsStore();

// =============================================================================
// Datasets Store
// =============================================================================

class DatasetsStore {
  cached = $state<CachedDatasetInfo[]>([]);
  searchResults = $state<DatasetSearchResult[]>([]);
  trending = $state<DatasetSearchResult[]>([]);
  loading = $state(false);
  searching = $state(false);
  error = $state<string | null>(null);
  searchQuery = $state('');

  async refreshCached() {
    this.loading = true;
    this.error = null;
    try {
      this.cached = await api.listCachedDatasets();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  async search(query: string) {
    if (!query.trim()) {
      this.searchResults = [];
      return;
    }
    this.searching = true;
    this.searchQuery = query;
    this.error = null;
    try {
      this.searchResults = await api.searchHubDatasets(query, 20);
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.searching = false;
    }
  }

  async loadTrending() {
    this.error = null;
    try {
      this.trending = await api.getTrendingDatasets(10);
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    }
  }

  clearSearch() {
    this.searchResults = [];
    this.searchQuery = '';
  }
}

export const datasetsStore = new DatasetsStore();

// =============================================================================
// Training Store
// =============================================================================

class TrainingStore {
  runs = $state<TrainingRun[]>([]);
  selectedRunId = $state<string | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  get selectedRun() {
    return this.runs.find(r => r.id === this.selectedRunId) ?? null;
  }

  get activeRuns() {
    return this.runs.filter(r => r.status === 'running' || r.status === 'pending');
  }

  get completedRuns() {
    return this.runs.filter(r => r.status === 'completed');
  }

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.runs = await api.listTrainingRuns();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  async start(config: api.TrainSpec) {
    this.loading = true;
    this.error = null;
    try {
      const runId = await api.startTraining(config, (data) => {
        // Stream metrics from the backend Channel directly into the run
        const run = this.runs.find(r => r.id === runId);
        if (!run) return;

        if (data.event === 'train_start') {
          run.status_message = data.status_message as string;
        } else if (data.event === 'step') {
          run.status_message = null;
          if (data.step != null) run.step = data.step as number;
          if (data.total_steps != null) run.total_steps = data.total_steps as number;
          if (data.total_epochs != null) run.total_epochs = data.total_epochs as number;
          if (data.epoch != null) run.epoch = data.epoch as number;
          if (data.loss != null) run.loss = data.loss as number;
          if (data.best_loss != null) run.best_loss = data.best_loss as number;
          if (data.lr != null) run.learning_rate = data.lr as number;
          if (data.tok_sec != null) run.tokens_per_second = data.tok_sec as number;
          if (data.grad_norm != null) run.grad_norm = data.grad_norm as number;
          if (data.eta_seconds != null) run.eta_seconds = data.eta_seconds as number;
        }
        // Trigger Svelte 5 reactivity
        this.runs = [...this.runs];
      });
      this.selectedRunId = runId;
      await this.refresh();
      return runId;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  async stop(runId: string) {
    try {
      await api.stopTraining(runId);
      await this.refresh();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    }
  }

  updateRun(updatedRun: TrainingRun) {
    this.runs = this.runs.map(r => (r.id === updatedRun.id ? updatedRun : r));
  }

  addRun(run: TrainingRun) {
    this.runs = [...this.runs, run];
  }
}

export const trainingStore = new TrainingStore();

// =============================================================================
// Serve Store — tracks running `pmetal serve` HTTP instances.
// =============================================================================

class ServeStore {
  instances = $state<api.ServeInstance[]>([]);
  selectedId = $state<string | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  get selected() {
    return this.instances.find(i => i.id === this.selectedId) ?? null;
  }

  get running() {
    return this.instances.filter(
      i => i.status === 'running' || i.status === 'starting',
    );
  }

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.instances = await api.listServeInstances();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  async start(config: api.ServeSpec) {
    this.loading = true;
    this.error = null;
    try {
      const id = await api.startServe(config);
      this.selectedId = id;
      await this.refresh();
      return id;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  async stop(instanceId: string) {
    try {
      await api.stopServe(instanceId);
      await this.refresh();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    }
  }

  addInstance(instance: api.ServeInstance) {
    this.instances = [...this.instances, instance];
  }

  updateInstance(updated: api.ServeInstance) {
    this.instances = this.instances.map(i => (i.id === updated.id ? updated : i));
  }
}

export const serveStore = new ServeStore();

// =============================================================================
// Bench Store — tracks one-shot `pmetal bench` runs.
// =============================================================================

class BenchStore {
  runs = $state<api.BenchRun[]>([]);
  selectedId = $state<string | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  get selected() {
    return this.runs.find(r => r.id === this.selectedId) ?? null;
  }

  get activeRuns() {
    return this.runs.filter(r => r.status === 'running' || r.status === 'pending');
  }

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.runs = await api.listBenchRuns();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  async start(config: api.BenchSpec) {
    this.loading = true;
    this.error = null;
    try {
      const id = await api.startBench(config);
      this.selectedId = id;
      await this.refresh();
      return id;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  async stop(runId: string) {
    await api.stopBench(runId);
    await this.refresh();
  }

  addRun(run: api.BenchRun) {
    this.runs = [...this.runs, run];
  }

  updateRun(updated: api.BenchRun) {
    this.runs = this.runs.map(r => (r.id === updated.id ? updated : r));
  }
}

export const benchStore = new BenchStore();

// =============================================================================
// Eval Store — tracks one-shot `pmetal eval` runs.
// =============================================================================

class EvalStore {
  runs = $state<api.EvalRun[]>([]);
  selectedId = $state<string | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  get selected() {
    return this.runs.find(r => r.id === this.selectedId) ?? null;
  }

  get activeRuns() {
    return this.runs.filter(r => r.status === 'running' || r.status === 'pending');
  }

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.runs = await api.listEvalRuns();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  async start(config: api.EvalSpec) {
    this.loading = true;
    this.error = null;
    try {
      const id = await api.startEval(config);
      this.selectedId = id;
      await this.refresh();
      return id;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  async stop(runId: string) {
    await api.stopEval(runId);
    await this.refresh();
  }

  addRun(run: api.EvalRun) {
    this.runs = [...this.runs, run];
  }

  updateRun(updated: api.EvalRun) {
    this.runs = this.runs.map(r => (r.id === updated.id ? updated : r));
  }
}

export const evalStore = new EvalStore();

// =============================================================================
// Pretrain Store — tracks long-running `pmetal pretrain` runs.
// =============================================================================

class PretrainStore {
  runs = $state<api.PretrainRun[]>([]);
  selectedId = $state<string | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  get selected() {
    return this.runs.find(r => r.id === this.selectedId) ?? null;
  }

  get activeRuns() {
    return this.runs.filter(r => r.status === 'running' || r.status === 'pending');
  }

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.runs = await api.listPretrainRuns();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  async start(config: api.PretrainSpec) {
    this.loading = true;
    this.error = null;
    try {
      const id = await api.startPretrain(config);
      this.selectedId = id;
      await this.refresh();
      return id;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  async stop(runId: string) {
    await api.stopPretrain(runId);
    await this.refresh();
  }

  addRun(run: api.PretrainRun) {
    this.runs = [...this.runs, run];
  }

  updateRun(updated: api.PretrainRun) {
    this.runs = this.runs.map(r => (r.id === updated.id ? updated : r));
  }
}

export const pretrainStore = new PretrainStore();

// =============================================================================
// GRPO Store
// =============================================================================

class GrpoStore {
  runs = $state<GrpoRun[]>([]);
  selectedRunId = $state<string | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  get selectedRun() {
    return this.runs.find(r => r.id === this.selectedRunId) ?? null;
  }

  get activeRuns() {
    return this.runs.filter(r => r.status === 'running' || r.status === 'pending');
  }

  get completedRuns() {
    return this.runs.filter(r => r.status === 'completed');
  }

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.runs = await api.listGrpoRuns();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  async start(config: api.GrpoSpec) {
    this.loading = true;
    this.error = null;
    try {
      const runId = await api.startGrpo(config);
      this.selectedRunId = runId;
      await this.refresh();
      return runId;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  async stop(runId: string) {
    try {
      await api.stopGrpo(runId);
      await this.refresh();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    }
  }

  updateRun(updatedRun: GrpoRun) {
    this.runs = this.runs.map(r => (r.id === updatedRun.id ? updatedRun : r));
  }

  addRun(run: GrpoRun) {
    this.runs = [...this.runs, run];
  }
}

export const grpoStore = new GrpoStore();

// =============================================================================
// Distillation Store
// =============================================================================

class DistillationStore {
  runs = $state<DistillationRun[]>([]);
  selectedRunId = $state<string | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  get selectedRun() {
    return this.runs.find(r => r.id === this.selectedRunId) ?? null;
  }

  get activeRuns() {
    return this.runs.filter(
      r =>
        r.status === 'pending' ||
        r.status === 'loading_models' ||
        r.status === 'generating_signals' ||
        r.status === 'training'
    );
  }

  get completedRuns() {
    return this.runs.filter(r => r.status === 'completed');
  }

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.runs = await api.listDistillationRuns();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  async start(config: api.DistillSpec) {
    this.loading = true;
    this.error = null;
    try {
      const runId = await api.startDistillation(config);
      this.selectedRunId = runId;
      await this.refresh();
      return runId;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  async stop(runId: string) {
    try {
      await api.stopDistillation(runId);
      await this.refresh();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    }
  }

  updateRun(updatedRun: DistillationRun) {
    this.runs = this.runs.map(r => (r.id === updatedRun.id ? updatedRun : r));
  }

  addRun(run: DistillationRun) {
    this.runs = [...this.runs, run];
  }
}

export const distillationStore = new DistillationStore();

// =============================================================================
// Dashboard Store
// =============================================================================

class DashboardStore {
  stats = $state<DashboardStats | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.stats = await api.getDashboardStats();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }
}

export const dashboardStore = new DashboardStore();

// =============================================================================
// Config Store
// =============================================================================

class ConfigStore {
  config = $state<AppConfig | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  async load() {
    this.loading = true;
    this.error = null;
    try {
      this.config = await api.getConfig();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }

  async save(config: AppConfig) {
    this.loading = true;
    this.error = null;
    try {
      await api.setConfig(config);
      this.config = config;
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    } finally {
      this.loading = false;
    }
  }

  get theme() {
    return this.config?.theme ?? 'system';
  }
}

export const configStore = new ConfigStore();

// =============================================================================
// Device Store
// =============================================================================

class DeviceStore {
  info = $state<DeviceInfo | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.info = await api.getDeviceInfo();
    } catch (e) {
      this.error = e instanceof Error ? e.message : String(e);
    } finally {
      this.loading = false;
    }
  }
}

export const deviceStore = new DeviceStore();

// =============================================================================
// Initialize stores
// =============================================================================

let unlistenFns: (() => void)[] = [];

export async function initializeStores() {
  await Promise.all([
    configStore.load(),
    modelsStore.refresh(),
    datasetsStore.refreshCached(),
    datasetsStore.loadTrending(),
    trainingStore.refresh(),
    grpoStore.refresh(),
    distillationStore.refresh(),
    serveStore.refresh(),
    benchStore.refresh(),
    evalStore.refresh(),
    pretrainStore.refresh(),
    dashboardStore.refresh(),
    deviceStore.refresh(),
  ]);

  // Set up event listeners for training updates and store unlisten functions
  unlistenFns.push(await api.onTrainingStarted((run) => {
    trainingStore.addRun(run);
    dashboardStore.refresh();
  }));

  unlistenFns.push(await api.onTrainingUpdate((run) => {
    trainingStore.updateRun(run);
  }));

  unlistenFns.push(await api.onTrainingStopped(() => {
    trainingStore.refresh();
    dashboardStore.refresh();
  }));

  // Set up event listeners for GRPO updates
  unlistenFns.push(await api.onGrpoStarted((run) => {
    grpoStore.addRun(run);
    dashboardStore.refresh();
  }));

  unlistenFns.push(await api.onGrpoUpdate((run) => {
    grpoStore.updateRun(run);
  }));

  unlistenFns.push(await api.onGrpoStopped(() => {
    grpoStore.refresh();
    dashboardStore.refresh();
  }));

  // Set up event listeners for distillation updates
  unlistenFns.push(await api.onDistillationStarted((run) => {
    distillationStore.addRun(run);
    dashboardStore.refresh();
  }));

  unlistenFns.push(await api.onDistillationUpdate((run) => {
    distillationStore.updateRun(run);
  }));

  unlistenFns.push(await api.onDistillationStopped(() => {
    distillationStore.refresh();
    dashboardStore.refresh();
  }));

  // Set up event listeners for serve instances. Each `serve-update` event
  // carries the full instance (with refreshed log_tail) so the store
  // replaces the entry rather than mutating in place.
  unlistenFns.push(await api.onServeStarted((instance) => {
    serveStore.addInstance(instance);
    dashboardStore.refresh();
  }));

  unlistenFns.push(await api.onServeUpdate((instance) => {
    serveStore.updateInstance(instance);
  }));

  unlistenFns.push(await api.onServeStopped(() => {
    serveStore.refresh();
    dashboardStore.refresh();
  }));

  // Bench
  unlistenFns.push(await api.onBenchStarted((run) => {
    benchStore.addRun(run);
    dashboardStore.refresh();
  }));
  unlistenFns.push(await api.onBenchUpdate((run) => benchStore.updateRun(run)));
  unlistenFns.push(await api.onBenchStopped(() => {
    benchStore.refresh();
    dashboardStore.refresh();
  }));

  // Eval
  unlistenFns.push(await api.onEvalStarted((run) => {
    evalStore.addRun(run);
    dashboardStore.refresh();
  }));
  unlistenFns.push(await api.onEvalUpdate((run) => evalStore.updateRun(run)));
  unlistenFns.push(await api.onEvalStopped(() => {
    evalStore.refresh();
    dashboardStore.refresh();
  }));

  // Pretrain
  unlistenFns.push(await api.onPretrainStarted((run) => {
    pretrainStore.addRun(run);
  }));
  unlistenFns.push(await api.onPretrainUpdate((run) => pretrainStore.updateRun(run)));
  unlistenFns.push(await api.onPretrainStopped(() => {
    pretrainStore.refresh();
  }));
}

export function cleanupStores() {
  unlistenFns.forEach(fn => fn());
  unlistenFns = [];
}
