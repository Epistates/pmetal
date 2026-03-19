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
  loading = $state(false);
  error = $state<string | null>(null);

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.models = await api.listModels();
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

  async start(config: api.TrainingConfig) {
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

  async start(config: api.GrpoConfig) {
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

  async start(config: api.DistillationConfig) {
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
}

export function cleanupStores() {
  unlistenFns.forEach(fn => fn());
  unlistenFns = [];
}
