/**
 * Svelte 5 stores (runes) for what is not a job: models, datasets, downloads,
 * settings and the machine. Jobs are in `jobs.svelte.ts`.
 */

import * as api from './api';
import type {
  AppConfig,
  CachedDatasetInfo,
  CachedModel,
  DatasetSearchResult,
  DownloadView,
  FormInfo,
  SystemInfo,
} from './api';
import { jobsStore } from './jobs.svelte';

// =============================================================================
// Models
// =============================================================================

class ModelsStore {
  models = $state<CachedModel[]>([]);
  loading = $state(false);
  error = $state<string | null>(null);

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      this.models = await api.refreshModels();
    } catch (e) {
      this.error = api.errorMessage(e);
    } finally {
      this.loading = false;
    }
  }

  /** Take the list the backend has, without scanning again. */
  async reload() {
    try {
      this.models = await api.listModels();
    } catch (e) {
      this.error = api.errorMessage(e);
    }
  }

  async delete(modelId: string) {
    await api.deleteModel(modelId);
    this.models = this.models.filter((m) => m.id !== modelId);
  }

  async addDirectory(path: string) {
    this.models = await api.addModelDirectory(path);
    await configStore.load();
  }

  async removeDirectory(path: string) {
    this.models = await api.removeModelDirectory(path);
    await configStore.load();
  }

  /** The model a job form's reference names, if it is one of these. */
  byReference(reference: string): CachedModel | undefined {
    return this.models.find((m) => m.reference === reference || m.id === reference);
  }
}

export const modelsStore = new ModelsStore();

// =============================================================================
// Datasets
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
      this.error = api.errorMessage(e);
    } finally {
      this.loading = false;
    }
  }

  async search(query: string) {
    if (!query.trim()) {
      this.clearSearch();
      return;
    }
    this.searching = true;
    this.searchQuery = query;
    this.error = null;
    try {
      this.searchResults = await api.searchHubDatasets(query, 20);
    } catch (e) {
      this.error = api.errorMessage(e);
    } finally {
      this.searching = false;
    }
  }

  async loadTrending() {
    try {
      this.trending = await api.getTrendingDatasets(10);
    } catch (e) {
      // Trending is a convenience; being offline is not an error worth a banner.
      console.warn('trending datasets unavailable:', api.errorMessage(e));
    }
  }

  clearSearch() {
    this.searchResults = [];
    this.searchQuery = '';
  }
}

export const datasetsStore = new DatasetsStore();

// =============================================================================
// Downloads
// =============================================================================

class DownloadsStore {
  items = $state<DownloadView[]>([]);
  error = $state<string | null>(null);

  get active() {
    return this.items.filter((d) => d.state === 'queued' || d.state === 'downloading');
  }

  async refresh() {
    try {
      this.items = await api.listDownloads();
    } catch (e) {
      this.error = api.errorMessage(e);
    }
  }

  update(view: DownloadView) {
    const at = this.items.findIndex((d) => d.id === view.id);
    if (at >= 0) this.items[at] = view;
    else this.items = [view, ...this.items];
    if (view.state === 'completed') {
      void modelsStore.refresh();
      void datasetsStore.refreshCached();
    }
  }

  async startModel(modelId: string, revision?: string) {
    this.error = null;
    this.update(await api.downloadModel(modelId, revision));
  }

  async startDataset(datasetId: string) {
    this.error = null;
    this.update(await api.downloadDataset(datasetId));
  }

  async cancel(id: string) {
    await api.cancelDownload(id);
  }

  async clearFinished() {
    await api.clearFinishedDownloads();
    this.items = this.active;
  }
}

export const downloadsStore = new DownloadsStore();

// =============================================================================
// Job forms
// =============================================================================

class FormsStore {
  forms = $state<FormInfo[]>([]);
  loaded = $state(false);
  error = $state<string | null>(null);

  async load() {
    try {
      this.forms = await api.listJobForms();
      this.error = null;
    } catch (e) {
      this.error = api.errorMessage(e);
    } finally {
      this.loaded = true;
    }
  }

  get(kind: string): FormInfo | undefined {
    return this.forms.find((f) => f.key === kind);
  }
}

export const formsStore = new FormsStore();

// =============================================================================
// Settings
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
      this.error = api.errorMessage(e);
    } finally {
      this.loading = false;
    }
  }

  async save(patch: { default_model?: string; theme?: api.Theme }) {
    this.error = null;
    try {
      this.config = await api.setConfig(patch);
    } catch (e) {
      this.error = api.errorMessage(e);
      throw e;
    }
  }

  async setToken(token: string) {
    this.config = await api.setHfToken(token);
  }

  async clearToken() {
    this.config = await api.clearHfToken();
  }

  get theme() {
    return this.config?.theme ?? 'system';
  }
}

export const configStore = new ConfigStore();

// =============================================================================
// The machine
// =============================================================================

class DeviceStore {
  info = $state<SystemInfo | null>(null);
  error = $state<string | null>(null);

  async refresh() {
    try {
      this.info = await api.getSystemInfo();
      this.error = null;
    } catch (e) {
      this.error = api.errorMessage(e);
    }
  }
}

export const deviceStore = new DeviceStore();

// =============================================================================
// Start-up
// =============================================================================

let unlisteners: (() => void)[] = [];

/** Load everything the pages draw and start following the backend. */
export async function initializeStores() {
  // Listen first, so nothing that happens while loading is missed.
  unlisteners.push(
    await api.onJobUpdate((job) => jobsStore.upsert(job)),
    await api.onJobOutput((e) => jobsStore.appendOutput(e.job_id, e.lines)),
    await api.onJobsResync(() => void jobsStore.refresh()),
    await api.onDownloadUpdate((d) => downloadsStore.update(d)),
    await api.onModelsUpdated(() => void modelsStore.reload()),
  );
  await Promise.allSettled([
    configStore.load(),
    formsStore.load(),
    jobsStore.refresh(),
    modelsStore.reload(),
    downloadsStore.refresh(),
    datasetsStore.refreshCached(),
    deviceStore.refresh(),
  ]);
  void datasetsStore.loadTrending();
}

export function cleanupStores() {
  unlisteners.forEach((fn) => fn());
  unlisteners = [];
}
