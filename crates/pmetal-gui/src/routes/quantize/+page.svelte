<script lang="ts">
  import { modelsStore } from '$lib/stores.svelte';
  import { quantizeModel, fuseLora } from '$lib/api';

  const quantMethods = [
    { value: 'dynamic_2.0', label: 'Dynamic 2.0', description: 'PMetal dynamic quantization — best quality/speed tradeoff' },
    { value: 'Q8_0', label: 'Q8_0', description: '8-bit quantization — near-lossless, 2x compression' },
    { value: 'Q6_K', label: 'Q6_K', description: '6-bit K-quant — excellent quality' },
    { value: 'Q5_K_M', label: 'Q5_K_M', description: '5-bit medium K-quant — good balance' },
    { value: 'Q4_K_M', label: 'Q4_K_M', description: '4-bit medium K-quant — recommended for most models' },
    { value: 'Q3_K_M', label: 'Q3_K_M', description: '3-bit medium K-quant — aggressive compression' },
    { value: 'Q2_K', label: 'Q2_K', description: '2-bit K-quant — maximum compression, lower quality' },
  ];

  // Form state
  let selectedModel = $state('');
  let selectedMethod = $state('dynamic_2.0');
  let loraAdapterPath = $state('');
  let importanceMatrixPath = $state('');
  let outputPath = $state('');

  // Fuse-first option
  let fuseBeforeQuantize = $state(false);

  // UI state
  let isSubmitting = $state(false);
  let formError = $state<string | null>(null);
  let formSuccess = $state<string | null>(null);
  let statusMessage = $state<string | null>(null);

  let models = $derived(modelsStore.models);

  function estimateOutputSize(modelId: string, method: string): string {
    const m = models.find(m => m.id === modelId);
    if (!m) return '--';
    const bpp: Record<string, number> = {
      'dynamic_2.0': 2.1,
      'Q8_0': 1.05,
      'Q6_K': 0.80,
      'Q5_K_M': 0.68,
      'Q4_K_M': 0.58,
      'Q3_K_M': 0.48,
      'Q2_K': 0.37,
    };
    const factor = bpp[method] ?? 1.0;
    const originalGb = m.size / (1024 * 1024 * 1024);
    const estimatedGb = originalGb * factor;
    return `~${estimatedGb.toFixed(1)} GB`;
  }

  async function handleSubmit(e: Event) {
    e.preventDefault();
    formError = null;
    formSuccess = null;
    statusMessage = null;

    if (!selectedModel) { formError = 'Please select a model'; return; }
    if (!outputPath.trim()) { formError = 'Please specify an output path'; return; }

    isSubmitting = true;

    let targetModel = selectedModel;

    try {
      // Step 1: Optionally fuse LoRA before quantizing
      if (fuseBeforeQuantize && loraAdapterPath.trim()) {
        statusMessage = 'Fusing LoRA adapter...';
        const fuseOutput = outputPath.trim() + '-fused-tmp';
        const fuseResult = await fuseLora(selectedModel, loraAdapterPath.trim(), fuseOutput);
        targetModel = fuseResult.output_dir;
        statusMessage = 'LoRA fused. Starting quantization...';
      }

      // Step 2: Quantize
      statusMessage = `Quantizing with ${selectedMethod}...`;
      const result = await quantizeModel(targetModel, selectedMethod, outputPath.trim());
      formSuccess = `Quantization complete. Output: ${result}`;
      statusMessage = null;
    } catch (e) {
      formError = e instanceof Error ? e.message : String(e);
      statusMessage = null;
    } finally {
      isSubmitting = false;
    }
  }
</script>

<div class="space-y-6 max-w-3xl">
  <!-- Header -->
  <div>
    <h1 class="text-2xl font-bold text-surface-900 dark:text-surface-100">Quantize Model</h1>
    <p class="text-surface-500 dark:text-surface-400 mt-1">Reduce model size using GGUF-compatible quantization methods</p>
  </div>

  <form onsubmit={handleSubmit} class="space-y-4">
    <!-- Model Selection -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Source Model</h3>
      </div>
      <div class="card-body space-y-4">
        <div>
          <label class="label" for="quant-model">Cached Model</label>
          <select id="quant-model" class="input" bind:value={selectedModel}>
            <option value="">Select a model to quantize...</option>
            {#each models as model}
              <option value={model.id}>{model.id} ({model.size_formatted})</option>
            {/each}
          </select>
        </div>

        <!-- Optional LoRA fuse -->
        <div class="p-4 rounded-lg bg-surface-50 dark:bg-surface-700/50 border border-surface-200 dark:border-surface-700 space-y-3">
          <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" class="rounded border-surface-300" bind:checked={fuseBeforeQuantize} />
            <span class="text-sm font-medium text-surface-700 dark:text-surface-300">Fuse LoRA adapter before quantizing</span>
          </label>
          {#if fuseBeforeQuantize}
            <div>
              <label class="label" for="quant-lora">LoRA Adapter Path</label>
              <input
                id="quant-lora"
                type="text"
                class="input"
                placeholder="/path/to/lora/adapter"
                bind:value={loraAdapterPath}
              />
            </div>
          {/if}
        </div>

        <div>
          <label class="label" for="quant-imatrix">Importance Matrix (optional)</label>
          <input
            id="quant-imatrix"
            type="text"
            class="input"
            placeholder="/path/to/imatrix.dat (improves lower-bit quality)"
            bind:value={importanceMatrixPath}
          />
          <p class="text-xs text-surface-500 mt-1">Generated from calibration data. Improves quality for Q4 and below.</p>
        </div>
      </div>
    </div>

    <!-- Quantization Method -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Quantization Method</h3>
      </div>
      <div class="card-body space-y-3">
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {#each quantMethods as method}
            <button
              type="button"
              class="p-3 rounded-lg border text-left transition-all {selectedMethod === method.value
                ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30'
                : 'border-surface-200 dark:border-surface-700 hover:border-surface-300 dark:hover:border-surface-600'}"
              onclick={() => (selectedMethod = method.value)}
            >
              <div class="flex items-center justify-between">
                <p class="text-sm font-semibold text-surface-900 dark:text-surface-100">{method.label}</p>
                {#if selectedModel}
                  <span class="text-xs text-surface-500">{estimateOutputSize(selectedModel, method.value)}</span>
                {/if}
              </div>
              <p class="text-xs text-surface-500 mt-0.5">{method.description}</p>
            </button>
          {/each}
        </div>

        {#if selectedModel}
          {@const originalModel = models.find(m => m.id === selectedModel)}
          {#if originalModel}
            <div class="pt-2 flex items-center gap-4 text-sm text-surface-600 dark:text-surface-400">
              <span>Original: <strong>{originalModel.size_formatted}</strong></span>
              <span>→</span>
              <span>Estimated: <strong class="text-primary-600 dark:text-primary-400">{estimateOutputSize(selectedModel, selectedMethod)}</strong></span>
            </div>
          {/if}
        {/if}
      </div>
    </div>

    <!-- Output -->
    <div class="card">
      <div class="card-header">
        <h3 class="font-semibold text-surface-900 dark:text-surface-100">Output</h3>
      </div>
      <div class="card-body">
        <label class="label" for="quant-output">Output Path</label>
        <input
          id="quant-output"
          type="text"
          class="input"
          placeholder="/path/to/quantized-model"
          bind:value={outputPath}
        />
        <p class="text-xs text-surface-500 mt-1">Directory where the quantized model will be saved</p>
      </div>
    </div>

    <!-- Status messages -->
    {#if statusMessage}
      <div class="p-4 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800 text-primary-700 dark:text-primary-300 text-sm flex items-center gap-2" role="status">
        <div class="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin flex-shrink-0" aria-hidden="true"></div>
        {statusMessage}
      </div>
    {/if}
    {#if formError}
      <div class="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
        {formError}
      </div>
    {/if}
    {#if formSuccess}
      <div class="p-4 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 text-green-700 dark:text-green-300 text-sm" role="status">
        {formSuccess}
      </div>
    {/if}

    <button type="submit" class="btn-primary w-full" disabled={isSubmitting || !selectedModel || !outputPath.trim()}>
      {#if isSubmitting}
        <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" aria-hidden="true"></div>
        {fuseBeforeQuantize && loraAdapterPath ? 'Fusing & Quantizing...' : 'Quantizing...'}
      {:else}
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
        </svg>
        Quantize Model
      {/if}
    </button>
  </form>

  <!-- Guide -->
  <div class="card">
    <div class="card-header">
      <h3 class="font-semibold text-surface-900 dark:text-surface-100">Quantization Guide</h3>
    </div>
    <div class="card-body space-y-3 text-sm text-surface-600 dark:text-surface-400">
      <p><strong class="text-surface-800 dark:text-surface-200">Dynamic 2.0</strong> — PMetal's default. Analyzes weight distributions per-layer and selects optimal bit-width automatically. Best for general use.</p>
      <p><strong class="text-surface-800 dark:text-surface-200">Q4_K_M</strong> — The recommended GGUF format for running on Apple Silicon. Good balance of quality and size. Works with all compatible tools.</p>
      <p><strong class="text-surface-800 dark:text-surface-200">Importance matrix</strong> — Run a calibration pass on representative data to build an imatrix file. Significantly improves Q4 and Q3 quality, especially for instruction-tuned models.</p>
      <p class="pt-2 border-t border-surface-200 dark:border-surface-700 text-xs">All quantization runs on the Apple Neural Engine (ANE) or GPU via Metal. Expect 5–20 minutes depending on model size.</p>
    </div>
  </div>
</div>
