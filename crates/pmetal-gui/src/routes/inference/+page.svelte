<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import { page } from '$app/stores';
  import { modelsStore } from '$lib/stores.svelte';
  import { startInference, stopInference, onInferenceToken, onInferenceDone, onInferenceError, listTrainedAdapters, getModelDefaults } from '$lib/api';
  import type { TrainedAdapter, ModelDefaults, InferenceMetrics } from '$lib/api';
  import { renderMarkdown } from '$lib/utils';
  import type { UnlistenFn } from '@tauri-apps/api/event';

  const DEFAULT_MAX_TOKENS = 1024;

  interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    thinking?: string;
    isStreaming?: boolean;
    metrics?: InferenceMetrics;
  }

  // Model config
  let selectedModel = $state('');
  let loraPath = $state('');
  let trainedAdapters = $state<TrainedAdapter[]>([]);
  let loraCustomPath = $state(false);
  let systemMessage = $state('You are a helpful assistant.');

  /** Human-readable label for current config. */
  let configLabel = $derived.by(() => {
    const model = selectedModel ? (selectedModel.split('/').pop() ?? selectedModel) : '';
    if (!loraPath) return model;
    const adapter = trainedAdapters.find(a => a.path === loraPath);
    if (adapter) return adapter.name;
    // Custom path fallback
    const adapterShort = loraPath.split('/').pop() ?? 'LoRA';
    return `${model} + ${adapterShort}`;
  });

  // Generation params (auto-filled from model's generation_config.json)
  let temperature = $state(0.7);
  let topK = $state(50);
  let topP = $state(0.9);
  let maxTokens = $state(DEFAULT_MAX_TOKENS);
  let repetitionPenalty = $state(1.1);
  let minP = $state<number | null>(null);
  let frequencyPenalty = $state<number | null>(null);
  let presencePenalty = $state<number | null>(null);
  let seed = $state<number | null>(null);
  let fp8 = $state(false);
  let noThinking = $state(false);
  let expertsDir = $state('');
  let kvQuant = $state<string>('auto'); // 'auto' | '0' | '4' | '8' | 'tq4' | 'tq8' | 'tq2_5' | 'tq3_5'
  let showParams = $state(false);
  let defaultsSource = $state(''); // which model the current defaults came from

  /** Load model defaults when model selection changes. */
  $effect(() => {
    const model = selectedModel;
    if (!model || model === defaultsSource) return;
    defaultsSource = model;
    getModelDefaults(model).then((d) => {
      if (d.temperature != null) temperature = d.temperature;
      if (d.top_k != null) topK = d.top_k;
      if (d.top_p != null) topP = d.top_p;
      if (d.max_new_tokens != null) maxTokens = normalizeMaxTokensValue(d.max_new_tokens);
      if (d.repetition_penalty != null) repetitionPenalty = d.repetition_penalty;
    }).catch(() => {});
  });

  // Chat state
  let messages = $state<ChatMessage[]>([]);
  let userInput = $state('');
  let isGenerating = $state(false);
  let inferenceStatus = $state<'idle' | 'loading' | 'generating' | 'done'>('idle');
  let error = $state<string | null>(null);
  let showThinking = $state(true);
  let expandedThinking = $state(new Set<number>());
  const truncatedThinkingReply =
    '[Response truncated - model was still thinking. Disable thinking or increase Max Tokens.]';

  function toggleThinking(idx: number) {
    expandedThinking = new Set(expandedThinking);
    if (expandedThinking.has(idx)) {
      expandedThinking.delete(idx);
    } else {
      expandedThinking.add(idx);
    }
  }

  // Scroll ref
  let messagesEl = $state<HTMLElement | null>(null);

  // Listeners
  let unlistenToken: UnlistenFn | null = null;
  let unlistenDone: UnlistenFn | null = null;
  let unlistenError: UnlistenFn | null = null;

  let models = $derived(modelsStore.models);

  onMount(() => {
    // Deep-link: pre-select model from query param
    const modelParam = $page.url.searchParams.get('model');
    if (modelParam) selectedModel = modelParam;

    // Load trained adapters for the dropdown
    listTrainedAdapters().then(a => trainedAdapters = a).catch(() => {});
  });

  onDestroy(() => {
    unlistenToken?.();
    unlistenDone?.();
    unlistenError?.();
  });

  async function scrollToBottom() {
    await tick();
    if (messagesEl) {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  function parseThinking(text: string): { thinking: string | null; reply: string } {
    const match = text.match(/^<think>([\s\S]*?)<\/think>\s*/);
    if (match) {
      return { thinking: match[1].trim(), reply: text.slice(match[0].length) };
    }
    return { thinking: null, reply: text };
  }

  function normalizeMaxTokensValue(value: number | null | undefined): number {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return DEFAULT_MAX_TOKENS;
    return Math.max(1, Math.floor(parsed));
  }

  function normalizeMaxTokensInput() {
    maxTokens = normalizeMaxTokensValue(maxTokens);
  }

  async function handleSend() {
    if (!userInput.trim() || isGenerating) return;
    if (!selectedModel) { error = 'Please select a model first'; return; }

    error = null;
    const prompt = userInput.trim();
    userInput = '';

    // Snapshot current history before adding new user message
    const historySnapshot = [...messages];

    // Add user message
    messages = [...messages, { role: 'user', content: prompt }];

    // Add placeholder assistant message
    messages = [...messages, { role: 'assistant', content: '', isStreaming: true }];
    await scrollToBottom();

    isGenerating = true;
    inferenceStatus = 'loading';
    let currentContent = '';
    let firstTokenReceived = false;

    try {
      // Set up listeners before invoking
      unlistenToken?.();
      unlistenDone?.();
      unlistenError?.();

      unlistenToken = await onInferenceToken((token) => {
        if (!firstTokenReceived) {
          firstTokenReceived = true;
          inferenceStatus = 'generating';
        }
        currentContent += token;
        messages = messages.map((m, i) => {
          if (i === messages.length - 1) {
            return { ...m, content: currentContent };
          }
          return m;
        });
        scrollToBottom();
      });

      unlistenDone = await onInferenceDone((metrics) => {
        const parsed = metrics?.response_text != null
          ? {
              thinking: metrics.thinking ?? null,
              reply: metrics.truncated_thinking ? truncatedThinkingReply : metrics.response_text,
            }
          : parseThinking(currentContent);
        messages = messages.map((m, i) => {
          if (i === messages.length - 1) {
            return {
              ...m,
              content: parsed.reply,
              thinking: parsed.thinking ?? undefined,
              isStreaming: false,
              metrics: metrics ?? undefined,
            };
          }
          return m;
        });
        isGenerating = false;
        inferenceStatus = 'done';
        scrollToBottom();
      });

      unlistenError = await onInferenceError((message) => {
        error = message;
        messages = messages.filter((m, i) => !(i === messages.length - 1 && m.isStreaming));
        isGenerating = false;
        inferenceStatus = 'idle';
      });

      await startInference(
        {
          model: selectedModel,
          lora: loraPath || undefined,
          prompt,
          system: systemMessage || undefined,
          temperature,
          top_k: topK,
          top_p: topP,
          min_p: minP,
          max_tokens: normalizeMaxTokensValue(maxTokens),
          repetition_penalty: repetitionPenalty,
          frequency_penalty: frequencyPenalty,
          presence_penalty: presencePenalty,
          seed,
          fp8: fp8 || undefined,
          no_thinking: noThinking || undefined,
          experts_dir: expertsDir || undefined,
          kv_quant:
            kvQuant === 'tq4'
              ? 4
              : kvQuant === 'tq8'
                ? 8
                : kvQuant === 'auto'
                  ? undefined
                  : parseInt(kvQuant),
          no_kv_quant: kvQuant === '0' ? true : undefined,
          kv_turboquant: kvQuant === 'tq4' || kvQuant === 'tq8' ? true : undefined,
          kv_turboquant_preset:
            kvQuant === 'tq2_5'
              ? 'q2_5'
              : kvQuant === 'tq3_5'
                ? 'q3_5'
                : undefined,
        },
        historySnapshot
          .filter((m) => m.role === 'user' || m.role === 'assistant')
          .map((m) => ({ role: m.role, content: m.content })),
      );
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      // Remove the streaming placeholder on error
      messages = messages.slice(0, -1);
      isGenerating = false;
      inferenceStatus = 'idle';
    }
  }

  async function handleStop() {
    await stopInference().catch(console.error);
    messages = messages.map((m, i) => {
      if (i === messages.length - 1 && m.isStreaming) {
        const { thinking, reply } = parseThinking(m.content);
        return { ...m, content: reply, thinking: thinking ?? undefined, isStreaming: false };
      }
      return m;
    });
    isGenerating = false;
    inferenceStatus = 'idle';
  }

  function clearChat() {
    messages = [];
    error = null;
  }

  /** Copy text to clipboard with brief visual feedback. */
  let copiedIdx = $state<number | null>(null);
  async function copyMessage(idx: number) {
    const msg = messages[idx];
    if (!msg) return;
    const text = msg.thinking ? `<think>\n${msg.thinking}\n</think>\n${msg.content}` : msg.content;
    await navigator.clipboard.writeText(text);
    copiedIdx = idx;
    setTimeout(() => { if (copiedIdx === idx) copiedIdx = null; }, 1500);
  }

  /** Delete assistant response at idx and regenerate from the preceding user message. */
  async function regenerate(idx: number) {
    // Find the user message that preceded this assistant response
    if (idx < 1 || messages[idx]?.role !== 'assistant') return;
    const userMsg = messages[idx - 1];
    if (userMsg?.role !== 'user') return;

    // Remove the assistant response
    messages = messages.slice(0, idx);

    // Re-send the user's prompt
    userInput = userMsg.content;
    // Remove the user message too — handleSend will re-add it
    messages = messages.slice(0, idx - 1);
    await handleSend();
  }

  function handleKeyDown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }
</script>

<!-- Full-height chat layout — override page padding via negative margin trick -->
<div class="flex flex-col" style="height: calc(100vh - 4rem); margin: -1.5rem;">
  <!-- Top config bar -->
  <div class="flex-shrink-0 px-4 py-3 bg-white dark:bg-surface-800 border-b border-surface-200 dark:border-surface-700">
    <div class="flex items-center gap-3 flex-wrap">
      <!-- Model selector -->
      <div class="flex-1 min-w-[200px] max-w-[400px]">
        <label for="inf-model" class="sr-only">Select model</label>
        <select id="inf-model" class="input text-sm" bind:value={selectedModel}>
          <option value="">Select a model...</option>
          {#each models as model}
            <option value={model.id}>{model.id.split('/').pop()} ({model.size_formatted})</option>
          {/each}
        </select>
      </div>

      <!-- LoRA adapter -->
      <div class="flex-1 min-w-[160px] max-w-[320px]">
        <label for="inf-lora" class="sr-only">LoRA adapter</label>
        {#if trainedAdapters.length > 0 && !loraCustomPath}
          <select id="inf-lora" class="input text-sm" bind:value={loraPath} onchange={(e) => {
            const selectedValue = (e.target as HTMLSelectElement).value;
            if (selectedValue === '__custom__') { loraCustomPath = true; loraPath = ''; return; }
            // Auto-select matching base model when a LoRA adapter is chosen
            const adapter = trainedAdapters.find(a => a.path === selectedValue);
            if (adapter?.base_model) {
              const match = models.find(m => m.id === adapter.base_model || m.id.endsWith('/' + adapter.base_model));
              if (match) selectedModel = match.id;
            }
          }}>
            <option value="">No adapter (base model)</option>
            {#each trainedAdapters as adapter}
              <option value={adapter.path}>
                {adapter.name}{adapter.rank ? ` r=${adapter.rank}` : ''}
              </option>
            {/each}
            <option value="__custom__">Custom path...</option>
          </select>
        {:else}
          <div class="flex gap-1">
            <input
              id="inf-lora"
              type="text"
              class="input text-sm flex-1"
              placeholder="LoRA adapter path (optional)"
              bind:value={loraPath}
            />
            {#if trainedAdapters.length > 0}
              <button type="button" class="btn-ghost btn-sm text-xs" onclick={() => { loraCustomPath = false; loraPath = ''; }}>List</button>
            {/if}
          </div>
        {/if}
      </div>

      <div class="w-28">
        <label class="label text-xs" for="inf-maxtok">Max Tokens</label>
        <input
          id="inf-maxtok"
          type="number"
          class="input text-sm"
          min="1"
          step="1"
          bind:value={maxTokens}
          onblur={normalizeMaxTokensInput}
        />
      </div>

      <!-- Parameters toggle -->
      <button
        class="btn-secondary btn-sm"
        aria-label="Toggle generation parameters"
        aria-expanded={showParams}
        onclick={() => (showParams = !showParams)}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
        </svg>
        Params
      </button>

      <label class="flex items-center gap-2 text-sm cursor-pointer">
        <input type="checkbox" class="rounded" bind:checked={showThinking} />
        <span class="text-surface-600 dark:text-surface-400">Show thinking</span>
      </label>

      <button
        class="btn-ghost btn-sm text-surface-500"
        aria-label="Clear conversation"
        onclick={clearChat}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
        </svg>
      </button>
    </div>

    <!-- Generation params collapsible -->
    {#if showParams}
      <div class="mt-3 pt-3 border-t border-surface-200 dark:border-surface-700 space-y-3">
        <!-- Row 1: core sampling -->
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label class="label" for="inf-temp">Temperature</label>
            <input id="inf-temp" type="number" class="input text-sm" step="0.05" min="0" max="2" bind:value={temperature} />
          </div>
          <div>
            <label class="label" for="inf-topk">Top-K</label>
            <input id="inf-topk" type="number" class="input text-sm" min="0" bind:value={topK} />
          </div>
          <div>
            <label class="label" for="inf-topp">Top-P</label>
            <input id="inf-topp" type="number" class="input text-sm" step="0.05" min="0" max="1" bind:value={topP} />
          </div>
          <div>
            <label class="label" for="inf-minp">Min-P</label>
            <input id="inf-minp" type="number" class="input text-sm" step="0.01" min="0" max="1" bind:value={minP} />
          </div>
        </div>
        <!-- Row 2: penalties, seed, KV cache -->
        <div class="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div>
            <label class="label" for="inf-rep">Rep. Penalty</label>
            <input id="inf-rep" type="number" class="input text-sm" step="0.05" min="1" bind:value={repetitionPenalty} />
          </div>
          <div>
            <label class="label" for="inf-freq">Freq. Penalty</label>
            <input id="inf-freq" type="number" class="input text-sm" step="0.05" min="0" max="2" bind:value={frequencyPenalty} />
          </div>
          <div>
            <label class="label" for="inf-pres">Pres. Penalty</label>
            <input id="inf-pres" type="number" class="input text-sm" step="0.05" min="0" max="2" bind:value={presencePenalty} />
          </div>
          <div>
            <label class="label" for="inf-seed">Seed</label>
            <input id="inf-seed" type="number" class="input text-sm" min="0" placeholder="Random" bind:value={seed} />
          </div>
          <div>
            <label class="label" for="inf-kvq">KV Cache</label>
            <select id="inf-kvq" class="input text-sm" bind:value={kvQuant}>
              <option value="auto">Auto</option>
              <option value="0">FP16</option>
              <option value="8">Q8</option>
              <option value="4">Q4</option>
              <option value="tq4">TurboQ4</option>
              <option value="tq8">TurboQ8</option>
              <option value="tq2_5">TurboQ2.5</option>
              <option value="tq3_5">TurboQ3.5</option>
            </select>
          </div>
        </div>
        <!-- Row 3: toggles + experts dir -->
        <div class="grid grid-cols-2 md:grid-cols-5 gap-4 items-end">
          <label class="flex items-center gap-2 text-sm cursor-pointer">
            <input type="checkbox" class="rounded" bind:checked={fp8} />
            <span class="text-surface-600 dark:text-surface-400">FP8 weights</span>
          </label>
          <label class="flex items-center gap-2 text-sm cursor-pointer">
            <input type="checkbox" class="rounded" bind:checked={noThinking} />
            <span class="text-surface-600 dark:text-surface-400">Disable thinking</span>
          </label>
          <div class="col-span-2 md:col-span-3">
            <label class="label" for="inf-experts">Experts Dir</label>
            <input id="inf-experts" type="text" class="input text-sm" placeholder="Path to packed experts (optional)" bind:value={expertsDir} />
          </div>
        </div>
      </div>
    {/if}
  </div>

  <!-- System message -->
  <div class="flex-shrink-0 px-4 py-2 bg-surface-50 dark:bg-surface-900/50 border-b border-surface-200 dark:border-surface-700">
    <div class="flex items-center gap-2">
      <label for="inf-system" class="text-xs font-medium text-surface-500 uppercase tracking-wider whitespace-nowrap">System:</label>
      <input
        id="inf-system"
        type="text"
        class="flex-1 text-sm bg-transparent border-none outline-none text-surface-700 dark:text-surface-300 placeholder-surface-400"
        placeholder="System message..."
        bind:value={systemMessage}
      />
    </div>
  </div>

  <!-- Messages area -->
  <div
    class="flex-1 overflow-y-auto scrollbar-thin p-4 space-y-4"
    bind:this={messagesEl}
    aria-label="Conversation"
    aria-live="polite"
    style="min-height: 0;"
  >
    {#if messages.length === 0}
      <div class="flex items-center justify-center h-full">
        <div class="text-center">
          <div class="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent-400 to-orange-600 flex items-center justify-center mx-auto mb-4" aria-hidden="true">
            <span class="text-white font-bold text-2xl">P</span>
          </div>
          <h3 class="text-lg font-semibold text-surface-900 dark:text-surface-100 mb-1">PMetal Inference</h3>
          <p class="text-surface-500 dark:text-surface-400 text-sm max-w-xs">
            Select a model above and start chatting. Supports streaming, LoRA adapters, and thinking tokens.
          </p>
        </div>
      </div>
    {:else if selectedModel}
      <!-- Config badge: shows what model+adapter combo is active -->
      <div class="flex justify-center mb-2">
        <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-surface-100 dark:bg-surface-800 border border-surface-200 dark:border-surface-700 text-xs text-surface-500 dark:text-surface-400">
          <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
          {configLabel}
        </span>
      </div>
    {/if}

    {#each messages as message, idx}
      {#if message.role === 'user'}
        <!-- User message -->
        <div class="group flex justify-end gap-1 items-end" role="article" aria-label="User message">
          <button
            class="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded text-surface-400 hover:text-surface-600 dark:hover:text-surface-300"
            aria-label="Copy message"
            onclick={() => copyMessage(idx)}
          >
            {#if copiedIdx === idx}
              <svg class="w-3.5 h-3.5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>
            {:else}
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
            {/if}
          </button>
          <div class="max-w-[70%] bg-primary-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </div>
        </div>
      {:else}
        <!-- Assistant message -->
        <div class="group flex justify-start gap-1 items-end" role="article" aria-label="Assistant message">
          <div class="max-w-[80%] space-y-2">
            <!-- Thinking block (collapsible, global showThinking filter) -->
            {#if message.thinking && showThinking}
              <div class="border border-surface-200 dark:border-surface-700 rounded-lg overflow-hidden">
                <button
                  class="w-full flex items-center justify-between px-3 py-2 text-xs text-surface-500 dark:text-surface-400 hover:bg-surface-100 dark:hover:bg-surface-800 transition-colors"
                  aria-expanded={expandedThinking.has(idx)}
                  onclick={() => toggleThinking(idx)}
                >
                  <span class="flex items-center gap-1.5">
                    <svg class="w-3 h-3 text-surface-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    <span class="font-medium uppercase tracking-wider">Thinking</span>
                    <span class="text-surface-400 dark:text-surface-500">({message.thinking.length} chars)</span>
                  </span>
                  <span class="text-surface-400 dark:text-surface-500">
                    {expandedThinking.has(idx) ? '▼' : '▶'}
                  </span>
                </button>
                {#if expandedThinking.has(idx)}
                  <div class="px-3 py-2 text-xs text-surface-600 dark:text-surface-400 font-mono whitespace-pre-wrap leading-relaxed border-t border-surface-200 dark:border-surface-700 max-h-96 overflow-y-auto bg-surface-50 dark:bg-surface-900/50">
                    {message.thinking}
                  </div>
                {/if}
              </div>
            {/if}

            <!-- Response with markdown rendering -->
            <div class="rounded-2xl rounded-tl-sm bg-white dark:bg-surface-800 border border-surface-200 dark:border-surface-700 px-4 py-3 text-sm leading-relaxed text-surface-900 dark:text-surface-100 markdown-content">
              {#if message.content}
                {@html renderMarkdown(message.content)}
              {/if}
              {#if message.isStreaming}
                <span class="inline-block w-2 h-4 bg-primary-500 animate-pulse ml-0.5 align-text-bottom" aria-label="Generating..." aria-live="polite"></span>
              {/if}
            </div>
            <!-- Inference metrics -->
            {#if message.metrics && !message.isStreaming}
              <div class="flex items-center gap-3 px-2 text-[10px] text-surface-400 dark:text-surface-500 font-mono">
                {#if message.metrics.tok_per_sec != null}
                  <span>{message.metrics.tok_per_sec.toFixed(1)} tok/s</span>
                {/if}
                {#if message.metrics.ttft_ms != null}
                  <span>TTFT {message.metrics.ttft_ms.toFixed(0)}ms</span>
                {/if}
                <span>{message.metrics.generated_tokens} tokens</span>
                <span>{(message.metrics.total_ms / 1000).toFixed(1)}s</span>
              </div>
            {/if}
          </div>
          <!-- Action buttons: copy + regenerate (hover reveal) -->
          {#if !message.isStreaming}
            <div class="flex flex-col gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                class="p-1 rounded text-surface-400 hover:text-surface-600 dark:hover:text-surface-300"
                aria-label="Copy response"
                onclick={() => copyMessage(idx)}
              >
                {#if copiedIdx === idx}
                  <svg class="w-3.5 h-3.5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" /></svg>
                {:else}
                  <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                {/if}
              </button>
              <button
                class="p-1 rounded text-surface-400 hover:text-surface-600 dark:hover:text-surface-300"
                aria-label="Regenerate response"
                disabled={isGenerating}
                onclick={() => regenerate(idx)}
              >
                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
              </button>
            </div>
          {/if}
        </div>
      {/if}
    {/each}
  </div>

  <!-- Error -->
  {#if error}
    <div class="flex-shrink-0 mx-4 mb-2 p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 text-sm" role="alert">
      {error}
    </div>
  {/if}

  <!-- Input area -->
  <div class="flex-shrink-0 p-4 bg-white dark:bg-surface-800 border-t border-surface-200 dark:border-surface-700">
    <!-- Status indicator bar -->
    {#if inferenceStatus === 'loading'}
      <div class="flex items-center gap-2 mb-2 text-xs text-surface-400 dark:text-surface-500">
        <span class="inline-block w-3 h-3 rounded-full border-2 border-primary-400 border-t-transparent animate-spin" aria-hidden="true"></span>
        <span>Loading model...</span>
      </div>
    {:else if inferenceStatus === 'generating'}
      <div class="flex items-center gap-2 mb-2 text-xs text-primary-500 dark:text-primary-400">
        <span class="inline-block w-2 h-2 rounded-full bg-primary-500 animate-pulse" aria-hidden="true"></span>
        <span>Generating...</span>
      </div>
    {/if}
    <div class="flex gap-2 items-end">
      <label for="inf-input" class="sr-only">Type a message</label>
      <textarea
        id="inf-input"
        class="input flex-1 resize-none min-h-[44px] max-h-[120px]"
        placeholder="Type a message... (Enter to send, Shift+Enter for newline)"
        rows="1"
        bind:value={userInput}
        onkeydown={handleKeyDown}
        disabled={isGenerating}
      ></textarea>

      {#if isGenerating}
        <button class="btn-danger" aria-label="Stop generation" onclick={handleStop}>
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
          </svg>
        </button>
      {:else}
        <button
          class="btn-primary"
          aria-label="Send message"
          onclick={handleSend}
          disabled={!userInput.trim() || !selectedModel}
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      {/if}
    </div>
  </div>
</div>
