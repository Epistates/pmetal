<script lang="ts">
  import { onMount, onDestroy, tick } from 'svelte';
  import { page } from '$app/stores';
  import { modelsStore } from '$lib/stores.svelte';
  import { startInference, stopInference, onInferenceToken, onInferenceDone, onInferenceError } from '$lib/api';
  import { renderMarkdown } from '$lib/utils';
  import type { UnlistenFn } from '@tauri-apps/api/event';

  interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    thinking?: string;
    isStreaming?: boolean;
  }

  // Model config
  let selectedModel = $state('');
  let loraPath = $state('');
  let systemMessage = $state('You are a helpful assistant.');

  // Generation params
  let temperature = $state(0.7);
  let topK = $state(50);
  let topP = $state(0.9);
  let maxTokens = $state(1024);
  let repetitionPenalty = $state(1.1);
  let showParams = $state(false);

  // Chat state
  let messages = $state<ChatMessage[]>([]);
  let userInput = $state('');
  let isGenerating = $state(false);
  let error = $state<string | null>(null);
  let showThinking = $state(true);

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

  /**
   * Build a formatted conversation prompt from the full message history.
   * We send the accumulated conversation so the model has context.
   */
  function buildConversationPrompt(history: ChatMessage[], newUserMessage: string): string {
    const lines: string[] = [];
    for (const msg of history) {
      if (msg.role === 'user') {
        lines.push(`User: ${msg.content}`);
      } else if (msg.role === 'assistant' && msg.content) {
        lines.push(`Assistant: ${msg.content}`);
      }
    }
    lines.push(`User: ${newUserMessage}`);
    lines.push('Assistant:');
    return lines.join('\n');
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
    let currentContent = '';

    try {
      // Set up listeners before invoking
      unlistenToken?.();
      unlistenDone?.();
      unlistenError?.();

      unlistenToken = await onInferenceToken((token) => {
        currentContent += token;
        messages = messages.map((m, i) => {
          if (i === messages.length - 1) {
            return { ...m, content: currentContent };
          }
          return m;
        });
        scrollToBottom();
      });

      unlistenDone = await onInferenceDone(() => {
        const { thinking, reply } = parseThinking(currentContent);
        messages = messages.map((m, i) => {
          if (i === messages.length - 1) {
            return { ...m, content: reply, thinking: thinking ?? undefined, isStreaming: false };
          }
          return m;
        });
        isGenerating = false;
        scrollToBottom();
      });

      unlistenError = await onInferenceError((message) => {
        error = message;
        messages = messages.filter((m, i) => !(i === messages.length - 1 && m.isStreaming));
        isGenerating = false;
      });

      // Build full conversation prompt for context
      const conversationPrompt = buildConversationPrompt(historySnapshot, prompt);

      await startInference({
        model: selectedModel,
        lora_path: loraPath || null,
        prompt: conversationPrompt,
        system_message: systemMessage || null,
        temperature,
        top_k: topK,
        top_p: topP,
        max_tokens: maxTokens,
        repetition_penalty: repetitionPenalty,
      });
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      // Remove the streaming placeholder on error
      messages = messages.slice(0, -1);
      isGenerating = false;
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
  }

  function clearChat() {
    messages = [];
    error = null;
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

      <!-- LoRA path -->
      <div class="flex-1 min-w-[160px] max-w-[280px]">
        <label for="inf-lora" class="sr-only">LoRA adapter path</label>
        <input
          id="inf-lora"
          type="text"
          class="input text-sm"
          placeholder="LoRA adapter path (optional)"
          bind:value={loraPath}
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
      <div class="mt-3 pt-3 border-t border-surface-200 dark:border-surface-700 grid grid-cols-2 md:grid-cols-5 gap-4">
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
          <label class="label" for="inf-maxtok">Max Tokens</label>
          <input id="inf-maxtok" type="number" class="input text-sm" min="1" bind:value={maxTokens} />
        </div>
        <div>
          <label class="label" for="inf-rep">Rep. Penalty</label>
          <input id="inf-rep" type="number" class="input text-sm" step="0.05" min="1" bind:value={repetitionPenalty} />
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
    {/if}

    {#each messages as message, idx}
      {#if message.role === 'user'}
        <!-- User message -->
        <div class="flex justify-end" role="article" aria-label="User message">
          <div class="max-w-[70%] bg-primary-600 text-white rounded-2xl rounded-tr-sm px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </div>
        </div>
      {:else}
        <!-- Assistant message -->
        <div class="flex justify-start" role="article" aria-label="Assistant message">
          <div class="max-w-[80%] space-y-2">
            <!-- Thinking block -->
            {#if message.thinking && showThinking}
              <div class="rounded-xl bg-surface-100 dark:bg-surface-800 border border-surface-200 dark:border-surface-700 p-3">
                <div class="flex items-center gap-2 mb-2">
                  <svg class="w-3 h-3 text-surface-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                  <span class="text-xs font-medium text-surface-500 uppercase tracking-wider">Thinking</span>
                </div>
                <p class="text-xs text-surface-600 dark:text-surface-400 font-mono whitespace-pre-wrap leading-relaxed">{message.thinking}</p>
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
          </div>
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
