/**
 * The sidebar. Jobs are listed by the key of the spec they run, so a key a
 * spec no longer has fails type-checking, and a test fails for a spec with no
 * place here.
 */

import type { JobKey } from './specs.generated';

export interface NavItem {
  path: string;
  label: string;
  sublabel?: string;
  /** SVG path data (24x24, stroked). */
  icon: string;
}

export interface NavSection {
  title: string;
  items: NavItem[];
}

const icons = {
  home: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6',
  bolt: 'M13 10V3L4 14h7v7l9-11h-7z',
  chart: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z',
  flask: 'M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z',
  bulb: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z',
  chat: 'M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z',
  box: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10',
  db: 'M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4',
  merge: 'M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z',
  chip: 'M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z',
  thumbs: 'M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5',
  cloud: 'M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z',
  server: 'M5 12h14M12 5v14',
  music: 'M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3',
  shield: 'M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z',
  cog: 'M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z',
  list: 'M4 6h16M4 10h16M4 14h16M4 18h16',
};

/** How a job appears in the sidebar. */
const job = (key: JobKey, label: string, sublabel: string | undefined, icon: string): NavItem => ({
  path: `/run/${key}`,
  label,
  sublabel,
  icon,
});

export const SECTIONS: NavSection[] = [
  {
    title: 'Overview',
    items: [
      { path: '/', label: 'Dashboard', icon: icons.home },
      { path: '/jobs', label: 'Jobs', sublabel: 'Everything running and run', icon: icons.list },
    ],
  },
  {
    title: 'Train',
    items: [
      job('train', 'Fine-tune', 'LoRA, QLoRA, full', icons.bolt),
      job('grpo', 'GRPO', 'Reinforcement learning', icons.chart),
      job('rlkd', 'RLKD', 'RL + distillation', icons.chart),
      job('preference', 'Preference', 'DPO, SimPO, ORPO, KTO', icons.thumbs),
      job('distill', 'Distillation', 'Teacher to student', icons.flask),
      job('embed_train', 'Embeddings', 'Contrastive training', icons.box),
      job('pretrain', 'Pretrain', 'From scratch', icons.bulb),
    ],
  },
  {
    title: 'Run',
    items: [
      { path: '/inference', label: 'Chat', sublabel: 'Talk to a model', icon: icons.chat },
      job('serve', 'Serve', 'OpenAI-compatible API', icons.server),
      job('dflash', 'DFlash', 'Speculative decoding', icons.bolt),
    ],
  },
  {
    title: 'Library',
    items: [
      { path: '/models', label: 'Models', sublabel: 'On this Mac and the Hub', icon: icons.box },
      { path: '/datasets', label: 'Datasets', sublabel: 'On this Mac and the Hub', icon: icons.db },
    ],
  },
  {
    title: 'Tools',
    items: [
      job('merge', 'Merge', 'Combine models', icons.merge),
      job('fuse', 'Fuse adapter', 'Into base weights', icons.merge),
      job('quantize', 'Quantize', 'GGUF and MLX', icons.chip),
      job('imatrix', 'Importance matrix', 'Guide quantization', icons.chip),
      job('ollama_modelfile', 'Modelfile', 'For Ollama', icons.cloud),
      job('pack_experts', 'Pack experts', 'Mixture of experts', icons.chip),
    ],
  },
  {
    title: 'Measure',
    items: [
      job('eval', 'Evaluate', 'Perplexity', icons.shield),
      job('bench', 'Benchmark', 'Training throughput', icons.music),
      job('bench_workload', 'Workload', 'Inference and training', icons.music),
    ],
  },
  {
    title: 'Advanced',
    items: [
      job('train_mtp', 'MTP predictor', undefined, icons.bolt),
      job('train_draft', 'DFlash draft', undefined, icons.bolt),
      job('train_diffusion', 'Diffusion LoRA', undefined, icons.flask),
      job('tokenize', 'Tokenize', 'Text to shards', icons.db),
      job('vocode', 'Vocode', 'Mel to audio', icons.music),
      job('decide', 'Decide', 'Typed decisions', icons.shield),
    ],
  },
  {
    title: 'App',
    items: [{ path: '/settings', label: 'Settings', icon: icons.cog }],
  },
];

/** The job keys the sidebar links to. */
export function navJobKeys(): string[] {
  return SECTIONS.flatMap((s) => s.items)
    .map((i) => /^\/run\/(.+)$/.exec(i.path)?.[1])
    .filter((k): k is string => !!k);
}
