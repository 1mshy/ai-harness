export type Difficulty = "easy" | "medium" | "hard";

export interface PromptItem {
  id: string;
  title: string;
  category: string;
  text: string;
  difficulty: Difficulty;
  targetTokens: number;
}

export interface Mix {
  easy: number;
  medium: number;
  hard: number;
}

export type RunMode = "count" | "duration" | "infinite";
export type Selection = "sequential" | "random";

export interface RunConfig {
  baseUrl: string;
  model: string;
  apiKey: string;

  concurrency: number;
  mode: RunMode;
  totalRequests: number;
  durationSecs: number;
  rampUpSecs: number;
  thinkTimeMs: number;
  requestTimeoutMs: number;
  maxRetries: number;
  stopOnError: boolean;

  stream: boolean;
  maxTokens: number;
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  presencePenalty: number;
  frequencyPenalty: number;
  repetitionPenalty: number;
  seed: number;
  ignoreEos: boolean;
  systemPrompt: string;

  mix: Mix;
  selection: Selection;
  prefixCacheBust: boolean;

  metricsPollMs: number;
  streamFlushMs: number;
}

// ---- events from Rust ------------------------------------------------------

export interface RequestStart {
  id: string;
  seq: number;
  worker: number;
  promptId: string;
  title: string;
  category: string;
  difficulty: Difficulty;
  promptChars: number;
  attempt: number;
  startedAt: number;
}

export interface Delta {
  id: string;
  text: string;
}

export interface RequestEnd {
  id: string;
  seq: number;
  worker: number;
  promptId: string;
  difficulty: Difficulty;
  ok: boolean;
  status: number;
  error: string | null;
  ttftMs: number | null;
  totalMs: number;
  outputTps: number | null;
  promptTokens: number;
  completionTokens: number;
  finishReason: string | null;
  text: string;
  finishedAt: number;
}

export type RunStateName =
  | "idle"
  | "running"
  | "stopping"
  | "done"
  | "cancelled"
  | "error";

export interface RunState {
  state: RunStateName;
  completed: number;
  failed: number;
  inFlight: number;
  dispatched: number;
  target: number | null;
  elapsedMs: number;
  message: string | null;
}

export interface ServerMetrics {
  ok: boolean;
  at: number;
  error: string | null;
  numRunning: number | null;
  numWaiting: number | null;
  kvCacheUsage: number | null;
  prefixHitRate: number | null;
  promptTokensTotal: number | null;
  generationTokensTotal: number | null;
  preemptionsTotal: number | null;
  requestsSuccessTotal: number | null;
  gpuCacheHitTokens: number | null;
  gpuCacheQueryTokens: number | null;
  specDraftTokens: number | null;
  specAcceptedTokens: number | null;
  specAcceptanceRate: number | null;
}

export interface ModelInfo {
  id: string;
  ownedBy: string;
  maxModelLen: number | null;
}

export interface Probe {
  reachable: boolean;
  error: string | null;
  version: string | null;
  models: ModelInfo[];
  metricsAvailable: boolean;
  latencyMs: number;
  /** Speculative-decoding servers reject min_p and logit_bias with a 400. */
  specDecoding: boolean;
}

// ---- derived, frontend-only ------------------------------------------------

/** A request currently streaming, or one that just finished. */
export interface LiveStream {
  id: string;
  seq: number;
  worker: number;
  promptId: string;
  title: string;
  category: string;
  difficulty: Difficulty;
  promptChars: number;
  startedAt: number;
  text: string;
  chars: number;
  done: boolean;
  ok: boolean;
  error: string | null;
  ttftMs: number | null;
  totalMs: number | null;
  outputTps: number | null;
  completionTokens: number;
  promptTokens: number;
  finishReason: string | null;
}

export interface CompletedRow extends RequestEnd {
  title: string;
  category: string;
}
