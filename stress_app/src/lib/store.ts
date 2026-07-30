import { create } from "zustand";
import type {
  CompletedRow,
  Delta,
  Difficulty,
  LiveStream,
  Probe,
  PromptItem,
  RequestEnd,
  RequestStart,
  RunConfig,
  RunState,
  ServerMetrics,
} from "./types";
import { ALL_PROMPTS } from "./prompts";

export const DEFAULT_CONFIG: RunConfig = {
  baseUrl: "http://10.150.0.30:1234/v1",
  model: "nvidia/Gemma-4-31B-IT-NVFP4",
  apiKey: "",

  concurrency: 8,
  mode: "count",
  totalRequests: 100,
  durationSecs: 60,
  rampUpSecs: 0,
  thinkTimeMs: 0,
  requestTimeoutMs: 120_000,
  maxRetries: 0,
  stopOnError: false,

  stream: true,
  maxTokens: 512,
  temperature: 0.7,
  topP: 0.95,
  topK: 0,
  minP: 0,
  presencePenalty: 0,
  frequencyPenalty: 0,
  repetitionPenalty: 0,
  seed: -1,
  ignoreEos: false,
  systemPrompt: "",

  mix: { easy: 1, medium: 1, hard: 1 },
  selection: "random",
  prefixCacheBust: false,

  metricsPollMs: 1000,
  streamFlushMs: 60,
};

const IDLE_RUN: RunState = {
  state: "idle",
  completed: 0,
  failed: 0,
  inFlight: 0,
  dispatched: 0,
  target: null,
  elapsedMs: 0,
  message: null,
};

export interface LogLine {
  id: number;
  t: number;
  level: "info" | "ok" | "warn" | "fail";
  text: string;
}

/** One second of aggregate run telemetry, kept for the charts. */
export interface Sample {
  t: number;
  outTps: number;
  reqPerSec: number;
  inFlight: number;
  p50Ttft: number | null;
  p99Ttft: number | null;
  kvCache: number | null;
  queued: number | null;
  running: number | null;
}

const MAX_LOG = 600;
const MAX_CARDS = 60;
const MAX_SAMPLES = 900;
const CARD_TAIL = 4000;

interface Store {
  config: RunConfig;
  setConfig: (patch: Partial<RunConfig>) => void;
  resetConfig: () => void;

  enabled: Record<string, boolean>;
  togglePrompt: (id: string) => void;
  setPromptsEnabled: (ids: string[], on: boolean) => void;
  selectedPrompts: () => PromptItem[];

  probe: Probe | null;
  setProbe: (p: Probe | null) => void;

  run: RunState;
  live: Record<string, LiveStream>;
  order: string[];
  completed: CompletedRow[];
  log: LogLine[];
  metrics: ServerMetrics | null;
  series: Sample[];

  /** Cumulative completion tokens; the per-second sampler diffs this. */
  tokensTotal: number;
  lastSample: { t: number; tokens: number; completed: number };

  onStart: (e: RequestStart) => void;
  onDelta: (batch: Delta[]) => void;
  onEnd: (e: RequestEnd) => void;
  onRunState: (s: RunState) => void;
  onMetrics: (m: ServerMetrics) => void;
  sample: () => void;

  addLog: (level: LogLine["level"], text: string) => void;
  resetRun: () => void;
}

let logSeq = 0;

const promptById = new Map(ALL_PROMPTS.map((p) => [p.id, p]));

function loadConfig(): RunConfig {
  try {
    const raw = localStorage.getItem("rig.config");
    if (!raw) return DEFAULT_CONFIG;
    // Merge so a config saved by an older build still gains new fields.
    return { ...DEFAULT_CONFIG, ...JSON.parse(raw) };
  } catch {
    return DEFAULT_CONFIG;
  }
}

function loadEnabled(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem("rig.enabled");
    if (raw) return JSON.parse(raw);
  } catch {
    /* fall through to "everything on" */
  }
  return {};
}

/** Absent from the map means enabled — a fresh corpus is fully armed. */
function isOn(enabled: Record<string, boolean>, id: string): boolean {
  return enabled[id] !== false;
}

export const useStore = create<Store>((set, get) => ({
  config: loadConfig(),
  setConfig: (patch) =>
    set((s) => {
      const config = { ...s.config, ...patch };
      try {
        localStorage.setItem("rig.config", JSON.stringify(config));
      } catch {
        /* private mode — settings just won't persist */
      }
      return { config };
    }),
  resetConfig: () => {
    try {
      localStorage.removeItem("rig.config");
    } catch {
      /* ignore */
    }
    set({ config: DEFAULT_CONFIG });
  },

  enabled: loadEnabled(),
  togglePrompt: (id) =>
    set((s) => {
      const enabled = { ...s.enabled, [id]: !isOn(s.enabled, id) };
      try {
        localStorage.setItem("rig.enabled", JSON.stringify(enabled));
      } catch {
        /* ignore */
      }
      return { enabled };
    }),
  setPromptsEnabled: (ids, on) =>
    set((s) => {
      const enabled = { ...s.enabled };
      for (const id of ids) enabled[id] = on;
      try {
        localStorage.setItem("rig.enabled", JSON.stringify(enabled));
      } catch {
        /* ignore */
      }
      return { enabled };
    }),
  selectedPrompts: () => {
    const { enabled, config } = get();
    const weights = config.mix;
    return ALL_PROMPTS.filter((p) => {
      if (!isOn(enabled, p.id)) return false;
      // A difficulty weighted to zero is excluded from the payload entirely,
      // so Rust never has to sample from a bucket the user disabled.
      return weights[p.difficulty] > 0;
    });
  },

  probe: null,
  setProbe: (probe) => set({ probe }),

  run: IDLE_RUN,
  live: {},
  order: [],
  completed: [],
  log: [],
  metrics: null,
  series: [],
  tokensTotal: 0,
  lastSample: { t: 0, tokens: 0, completed: 0 },

  onStart: (e) =>
    set((s) => {
      const card: LiveStream = {
        id: e.id,
        seq: e.seq,
        worker: e.worker,
        promptId: e.promptId,
        title: e.title,
        category: e.category,
        difficulty: e.difficulty,
        promptChars: e.promptChars,
        startedAt: e.startedAt,
        text: "",
        chars: 0,
        done: false,
        ok: false,
        error: null,
        ttftMs: null,
        totalMs: null,
        outputTps: null,
        completionTokens: 0,
        promptTokens: 0,
        finishReason: null,
      };

      const live = { ...s.live, [e.id]: card };
      let order = [e.id, ...s.order.filter((id) => id !== e.id)];

      // Evict finished cards past the cap; in-flight cards always survive.
      if (order.length > MAX_CARDS) {
        const keep: string[] = [];
        const dropped: string[] = [];
        for (const id of order) {
          if (keep.length < MAX_CARDS || !live[id]?.done) keep.push(id);
          else dropped.push(id);
        }
        for (const id of dropped) delete live[id];
        order = keep;
      }

      return { live, order };
    }),

  onDelta: (batch) =>
    set((s) => {
      if (batch.length === 0) return {};
      // Coalesce the batch per id first so each card is rewritten once.
      const merged = new Map<string, string>();
      for (const d of batch) {
        merged.set(d.id, (merged.get(d.id) ?? "") + d.text);
      }

      const live = { ...s.live };
      let touched = false;
      for (const [id, chunk] of merged) {
        const card = live[id];
        if (!card) continue;
        const text = card.text + chunk;
        live[id] = {
          ...card,
          // Only the tail is rendered, so only the tail is retained per card.
          // The authoritative full text arrives with request-end.
          text: text.length > CARD_TAIL ? text.slice(-CARD_TAIL) : text,
          chars: card.chars + chunk.length,
        };
        touched = true;
      }
      return touched ? { live } : {};
    }),

  onEnd: (e) =>
    set((s) => {
      const card = s.live[e.id];
      const meta = promptById.get(e.promptId);
      const live = card
        ? {
            ...s.live,
            [e.id]: {
              ...card,
              done: true,
              ok: e.ok,
              error: e.error,
              ttftMs: e.ttftMs,
              totalMs: e.totalMs,
              outputTps: e.outputTps,
              completionTokens: e.completionTokens,
              promptTokens: e.promptTokens,
              finishReason: e.finishReason,
              text: e.ok && e.text ? e.text.slice(-CARD_TAIL) : card.text,
              chars: e.text ? e.text.length : card.chars,
            },
          }
        : s.live;

      const row: CompletedRow = {
        ...e,
        title: meta?.title ?? e.promptId,
        category: meta?.category ?? "—",
      };

      const line: LogLine = e.ok
        ? {
            id: ++logSeq,
            t: e.finishedAt,
            level: "ok",
            text: `#${e.seq} ${e.promptId} · ttft ${
              e.ttftMs != null ? Math.round(e.ttftMs) : "—"
            }ms · ${e.completionTokens} tok · ${Math.round(e.totalMs)}ms`,
          }
        : {
            id: ++logSeq,
            t: e.finishedAt,
            level: "fail",
            text: `#${e.seq} ${e.promptId} FAILED · ${e.error ?? "unknown error"}`,
          };

      return {
        live,
        completed: [...s.completed, row],
        tokensTotal: s.tokensTotal + e.completionTokens,
        log: [line, ...s.log].slice(0, MAX_LOG),
      };
    }),

  onRunState: (r) =>
    set((s) => {
      if (r.state !== s.run.state && r.message) {
        const line: LogLine = {
          id: ++logSeq,
          t: Date.now(),
          level: "warn",
          text: r.message,
        };
        return { run: r, log: [line, ...s.log].slice(0, MAX_LOG) };
      }
      return { run: r };
    }),

  onMetrics: (m) => set({ metrics: m }),

  sample: () =>
    set((s) => {
      if (s.run.state !== "running") return {};
      const now = Date.now();
      const last = s.lastSample;
      if (last.t === 0) {
        return {
          lastSample: {
            t: now,
            tokens: s.tokensTotal,
            completed: s.run.completed,
          },
        };
      }
      const dt = (now - last.t) / 1000;
      if (dt <= 0) return {};

      // Percentiles over the trailing window, not the whole run, so the chart
      // shows degradation as it happens rather than a smoothed lifetime figure.
      const recent = s.completed.slice(-60).filter((c) => c.ok && c.ttftMs != null);
      const ttfts = recent.map((c) => c.ttftMs as number).sort((a, b) => a - b);
      const at = (p: number) =>
        ttfts.length
          ? ttfts[
              Math.min(
                ttfts.length - 1,
                Math.max(0, Math.ceil((p / 100) * ttfts.length) - 1),
              )
            ]
          : null;

      const sampleRow: Sample = {
        t: now,
        outTps: Math.max(0, (s.tokensTotal - last.tokens) / dt),
        reqPerSec: Math.max(0, (s.run.completed - last.completed) / dt),
        inFlight: s.run.inFlight,
        p50Ttft: at(50),
        p99Ttft: at(99),
        kvCache: s.metrics?.kvCacheUsage ?? null,
        queued: s.metrics?.numWaiting ?? null,
        running: s.metrics?.numRunning ?? null,
      };

      return {
        series: [...s.series, sampleRow].slice(-MAX_SAMPLES),
        lastSample: {
          t: now,
          tokens: s.tokensTotal,
          completed: s.run.completed,
        },
      };
    }),

  addLog: (level, text) =>
    set((s) => ({
      log: [{ id: ++logSeq, t: Date.now(), level, text }, ...s.log].slice(
        0,
        MAX_LOG,
      ),
    })),

  resetRun: () =>
    set({
      run: IDLE_RUN,
      live: {},
      order: [],
      completed: [],
      series: [],
      tokensTotal: 0,
      lastSample: { t: 0, tokens: 0, completed: 0 },
      log: [],
    }),
}));

export const DIFFICULTIES: Difficulty[] = ["easy", "medium", "hard"];
