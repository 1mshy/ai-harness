import { useCallback, useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useStore } from "./lib/store";
import { ALL_PROMPTS } from "./lib/prompts";
import type {
  Delta,
  Probe,
  RequestEnd,
  RequestStart,
  RunState,
  ServerMetrics,
} from "./lib/types";
import { num, pct, secs } from "./lib/format";
import SettingsModal from "./components/SettingsModal";
import Console from "./pages/Console";
import Metrics from "./pages/Metrics";
import Prompts from "./pages/Prompts";
import Results from "./pages/Results";

const PAGES = [
  { id: "console", label: "Console", sub: "Live output streams" },
  { id: "metrics", label: "Metrics", sub: "Throughput and latency" },
  { id: "prompts", label: "Corpus", sub: "Prompt library" },
  { id: "results", label: "Results", sub: "Completed requests" },
] as const;

type PageId = (typeof PAGES)[number]["id"];

export default function App() {
  const [page, setPage] = useState<PageId>("console");
  const [settingsOpen, setSettingsOpen] = useState(false);

  const config = useStore((s) => s.config);
  const run = useStore((s) => s.run);
  const metrics = useStore((s) => s.metrics);
  const probe = useStore((s) => s.probe);
  const completedCount = useStore((s) => s.completed.length);
  const liveCount = useStore((s) => s.order.length);
  const selectedPrompts = useStore((s) => s.selectedPrompts);
  const addLog = useStore((s) => s.addLog);
  const resetRun = useStore((s) => s.resetRun);
  const setProbe = useStore((s) => s.setProbe);

  const isRunning = run.state === "running" || run.state === "stopping";

  // ---- event wiring ------------------------------------------------------
  useEffect(() => {
    const store = useStore.getState;
    const unlisteners: Array<() => void> = [];
    let cancelled = false;

    const wire = async () => {
      const subs = await Promise.all([
        listen<RequestStart>("rig://request-start", (e) =>
          store().onStart(e.payload),
        ),
        listen<Delta[]>("rig://delta", (e) => store().onDelta(e.payload)),
        listen<RequestEnd>("rig://request-end", (e) => store().onEnd(e.payload)),
        listen<RunState>("rig://run-state", (e) => store().onRunState(e.payload)),
        listen<ServerMetrics>("rig://server-metrics", (e) =>
          store().onMetrics(e.payload),
        ),
      ]);
      if (cancelled) {
        subs.forEach((u) => u());
        return;
      }
      unlisteners.push(...subs);
    };

    void wire();
    return () => {
      cancelled = true;
      unlisteners.forEach((u) => u());
    };
  }, []);

  // ---- per-second sampler for the charts ---------------------------------
  useEffect(() => {
    const id = window.setInterval(() => useStore.getState().sample(), 1000);
    return () => window.clearInterval(id);
  }, []);

  // ---- probe the server once on launch -----------------------------------
  useEffect(() => {
    void (async () => {
      try {
        const { baseUrl, apiKey } = useStore.getState().config;
        const result = await invoke<Probe>("probe_server", { baseUrl, apiKey });
        setProbe(result);
        addLog(
          result.reachable ? "ok" : "fail",
          result.reachable
            ? `connected · vllm ${result.version ?? "?"} · ${result.models
                .map((m) => m.id)
                .join(", ")}`
            : `server unreachable · ${result.error ?? "no response"}`,
        );
      } catch (e) {
        addLog("fail", `probe error · ${String(e)}`);
      }
    })();
  }, [addLog, setProbe]);

  // ---- controls ----------------------------------------------------------
  const start = useCallback(async () => {
    const prompts = selectedPrompts();
    if (prompts.length === 0) {
      addLog(
        "warn",
        "No prompts armed — enable a difficulty in Prompt mix, or re-enable prompts on the Corpus page.",
      );
      return;
    }
    resetRun();
    useStore.setState({ run: { ...useStore.getState().run, state: "running" } });
    addLog(
      "info",
      `run start · ${prompts.length} prompts armed · concurrency ${config.concurrency} · ${
        config.mode === "count"
          ? `${config.totalRequests} requests`
          : config.mode === "duration"
            ? `${config.durationSecs}s`
            : "until stopped"
      }`,
    );
    try {
      await invoke("start_run", { config, prompts });
    } catch (e) {
      addLog("fail", `could not start · ${String(e)}`);
      useStore.setState({ run: { ...useStore.getState().run, state: "error" } });
    }
  }, [config, selectedPrompts, addLog, resetRun]);

  const stop = useCallback(async () => {
    useStore.setState({ run: { ...useStore.getState().run, state: "stopping" } });
    addLog("warn", "stop requested — draining in-flight requests");
    try {
      await invoke("stop_run");
    } catch (e) {
      addLog("fail", `stop failed · ${String(e)}`);
    }
  }, [addLog]);

  // ---- keyboard ----------------------------------------------------------
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key === ",") {
        e.preventDefault();
        setSettingsOpen((v) => !v);
      }
      if (mod && e.key === "Enter") {
        e.preventDefault();
        if (isRunning) void stop();
        else void start();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isRunning, start, stop]);

  const counts: Record<PageId, string> = useMemo(
    () => ({
      console: liveCount > 0 ? String(liveCount) : "",
      metrics: run.state === "running" ? "live" : "",
      prompts: String(ALL_PROMPTS.length),
      results: completedCount > 0 ? String(completedCount) : "",
    }),
    [liveCount, completedCount, run.state],
  );

  const active = PAGES.find((p) => p.id === page)!;
  const progress =
    run.target && run.target > 0 ? (run.completed + run.failed) / run.target : null;

  return (
    <>
      <div className="drag-strip" data-tauri-drag-region />
      <div className="shell">
        <nav className="rail" aria-label="Sections">
          <div className="rail__brand">
            <span className="rail__mark">
              stress<em>·</em>rig
            </span>
            <span className="rail__ver">v0.1</span>
          </div>

          <div className="rail__nav">
            {PAGES.map((p) => (
              <button
                key={p.id}
                className="rail__item"
                aria-current={page === p.id ? "page" : undefined}
                onClick={() => setPage(p.id)}
              >
                <span className="truncate">{p.label}</span>
                <span className="rail__count">{counts[p.id]}</span>
              </button>
            ))}
          </div>

          <div className="rail__spacer" />

          <div className="rail__foot">
            <div className="row" style={{ gap: "var(--space-2xs)" }}>
              <span
                className={`dot ${
                  isRunning ? "dot--live" : probe?.reachable ? "dot--ok" : "dot--fail"
                }`}
              />
              <span className="rail__ver truncate">
                {isRunning ? "running" : probe?.reachable ? "server up" : "server down"}
              </span>
            </div>
            <button
              className="btn btn--ghost btn--sm"
              onClick={() => setSettingsOpen(true)}
            >
              Settings ⌘,
            </button>
          </div>
        </nav>

        <main className="main">
          <header className="topbar">
            <div className="topbar__left">
              <h1>{active.label}</h1>
              <span className="topbar__sub truncate">{active.sub}</span>
            </div>
            <div className="topbar__right">
              <button className="btn" onClick={() => setSettingsOpen(true)}>
                Settings
              </button>
              {isRunning ? (
                <button
                  className="btn btn--danger"
                  onClick={() => void stop()}
                  disabled={run.state === "stopping"}
                  data-state={run.state === "stopping" ? "loading" : undefined}
                >
                  {run.state === "stopping" ? "Draining…" : "Stop ⌘↵"}
                </button>
              ) : (
                <button className="btn btn--primary" onClick={() => void start()}>
                  Start run ⌘↵
                </button>
              )}
            </div>
          </header>

          <div className={page === "console" ? "page page--flush" : "page"}>
            {page === "console" && <Console />}
            {page === "metrics" && <Metrics />}
            {page === "prompts" && <Prompts />}
            {page === "results" && <Results />}
          </div>
        </main>

        <footer className="statusbar">
          <span className="truncate">{config.model}</span>
          <span className="statusbar__sep" />
          <span>
            conc <b>{config.concurrency}</b>
          </span>
          <span className="statusbar__sep" />
          <span>
            done <b>{num(run.completed)}</b>
            {run.target ? <span className="dim">/{num(run.target)}</span> : null}
          </span>
          <span>
            in flight <b>{num(run.inFlight)}</b>
          </span>
          <span>
            failed{" "}
            <b style={run.failed > 0 ? { color: "var(--color-fail)" } : undefined}>
              {num(run.failed)}
            </b>
          </span>
          {progress != null ? (
            <span>
              <b>{pct(Math.min(1, progress))}</b>
            </span>
          ) : null}

          <span className="statusbar__push" />

          {metrics?.ok ? (
            <>
              <span>
                kv <b>{pct(metrics.kvCacheUsage, 1)}</b>
              </span>
              <span>
                queued <b>{num(metrics.numWaiting)}</b>
              </span>
              <span>
                srv running <b>{num(metrics.numRunning)}</b>
              </span>
              <span className="statusbar__sep" />
            </>
          ) : null}
          <span>
            elapsed <b>{secs(run.elapsedMs)}</b>
          </span>
        </footer>
      </div>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        locked={isRunning}
      />
    </>
  );
}
