import { useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useStore, DEFAULT_CONFIG } from "../lib/store";
import { PROMPT_COUNTS } from "../lib/prompts";
import type { Probe, RunMode, Selection } from "../lib/types";
import { Field, NumberInput, SliderInput, Switch } from "./ui";
import { ms, num } from "../lib/format";

const TABS = [
  { id: "connection", label: "Connection" },
  { id: "load", label: "Load shape" },
  { id: "sampling", label: "Sampling" },
  { id: "corpus", label: "Prompt mix" },
  { id: "advanced", label: "Advanced" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function SettingsModal({
  open,
  onClose,
  locked,
}: {
  open: boolean;
  onClose: () => void;
  locked: boolean;
}) {
  const config = useStore((s) => s.config);
  const setConfig = useStore((s) => s.setConfig);
  const resetConfig = useStore((s) => s.resetConfig);
  const probe = useStore((s) => s.probe);
  const setProbe = useStore((s) => s.setProbe);
  const addLog = useStore((s) => s.addLog);

  const [tab, setTab] = useState<TabId>("connection");
  const [testing, setTesting] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    // Move focus into the dialog so Escape and Tab behave.
    dialogRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  async function testConnection() {
    setTesting(true);
    try {
      const result = await invoke<Probe>("probe_server", {
        baseUrl: config.baseUrl,
        apiKey: config.apiKey,
      });
      setProbe(result);
      if (result.reachable) {
        addLog(
          "ok",
          `probe ok · ${result.models.length} model(s) · vllm ${
            result.version ?? "?"
          } · ${Math.round(result.latencyMs)}ms`,
        );
        // Adopt the served model if the configured one isn't on the server.
        if (
          result.models.length > 0 &&
          !result.models.some((m) => m.id === config.model)
        ) {
          setConfig({ model: result.models[0].id });
        }
      } else {
        addLog("fail", `probe failed · ${result.error ?? "unreachable"}`);
      }
    } catch (e) {
      addLog("fail", `probe error · ${String(e)}`);
    } finally {
      setTesting(false);
    }
  }

  const poolEasy = config.mix.easy > 0 ? PROMPT_COUNTS.easy : 0;
  const poolMedium = config.mix.medium > 0 ? PROMPT_COUNTS.medium : 0;
  const poolHard = config.mix.hard > 0 ? PROMPT_COUNTS.hard : 0;
  const weightTotal = config.mix.easy + config.mix.medium + config.mix.hard;
  const share = (w: number) => (weightTotal > 0 ? `${Math.round((w / weightTotal) * 100)}%` : "0%");

  return (
    <div
      className="overlay"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label="Run settings"
        tabIndex={-1}
        ref={dialogRef}
      >
        <header className="modal__head">
          <h2 className="modal__title">Run settings</h2>
          <div className="row">
            {locked ? (
              <span className="badge badge--muted">
                run in progress · applies to next run
              </span>
            ) : null}
            <button className="btn btn--ghost" onClick={onClose}>
              Close ⎋
            </button>
          </div>
        </header>

        <div className="modal__grid">
          <div className="modal__tabs" role="tablist" aria-label="Settings sections">
            {TABS.map((t) => (
              <button
                key={t.id}
                className="modal__tab"
                role="tab"
                aria-selected={tab === t.id}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div className="modal__panel" role="tabpanel">
            {tab === "connection" && (
              <div className="form-section">
                <div className="stack">
                  <Field
                    label="Base URL"
                    hint="OpenAI-compatible root. /chat/completions is appended; /metrics is read from the server root."
                  >
                    <input
                      className="input"
                      value={config.baseUrl}
                      spellCheck={false}
                      onChange={(e) => setConfig({ baseUrl: e.target.value })}
                    />
                  </Field>

                  <div className="form-grid">
                    <Field label="Model" hint="Populated by Test connection.">
                      {probe && probe.models.length > 0 ? (
                        <select
                          className="select"
                          value={config.model}
                          onChange={(e) => setConfig({ model: e.target.value })}
                        >
                          {probe.models.map((m) => (
                            <option key={m.id} value={m.id}>
                              {m.id}
                            </option>
                          ))}
                          {!probe.models.some((m) => m.id === config.model) && (
                            <option value={config.model}>{config.model}</option>
                          )}
                        </select>
                      ) : (
                        <input
                          className="input"
                          value={config.model}
                          spellCheck={false}
                          onChange={(e) => setConfig({ model: e.target.value })}
                        />
                      )}
                    </Field>

                    <Field label="API key" hint="Blank for an open vLLM server.">
                      <input
                        className="input"
                        type="password"
                        value={config.apiKey}
                        placeholder="optional"
                        onChange={(e) => setConfig({ apiKey: e.target.value })}
                      />
                    </Field>
                  </div>

                  <div className="row">
                    <button
                      className="btn"
                      onClick={testConnection}
                      disabled={testing}
                      data-state={testing ? "loading" : undefined}
                    >
                      {testing ? "Probing…" : "Test connection"}
                    </button>
                    {probe ? (
                      <span
                        className={`badge badge--${probe.reachable ? "ok" : "fail"}`}
                      >
                        {probe.reachable ? "reachable" : "unreachable"}
                      </span>
                    ) : null}
                  </div>

                  {probe ? (
                    <div className="panel">
                      <div className="panel__body">
                        {probe.reachable ? (
                          <div className="stack">
                            <div className="row mono" style={{ fontSize: "var(--text-2xs)" }}>
                              <span className="dim">engine</span>
                              <b>{probe.version ?? "unknown"}</b>
                              <span className="statusbar__sep" />
                              <span className="dim">round trip</span>
                              <b>{ms(probe.latencyMs)}</b>
                              <span className="statusbar__sep" />
                              <span className="dim">/metrics</span>
                              <b>
                                {probe.metricsAvailable ? "available" : "not exposed"}
                              </b>
                            </div>
                            {probe.models.map((m) => (
                              <div
                                key={m.id}
                                className="row mono"
                                style={{ fontSize: "var(--text-2xs)" }}
                              >
                                <span className="badge badge--muted">{m.id}</span>
                                <span className="dim">
                                  ctx{" "}
                                  {m.maxModelLen ? num(m.maxModelLen) : "unknown"}
                                </span>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="field__error">{probe.error}</p>
                        )}
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            )}

            {tab === "load" && (
              <div className="stack">
                <div className="form-section">
                  <h3 className="form-section__title">Concurrency</h3>
                  <Field
                    label="Parallel workers"
                    hint="Each worker holds one in-flight request. This is the primary pressure knob."
                  >
                    <SliderInput
                      value={config.concurrency}
                      onChange={(v) => setConfig({ concurrency: v })}
                      min={1}
                      max={256}
                    />
                  </Field>
                </div>

                <div className="form-section">
                  <h3 className="form-section__title">Stop condition</h3>
                  <div className="form-grid">
                    <Field label="Mode">
                      <select
                        className="select"
                        value={config.mode}
                        onChange={(e) =>
                          setConfig({ mode: e.target.value as RunMode })
                        }
                      >
                        <option value="count">Fixed request count</option>
                        <option value="duration">Fixed duration</option>
                        <option value="infinite">Run until stopped</option>
                      </select>
                    </Field>

                    {config.mode === "count" && (
                      <Field label="Total requests">
                        <NumberInput
                          value={config.totalRequests}
                          onChange={(v) => setConfig({ totalRequests: v })}
                          min={1}
                          max={100000}
                        />
                      </Field>
                    )}

                    {config.mode === "duration" && (
                      <Field label="Duration (s)">
                        <NumberInput
                          value={config.durationSecs}
                          onChange={(v) => setConfig({ durationSecs: v })}
                          min={1}
                          max={86400}
                        />
                      </Field>
                    )}
                  </div>
                </div>

                <div className="form-section">
                  <h3 className="form-section__title">Pacing</h3>
                  <div className="form-grid">
                    <Field
                      label="Ramp-up (s)"
                      hint="Stagger worker starts instead of a thundering herd at t=0."
                    >
                      <NumberInput
                        value={config.rampUpSecs}
                        onChange={(v) => setConfig({ rampUpSecs: v })}
                        min={0}
                        max={600}
                      />
                    </Field>
                    <Field
                      label="Think time (ms)"
                      hint="Idle gap a worker waits after each response."
                    >
                      <NumberInput
                        value={config.thinkTimeMs}
                        onChange={(v) => setConfig({ thinkTimeMs: v })}
                        min={0}
                        max={60000}
                        step={50}
                      />
                    </Field>
                    <Field label="Request timeout (ms)">
                      <NumberInput
                        value={config.requestTimeoutMs}
                        onChange={(v) => setConfig({ requestTimeoutMs: v })}
                        min={1000}
                        max={1800000}
                        step={1000}
                      />
                    </Field>
                    <Field
                      label="Retries per request"
                      hint="Retried requests are re-dispatched with exponential backoff."
                    >
                      <NumberInput
                        value={config.maxRetries}
                        onChange={(v) => setConfig({ maxRetries: v })}
                        min={0}
                        max={10}
                      />
                    </Field>
                  </div>
                  <div className="row" style={{ marginTop: "var(--space-sm)" }}>
                    <Switch
                      checked={config.stopOnError}
                      onChange={(v) => setConfig({ stopOnError: v })}
                      label="Abort the whole run on first failure"
                    />
                  </div>
                </div>
              </div>
            )}

            {tab === "sampling" && (
              <div className="stack">
                <div className="form-section">
                  <h3 className="form-section__title">Decoding</h3>
                  <div className="form-grid">
                    <Field
                      label="Max tokens"
                      hint="Upper bound on completion length."
                    >
                      <NumberInput
                        value={config.maxTokens}
                        onChange={(v) => setConfig({ maxTokens: v })}
                        min={1}
                        max={32768}
                      />
                    </Field>
                    <Field label="Temperature">
                      <SliderInput
                        value={config.temperature}
                        onChange={(v) => setConfig({ temperature: v })}
                        min={0}
                        max={2}
                        step={0.05}
                      />
                    </Field>
                    <Field label="Top-p">
                      <SliderInput
                        value={config.topP}
                        onChange={(v) => setConfig({ topP: v })}
                        min={0.01}
                        max={1}
                        step={0.01}
                      />
                    </Field>
                    <Field label="Top-k" hint="0 leaves it unset.">
                      <NumberInput
                        value={config.topK}
                        onChange={(v) => setConfig({ topK: v })}
                        min={0}
                        max={1000}
                      />
                    </Field>
                    <Field label="Min-p" hint="0 leaves it unset.">
                      <SliderInput
                        value={config.minP}
                        onChange={(v) => setConfig({ minP: v })}
                        min={0}
                        max={1}
                        step={0.01}
                      />
                    </Field>
                    <Field
                      label="Repetition penalty"
                      hint="0 leaves it unset. vLLM extension."
                    >
                      <SliderInput
                        value={config.repetitionPenalty}
                        onChange={(v) => setConfig({ repetitionPenalty: v })}
                        min={0}
                        max={2}
                        step={0.05}
                      />
                    </Field>
                    <Field label="Presence penalty">
                      <SliderInput
                        value={config.presencePenalty}
                        onChange={(v) => setConfig({ presencePenalty: v })}
                        min={-2}
                        max={2}
                        step={0.1}
                      />
                    </Field>
                    <Field label="Frequency penalty">
                      <SliderInput
                        value={config.frequencyPenalty}
                        onChange={(v) => setConfig({ frequencyPenalty: v })}
                        min={-2}
                        max={2}
                        step={0.1}
                      />
                    </Field>
                    <Field label="Seed" hint="-1 leaves it unset.">
                      <NumberInput
                        value={config.seed}
                        onChange={(v) => setConfig({ seed: v })}
                        min={-1}
                        max={2147483647}
                      />
                    </Field>
                  </div>
                </div>

                <div className="form-section">
                  <h3 className="form-section__title">Protocol</h3>
                  <div className="stack">
                    <Switch
                      checked={config.stream}
                      onChange={(v) => setConfig({ stream: v })}
                      label="Stream responses (required for true time-to-first-token)"
                    />
                    <Switch
                      checked={config.ignoreEos}
                      onChange={(v) => setConfig({ ignoreEos: v })}
                      label="ignore_eos — generate to max_tokens regardless of stop"
                    />
                    <p className="field__hint">
                      ignore_eos pins every completion to the same length, which
                      isolates decode throughput from the model's natural
                      stopping behaviour. Useful for comparing runs; not
                      representative of real traffic.
                    </p>
                  </div>
                </div>

                <div className="form-section">
                  <h3 className="form-section__title">System prompt</h3>
                  <Field
                    label="Prepended to every request"
                    hint="Leave blank to send the user turn alone."
                  >
                    <textarea
                      className="textarea"
                      value={config.systemPrompt}
                      placeholder="optional"
                      onChange={(e) => setConfig({ systemPrompt: e.target.value })}
                    />
                  </Field>
                </div>
              </div>
            )}

            {tab === "corpus" && (
              <div className="stack">
                <div className="form-section">
                  <h3 className="form-section__title">Difficulty weighting</h3>
                  <p className="field__hint" style={{ marginBottom: "var(--space-sm)" }}>
                    Relative weights, not counts. A difficulty set to 0 is
                    excluded from the run entirely.
                  </p>
                  <div className="stack">
                    <Field
                      label={`Easy — ${share(config.mix.easy)} of traffic · ${poolEasy} prompts available`}
                    >
                      <SliderInput
                        value={config.mix.easy}
                        onChange={(v) =>
                          setConfig({ mix: { ...config.mix, easy: v } })
                        }
                        min={0}
                        max={10}
                      />
                    </Field>
                    <Field
                      label={`Medium — ${share(config.mix.medium)} of traffic · ${poolMedium} prompts available`}
                    >
                      <SliderInput
                        value={config.mix.medium}
                        onChange={(v) =>
                          setConfig({ mix: { ...config.mix, medium: v } })
                        }
                        min={0}
                        max={10}
                      />
                    </Field>
                    <Field
                      label={`Hard — ${share(config.mix.hard)} of traffic · ${poolHard} prompts available`}
                    >
                      <SliderInput
                        value={config.mix.hard}
                        onChange={(v) =>
                          setConfig({ mix: { ...config.mix, hard: v } })
                        }
                        min={0}
                        max={10}
                      />
                    </Field>
                  </div>
                </div>

                <div className="form-section">
                  <h3 className="form-section__title">Selection</h3>
                  <div className="form-grid">
                    <Field
                      label="Order within a difficulty"
                      hint="Sequential walks the corpus; random samples with replacement."
                    >
                      <select
                        className="select"
                        value={config.selection}
                        onChange={(e) =>
                          setConfig({ selection: e.target.value as Selection })
                        }
                      >
                        <option value="random">Random</option>
                        <option value="sequential">Sequential</option>
                      </select>
                    </Field>
                  </div>
                  <div className="stack" style={{ marginTop: "var(--space-sm)" }}>
                    <Switch
                      checked={config.prefixCacheBust}
                      onChange={(v) => setConfig({ prefixCacheBust: v })}
                      label="Defeat prefix cache (unique nonce per request)"
                    />
                    <p className="field__hint">
                      vLLM caches shared prompt prefixes. Re-running the same
                      corpus will otherwise report throughput well above what
                      cold traffic would see. Turn this on when you want honest
                      prefill numbers; leave it off to measure the cache itself.
                    </p>
                  </div>
                </div>
              </div>
            )}

            {tab === "advanced" && (
              <div className="stack">
                <div className="form-section">
                  <h3 className="form-section__title">Telemetry</h3>
                  <div className="form-grid">
                    <Field
                      label="Server metrics poll (ms)"
                      hint="Scrape interval for the vLLM Prometheus endpoint."
                    >
                      <NumberInput
                        value={config.metricsPollMs}
                        onChange={(v) => setConfig({ metricsPollMs: v })}
                        min={200}
                        max={30000}
                        step={100}
                      />
                    </Field>
                    <Field
                      label="Stream flush (ms)"
                      hint="Deltas are batched at this cadence before crossing into the UI. Lower is smoother; higher costs less."
                    >
                      <NumberInput
                        value={config.streamFlushMs}
                        onChange={(v) => setConfig({ streamFlushMs: v })}
                        min={16}
                        max={1000}
                        step={10}
                      />
                    </Field>
                  </div>
                </div>

                <div className="form-section">
                  <h3 className="form-section__title">Reset</h3>
                  <div className="row">
                    <button
                      className="btn btn--danger"
                      onClick={() => {
                        resetConfig();
                        setTab("connection");
                      }}
                    >
                      Restore defaults
                    </button>
                    <span className="field__hint">
                      Returns every field to {DEFAULT_CONFIG.baseUrl} at
                      concurrency {DEFAULT_CONFIG.concurrency}.
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <footer className="modal__foot">
          <span className="field__hint">
            Settings save as you type and persist between launches.
          </span>
          <button className="btn btn--primary" onClick={onClose}>
            Done
          </button>
        </footer>
      </div>
    </div>
  );
}
