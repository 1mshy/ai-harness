import { memo, useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../lib/store";
import type { LiveStream } from "../lib/types";
import { bytes, clock, ms, num } from "../lib/format";
import { Badge, Empty } from "../components/ui";

type Filter = "all" | "live" | "failed";

/** One request's output, streaming in. Subscribes to its own slice so a delta
 *  on request #7 does not re-render the other 63 cards. */
const StreamCard = memo(function StreamCard({ id }: { id: string }) {
  const s = useStore((st) => st.live[id]) as LiveStream | undefined;
  const bodyRef = useRef<HTMLPreElement>(null);
  const pinnedRef = useRef(true);

  // Follow the tail while the user is at the bottom; stop the moment they
  // scroll up to read something, and resume when they return.
  useEffect(() => {
    const el = bodyRef.current;
    if (!el || !pinnedRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [s?.text]);

  if (!s) return null;

  const status = s.done ? (s.ok ? "ok" : "fail") : "live";

  return (
    <article className="stream" data-live={!s.done} data-status={status}>
      <header className="stream__head">
        <span
          className={`dot ${
            status === "live" ? "dot--live" : status === "ok" ? "dot--ok" : "dot--fail"
          }`}
          aria-hidden="true"
        />
        <span className="stream__title" title={`${s.promptId} — ${s.title}`}>
          {s.title}
        </span>
        <span className="row" style={{ gap: "var(--space-2xs)", flexWrap: "nowrap" }}>
          <Badge kind={s.difficulty}>{s.difficulty}</Badge>
          <span className="stream__seq">#{s.seq}</span>
        </span>
      </header>

      {s.done && !s.ok ? (
        <p className="stream__err">
          {s.error ?? "request failed"}
        </p>
      ) : (
        <pre className="stream__body" ref={bodyRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            pinnedRef.current =
              el.scrollHeight - el.scrollTop - el.clientHeight < 24;
          }}
        >
          {s.text}
          {!s.done && s.text ? <span className="stream__caret" /> : null}
        </pre>
      )}

      <footer className="stream__foot">
        <span>
          w<b>{s.worker}</b>
        </span>
        <span>
          ttft <b>{ms(s.ttftMs)}</b>
        </span>
        <span>
          tok <b>{num(s.completionTokens || null)}</b>
        </span>
        <span>
          tok/s <b>{s.outputTps ? s.outputTps.toFixed(1) : "—"}</b>
        </span>
        <span>
          total <b>{ms(s.totalMs)}</b>
        </span>
        <span className="dim">{bytes(s.chars)}</span>
        {s.finishReason && s.finishReason !== "stop" ? (
          <span className="badge badge--muted">{s.finishReason}</span>
        ) : null}
      </footer>
    </article>
  );
});

export default function Console() {
  const order = useStore((st) => st.order);
  const live = useStore((st) => st.live);
  const log = useStore((st) => st.log);
  const runState = useStore((st) => st.run.state);
  const [cols, setCols] = useState(2);
  const [filter, setFilter] = useState<Filter>("all");

  const visible = useMemo(() => {
    if (filter === "all") return order;
    return order.filter((id) => {
      const s = live[id];
      if (!s) return false;
      if (filter === "live") return !s.done;
      return s.done && !s.ok;
    });
  }, [order, live, filter]);

  return (
    <div className="console">
      <div className="console__streams" data-cols={cols}>
        <div
          className="row"
          style={{
            gridColumn: "1 / -1",
            justifyContent: "space-between",
            marginBottom: "var(--space-3xs)",
          }}
        >
          <div className="row" role="group" aria-label="Filter streams">
            {(["all", "live", "failed"] as const).map((f) => (
              <button
                key={f}
                className={`btn btn--sm${filter === f ? "" : " btn--ghost"}`}
                onClick={() => setFilter(f)}
                aria-pressed={filter === f}
              >
                {f}
              </button>
            ))}
          </div>
          <div className="row" role="group" aria-label="Grid density">
            <span className="field__label">columns</span>
            {[1, 2, 3, 4].map((c) => (
              <button
                key={c}
                className={`btn btn--sm${cols === c ? "" : " btn--ghost"}`}
                onClick={() => setCols(c)}
                aria-pressed={cols === c}
              >
                {c}
              </button>
            ))}
          </div>
        </div>

        {visible.length === 0 ? (
          <div style={{ gridColumn: "1 / -1" }}>
            <Empty
              title={
                runState === "running"
                  ? "Waiting for the first request to open"
                  : "No streams yet"
              }
              hint={
                filter !== "all"
                  ? "Nothing matches this filter right now."
                  : "Press Start run (⌘↵) to begin. Every in-flight request opens a card here and fills in as tokens arrive."
              }
            />
          </div>
        ) : (
          visible.map((id) => <StreamCard key={id} id={id} />)
        )}
      </div>

      <aside className="console__side">
        <div className="panel__head">
          <h2 className="panel__title">Event log</h2>
          <span className="rail__ver">{log.length}</span>
        </div>
        <div className="log">
          {log.length === 0 ? (
            <p className="field__hint" style={{ padding: "var(--space-xs)" }}>
              Request completions, failures, and run transitions land here
              newest-first.
            </p>
          ) : (
            log.map((l) => (
              <div className="log__row" key={l.id} data-level={l.level}>
                <span className="log__t">{clock(l.t)}</span>
                <span className="log__msg">{l.text}</span>
              </div>
            ))
          )}
        </div>
      </aside>
    </div>
  );
}
