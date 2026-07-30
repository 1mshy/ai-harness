use crate::telemetry::{self, now_ms};
use crate::types::*;
use futures_util::StreamExt;
use rand::Rng;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};
use tokio::sync::mpsc;

const MAX_STORED_CHARS: usize = 200_000;

/// Prompt pools, split by difficulty, with a per-pool cursor for sequential
/// selection.
struct Pools {
    easy: Vec<PromptItem>,
    medium: Vec<PromptItem>,
    hard: Vec<PromptItem>,
    cur_easy: AtomicU64,
    cur_medium: AtomicU64,
    cur_hard: AtomicU64,
}

impl Pools {
    fn build(items: Vec<PromptItem>) -> Self {
        let mut easy = Vec::new();
        let mut medium = Vec::new();
        let mut hard = Vec::new();
        for p in items {
            match p.difficulty.as_str() {
                "easy" => easy.push(p),
                "medium" => medium.push(p),
                "hard" => hard.push(p),
                _ => medium.push(p),
            }
        }
        Pools {
            easy,
            medium,
            hard,
            cur_easy: AtomicU64::new(0),
            cur_medium: AtomicU64::new(0),
            cur_hard: AtomicU64::new(0),
        }
    }

    fn total(&self) -> usize {
        self.easy.len() + self.medium.len() + self.hard.len()
    }

    /// Weighted difficulty pick, then sequential or random selection inside it.
    fn pick(&self, mix: &Mix, sequential: bool) -> Option<&PromptItem> {
        let mut buckets: Vec<(&Vec<PromptItem>, &AtomicU64, u32)> = Vec::with_capacity(3);
        if !self.easy.is_empty() && mix.easy > 0 {
            buckets.push((&self.easy, &self.cur_easy, mix.easy));
        }
        if !self.medium.is_empty() && mix.medium > 0 {
            buckets.push((&self.medium, &self.cur_medium, mix.medium));
        }
        if !self.hard.is_empty() && mix.hard > 0 {
            buckets.push((&self.hard, &self.cur_hard, mix.hard));
        }
        if buckets.is_empty() {
            return None;
        }

        let total_weight: u32 = buckets.iter().map(|b| b.2).sum();
        // Scope the RNG so it is dropped before any await in the caller.
        let (roll, rand_idx) = {
            let mut rng = rand::thread_rng();
            (rng.gen_range(0..total_weight), rng.gen::<u64>())
        };

        let mut acc = 0u32;
        for (pool, cursor, weight) in buckets {
            acc += weight;
            if roll < acc {
                let idx = if sequential {
                    cursor.fetch_add(1, Ordering::Relaxed)
                } else {
                    rand_idx
                };
                return pool.get((idx % pool.len() as u64) as usize);
            }
        }
        None
    }
}

struct Counters {
    dispatched: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    in_flight: AtomicU64,
    seq: AtomicU64,
}

/// Outcome of one HTTP call, before it is turned into a `RequestEnd`.
struct Attempt {
    ok: bool,
    status: u16,
    error: Option<String>,
    ttft_ms: Option<f64>,
    total_ms: f64,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: Option<String>,
    text: String,
}

pub async fn run(
    app: AppHandle,
    cfg: RunConfig,
    prompts: Vec<PromptItem>,
    cancel: Arc<AtomicBool>,
) {
    let pools = Arc::new(Pools::build(prompts));
    if pools.total() == 0 {
        emit_state(&app, "error", &Counters::new(), None, 0.0, Some("No prompts selected — enable at least one difficulty in the prompt library.".into()));
        cancel.store(true, Ordering::Relaxed);
        return;
    }

    let concurrency = cfg.concurrency.clamp(1, 512);
    let client = match reqwest::Client::builder()
        .pool_max_idle_per_host(concurrency * 2)
        .connect_timeout(Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            emit_state(&app, "error", &Counters::new(), None, 0.0, Some(format!("HTTP client: {e}")));
            cancel.store(true, Ordering::Relaxed);
            return;
        }
    };

    let counters = Arc::new(Counters::new());
    let cfg = Arc::new(cfg);
    let started = Instant::now();

    let target: Option<u64> = match cfg.mode.as_str() {
        "count" => Some(cfg.total_requests as u64),
        _ => None,
    };
    let deadline: Option<Instant> = match cfg.mode.as_str() {
        "duration" => Some(started + Duration::from_secs(cfg.duration_secs.max(1))),
        _ => None,
    };

    // --- delta coalescing -------------------------------------------------
    // One IPC message per token would drown the webview at high concurrency,
    // so deltas are buffered and flushed on a fixed cadence instead.
    let (tx, mut rx) = mpsc::unbounded_channel::<Delta>();
    let flush_ms = cfg.stream_flush_ms.clamp(16, 1000);
    let flush_app = app.clone();
    let flusher = tauri::async_runtime::spawn(async move {
        let mut buf: Vec<Delta> = Vec::with_capacity(512);
        let mut ticker = tokio::time::interval(Duration::from_millis(flush_ms));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Some(d) => buf.push(d),
                        None => {
                            if !buf.is_empty() {
                                let _ = flush_app.emit("rig://delta", &buf);
                            }
                            break;
                        }
                    }
                }
                _ = ticker.tick() => {
                    if !buf.is_empty() {
                        let _ = flush_app.emit("rig://delta", &buf);
                        buf.clear();
                    }
                }
            }
        }
    });

    // --- progress heartbeat ----------------------------------------------
    let hb_app = app.clone();
    let hb_counters = counters.clone();
    let hb_cancel = cancel.clone();
    let heartbeat = tauri::async_runtime::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_millis(250));
        while !hb_cancel.load(Ordering::Relaxed) {
            ticker.tick().await;
            emit_state(
                &hb_app,
                "running",
                &hb_counters,
                target,
                started.elapsed().as_secs_f64() * 1000.0,
                None,
            );
        }
    });

    // --- vLLM /metrics poller --------------------------------------------
    let metrics_task = tauri::async_runtime::spawn(telemetry::poll_loop(
        app.clone(),
        client.clone(),
        cfg.base_url.clone(),
        cfg.metrics_poll_ms,
        cancel.clone(),
    ));

    // --- workers ----------------------------------------------------------
    let mut workers = Vec::with_capacity(concurrency);
    for worker_id in 0..concurrency {
        let app = app.clone();
        let cfg = cfg.clone();
        let pools = pools.clone();
        let client = client.clone();
        let counters = counters.clone();
        let cancel = cancel.clone();
        let tx = tx.clone();

        workers.push(tauri::async_runtime::spawn(async move {
            // Ramp: stagger worker start so the server sees load climb rather
            // than a thundering herd at t=0.
            if cfg.ramp_up_secs > 0 && concurrency > 1 {
                let offset = (cfg.ramp_up_secs as f64 * 1000.0) * (worker_id as f64)
                    / (concurrency as f64);
                tokio::time::sleep(Duration::from_millis(offset as u64)).await;
            }

            loop {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                if let Some(d) = deadline {
                    if Instant::now() >= d {
                        break;
                    }
                }
                if let Some(t) = target {
                    // Claim a slot before doing any work.
                    let claimed = counters.dispatched.fetch_add(1, Ordering::SeqCst);
                    if claimed >= t {
                        counters.dispatched.fetch_sub(1, Ordering::SeqCst);
                        break;
                    }
                } else {
                    counters.dispatched.fetch_add(1, Ordering::SeqCst);
                }

                let Some(prompt) = pools.pick(&cfg.mix, cfg.selection == "sequential").cloned()
                else {
                    break;
                };

                let seq = counters.seq.fetch_add(1, Ordering::SeqCst);
                let id = format!("r{seq}");
                let mut attempt_no = 0u32;

                loop {
                    counters.in_flight.fetch_add(1, Ordering::SeqCst);
                    let _ = app.emit(
                        "rig://request-start",
                        &RequestStart {
                            id: id.clone(),
                            seq,
                            worker: worker_id,
                            prompt_id: prompt.id.clone(),
                            title: prompt.title.clone(),
                            category: prompt.category.clone(),
                            difficulty: prompt.difficulty.clone(),
                            prompt_chars: prompt.text.len(),
                            attempt: attempt_no,
                            started_at: now_ms(),
                        },
                    );

                    let outcome = execute(&client, &cfg, &prompt, &id, &tx, &cancel).await;
                    counters.in_flight.fetch_sub(1, Ordering::SeqCst);

                    let retryable = !outcome.ok
                        && attempt_no < cfg.max_retries
                        && !cancel.load(Ordering::Relaxed);

                    let mut text = outcome.text.clone();
                    if text.len() > MAX_STORED_CHARS {
                        text.truncate(MAX_STORED_CHARS);
                        text.push_str("\n… [truncated]");
                    }

                    let _ = app.emit(
                        "rig://request-end",
                        &RequestEnd {
                            id: id.clone(),
                            seq,
                            worker: worker_id,
                            prompt_id: prompt.id.clone(),
                            difficulty: prompt.difficulty.clone(),
                            ok: outcome.ok,
                            status: outcome.status,
                            error: outcome.error.clone(),
                            ttft_ms: outcome.ttft_ms,
                            total_ms: outcome.total_ms,
                            output_tps: decode_tps(&outcome),
                            prompt_tokens: outcome.prompt_tokens,
                            completion_tokens: outcome.completion_tokens,
                            finish_reason: outcome.finish_reason.clone(),
                            text,
                            finished_at: now_ms(),
                        },
                    );

                    if outcome.ok {
                        counters.completed.fetch_add(1, Ordering::SeqCst);
                        break;
                    }

                    if retryable {
                        attempt_no += 1;
                        // Exponential backoff, capped.
                        let backoff = (200u64 << attempt_no.min(5)).min(5_000);
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                        continue;
                    }

                    counters.failed.fetch_add(1, Ordering::SeqCst);
                    if cfg.stop_on_error {
                        cancel.store(true, Ordering::Relaxed);
                    }
                    break;
                }

                if cfg.think_time_ms > 0 && !cancel.load(Ordering::Relaxed) {
                    tokio::time::sleep(Duration::from_millis(cfg.think_time_ms)).await;
                }
            }
        }));
    }

    drop(tx);
    for w in workers {
        let _ = w.await;
    }

    // Workers are done — wind the support tasks down.
    let was_cancelled = cancel.load(Ordering::Relaxed);
    cancel.store(true, Ordering::Relaxed);
    let _ = flusher.await;
    heartbeat.abort();
    metrics_task.abort();

    // One last metrics read so the dashboard shows the post-run resting state.
    let _ = app.emit(
        "rig://server-metrics",
        &telemetry::scrape(&client, &cfg.base_url).await,
    );

    emit_state(
        &app,
        if was_cancelled { "cancelled" } else { "done" },
        &counters,
        target,
        started.elapsed().as_secs_f64() * 1000.0,
        None,
    );
}

fn decode_tps(a: &Attempt) -> Option<f64> {
    let ttft = a.ttft_ms?;
    let decode_ms = a.total_ms - ttft;
    if decode_ms <= 0.0 || a.completion_tokens <= 1 {
        return None;
    }
    Some((a.completion_tokens as f64) / (decode_ms / 1000.0))
}

impl Counters {
    fn new() -> Self {
        Counters {
            dispatched: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            in_flight: AtomicU64::new(0),
            seq: AtomicU64::new(0),
        }
    }
}

fn emit_state(
    app: &AppHandle,
    state: &str,
    c: &Counters,
    target: Option<u64>,
    elapsed_ms: f64,
    message: Option<String>,
) {
    let _ = app.emit(
        "rig://run-state",
        &RunState {
            state: state.to_string(),
            completed: c.completed.load(Ordering::Relaxed),
            failed: c.failed.load(Ordering::Relaxed),
            in_flight: c.in_flight.load(Ordering::Relaxed),
            dispatched: c.dispatched.load(Ordering::Relaxed),
            target,
            elapsed_ms,
            message,
        },
    );
}

/// Build the chat-completions body. vLLM accepts its extended sampling params
/// (`top_k`, `min_p`, `repetition_penalty`, `ignore_eos`) at the top level of
/// the OpenAI-compatible payload.
fn build_body(cfg: &RunConfig, prompt_text: &str) -> serde_json::Value {
    let mut messages = Vec::new();
    if !cfg.system_prompt.trim().is_empty() {
        messages.push(serde_json::json!({
            "role": "system",
            "content": cfg.system_prompt,
        }));
    }
    messages.push(serde_json::json!({ "role": "user", "content": prompt_text }));

    let mut body = serde_json::json!({
        "model": cfg.model,
        "messages": messages,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "presence_penalty": cfg.presence_penalty,
        "frequency_penalty": cfg.frequency_penalty,
        "stream": cfg.stream,
    });

    let map = body.as_object_mut().unwrap();
    if cfg.stream {
        map.insert(
            "stream_options".into(),
            serde_json::json!({ "include_usage": true }),
        );
    }
    if cfg.top_k > 0 {
        map.insert("top_k".into(), serde_json::json!(cfg.top_k));
    }
    if cfg.min_p > 0.0 {
        map.insert("min_p".into(), serde_json::json!(cfg.min_p));
    }
    if cfg.repetition_penalty > 0.0 {
        map.insert(
            "repetition_penalty".into(),
            serde_json::json!(cfg.repetition_penalty),
        );
    }
    if cfg.seed >= 0 {
        map.insert("seed".into(), serde_json::json!(cfg.seed));
    }
    if cfg.ignore_eos {
        map.insert("ignore_eos".into(), serde_json::json!(true));
    }
    body
}

async fn execute(
    client: &reqwest::Client,
    cfg: &RunConfig,
    prompt: &PromptItem,
    id: &str,
    tx: &mpsc::UnboundedSender<Delta>,
    cancel: &Arc<AtomicBool>,
) -> Attempt {
    let text = if cfg.prefix_cache_bust {
        // A unique preamble defeats vLLM's prefix cache, which would otherwise
        // make repeat runs of the same corpus look far faster than they are.
        let nonce: u64 = rand::thread_rng().gen();
        format!("[trace {nonce:016x}]\n{}", prompt.text)
    } else {
        prompt.text.clone()
    };

    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let body = build_body(cfg, &text);

    let mut req = client.post(&url).json(&body);
    if !cfg.api_key.trim().is_empty() {
        req = req.bearer_auth(cfg.api_key.trim());
    }

    let timeout = Duration::from_millis(cfg.request_timeout_ms.max(1000));
    let start = Instant::now();

    let fut = async {
        let resp = req.send().await.map_err(|e| e.to_string())?;
        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let detail = resp.text().await.unwrap_or_default();
            let detail = detail.chars().take(400).collect::<String>();
            return Err(format!("HTTP {status}: {detail}"));
        }
        if cfg.stream {
            read_stream(resp, id, tx, cancel, start).await
        } else {
            read_whole(resp, start).await
        }
    };

    match tokio::time::timeout(timeout, fut).await {
        Ok(Ok(mut a)) => {
            a.total_ms = start.elapsed().as_secs_f64() * 1000.0;
            a
        }
        Ok(Err(e)) => Attempt {
            ok: false,
            status: 0,
            error: Some(e),
            ttft_ms: None,
            total_ms: start.elapsed().as_secs_f64() * 1000.0,
            prompt_tokens: 0,
            completion_tokens: 0,
            finish_reason: None,
            text: String::new(),
        },
        Err(_) => Attempt {
            ok: false,
            status: 0,
            error: Some(format!("timeout after {}ms", cfg.request_timeout_ms)),
            ttft_ms: None,
            total_ms: start.elapsed().as_secs_f64() * 1000.0,
            prompt_tokens: 0,
            completion_tokens: 0,
            finish_reason: None,
            text: String::new(),
        },
    }
}

async fn read_whole(resp: reqwest::Response, start: Instant) -> Result<Attempt, String> {
    let v: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let text = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let finish_reason = v["choices"][0]["finish_reason"].as_str().map(String::from);
    Ok(Attempt {
        ok: true,
        status: 200,
        error: None,
        // Without streaming there is no first-token signal; the whole response
        // is the first token as far as the client can tell.
        ttft_ms: Some(start.elapsed().as_secs_f64() * 1000.0),
        total_ms: 0.0,
        prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
        finish_reason,
        text,
    })
}

/// One decoded SSE frame from the chat-completions stream.
#[derive(Debug, Default, PartialEq)]
struct Frame {
    done: bool,
    content: Option<String>,
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
    finish_reason: Option<String>,
}

/// Decode a single SSE line. Returns `None` for keepalives, comments, blank
/// lines, and anything that is not a `data:` frame.
fn parse_sse_line(line: &str) -> Option<Frame> {
    let payload = line.trim().strip_prefix("data:")?.trim();
    if payload.is_empty() {
        return None;
    }
    if payload == "[DONE]" {
        return Some(Frame {
            done: true,
            ..Default::default()
        });
    }

    let v: serde_json::Value = serde_json::from_str(payload).ok()?;
    let mut frame = Frame::default();

    if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
        frame.prompt_tokens = u["prompt_tokens"].as_u64().map(|n| n as u32);
        frame.completion_tokens = u["completion_tokens"].as_u64().map(|n| n as u32);
    }
    if let Some(fr) = v["choices"][0]["finish_reason"].as_str() {
        frame.finish_reason = Some(fr.to_string());
    }
    if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
        if !delta.is_empty() {
            frame.content = Some(delta.to_string());
        }
    }
    Some(frame)
}

async fn read_stream(
    resp: reqwest::Response,
    id: &str,
    tx: &mpsc::UnboundedSender<Delta>,
    cancel: &Arc<AtomicBool>,
    start: Instant,
) -> Result<Attempt, String> {
    let mut stream = resp.bytes_stream();
    let mut pending = String::new();
    let mut full = String::new();
    let mut ttft_ms: Option<f64> = None;
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;
    let mut finish_reason: Option<String> = None;

    'outer: while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        let bytes = chunk.map_err(|e| e.to_string())?;
        pending.push_str(&String::from_utf8_lossy(&bytes));

        // SSE frames are newline-delimited; a chunk can split one mid-line, so
        // only complete lines are consumed and the remainder is carried over.
        while let Some(nl) = pending.find('\n') {
            let line: String = pending.drain(..=nl).collect();
            let Some(frame) = parse_sse_line(&line) else {
                continue;
            };
            if frame.done {
                break 'outer;
            }
            if let Some(p) = frame.prompt_tokens {
                prompt_tokens = p;
            }
            if let Some(c) = frame.completion_tokens {
                completion_tokens = c;
            }
            if let Some(fr) = frame.finish_reason {
                finish_reason = Some(fr);
            }
            if let Some(delta) = frame.content {
                if ttft_ms.is_none() {
                    ttft_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                }
                full.push_str(&delta);
                let _ = tx.send(Delta {
                    id: id.to_string(),
                    text: delta,
                });
            }
        }
    }

    if completion_tokens == 0 && !full.is_empty() {
        // Server did not return usage; approximate so throughput math still works.
        completion_tokens = (full.len() as f64 / 4.0).round() as u32;
    }

    Ok(Attempt {
        ok: true,
        status: 200,
        error: None,
        ttft_ms,
        total_ms: 0.0,
        prompt_tokens,
        completion_tokens,
        finish_reason,
        text: full,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt(id: &str, difficulty: &str) -> PromptItem {
        PromptItem {
            id: id.into(),
            title: id.into(),
            category: "test".into(),
            text: "hello".into(),
            difficulty: difficulty.into(),
            target_tokens: 16,
        }
    }

    // Frames below are verbatim from vLLM 0.23.1's /v1/chat/completions stream.
    #[test]
    fn decodes_a_content_delta() {
        let line = r#"data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]}"#;
        let f = parse_sse_line(line).expect("frame");
        assert_eq!(f.content.as_deref(), Some("OK"));
        assert!(!f.done);
        assert_eq!(f.finish_reason, None);
    }

    #[test]
    fn decodes_finish_reason_without_content() {
        let line = r#"data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop","stop_reason":106}]}"#;
        let f = parse_sse_line(line).expect("frame");
        assert_eq!(f.finish_reason.as_deref(), Some("stop"));
        assert_eq!(f.content, None);
    }

    #[test]
    fn decodes_trailing_usage_frame() {
        let line = r#"data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":15,"total_tokens":17,"completion_tokens":2}}"#;
        let f = parse_sse_line(line).expect("frame");
        assert_eq!(f.prompt_tokens, Some(15));
        assert_eq!(f.completion_tokens, Some(2));
    }

    #[test]
    fn recognises_done_sentinel() {
        assert!(parse_sse_line("data: [DONE]").expect("frame").done);
    }

    #[test]
    fn ignores_non_data_lines() {
        assert_eq!(parse_sse_line(""), None);
        assert_eq!(parse_sse_line("\n"), None);
        assert_eq!(parse_sse_line(": keepalive"), None);
        assert_eq!(parse_sse_line("event: message"), None);
        // Malformed JSON must be skipped, not abort the stream.
        assert_eq!(parse_sse_line("data: {not json"), None);
    }

    #[test]
    fn null_usage_is_not_read_as_zero() {
        // Mid-stream frames carry usage:null; treating that as 0 would wipe the
        // real counts that arrive in the final frame.
        let line = r#"data: {"choices":[{"index":0,"delta":{"content":"x"}}],"usage":null}"#;
        let f = parse_sse_line(line).expect("frame");
        assert_eq!(f.prompt_tokens, None);
        assert_eq!(f.completion_tokens, None);
    }

    #[test]
    fn zero_weight_difficulty_is_never_selected() {
        let pools = Pools::build(vec![
            prompt("e1", "easy"),
            prompt("m1", "medium"),
            prompt("h1", "hard"),
        ]);
        let mix = Mix { easy: 0, medium: 1, hard: 0 };
        for _ in 0..200 {
            let picked = pools.pick(&mix, false).expect("a prompt");
            assert_eq!(picked.difficulty, "medium");
        }
    }

    #[test]
    fn sequential_selection_walks_the_pool() {
        let pools = Pools::build(vec![
            prompt("e1", "easy"),
            prompt("e2", "easy"),
            prompt("e3", "easy"),
        ]);
        let mix = Mix { easy: 1, medium: 0, hard: 0 };
        let ids: Vec<String> = (0..3)
            .map(|_| pools.pick(&mix, true).unwrap().id.clone())
            .collect();
        assert_eq!(ids, vec!["e1", "e2", "e3"]);
        // Cursor wraps rather than running off the end.
        assert_eq!(pools.pick(&mix, true).unwrap().id, "e1");
    }

    #[test]
    fn empty_pool_yields_nothing() {
        let pools = Pools::build(vec![]);
        assert!(pools.pick(&Mix::default(), false).is_none());
    }

    #[test]
    fn unknown_difficulty_falls_back_to_medium() {
        let pools = Pools::build(vec![prompt("x1", "expert")]);
        assert_eq!(pools.medium.len(), 1);
    }
}
