use crate::types::ServerMetrics;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Emitter};

/// vLLM exposes Prometheus text on the server root, not under `/v1`.
pub fn metrics_url(base_url: &str) -> String {
    let root = base_url.trim_end_matches('/');
    let root = root.strip_suffix("/v1").unwrap_or(root);
    format!("{}/metrics", root.trim_end_matches('/'))
}

/// Minimal Prometheus text-format reader. Sums every label series under a
/// metric name, which is what we want for the counters here (they are split by
/// `model_name`, `engine`, and — on counters like `request_success_total` —
/// `finished_reason`).
fn parse_prometheus(body: &str) -> HashMap<String, f64> {
    let mut out: HashMap<String, f64> = HashMap::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((lhs, rhs)) = line.rsplit_once(' ') else {
            continue;
        };
        let Ok(value) = rhs.trim().parse::<f64>() else {
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        let name = lhs.split('{').next().unwrap_or(lhs).trim();
        if name.is_empty() {
            continue;
        }
        *out.entry(name.to_string()).or_insert(0.0) += value;
    }
    out
}

fn get(m: &HashMap<String, f64>, key: &str) -> Option<f64> {
    m.get(key).copied()
}

pub async fn scrape(client: &reqwest::Client, base_url: &str) -> ServerMetrics {
    let now = now_ms();
    let url = metrics_url(base_url);
    let res = client
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await;

    let body = match res {
        Ok(r) if r.status().is_success() => match r.text().await {
            Ok(t) => t,
            Err(e) => return failed(now, format!("read body: {e}")),
        },
        Ok(r) => return failed(now, format!("HTTP {}", r.status().as_u16())),
        Err(e) => return failed(now, e.to_string()),
    };

    let m = parse_prometheus(&body);

    // Prefix-cache hit rate is exposed as two token counters, not a ratio.
    let hits = get(&m, "vllm:prefix_cache_hits_total")
        .or_else(|| get(&m, "vllm:gpu_prefix_cache_hits_total"));
    let queries = get(&m, "vllm:prefix_cache_queries_total")
        .or_else(|| get(&m, "vllm:gpu_prefix_cache_queries_total"));
    let hit_rate = match (hits, queries) {
        (Some(h), Some(q)) if q > 0.0 => Some(h / q),
        _ => None,
    };

    ServerMetrics {
        ok: true,
        at: now,
        error: None,
        num_running: get(&m, "vllm:num_requests_running"),
        num_waiting: get(&m, "vllm:num_requests_waiting"),
        kv_cache_usage: get(&m, "vllm:kv_cache_usage_perc")
            .or_else(|| get(&m, "vllm:gpu_cache_usage_perc")),
        prefix_hit_rate: hit_rate,
        prompt_tokens_total: get(&m, "vllm:prompt_tokens_total"),
        generation_tokens_total: get(&m, "vllm:generation_tokens_total"),
        preemptions_total: get(&m, "vllm:num_preemptions_total"),
        requests_success_total: get(&m, "vllm:request_success_total"),
        gpu_cache_hit_tokens: hits,
        gpu_cache_query_tokens: queries,
    }
}

fn failed(at: u64, msg: String) -> ServerMetrics {
    ServerMetrics {
        ok: false,
        at,
        error: Some(msg),
        num_running: None,
        num_waiting: None,
        kv_cache_usage: None,
        prefix_hit_rate: None,
        prompt_tokens_total: None,
        generation_tokens_total: None,
        preemptions_total: None,
        requests_success_total: None,
        gpu_cache_hit_tokens: None,
        gpu_cache_query_tokens: None,
    }
}

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Background poller. Runs for the lifetime of a test run and emits one
/// `rig://server-metrics` event per tick.
pub async fn poll_loop(
    app: AppHandle,
    client: reqwest::Client,
    base_url: String,
    interval_ms: u64,
    cancel: Arc<AtomicBool>,
) {
    let interval_ms = interval_ms.max(200);
    while !cancel.load(Ordering::Relaxed) {
        let snapshot = scrape(&client, &base_url).await;
        let _ = app.emit("rig://server-metrics", &snapshot);
        tokio::time::sleep(Duration::from_millis(interval_ms)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_v1_suffix_for_the_metrics_root() {
        assert_eq!(
            metrics_url("http://10.150.0.30:1234/v1"),
            "http://10.150.0.30:1234/metrics"
        );
        assert_eq!(
            metrics_url("http://10.150.0.30:1234/v1/"),
            "http://10.150.0.30:1234/metrics"
        );
        assert_eq!(
            metrics_url("http://10.150.0.30:1234"),
            "http://10.150.0.30:1234/metrics"
        );
    }

    #[test]
    fn sums_series_across_label_sets() {
        // request_success_total is split by finished_reason; the dashboard wants
        // the total, so every label set for a name is added together.
        let body = "\
# HELP vllm:request_success_total Count of successful requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason=\"stop\",model_name=\"g\"} 12.0
vllm:request_success_total{finished_reason=\"length\",model_name=\"g\"} 5.0
vllm:num_requests_running{model_name=\"g\"} 3.0
";
        let m = parse_prometheus(body);
        assert_eq!(m.get("vllm:request_success_total"), Some(&17.0));
        assert_eq!(m.get("vllm:num_requests_running"), Some(&3.0));
    }

    #[test]
    fn skips_comments_blanks_and_non_numeric_values() {
        let body = "\
# HELP something A comment
# TYPE something gauge

vllm:kv_cache_usage_perc{model_name=\"g\"} 0.42
vllm:broken{model_name=\"g\"} NaN
garbage line without value
";
        let m = parse_prometheus(body);
        assert_eq!(m.get("vllm:kv_cache_usage_perc"), Some(&0.42));
        assert!(!m.contains_key("vllm:broken"), "NaN must not be recorded");
        assert_eq!(m.len(), 1);
    }
}
