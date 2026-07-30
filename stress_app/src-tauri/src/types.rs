use serde::{Deserialize, Serialize};

/// One prompt from the corpus. The frontend owns the corpus (it is bundled as
/// JSON by Vite) and hands the selected pool over when a run starts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptItem {
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub category: String,
    pub text: String,
    pub difficulty: String,
    #[serde(default)]
    pub target_tokens: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Mix {
    #[serde(default = "one")]
    pub easy: u32,
    #[serde(default = "one")]
    pub medium: u32,
    #[serde(default = "one")]
    pub hard: u32,
}

impl Default for Mix {
    fn default() -> Self {
        Mix { easy: 1, medium: 1, hard: 1 }
    }
}

fn one() -> u32 {
    1
}

/// Everything the settings modal can drive. Every field carries a default so
/// the frontend can send a partial config without the command rejecting it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunConfig {
    pub base_url: String,
    pub model: String,

    #[serde(default)]
    pub api_key: String,

    // ---- load shape ----
    #[serde(default = "d_concurrency")]
    pub concurrency: usize,
    /// "count" | "duration" | "infinite"
    #[serde(default = "d_mode")]
    pub mode: String,
    #[serde(default = "d_total")]
    pub total_requests: usize,
    #[serde(default = "d_duration")]
    pub duration_secs: u64,
    #[serde(default)]
    pub ramp_up_secs: u64,
    #[serde(default)]
    pub think_time_ms: u64,
    #[serde(default = "d_timeout")]
    pub request_timeout_ms: u64,
    #[serde(default)]
    pub max_retries: u32,
    #[serde(default)]
    pub stop_on_error: bool,

    // ---- sampling ----
    #[serde(default = "d_true")]
    pub stream: bool,
    #[serde(default = "d_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "d_temperature")]
    pub temperature: f32,
    #[serde(default = "d_top_p")]
    pub top_p: f32,
    /// <= 0 means "leave unset"
    #[serde(default)]
    pub top_k: i32,
    /// <= 0 means "leave unset"
    #[serde(default)]
    pub min_p: f32,
    #[serde(default)]
    pub presence_penalty: f32,
    #[serde(default)]
    pub frequency_penalty: f32,
    /// <= 0 means "leave unset"
    #[serde(default)]
    pub repetition_penalty: f32,
    /// < 0 means "leave unset"
    #[serde(default = "d_seed")]
    pub seed: i64,
    /// vLLM extension: keep generating to max_tokens even past EOS. Pins the
    /// output length so decode throughput is measured against a fixed budget.
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default)]
    pub system_prompt: String,

    // ---- prompt selection ----
    #[serde(default)]
    pub mix: Mix,
    /// "sequential" | "random"
    #[serde(default = "d_selection")]
    pub selection: String,
    /// Prepend a unique nonce to every prompt so vLLM's prefix cache cannot
    /// serve a warm prefix. Without this, repeated prompts inflate throughput.
    #[serde(default)]
    pub prefix_cache_bust: bool,

    // ---- telemetry ----
    #[serde(default = "d_metrics_poll")]
    pub metrics_poll_ms: u64,
    #[serde(default = "d_flush")]
    pub stream_flush_ms: u64,
}

fn d_concurrency() -> usize {
    8
}
fn d_mode() -> String {
    "count".into()
}
fn d_total() -> usize {
    100
}
fn d_duration() -> u64 {
    60
}
fn d_timeout() -> u64 {
    120_000
}
fn d_true() -> bool {
    true
}
fn d_max_tokens() -> u32 {
    512
}
fn d_temperature() -> f32 {
    0.7
}
fn d_top_p() -> f32 {
    0.95
}
fn d_seed() -> i64 {
    -1
}
fn d_selection() -> String {
    "random".into()
}
fn d_metrics_poll() -> u64 {
    1000
}
fn d_flush() -> u64 {
    60
}

// ---------------------------------------------------------------------------
// Event payloads
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestStart {
    pub id: String,
    pub seq: u64,
    pub worker: usize,
    pub prompt_id: String,
    pub title: String,
    pub category: String,
    pub difficulty: String,
    pub prompt_chars: usize,
    pub attempt: u32,
    pub started_at: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Delta {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestEnd {
    pub id: String,
    pub seq: u64,
    pub worker: usize,
    pub prompt_id: String,
    pub difficulty: String,
    pub ok: bool,
    pub status: u16,
    pub error: Option<String>,
    /// Time to first token, ms. None when the request failed before any token.
    pub ttft_ms: Option<f64>,
    pub total_ms: f64,
    /// Decode-only throughput: completion tokens / (total - ttft).
    pub output_tps: Option<f64>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: Option<String>,
    pub text: String,
    pub finished_at: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunState {
    /// "running" | "stopping" | "done" | "cancelled" | "error"
    pub state: String,
    pub completed: u64,
    pub failed: u64,
    pub in_flight: u64,
    pub dispatched: u64,
    pub target: Option<u64>,
    pub elapsed_ms: f64,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerMetrics {
    pub ok: bool,
    pub at: u64,
    pub error: Option<String>,
    pub num_running: Option<f64>,
    pub num_waiting: Option<f64>,
    pub kv_cache_usage: Option<f64>,
    pub prefix_hit_rate: Option<f64>,
    pub prompt_tokens_total: Option<f64>,
    pub generation_tokens_total: Option<f64>,
    pub preemptions_total: Option<f64>,
    pub requests_success_total: Option<f64>,
    pub gpu_cache_hit_tokens: Option<f64>,
    pub gpu_cache_query_tokens: Option<f64>,
}
