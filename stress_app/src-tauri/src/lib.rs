mod runner;
mod telemetry;
mod types;

use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Manager, State};
use types::{PromptItem, RunConfig, ServerMetrics};

#[derive(Default)]
struct AppState {
    cancel: Mutex<Option<Arc<AtomicBool>>>,
    running: AtomicBool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelInfo {
    id: String,
    owned_by: String,
    max_model_len: Option<u64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Probe {
    reachable: bool,
    error: Option<String>,
    version: Option<String>,
    models: Vec<ModelInfo>,
    metrics_available: bool,
    latency_ms: f64,
    /// True when the engine reports speculative-decoding counters. Such servers
    /// reject `min_p` and `logit_bias` outright, so the UI disables those.
    spec_decoding: bool,
}

fn http() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(8))
        .build()
        .map_err(|e| e.to_string())
}

/// Ask the server what it is running: model list, engine version, and whether
/// the Prometheus endpoint is exposed.
#[tauri::command]
async fn probe_server(base_url: String, api_key: String) -> Probe {
    let started = std::time::Instant::now();
    let client = match http() {
        Ok(c) => c,
        Err(e) => {
            return Probe {
                reachable: false,
                error: Some(e),
                version: None,
                models: vec![],
                metrics_available: false,
                spec_decoding: false,
                latency_ms: 0.0,
            }
        }
    };

    let base = base_url.trim_end_matches('/').to_string();
    let mut req = client.get(format!("{base}/models")).timeout(Duration::from_secs(10));
    if !api_key.trim().is_empty() {
        req = req.bearer_auth(api_key.trim());
    }

    let models_res = req.send().await;
    let (reachable, error, models) = match models_res {
        Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
            Ok(v) => {
                let list = v["data"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .map(|m| ModelInfo {
                                id: m["id"].as_str().unwrap_or_default().to_string(),
                                owned_by: m["owned_by"].as_str().unwrap_or_default().to_string(),
                                max_model_len: m["max_model_len"].as_u64(),
                            })
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                (true, None, list)
            }
            Err(e) => (false, Some(format!("bad /models payload: {e}")), vec![]),
        },
        Ok(r) => (false, Some(format!("HTTP {}", r.status().as_u16())), vec![]),
        Err(e) => (false, Some(e.to_string()), vec![]),
    };

    // The engine version lives on the server root, alongside /metrics.
    let root = base.strip_suffix("/v1").unwrap_or(&base).trim_end_matches('/');
    let version = client
        .get(format!("{root}/version"))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .ok()
        .filter(|r| r.status().is_success());
    let version = match version {
        Some(r) => r
            .json::<serde_json::Value>()
            .await
            .ok()
            .and_then(|v| v["version"].as_str().map(String::from)),
        None => None,
    };

    let snapshot = telemetry::scrape(&client, &base).await;

    Probe {
        reachable,
        error,
        version,
        models,
        metrics_available: snapshot.ok,
        spec_decoding: snapshot.spec_draft_tokens.unwrap_or(0.0) > 0.0,
        latency_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

#[tauri::command]
async fn fetch_metrics(base_url: String) -> ServerMetrics {
    match http() {
        Ok(client) => telemetry::scrape(&client, &base_url).await,
        Err(e) => telemetry::failed(telemetry::now_ms(), e),
    }
}

#[tauri::command]
fn start_run(
    app: AppHandle,
    state: State<'_, AppState>,
    config: RunConfig,
    prompts: Vec<PromptItem>,
) -> Result<(), String> {
    if state.running.swap(true, Ordering::SeqCst) {
        return Err("A run is already in progress.".into());
    }
    if prompts.is_empty() {
        state.running.store(false, Ordering::SeqCst);
        return Err("No prompts selected.".into());
    }

    let cancel = Arc::new(AtomicBool::new(false));
    *state.cancel.lock().map_err(|e| e.to_string())? = Some(cancel.clone());

    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        runner::run(handle.clone(), config, prompts, cancel).await;
        if let Some(st) = handle.try_state::<AppState>() {
            st.running.store(false, Ordering::SeqCst);
            if let Ok(mut guard) = st.cancel.lock() {
                *guard = None;
            }
        }
    });

    Ok(())
}

#[tauri::command]
fn stop_run(state: State<'_, AppState>) -> Result<(), String> {
    if let Ok(guard) = state.cancel.lock() {
        if let Some(flag) = guard.as_ref() {
            flag.store(true, Ordering::SeqCst);
        }
    }
    Ok(())
}

#[tauri::command]
fn is_running(state: State<'_, AppState>) -> bool {
    state.running.load(Ordering::SeqCst)
}

/// Write an export next to the user's other downloads and hand back the path.
/// Keeps the frontend free of filesystem permissions for a one-shot action.
#[tauri::command]
fn export_file(app: AppHandle, filename: String, contents: String) -> Result<String, String> {
    let dir = app
        .path()
        .download_dir()
        .or_else(|_| app.path().home_dir())
        .map_err(|e| e.to_string())?;

    // Reject path traversal — the frontend supplies this name.
    let safe = filename
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .collect::<String>();
    if safe.is_empty() {
        return Err("Invalid filename.".into());
    }

    let path = dir.join(safe);
    std::fs::write(&path, contents).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            probe_server,
            fetch_metrics,
            start_run,
            stop_run,
            is_running,
            export_file
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
