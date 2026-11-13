//! First-run setup: find what is already available, pick the best of it, save the choices,
//! and say what is missing (with a preview of what a managed install would add).

use crate::config::Config;
use crate::endpoints::{DEFAULT_STT_MODEL, Endpoints};
use crate::hardware::{self, Hardware, Tier, gib};
use crate::latency::{self, Latency, ModelSpeed, Rating, Targets};
use crate::profile;
use crate::speech::find_in_path;
use anyhow::Result;
use std::fmt::Write as _;
use std::io::{BufRead, Write};
use std::time::Duration;

/// Places a Kokoro speech endpoint is commonly found, best guess first.
pub const KOKORO_CANDIDATES: [&str; 2] = [
    "http://127.0.0.1:8880/v1/audio/speech",
    "http://127.0.0.1:8080/v1/audio/speech",
];

/// What the probe found on this machine.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Findings {
    /// The model-list URL that was asked, for messages.
    pub models_url: String,
    /// Model ids served by the local server, or `None` if it could not be reached.
    pub models: Option<Vec<String>>,
    /// A Kokoro endpoint that answered with audio.
    pub kokoro_url: Option<String>,
    /// The system speech command (`say` or `espeak-ng`), if installed.
    pub system_voice: Option<&'static str>,
    /// A program that can play audio, if one is installed.
    pub player: Option<&'static str>,
    /// Whether the `claude` CLI is installed.
    pub claude: bool,
    /// This machine's memory, GPU and disk.
    pub hardware: Option<Hardware>,
    /// How fast the models answer here; `None` if not timed.
    pub latency: Option<Latency>,
}

/// Look around this machine: what is available, what the hardware is, and (unless `quick`) how
/// fast the models answer. Timing can take a minute if the models have to load first.
pub async fn probe(client: &reqwest::Client, endpoints: &Endpoints, quick: bool) -> Findings {
    // The configured speech endpoint first, then the usual places.
    let mut candidates = vec![endpoints.tts.clone()];
    for usual in KOKORO_CANDIDATES {
        if !candidates.iter().any(|c| c == usual) {
            candidates.push(usual.to_string());
        }
    }
    let mut found = probe_at(client, &endpoints.models, &candidates).await;
    found.hardware = tokio::task::spawn_blocking(hardware::detect).await.ok();
    if !quick && let Some(ids) = &found.models {
        let chat = pick_chat_model(ids);
        let smart = pick_smart_model(ids).filter(|smart| Some(smart) != chat.as_ref());
        let whisper = pick_whisper(ids);
        let targets = Targets {
            whisper: whisper.is_some(),
            stt_model: whisper.unwrap_or_default(),
            endpoints: endpoints.clone(),
            chat_model: chat.as_deref(),
            smart_model: smart.as_deref(),
            kokoro_url: found.kokoro_url.as_deref(),
        };
        found.latency = Some(latency::measure(client, &targets).await);
    }
    found
}

/// Like [`probe`], with the model-list URL and Kokoro candidates given explicitly.
pub async fn probe_at(
    client: &reqwest::Client,
    models_url: &str,
    kokoro_candidates: &[String],
) -> Findings {
    let mut kokoro_url = None;
    for url in kokoro_candidates {
        if kokoro_answers(client, url).await {
            kokoro_url = Some(url.clone());
            break;
        }
    }
    let system_command = if cfg!(target_os = "macos") {
        "say"
    } else {
        "espeak-ng"
    };
    Findings {
        models_url: models_url.to_string(),
        models: list_models(client, models_url).await,
        kokoro_url,
        system_voice: find_in_path(system_command).map(|_| system_command),
        player: ["afplay", "paplay", "pw-play", "aplay"]
            .into_iter()
            .find(|name| find_in_path(name).is_some()),
        claude: find_in_path("claude").is_some(),
        hardware: None,
        latency: None,
    }
}

async fn list_models(client: &reqwest::Client, models_url: &str) -> Option<Vec<String>> {
    let body: serde_json::Value = client
        .get(models_url)
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .await
        .ok()?;
    let ids = body
        .get("data")?
        .as_array()?
        .iter()
        .filter_map(|m| m.get("id")?.as_str().map(ToString::to_string))
        .collect();
    Some(ids)
}

/// A Kokoro endpoint counts only if it really returns audio for a tiny request.
async fn kokoro_answers(client: &reqwest::Client, url: &str) -> bool {
    let request = serde_json::json!({
        "model": "kokoro",
        "input": "Ready.",
        "voice": "af_heart",
        "response_format": "wav",
    });
    let Ok(response) = client
        .post(url)
        .timeout(Duration::from_secs(10))
        .json(&request)
        .send()
        .await
    else {
        return false;
    };
    match response.error_for_status() {
        Ok(ok) => ok.bytes().await.is_ok_and(|b| !b.is_empty()),
        Err(_) => false,
    }
}

const REASONING: [&str; 4] = ["reason", "think", "r1", "-o1"];

fn is_reasoning(id: &str) -> bool {
    let id = id.to_lowercase();
    REASONING.iter().any(|word| id.contains(word))
}

/// How suitable a model is for a voice assistant, or `None` if it is not a chat model at all.
/// Reasoning models are slow to start answering, so they rank below plain ones.
fn chat_score(id: &str) -> Option<i32> {
    const NOT_CHAT: [&str; 8] = [
        "whisper", "embed", "rerank", "tts", "kokoro", "-vl", "vision", "clip",
    ];
    let lowered = id.to_lowercase();
    let id = lowered.as_str();
    if NOT_CHAT.iter().any(|word| id.contains(word)) {
        return None;
    }
    let mut score = 100;
    if is_reasoning(id) {
        score -= 50;
    }
    if id.contains("coder") {
        score -= 5;
    }
    Some(score)
}

/// The best chat model among `ids` for talking, if any.
///
/// # Examples
///
/// ```
/// use voice_notes::setup::pick_chat_model;
///
/// let ids = ["nomic-embed", "qwen-reasoner", "qwen3-coder-30b", "whisper"].map(String::from);
/// assert_eq!(pick_chat_model(&ids).as_deref(), Some("qwen3-coder-30b"));
/// ```
#[must_use]
pub fn pick_chat_model(ids: &[String]) -> Option<String> {
    ids.iter()
        .filter_map(|id| chat_score(id).map(|score| (score, id)))
        .max_by(|(a, a_id), (b, b_id)| a.cmp(b).then_with(|| b_id.cmp(a_id)))
        .map(|(_, id)| id.clone())
}

/// The reasoning model among `ids` for the `smart` and `think` profiles, if any.
///
/// # Examples
///
/// ```
/// use voice_notes::setup::pick_smart_model;
///
/// let ids = ["qwen-reasoner", "qwen3-coder-30b", "whisper"].map(String::from);
/// assert_eq!(pick_smart_model(&ids).as_deref(), Some("qwen-reasoner"));
/// assert_eq!(pick_smart_model(&["qwen3-coder-30b".to_string()]), None);
/// ```
#[must_use]
pub fn pick_smart_model(ids: &[String]) -> Option<String> {
    ids.iter()
        .filter(|id| chat_score(id).is_some() && is_reasoning(id))
        .min()
        .cloned()
}

/// The Whisper model among `ids`, if any.
#[must_use]
pub fn pick_whisper(ids: &[String]) -> Option<String> {
    ids.iter()
        .find(|id| id.to_lowercase().contains("whisper"))
        .cloned()
}

/// A job a model or service does for the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Speech to text.
    Stt,
    /// Answering in talk mode (and cleaning notes).
    Chat,
    /// Speaking replies.
    Voice,
}

impl Role {
    fn label(self) -> &'static str {
        match self {
            Self::Stt => "Speech-to-text",
            Self::Chat => "Chat",
            Self::Voice => "Voice",
        }
    }
}

/// How well a role is covered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// The best option is available.
    Ready(String),
    /// A working but lesser option is available.
    Fallback(String),
    /// Nothing is available.
    Missing,
}

/// One line of the plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Item {
    pub role: Role,
    pub status: Status,
}

/// What to use for each role, and the config that makes the app use it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub items: Vec<Item>,
    pub config: Config,
}

/// Pick the best available option for every role.
#[must_use]
pub fn plan(found: &Findings) -> Plan {
    let ids: &[String] = found.models.as_deref().unwrap_or_default();
    let mut config = Config::default();

    let stt = match pick_whisper(ids) {
        Some(id) => {
            if id != DEFAULT_STT_MODEL {
                // The server names it differently, so remember the name to ask for.
                config.stt_model = Some(id.clone());
            }
            Status::Ready(format!("{id} (on the server)"))
        }
        None => Status::Missing,
    };

    let chat = if let Some(model) = pick_chat_model(ids) {
        config.backend = Some("local".to_string());
        // `smart` and `think` only make sense with a different, reasoning model.
        config.smart_model = pick_smart_model(ids).filter(|smart| *smart != model);
        config.chat_model = Some(model.clone());
        Status::Ready(model)
    } else if found.claude {
        Status::Fallback("only the claude CLI (opt in with --backend claude)".to_string())
    } else {
        Status::Missing
    };

    let voice = if let Some(url) = &found.kokoro_url {
        config.tts = Some("kokoro".to_string());
        config.tts_url = Some(url.clone());
        Status::Ready(format!("Kokoro at {url}"))
    } else if let Some(command) = found.system_voice {
        config.tts = Some("system".to_string());
        Status::Fallback(format!(
            "the system voice ({command}), no Kokoro server found"
        ))
    } else {
        Status::Missing
    };

    Plan {
        items: vec![
            Item {
                role: Role::Stt,
                status: stt,
            },
            Item {
                role: Role::Chat,
                status: chat,
            },
            Item {
                role: Role::Voice,
                status: voice,
            },
        ],
        config,
    }
}

/// A model or service a managed install would provide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Component {
    pub name: &'static str,
    pub role: Role,
    /// Download size: measured from the real file unless `approximate`.
    pub bytes: u64,
    pub approximate: bool,
    /// An extra that is not needed to talk (the reasoning model for `smart` and `think`).
    pub optional: bool,
    pub source: &'static str,
}

const fn component(
    name: &'static str,
    role: Role,
    bytes: u64,
    approximate: bool,
    optional: bool,
    source: &'static str,
) -> Component {
    Component {
        name,
        role,
        bytes,
        approximate,
        optional,
        source,
    }
}

const WHISPER_SMALL: Component = component(
    "Whisper small",
    Role::Stt,
    488_636_416,
    true,
    false,
    "whisper.cpp model ggml-small.bin",
);
const WHISPER_TURBO: Component = component(
    "Whisper large-v3-turbo (q5_0)",
    Role::Stt,
    574_041_195,
    false,
    false,
    "whisper.cpp model ggml-large-v3-turbo-q5_0.bin",
);
const CHAT_4B: Component = component(
    "Qwen3-4B (Q4_K_M)",
    Role::Chat,
    2_500_000_000,
    true,
    false,
    "GGUF, served by llama.cpp",
);
const CHAT_8B: Component = component(
    "Qwen3-8B (Q4_K_M)",
    Role::Chat,
    5_000_000_000,
    true,
    false,
    "GGUF, served by llama.cpp",
);
const CHAT_14B: Component = component(
    "Qwen3-14B (Q4_K_M)",
    Role::Chat,
    9_000_000_000,
    true,
    false,
    "GGUF, served by llama.cpp",
);
const CHAT_30B: Component = component(
    "Qwen3-Coder-30B-A3B-Instruct (Q4_K_XL)",
    Role::Chat,
    17_665_334_432,
    false,
    false,
    "Unsloth GGUF, served by llama.cpp",
);
const REASONER: Component = component(
    "Qwen3.8-27B (Q4_K_M), for smart and think",
    Role::Chat,
    16_564_259_744,
    false,
    true,
    "Unsloth GGUF, served by llama.cpp",
);
const KOKORO: Component = component(
    "Kokoro v1.0 (82M)",
    Role::Voice,
    327_212_226,
    false,
    false,
    "hexgrad/Kokoro-82M, served by Kokoro-FastAPI",
);

/// The models suggested for a machine of this size. Sizes marked approximate are estimates;
/// the others were measured from the real files.
#[must_use]
pub fn catalogue(tier: Tier) -> Vec<Component> {
    match tier {
        Tier::Minimal => vec![WHISPER_SMALL, CHAT_4B, KOKORO],
        Tier::Light => vec![WHISPER_TURBO, CHAT_8B, KOKORO],
        Tier::Standard => vec![WHISPER_TURBO, CHAT_14B, KOKORO],
        Tier::Power => vec![WHISPER_TURBO, CHAT_30B, KOKORO, REASONER],
    }
}

/// The next smaller size class.
fn smaller(tier: Tier) -> Tier {
    match tier {
        Tier::Power => Tier::Standard,
        Tier::Standard => Tier::Light,
        Tier::Light | Tier::Minimal => Tier::Minimal,
    }
}

/// The size in billions of parameters written in a model name, e.g. `30` for
/// `qwen3-coder-30b` or `Qwen3-8B (Q4_K_M)`. Names without one give `None`.
///
/// # Examples
///
/// ```
/// use voice_notes::setup::parameters_billions;
///
/// assert_eq!(parameters_billions("qwen3-coder-30b"), Some(30.0));
/// assert_eq!(parameters_billions("Qwen3-8B (Q4_K_M)"), Some(8.0));
/// assert_eq!(parameters_billions("llama-3.2-1.5b-instruct"), Some(1.5));
/// assert_eq!(parameters_billions("whisper"), None);
/// ```
#[must_use]
pub fn parameters_billions(name: &str) -> Option<f64> {
    let lower = name.to_lowercase();
    let bytes = lower.as_bytes();
    let mut best = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            let after_b = bytes.get(i) == Some(&b'b');
            let boundary = bytes.get(i + 1).is_none_or(|c| !c.is_ascii_alphanumeric());
            let preceded = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
            if after_b
                && boundary
                && preceded
                && let Ok(value) = lower[start..i].parse::<f64>()
            {
                best = best.or(Some(value));
            }
        } else {
            i += 1;
        }
    }
    best
}

/// When the installed chat model is too slow: the model that fits this machine, or, if the
/// installed one is already that small, the next size down.
fn faster_chat(tier: Tier, installed: Option<&str>) -> Option<Component> {
    let chat = |t: Tier| {
        catalogue(t)
            .into_iter()
            .find(|c| c.role == Role::Chat && !c.optional)
    };
    let fitting = chat(tier)?;
    let installed_size = installed.and_then(parameters_billions);
    let fitting_size = parameters_billions(fitting.name);
    match (installed_size, fitting_size) {
        (Some(have), Some(fit)) if have <= fit => chat(smaller(tier)),
        _ => Some(fitting),
    }
}

/// What a managed install would add on this machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Preview {
    /// The size class the suggestions are for.
    pub tier: Tier,
    /// False when the hardware could not be read and the middle size is assumed.
    pub tier_known: bool,
    /// Models for roles that are missing, or for a chat model that is too slow here.
    pub required: Vec<Component>,
    /// Optional extras, such as the reasoning model.
    pub extras: Vec<Component>,
}

/// Work out what a managed install would add: for each role that is missing, the model that
/// suits this machine; for a chat model that is too slow, one size smaller.
#[must_use]
pub fn managed_preview(
    plan: &Plan,
    hardware: Option<&Hardware>,
    timing: Option<&Latency>,
) -> Preview {
    let known = hardware.and_then(Hardware::tier);
    let tier = known.unwrap_or(Tier::Light);
    let chat_too_slow = timing
        .and_then(|l| l.chat.as_ref())
        .is_some_and(|c| latency::rate_chat(c.warm) == Rating::Slow);
    let missing = |role: Role| {
        plan.items
            .iter()
            .any(|i| i.role == role && !matches!(i.status, Status::Ready(_)))
    };

    let (extras, wanted): (Vec<Component>, Vec<Component>) =
        catalogue(tier).into_iter().partition(|c| c.optional);
    let mut required: Vec<Component> = wanted.into_iter().filter(|c| missing(c.role)).collect();
    if chat_too_slow && !missing(Role::Chat) {
        required.extend(faster_chat(tier, plan.config.chat_model.as_deref()));
    }
    Preview {
        tier,
        tier_known: known.is_some(),
        required,
        extras: if plan.config.smart_model.is_none() {
            extras
        } else {
            Vec::new()
        },
    }
}

/// A size in decimal units, e.g. `17.7 GB` or `327 MB`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn human_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}

fn size_text(c: &Component) -> String {
    let size = human_size(c.bytes);
    if c.approximate {
        format!("~{size}")
    } else {
        size
    }
}

fn manual_hint(role: Role) -> &'static str {
    match role {
        Role::Stt | Role::Chat => {
            "Serve Whisper and a chat model through an OpenAI-compatible server such as \
             llama-swap on 127.0.0.1:8080."
        }
        Role::Voice => {
            "Run Kokoro-FastAPI (https://github.com/remsky/Kokoro-FastAPI), by default on \
             port 8880."
        }
    }
}

fn seconds(d: std::time::Duration) -> String {
    format!("{:.1} s", d.as_secs_f64())
}

fn mark(rating: Rating) -> &'static str {
    match rating {
        Rating::Snappy => "ok",
        Rating::Fine => "~~",
        Rating::Slow => "!!",
    }
}

/// One report line for a chat model's timing.
fn model_speed_line(label: &str, speed: &ModelSpeed, note: &str) -> String {
    let rating = latency::rate_chat(speed.warm);
    let mut line = format!(
        "  [{}] {label:<15} {}: first sentence in {}",
        mark(rating),
        speed.model,
        seconds(speed.warm)
    );
    if speed.loaded_on_demand() {
        let _ = write!(
            line,
            " (the first reply took {} while the model loaded)",
            seconds(speed.cold)
        );
    }
    if rating == Rating::Slow {
        line.push_str(note);
    }
    line
}

/// The "Speed" section of the report.
fn speed_lines(timing: &Latency) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(d) = timing.stt {
        lines.push(format!(
            "  [{}] {:<15} transcribes a 3 s clip in {}",
            mark(latency::rate_stt(d)),
            "Speech-to-text",
            seconds(d)
        ));
    }
    if let Some(chat) = &timing.chat {
        lines.push(model_speed_line("Chat", chat, ": too slow for talking"));
    }
    if let Some(smart) = &timing.smart {
        lines.push(model_speed_line(
            "Smart",
            smart,
            ": fine for `think`, slow for conversation",
        ));
    }
    if let Some(d) = timing.voice {
        lines.push(format!(
            "  [{}] {:<15} Kokoro speaks one sentence in {}",
            mark(latency::rate_voice(d)),
            "Voice",
            seconds(d)
        ));
    }
    if lines.is_empty() {
        lines.push("  (nothing could be timed)".to_string());
    }
    lines
}

/// The "Your machine" section of the report.
fn machine_lines(hw: &Hardware) -> Vec<String> {
    let mut lines = vec![format!("  {}", hw.summary())];
    match (hw.tier(), hw.model_memory()) {
        (Some(tier), Some(memory)) => lines.push(format!(
            "  Suggested size: {} (about {} for models)",
            tier.label(),
            gib(memory)
        )),
        _ => lines.push("  Memory not detected, so no size is suggested.".to_string()),
    }
    if hw.cpu_only() {
        lines.push("  No GPU found: replies will be slower, so prefer small models.".to_string());
    }
    lines
}

/// The setup report as text.
#[must_use]
pub fn render(found: &Findings, plan: &Plan) -> String {
    let mut out = String::from("Voice Notes setup\n\nWhat I found\n");
    for item in &plan.items {
        let (mark, text) = match &item.status {
            Status::Ready(what) => ("ok", what.clone()),
            Status::Fallback(what) => ("~~", what.clone()),
            Status::Missing => ("!!", "nothing found".to_string()),
        };
        let _ = writeln!(out, "  [{mark}] {:<15} {text}", item.role.label());
    }
    let profiles = profile::available_with(&plan.config, found.claude);
    let names: Vec<&str> = profiles.iter().map(|p| p.name.as_str()).collect();
    let _ = writeln!(out, "  [ok] {:<15} {}", "Profiles", names.join(", "));
    let (mark, what, text) = match (found.models.is_some(), found.player) {
        (false, _) => {
            let url = if found.models_url.is_empty() {
                Endpoints::default().models
            } else {
                found.models_url.clone()
            };
            ("!!", "Server", format!("no server answered at {url}"))
        }
        (true, None) => ("!!", "Playback", "no audio player found".to_string()),
        (true, Some(player)) => ("ok", "Playback", player.to_string()),
    };
    let _ = writeln!(out, "  [{mark}] {what:<15} {text}");
    if found.claude {
        let _ = writeln!(
            out,
            "  [ok] {:<15} claude CLI available (use --backend claude; what you say is then sent to Anthropic)",
            "Optional"
        );
    }

    if let Some(hw) = &found.hardware {
        out.push_str("\nYour machine\n");
        for line in machine_lines(hw) {
            let _ = writeln!(out, "{line}");
        }
    }
    if let Some(timing) = &found.latency {
        out.push_str("\nSpeed\n");
        for line in speed_lines(timing) {
            let _ = writeln!(out, "{line}");
        }
    }

    let preview = managed_preview(plan, found.hardware.as_ref(), found.latency.as_ref());
    let mut hints: Vec<String> = plan
        .items
        .iter()
        .filter(|i| !matches!(i.status, Status::Ready(_)))
        .map(|i| format!("  - {}: {}", i.role.label(), manual_hint(i.role)))
        .collect();
    if let Some(chat) = found.latency.as_ref().and_then(|l| l.chat.as_ref())
        && latency::rate_chat(chat.warm) == Rating::Slow
    {
        hints.push(format!(
            "  - Chat: replies start slowly here ({}). A smaller model will feel much better.",
            seconds(chat.warm)
        ));
    }

    if hints.is_empty() && preview.required.is_empty() {
        out.push_str("\nEverything needed is available.\n");
    } else {
        out.push_str("\nWhat is missing or could be better\n");
        for hint in &hints {
            let _ = writeln!(out, "{hint}");
        }
        if !preview.required.is_empty() {
            push_preview(&mut out, &preview, found.hardware.as_ref());
        }
    }
    if !preview.extras.is_empty() {
        out.push_str("\nOptional, for the smart and think profiles (preview)\n");
        for c in &preview.extras {
            let _ = writeln!(out, "  + {:<42} {:>9}   {}", c.name, size_text(c), c.source);
        }
    }
    out
}

/// The managed-install preview, sized for this machine.
fn push_preview(out: &mut String, preview: &Preview, hardware: Option<&Hardware>) {
    out.push_str("\nWhat a managed install would add (preview, not available yet)\n");
    if preview.tier_known {
        let _ = writeln!(
            out,
            "  Chosen for your machine: {} size.",
            preview.tier.label()
        );
    } else {
        let _ = writeln!(
            out,
            "  Machine not detected, so assuming the {} size.",
            preview.tier.label()
        );
    }
    let mut total = 0;
    for c in &preview.required {
        total += c.bytes;
        let _ = writeln!(out, "  + {:<42} {:>9}   {}", c.name, size_text(c), c.source);
    }
    let approximate = preview.required.iter().any(|c| c.approximate);
    let _ = writeln!(
        out,
        "  Total download: {}{}. Sizes are measured from real files where possible; \"~\" marks estimates.",
        if approximate { "about " } else { "" },
        human_size(total)
    );
    if let Some(free) = hardware.and_then(|h| h.disk_free_bytes)
        && total.saturating_mul(11) / 10 > free
    {
        let _ = writeln!(
            out,
            "  [!!] Not enough free disk space: {} free.",
            gib(free)
        );
    }
}

/// Combine a saved config with newly chosen values; the new values win where present.
#[must_use]
pub fn overlay(base: Config, new: &Config) -> Config {
    Config {
        backend: new.backend.clone().or(base.backend),
        chat_model: new.chat_model.clone().or(base.chat_model),
        tts: new.tts.clone().or(base.tts),
        tts_url: new.tts_url.clone().or(base.tts_url),
        tts_model: new.tts_model.clone().or(base.tts_model),
        tts_voice: new.tts_voice.clone().or(base.tts_voice),
        voice: new.voice.clone().or(base.voice),
        language: new.language.clone().or(base.language),
        server: new.server.clone().or(base.server),
        stt_url: new.stt_url.clone().or(base.stt_url),
        chat_url: new.chat_url.clone().or(base.chat_url),
        models_url: new.models_url.clone().or(base.models_url),
        stt_model: new.stt_model.clone().or(base.stt_model),
        profile: new.profile.clone().or(base.profile),
        smart_model: new.smart_model.clone().or(base.smart_model),
        profiles: {
            let mut profiles = base.profiles;
            profiles.extend(new.profiles.clone());
            profiles
        },
    }
}

/// Run the setup command: report, then (unless `check_only`) save after asking.
///
/// # Errors
///
/// Returns an error if the saved config cannot be read or the new one cannot be written.
pub async fn run(endpoints: &Endpoints, yes: bool, check_only: bool, quick: bool) -> Result<()> {
    let client = reqwest::Client::new();
    if quick {
        println!("Looking around...\n");
    } else {
        println!("Looking around and timing your models (this can take a minute)...\n");
    }
    let found = probe(&client, endpoints, quick).await;
    let plan = plan(&found);
    print!("{}", render(&found, &plan));
    if check_only {
        return Ok(());
    }

    let merged = overlay(Config::load()?, &plan.config);
    if plan.config == Config::default() {
        println!(
            "\nNothing to save yet. Run `voice-notes setup` again once something is available."
        );
        return Ok(());
    }
    let Some(path) = Config::path() else {
        println!("\nCannot work out where to save the config (no HOME).");
        return Ok(());
    };
    println!(
        "\nSettings to save to {}:\n{}",
        path.display(),
        merged.render()?
    );
    if !yes && !confirm("Save these settings? [Y/n] ")? {
        println!("Not saved.");
        return Ok(());
    }
    let written = merged.save()?;
    println!(
        "Saved to {}. Flags still override these settings.",
        written.display()
    );
    Ok(())
}

/// Ask a yes/no question on the terminal; an empty answer means yes.
fn confirm(question: &str) -> Result<bool> {
    print!("{question}");
    std::io::stdout().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(!line.trim().to_lowercase().starts_with('n'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::Gpu;

    fn ids(list: &[&str]) -> Vec<String> {
        list.iter().map(ToString::to_string).collect()
    }

    /// Your real server: the fast instruct model wins; embeddings, vision and speech are never
    /// chat models, and the reasoning model ranks below the plain one.
    #[test]
    fn picks_the_best_chat_model() {
        let server = ids(&[
            "nomic-embed",
            "qwen-reasoner",
            "qwen25-vl-7b",
            "qwen3-coder-30b",
            "whisper",
        ]);
        assert_eq!(pick_chat_model(&server).as_deref(), Some("qwen3-coder-30b"));
        assert_eq!(pick_whisper(&server).as_deref(), Some("whisper"));
        assert_eq!(
            pick_chat_model(&ids(&["whisper", "qwen-reasoner"])).as_deref(),
            Some("qwen-reasoner"),
            "a reasoning model is still better than nothing"
        );
        assert_eq!(
            pick_chat_model(&ids(&["whisper", "nomic-embed", "kokoro"])),
            None
        );
        assert_eq!(pick_chat_model(&[]), None);
        assert_eq!(
            pick_chat_model(&ids(&["llama-3-8b", "mistral-7b"])).as_deref(),
            Some("llama-3-8b"),
            "ties go to the first name alphabetically"
        );
        assert_eq!(pick_whisper(&ids(&["qwen3"])), None);
    }

    fn everything() -> Findings {
        Findings {
            models: Some(ids(&["whisper", "qwen3-coder-30b", "nomic-embed"])),
            kokoro_url: Some("http://127.0.0.1:8880/v1/audio/speech".into()),
            system_voice: Some("espeak-ng"),
            player: Some("paplay"),
            claude: true,
            ..Findings::default()
        }
    }

    /// With everything present the plan is all ready, the config records the choices, and no
    /// managed install is offered.
    #[test]
    fn full_setup_needs_nothing() {
        let found = everything();
        let plan = plan(&found);
        assert!(
            plan.items
                .iter()
                .all(|i| matches!(i.status, Status::Ready(_)))
        );
        assert_eq!(plan.config.chat_model.as_deref(), Some("qwen3-coder-30b"));
        assert_eq!(plan.config.backend.as_deref(), Some("local"));
        assert_eq!(plan.config.tts.as_deref(), Some("kokoro"));
        assert_eq!(
            plan.config.tts_url.as_deref(),
            Some("http://127.0.0.1:8880/v1/audio/speech")
        );
        let preview = managed_preview(&plan, None, None);
        assert!(preview.required.is_empty() && preview.extras.is_empty());
        assert!(render(&found, &plan).contains("Everything needed is available."));
    }

    /// Without Kokoro the system voice is the fallback, and the preview offers Kokoro only.
    #[test]
    fn missing_kokoro_falls_back_and_previews_it() {
        let found = Findings {
            kokoro_url: None,
            ..everything()
        };
        let plan = plan(&found);
        assert!(matches!(plan.items[2].status, Status::Fallback(_)));
        assert_eq!(plan.config.tts.as_deref(), Some("system"));
        let preview = managed_preview(&plan, None, None);
        assert_eq!(preview.required.len(), 1);
        assert_eq!(preview.required[0].role, Role::Voice);
        let text = render(&found, &plan);
        assert!(text.contains("Kokoro-FastAPI") && text.contains("327 MB"));
    }

    /// With nothing at all, every role is missing, the preview lists all three components with
    /// their total size, and there is nothing to save.
    #[test]
    fn empty_machine_previews_everything() {
        let found = Findings::default();
        let plan = plan(&found);
        assert!(plan.items.iter().all(|i| i.status == Status::Missing));
        assert_eq!(plan.config, Config::default());
        let preview = managed_preview(&plan, None, None);
        assert_eq!(preview.required.len(), 3);
        assert!(!preview.tier_known && preview.tier == Tier::Light);
        let text = render(&found, &plan);
        assert!(text.contains("no server answered"));
        assert!(text.contains("preview, not available yet"));
        assert!(
            text.contains("Machine not detected, so assuming the light size"),
            "{text}"
        );
        assert!(text.contains("Total download: about 5.9 GB"), "{text}");
    }

    /// Only the claude CLI present: chat is a fallback and is not enabled without opting in.
    #[test]
    fn claude_only_is_not_chosen_automatically() {
        let found = Findings {
            claude: true,
            ..Findings::default()
        };
        let plan = plan(&found);
        assert!(matches!(plan.items[1].status, Status::Fallback(_)));
        assert!(plan.config.backend.is_none() && plan.config.chat_model.is_none());
    }

    /// New choices replace old ones, while unrelated saved settings are kept.
    #[test]
    fn overlay_keeps_unrelated_settings() {
        let saved = Config {
            language: Some("hr".into()),
            chat_model: Some("old".into()),
            ..Config::default()
        };
        let new = Config {
            chat_model: Some("new".into()),
            ..Config::default()
        };
        let merged = overlay(saved, &new);
        assert_eq!(merged.chat_model.as_deref(), Some("new"));
        assert_eq!(merged.language.as_deref(), Some("hr"));
    }

    /// Sizes read naturally.
    #[test]
    fn formats_sizes() {
        assert_eq!(human_size(327_212_226), "327 MB");
        assert_eq!(human_size(17_665_334_432), "17.7 GB");
        assert_eq!(human_size(574_041_195), "574 MB");
    }

    /// Serve one canned HTTP response on a local port and return its base URL.
    async fn serve_once(content_type: &'static str, body: &'static [u8]) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(head.as_bytes()).await;
            let _ = socket.write_all(body).await;
        });
        base
    }

    /// The probe reads the model list from a server and accepts a Kokoro endpoint only if it
    /// returns audio; an unreachable server and endpoint are simply absent.
    #[tokio::test]
    async fn probe_reads_server_and_kokoro() {
        let client = reqwest::Client::new();
        let server = serve_once(
            "application/json",
            br#"{"data":[{"id":"whisper"},{"id":"qwen3-coder-30b"}]}"#,
        )
        .await;
        let kokoro = format!(
            "{}/v1/audio/speech",
            serve_once("audio/wav", b"RIFFdata").await
        );
        let found = probe_at(&client, &server, std::slice::from_ref(&kokoro)).await;
        assert_eq!(found.models, Some(ids(&["whisper", "qwen3-coder-30b"])));
        assert_eq!(found.kokoro_url, Some(kokoro));

        let dead = "http://127.0.0.1:1";
        let found = probe_at(&client, dead, &[format!("{dead}/v1/audio/speech")]).await;
        assert_eq!(found.models, None);
        assert_eq!(found.kokoro_url, None);
    }

    /// An endpoint that answers with nothing is not counted as Kokoro.
    #[tokio::test]
    async fn empty_audio_does_not_count() {
        let client = reqwest::Client::new();
        let empty = format!("{}/v1/audio/speech", serve_once("audio/wav", b"").await);
        assert!(!kokoro_answers(&client, &empty).await);
    }

    /// Your server: the reasoning model becomes the `smart` model, distinct from the chat model.
    #[test]
    fn finds_the_smart_model() {
        let server = ids(&[
            "nomic-embed",
            "qwen-reasoner",
            "qwen25-vl-7b",
            "qwen3-coder-30b",
            "whisper",
        ]);
        assert_eq!(pick_smart_model(&server).as_deref(), Some("qwen-reasoner"));
        assert_eq!(
            pick_smart_model(&ids(&["qwen3-coder-30b", "whisper"])),
            None
        );
        assert_eq!(pick_smart_model(&ids(&["whisper", "nomic-embed"])), None);

        let found = Findings {
            models: Some(server),
            ..everything()
        };
        let plan = plan(&found);
        assert_eq!(plan.config.chat_model.as_deref(), Some("qwen3-coder-30b"));
        assert_eq!(plan.config.smart_model.as_deref(), Some("qwen-reasoner"));
    }

    /// With only a reasoning model, it is the chat model and there is no separate smart model.
    #[test]
    fn lone_reasoning_model_is_not_duplicated() {
        let found = Findings {
            models: Some(ids(&["whisper", "qwen-reasoner"])),
            ..everything()
        };
        let plan = plan(&found);
        assert_eq!(plan.config.chat_model.as_deref(), Some("qwen-reasoner"));
        assert_eq!(plan.config.smart_model, None);
    }

    /// The report lists the profiles this machine can run.
    #[test]
    fn report_lists_profiles() {
        let found = Findings {
            models: Some(ids(&["whisper", "qwen-reasoner", "qwen3-coder-30b"])),
            claude: true,
            ..everything()
        };
        let text = render(&found, &plan(&found));
        assert!(
            text.contains("Profiles") && text.contains("fast, dev, smart, think, claude"),
            "{text}"
        );
        let no_claude = Findings {
            claude: false,
            ..everything()
        };
        let basic = render(&no_claude, &plan(&no_claude));
        assert!(basic.contains("fast, dev\n"), "{basic}");
    }

    /// Saving keeps your own profiles and adds newly found settings.
    #[test]
    fn overlay_merges_profiles() {
        let mut saved = Config::default();
        saved
            .profiles
            .insert("mine".into(), crate::config::ProfileConfig::default());
        saved.profile = Some("dev".into());
        let new = Config {
            smart_model: Some("qwen-reasoner".into()),
            ..Config::default()
        };
        let merged = overlay(saved, &new);
        assert!(merged.profiles.contains_key("mine"));
        assert_eq!(merged.profile.as_deref(), Some("dev"));
        assert_eq!(merged.smart_model.as_deref(), Some("qwen-reasoner"));
    }

    const GIB: u64 = 1 << 30;

    fn machine(gpu: Gpu, ram_gib: u64) -> Hardware {
        Hardware {
            os: "linux",
            arch: "x86_64",
            cores: 16,
            ram_bytes: Some(ram_gib * GIB),
            gpu,
            disk_free_bytes: Some(500 * GIB),
        }
    }

    fn power_machine() -> Hardware {
        machine(
            Gpu::Shared {
                name: "Strix".into(),
                bytes: 80 * GIB,
            },
            125,
        )
    }

    fn names(components: &[Component]) -> Vec<&str> {
        components.iter().map(|c| c.name).collect()
    }

    /// Each size class suggests a different, fitting set of models.
    #[test]
    fn catalogue_follows_the_tier() {
        let small = catalogue(Tier::Minimal);
        assert!(
            names(&small).contains(&"Qwen3-4B (Q4_K_M)")
                && names(&small).contains(&"Whisper small")
        );
        assert!(names(&catalogue(Tier::Light)).contains(&"Qwen3-8B (Q4_K_M)"));
        assert!(names(&catalogue(Tier::Standard)).contains(&"Qwen3-14B (Q4_K_M)"));
        let power = catalogue(Tier::Power);
        assert!(names(&power).contains(&"Qwen3-Coder-30B-A3B-Instruct (Q4_K_XL)"));
        assert!(
            power.iter().any(|c| c.optional),
            "only the biggest tier offers a reasoning model"
        );
        for tier in [Tier::Minimal, Tier::Light, Tier::Standard] {
            assert!(catalogue(tier).iter().all(|c| !c.optional));
        }
        // Every set can do all three jobs.
        for tier in [Tier::Minimal, Tier::Light, Tier::Standard, Tier::Power] {
            let roles: Vec<Role> = catalogue(tier)
                .iter()
                .filter(|c| !c.optional)
                .map(|c| c.role)
                .collect();
            assert!(
                roles.contains(&Role::Stt)
                    && roles.contains(&Role::Chat)
                    && roles.contains(&Role::Voice)
            );
        }
    }

    /// Sizes taken from real files are exact; the rest are marked as estimates.
    #[test]
    fn measured_and_estimated_sizes() {
        let power = catalogue(Tier::Power);
        let coder = power
            .iter()
            .find(|c| c.role == Role::Chat && !c.optional)
            .unwrap();
        assert!(!coder.approximate && coder.bytes == 17_665_334_432);
        let reasoner = power.iter().find(|c| c.optional).unwrap();
        assert!(!reasoner.approximate && reasoner.bytes == 16_564_259_744);
        let light_chat = catalogue(Tier::Light)
            .into_iter()
            .find(|c| c.role == Role::Chat)
            .unwrap();
        assert!(light_chat.approximate);
        assert_eq!(size_text(&light_chat), "~5.0 GB");
        assert_eq!(size_text(coder), "17.7 GB");
    }

    /// A weak machine is offered small models, a strong one the big pair.
    #[test]
    fn preview_is_sized_for_the_machine() {
        let found = Findings::default();
        let plan = plan(&found);
        let laptop = machine(Gpu::None, 8); // CPU only: 4 GiB for models
        let preview = managed_preview(&plan, Some(&laptop), None);
        assert_eq!(preview.tier, Tier::Minimal);
        assert!(preview.tier_known);
        assert!(names(&preview.required).contains(&"Qwen3-4B (Q4_K_M)"));
        assert_eq!(preview.extras, []);

        let big = power_machine();
        let preview = managed_preview(&plan, Some(&big), None);
        assert_eq!(preview.tier, Tier::Power);
        assert!(names(&preview.required).contains(&"Qwen3-Coder-30B-A3B-Instruct (Q4_K_XL)"));
        assert_eq!(
            preview.extras.len(),
            1,
            "the reasoning model is offered as an extra"
        );
        let total: u64 = preview.required.iter().map(|c| c.bytes).sum();
        assert_eq!(human_size(total), "18.6 GB");
    }

    /// The reasoning extra is not offered when one is already configured.
    #[test]
    fn no_extra_when_a_smart_model_exists() {
        let found = Findings {
            models: Some(ids(&["whisper", "qwen-reasoner", "qwen3-coder-30b"])),
            ..everything()
        };
        let plan = plan(&found);
        let preview = managed_preview(&plan, Some(&power_machine()), None);
        assert!(preview.required.is_empty() && preview.extras.is_empty());
    }

    fn secs(s: f64) -> std::time::Duration {
        std::time::Duration::from_secs_f64(s)
    }

    fn timing(chat_warm: f64) -> Latency {
        Latency {
            stt: Some(secs(0.4)),
            chat: Some(ModelSpeed {
                model: "qwen3-coder-30b".into(),
                cold: secs(chat_warm),
                warm: secs(chat_warm),
            }),
            smart: None,
            voice: Some(secs(0.5)),
        }
    }

    /// A chat model that is too slow here is replaced in the preview by one a size smaller.
    #[test]
    fn slow_chat_suggests_a_smaller_model() {
        let found = everything();
        let plan = plan(&found);
        let hw = power_machine();
        let slow = timing(9.0);
        let preview = managed_preview(&plan, Some(&hw), Some(&slow));
        assert_eq!(names(&preview.required), ["Qwen3-14B (Q4_K_M)"]);
        let fast = timing(0.7);
        assert_eq!(managed_preview(&plan, Some(&hw), Some(&fast)).required, []);
        let weak = machine(Gpu::None, 8);
        let slow_weak = managed_preview(&plan, Some(&weak), Some(&slow));
        assert_eq!(
            names(&slow_weak.required),
            ["Qwen3-4B (Q4_K_M)"],
            "nothing smaller than minimal"
        );
    }

    /// The report has a machine section with the suggested size, and warns when there is no GPU.
    #[test]
    fn report_describes_the_machine() {
        let found = Findings {
            hardware: Some(power_machine()),
            ..everything()
        };
        let text = render(&found, &plan(&found));
        assert!(
            text.contains("Your machine") && text.contains("Strix"),
            "{text}"
        );
        assert!(
            text.contains("Suggested size: power (about 80 GB for models)"),
            "{text}"
        );

        let cpu = Findings {
            hardware: Some(machine(Gpu::None, 16)),
            ..everything()
        };
        let text = render(&cpu, &plan(&cpu));
        assert!(
            text.contains("No GPU found: replies will be slower"),
            "{text}"
        );
        assert!(text.contains("Suggested size: light"), "{text}");

        let unknown = Hardware {
            ram_bytes: None,
            ..machine(Gpu::None, 1)
        };
        let found = Findings {
            hardware: Some(unknown),
            ..everything()
        };
        assert!(render(&found, &plan(&found)).contains("Memory not detected"));
    }

    /// The speed section rates each part, and says so when a model loaded on demand.
    #[test]
    fn report_rates_the_speed() {
        let mut latency = timing(0.7);
        latency.chat = Some(ModelSpeed {
            model: "qwen3-coder-30b".into(),
            cold: secs(4.6),
            warm: secs(0.7),
        });
        latency.smart = Some(ModelSpeed {
            model: "qwen-reasoner".into(),
            cold: secs(9.0),
            warm: secs(6.0),
        });
        let found = Findings {
            latency: Some(latency),
            ..everything()
        };
        let text = render(&found, &plan(&found));
        assert!(text.contains("Speed"), "{text}");
        assert!(
            text.contains("[ok] Speech-to-text  transcribes a 3 s clip in 0.4 s"),
            "{text}"
        );
        assert!(text.contains("[ok] Chat            qwen3-coder-30b: first sentence in 0.7 s (the first reply took 4.6 s while the model loaded)"), "{text}");
        assert!(text.contains("[!!] Smart           qwen-reasoner: first sentence in 6.0 s: fine for `think`, slow for conversation"), "{text}");
        assert!(
            text.contains("[ok] Voice           Kokoro speaks one sentence in 0.5 s"),
            "{text}"
        );
        assert!(
            text.contains("Everything needed is available."),
            "a slow smart model is not a problem"
        );
    }

    /// A slow chat model is flagged, explained and replaced in the managed-install preview.
    #[test]
    fn report_flags_slow_chat() {
        let found = Findings {
            hardware: Some(machine(
                Gpu::Dedicated {
                    name: "RTX".into(),
                    vram_bytes: 12 * GIB,
                },
                32,
            )),
            latency: Some(timing(9.0)),
            ..everything()
        };
        let text = render(&found, &plan(&found));
        assert!(
            text.contains("[!!] Chat") && text.contains(": too slow for talking"),
            "{text}"
        );
        assert!(text.contains("replies start slowly here (9.0 s)"), "{text}");
        assert!(
            text.contains("Chosen for your machine: light size."),
            "{text}"
        );
        assert!(
            text.contains("Qwen3-8B") && !text.contains("Everything needed"),
            "{text}"
        );
    }

    /// Nothing timed gives an explicit note instead of an empty section.
    #[test]
    fn empty_timing_is_explained() {
        let found = Findings {
            latency: Some(Latency::default()),
            ..everything()
        };
        assert!(render(&found, &plan(&found)).contains("(nothing could be timed)"));
    }

    /// The preview warns when the downloads would not fit on the disk.
    #[test]
    fn warns_about_disk_space() {
        let mut hw = power_machine();
        hw.disk_free_bytes = Some(5 * GIB);
        let found = Findings {
            hardware: Some(hw),
            ..Findings::default()
        };
        let text = render(&found, &plan(&found));
        assert!(
            text.contains("[!!] Not enough free disk space: 5.0 GB free."),
            "{text}"
        );
        let roomy = Findings {
            hardware: Some(power_machine()),
            ..Findings::default()
        };
        assert!(!render(&roomy, &plan(&roomy)).contains("Not enough free disk"));
    }

    /// The preview says plainly which sizes are estimates.
    #[test]
    fn preview_marks_estimates() {
        let found = Findings {
            hardware: Some(machine(Gpu::None, 24)), // 12 GiB for models: the light size
            ..Findings::default()
        };
        let text = render(&found, &plan(&found));
        assert!(
            text.contains("~5.0 GB") && text.contains("Total download: about"),
            "{text}"
        );
        let power = Findings {
            hardware: Some(power_machine()),
            ..Findings::default()
        };
        let text = render(&power, &plan(&power));
        assert!(
            text.contains("Total download: 18.6 GB") && !text.contains("about 18"),
            "{text}"
        );
    }

    /// When the server calls the Whisper model something else, setup remembers that name; the
    /// default name is not written.
    #[test]
    fn remembers_a_different_whisper_name() {
        let found = Findings {
            models: Some(ids(&["whisper-large-v3", "qwen3-coder-30b"])),
            ..everything()
        };
        assert_eq!(
            plan(&found).config.stt_model.as_deref(),
            Some("whisper-large-v3")
        );
        let plain = plan(&everything());
        assert_eq!(
            plain.config.stt_model, None,
            "the default name needs no saving"
        );
    }

    /// The report names the model-list URL that was asked, so a wrong address is easy to spot.
    #[test]
    fn report_names_the_server_that_did_not_answer() {
        let found = Findings {
            models_url: "http://gpu-box:9000/v1/models".into(),
            ..Findings::default()
        };
        let text = render(&found, &plan(&found));
        assert!(
            text.contains("no server answered at http://gpu-box:9000/v1/models"),
            "{text}"
        );
        let unset = render(&Findings::default(), &plan(&Findings::default()));
        assert!(
            unset.contains("no server answered at http://127.0.0.1:8080/v1/models"),
            "{unset}"
        );
    }

    /// The probe asks the model-list URL it is given, whatever its path.
    #[tokio::test]
    async fn probe_asks_the_given_models_url() {
        let (base, request) = crate::test_support::capture_once(
            "application/json",
            r#"{"data":[{"id":"whisper"},{"id":"my-chat"}]}"#,
        )
        .await;
        let url = format!("{base}/custom/models");
        let found = probe_at(&reqwest::Client::new(), &url, &[]).await;
        assert_eq!(found.models, Some(ids(&["whisper", "my-chat"])));
        assert_eq!(found.models_url, url);
        assert!(request.await.unwrap().starts_with("GET /custom/models "));
    }

    /// Saved endpoint keys survive a later setup run, and new values replace old ones.
    #[test]
    fn overlay_keeps_endpoint_settings() {
        let saved = Config {
            server: Some("http://gpu-box:8080".into()),
            stt_url: Some("http://stt/x".into()),
            ..Config::default()
        };
        let new = Config {
            stt_model: Some("whisper-large-v3".into()),
            ..Config::default()
        };
        let merged = overlay(saved, &new);
        assert_eq!(merged.server.as_deref(), Some("http://gpu-box:8080"));
        assert_eq!(merged.stt_url.as_deref(), Some("http://stt/x"));
        assert_eq!(merged.stt_model.as_deref(), Some("whisper-large-v3"));
    }
}
