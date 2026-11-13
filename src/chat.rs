//! Conversation with a chat backend: your local models, or Claude through the CLI.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::RandomState;
use std::fmt::Write as _;
use std::hash::{BuildHasher, Hasher};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::sync::mpsc::UnboundedSender;

/// Style instructions so answers sound right when read aloud.
pub const SPOKEN_STYLE: &str = "You are a voice assistant. Your answers are read aloud, so reply \
in one to three short, plain sentences. Never use markdown, lists, code blocks, emoji or URLs.";

/// How many past messages are sent to a local model.
const MAX_HISTORY: usize = 20;
/// Reasoning models can think for a long time.
const REPLY_TIMEOUT: Duration = Duration::from_secs(300);

/// A chat message in the OpenAI-compatible format.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    #[must_use]
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

/// Who answers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Backend {
    /// A model on the local OpenAI-compatible server (for example `qwen-reasoner`). With
    /// `think` off, reasoning models skip their long thinking phase, which is much faster.
    Local { model: String, think: bool },
    /// The `claude` CLI in headless mode, limited to read-only tools.
    Claude,
}

impl Backend {
    /// Short name for the screen.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Local { model, .. } => model.clone(),
            Self::Claude => "claude".to_string(),
        }
    }
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_template_kwargs: Option<TemplateKwargs>,
}

#[derive(Serialize)]
struct TemplateKwargs {
    enable_thinking: bool,
}

#[derive(Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
}

#[derive(Deserialize)]
struct StreamDelta {
    content: Option<String>,
}

/// One line of a server-sent-events stream.
#[derive(Debug, PartialEq, Eq)]
enum Sse {
    /// A piece of the answer.
    Delta(String),
    /// The stream is finished.
    Done,
    /// Nothing of interest (comments, role-only chunks, blank lines).
    Skip,
}

fn parse_sse_line(line: &str) -> Sse {
    let Some(data) = line.trim().strip_prefix("data:") else {
        return Sse::Skip;
    };
    let data = data.trim();
    if data == "[DONE]" {
        return Sse::Done;
    }
    serde_json::from_str::<StreamChunk>(data)
        .ok()
        .and_then(|c| c.choices.into_iter().next())
        .and_then(|c| c.delta.content)
        .filter(|t| !t.is_empty())
        .map_or(Sse::Skip, Sse::Delta)
}

/// The part of `raw` a listener should hear so far: reasoning removed, and a possible start
/// of a `<think>` tag held back until it is clear what it is.
fn visible_text(raw: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let text = if let Some(end) = raw.rfind(CLOSE) {
        &raw[end + CLOSE.len()..]
    } else if let Some(start) = raw.find(OPEN) {
        &raw[..start]
    } else {
        raw
    };
    let mut text = text.trim_start().to_string();
    for keep in (1..OPEN.len()).rev() {
        if text.ends_with(&OPEN[..keep]) {
            text.truncate(text.len() - keep);
            break;
        }
    }
    text
}

/// Remove a reasoning model's `<think>…</think>` block, leaving only the answer.
///
/// # Examples
///
/// ```
/// use voice_notes::chat::strip_reasoning;
///
/// assert_eq!(strip_reasoning("<think>hmm, 2+2</think>\n\nFour."), "Four.");
/// assert_eq!(strip_reasoning("no thinking here"), "no thinking here");
/// ```
#[must_use]
pub fn strip_reasoning(text: &str) -> String {
    const CLOSE: &str = "</think>";
    let mut s = text.to_string();
    if let Some(end) = s.rfind(CLOSE) {
        // Also covers templates that omit the opening tag.
        s = s[end + CLOSE.len()..].to_string();
    } else if let Some(start) = s.find("<think>") {
        s.truncate(start);
    }
    s.trim().to_string()
}

/// Arguments for one headless `claude` call; the prompt itself goes on stdin.
#[must_use]
pub fn claude_args(session: &str, resume: bool) -> Vec<String> {
    let mut args = vec!["-p".to_string(), "--output-format".into(), "text".into()];
    if resume {
        args.extend(["--resume".to_string(), session.to_string()]);
    } else {
        args.extend(["--session-id".to_string(), session.to_string()]);
    }
    args.extend([
        "--append-system-prompt".to_string(),
        SPOKEN_STYLE.to_string(),
        // Attached form: the option takes a list and would otherwise swallow later arguments.
        "--allowedTools=Read,Grep,Glob".to_string(),
    ]);
    args
}

/// A random version-4-style UUID for naming a Claude session.
fn new_session_id() -> String {
    let hasher = RandomState::new();
    let mut parts = [0u64; 2];
    for (i, part) in parts.iter_mut().enumerate() {
        let mut h = hasher.build_hasher();
        h.write_usize(i);
        *part = h.finish();
    }
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&parts[0].to_be_bytes());
    b[8..].copy_from_slice(&parts[1].to_be_bytes());
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let hex = b.iter().fold(String::new(), |mut acc, x| {
        let _ = write!(acc, "{x:02x}");
        acc
    });
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Ask the local server for a streamed answer, forwarding the visible text as it arrives.
async fn stream_local(
    client: &reqwest::Client,
    model: &str,
    think: bool,
    messages: &[ChatMessage],
    deltas: Option<&UnboundedSender<String>>,
) -> Result<String> {
    let mut response = client
        .post("http://127.0.0.1:8080/v1/chat/completions")
        .timeout(REPLY_TIMEOUT)
        .json(&Request {
            model,
            messages,
            stream: true,
            chat_template_kwargs: (!think).then_some(TemplateKwargs {
                enable_thinking: false,
            }),
        })
        .send()
        .await?
        .error_for_status()?;

    let mut raw = String::new();
    let mut sent = 0;
    let mut pending: Vec<u8> = Vec::new();
    'stream: while let Some(chunk) = response.chunk().await? {
        pending.extend_from_slice(&chunk);
        // Decode whole lines only, so multi-byte characters are never split.
        while let Some(newline) = pending.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = pending.drain(..=newline).collect();
            match parse_sse_line(&String::from_utf8_lossy(&line)) {
                Sse::Delta(piece) => {
                    raw.push_str(&piece);
                    let visible = visible_text(&raw);
                    if let Some(new) = visible.get(sent..).filter(|n| !n.is_empty()) {
                        if let Some(tx) = deltas {
                            let _ = tx.send(new.to_string());
                        }
                        sent = visible.len();
                    }
                }
                Sse::Done => break 'stream,
                Sse::Skip => {}
            }
        }
    }
    Ok(strip_reasoning(&raw))
}

/// A multi-turn conversation that remembers what was said.
pub struct Conversation {
    backend: Backend,
    history: Vec<ChatMessage>,
    session: String,
    started: bool,
}

impl Conversation {
    #[must_use]
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            history: Vec::new(),
            session: new_session_id(),
            started: false,
        }
    }

    /// Forget everything and start a fresh conversation.
    pub fn reset(&mut self) {
        self.history.clear();
        self.session = new_session_id();
        self.started = false;
    }

    /// Send `user_text` and return the reply, ready to be shown and spoken.
    ///
    /// If `deltas` is given, the answer is also sent through it piece by piece as the model
    /// writes it (reasoning removed), so it can be shown and spoken before it is complete.
    ///
    /// # Errors
    ///
    /// Returns an error if the server or the `claude` CLI cannot be reached or fails, or if it
    /// answers with nothing.
    pub async fn reply(
        &mut self,
        client: &reqwest::Client,
        user_text: &str,
        deltas: Option<&UnboundedSender<String>>,
    ) -> Result<String> {
        let reply = match self.backend.clone() {
            Backend::Local { model, think } => {
                self.reply_local(client, &model, think, user_text, deltas)
                    .await?
            }
            Backend::Claude => {
                let reply = self.reply_claude(user_text).await?;
                if let Some(tx) = deltas {
                    let _ = tx.send(reply.clone());
                }
                reply
            }
        };
        if reply.is_empty() {
            bail!("the assistant sent an empty answer");
        }
        Ok(reply)
    }

    async fn reply_local(
        &mut self,
        client: &reqwest::Client,
        model: &str,
        think: bool,
        user_text: &str,
        deltas: Option<&UnboundedSender<String>>,
    ) -> Result<String> {
        self.history.push(ChatMessage::new("user", user_text));
        let start = self.history.len().saturating_sub(MAX_HISTORY);
        let mut messages = vec![ChatMessage::new("system", SPOKEN_STYLE)];
        messages.extend_from_slice(&self.history[start..]);

        match stream_local(client, model, think, &messages, deltas).await {
            Ok(reply) => {
                self.history.push(ChatMessage::new("assistant", &reply));
                Ok(reply)
            }
            Err(e) => {
                self.history.pop();
                Err(e)
            }
        }
    }

    async fn reply_claude(&mut self, user_text: &str) -> Result<String> {
        let args = claude_args(&self.session, self.started);
        let mut child = Command::new("claude")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .context("could not run `claude` (is Claude Code installed and on PATH?)")?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(user_text.as_bytes()).await?;
        }
        let output = tokio::time::timeout(REPLY_TIMEOUT, child.wait_with_output())
            .await
            .context("claude took too long to answer")??;
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            bail!("claude failed: {}", err.trim());
        }
        self.started = true;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reasoning blocks are removed whether or not they are well formed.
    #[test]
    fn strips_reasoning_variants() {
        assert_eq!(strip_reasoning("<think>a</think>Answer."), "Answer.");
        assert_eq!(strip_reasoning("a b c</think>\nAnswer."), "Answer.");
        assert_eq!(strip_reasoning("Answer.<think>never closed"), "Answer.");
        assert_eq!(
            strip_reasoning("<think>x</think>one<think>y</think>two"),
            "two"
        );
        assert_eq!(strip_reasoning("  plain  "), "plain");
    }

    /// The first Claude call names the session, later calls resume it, and the prompt is never
    /// an argument.
    #[test]
    fn claude_args_start_then_resume() {
        let first = claude_args("abc", false);
        assert!(first.windows(2).any(|w| w == ["--session-id", "abc"]));
        assert!(!first.contains(&"--resume".to_string()));
        let later = claude_args("abc", true);
        assert!(later.windows(2).any(|w| w == ["--resume", "abc"]));
        assert!(later.contains(&"--allowedTools=Read,Grep,Glob".to_string()));
        assert_eq!(later[0], "-p");
    }

    /// Session ids look like version-4 UUIDs and differ between conversations.
    #[test]
    fn session_ids_are_uuids() {
        let a = new_session_id();
        assert_eq!(a.len(), 36);
        assert_eq!(a.matches('-').count(), 4);
        assert_eq!(a.as_bytes()[14], b'4');
        assert!(matches!(a.as_bytes()[19], b'8' | b'9' | b'a' | b'b'));
        assert_ne!(a, new_session_id());
    }

    /// Resetting starts a new Claude session.
    #[test]
    fn reset_changes_session() {
        let mut c = Conversation::new(Backend::Claude);
        c.started = true;
        let before = c.session.clone();
        c.reset();
        assert!(!c.started);
        assert_ne!(before, c.session);
    }

    /// Backend labels shown in the header.
    #[test]
    fn backend_labels() {
        assert_eq!(Backend::Claude.label(), "claude");
        assert_eq!(
            Backend::Local {
                model: "qwen-reasoner".into(),
                think: false
            }
            .label(),
            "qwen-reasoner"
        );
    }

    /// Each kind of stream line is understood; anything else is skipped.
    #[test]
    fn parses_sse_lines() {
        let delta = r#"data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}"#;
        assert_eq!(parse_sse_line(delta), Sse::Delta("Hi".into()));
        assert_eq!(parse_sse_line("data: [DONE]"), Sse::Done);
        assert_eq!(parse_sse_line(""), Sse::Skip);
        assert_eq!(parse_sse_line(": keep-alive"), Sse::Skip);
        let role_only = r#"data: {"choices":[{"delta":{"role":"assistant","content":null}}]}"#;
        assert_eq!(parse_sse_line(role_only), Sse::Skip);
        assert_eq!(parse_sse_line("data: {not json"), Sse::Skip);
    }

    /// Reasoning stays hidden while streaming, and a half-written tag is held back.
    #[test]
    fn visible_text_hides_reasoning_while_streaming() {
        assert_eq!(visible_text(""), "");
        assert_eq!(visible_text("<"), "");
        assert_eq!(visible_text("<thin"), "");
        assert_eq!(visible_text("<think>\nplanning"), "");
        assert_eq!(visible_text("<think>\n\n</think>\n\nTwo"), "Two");
        assert_eq!(visible_text("Two plus two is"), "Two plus two is");
        assert_eq!(visible_text("a < b"), "a < b");
        assert_eq!(visible_text("Hello <"), "Hello ");
    }

    /// Visible text only ever grows as the raw stream grows, so deltas can be sliced by length.
    #[test]
    fn visible_text_grows_monotonically() {
        let full = "<think>\n\n</think>\n\nTwo plus two equals four.";
        let mut last = 0;
        for end in (0..=full.len()).filter(|i| full.is_char_boundary(*i)) {
            let visible = visible_text(&full[..end]);
            assert!(visible.len() >= last, "shrank at {end}");
            last = visible.len();
        }
        assert_eq!(visible_text(full), "Two plus two equals four.");
    }

    /// Thinking is switched off in the request unless asked for.
    #[test]
    fn request_disables_thinking_by_default() {
        let messages = [ChatMessage::new("user", "hi")];
        let off = Request {
            model: "m",
            messages: &messages,
            stream: true,
            chat_template_kwargs: Some(TemplateKwargs {
                enable_thinking: false,
            }),
        };
        let json = serde_json::to_string(&off).unwrap();
        assert!(json.contains("\"enable_thinking\":false") && json.contains("\"stream\":true"));
        let on = Request {
            model: "m",
            messages: &messages,
            stream: true,
            chat_template_kwargs: None,
        };
        assert!(
            !serde_json::to_string(&on)
                .unwrap()
                .contains("chat_template_kwargs")
        );
    }
}
