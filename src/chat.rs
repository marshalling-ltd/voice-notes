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
    /// A model on the local OpenAI-compatible server (for example `qwen-reasoner`).
    Local { model: String },
    /// The `claude` CLI in headless mode, limited to read-only tools.
    Claude,
}

impl Backend {
    /// Short name for the screen.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Local { model } => model.clone(),
            Self::Claude => "claude".to_string(),
        }
    }
}

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
}

#[derive(Deserialize)]
struct Response {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ReplyMessage,
}

#[derive(Deserialize)]
struct ReplyMessage {
    content: Option<String>,
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
    /// # Errors
    ///
    /// Returns an error if the server or the `claude` CLI cannot be reached or fails, or if it
    /// answers with nothing.
    pub async fn reply(&mut self, client: &reqwest::Client, user_text: &str) -> Result<String> {
        let reply = match self.backend.clone() {
            Backend::Local { model } => self.reply_local(client, &model, user_text).await?,
            Backend::Claude => self.reply_claude(user_text).await?,
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
        user_text: &str,
    ) -> Result<String> {
        self.history.push(ChatMessage::new("user", user_text));
        let start = self.history.len().saturating_sub(MAX_HISTORY);
        let mut messages = vec![ChatMessage::new("system", SPOKEN_STYLE)];
        messages.extend_from_slice(&self.history[start..]);

        let result = async {
            let response = client
                .post("http://127.0.0.1:8080/v1/chat/completions")
                .timeout(REPLY_TIMEOUT)
                .json(&Request {
                    model,
                    messages: &messages,
                })
                .send()
                .await?
                .error_for_status()?;
            let body: Response = response.json().await?;
            let content = body
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)
                .unwrap_or_default();
            anyhow::Ok(strip_reasoning(&content))
        }
        .await;

        match result {
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
                model: "qwen-reasoner".into()
            }
            .label(),
            "qwen-reasoner"
        );
    }
}
