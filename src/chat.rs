//! Conversation with a chat backend: your local models, or Claude through the CLI.

use crate::endpoints::Endpoints;
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

/// Style for the `dev` profile: technical help that still sounds right when spoken.
pub const DEV_STYLE: &str = "You are a voice assistant for a software developer. Answers are read \
aloud, so reply in a few short, plain sentences. Be technically precise. Never read code or \
commands character by character: describe what they do instead. Never use markdown, lists, \
code blocks, emoji or URLs.";

/// A piece of an answer as it streams in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplyDelta {
    /// The model's reasoning: shown on screen, never spoken.
    Thinking(String),
    /// The answer itself.
    Text(String),
}

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
    /// True if what you say leaves this machine: the Claude backend sends transcripts to
    /// Anthropic, while local models keep everything on this machine.
    ///
    /// # Examples
    ///
    /// ```
    /// use voice_notes::chat::Backend;
    ///
    /// assert!(Backend::Claude.is_remote());
    /// assert!(!Backend::Local { model: "qwen3-coder-30b".into(), think: false }.is_remote());
    /// ```
    #[must_use]
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Claude)
    }

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

/// Drop a trailing piece of `tag` (a tag still being written) until it is clear what it is.
fn hold_back(text: &str, tag: &str) -> String {
    let mut text = text.to_string();
    for keep in (1..tag.len()).rev() {
        if text.ends_with(&tag[..keep]) {
            text.truncate(text.len() - keep);
            break;
        }
    }
    text
}

/// Split the raw stream so far into `(thinking, answer)`. A half-written tag is held back.
fn split_think(raw: &str) -> (String, String) {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    if let Some(end) = raw.rfind(CLOSE) {
        let before = &raw[..end];
        let thinking = before.trim_start();
        let thinking = thinking.strip_prefix(OPEN).unwrap_or(thinking).trim();
        (
            thinking.to_string(),
            raw[end + CLOSE.len()..].trim_start().to_string(),
        )
    } else if let Some(start) = raw.find(OPEN) {
        let thinking = raw[start + OPEN.len()..].trim_start();
        (
            hold_back(thinking, CLOSE),
            raw[..start].trim_start().to_string(),
        )
    } else {
        (String::new(), hold_back(raw.trim_start(), OPEN))
    }
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
pub fn claude_args(session: &str, resume: bool, prompt: &str) -> Vec<String> {
    let mut args = vec!["-p".to_string(), "--output-format".into(), "text".into()];
    if resume {
        args.extend(["--resume".to_string(), session.to_string()]);
    } else {
        args.extend(["--session-id".to_string(), session.to_string()]);
    }
    args.extend([
        "--append-system-prompt".to_string(),
        prompt.to_string(),
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
    url: &str,
    model: &str,
    think: bool,
    messages: &[ChatMessage],
    deltas: Option<&UnboundedSender<ReplyDelta>>,
) -> Result<String> {
    let mut response = client
        .post(url)
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
    let mut sent_answer = 0;
    let mut sent_thinking = 0;
    let mut pending: Vec<u8> = Vec::new();
    'stream: while let Some(chunk) = response.chunk().await? {
        pending.extend_from_slice(&chunk);
        // Decode whole lines only, so multi-byte characters are never split.
        while let Some(newline) = pending.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = pending.drain(..=newline).collect();
            match parse_sse_line(&String::from_utf8_lossy(&line)) {
                Sse::Delta(piece) => {
                    raw.push_str(&piece);
                    let (thinking, answer) = split_think(&raw);
                    if let Some(new) = thinking.get(sent_thinking..).filter(|n| !n.is_empty()) {
                        if let Some(tx) = deltas {
                            let _ = tx.send(ReplyDelta::Thinking(new.to_string()));
                        }
                        sent_thinking = thinking.len();
                    }
                    if let Some(new) = answer.get(sent_answer..).filter(|n| !n.is_empty()) {
                        if let Some(tx) = deltas {
                            let _ = tx.send(ReplyDelta::Text(new.to_string()));
                        }
                        sent_answer = answer.len();
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
    url: String,
    prompt: String,
    history: Vec<ChatMessage>,
    session: String,
    started: bool,
}

impl Conversation {
    #[must_use]
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            url: Endpoints::default().chat,
            prompt: SPOKEN_STYLE.to_string(),
            history: Vec::new(),
            session: new_session_id(),
            started: false,
        }
    }

    /// Send chat requests to `url` instead of the default local server.
    #[must_use]
    pub fn with_url(mut self, url: String) -> Self {
        self.url = url;
        self
    }

    /// Use `prompt` as the system prompt instead of the default spoken style.
    #[must_use]
    pub fn with_prompt(mut self, prompt: Option<String>) -> Self {
        if let Some(prompt) = prompt {
            self.prompt = prompt;
        }
        self
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
    /// writes it, so it can be shown and spoken before it is complete. Reasoning arrives as
    /// [`ReplyDelta::Thinking`] and is never part of the returned answer.
    ///
    /// # Errors
    ///
    /// Returns an error if the server or the `claude` CLI cannot be reached or fails, or if it
    /// answers with nothing.
    pub async fn reply(
        &mut self,
        client: &reqwest::Client,
        user_text: &str,
        deltas: Option<&UnboundedSender<ReplyDelta>>,
    ) -> Result<String> {
        let reply = match self.backend.clone() {
            Backend::Local { model, think } => {
                self.reply_local(client, &model, think, user_text, deltas)
                    .await?
            }
            Backend::Claude => {
                let reply = self.reply_claude(user_text).await?;
                if let Some(tx) = deltas {
                    let _ = tx.send(ReplyDelta::Text(reply.clone()));
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
        deltas: Option<&UnboundedSender<ReplyDelta>>,
    ) -> Result<String> {
        self.history.push(ChatMessage::new("user", user_text));
        let start = self.history.len().saturating_sub(MAX_HISTORY);
        let mut messages = vec![ChatMessage::new("system", &self.prompt)];
        messages.extend_from_slice(&self.history[start..]);

        match stream_local(client, &self.url, model, think, &messages, deltas).await {
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
        let args = claude_args(&self.session, self.started, &self.prompt);
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

    /// The part of `raw` a listener should hear so far: reasoning removed.
    fn visible_text(raw: &str) -> String {
        split_think(raw).1
    }

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
        let first = claude_args("abc", false, SPOKEN_STYLE);
        assert!(first.windows(2).any(|w| w == ["--session-id", "abc"]));
        assert!(!first.contains(&"--resume".to_string()));
        let later = claude_args("abc", true, DEV_STYLE);
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
        assert!(Backend::Claude.is_remote());
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

    /// The stream is split into thinking and answer, holding back half-written tags.
    #[test]
    fn splits_thinking_from_answer() {
        let split = |raw: &str| split_think(raw);
        assert_eq!(split(""), (String::new(), String::new()));
        assert_eq!(
            split("plain answer"),
            (String::new(), "plain answer".into())
        );
        assert_eq!(
            split("<think>\nLet me see"),
            ("Let me see".into(), String::new())
        );
        assert_eq!(
            split("<think>\nLet me see</"),
            ("Let me see".into(), String::new())
        );
        assert_eq!(
            split("<think>\nLet me see\n</think>\n\nFour."),
            ("Let me see".into(), "Four.".into())
        );
        assert_eq!(
            split("<think>\n\n</think>\n\nHi"),
            (String::new(), "Hi".into())
        );
        assert_eq!(
            split("Answer.<think>later"),
            ("later".into(), "Answer.".into())
        );
        assert_eq!(split("Tag <thin"), (String::new(), "Tag ".into()));
    }

    /// Thinking text only grows while it is being written, so it can be sliced by length.
    #[test]
    fn thinking_grows_while_streaming() {
        let full = "<think>\nFirst, add them.\n</think>\n\nFour.";
        let mut last = 0;
        for end in (0..full.find("</think>").unwrap()).filter(|i| full.is_char_boundary(*i)) {
            let (thinking, answer) = split_think(&full[..end]);
            assert!(thinking.len() >= last, "shrank at {end}");
            assert_eq!(answer, "");
            last = thinking.len();
        }
    }

    /// The dev profile's prompt differs from the default one, and a custom prompt replaces it.
    #[test]
    fn prompts() {
        assert_ne!(DEV_STYLE, SPOKEN_STYLE);
        assert!(DEV_STYLE.contains("software developer"));
        let default = Conversation::new(Backend::Claude);
        assert_eq!(default.prompt, SPOKEN_STYLE);
        let dev = Conversation::new(Backend::Claude).with_prompt(Some(DEV_STYLE.to_string()));
        assert_eq!(dev.prompt, DEV_STYLE);
        let unchanged = Conversation::new(Backend::Claude).with_prompt(None);
        assert_eq!(unchanged.prompt, SPOKEN_STYLE);
    }

    /// Claude is given the profile's prompt, not always the default.
    #[test]
    fn claude_gets_the_profile_prompt() {
        let args = claude_args("abc", false, DEV_STYLE);
        let at = args
            .iter()
            .position(|a| a == "--append-system-prompt")
            .unwrap();
        assert_eq!(args[at + 1], DEV_STYLE);
    }

    /// Chat requests go to the local server unless another URL is given.
    #[test]
    fn chat_url_defaults_and_can_be_changed() {
        let default = Conversation::new(Backend::Claude);
        assert_eq!(default.url, "http://127.0.0.1:8080/v1/chat/completions");
        let custom = Conversation::new(Backend::Claude)
            .with_url("http://gpu-box:9000/v1/chat/completions".into());
        assert_eq!(custom.url, "http://gpu-box:9000/v1/chat/completions");
    }

    const SSE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"Four\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\" it is.\"}}]}\n\ndata: [DONE]\n\n";

    /// A streamed reply is read from the configured URL, with the configured model, and the
    /// pieces arrive in order.
    #[tokio::test]
    async fn streams_from_the_configured_url() {
        let (base, request) = crate::test_support::capture_once("text/event-stream", SSE).await;
        let backend = Backend::Local {
            model: "my-chat-model".into(),
            think: false,
        };
        let mut conversation =
            Conversation::new(backend).with_url(format!("{base}/elsewhere/chat"));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let reply = conversation
            .reply(&reqwest::Client::new(), "What is two plus two?", Some(&tx))
            .await
            .unwrap();
        assert_eq!(reply, "Four it is.");

        let mut pieces = Vec::new();
        while let Ok(delta) = rx.try_recv() {
            pieces.push(delta);
        }
        assert_eq!(
            pieces,
            [
                ReplyDelta::Text("Four".into()),
                ReplyDelta::Text(" it is.".into())
            ]
        );
        let sent = request.await.unwrap();
        assert!(sent.starts_with("POST /elsewhere/chat "), "{sent}");
        assert!(
            sent.contains("\"model\":\"my-chat-model\"") && sent.contains("\"stream\":true"),
            "{sent}"
        );
        assert!(sent.contains("What is two plus two?"));
    }

    /// An unreachable chat server is an error, and the failed question is forgotten.
    #[tokio::test]
    async fn unreachable_chat_server_is_an_error() {
        let backend = Backend::Local {
            model: "m".into(),
            think: false,
        };
        let mut conversation =
            Conversation::new(backend).with_url("http://127.0.0.1:1/v1/chat/completions".into());
        assert!(
            conversation
                .reply(&reqwest::Client::new(), "hi", None)
                .await
                .is_err()
        );
        assert!(
            conversation.history.is_empty(),
            "the failed question is not kept"
        );
    }
}
