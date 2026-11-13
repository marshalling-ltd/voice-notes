//! Text-to-speech: a Kokoro server (OpenAI-compatible `/v1/audio/speech`), or the system voice
//! (`say` on macOS, `espeak-ng` elsewhere) as a fallback.

use anyhow::{Context, Result, anyhow, bail};
use serde::Serialize;
use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Duration;

/// Longest text spoken in one go.
const MAX_SPOKEN_CHARS: usize = 2000;

/// Turn model output into something pleasant to hear: no code blocks, markdown marks, link
/// targets or URLs, and a sane length.
///
/// # Examples
///
/// ```
/// use voice_notes::speech::speakable;
///
/// let text = "**Sure.** See [the docs](https://example.com) or run:\n```sh\nls -la\n```\nDone_now!";
/// assert_eq!(speakable(text), "Sure. See the docs or run: Done now!");
/// ```
#[must_use]
pub fn speakable(text: &str) -> String {
    let mut prose = String::new();
    let mut in_code = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_code = !in_code;
        } else if !in_code {
            prose.push_str(line);
            prose.push('\n');
        }
    }

    let prose = strip_links(&prose);
    let marks_removed: String = prose
        .chars()
        .map(|c| match c {
            '*' | '#' | '`' | '>' | '|' | '~' | '_' => ' ',
            other => other,
        })
        .collect();
    let words: Vec<&str> = marks_removed
        .split_whitespace()
        .filter(|w| !w.starts_with("http://") && !w.starts_with("https://"))
        .collect();
    let mut out = words.join(" ");

    if out.chars().count() > MAX_SPOKEN_CHARS {
        out = out.chars().take(MAX_SPOKEN_CHARS).collect();
        if let Some(cut) = out.rfind(' ') {
            out.truncate(cut);
        }
    }
    out
}

/// Replace `[label](target)` with `label`.
fn strip_links(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find('[') {
        let after = &rest[open + 1..];
        let link = after.find("](").and_then(|close| {
            after[close + 2..]
                .find(')')
                .map(|end| (close, close + 2 + end))
        });
        if let Some((close, end)) = link {
            out.push_str(&rest[..open]);
            out.push_str(&after[..close]);
            rest = &after[end + 1..];
        } else {
            out.push_str(&rest[..=open]);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

/// Cuts streamed text into sentences so each can be spoken as soon as it is complete.
///
/// A sentence ends at `.`, `!` or `?` followed by whitespace. After 50 characters a comma,
/// semicolon or colon also ends a chunk, so long sentences start playing early.
///
/// # Examples
///
/// ```
/// use voice_notes::speech::SentenceSplitter;
///
/// let mut splitter = SentenceSplitter::default();
/// assert!(splitter.push("Hello the").is_empty());
/// assert_eq!(splitter.push("re. How are"), vec!["Hello there."]);
/// assert_eq!(splitter.push(" you? Fine"), vec!["How are you?"]);
/// assert_eq!(splitter.finish().as_deref(), Some("Fine"));
/// ```
#[derive(Default)]
pub struct SentenceSplitter {
    buf: String,
}

impl SentenceSplitter {
    /// Characters before a comma may end a chunk.
    const COMMA_AFTER: usize = 50;

    /// Add streamed text; returns every sentence completed by it.
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        self.buf.push_str(delta);
        let mut out = Vec::new();
        while let Some(end) = self.boundary() {
            let rest = self.buf.split_off(end);
            let sentence = std::mem::replace(&mut self.buf, rest);
            let sentence = sentence.trim();
            if !sentence.is_empty() {
                out.push(sentence.to_string());
            }
        }
        out
    }

    /// The unfinished remainder, once the stream has ended.
    pub fn finish(&mut self) -> Option<String> {
        let rest = std::mem::take(&mut self.buf);
        let rest = rest.trim();
        (!rest.is_empty()).then(|| rest.to_string())
    }

    /// Byte offset just past the first complete sentence in the buffer, if any.
    fn boundary(&self) -> Option<usize> {
        let mut chars = self.buf.char_indices().peekable();
        while let Some((i, c)) = chars.next() {
            let Some(&(_, next)) = chars.peek() else {
                break; // the terminator is last: wait to see what follows
            };
            let sentence_end = matches!(c, '.' | '!' | '?') && next.is_whitespace();
            let clause_end = matches!(c, ',' | ';' | ':')
                && next.is_whitespace()
                && self.buf[..i].chars().count() >= Self::COMMA_AFTER;
            if sentence_end || clause_end {
                return Some(i + c.len_utf8());
            }
        }
        None
    }
}

/// How to reach a Kokoro text-to-speech server.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KokoroConfig {
    /// Full URL of the speech endpoint, e.g. `http://127.0.0.1:8080/v1/audio/speech`.
    pub url: String,
    /// Model name the server expects, usually `kokoro`.
    pub model: String,
    /// Voice name, e.g. `af_heart`.
    pub voice: String,
}

/// What to speak with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechConfig {
    /// Kokoro server to use first; `None` always uses the system voice.
    pub kokoro: Option<KokoroConfig>,
    /// Voice for the system engine (`espeak-ng` voice such as `en`, or a macOS `say` voice).
    pub system_voice: String,
}

impl SpeechConfig {
    /// Speak with the system voice only.
    #[must_use]
    pub fn system(voice: impl Into<String>) -> Self {
        Self {
            kokoro: None,
            system_voice: voice.into(),
        }
    }
}

#[derive(Serialize)]
struct SpeechRequest<'a> {
    model: &'a str,
    input: &'a str,
    voice: &'a str,
    response_format: &'a str,
}

/// Ask a Kokoro server for the audio (WAV) of `text`.
pub(crate) async fn fetch_audio(
    client: &reqwest::Client,
    cfg: &KokoroConfig,
    text: &str,
) -> Result<Vec<u8>> {
    let response = client
        .post(&cfg.url)
        .timeout(Duration::from_secs(60))
        .json(&SpeechRequest {
            model: &cfg.model,
            input: text,
            voice: &cfg.voice,
            response_format: "wav",
        })
        .send()
        .await?
        .error_for_status()?;
    let bytes = response.bytes().await?;
    if bytes.is_empty() {
        bail!("the speech server sent no audio");
    }
    Ok(bytes.to_vec())
}

/// Look for an executable on `PATH`.
///
/// # Examples
///
/// ```
/// use voice_notes::speech::find_in_path;
///
/// assert!(find_in_path("sh").is_some());
/// assert!(find_in_path("definitely-not-a-real-program-xyz").is_none());
/// ```
#[must_use]
pub fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| candidate.is_file())
}

/// A command that plays the WAV file at `path`, using what the platform provides.
fn player_command(path: &Path) -> Result<Command> {
    let (program, args): (&str, &[&str]) = if cfg!(target_os = "macos") {
        ("afplay", &[])
    } else {
        // PipeWire/PulseAudio first, so playback follows the desktop's output device.
        [
            ("paplay", &[][..]),
            ("pw-play", &[][..]),
            ("aplay", &["-q"][..]),
            (
                "ffplay",
                &["-nodisp", "-autoexit", "-loglevel", "quiet"][..],
            ),
        ]
        .into_iter()
        .find(|(name, _)| find_in_path(name).is_some())
        .ok_or_else(|| {
            anyhow!("no audio player found; install pulseaudio-utils, pipewire or alsa-utils")
        })?
    };
    let mut command = Command::new(program);
    command.args(args).arg(path);
    Ok(command)
}

/// The command that speaks text read from stdin with the platform's own voice.
fn system_command(voice: &str) -> Command {
    if cfg!(target_os = "macos") {
        let mut command = Command::new("say");
        if voice != "en" {
            command.args(["-v", voice]); // `en` is the espeak default, not a macOS voice
        }
        command
    } else {
        let mut command = Command::new("espeak-ng");
        command.args(["-v", voice, "-s", "165", "--stdin"]);
        command
    }
}

fn system_voice_missing() -> anyhow::Error {
    if cfg!(target_os = "macos") {
        anyhow!("`say` not found, so there is no system voice")
    } else {
        anyhow!("espeak-ng not found; install it with: sudo apt install espeak-ng")
    }
}

/// Something waiting to be spoken.
enum Item {
    /// Speak with the system voice.
    Text(String),
    /// Kokoro audio being fetched in the background; `text` is kept for the fallback.
    Audio {
        text: String,
        ready: Receiver<Result<PathBuf, String>>,
    },
}

/// What to do with the front of the queue.
#[derive(Debug, PartialEq, Eq)]
enum Next {
    /// Nothing queued.
    Idle,
    /// The front item is not ready yet.
    Wait,
    /// Speak this text with the system voice.
    System(String),
    /// Play this audio file.
    Play(PathBuf),
    /// Kokoro failed; speak `text` with the system voice instead.
    Fallback { text: String, reason: String },
}

/// Speaks text aloud, one utterance at a time, with audio for the next sentence prepared while
/// the current one plays.
pub struct Speaker {
    config: SpeechConfig,
    client: reqwest::Client,
    child: Option<Child>,
    playing: Option<PathBuf>,
    queue: VecDeque<Item>,
    counter: usize,
    /// Kokoro failed once, so everything now uses the system voice.
    fell_back: bool,
    notice: Option<String>,
}

impl Speaker {
    #[must_use]
    pub fn new(config: SpeechConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            child: None,
            playing: None,
            queue: VecDeque::new(),
            counter: 0,
            fell_back: false,
            notice: None,
        }
    }

    /// Speak `text` now, interrupting anything already being said. Returns immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if no voice or audio player can be started.
    pub fn speak(&mut self, text: &str) -> Result<()> {
        self.stop();
        self.enqueue(text)
    }

    /// Speak `text` after whatever is already queued, starting at once if nothing is playing.
    ///
    /// # Errors
    ///
    /// Returns an error if no voice or audio player can be started.
    pub fn enqueue(&mut self, text: &str) -> Result<()> {
        let text = speakable(text);
        if !text.is_empty() {
            let item = self.prepare(text);
            self.queue.push_back(item);
        }
        self.pump()
    }

    /// Start fetching Kokoro audio for `text` when possible, else queue it for the system voice.
    fn prepare(&mut self, text: String) -> Item {
        let kokoro = self.config.kokoro.clone().filter(|_| !self.fell_back);
        match (kokoro, tokio::runtime::Handle::try_current()) {
            (Some(cfg), Ok(runtime)) => {
                self.counter += 1;
                let path = std::env::temp_dir().join(format!(
                    "voice-notes-{}-{}.wav",
                    std::process::id(),
                    self.counter
                ));
                let (tx, ready) = mpsc::channel();
                let client = self.client.clone();
                let spoken = text.clone();
                runtime.spawn(async move {
                    let result = match fetch_audio(&client, &cfg, &spoken).await {
                        Ok(bytes) => std::fs::write(&path, bytes)
                            .map(|()| path.clone())
                            .map_err(|e| e.to_string()),
                        Err(e) => Err(format!("{e:#}")),
                    };
                    let wrote = result.is_ok();
                    if tx.send(result).is_err() && wrote {
                        let _ = std::fs::remove_file(&path); // nobody is waiting for it any more
                    }
                });
                Item::Audio { text, ready }
            }
            _ => Item::Text(text),
        }
    }

    /// Start the next queued utterance once the current one has finished. Call regularly.
    ///
    /// # Errors
    ///
    /// Returns an error if no voice or audio player can be started.
    pub fn pump(&mut self) -> Result<()> {
        if self.is_speaking() {
            return Ok(());
        }
        match self.poll_front() {
            Next::Idle | Next::Wait => Ok(()),
            Next::System(text) => self.start_system(&text),
            Next::Play(path) => self.start_player(path),
            Next::Fallback { text, reason } => {
                if !self.fell_back {
                    self.fell_back = true;
                    self.notice = Some(format!(
                        "Kokoro unavailable ({reason}); using the system voice"
                    ));
                }
                self.start_system(&text)
            }
        }
    }

    /// Look at the front of the queue and take it off if it is ready.
    fn poll_front(&mut self) -> Next {
        let polled = match self.queue.front() {
            None => return Next::Idle,
            Some(Item::Text(_)) => None,
            Some(Item::Audio { ready, .. }) => Some(ready.try_recv()),
        };
        match (self.queue.pop_front(), polled) {
            (Some(Item::Text(text)), _) => Next::System(text),
            (Some(Item::Audio { .. }), Some(Ok(Ok(path)))) => Next::Play(path),
            (Some(Item::Audio { text, .. }), Some(Ok(Err(reason)))) => {
                Next::Fallback { text, reason }
            }
            (Some(Item::Audio { text, .. }), Some(Err(TryRecvError::Disconnected))) => {
                Next::Fallback {
                    text,
                    reason: "the request was lost".to_string(),
                }
            }
            (Some(item @ Item::Audio { .. }), _) => {
                self.queue.push_front(item); // not ready yet
                Next::Wait
            }
            (None, _) => Next::Idle,
        }
    }

    /// If Kokoro failed, a one-off message explaining the switch to the system voice.
    pub fn take_notice(&mut self) -> Option<String> {
        self.notice.take()
    }

    /// Whether something is playing or still waiting to be played.
    pub fn is_busy(&mut self) -> bool {
        self.is_speaking() || !self.queue.is_empty()
    }

    fn start_system(&mut self, text: &str) -> Result<()> {
        let mut child = system_command(&self.config.system_voice)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => {
                    self.queue.clear();
                    system_voice_missing()
                }
                _ => anyhow!(e).context("could not start the system voice"),
            })?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(text.as_bytes())
                .context("could not send text to the system voice")?;
        }
        self.child = Some(child);
        Ok(())
    }

    fn start_player(&mut self, path: PathBuf) -> Result<()> {
        let spawned = player_command(&path).and_then(|mut command| {
            command
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .context("could not start the audio player")
        });
        match spawned {
            Ok(child) => {
                self.child = Some(child);
                self.playing = Some(path);
                Ok(())
            }
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                self.queue.clear();
                Err(e)
            }
        }
    }

    /// Whether speech is still playing.
    pub fn is_speaking(&mut self) -> bool {
        match self.child.as_mut().map(Child::try_wait) {
            Some(Ok(None)) => true,
            Some(_) => {
                self.child = None;
                self.remove_played_file();
                false
            }
            None => false,
        }
    }

    fn remove_played_file(&mut self) {
        if let Some(path) = self.playing.take() {
            let _ = std::fs::remove_file(path);
        }
    }

    /// Cut speech off and forget anything queued.
    pub fn stop(&mut self) {
        self.queue.clear();
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.remove_played_file();
    }
}

impl Drop for Speaker {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Code blocks, markdown marks, link targets and bare URLs are removed.
    #[test]
    fn cleans_markdown_for_speech() {
        assert_eq!(
            speakable("# Title\n- **bold** and `code`"),
            "Title - bold and code"
        );
        assert_eq!(speakable("see https://example.com now"), "see now");
        assert_eq!(speakable("```\nonly code\n```"), "");
        assert_eq!(speakable("[a](b) [c](d)"), "a c");
        assert_eq!(speakable("a [not a link"), "a [not a link");
    }

    /// Long text is cut at a word boundary.
    #[test]
    fn truncates_long_text() {
        let long = "word ".repeat(1000);
        let out = speakable(&long);
        assert!(out.chars().count() <= MAX_SPOKEN_CHARS);
        assert!(out.ends_with("word"));
    }

    /// A speaker that never spoke is not speaking, and stopping is harmless.
    #[test]
    fn idle_speaker() {
        let mut s = Speaker::new(SpeechConfig::system("en"));
        assert!(!s.is_speaking());
        s.stop();
        assert!(s.speak("   ").is_ok()); // nothing speakable, so no process needed
        assert!(!s.is_speaking());
    }

    /// Streamed text is cut into whole sentences, never mid-sentence or mid-number.
    #[test]
    fn splits_sentences_as_text_streams() {
        let mut sp = SentenceSplitter::default();
        assert_eq!(sp.push("The value is 3"), [] as [std::string::String; 0]);
        assert_eq!(sp.push(".5 today"), [] as [std::string::String; 0]); // a decimal point is not a sentence end
        assert_eq!(
            sp.push(". Next one! And"),
            vec!["The value is 3.5 today.", "Next one!"]
        );
        assert_eq!(sp.finish().as_deref(), Some("And"));
        assert_eq!(sp.finish(), None);
    }

    /// A terminator at the very end waits for the next character before cutting.
    #[test]
    fn waits_after_a_trailing_terminator() {
        let mut sp = SentenceSplitter::default();
        assert_eq!(sp.push("Done."), [] as [std::string::String; 0]);
        assert_eq!(sp.push(" Then"), vec!["Done."]);
    }

    /// Long sentences are cut at a comma so speech can start early; short ones are not.
    #[test]
    fn cuts_long_sentences_at_commas() {
        let mut sp = SentenceSplitter::default();
        assert_eq!(sp.push("Yes, I think so, "), [] as [std::string::String; 0]);
        let long = "This is a rather long opening clause that goes on, and then continues";
        assert_eq!(sp.push(long).len(), 1);
    }

    /// Queued speech is dropped by `stop`, and an empty queue means not busy.
    #[test]
    fn queue_is_cleared_on_stop() {
        let mut s = Speaker::new(SpeechConfig::system("en"));
        assert!(!s.is_busy());
        s.queue.push_back(Item::Text("pending".into()));
        assert!(s.is_busy());
        s.stop();
        assert!(!s.is_busy());
    }

    /// The request sent to Kokoro asks for WAV audio with the configured model and voice.
    #[test]
    fn speech_request_shape() {
        let body = serde_json::to_value(SpeechRequest {
            model: "kokoro",
            input: "Hi.",
            voice: "af_heart",
            response_format: "wav",
        })
        .unwrap();
        assert_eq!(body["model"], "kokoro");
        assert_eq!(body["input"], "Hi.");
        assert_eq!(body["voice"], "af_heart");
        assert_eq!(body["response_format"], "wav");
    }

    fn audio_item(
        result: Option<Result<PathBuf, String>>,
    ) -> (Item, mpsc::Sender<Result<PathBuf, String>>) {
        let (tx, ready) = mpsc::channel();
        if let Some(r) = result {
            tx.send(r).unwrap();
        }
        (
            Item::Audio {
                text: "hello".into(),
                ready,
            },
            tx,
        )
    }

    /// Ready audio is played, failed audio falls back to the system voice, and audio that is
    /// still being fetched keeps its place at the front of the queue.
    #[test]
    fn queue_front_states() {
        let mut s = Speaker::new(SpeechConfig::system("en"));
        assert_eq!(s.poll_front(), Next::Idle);

        let (item, _keep) = audio_item(None);
        s.queue.push_back(item);
        assert_eq!(s.poll_front(), Next::Wait);
        assert_eq!(s.queue.len(), 1);

        let (item, _keep) = audio_item(Some(Ok(PathBuf::from("/tmp/a.wav"))));
        s.queue.clear();
        s.queue.push_back(item);
        assert_eq!(s.poll_front(), Next::Play(PathBuf::from("/tmp/a.wav")));

        let (item, _keep) = audio_item(Some(Err("boom".into())));
        s.queue.push_back(item);
        assert_eq!(
            s.poll_front(),
            Next::Fallback {
                text: "hello".into(),
                reason: "boom".into()
            }
        );

        s.queue.push_back(Item::Text("plain".into()));
        assert_eq!(s.poll_front(), Next::System("plain".into()));
    }

    /// A lost request (sender dropped) is treated as a failure, not a hang.
    #[test]
    fn dropped_request_falls_back() {
        let mut s = Speaker::new(SpeechConfig::system("en"));
        let (item, tx) = audio_item(None);
        drop(tx);
        s.queue.push_back(item);
        assert!(matches!(s.poll_front(), Next::Fallback { .. }));
    }

    /// Without a Kokoro config, or outside an async runtime, text goes to the system voice.
    #[test]
    fn plain_config_uses_system_voice() {
        let mut s = Speaker::new(SpeechConfig::system("en"));
        assert!(matches!(s.prepare("hi".into()), Item::Text(_)));
        let kokoro = SpeechConfig {
            kokoro: Some(KokoroConfig {
                url: "http://127.0.0.1:1/x".into(),
                model: "kokoro".into(),
                voice: "v".into(),
            }),
            system_voice: "en".into(),
        };
        let mut s = Speaker::new(kokoro);
        assert!(matches!(s.prepare("hi".into()), Item::Text(_))); // no runtime here
    }

    /// Serve one canned HTTP response on a local port and return its URL.
    async fn serve_once(status: &'static str, body: &'static [u8]) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/audio/speech", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let head = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nContent-Type: audio/wav\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(head.as_bytes()).await;
            let _ = socket.write_all(body).await;
        });
        url
    }

    fn cfg(url: String) -> KokoroConfig {
        KokoroConfig {
            url,
            model: "kokoro".into(),
            voice: "af_heart".into(),
        }
    }

    /// Audio bytes come back from a working server; errors and empty answers are failures.
    #[tokio::test]
    async fn fetches_audio_from_server() {
        let client = reqwest::Client::new();
        let url = serve_once("200 OK", b"RIFFfakewav").await;
        assert_eq!(
            fetch_audio(&client, &cfg(url), "hi").await.unwrap(),
            b"RIFFfakewav"
        );

        let url = serve_once("500 Internal Server Error", b"nope").await;
        assert!(fetch_audio(&client, &cfg(url), "hi").await.is_err());

        let url = serve_once("200 OK", b"").await;
        assert!(fetch_audio(&client, &cfg(url), "hi").await.is_err());
    }

    /// An unreachable server is reported as an error, never a hang.
    #[tokio::test]
    async fn unreachable_server_is_an_error() {
        let client = reqwest::Client::new();
        let result = fetch_audio(
            &client,
            &cfg("http://127.0.0.1:1/v1/audio/speech".into()),
            "hi",
        )
        .await;
        assert!(result.is_err());
    }

    /// `PATH` lookup finds a standard tool and rejects nonsense.
    #[test]
    fn finds_programs_in_path() {
        assert!(find_in_path("sh").is_some());
        assert!(find_in_path("definitely-not-a-real-program-xyz").is_none());
    }
}
