//! One full voice turn without a microphone, so it can run in CI or a script.
//!
//! A recording stands in for the microphone, the turn runs end to end (transcribe, ask the
//! chat backend, speak the reply), and the result is printed as text or JSON. Each stage that
//! can fail has its own exit code, so a failing run says which part broke.

use crate::audio::normalize_wav;
use crate::chat::{Conversation, ReplyDelta};
use crate::pipeline::Settings;
use crate::speech::{Speaker, fetch_audio, speakable};
use crate::transcribe::{TranscribeOptions, is_hallucination, transcribe_audio};
use serde_json::json;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// What to do in one turn.
#[derive(Debug, Clone)]
pub struct TurnOptions {
    /// The recording that stands in for the microphone.
    pub wav: PathBuf,
    /// Do not play the reply; still ask the speech service for it and check that audio came
    /// back.
    pub no_audio_output: bool,
}

/// The part of a turn that failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Reading the recording.
    Input,
    /// Speech to text.
    Transcribe,
    /// Asking the chat backend.
    Chat,
    /// Text to speech.
    Speech,
}

impl Stage {
    /// The process exit code for a failure in this stage: 10, 11, 12 or 13.
    #[must_use]
    pub fn exit_code(self) -> i32 {
        match self {
            Self::Input => 10,
            Self::Transcribe => 11,
            Self::Chat => 12,
            Self::Speech => 13,
        }
    }

    /// A short name for messages.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Input => "reading the recording",
            Self::Transcribe => "transcribing",
            Self::Chat => "asking the chat model",
            Self::Speech => "speaking the reply",
        }
    }
}

/// A turn that failed, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnError {
    pub stage: Stage,
    pub message: String,
}

impl std::fmt::Display for TurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "failed while {}: {}", self.stage.name(), self.message)
    }
}

impl std::error::Error for TurnError {}

fn fail<E: std::fmt::Display>(stage: Stage) -> impl FnOnce(E) -> TurnError {
    move |e| TurnError {
        stage,
        message: format!("{e:#}"),
    }
}

/// What happened to the spoken reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Speech {
    /// Audio came back from the speech service and was thrown away (`--no-audio-output`).
    Discarded { bytes: usize },
    /// The reply was played aloud.
    Played,
    /// Nothing was spoken, for the stated reason.
    Skipped(String),
}

/// How long each stage took.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timings {
    pub transcribe: Duration,
    pub chat: Duration,
    pub speech: Duration,
}

/// The outcome of a successful turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnReport {
    pub transcript: String,
    pub reply: String,
    /// How many pieces the reply arrived in. More than one means it was streamed.
    pub reply_chunks: usize,
    pub speech: Speech,
    pub timings: Timings,
    pub profile: String,
    pub backend: String,
}

fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

impl TurnReport {
    /// The report as one JSON object.
    #[must_use]
    pub fn to_json(&self) -> String {
        let (speech, bytes) = match &self.speech {
            Speech::Discarded { bytes } => ("discarded", Some(*bytes)),
            Speech::Played => ("played", None),
            Speech::Skipped(_) => ("skipped", None),
        };
        let reason = match &self.speech {
            Speech::Skipped(why) => Some(why.as_str()),
            _ => None,
        };
        json!({
            "transcript": self.transcript,
            "reply": self.reply,
            "reply_chunks": self.reply_chunks,
            "speech": speech,
            "audio_bytes": bytes,
            "speech_skipped_because": reason,
            "profile": self.profile,
            "backend": self.backend,
            "timings_ms": {
                "transcribe": millis(self.timings.transcribe),
                "chat": millis(self.timings.chat),
                "speech": millis(self.timings.speech),
            },
        })
        .to_string()
    }

    /// The report as plain text: transcript, reply and what happened to the speech.
    #[must_use]
    pub fn to_text(&self) -> String {
        let speech = match &self.speech {
            Speech::Discarded { bytes } => format!("{bytes} bytes of audio received and discarded"),
            Speech::Played => "played".to_string(),
            Speech::Skipped(why) => format!("skipped ({why})"),
        };
        format!(
            "Transcript: {}\nReply: {}\nSpeech: {speech}\n",
            self.transcript, self.reply
        )
    }
}

/// Run one turn: read the recording, transcribe it, ask the chat backend, speak the reply.
///
/// # Errors
///
/// Returns a [`TurnError`] naming the first stage that failed: an unreadable recording, a
/// transcription that fails or finds no speech, a chat backend that does not answer, or a speech
/// service that returns no audio.
pub async fn run_turn(
    client: &reqwest::Client,
    settings: &Settings,
    opts: &TurnOptions,
) -> Result<TurnReport, TurnError> {
    let mut timings = Timings::default();

    let data = std::fs::read(&opts.wav).map_err(|e| TurnError {
        stage: Stage::Input,
        message: format!("could not read {}: {e}", opts.wav.display()),
    })?;
    let wav = normalize_wav(&data).map_err(fail(Stage::Input))?;

    let start = Instant::now();
    let transcribe_opts = TranscribeOptions {
        language: settings.language.clone(),
        prompt: None,
        url: settings.endpoints.stt.clone(),
        model: settings.stt_model.clone(),
    };
    let transcript = transcribe_audio(client, &wav, &transcribe_opts)
        .await
        .map_err(fail(Stage::Transcribe))?
        .trim()
        .to_string();
    timings.transcribe = start.elapsed();
    if is_hallucination(&transcript) {
        return Err(TurnError {
            stage: Stage::Transcribe,
            message: if transcript.is_empty() {
                "no speech was recognised".to_string()
            } else {
                format!("no speech was recognised (the model returned {transcript:?})")
            },
        });
    }

    let start = Instant::now();
    let mut conversation = Conversation::new(settings.backend.clone())
        .with_url(settings.endpoints.chat.clone())
        .with_prompt(settings.system_prompt.clone());
    let (pieces_tx, mut pieces_rx) = tokio::sync::mpsc::unbounded_channel();
    let reply = conversation
        .reply(client, &transcript, Some(&pieces_tx))
        .await
        .map_err(fail(Stage::Chat))?;
    timings.chat = start.elapsed();
    let mut reply_chunks = 0;
    while let Ok(piece) = pieces_rx.try_recv() {
        if matches!(piece, ReplyDelta::Text(_)) {
            reply_chunks += 1;
        }
    }

    let start = Instant::now();
    let speech = speak(client, settings, &reply, opts.no_audio_output).await?;
    timings.speech = start.elapsed();

    Ok(TurnReport {
        transcript,
        reply,
        reply_chunks,
        speech,
        timings,
        profile: settings.profile.clone(),
        backend: settings.backend.label(),
    })
}

/// The speech stage. With `no_audio_output`, ask the Kokoro service for the audio and check that
/// some came back; otherwise play the reply and wait until it has finished.
async fn speak(
    client: &reqwest::Client,
    settings: &Settings,
    reply: &str,
    no_audio_output: bool,
) -> Result<Speech, TurnError> {
    if !settings.speak {
        return Ok(Speech::Skipped("speech is turned off".to_string()));
    }
    let text = speakable(reply);
    if text.is_empty() {
        return Ok(Speech::Skipped(
            "the reply has nothing to say aloud".to_string(),
        ));
    }
    match (&settings.speech.kokoro, no_audio_output) {
        (Some(kokoro), true) => {
            let audio = fetch_audio(client, kokoro, &text)
                .await
                .map_err(fail(Stage::Speech))?;
            Ok(Speech::Discarded { bytes: audio.len() })
        }
        (None, true) => Ok(Speech::Skipped(
            "the system voice cannot be captured; use Kokoro to test speech output".to_string(),
        )),
        (_, false) => {
            let mut speaker = Speaker::new(settings.speech.clone());
            speaker.speak(&text).map_err(fail(Stage::Speech))?;
            while speaker.is_busy() {
                speaker.pump().map_err(fail(Stage::Speech))?;
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            Ok(Speech::Played)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::Backend;
    use crate::endpoints::Endpoints;
    use crate::speech::{KokoroConfig, SpeechConfig};
    use crate::test_support::capture_once;

    const SSE: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"Four\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\" it is.\"}}]}\n\ndata: [DONE]\n\n";
    const DEAD: &str = "http://127.0.0.1:1";

    /// A short recording written to a file that is removed when the test ends.
    struct Clip(PathBuf);

    impl Clip {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "voice-notes-test-{}-{name}.wav",
                std::process::id()
            ));
            std::fs::write(&path, crate::latency::test_clip()).unwrap();
            Self(path)
        }
    }

    impl Drop for Clip {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Settings pointing at the given base URLs for speech-to-text, chat and speech.
    fn settings(stt: &str, chat: &str, tts: Option<&str>) -> Settings {
        Settings {
            clean: false,
            min_threshold: 0.01,
            language: Some("en".into()),
            verbose: false,
            endpoints: Endpoints {
                stt: format!("{stt}/stt"),
                chat: format!("{chat}/chat"),
                models: format!("{chat}/models"),
                tts: tts.map_or_else(|| format!("{DEAD}/tts"), |t| format!("{t}/tts")),
            },
            stt_model: "test-whisper".into(),
            clean_model: "test-clean".into(),
            backend: Backend::Local {
                model: "test-chat".into(),
                think: false,
            },
            system_prompt: None,
            profile: "fast".into(),
            profiles: vec!["fast".into()],
            speak: true,
            speech: SpeechConfig {
                kokoro: tts.map(|t| KokoroConfig {
                    url: format!("{t}/tts"),
                    model: "kokoro".into(),
                    voice: "af_heart".into(),
                }),
                system_voice: "en".into(),
            },
            talk: false,
        }
    }

    fn options(clip: &Clip) -> TurnOptions {
        TurnOptions {
            wav: clip.0.clone(),
            no_audio_output: true,
        }
    }

    /// Start mock services for speech-to-text, chat and speech, each answering once.
    async fn services(transcript: &str) -> (String, String, String) {
        let (stt, _) =
            capture_once("application/json", format!(r#"{{"text":"{transcript}"}}"#)).await;
        let (chat, _) = capture_once("text/event-stream", SSE).await;
        let (tts, _) = capture_once("audio/wav", "RIFFfake-audio-bytes").await;
        (stt, chat, tts)
    }

    /// A whole turn: the recording is transcribed, the chat backend answers, and the speech
    /// service's audio is received and discarded.
    #[tokio::test]
    async fn full_turn_succeeds() {
        let clip = Clip::new("full");
        let (stt, chat, tts) = services("what is two plus two").await;
        let report = run_turn(
            &reqwest::Client::new(),
            &settings(&stt, &chat, Some(&tts)),
            &options(&clip),
        )
        .await
        .unwrap();
        assert_eq!(report.transcript, "what is two plus two");
        assert_eq!(report.reply, "Four it is.");
        assert_eq!(
            report.reply_chunks, 2,
            "the mock server sends the reply in two pieces"
        );
        assert_eq!(
            report.speech,
            Speech::Discarded {
                bytes: "RIFFfake-audio-bytes".len()
            }
        );
        assert_eq!(
            (report.profile.as_str(), report.backend.as_str()),
            ("fast", "test-chat")
        );
    }

    /// The request to each service carries the configured model and the recording itself.
    #[tokio::test]
    async fn requests_carry_the_recording_and_model_names() {
        let clip = Clip::new("requests");
        let (stt, stt_request) =
            capture_once("application/json", r#"{"text":"hello there"}"#).await;
        let (chat, chat_request) = capture_once("text/event-stream", SSE).await;
        let (tts, tts_request) = capture_once("audio/wav", "RIFFaudio").await;
        run_turn(
            &reqwest::Client::new(),
            &settings(&stt, &chat, Some(&tts)),
            &options(&clip),
        )
        .await
        .unwrap();

        let sent = stt_request.await.unwrap();
        assert!(
            sent.starts_with("POST /stt ")
                && sent.contains("test-whisper")
                && sent.contains("RIFF"),
            "{sent}"
        );
        let sent = chat_request.await.unwrap();
        assert!(
            sent.starts_with("POST /chat ")
                && sent.contains("test-chat")
                && sent.contains("hello there"),
            "{sent}"
        );
        let sent = tts_request.await.unwrap();
        assert!(
            sent.starts_with("POST /tts ") && sent.contains("Four it is."),
            "{sent}"
        );
    }

    /// The text report has the transcript, the reply and what happened to the speech.
    #[tokio::test]
    async fn text_report() {
        let clip = Clip::new("text");
        let (stt, chat, tts) = services("hello there").await;
        let report = run_turn(
            &reqwest::Client::new(),
            &settings(&stt, &chat, Some(&tts)),
            &options(&clip),
        )
        .await
        .unwrap();
        let text = report.to_text();
        assert!(
            text.contains("Transcript: hello there\nReply: Four it is.\n"),
            "{text}"
        );
        assert!(
            text.contains("Speech: 20 bytes of audio received and discarded"),
            "{text}"
        );
    }

    /// The JSON report is valid and has every field a script needs.
    #[tokio::test]
    async fn json_report() {
        let clip = Clip::new("json");
        let (stt, chat, tts) = services("hello there").await;
        let report = run_turn(
            &reqwest::Client::new(),
            &settings(&stt, &chat, Some(&tts)),
            &options(&clip),
        )
        .await
        .unwrap();
        let value: serde_json::Value = serde_json::from_str(&report.to_json()).unwrap();
        assert_eq!(value["transcript"], "hello there");
        assert_eq!(value["reply"], "Four it is.");
        assert_eq!(value["reply_chunks"], 2);
        assert_eq!(value["speech"], "discarded");
        assert_eq!(value["audio_bytes"], 20);
        assert_eq!(value["profile"], "fast");
        for stage in ["transcribe", "chat", "speech"] {
            assert!(value["timings_ms"][stage].is_u64(), "{stage}");
        }
        assert!(value["speech_skipped_because"].is_null());
    }

    /// An unreadable or missing recording fails the input stage with exit code 10.
    #[tokio::test]
    async fn bad_recordings_fail_the_input_stage() {
        let missing = TurnOptions {
            wav: PathBuf::from("/definitely/not/here.wav"),
            no_audio_output: true,
        };
        let err = run_turn(
            &reqwest::Client::new(),
            &settings(DEAD, DEAD, None),
            &missing,
        )
        .await
        .unwrap_err();
        assert_eq!(err.stage, Stage::Input);
        assert_eq!(err.stage.exit_code(), 10);
        assert!(err.message.contains("could not read"), "{err}");

        let junk =
            std::env::temp_dir().join(format!("voice-notes-test-{}-junk.wav", std::process::id()));
        std::fs::write(&junk, b"this is not audio").unwrap();
        let err = run_turn(
            &reqwest::Client::new(),
            &settings(DEAD, DEAD, None),
            &TurnOptions {
                wav: junk.clone(),
                no_audio_output: true,
            },
        )
        .await
        .unwrap_err();
        let _ = std::fs::remove_file(&junk);
        assert_eq!(err.stage, Stage::Input);
        assert!(err.message.contains("WAV"), "{err}");
    }

    /// A speech-to-text service that is down fails the transcribe stage with exit code 11.
    #[tokio::test]
    async fn dead_transcriber_fails_stage_11() {
        let clip = Clip::new("stt-down");
        let (chat, _) = capture_once("text/event-stream", SSE).await;
        let err = run_turn(
            &reqwest::Client::new(),
            &settings(DEAD, &chat, None),
            &options(&clip),
        )
        .await
        .unwrap_err();
        assert_eq!((err.stage, err.stage.exit_code()), (Stage::Transcribe, 11));
    }

    /// A recording in which nothing is recognised fails the transcribe stage, whether the
    /// answer is empty or one of Whisper's stock phrases for silence.
    #[tokio::test]
    async fn no_speech_fails_stage_11() {
        for answer in ["", "you", "Thank you."] {
            let clip = Clip::new("silence");
            let (stt, chat, _) = services(answer).await;
            let err = run_turn(
                &reqwest::Client::new(),
                &settings(&stt, &chat, None),
                &options(&clip),
            )
            .await
            .unwrap_err();
            assert_eq!(err.stage, Stage::Transcribe, "{answer:?}");
            assert!(err.message.contains("no speech was recognised"), "{err}");
        }
    }

    /// A chat backend that is down fails the chat stage with exit code 12.
    #[tokio::test]
    async fn dead_chat_fails_stage_12() {
        let clip = Clip::new("chat-down");
        let (stt, _) = capture_once("application/json", r#"{"text":"hello there"}"#).await;
        let err = run_turn(
            &reqwest::Client::new(),
            &settings(&stt, DEAD, None),
            &options(&clip),
        )
        .await
        .unwrap_err();
        assert_eq!((err.stage, err.stage.exit_code()), (Stage::Chat, 12));
    }

    /// A speech service that is down, or returns nothing, fails the speech stage with exit
    /// code 13. There is no quiet fallback in a test run.
    #[tokio::test]
    async fn dead_or_empty_speech_fails_stage_13() {
        let clip = Clip::new("tts-down");
        let (stt, chat, _) = services("hello there").await;
        let err = run_turn(
            &reqwest::Client::new(),
            &settings(&stt, &chat, Some(DEAD)),
            &options(&clip),
        )
        .await
        .unwrap_err();
        assert_eq!((err.stage, err.stage.exit_code()), (Stage::Speech, 13));

        let (stt, _) = capture_once("application/json", r#"{"text":"hello there"}"#).await;
        let (chat, _) = capture_once("text/event-stream", SSE).await;
        let (tts, _) = capture_once("audio/wav", "").await;
        let err = run_turn(
            &reqwest::Client::new(),
            &settings(&stt, &chat, Some(&tts)),
            &options(&clip),
        )
        .await
        .unwrap_err();
        assert_eq!(err.stage, Stage::Speech, "an empty answer is not audio");
    }

    /// With speech turned off, or only a system voice that cannot be captured, the speech stage
    /// is skipped with a reason and the speech service is never asked.
    #[tokio::test]
    async fn speech_can_be_skipped() {
        let clip = Clip::new("skip");
        let (stt, chat, _) = services("hello there").await;
        let mut quiet = settings(&stt, &chat, Some(DEAD));
        quiet.speak = false;
        let report = run_turn(&reqwest::Client::new(), &quiet, &options(&clip))
            .await
            .unwrap();
        assert_eq!(
            report.speech,
            Speech::Skipped("speech is turned off".into())
        );
        let value: serde_json::Value = serde_json::from_str(&report.to_json()).unwrap();
        assert_eq!(value["speech"], "skipped");
        assert_eq!(value["speech_skipped_because"], "speech is turned off");

        let (stt, chat, _) = services("hello there").await;
        let report = run_turn(
            &reqwest::Client::new(),
            &settings(&stt, &chat, None),
            &options(&clip),
        )
        .await
        .unwrap();
        assert!(
            matches!(report.speech, Speech::Skipped(ref why) if why.contains("system voice")),
            "{report:?}"
        );
    }

    /// Each stage has its own exit code, and the error message says which stage failed.
    #[test]
    fn exit_codes_and_messages() {
        let codes: Vec<i32> = [Stage::Input, Stage::Transcribe, Stage::Chat, Stage::Speech]
            .map(Stage::exit_code)
            .to_vec();
        assert_eq!(codes, [10, 11, 12, 13]);
        let err = TurnError {
            stage: Stage::Chat,
            message: "connection refused".into(),
        };
        assert_eq!(
            err.to_string(),
            "failed while asking the chat model: connection refused"
        );
    }
}
