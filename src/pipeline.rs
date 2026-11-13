//! Shared plumbing between the interfaces: settings and the ordered transcription worker.

use crate::chat::Backend;
use crate::speech::SpeechConfig;
use crate::transcribe::{TranscribeOptions, is_hallucination, transcribe_audio};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

/// Options chosen on the command line.
// Each bool is an independent command-line switch, not a state machine.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug)]
pub struct Settings {
    /// Polish the final transcript with the chat model.
    pub clean: bool,
    /// Minimum speech level; the real gate is calibrated from ambient noise.
    pub min_threshold: f32,
    /// Spoken language code, or `None` to let Whisper detect it.
    pub language: Option<String>,
    /// Print the live input level (plain mode only).
    pub verbose: bool,
    /// Who answers in talk mode.
    pub backend: Backend,
    /// System prompt for the chat backend; `None` means the default spoken style.
    pub system_prompt: Option<String>,
    /// The profile in use.
    pub profile: String,
    /// Every profile that can be switched to, in the order `p` cycles them.
    pub profiles: Vec<String>,
    /// Speak replies aloud.
    pub speak: bool,
    /// How replies are spoken.
    pub speech: SpeechConfig,
    /// Start in talk mode instead of notes mode.
    pub talk: bool,
}

/// One utterance waiting to be transcribed.
pub struct Job {
    pub id: usize,
    /// 16 kHz mono 16-bit WAV bytes.
    pub wav: Vec<u8>,
}

/// Result of transcribing one [`Job`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Accepted text.
    Text { id: usize, text: String },
    /// Text rejected as a likely hallucination (may be empty).
    Dropped { id: usize, text: String },
    /// The request failed.
    Failed { id: usize, error: String },
}

impl Outcome {
    /// The id of the job this outcome belongs to.
    #[must_use]
    pub fn id(&self) -> usize {
        match self {
            Self::Text { id, .. } | Self::Dropped { id, .. } | Self::Failed { id, .. } => *id,
        }
    }
}

/// Start a worker that transcribes jobs one at a time, so outcomes arrive in the order the
/// jobs were sent. The worker stops when the returned sender is dropped.
#[must_use]
pub fn spawn_transcriber(
    client: reqwest::Client,
    opts: TranscribeOptions,
) -> (UnboundedSender<Job>, UnboundedReceiver<Outcome>) {
    let (job_tx, mut job_rx) = unbounded_channel::<Job>();
    let (out_tx, out_rx) = unbounded_channel::<Outcome>();
    tokio::spawn(async move {
        while let Some(job) = job_rx.recv().await {
            let id = job.id;
            let outcome = match transcribe_audio(&client, &job.wav, &opts).await {
                Ok(text) => {
                    let text = text.trim().to_string();
                    if is_hallucination(&text) {
                        Outcome::Dropped { id, text }
                    } else {
                        Outcome::Text { id, text }
                    }
                }
                Err(e) => Outcome::Failed {
                    id,
                    error: format!("{e:#}"),
                },
            };
            if out_tx.send(outcome).is_err() {
                break;
            }
        }
    });
    (job_tx, out_rx)
}
