//! Timing the models you have, so setup can say whether they feel fast enough for conversation.

use crate::chat::{Backend, Conversation, ReplyDelta};
use crate::speech::{KokoroConfig, SentenceSplitter, fetch_audio};
use crate::transcribe::{TranscribeOptions, transcribe_audio};
use std::time::{Duration, Instant};

/// What the chat models are asked while being timed: a two-sentence answer.
const PROMPT: &str = "Tell me two short facts about octopuses.";
/// Length of the clip used to time speech-to-text.
const CLIP_SECONDS: u32 = 3;
const CHAT_TIMEOUT: Duration = Duration::from_secs(120);
const QUICK_TIMEOUT: Duration = Duration::from_secs(30);

/// How a delay feels in conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rating {
    /// Feels instant.
    Snappy,
    /// Noticeable but fine.
    Fine,
    /// Too slow to enjoy talking.
    Slow,
}

fn rate(d: Duration, snappy: Duration, fine: Duration) -> Rating {
    if d <= snappy {
        Rating::Snappy
    } else if d <= fine {
        Rating::Fine
    } else {
        Rating::Slow
    }
}

/// Time until the first sentence of a reply: up to 1.5 s is snappy, up to 4 s is fine.
#[must_use]
pub fn rate_chat(d: Duration) -> Rating {
    rate(d, Duration::from_millis(1500), Duration::from_secs(4))
}

/// Time to transcribe a short clip: up to 1 s is snappy, up to 3 s is fine.
#[must_use]
pub fn rate_stt(d: Duration) -> Rating {
    rate(d, Duration::from_secs(1), Duration::from_secs(3))
}

/// Time to synthesize one sentence: up to 1 s is snappy, up to 2.5 s is fine.
#[must_use]
pub fn rate_voice(d: Duration) -> Rating {
    rate(d, Duration::from_secs(1), Duration::from_millis(2500))
}

/// Time to the first sentence of a reply from one model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpeed {
    pub model: String,
    /// The first reply, which includes loading the model if it was not loaded yet.
    pub cold: Duration,
    /// The second reply, with the model loaded: what a conversation feels like.
    pub warm: Duration,
}

impl ModelSpeed {
    /// True if the first reply was much slower because the model had to load.
    #[must_use]
    pub fn loaded_on_demand(&self) -> bool {
        self.cold > self.warm * 2 && self.cold > self.warm + Duration::from_secs(2)
    }
}

/// Everything that was timed; each part is `None` if it was not available or failed.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Latency {
    /// Time to transcribe a short clip.
    pub stt: Option<Duration>,
    /// The chat model used for `fast` and `dev`.
    pub chat: Option<ModelSpeed>,
    /// The reasoning model behind `smart` and `think`, with thinking off.
    pub smart: Option<ModelSpeed>,
    /// Time to synthesize one sentence with Kokoro.
    pub voice: Option<Duration>,
}

/// What to time.
#[derive(Debug, Default, Clone)]
pub struct Targets<'a> {
    pub whisper: bool,
    pub chat_model: Option<&'a str>,
    pub smart_model: Option<&'a str>,
    pub kokoro_url: Option<&'a str>,
}

/// A few seconds of a steady tone as a 16 kHz mono WAV, to give Whisper something to chew on.
///
/// # Panics
///
/// Never in practice: writing a WAV to memory cannot fail.
#[must_use]
pub fn test_clip() -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut bytes = Vec::new();
    {
        let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut bytes), spec)
            .expect("an in-memory WAV header can always be written");
        for i in 0..16_000 * CLIP_SECONDS {
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            let sample = (f64::from(i) * 220.0 * std::f64::consts::TAU / 16_000.0).sin() * 6_000.0;
            #[allow(clippy::cast_possible_truncation)]
            writer
                .write_sample(sample as i16)
                .expect("an in-memory WAV sample can always be written");
        }
        writer
            .finalize()
            .expect("an in-memory WAV can always be finished");
    }
    bytes
}

async fn time_stt(client: &reqwest::Client) -> Option<Duration> {
    let clip = test_clip();
    let opts = TranscribeOptions::default();
    let mut last = None;
    // The first request may load the model, so the second one is what counts.
    for _ in 0..2 {
        let start = Instant::now();
        tokio::time::timeout(QUICK_TIMEOUT, transcribe_audio(client, &clip, &opts))
            .await
            .ok()?
            .ok()?;
        last = Some(start.elapsed());
    }
    last
}

async fn time_voice(client: &reqwest::Client, url: &str) -> Option<Duration> {
    let cfg = KokoroConfig {
        url: url.to_string(),
        model: "kokoro".to_string(),
        voice: "af_heart".to_string(),
    };
    let mut last = None;
    for _ in 0..2 {
        let start = Instant::now();
        tokio::time::timeout(
            QUICK_TIMEOUT,
            fetch_audio(client, &cfg, "Octopuses have three hearts."),
        )
        .await
        .ok()?
        .ok()?;
        last = Some(start.elapsed());
    }
    last
}

/// Time from sending the question until the first whole sentence of the reply is complete.
/// A one-sentence reply counts as complete when it ends.
async fn first_sentence(client: &reqwest::Client, model: &str) -> Option<Duration> {
    let mut conversation = Conversation::new(Backend::Local {
        model: model.to_string(),
        think: false,
    });
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let start = Instant::now();
    let task = tokio::spawn({
        let client = client.clone();
        async move { conversation.reply(&client, PROMPT, Some(&tx)).await }
    });

    let mut splitter = SentenceSplitter::default();
    let mut first = None;
    let waited = tokio::time::timeout(CHAT_TIMEOUT, async {
        while let Some(delta) = rx.recv().await {
            if let ReplyDelta::Text(text) = delta
                && first.is_none()
                && !splitter.push(&text).is_empty()
            {
                first = Some(start.elapsed());
            }
        }
    })
    .await;
    if waited.is_err() {
        task.abort();
        return None;
    }
    task.await.ok()?.ok()?;
    first.or_else(|| Some(start.elapsed()))
}

async fn time_model(client: &reqwest::Client, model: &str) -> Option<ModelSpeed> {
    let cold = first_sentence(client, model).await?;
    let warm = first_sentence(client, model).await?;
    Some(ModelSpeed {
        model: model.to_string(),
        cold,
        warm,
    })
}

/// Time what can be timed. Every request goes to the local server, so this is only useful when
/// that server is running.
pub async fn measure(client: &reqwest::Client, targets: &Targets<'_>) -> Latency {
    let stt = if targets.whisper {
        time_stt(client).await
    } else {
        None
    };
    let voice = match targets.kokoro_url {
        Some(url) => time_voice(client, url).await,
        None => None,
    };
    let chat = match targets.chat_model {
        Some(model) => time_model(client, model).await,
        None => None,
    };
    let smart = match targets.smart_model {
        Some(model) => time_model(client, model).await,
        None => None,
    };
    Latency {
        stt,
        chat,
        smart,
        voice,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(s: f64) -> Duration {
        Duration::from_secs_f64(s)
    }

    /// Chat is snappy up to 1.5 s, fine up to 4 s, and slow beyond.
    #[test]
    fn chat_ratings() {
        assert_eq!(rate_chat(secs(0.3)), Rating::Snappy);
        assert_eq!(rate_chat(secs(1.5)), Rating::Snappy);
        assert_eq!(rate_chat(secs(1.6)), Rating::Fine);
        assert_eq!(rate_chat(secs(4.0)), Rating::Fine);
        assert_eq!(rate_chat(secs(4.1)), Rating::Slow);
        assert_eq!(rate_chat(secs(218.0)), Rating::Slow);
    }

    /// Speech-to-text and voice have their own, tighter scales.
    #[test]
    fn stt_and_voice_ratings() {
        assert_eq!(rate_stt(secs(0.4)), Rating::Snappy);
        assert_eq!(rate_stt(secs(2.0)), Rating::Fine);
        assert_eq!(rate_stt(secs(3.5)), Rating::Slow);
        assert_eq!(rate_voice(secs(0.5)), Rating::Snappy);
        assert_eq!(rate_voice(secs(2.0)), Rating::Fine);
        assert_eq!(rate_voice(secs(2.6)), Rating::Slow);
    }

    /// A first reply that was far slower than the second means the model had to load.
    #[test]
    fn detects_loading_on_demand() {
        let speed = |cold, warm| ModelSpeed {
            model: "m".into(),
            cold: secs(cold),
            warm: secs(warm),
        };
        assert!(speed(4.6, 0.7).loaded_on_demand());
        assert!(!speed(0.9, 0.7).loaded_on_demand(), "not much slower");
        assert!(
            !speed(2.0, 0.5).loaded_on_demand(),
            "slower, but by under two seconds"
        );
        assert!(
            !speed(5.0, 4.0).loaded_on_demand(),
            "a slow model is not a loading model"
        );
    }

    /// The test clip is a valid 16 kHz mono WAV of the promised length and is not silent.
    #[test]
    fn test_clip_is_a_real_wav() {
        let bytes = test_clip();
        let reader = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        let spec = reader.spec();
        assert_eq!(
            (spec.channels, spec.sample_rate, spec.bits_per_sample),
            (1, 16_000, 16)
        );
        let samples: Vec<i16> = reader.into_samples().map(Result::unwrap).collect();
        assert_eq!(samples.len(), 16_000 * CLIP_SECONDS as usize);
        assert!(
            samples.iter().any(|s| s.abs() > 1000),
            "the clip should not be silent"
        );
    }

    /// With nothing to time, nothing is timed and nothing is sent anywhere.
    #[tokio::test]
    async fn nothing_to_time() {
        let client = reqwest::Client::new();
        let latency = measure(&client, &Targets::default()).await;
        assert_eq!(latency, Latency::default());
    }

    /// An unreachable endpoint gives no timing rather than an error or a hang.
    #[tokio::test]
    async fn unreachable_voice_is_none() {
        let client = reqwest::Client::new();
        assert_eq!(
            time_voice(&client, "http://127.0.0.1:1/v1/audio/speech").await,
            None
        );
    }
}
