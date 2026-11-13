use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct TranscriptionResponse {
    text: String,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct CleanTranscriptRequest {
    model: &'static str,
    messages: Vec<ChatMessage>,
}

#[derive(Deserialize)]
struct CleanTranscriptResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(rename = "message")]
    message: ChatMessage,
}

/// Settings that steer Whisper away from inventing text.
#[derive(Clone, Default)]
pub struct TranscribeOptions {
    /// ISO language code such as "en"; `None` lets the server auto-detect.
    pub language: Option<String>,
    /// Short context prompt; leave `None` unless you have vocabulary to bias towards.
    pub prompt: Option<String>,
}

const STOCK_PHRASES: &[&str] = &[
    "you",
    "thank you",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "bye",
    "bye bye",
    "okay",
    "[music]",
    "(music)",
    "[blank_audio]",
    "[silence]",
];

/// Heuristics for text Whisper typically makes up from noise or near-silence.
#[must_use]
pub fn is_hallucination(text: &str) -> bool {
    let lower = text.trim().to_lowercase();
    let bare: String = lower
        .chars()
        .filter(|c| {
            c.is_alphanumeric()
                || c.is_whitespace()
                || *c == '_'
                || *c == '['
                || *c == ']'
                || *c == '('
                || *c == ')'
        })
        .collect();
    let bare = bare.split_whitespace().collect::<Vec<_>>().join(" ");

    if bare.is_empty() || STOCK_PHRASES.contains(&bare.as_str()) {
        return true;
    }
    // Credits learned from subtitle data.
    if lower.contains("amara.org") || lower.contains("subtitles by") {
        return true;
    }
    // Subtitle-style dialogue markers ("- Yeah? - I'll be there.").
    if lower.starts_with('-') && lower[1..].contains(" - ") {
        return true;
    }
    // Looping output: one word five or more times in a row.
    let words: Vec<&str> = bare.split_whitespace().collect();
    words.windows(5).any(|w| w.iter().all(|x| *x == w[0]))
}

pub async fn transcribe_audio(
    client: &Client,
    audio_data: &[u8],
    opts: &TranscribeOptions,
) -> Result<String> {
    let part = reqwest::multipart::Part::bytes(audio_data.to_vec())
        .file_name("audio.wav")
        .mime_str("audio/wav")?;

    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", "whisper")
        .text("response_format", "json")
        .text("temperature", "0");
    let form = match &opts.language {
        Some(lang) => form.text("language", lang.clone()),
        None => form,
    };
    let form = match &opts.prompt {
        Some(prompt) => form.text("prompt", prompt.clone()),
        None => form,
    };

    let response = client
        .post("http://127.0.0.1:8080/v1/audio/transcriptions")
        .multipart(form)
        .send()
        .await?
        .error_for_status()?;

    let transcription_response: TranscriptionResponse = response.json().await?;
    Ok(transcription_response.text)
}

pub async fn clean_transcript(client: &Client, raw_text: &str) -> Result<String> {
    let request = CleanTranscriptRequest {
        model: "qwen3-coder-30b",
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant that improves text quality by fixing punctuation, capitalization, and grammar without changing the meaning.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: format!("Please improve the following text: {raw_text}"),
            },
        ],
    };

    let response = client
        .post("http://127.0.0.1:8080/v1/chat/completions")
        .json(&request)
        .send()
        .await?
        .error_for_status()?;

    let clean_response: CleanTranscriptResponse = response.json().await?;
    Ok(clean_response.choices[0].message.content.clone())
}

#[cfg(test)]
mod tests {
    use super::is_hallucination;

    #[test]
    fn flags_known_hallucinations() {
        for t in [
            "you",
            " Thank you. ",
            "Thanks for watching!",
            "[BLANK_AUDIO]",
            "- Yeah? - I'll be there.",
            "Subtitles by the Amara.org community",
            "no no no no no no",
            "",
        ] {
            assert!(is_hallucination(t), "should flag {t:?}");
        }
    }

    #[test]
    fn keeps_real_speech() {
        for t in [
            "Are you working?",
            "Thank you for the update on the report.",
            "I think it's a little bit slow.",
            "No, no, I meant the other one.",
        ] {
            assert!(!is_hallucination(t), "should keep {t:?}");
        }
    }
}
