//! Where the services live. The client speaks OpenAI-compatible endpoints only, so anything
//! that answers the same requests can be swapped in: another machine, a container, a cloud.

use anyhow::{Result, bail};

/// The server that normally hosts speech-to-text, chat and the model list.
pub const DEFAULT_SERVER: &str = "http://127.0.0.1:8080";

/// Default speech endpoint (a Kokoro server).
pub const DEFAULT_TTS_URL: &str = "http://127.0.0.1:8880/v1/audio/speech";

/// The model name sent to the speech-to-text endpoint.
pub const DEFAULT_STT_MODEL: &str = "whisper";

/// The four endpoints the client uses.
///
/// | Job | Request |
/// | --- | --- |
/// | speech to text | `POST /v1/audio/transcriptions` |
/// | chat (streaming) | `POST /v1/chat/completions` |
/// | model list | `GET /v1/models` |
/// | text to speech | `POST /v1/audio/speech` |
///
/// # Examples
///
/// ```
/// use voice_notes::endpoints::Endpoints;
///
/// let local = Endpoints::default();
/// assert_eq!(local.chat, "http://127.0.0.1:8080/v1/chat/completions");
///
/// // One server for speech-to-text, chat and the model list:
/// let remote = Endpoints::from_base("http://gpu-box:8080/");
/// assert_eq!(remote.stt, "http://gpu-box:8080/v1/audio/transcriptions");
/// assert_eq!(remote.models, "http://gpu-box:8080/v1/models");
/// // Text to speech is a separate service and keeps its own address.
/// assert_eq!(remote.tts, local.tts);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoints {
    /// Full URL for speech to text.
    pub stt: String,
    /// Full URL for chat completions.
    pub chat: String,
    /// Full URL for the model list.
    pub models: String,
    /// Full URL for text to speech.
    pub tts: String,
}

impl Default for Endpoints {
    fn default() -> Self {
        Self::from_base(DEFAULT_SERVER)
    }
}

impl Endpoints {
    /// The usual layout for one server: speech to text, chat and the model list under
    /// `base`. Text to speech keeps its default address. A trailing slash on `base` is fine.
    #[must_use]
    pub fn from_base(base: &str) -> Self {
        let base = base.trim_end_matches('/');
        Self {
            stt: format!("{base}/v1/audio/transcriptions"),
            chat: format!("{base}/v1/chat/completions"),
            models: format!("{base}/v1/models"),
            tts: DEFAULT_TTS_URL.to_string(),
        }
    }

    /// Check that every endpoint is an `http://` or `https://` URL.
    ///
    /// # Errors
    ///
    /// Returns an error naming the first endpoint that is not.
    pub fn validate(&self) -> Result<()> {
        for (name, url) in [
            ("speech-to-text", &self.stt),
            ("chat", &self.chat),
            ("model list", &self.models),
            ("text-to-speech", &self.tts),
        ] {
            let ok = url
                .strip_prefix("http://")
                .or_else(|| url.strip_prefix("https://"))
                .is_some_and(|rest| !rest.is_empty() && !rest.starts_with('/'));
            if !ok {
                bail!("the {name} endpoint `{url}` must be an http:// or https:// URL");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults match the contract: three endpoints on 8080, speech on 8880.
    #[test]
    fn defaults_follow_the_contract() {
        let e = Endpoints::default();
        assert_eq!(e.stt, "http://127.0.0.1:8080/v1/audio/transcriptions");
        assert_eq!(e.chat, "http://127.0.0.1:8080/v1/chat/completions");
        assert_eq!(e.models, "http://127.0.0.1:8080/v1/models");
        assert_eq!(e.tts, "http://127.0.0.1:8880/v1/audio/speech");
        assert!(e.validate().is_ok());
    }

    /// A base URL fills in three endpoints, with or without a trailing slash.
    #[test]
    fn base_url_builds_the_paths() {
        let plain = Endpoints::from_base("https://example.com:9000");
        let slash = Endpoints::from_base("https://example.com:9000///");
        assert_eq!(plain, slash);
        assert_eq!(plain.chat, "https://example.com:9000/v1/chat/completions");
        assert_eq!(plain.tts, DEFAULT_TTS_URL, "speech is a separate service");
    }

    /// Anything that is not a plain http(s) URL is rejected with a message naming the endpoint.
    #[test]
    fn rejects_bad_urls() {
        for bad in [
            "",
            "localhost:8080/v1",
            "ftp://x/y",
            "http://",
            "https:///path",
            "8080",
        ] {
            let e = Endpoints {
                stt: bad.into(),
                ..Endpoints::default()
            };
            let err = e.validate().unwrap_err().to_string();
            assert!(
                err.contains("speech-to-text") && err.contains("http"),
                "{bad:?}: {err}"
            );
        }
        let e = Endpoints {
            tts: "nope".into(),
            ..Endpoints::default()
        };
        assert!(
            e.validate()
                .unwrap_err()
                .to_string()
                .contains("text-to-speech")
        );
        let e = Endpoints::from_base("https://remote.example/");
        assert!(e.validate().is_ok());
    }
}
