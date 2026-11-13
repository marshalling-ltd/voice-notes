//! Saved choices, so the app works without flags once `voice-notes setup` has run.
//!
//! Precedence, highest first: command-line flag, this file, built-in default.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Every field is optional: an absent field means "use the built-in default".
#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq, Eq)]
#[serde(default)]
pub struct Config {
    /// Who answers in talk mode: `local` or `claude`.
    pub backend: Option<String>,
    /// Model name on the local server.
    pub chat_model: Option<String>,
    /// How replies are spoken: `kokoro` or `system`.
    pub tts: Option<String>,
    /// Kokoro speech endpoint.
    pub tts_url: Option<String>,
    /// Model name sent to the Kokoro server.
    pub tts_model: Option<String>,
    /// Kokoro voice.
    pub tts_voice: Option<String>,
    /// System voice (`espeak-ng` voice, or a macOS `say` voice).
    pub voice: Option<String>,
    /// Spoken language code, or `auto`.
    pub language: Option<String>,
    /// The profile used at launch (`fast` if absent).
    pub profile: Option<String>,
    /// The reasoning model behind the `smart` and `think` profiles.
    pub smart_model: Option<String>,
    /// Your own profiles, or overrides of the built-in ones.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub profiles: BTreeMap<String, ProfileConfig>,
}

/// A profile defined in the config file, as `[profiles.NAME]`.
#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq, Eq)]
#[serde(default)]
pub struct ProfileConfig {
    /// `local` or `claude`.
    pub backend: Option<String>,
    /// Local model name.
    pub model: Option<String>,
    /// Let a reasoning model think before answering.
    pub think: Option<bool>,
    /// System prompt for this profile.
    pub prompt: Option<String>,
    /// One line shown in setup.
    pub description: Option<String>,
}

/// Where the config file lives: `$XDG_CONFIG_HOME` or `~/.config` on Linux, and
/// `~/Library/Application Support` on macOS.
#[must_use]
pub fn path_from(xdg: Option<&str>, home: Option<&str>, macos: bool) -> Option<PathBuf> {
    let base = if macos {
        PathBuf::from(home?).join("Library/Application Support")
    } else if let Some(xdg) = xdg.filter(|x| !x.is_empty()) {
        PathBuf::from(xdg)
    } else {
        PathBuf::from(home?).join(".config")
    };
    Some(base.join("voice-notes").join("config.toml"))
}

impl Config {
    /// The config file path on this machine.
    #[must_use]
    pub fn path() -> Option<PathBuf> {
        path_from(
            std::env::var("XDG_CONFIG_HOME").ok().as_deref(),
            std::env::var("HOME").ok().as_deref(),
            cfg!(target_os = "macos"),
        )
    }

    /// Parse a config from TOML text.
    ///
    /// # Errors
    ///
    /// Returns an error if the text is not valid TOML for this config.
    pub fn parse(text: &str) -> Result<Self> {
        toml::from_str(text).context("the config file is not valid")
    }

    /// Render as TOML text.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails (it should not for this type).
    pub fn render(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }

    /// Load the saved config; a missing file is an empty config.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read or parsed.
    pub fn load() -> Result<Self> {
        let Some(path) = Self::path() else {
            return Ok(Self::default());
        };
        match std::fs::read_to_string(&path) {
            Ok(text) => Self::parse(&text).with_context(|| format!("in {}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("could not read {}", path.display())),
        }
    }

    /// Save to the config file, creating its folder.
    ///
    /// # Errors
    ///
    /// Returns an error if the location is unknown or the file cannot be written.
    pub fn save(&self) -> Result<PathBuf> {
        let path = Self::path().context("cannot work out where to save the config (no HOME)")?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("could not create {}", dir.display()))?;
        }
        std::fs::write(&path, self.render()?)
            .with_context(|| format!("could not write {}", path.display()))?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A config survives a round trip through TOML.
    #[test]
    fn round_trips_through_toml() {
        let config = Config {
            backend: Some("local".into()),
            chat_model: Some("qwen3-coder-30b".into()),
            tts: Some("kokoro".into()),
            tts_url: Some("http://127.0.0.1:8880/v1/audio/speech".into()),
            ..Config::default()
        };
        let text = config.render().unwrap();
        assert!(text.contains("chat_model = \"qwen3-coder-30b\""));
        assert!(!text.contains("language"), "absent fields are not written");
        assert!(
            !text.contains("profiles"),
            "an empty profile table is not written"
        );
        assert_eq!(Config::parse(&text).unwrap(), config);
    }

    /// An empty file, or one with only some fields, is fine; junk is an error.
    #[test]
    fn partial_and_invalid_files() {
        assert_eq!(Config::parse("").unwrap(), Config::default());
        let some = Config::parse("language = \"hr\"\n").unwrap();
        assert_eq!(some.language.as_deref(), Some("hr"));
        assert!(some.chat_model.is_none());
        assert!(Config::parse("language = [1, 2").is_err());
        assert!(Config::parse("language = 5").is_err());
    }

    /// The path follows each platform's convention.
    #[test]
    fn paths_per_platform() {
        let linux = path_from(None, Some("/home/a"), false).unwrap();
        assert_eq!(
            linux,
            PathBuf::from("/home/a/.config/voice-notes/config.toml")
        );
        let xdg = path_from(Some("/x/cfg"), Some("/home/a"), false).unwrap();
        assert_eq!(xdg, PathBuf::from("/x/cfg/voice-notes/config.toml"));
        let empty_xdg = path_from(Some(""), Some("/home/a"), false).unwrap();
        assert_eq!(empty_xdg, linux);
        let mac = path_from(None, Some("/Users/a"), true).unwrap();
        assert_eq!(
            mac,
            PathBuf::from("/Users/a/Library/Application Support/voice-notes/config.toml")
        );
        assert!(path_from(None, None, false).is_none());
    }

    /// Profiles round-trip as `[profiles.NAME]` tables, written after the plain settings.
    #[test]
    fn profiles_round_trip() {
        let mut config = Config {
            profile: Some("dev".into()),
            smart_model: Some("qwen-reasoner".into()),
            ..Config::default()
        };
        config.profiles.insert(
            "quiet".into(),
            ProfileConfig {
                backend: Some("local".into()),
                model: Some("tiny".into()),
                think: Some(false),
                prompt: Some("Answer in five words.".into()),
                description: Some("terse".into()),
            },
        );
        let text = config.render().unwrap();
        assert!(text.contains("[profiles.quiet]") && text.contains("model = \"tiny\""));
        assert!(text.find("profile = ").unwrap() < text.find("[profiles.quiet]").unwrap());
        assert_eq!(Config::parse(&text).unwrap(), config);

        let parsed = Config::parse("[profiles.x]\nthink = true\n").unwrap();
        assert_eq!(parsed.profiles["x"].think, Some(true));
        assert!(parsed.profiles["x"].model.is_none());
    }
}
