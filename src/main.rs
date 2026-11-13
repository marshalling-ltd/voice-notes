use anyhow::{Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use std::io::IsTerminal;

mod audio;
mod chat;
mod config;
mod hardware;
mod latency;
mod pipeline;
mod plain;
mod profile;
mod setup;
mod speech;
mod transcribe;
mod ui;

use chat::Backend;
use config::Config;
use pipeline::Settings;
use profile::Profile;
use speech::{KokoroConfig, SpeechConfig};

const DEFAULT_TTS_URL: &str = "http://127.0.0.1:8880/v1/audio/speech";

/// How replies are spoken.
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
enum TtsArg {
    /// A Kokoro server, falling back to the system voice if it cannot be reached
    Kokoro,
    /// The system voice only (`say` on macOS, `espeak-ng` on Linux)
    System,
}

/// Who answers in talk mode.
#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
enum BackendArg {
    /// A model on the local server (see --chat-model)
    Local,
    /// The `claude` CLI, limited to read-only tools
    Claude,
}

#[derive(Subcommand, Clone)]
enum Command {
    /// Find the models and services you already have, pick the best, and save the choices
    Setup {
        /// Save without asking
        #[arg(long, short)]
        yes: bool,
        /// Only report what was found; do not save anything
        #[arg(long)]
        check: bool,
        /// Skip timing the models (faster, but no speed advice)
        #[arg(long)]
        quick: bool,
    },
}

// Each bool is an independent command-line switch.
#[allow(clippy::struct_excessive_bools)]
#[derive(Parser, Clone)]
#[command(about = "Speak and get text, or speak and get an answer back")]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    /// Clean the transcript with grammar and punctuation fixes
    #[arg(long)]
    clean: bool,

    /// Minimum RMS level (0.0-1.0) treated as speech; the actual gate is auto-calibrated from ambient noise
    #[arg(long, default_value_t = 0.01)]
    threshold: f32,

    /// Spoken language code passed to Whisper (e.g. en, hr); `auto` lets it detect [default: en]
    #[arg(long)]
    language: Option<String>,

    /// Use the line-by-line interface instead of the full-screen one
    #[arg(long)]
    plain: bool,

    /// Start in talk mode: speak, get an answer, hear it spoken
    #[arg(long)]
    talk: bool,

    /// Profile: how the assistant answers (`fast`, `dev`, `smart`, `think`, `claude`, or your
    /// own from the config file) [default: the saved one, else `fast`]
    #[arg(long)]
    profile: Option<String>,

    /// Override the profile's backend
    #[arg(long, value_enum)]
    backend: Option<BackendArg>,

    /// Override the profile's local model, e.g. qwen3-coder-30b or qwen-reasoner
    #[arg(long)]
    chat_model: Option<String>,

    /// Let the model think before answering (much slower; the `think` profile does this)
    #[arg(long)]
    think: bool,

    /// How replies are spoken [default: kokoro]
    #[arg(long, value_enum)]
    tts: Option<TtsArg>,

    /// Kokoro speech endpoint (OpenAI-compatible `/v1/audio/speech`)
    /// (default `http://127.0.0.1:8880/v1/audio/speech`)
    #[arg(long)]
    tts_url: Option<String>,

    /// Model name sent to the Kokoro server [default: kokoro]
    #[arg(long)]
    tts_model: Option<String>,

    /// Kokoro voice, e.g. `af_heart`, `af_bella`, `am_michael`, `bf_emma` (default `af_heart`)
    #[arg(long)]
    tts_voice: Option<String>,

    /// System voice for --tts system and as the fallback (espeak-ng voice such as en or en-gb;
    /// a `say` voice on macOS) [default: en]
    #[arg(long)]
    voice: Option<String>,

    /// Do not speak replies aloud
    #[arg(long)]
    no_speech: bool,

    /// Print the live input level (with --plain) so you can pick a good --threshold
    #[arg(long)]
    verbose: bool,
}

/// Combine flags, the saved config and built-in defaults, in that order of priority.
///
/// `profile_override` is a profile picked while running (the `p` key); flags that adjust the
/// profile apply only to the profile chosen at launch.
fn resolve(args: &Args, config: &Config, profile_override: Option<&str>) -> Result<Settings> {
    resolve_with(args, config, profile_override, profile::available(config))
}

fn resolve_with(
    args: &Args,
    config: &Config,
    profile_override: Option<&str>,
    profiles: Vec<Profile>,
) -> Result<Settings> {
    let wanted = profile_override
        .map(ToString::to_string)
        .or_else(|| args.profile.clone())
        .unwrap_or_else(|| profile::default_name(config));
    let chosen = profile::pick(&profiles, &wanted)?;

    let mut backend = chosen.backend.clone();
    if profile_override.is_none() {
        match args.backend {
            Some(BackendArg::Claude) => backend = Backend::Claude,
            Some(BackendArg::Local) if backend == Backend::Claude => {
                backend = Backend::Local {
                    model: config
                        .chat_model
                        .clone()
                        .unwrap_or_else(|| profile::DEFAULT_FAST_MODEL.to_string()),
                    think: false,
                };
            }
            _ => {}
        }
        if let Backend::Local { model, think } = &mut backend {
            if let Some(flag) = &args.chat_model {
                model.clone_from(flag);
            }
            *think |= args.think;
        }
    }

    let tts = args
        .tts
        .or_else(|| {
            config
                .tts
                .as_deref()
                .and_then(|s| TtsArg::from_str(s, true).ok())
        })
        .unwrap_or(TtsArg::Kokoro);
    let pick = |flag: &Option<String>, saved: &Option<String>, default: &str| {
        flag.clone()
            .or_else(|| saved.clone())
            .unwrap_or_else(|| default.to_string())
    };
    let language = pick(&args.language, &config.language, "en");

    Ok(Settings {
        clean: args.clean,
        min_threshold: args.threshold,
        language: (language != "auto").then_some(language),
        verbose: args.verbose,
        backend,
        system_prompt: chosen.prompt,
        profile: chosen.name,
        profiles: profiles.into_iter().map(|p| p.name).collect(),
        speak: !args.no_speech,
        speech: SpeechConfig {
            kokoro: (tts == TtsArg::Kokoro).then(|| KokoroConfig {
                url: pick(&args.tts_url, &config.tts_url, DEFAULT_TTS_URL),
                model: pick(&args.tts_model, &config.tts_model, "kokoro"),
                voice: pick(&args.tts_voice, &config.tts_voice, "af_heart"),
            }),
            ..SpeechConfig::system(pick(&args.voice, &config.voice, "en"))
        },
        talk: args.talk,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(Command::Setup { yes, check, quick }) = args.command {
        return setup::run(yes, check, quick).await;
    }

    let config = Config::load()?;
    let settings = resolve(&args, &config, None)?;
    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if args.plain || !interactive {
        if args.talk {
            bail!("--talk needs the full-screen interface (run in a terminal, without --plain)");
        }
        println!("Voice Notes - Speech-to-Text CLI");
        println!("==================================");
        return plain::run(settings).await;
    }

    let resolver_args = args.clone();
    let resolver: ui::Resolver =
        std::sync::Arc::new(move |config, profile| resolve(&resolver_args, config, profile));
    let text = ui::run(settings, resolver, config).await?;
    if !text.is_empty() {
        println!("{text}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chat::DEV_STYLE;
    use profile::{DEFAULT_FAST_MODEL, available_with};

    fn args(list: &[&str]) -> Args {
        Args::try_parse_from(std::iter::once("voice-notes").chain(list.iter().copied())).unwrap()
    }

    fn settings(list: &[&str], config: &Config, claude: bool) -> Result<Settings> {
        resolve_with(&args(list), config, None, available_with(config, claude))
    }

    fn local(model: &str, think: bool) -> Backend {
        Backend::Local {
            model: model.into(),
            think,
        }
    }

    fn with_smart() -> Config {
        Config {
            chat_model: Some("qwen3-coder-30b".into()),
            smart_model: Some("qwen-reasoner".into()),
            ..Config::default()
        }
    }

    /// With no flags and no config, the `fast` profile and built-in defaults apply.
    #[test]
    fn defaults_without_flags_or_config() {
        let s = settings(&[], &Config::default(), false).unwrap();
        assert_eq!(s.backend, local(DEFAULT_FAST_MODEL, false));
        assert_eq!(s.profile, "fast");
        assert_eq!(s.profiles, ["fast", "dev"]);
        assert_eq!(s.system_prompt, None);
        assert_eq!(s.language.as_deref(), Some("en"));
        let kokoro = s.speech.kokoro.unwrap();
        assert_eq!(kokoro.url, DEFAULT_TTS_URL);
        assert_eq!(kokoro.voice, "af_heart");
        assert_eq!(s.speech.system_voice, "en");
        assert!(s.speak && !s.talk);
    }

    /// Saved choices replace the defaults, and flags replace saved choices.
    #[test]
    fn flags_beat_config_beats_defaults() {
        let config = Config {
            chat_model: Some("saved-model".into()),
            tts_url: Some("http://saved/v1/audio/speech".into()),
            language: Some("hr".into()),
            ..Config::default()
        };
        let s = settings(&[], &config, false).unwrap();
        assert_eq!(s.backend, local("saved-model", false));
        assert_eq!(
            s.speech.kokoro.as_ref().unwrap().url,
            "http://saved/v1/audio/speech"
        );
        assert_eq!(s.language.as_deref(), Some("hr"));

        let flags = [
            "--chat-model",
            "flag-model",
            "--tts-url",
            "http://flag/x",
            "--language",
            "auto",
        ];
        let s = settings(&flags, &config, false).unwrap();
        assert_eq!(s.backend, local("flag-model", false));
        assert_eq!(s.speech.kokoro.as_ref().unwrap().url, "http://flag/x");
        assert_eq!(s.language, None, "`auto` means let Whisper detect it");
    }

    /// Each profile gives the model, thinking switch and prompt it promises.
    #[test]
    fn profiles_select_model_thinking_and_prompt() {
        let config = with_smart();
        let dev = settings(&["--profile", "dev"], &config, false).unwrap();
        assert_eq!(dev.backend, local("qwen3-coder-30b", false));
        assert_eq!(dev.system_prompt.as_deref(), Some(DEV_STYLE));
        assert_eq!(dev.profile, "dev");

        let smart = settings(&["--profile", "smart"], &config, false).unwrap();
        assert_eq!(smart.backend, local("qwen-reasoner", false));
        let think = settings(&["--profile", "think"], &config, false).unwrap();
        assert_eq!(think.backend, local("qwen-reasoner", true));
        assert_eq!(think.profiles, ["fast", "dev", "smart", "think"]);
    }

    /// The saved default profile is used, and a flag beats it.
    #[test]
    fn saved_profile_and_flag() {
        let config = Config {
            profile: Some("dev".into()),
            ..with_smart()
        };
        assert_eq!(settings(&[], &config, false).unwrap().profile, "dev");
        assert_eq!(
            settings(&["--profile", "think"], &config, false)
                .unwrap()
                .profile,
            "think"
        );
    }

    /// Flags that adjust the profile apply at launch, but not to a profile picked later with `p`.
    #[test]
    fn flags_adjust_the_launch_profile_only() {
        let config = with_smart();
        let launch = args(&["--chat-model", "flag-model", "--think"]);
        let s = resolve_with(&launch, &config, None, available_with(&config, false)).unwrap();
        assert_eq!(s.backend, local("flag-model", true));

        let switched = resolve_with(
            &launch,
            &config,
            Some("smart"),
            available_with(&config, false),
        )
        .unwrap();
        assert_eq!(switched.backend, local("qwen-reasoner", false));
        assert_eq!(switched.profile, "smart");
    }

    /// An old config that chose the claude backend still works when the CLI is there, and a
    /// flag can switch to or away from Claude.
    #[test]
    fn claude_backend() {
        let config = Config {
            backend: Some("claude".into()),
            ..Config::default()
        };
        let s = settings(&[], &config, true).unwrap();
        assert_eq!(s.backend, Backend::Claude);
        assert_eq!(s.profile, "claude");
        assert!(
            settings(&[], &config, false).is_err(),
            "no CLI, no claude profile"
        );

        let s = settings(&["--backend", "claude"], &Config::default(), false).unwrap();
        assert_eq!(s.backend, Backend::Claude);
        let s = settings(&["--backend", "local"], &config, true).unwrap();
        assert!(matches!(s.backend, Backend::Local { .. }));
    }

    /// The saved voice engine is honoured, and a flag beats it.
    #[test]
    fn saved_tts() {
        let config = Config {
            tts: Some("system".into()),
            ..Config::default()
        };
        assert!(
            settings(&[], &config, false)
                .unwrap()
                .speech
                .kokoro
                .is_none()
        );
        let s = settings(&["--tts", "kokoro"], &config, false).unwrap();
        assert!(s.speech.kokoro.is_some());
        let bad = Config {
            tts: Some("nonsense".into()),
            ..Config::default()
        };
        assert!(settings(&[], &bad, false).unwrap().speech.kokoro.is_some());
    }

    /// An unknown profile is an error that names the ones that exist.
    #[test]
    fn unknown_profile_is_an_error() {
        let err = settings(&["--profile", "nope"], &Config::default(), false)
            .unwrap_err()
            .to_string();
        assert!(err.contains("`nope`") && err.contains("fast, dev"), "{err}");
    }

    /// The setup subcommand parses with its options.
    #[test]
    fn parses_setup_subcommand() {
        let parsed = args(&["setup", "--yes"]);
        assert!(matches!(
            parsed.command,
            Some(Command::Setup {
                yes: true,
                check: false,
                quick: false
            })
        ));
        let parsed = args(&["setup", "--check"]);
        assert!(matches!(
            parsed.command,
            Some(Command::Setup {
                yes: false,
                check: true,
                quick: false
            })
        ));
        let parsed = args(&["setup", "--quick", "--check"]);
        assert!(matches!(
            parsed.command,
            Some(Command::Setup {
                yes: false,
                check: true,
                quick: true
            })
        ));
        assert!(args(&["--talk"]).command.is_none());
    }
}
