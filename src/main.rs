use anyhow::{Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use std::io::IsTerminal;

mod audio;
// Used by the downloader and the models screen (tickets #24 and #25); not by the program yet.
#[allow(dead_code)]
mod catalogue;
mod chat;
mod config;
mod devices;
// Used by the models screen and first-run setup (VN-19 and VN-30); not by the program yet.
#[allow(dead_code)]
mod download;
mod endpoints;
mod hardware;
mod headless;
mod latency;
mod pipeline;
mod plain;
mod playback;
mod profile;
mod setup;
mod speech;
mod transcribe;
mod ui;

#[cfg(test)]
mod test_support;

use chat::Backend;
use config::Config;
use endpoints::{DEFAULT_STT_MODEL, Endpoints};
use pipeline::Settings;
use profile::Profile;
use speech::{KokoroConfig, SpeechConfig};

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
    /// Run one voice turn without a microphone: a recording is transcribed, answered and
    /// spoken. For tests and scripts. Exit code: 0 ok, 10 recording, 11 transcribing, 12 chat,
    /// 13 speech
    Turn {
        /// The recording that stands in for the microphone (any WAV file)
        #[arg(long)]
        input_wav: std::path::PathBuf,
        /// Do not play the reply; still fetch it from the speech service and check it
        #[arg(long)]
        no_audio_output: bool,
        /// Print the result as one JSON object instead of text
        #[arg(long)]
        json: bool,
    },
    /// List the microphones and speakers; the default and the saved choice are marked
    Devices,
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
    #[arg(long, global = true)]
    language: Option<String>,

    /// Use the line-by-line interface instead of the full-screen one
    #[arg(long)]
    plain: bool,

    /// Start in talk mode: speak, get an answer, hear it spoken
    #[arg(long)]
    talk: bool,

    /// Profile: how the assistant answers (`fast`, `dev`, `smart`, `think`, `claude`, or your
    /// own from the config file) [default: the saved one, else `fast`]
    #[arg(long, global = true)]
    profile: Option<String>,

    /// Override the profile's backend
    #[arg(long, global = true, value_enum)]
    backend: Option<BackendArg>,

    /// Override the profile's local model, e.g. qwen3-coder-30b or qwen-reasoner
    #[arg(long, global = true)]
    chat_model: Option<String>,

    /// Let the model think before answering (much slower; the `think` profile does this)
    #[arg(long, global = true)]
    think: bool,

    /// How replies are spoken [default: kokoro]
    #[arg(long, global = true, value_enum)]
    tts: Option<TtsArg>,

    /// Speech endpoint, `POST /v1/audio/speech` (default `http://127.0.0.1:8880/v1/audio/speech`)
    #[arg(long, global = true)]
    tts_url: Option<String>,

    /// One server for speech-to-text, chat and the model list, e.g. `http://gpu-box:8080`
    /// (default `http://127.0.0.1:8080`); the three URLs below override it one by one
    #[arg(long, global = true)]
    server: Option<String>,

    /// Speech-to-text endpoint, `POST /v1/audio/transcriptions`
    #[arg(long, global = true)]
    stt_url: Option<String>,

    /// Chat endpoint, `POST /v1/chat/completions`
    #[arg(long, global = true)]
    chat_url: Option<String>,

    /// Model-list endpoint, `GET /v1/models`
    #[arg(long, global = true)]
    models_url: Option<String>,

    /// Model name sent to the speech-to-text endpoint [default: whisper]
    #[arg(long, global = true)]
    stt_model: Option<String>,

    /// Model name sent to the Kokoro server [default: kokoro]
    #[arg(long, global = true)]
    tts_model: Option<String>,

    /// Kokoro voice, e.g. `af_heart`, `af_bella`, `am_michael`, `bf_emma` (default `af_heart`)
    #[arg(long, global = true)]
    tts_voice: Option<String>,

    /// System voice for --tts system and as the fallback (espeak-ng voice such as en or en-gb;
    /// a `say` voice on macOS) [default: en]
    #[arg(long, global = true)]
    voice: Option<String>,

    /// Do not speak replies aloud
    #[arg(long, global = true)]
    no_speech: bool,

    /// Microphone to record from: a device id or a name (see `voice-notes devices`)
    /// [default: the saved one, else the system default]
    #[arg(long, global = true)]
    input_device: Option<String>,

    /// Speaker to play replies on: a device id or a name [default: the saved one, else the
    /// system default]
    #[arg(long, global = true)]
    output_device: Option<String>,

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

/// Work out the four endpoints. From highest priority to lowest: an individual flag, the
/// `--server` flag, an individual key in the config file, the config's `server`, the default.
fn resolve_endpoints(args: &Args, config: &Config) -> Result<Endpoints> {
    let flag_server = args.server.as_deref().map(Endpoints::from_base);
    let saved_server = config.server.as_deref().map(Endpoints::from_base);
    let default = Endpoints::default();
    let choose =
        |flag: &Option<String>, saved: &Option<String>, field: fn(&Endpoints) -> &String| {
            flag.clone()
                .or_else(|| flag_server.as_ref().map(|e| field(e).clone()))
                .or_else(|| saved.clone())
                .or_else(|| saved_server.as_ref().map(|e| field(e).clone()))
                .unwrap_or_else(|| field(&default).clone())
        };
    let endpoints = Endpoints {
        stt: choose(&args.stt_url, &config.stt_url, |e| &e.stt),
        chat: choose(&args.chat_url, &config.chat_url, |e| &e.chat),
        models: choose(&args.models_url, &config.models_url, |e| &e.models),
        // Speech is a separate service, so `server` does not move it.
        tts: args
            .tts_url
            .clone()
            .or_else(|| config.tts_url.clone())
            .unwrap_or(default.tts),
    };
    endpoints.validate()?;
    Ok(endpoints)
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
    let endpoints = resolve_endpoints(args, config)?;

    Ok(Settings {
        stt_model: pick(&args.stt_model, &config.stt_model, DEFAULT_STT_MODEL),
        clean_model: config
            .chat_model
            .clone()
            .unwrap_or_else(|| profile::DEFAULT_FAST_MODEL.to_string()),
        clean: args.clean,
        min_threshold: args.threshold,
        language: (language != "auto").then_some(language),
        verbose: args.verbose,
        input_device: args
            .input_device
            .clone()
            .or_else(|| config.input_device.clone()),
        backend,
        system_prompt: chosen.prompt,
        profile: chosen.name,
        profiles: profiles.into_iter().map(|p| p.name).collect(),
        speak: !args.no_speech,
        speech: SpeechConfig {
            kokoro: (tts == TtsArg::Kokoro).then(|| KokoroConfig {
                url: endpoints.tts.clone(),
                model: pick(&args.tts_model, &config.tts_model, "kokoro"),
                voice: pick(&args.tts_voice, &config.tts_voice, "af_heart"),
            }),
            output_device: args
                .output_device
                .clone()
                .or_else(|| config.output_device.clone()),
            ..SpeechConfig::system(pick(&args.voice, &config.voice, "en"))
        },
        talk: args.talk,
        endpoints,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(Command::Turn {
        input_wav,
        no_audio_output,
        json,
    }) = &args.command
    {
        let config = Config::load()?;
        let settings = resolve(&args, &config, None)?;
        let opts = headless::TurnOptions {
            wav: input_wav.clone(),
            no_audio_output: *no_audio_output,
        };
        return match headless::run_turn(&reqwest::Client::new(), &settings, &opts).await {
            Ok(report) => {
                if *json {
                    println!("{}", report.to_json());
                } else {
                    print!("{}", report.to_text());
                }
                Ok(())
            }
            Err(e) => {
                eprintln!("voice-notes: {e}");
                std::process::exit(e.stage.exit_code());
            }
        };
    }
    if let Some(Command::Devices) = args.command {
        let config = Config::load()?;
        let input = args.input_device.as_ref().or(config.input_device.as_ref());
        let output = args
            .output_device
            .as_ref()
            .or(config.output_device.as_ref());
        for (direction, wanted) in [
            (devices::Direction::Input, input),
            (devices::Direction::Output, output),
        ] {
            let list = devices::list(direction)?;
            print!(
                "{}",
                devices::render(direction, &list, wanted.map(String::as_str))
            );
        }
        return Ok(());
    }
    if let Some(Command::Setup { yes, check, quick }) = args.command {
        let config = Config::load()?;
        let endpoints = resolve_endpoints(&args, &config)?;
        return setup::run(&endpoints, yes, check, quick).await;
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

    /// The saved devices are used when no flag is given, and a flag wins over them.
    #[test]
    fn audio_device_flags_beat_the_saved_choice() {
        let saved = Config {
            input_device: Some("saved-mic".into()),
            output_device: Some("saved-speaker".into()),
            ..Config::default()
        };
        let none = settings(&[], &Config::default(), false).unwrap();
        assert_eq!(none.input_device, None);
        assert_eq!(none.speech.output_device, None);

        let from_config = settings(&[], &saved, false).unwrap();
        assert_eq!(from_config.input_device.as_deref(), Some("saved-mic"));
        assert_eq!(
            from_config.speech.output_device.as_deref(),
            Some("saved-speaker")
        );

        let flags = [
            "--input-device",
            "flag-mic",
            "--output-device",
            "flag-speaker",
        ];
        let from_flags = settings(&flags, &saved, false).unwrap();
        assert_eq!(from_flags.input_device.as_deref(), Some("flag-mic"));
        assert_eq!(
            from_flags.speech.output_device.as_deref(),
            Some("flag-speaker")
        );
    }

    /// The device flags work after a subcommand too, like the other global flags.
    #[test]
    fn device_flags_are_global_and_the_devices_command_exists() {
        let a = args(&["devices", "--input-device", "x"]);
        assert!(matches!(a.command, Some(Command::Devices)));
        assert_eq!(a.input_device.as_deref(), Some("x"));
        let turn = args(&["turn", "--input-wav", "a.wav", "--output-device", "y"]);
        assert_eq!(turn.output_device.as_deref(), Some("y"));
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
        assert_eq!(kokoro.url, endpoints::DEFAULT_TTS_URL);
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
        let parsed = args(&[
            "setup",
            "--check",
            "--server",
            "http://gpu-box:8080",
            "--stt-model",
            "tiny",
        ]);
        assert!(matches!(
            parsed.command,
            Some(Command::Setup { check: true, .. })
        ));
        assert_eq!(parsed.server.as_deref(), Some("http://gpu-box:8080"));
        assert_eq!(parsed.stt_model.as_deref(), Some("tiny"));
        let turn = args(&[
            "turn",
            "--input-wav",
            "q.wav",
            "--no-audio-output",
            "--json",
            "--no-speech",
            "--profile",
            "dev",
            "--tts",
            "system",
            "--language",
            "hr",
        ]);
        assert!(matches!(
            turn.command,
            Some(Command::Turn {
                json: true,
                no_audio_output: true,
                ..
            })
        ));
        assert!(turn.no_speech && turn.profile.as_deref() == Some("dev"));
        assert_eq!(turn.tts, Some(TtsArg::System));
        assert_eq!(turn.language.as_deref(), Some("hr"));
        assert!(
            Args::try_parse_from(["voice-notes", "turn"]).is_err(),
            "--input-wav is required"
        );
        let before = args(&["--server", "http://gpu-box:8080", "setup", "--check"]);
        assert_eq!(before.server.as_deref(), Some("http://gpu-box:8080"));
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

    fn endpoints(list: &[&str], config: &Config) -> Result<Endpoints> {
        resolve_endpoints(&args(list), config)
    }

    /// With nothing configured the local defaults apply.
    #[test]
    fn endpoints_default_to_the_local_server() {
        let e = endpoints(&[], &Config::default()).unwrap();
        assert_eq!(e, Endpoints::default());
        assert_eq!(e.tts, endpoints::DEFAULT_TTS_URL);
    }

    /// `--server` moves speech-to-text, chat and the model list at once, but not speech.
    #[test]
    fn server_flag_sets_three_endpoints() {
        let e = endpoints(&["--server", "http://gpu-box:9000/"], &Config::default()).unwrap();
        assert_eq!(e.stt, "http://gpu-box:9000/v1/audio/transcriptions");
        assert_eq!(e.chat, "http://gpu-box:9000/v1/chat/completions");
        assert_eq!(e.models, "http://gpu-box:9000/v1/models");
        assert_eq!(e.tts, endpoints::DEFAULT_TTS_URL);
    }

    /// Each endpoint can be set on its own, by flag or in the config file.
    #[test]
    fn single_endpoints_by_flag_and_config() {
        let config = Config {
            chat_url: Some("http://saved/chat".into()),
            tts_url: Some("http://saved/speech".into()),
            ..Config::default()
        };
        let e = endpoints(&[], &config).unwrap();
        assert_eq!(e.chat, "http://saved/chat");
        assert_eq!(e.tts, "http://saved/speech");
        assert_eq!(
            e.stt,
            Endpoints::default().stt,
            "the others keep their defaults"
        );

        let e = endpoints(
            &[
                "--chat-url",
                "http://flag/chat",
                "--stt-url",
                "http://flag/stt",
                "--models-url",
                "http://flag/models",
            ],
            &config,
        )
        .unwrap();
        assert_eq!(
            (e.chat.as_str(), e.stt.as_str(), e.models.as_str()),
            ("http://flag/chat", "http://flag/stt", "http://flag/models")
        );
        assert_eq!(e.tts, "http://saved/speech");
    }

    /// Priority from highest to lowest: individual flag, `--server`, individual config key,
    /// config `server`, default.
    #[test]
    fn endpoint_precedence() {
        let config = Config {
            server: Some("http://config-server".into()),
            chat_url: Some("http://config-chat".into()),
            ..Config::default()
        };
        // Config only: the individual key beats the config's server.
        let e = endpoints(&[], &config).unwrap();
        assert_eq!(e.chat, "http://config-chat");
        assert_eq!(e.stt, "http://config-server/v1/audio/transcriptions");
        // The server flag beats the config's individual key.
        let e = endpoints(&["--server", "http://flag-server"], &config).unwrap();
        assert_eq!(e.chat, "http://flag-server/v1/chat/completions");
        // An individual flag beats everything.
        let e = endpoints(
            &[
                "--server",
                "http://flag-server",
                "--chat-url",
                "http://flag-chat",
            ],
            &config,
        )
        .unwrap();
        assert_eq!(e.chat, "http://flag-chat");
        assert_eq!(e.stt, "http://flag-server/v1/audio/transcriptions");
    }

    /// A malformed endpoint stops the program with a message that names it.
    #[test]
    fn bad_endpoint_is_an_error() {
        let err = endpoints(&["--chat-url", "localhost:8080"], &Config::default())
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("chat") && err.contains("localhost:8080"),
            "{err}"
        );
        let bad_config = Config {
            server: Some("gpu-box:8080".into()),
            ..Config::default()
        };
        assert!(endpoints(&[], &bad_config).is_err());
        assert!(settings(&["--stt-url", "nope"], &Config::default(), false).is_err());
    }

    /// The endpoints and model names reach the settings the app runs with.
    #[test]
    fn settings_carry_endpoints_and_model_names() {
        let s = settings(&[], &Config::default(), false).unwrap();
        assert_eq!(s.endpoints, Endpoints::default());
        assert_eq!(s.stt_model, "whisper");
        assert_eq!(s.clean_model, profile::DEFAULT_FAST_MODEL);
        assert_eq!(s.speech.kokoro.unwrap().url, endpoints::DEFAULT_TTS_URL);

        let config = Config {
            stt_model: Some("whisper-large-v3".into()),
            chat_model: Some("my-chat".into()),
            server: Some("http://gpu-box:8080".into()),
            ..Config::default()
        };
        let s = settings(&[], &config, false).unwrap();
        assert_eq!(s.stt_model, "whisper-large-v3");
        assert_eq!(s.clean_model, "my-chat");
        assert_eq!(s.endpoints.chat, "http://gpu-box:8080/v1/chat/completions");
        let s = settings(
            &["--stt-model", "tiny", "--tts-url", "http://voice/speech"],
            &config,
            false,
        )
        .unwrap();
        assert_eq!(s.stt_model, "tiny");
        assert_eq!(s.endpoints.tts, "http://voice/speech");
        assert_eq!(s.speech.kokoro.unwrap().url, "http://voice/speech");
    }
}
