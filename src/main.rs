use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use std::io::IsTerminal;

mod audio;
mod chat;
mod pipeline;
mod plain;
mod speech;
mod transcribe;
mod ui;

use chat::Backend;
use pipeline::Settings;

/// Who answers in talk mode.
#[derive(Clone, Copy, ValueEnum)]
enum BackendArg {
    /// A model on the local server (see --chat-model)
    Local,
    /// The `claude` CLI, limited to read-only tools
    Claude,
}

// Each bool is an independent command-line switch.
#[allow(clippy::struct_excessive_bools)]
#[derive(Parser)]
struct Args {
    /// Clean the transcript with grammar and punctuation fixes
    #[arg(long)]
    clean: bool,

    /// Minimum RMS level (0.0-1.0) treated as speech; the actual gate is auto-calibrated from ambient noise
    #[arg(long, default_value_t = 0.01)]
    threshold: f32,

    /// Spoken language code passed to Whisper (e.g. en, hr); `auto` lets it detect
    #[arg(long, default_value = "en")]
    language: String,

    /// Use the line-by-line interface instead of the full-screen one
    #[arg(long)]
    plain: bool,

    /// Start in talk mode: speak, get an answer, hear it spoken
    #[arg(long)]
    talk: bool,

    /// Who answers in talk mode
    #[arg(long, value_enum, default_value = "local")]
    backend: BackendArg,

    /// Local model for talk mode, e.g. qwen3-coder-30b (fast) or qwen-reasoner (slower, smarter)
    #[arg(long, default_value = "qwen3-coder-30b")]
    chat_model: String,

    /// espeak-ng voice used to speak replies (e.g. en, en-gb)
    #[arg(long, default_value = "en")]
    voice: String,

    /// Do not speak replies aloud
    #[arg(long)]
    no_speech: bool,

    /// Print the live input level (with --plain) so you can pick a good --threshold
    #[arg(long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let settings = Settings {
        clean: args.clean,
        min_threshold: args.threshold,
        language: (args.language != "auto").then(|| args.language.clone()),
        verbose: args.verbose,
        backend: match args.backend {
            BackendArg::Local => Backend::Local {
                model: args.chat_model.clone(),
            },
            BackendArg::Claude => Backend::Claude,
        },
        speak: !args.no_speech,
        voice: args.voice.clone(),
        talk: args.talk,
    };

    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if args.plain || !interactive {
        if args.talk {
            bail!("--talk needs the full-screen interface (run in a terminal, without --plain)");
        }
        println!("Voice Notes - Speech-to-Text CLI");
        println!("==================================");
        return plain::run(settings).await;
    }

    let text = ui::run(settings).await?;
    if !text.is_empty() {
        println!("{text}");
    }
    Ok(())
}
