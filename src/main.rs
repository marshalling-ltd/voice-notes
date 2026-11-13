use anyhow::Result;
use clap::Parser;
use std::io::IsTerminal;

mod audio;
mod pipeline;
mod plain;
mod transcribe;
mod ui;

use pipeline::Settings;

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
    };

    let interactive = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if args.plain || !interactive {
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
