use anyhow::Result;
use clap::Parser;
use std::io::{self, Write};
use std::time::Duration;
use tokio::sync::mpsc;

mod audio;
mod transcribe;

use audio::AudioRecorder;
use transcribe::{TranscribeOptions, clean_transcript, is_hallucination, transcribe_audio};

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

    /// Print the live input level so you can pick a good --threshold
    #[arg(long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Voice Notes - Speech-to-Text CLI");
    println!("==================================");

    let args = Args::parse();
    let mut recorder = AudioRecorder::new(args.threshold, args.verbose);

    println!("Press Enter to start recording...");
    io::stdin().read_line(&mut String::new())?;

    recorder.start_recording()?;
    println!("Calibrating to background noise, stay quiet for a moment...");
    let mut announced = false;

    // One worker transcribes segments in order, so text prints in speaking order.
    let (tx, mut rx) = mpsc::unbounded_channel::<Vec<u8>>();
    let client = reqwest::Client::new();
    let worker_client = client.clone();
    let opts = TranscribeOptions {
        language: (args.language != "auto").then(|| args.language.clone()),
        prompt: None,
    };
    let worker = tokio::spawn(async move {
        let mut parts: Vec<String> = Vec::new();
        while let Some(wav) = rx.recv().await {
            match transcribe_audio(&worker_client, &wav, &opts).await {
                Ok(text) => {
                    let text = text.trim().to_string();
                    if is_hallucination(&text) {
                        if !text.is_empty() {
                            eprintln!("\n[dropped likely hallucination: {text:?}]");
                        }
                    } else {
                        print!("{text} ");
                        let _ = io::stdout().flush();
                        parts.push(text);
                    }
                }
                Err(e) => eprintln!("\n[transcription failed: {e:#}]"),
            }
        }
        parts
    });

    // Wait for Enter on a blocking thread while we poll for finished segments.
    let mut enter = tokio::task::spawn_blocking(|| io::stdin().read_line(&mut String::new()));
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    loop {
        tokio::select! {
            res = &mut enter => {
                res??;
                break;
            }
            _ = tick.tick() => {
                for wav in recorder.poll_segments()? {
                    let _ = tx.send(wav);
                }
                if !announced
                    && let Some(gate) = recorder.calibrated_threshold() {
                        announced = true;
                        println!("\nReady (speech gate {gate:.3}). Speak; text appears after each pause. Press Enter to stop.\n");
                    }
            }
        }
    }

    if let Some(wav) = recorder.stop_recording()? {
        let _ = tx.send(wav);
    }
    drop(tx);
    println!("\n\nFinishing transcription...");
    let parts = worker.await?;
    let raw_transcript = parts.join(" ");

    println!("\nRaw transcript:");
    println!("{raw_transcript}");

    if args.clean && !raw_transcript.is_empty() {
        println!("\nCleaning transcript...");
        let cleaned_transcript = clean_transcript(&client, &raw_transcript).await?;
        println!("\nCleaned transcript:");
        println!("{cleaned_transcript}");
    }

    Ok(())
}
