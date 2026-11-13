//! Line-by-line interface for non-interactive terminals and `--plain`.

use crate::audio::AudioRecorder;
use crate::pipeline::{Job, Outcome, Settings, spawn_transcriber};
use crate::transcribe::{TranscribeOptions, clean_transcript};
use anyhow::Result;
use std::io::{self, Write};
use std::time::Duration;

pub async fn run(settings: Settings) -> Result<()> {
    let mut recorder = AudioRecorder::new(settings.min_threshold, settings.verbose);

    println!("Press Enter to start recording...");
    io::stdin().read_line(&mut String::new())?;

    recorder.start_recording()?;
    println!("Calibrating to background noise, stay quiet for a moment...");
    let mut announced = false;

    let client = reqwest::Client::new();
    let opts = TranscribeOptions {
        language: settings.language.clone(),
        prompt: None,
    };
    let (tx, mut rx) = spawn_transcriber(client.clone(), opts);
    let mut next_id = 0;
    let mut parts: Vec<String> = Vec::new();

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
                    let _ = tx.send(Job { id: next_id, wav });
                    next_id += 1;
                }
                while let Ok(outcome) = rx.try_recv() {
                    show(outcome, &mut parts);
                }
                if !announced && let Some(gate) = recorder.calibrated_threshold() {
                    announced = true;
                    println!("\nReady (speech gate {gate:.3}). Speak; text appears after each pause. Press Enter to stop.\n");
                }
            }
        }
    }

    if let Some(wav) = recorder.stop_recording()? {
        let _ = tx.send(Job { id: next_id, wav });
    }
    drop(tx);
    println!("\n\nFinishing transcription...");
    while let Some(outcome) = rx.recv().await {
        show(outcome, &mut parts);
    }
    let raw_transcript = parts.join(" ");

    println!("\nRaw transcript:");
    println!("{raw_transcript}");

    if settings.clean && !raw_transcript.is_empty() {
        println!("\nCleaning transcript...");
        let cleaned = clean_transcript(&client, &raw_transcript).await?;
        println!("\nCleaned transcript:");
        println!("{cleaned}");
    }
    Ok(())
}

fn show(outcome: Outcome, parts: &mut Vec<String>) {
    match outcome {
        Outcome::Text { text, .. } => {
            print!("{text} ");
            let _ = io::stdout().flush();
            parts.push(text);
        }
        Outcome::Dropped { text, .. } => {
            if !text.is_empty() {
                eprintln!("\n[dropped likely hallucination: {text:?}]");
            }
        }
        Outcome::Failed { error, .. } => eprintln!("\n[transcription failed: {error}]"),
    }
}
