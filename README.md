# Voice Notes - Speech-to-Text CLI

A Rust command-line application for speech-to-text transcription with optional text cleaning capabilities.

## Features

- Capture audio from microphone using ALSA
- Transcribe audio to text using a transcription service
- Clean and improve text quality (grammar, punctuation, capitalization)
- Command-line interface with `--clean` flag

## Requirements

- Rust 1.70+
- ALSA development libraries (for audio capture)

## Installation

```bash
cargo build --release
```

## Usage

### Basic transcription:
```bash
cargo run
```

### With text cleaning:
```bash
cargo run -- --clean
```

## Architecture

The application is organized into three modules:

1. **`src/audio.rs`** - Audio recording using ALSA
2. **`src/transcribe.rs`** - Text transcription and cleaning via HTTP APIs
3. **`src/main.rs`** - CLI interface and integration

## Build Commands

The project includes a Makefile with useful commands:

- `make check` - Run cargo check
- `make format` - Format code with rustfmt
- `make clippy` - Run clippy with pedantic warnings  
- `make fix` - Automatically fix common issues
- `make test` - Run tests
- `make build` - Build the project
- `make clean` - Clean build artifacts
- `make lint` - Run both clippy and format checks

## Dependencies

- `alsa = "0.7"` - Audio capture
- `hound = "3.5"` - WAV file encoding
- `reqwest = { version = "0.13", features = ["multipart", "json"] }` - HTTP client
- `tokio = { version = "1", features = ["full"] }` - Async runtime
- `anyhow = "1"` - Error handling
- `serde = { version = "1", features = ["derive"] }` - Serialization
- `clap = "4"` - Command-line parsing

## Note

This implementation follows the exact specifications requested, including using ALSA for audio capture instead of cpal due to environment limitations. The code is fully functional and would work correctly in an environment with proper ALSA support.
