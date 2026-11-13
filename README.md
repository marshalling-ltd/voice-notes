# Voice Notes

Speak and get text, or speak and get an answer back. Runs on **Linux** and **macOS**.

- **Notes mode:** press Enter, talk, and text appears after each pause while you are still recording. Optionally polish the final transcript (grammar, punctuation, capitalization).
- **Talk mode:** press Enter, ask something, press Enter again. A chat model (local, or Claude) answers, and the answer is spoken back sentence by sentence as it is written.

## How it works

- Records from the default microphone with `cpal` (PipeWire/PulseAudio on Linux, CoreAudio on macOS).
- Measures about one second of background noise at the start and sets a speech gate from it, so it adapts to noisy or hot microphones.
- Cuts an utterance after a pause and transcribes it against a local Whisper-compatible server (`http://127.0.0.1:8080/v1/audio/transcriptions`) while recording continues.
- Drops weak segments and likely Whisper hallucinations (stock phrases, subtitle credits, looping text).
- Talk mode streams the reply from the chat model, hides any `<think>` reasoning, cuts it into sentences, and speaks each sentence as soon as it is complete.

## Requirements

| | Linux | macOS |
| --- | --- | --- |
| Rust | nightly (see `make requirements`) | nightly |
| Audio | `libasound2-dev pkg-config build-essential` to build | nothing extra |
| Playback | one of `paplay`, `pw-play`, `aplay` | `afplay` (built in) |
| System voice | `espeak-ng` | `say` (built in) |

You also need an OpenAI-compatible server on `127.0.0.1:8080` serving Whisper (`/v1/audio/transcriptions`) and, for talk mode and `--clean`, a chat model (`/v1/chat/completions`).

On macOS the first run asks for microphone access for your terminal app. If it was denied, enable it under System Settings → Privacy & Security → Microphone.

## Usage

```sh
cargo run                           # notes mode, full-screen interface
cargo run -- --clean                # also polish the final transcript
cargo run -- --talk                 # talk mode with the fast local model
cargo run -- --talk --chat-model qwen-reasoner      # a reasoning model (slower)
cargo run -- --talk --backend claude                # answers from Claude (see below)
cargo run -- --plain                # line-by-line interface (also used when not in a terminal)
```

Keys: `Enter` start/stop (or skip speech), `t` notes/talk, `r` read aloud, `c` clean (notes), `n` new conversation (talk), `↑↓` scroll, `q` quit.

Other options: `--language hr` (spoken language, default `en`, `auto` to detect), `--threshold` (minimum speech level), `--no-speech` (text only), `--think` (let reasoning models think; much slower), `--verbose` (with `--plain`, show the live input level).

### Chat backends

- `--backend local` (default): a model on the local server, chosen with `--chat-model` (default `qwen3-coder-30b`). Reasoning models answer far faster with thinking off, which is the default; `--think` turns it on.
- `--backend claude`: runs the `claude` CLI headless, restricted to read-only tools (`Read`, `Grep`, `Glob`), with the conversation kept across turns. What you say is sent to Anthropic and billed to your account.

### Voice

`--tts kokoro` (default) asks a [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech server for each sentence, preparing the next sentence while the current one plays. Any server with an OpenAI-compatible `/v1/audio/speech` endpoint works; [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) is one. Configure it with:

```sh
--tts-url http://127.0.0.1:8880/v1/audio/speech   # default (Kokoro-FastAPI port)
--tts-model kokoro
--tts-voice af_heart
```

If the server cannot be reached, the app says so once and switches to the system voice (`--voice`, default `en`), so talking keeps working. `--tts system` skips Kokoro entirely.

### Tuning

If nothing appears while you speak, run `cargo run -- --plain --verbose` and compare the `ac-level` while speaking with the gate. If a quiet room reads very high, lower the microphone gain, for example `wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.3` on Linux.

## Project layout

| File | Purpose |
| --- | --- |
| `src/audio.rs` | Microphone capture, noise calibration, silence-based segmentation, WAV encoding |
| `src/transcribe.rs` | Whisper and chat HTTP calls, hallucination filter |
| `src/chat.rs` | Conversation with a local model (streaming) or Claude, reasoning removal |
| `src/speech.rs` | Sentence splitting, Kokoro and system speech, playback queue |
| `src/pipeline.rs` | Settings and the ordered transcription worker |
| `src/ui.rs` | Full-screen interface (ratatui) |
| `src/plain.rs` | Line-by-line interface |
| `src/main.rs` | Command-line options |

## Contributing

1. Fork `marshalling-ltd/voice-notes` and clone your fork (`origin`); add the org repo as `upstream`.
2. One-time setup (Rust nightly and the cargo tools):

   ```sh
   make requirements   # on Linux also: sudo apt-get install -y libasound2-dev pkg-config build-essential
   cargo run           # start the app
   make check          # cargo update/upgrade, machete, audit, deny, typos, fmt, clippy pedantic, tests
   ```

3. Branch off `upstream/master`, make a focused change with tests, **run `make check` before every push**, push to your fork and open a pull request against `master`. CI runs the same checks on Linux and macOS and must pass.

Other targets: `make test`, `make nextest`, `make test-cov`, `make clean`.

## Releasing

CI (`.github/workflows/ci.yml`) runs on every pull request and push to `master`. Releases are built by `.github/workflows/release.yml`:

```sh
make bump-patch                 # bump the version in Cargo.toml and Cargo.lock, then commit it
git tag v0.1.1 && git push upstream v0.1.1
```

- A `v*` tag must match the version in `Cargo.toml`; it publishes a GitHub release with a tarball and `.sha256` for Linux x64, macOS Apple Silicon and macOS Intel.
- Pushing to the `release` branch (or running the workflow manually) refreshes the rolling `latest` pre-release.
