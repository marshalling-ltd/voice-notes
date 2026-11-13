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
cargo run -- --talk                 # talk mode with the default (`fast`) profile
cargo run -- --talk --profile dev   # technical answers
cargo run -- --talk --profile think # a reasoning model that shows its thinking
cargo run -- setup                  # find your models and services and save the best choices
cargo run -- --plain                # line-by-line interface (also used when not in a terminal)
```

Keys: `Enter` start/stop (or skip speech), `p` next profile, `t` notes/talk, `r` read aloud, `u` setup, `c` clean (notes), `n` new conversation (talk), `↑↓` scroll, `q` quit.

Other options: `--language hr` (spoken language, default `en`, `auto` to detect), `--threshold` (minimum speech level), `--no-speech` (text only), `--verbose` (with `--plain`, show the live input level). `--backend`, `--chat-model` and `--think` adjust the launch profile.

### Profiles

A profile is a named combination of backend, model, thinking and prompt. Switch live with `p` (the conversation starts afresh), or start with `--profile NAME`.

| profile | model | thinking | for |
| --- | --- | --- | --- |
| `fast` (default) | your fast chat model (default `qwen3-coder-30b`) | off | quick everyday chat |
| `dev` | the same model | off | technical answers, with a developer-oriented spoken prompt |
| `smart` | a reasoning model | off | nicer answers, a few seconds slower |
| `think` | a reasoning model | on | hard problems: the thinking is shown on screen, only the answer is spoken |
| `claude` | the `claude` CLI | n/a | deep help; read-only tools (`Read`, `Grep`, `Glob`) |

`smart` and `think` appear once `voice-notes setup` has found a reasoning model; `claude` appears when the CLI is installed (what you say is then sent to Anthropic and billed to your account). Reasoning models start answering much faster with thinking off, which is why only `think` turns it on.

Define your own, or adjust a built-in one, in the config file:

```toml
profile = "dev"                    # the profile used at launch

[profiles.dev]
model = "my-coder"                 # keeps the dev prompt unless you set `prompt`

[profiles.terse]
model = "qwen3-coder-30b"
think = false
prompt = "Answer in at most five words."
description = "Very short answers"
```

### Privacy

Voice Notes has no telemetry or analytics. It sends audio and text only to the endpoints you configure, which by default are on your own machine (`127.0.0.1`).

The one exception you can switch on is the **Claude backend** (the `claude` profile, or `--backend claude`): what you say is then sent to Anthropic, under Anthropic's terms, and billed to your account. While it is in use the Conversation panel says so on its border, and again before your first question. Local models never send anything off your machine.

### Setup

`voice-notes setup` looks at your machine and what you already run, picks the best option for each job, and saves the choices after asking (`--yes` skips the question, `--check` only reports, `--quick` skips timing the models). In the full-screen interface the same screen opens on first launch, and again with `u`; `s` skips it while it is still working.

It reports:

- **What it found:** the Whisper and chat models on the server, a Kokoro endpoint, the system voice, a player, the `claude` CLI, and which profiles that makes possible.
- **Your machine:** OS, cores, RAM, GPU and free disk. Memory available to models is a card's VRAM (NVIDIA, or discrete AMD), the shared pool for Apple Silicon and AMD APUs (read from the kernel's `amdgpu` files on Linux), or half the RAM when there is no GPU.
- **Speed:** it times a short clip through Whisper, a two-sentence reply from the chat model (cold, then warm, so you can see when a model had to load first), from the reasoning model, and one sentence from Kokoro, and rates each for conversation. Time to the first sentence is what matters: up to 1.5 s feels instant, up to 4 s is fine, beyond that is too slow for talking.
- **What would help:** missing pieces, and a chat model that is too slow here, come with a suggestion sized for your machine:

| memory for models | speech-to-text | chat | extras |
| --- | --- | --- | --- |
| under 8 GB (minimal) | Whisper small | Qwen3-4B | |
| 8 to 16 GB (light) | large-v3-turbo | Qwen3-8B | |
| 16 to 24 GB (standard) | large-v3-turbo | Qwen3-14B | |
| 24 GB or more (power) | large-v3-turbo | Qwen3-Coder-30B-A3B | a reasoning model for `smart` and `think` |

Sizes that were measured from real files are exact; estimates are marked with `~`. If the installed chat model is slower than your machine warrants, the suggestion is the model that fits (or the next size down if you are already there). This is a preview: nothing is downloaded yet.

Saved settings live in `~/.config/voice-notes/config.toml` (Linux) or `~/Library/Application Support/voice-notes/config.toml` (macOS). Precedence is: command-line flag, then saved settings, then built-in defaults.

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
| `src/chat.rs` | Conversation with a local model (streaming) or Claude, reasoning handling |
| `src/profile.rs` | Profiles: fast, dev, smart, think, claude and your own |
| `src/config.rs` | Saved settings |
| `src/setup.rs` | Finds models and services, picks the best, suggests models for your machine, saves the choices |
| `src/hardware.rs` | Detects memory, GPU and disk; size classes |
| `src/latency.rs` | Times Whisper, the chat models and Kokoro |
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
