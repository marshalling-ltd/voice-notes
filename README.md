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
| Playback | nothing extra (a sound server: PipeWire or PulseAudio) | nothing extra |
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

Keys: `Enter` start/stop (or skip speech), `p` next profile, `t` notes/talk, `r` read aloud, `u` setup, `d` audio devices, `m` models, `c` clean (notes), `n` new conversation (talk), `↑↓` scroll, `q` quit.

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

### Running without a microphone (tests and scripts)

`voice-notes turn` runs one whole voice turn with a recording standing in for the microphone, then prints the result and exits. It needs no terminal and no audio hardware, so it works in CI.

```sh
voice-notes turn --input-wav question.wav --no-audio-output          # text
voice-notes turn --input-wav question.wav --no-audio-output --json   # one JSON object
```

The turn is: read the recording, transcribe it, ask the chat backend, speak the reply.

- `--input-wav FILE` is any WAV file: mono or stereo, any sample rate, integer or float samples. Files streamed from a speech server, which declare an unknown length, are accepted too.
- `--no-audio-output` does not play the reply. It still asks the speech service for it and checks that audio came back, then throws the audio away, so a test can prove the whole loop without speakers. Without the flag, the reply is played and the command waits until it has finished. With `--no-speech`, or with only a system voice (which cannot be captured), the speech stage is skipped and the report says why.
- All the usual settings apply, before or after `turn`: `--server`, `--stt-url`, `--chat-url`, `--tts-url`, `--profile`, `--backend`, `--chat-model`, `--language` and so on.

Text output:

```
Transcript: What is the capital of France?
Reply: The capital of France is Paris.
Speech: 95178 bytes of audio received and discarded
```

JSON output has `transcript`, `reply`, `reply_chunks` (how many pieces the reply arrived in; more than one means it was streamed), `speech` (`discarded`, `played` or `skipped`), `audio_bytes`, `speech_skipped_because`, `profile`, `backend` and `timings_ms` (`transcribe`, `chat`, `speech`).

**Exit codes.** A failing turn prints which stage failed on stderr and exits with a code that names it:

| Code | Meaning |
| --- | --- |
| 0 | The turn worked |
| 10 | The recording is missing or not a WAV file |
| 11 | Transcription failed or recognised no speech (an empty answer, or a stock phrase such as "Thank you." that models invent for silence) |
| 12 | The chat backend did not answer |
| 13 | The speech service failed or returned no audio |
| 2 | The command line was invalid |

In a CI step this becomes `voice-notes turn --input-wav test.wav --no-audio-output --json`, and a non-zero exit fails the job.

### Endpoints and the API contract

The app talks to services over HTTP and speaks OpenAI-compatible requests only, so anything that answers the same requests can be swapped in: another machine, a container, a cloud service.

| Job | Request | Default | Flag | Config key |
| --- | --- | --- | --- | --- |
| Speech to text | `POST /v1/audio/transcriptions` | `http://127.0.0.1:8080/v1/audio/transcriptions` | `--stt-url` | `stt_url` |
| Chat (streaming) | `POST /v1/chat/completions` | `http://127.0.0.1:8080/v1/chat/completions` | `--chat-url` | `chat_url` |
| Model list | `GET /v1/models` | `http://127.0.0.1:8080/v1/models` | `--models-url` | `models_url` |
| Text to speech | `POST /v1/audio/speech` | `http://127.0.0.1:8880/v1/audio/speech` | `--tts-url` | `tts_url` |

Every value is a full `http://` or `https://` URL, and anything else stops the program with a message naming the endpoint.

For the usual case of one server hosting the first three, set it once: `--server http://gpu-box:8080` (config key `server`). It moves speech to text, chat and the model list together; text to speech is a separate service and keeps its own address.

Model names are configurable too: `--stt-model` / `stt_model` (default `whisper`), `--tts-model` / `tts_model` (default `kokoro`), and the chat model through the profile (`--chat-model`, or `chat_model` in the config file, which also picks the model that polishes transcripts with `--clean`).

Each setting is taken from the first place that has it: an individual flag, then `--server`, then the same key in the config file, then the config file's `server`, then the default. For example, with `server = "http://gpu-box:8080"` in the config file and `--chat-url http://other:8080/v1/chat/completions` on the command line, chat goes to `other` while speech to text and the model list go to `gpu-box`.

```sh
cargo run -- --server http://gpu-box:8080                       # one remote server
cargo run -- --stt-url http://stt-box:9000/v1/audio/transcriptions --stt-model whisper-large-v3
voice-notes setup --server http://gpu-box:8080                  # check that server instead
```

### Privacy

Voice Notes has no telemetry or analytics. It sends audio and text only to the endpoints you configure, which by default are on your own machine (`127.0.0.1`).

The one exception you can switch on is the **Claude backend** (the `claude` profile, or `--backend claude`): what you say is then sent to Anthropic, under Anthropic's terms, and billed to your account. While it is in use the Conversation panel says so on its border, and again before your first question. Local models never send anything off your machine.

### Setup

`voice-notes setup` looks at your machine and what you already run, picks the best option for each job, and saves the choices after asking (`--yes` skips the question, `--check` only reports, `--quick` skips timing the models). In the full-screen interface the same screen opens on first launch, and again with `u`; `s` skips it while it is still working.

It reports:

- **What it found:** the Whisper and chat models on the server, a Kokoro endpoint, the system voice, a player, the `claude` CLI, and which profiles that makes possible.
- **Your machine:** OS, cores, RAM, GPU and free disk. Memory available to models is a card's VRAM (NVIDIA, or discrete AMD), the shared pool for Apple Silicon (70% of the unified memory) and AMD APUs (read from the kernel's `amdgpu` files on Linux), or half the RAM when there is no GPU. On an Intel Mac a discrete card counts by its own memory (read from `system_profiler`), while Intel graphics share system memory like the CPU and count as no GPU. Cards report slightly less than their nominal size (a "24 GB" card shows 24 564 MiB), so memory is counted 2% generously and each card gets the class on its box. An Intel build running under Rosetta on an Apple Silicon Mac is recognised as Apple Silicon.
- **Speed:** it times a short clip through Whisper, a two-sentence reply from the chat model (cold, then warm, so you can see when a model had to load first), from the reasoning model, and one sentence from Kokoro, and rates each for conversation. Time to the first sentence is what matters: up to 1.5 s feels instant, up to 4 s is fine, beyond that is too slow for talking.
- **What would help:** missing pieces, and a chat model that is too slow here, come with a suggestion sized for your machine:

| memory for models | speech-to-text | chat | extras |
| --- | --- | --- | --- |
| under 8 GB (minimal) | Whisper small | Qwen3-4B | |
| 8 to 16 GB (light) | large-v3-turbo | Qwen3-8B | |
| 16 to 24 GB (standard) | large-v3-turbo | Qwen3-14B | |
| 24 GB or more (power) | large-v3-turbo | Qwen3-Coder-30B-A3B | Qwen3.8-27B, the reasoning model for `smart` and `think` |

The sizes are those of the real files in `models.toml`. If the installed chat model is slower than your machine warrants, the suggestion is the model that fits (or the next size down if you are already there). The suggested models can be downloaded from the app: see *Choosing models* below, or the models screen (`m`).

`voice-notes hardware` prints what was found about this machine, the memory models can use and the rule that gave it, and the size class; `voice-notes hardware --raw` adds the raw output of every command the detection read (`sysctl`, `system_profiler`, `nvidia-smi`, `lspci`, `/proc/meminfo`, `df`), which is what to include when reporting a wrong result.

**Choosing models.** On the setup screen press `c` to pick the model for each job: chat (the `fast` and `dev` profiles), smart (the `smart` and `think` profiles) and speech-to-text. Each row starts on the recommended model; `↑↓` moves between jobs, `←→` steps through the models your server offers (embedding, vision and speech models are never offered as chat models), Enter accepts and Esc cancels. For example, swap the coder model for the reasoning model as your chat model, then press Enter on the setup screen to save. A saved choice is used straight away, for all three jobs, without restarting the app. Your server loads a model the first time it is used, so the first reply after a change can be slow. The same screen is reachable later with `u`.

**Models that are not installed.** Next to the models your server serves, each row also offers the models of the catalogue that fit your machine and are not served yet, marked `↓` with their download size (the class's own model first, then the smaller ones). On a fresh machine, where nothing is served, each row starts on the class's own model. Picking one and pressing Enter shows its licence name, a link to read it and the total download size, and nothing is downloaded until you accept with Enter or `y`; `Esc` declines. The download runs on the models screen with a progress bar, and can be cancelled with `x`.

When the downloads are finished the app asks your server which models it serves. The ones it serves are saved and used straight away, without restarting the app. The app does not manage your server, so a model that the server does not serve yet is left unsaved, and the screen says what to do: start the stack for that class (`stack/macos.sh start light`, or `docker compose -f stack/compose.yaml --profile light up`) or add the file to your llama-swap configuration, and then pick it again. The name saved for a downloaded model is the one the stack's configuration serves it under (`served_as` in `models.toml`, for example `qwen3-8b`, or `qwen-reasoner` for the reasoning model).

Saved settings live in `~/.config/voice-notes/config.toml` (Linux) or `~/Library/Application Support/voice-notes/config.toml` (macOS). Precedence is: command-line flag, then saved settings, then built-in defaults.

### Voice

`--tts kokoro` (default) asks a [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) text-to-speech server for each sentence, preparing the next sentence while the current one plays. Any server with an OpenAI-compatible `/v1/audio/speech` endpoint works; [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) is one. Configure it with:

```sh
--tts-url http://127.0.0.1:8880/v1/audio/speech   # default (Kokoro-FastAPI port)
--tts-model kokoro
--tts-voice af_heart
```

If the server cannot be reached, the app says so once and switches to the system voice (`--voice`, default `en`), so talking keeps working. `--tts system` skips Kokoro entirely.

### Models

Press `m` to open the models screen. It looks at your machine (memory for models, GPU), shows which of four size classes fits (`minimal`, `light`, `standard`, `power`), and lists the models that class starts with: speech to text, chat, text to speech and, for `power`, a reasoning model. Each row says whether the model is installed, partly downloaded, or missing, with its size. `←` `→` look at another class, `↑` `↓` choose a row, and the licence of the highlighted model is shown below the list.

`Enter` asks to download the highlighted model and `a` everything of the class that is missing. Before anything is downloaded, a screen lists each model with its licence name, a link to read it, and the size; nothing is downloaded until you accept with `Enter` or `y`, and `Esc` or `n` cancels. The downloads then run one after the other, with a progress bar. `x` cancels; what arrived is kept, and the next `Enter` continues from there. Each finished file is checked against its sha256, and a file that does not match is deleted. The models come from `models.toml`, and they are downloaded to `~/.local/share/voice-notes-stack/models` (or `$VOICE_NOTES_STACK_HOME/models`), the folder `stack/macos.sh` serves models from; set `models_dir` in the config file to use another folder. The download does not change which model the app uses: that is still chosen with `--chat-model` or `voice-notes setup`.

### Microphone and speaker

By default the app uses the system's default microphone and speaker. To use others, press `d` in the app: one row each for the microphone and the speaker, `←` `→` to change, `Enter` to save. The choice is saved in the config file and used again after a restart.

```sh
voice-notes devices                       # list them, with the default and the saved choice marked
voice-notes --input-device "Blue Yeti"    # a device's id, its name, or a unique part of the name
voice-notes --output-device pulseaudio:alsa_output.usb-headset
```

In the config file they are `input_device` and `output_device`. If a chosen device is not there (a USB microphone that is not plugged in today), the default is used and the warnings line says so, so yesterday's choice is never a dead end. Replies from Kokoro are played through the chosen speaker by the app itself on both Linux and macOS. The system voice (the fallback, `espeak-ng` or `say`) always uses the default speaker, because those tools cannot be pointed at another one. If the chosen speaker cannot be opened, the reply is played by the platform's own player (`paplay`, `pw-play`, `aplay` or `ffplay` on Linux, `afplay` on macOS) and a warning says why.

### Tuning

Problems with the microphone appear in red on the line above the key hints, never as a crash: no microphone found, access denied (on macOS the message names the System Settings page; the permission is for the terminal app you run `voice-notes` in), the device unplugged or switched off while recording, or another program using it. A recording that was running stops, and pressing Enter starts again once the problem is fixed.

If nothing appears while you speak, run `cargo run -- --plain --verbose` and compare the `ac-level` while speaking with the gate. If a quiet room reads very high, lower the microphone gain, for example `wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.3` on Linux.

## Project layout

| File | Purpose |
| --- | --- |
| `src/audio.rs` | Microphone capture, noise calibration, silence-based segmentation, WAV encoding |
| `src/devices.rs` | Lists the microphones and speakers and finds the one chosen |
| `src/playback.rs` | Plays a reply through the chosen speaker |
| `src/transcribe.rs` | Whisper and chat HTTP calls, hallucination filter |
| `src/chat.rs` | Conversation with a local model (streaming) or Claude, reasoning handling |
| `src/profile.rs` | Profiles: fast, dev, smart, think, claude and your own |
| `src/config.rs` | Saved settings |
| `src/catalogue.rs` | Reads `models.toml`: every model with its address, checksum, size and licence, and what each size class starts with |
| `models.toml` | The model catalogue, embedded in the program |
| `src/modelscreen.rs` | The models screen: the classes' models, what is installed, and what the keys do |
| `src/download.rs` | Downloads a catalogue model: resumes, checks free disk space and the sha256, reports progress |
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

3. Branch off `upstream/master`, make a focused change with tests, **run `make check` before every push**, push to your fork and open a pull request against `master`. CI runs the same checks on Linux and macOS and must pass. A smoke test also runs the whole voice loop against the service stack in `stack/` (see `stack/README.md`); you can run it yourself with `cargo build && stack/smoke.sh` if Docker is available. The container images in `stack/compose.yaml` are pinned by digest; changing one is a pull request that must pass that smoke test (see `stack/README.md`).

Other targets: `make test`, `make nextest`, `make test-cov`, `make clean`.

## Releasing

CI (`.github/workflows/ci.yml`) runs on every pull request and push to `master`. Releases are built by `.github/workflows/release.yml`:

```sh
make bump-patch                 # bump the version in Cargo.toml and Cargo.lock, then commit it
git tag v0.1.1 && git push upstream v0.1.1
```

- A `v*` tag must match the version in `Cargo.toml`; it publishes a GitHub release with a tarball and `.sha256` for Linux x64, macOS Apple Silicon and macOS Intel.
- Pushing to the `release` branch (or running the workflow manually) refreshes the rolling `latest` pre-release.
