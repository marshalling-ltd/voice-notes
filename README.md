# Voice Notes

Speech-to-text CLI. Press Enter, talk, and text appears after each pause while you are still recording. Optionally cleans the final transcript (grammar, punctuation, capitalization).

## How it works

- Records from the default microphone with `cpal` (PulseAudio host, served by PipeWire or PulseAudio; falls back to the ALSA default).
- Measures about one second of background noise at the start and sets a speech gate from it, so it adapts to noisy or hot microphones.
- Cuts an utterance after a pause and transcribes it against a local Whisper-compatible server (`http://127.0.0.1:8080/v1/audio/transcriptions`) while recording continues.
- Drops weak segments and likely Whisper hallucinations (stock phrases, subtitle credits, looping text).
- With `--clean`, the combined transcript is polished through `/v1/chat/completions` on the same server.

## Usage

```sh
cargo run                       # live transcription
cargo run -- --clean            # also clean up the final transcript
cargo run -- --language hr      # spoken language (default en, `auto` to detect)
cargo run -- --verbose          # show the live input level and calibrated gate
cargo run -- --threshold 0.02   # minimum speech level (default 0.01)
```

If text lags or nothing appears, run with `--verbose` and compare the `ac-level` while speaking with the gate. If a quiet room reads very high, lower the microphone gain, for example `wpctl set-volume @DEFAULT_AUDIO_SOURCE@ 0.3`.

## Project layout

| File | Purpose |
| --- | --- |
| `src/audio.rs` | Microphone capture, noise calibration, silence-based segmentation, WAV encoding |
| `src/transcribe.rs` | Whisper and chat HTTP calls, hallucination filter |
| `src/main.rs` | CLI, live loop, ordered transcription worker |

## Contributing

1. Fork `marshalling-ltd/voice-notes` and clone your fork (`origin`); add the org repo as `upstream`.
2. One-time setup (Rust nightly, cargo tools and the ALSA headers):

   ```sh
   make requirements   # needs: sudo apt-get install -y libasound2-dev pkg-config build-essential
   cargo run           # start the app
   make check          # cargo update/upgrade, machete, audit, deny, typos, fmt, clippy pedantic, tests
   ```

3. Branch off `upstream/master`, make a focused change with tests, **run `make check` before every push**, push to your fork and open a pull request against `master`. CI runs the same checks and must pass.

Other targets: `make test`, `make nextest`, `make test-cov`, `make clean`.
