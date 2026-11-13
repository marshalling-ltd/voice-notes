# Old-machine test checklist

Goal: find out what breaks when someone other than the author tries the app on ordinary hardware. Note everything, including things that "just worked". Write timings in seconds.

## 0. Before leaving (on the main machine)

- [ ] Services are up: `curl -s 127.0.0.1:8080/v1/models | head -c 200` lists models, and `curl -s -o /dev/null -w "%{http_code}\n" -X POST 127.0.0.1:8880/v1/audio/speech -H 'Content-Type: application/json' -d '{"model":"kokoro","input":"Hi.","voice":"af_heart","response_format":"wav"}'` prints `200`.
- [ ] The main machine is reachable from the old one (same network, SSH works).
- [ ] This checklist and a way to copy files back (a USB stick, or `scp`).

## 1. Describe the old machine

Run and paste the output into the notes:

```
uname -a ; lscpu | head -20 ; free -h ; lspci | grep -i -E "vga|3d|display"
pactl info 2>&1 | head -5 ; aplay -l ; arecord -l
echo "$XDG_SESSION_TYPE  terminal: $(tput cols)x$(tput lines)"
```

Which sound system does it use (PipeWire, PulseAudio, plain ALSA)? Is a microphone present and not muted?

## 2. Get the app running

Pick one.

**A. Build on the old machine** (most reliable): install Rust with `rustup` (distribution packages are usually too old), install `libasound2-dev pkg-config build-essential`, then `cargo build --release`. Note how long it took.

**B. Copy a binary** built elsewhere. If it fails with a `GLIBC_` error, the old machine's Linux is older than the build machine's: use A.

- [ ] The app starts: `./voice-notes --help` prints help.

## 3. Choose the model setup

**Test A: client only, models on the main machine.** On the old machine run, and keep it open:

```
ssh -N -L 8080:127.0.0.1:8080 -L 8880:127.0.0.1:8880 <user>@<main-machine>
```

This checks microphone, speakers, terminal and first-run behaviour, but **not** how the models perform on weak hardware.

**Test B: everything on the old machine** with small models (Whisper small, a chat model of about 4B, Kokoro or the system voice). Needs manual setup. This checks the minimal and light size classes and the latency advice. Note every setup step that was painful.

## 4. Tests

Mark each ✔ or ✘ and write what you saw.

| # | Do this | Expect | Result | Notes |
| --- | --- | --- | --- | --- |
| 1 | Delete `~/.config/voice-notes`, run `./voice-notes setup --check` | A report: services, machine, size class, speed. Takes under a minute. | | time: |
| 2 | Run `./voice-notes setup --check --quick` | Same without the speed section | | |
| 3 | Run `./voice-notes` in a terminal (delete the config folder first) | Setup screen appears on first launch; `s` skips it, Enter saves | | |
| 4 | Look at the main screen at 80x24, then resize wider/narrower | Nothing is cut off or garbled; footer keys readable | | |
| 5 | Press Enter, stay quiet 2 seconds, then speak a sentence, pause, speak another, Enter | Status shows CALIBRATING, then LISTENING. Meter moves with your voice. Each sentence appears after its pause. | | calibrated gate: |
| 6 | Same, in a noisy room or with the mic very loud/quiet | Still segments sensibly; if not, note the meter values (`--plain --verbose`) | | |
| 7 | Press `t` for talk mode and ask a question | Reply streams in; spoken aloud; your words shown as you speak | | time from Enter (stop) to first spoken word: |
| 8 | Press Enter while it speaks | Speech stops | | |
| 9 | Press `r` after a reply | Last reply is read aloud again | | |
| 10 | Press `p` several times | Profile name in the header changes; each profile answers | | which profiles appeared: |
| 11 | Press `n`, ask a follow-up that needs the earlier answer | It does not remember (new conversation) | | |
| 12 | Quit with `q`; then run `reset` only if the terminal looks broken | Terminal is restored cleanly | | |
| 13 | `./voice-notes --plain` (no full-screen UI) | Line-by-line mode works | | |
| 14 | `./voice-notes --talk --tts system` | Speaks with the system voice (`espeak-ng`) | | |

## 5. Break things on purpose

| # | Do this | Expect | Result |
| --- | --- | --- | --- |
| 15 | Stop the SSH tunnel (or the servers) mid-session, then speak | A message in the footer or reply pane; no crash | |
| 16 | Stop only the speech server | Text still appears; one notice about falling back to the system voice | |
| 17 | Mute or unplug the microphone, press Enter | A readable error, no crash | |
| 18 | Start with no microphone at all | A readable error, no crash | |
| 19 | Press keys quickly and repeatedly (Enter, Enter, Enter; `t`, `p`) | No stuck state, no crash | |

## 6. What to send back

- The output of section 1 and of test 1 (`setup --check`).
- The exact text of any error, and what you pressed just before it.
- For any crash: re-run with `RUST_BACKTRACE=1` and copy the backtrace.
- The timings: build time, first launch, calibration, stop-to-first-spoken-word.
- Your overall impression: what felt slow, confusing or broken.

## Known limits (not bugs for this test)

- The model servers are not installed for you; the managed install is not built yet.
- Windows and macOS are not part of this test.
- The first reply after a model has been idle can take many seconds while it loads.
- The `think` profile is slow on purpose.
