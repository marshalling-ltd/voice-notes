#!/usr/bin/env bash
# Smoke test: start the stack, run one full voice turn with a recording, and check the result.
#
# Used by CI (.github/workflows/smoke.yml and nightly.yml) and runnable by hand:
#
#   cargo build && stack/smoke.sh
#
# It checks that a recording of "What is the capital of France?" is transcribed, that the chat
# model answers in a streamed reply, and that the speech service returns audio. The exit code is
# 0 only if all of that held. The stack is removed afterwards, and if anything failed the logs of
# every service are saved in $OUT_DIR.
#
# Settings, all optional (environment variables):
#   VOICE_NOTES           the client binary                   (default: target/debug/voice-notes)
#   CLIP                  the recording to send               (default: stack/test-clip.wav)
#   STACK_PROFILE         compose profile to start            (default: ci)
#   CHAT_MODEL            model name to ask for chat          (default: qwen3-0.6b; with another
#                         profile, name its chat model, e.g. qwen3-4b for minimal)
#   LLAMA_SWAP_PORT, KOKORO_PORT   host ports to use          (defaults: 8080, 8880)
#   MODELS_FILE, LLAMA_SWAP_CONFIG_FILE   a different model set (paths relative to stack/)
#   COMPOSE_PROJECT       compose project name                (default: voice-notes-smoke)
#   OUT_DIR               where results and logs go           (default: smoke-out)
#   WAIT_TIMEOUT          seconds to wait for the stack       (default: 900)
#   KEEP_STACK            if set, leave the stack running afterwards
#   STACK_NATIVE          if set, test the native macOS stack (macos.sh) instead of containers;
#                         the profile's models are installed and run without Docker
#   STACK_GPU             if set, also pass the GPU through to llama-swap (compose.gpu.yaml)

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"

VOICE_NOTES="${VOICE_NOTES:-$root/target/debug/voice-notes}"
CLIP="${CLIP:-$here/test-clip.wav}"
CHAT_MODEL="${CHAT_MODEL:-qwen3-0.6b}"
PROJECT="${COMPOSE_PROJECT:-voice-notes-smoke}"
OUT="${OUT_DIR:-$root/smoke-out}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-900}"
export LLAMA_SWAP_PORT="${LLAMA_SWAP_PORT:-8080}" KOKORO_PORT="${KOKORO_PORT:-8880}"

compose=(docker compose -p "$PROJECT" -f "$here/compose.yaml" --profile "${STACK_PROFILE:-ci}")
if [[ -n "${STACK_GPU:-}" ]]; then
  compose+=(-f "$here/compose.gpu.yaml")
fi

say() { printf 'smoke: %s\n' "$*"; }
fail() { printf 'smoke: FAIL: %s\n' "$*" >&2; exit 1; }

collect_logs() {
  if [[ -n "${STACK_NATIVE:-}" ]]; then
    cp "${VOICE_NOTES_STACK_HOME:-$HOME/.local/share/voice-notes-stack}"/run/*.log "$OUT/" 2>/dev/null || true
    return
  fi
  "${compose[@]}" ps -a >"$OUT/compose-ps.txt" 2>&1 || true
  "${compose[@]}" logs --no-color >"$OUT/compose-logs.txt" 2>&1 || true
}

cleanup() {
  local status=$?
  if [[ $status -ne 0 ]]; then
    say "failed with exit code $status; saving the service logs in $OUT"
    collect_logs
  fi
  if [[ -z "${KEEP_STACK:-}" ]]; then
    if [[ -n "${STACK_NATIVE:-}" ]]; then
      "$here/macos.sh" stop >/dev/null 2>&1 || true
    else
      "${compose[@]}" down -v >/dev/null 2>&1 || true
    fi
  fi
  exit "$status"
}

mkdir -p "$OUT"
trap cleanup EXIT

[[ -x "$VOICE_NOTES" ]] || fail "the client is not built: $VOICE_NOTES (run cargo build)"
[[ -f "$CLIP" ]] || fail "the test recording is missing: $CLIP"
if [[ -z "${STACK_NATIVE:-}" ]]; then
  docker info >/dev/null 2>&1 || fail "cannot talk to Docker (is the daemon running, and are you allowed to use it?)"
fi

say "starting the stack (profile: ${STACK_PROFILE:-ci}; models: ${MODELS_FILE:-from the profile})"
start=$(date +%s)
if [[ -n "${STACK_NATIVE:-}" ]]; then
  START_TIMEOUT="$WAIT_TIMEOUT" "$here/macos.sh" start "${STACK_PROFILE:-ci}"
else
  "${compose[@]}" up -d --wait --wait-timeout "$WAIT_TIMEOUT"
fi
say "the stack is healthy after $(( $(date +%s) - start )) s"

say "running one voice turn with $(basename "$CLIP")"
set +e
"$VOICE_NOTES" turn --input-wav "$CLIP" --no-audio-output --json \
  --server "http://127.0.0.1:${LLAMA_SWAP_PORT}" \
  --tts-url "http://127.0.0.1:${KOKORO_PORT}/v1/audio/speech" \
  --chat-model "$CHAT_MODEL" >"$OUT/turn.json" 2>"$OUT/turn.stderr"
code=$?
set -e
if [[ $code -ne 0 ]]; then
  case $code in
    10) stage="reading the recording" ;;
    11) stage="transcribing" ;;
    12) stage="asking the chat model" ;;
    13) stage="speaking the reply" ;;
    *)  stage="starting the client" ;;
  esac
  cat "$OUT/turn.stderr" >&2
  fail "the voice turn failed while ${stage} (exit code ${code})"
fi

python3 - "$OUT/turn.json" "$OUT/summary.md" <<'PY' || fail "the voice turn did not meet the expectations above"
import json
import sys

turn = json.load(open(sys.argv[1]))
transcript = turn.get("transcript", "").strip()
reply = turn.get("reply", "").strip()
chunks = turn.get("reply_chunks", 0)
audio = turn.get("audio_bytes") or 0
timings = turn.get("timings_ms", {})

checks = [
    ("the recording was transcribed", bool(transcript), repr(transcript)),
    (
        "the transcript matches the recording",
        "capital" in transcript.lower() or "france" in transcript.lower(),
        "expected it to mention the capital or France",
    ),
    ("the chat model answered", bool(reply), repr(reply)),
    ("the reply was streamed", chunks >= 2, f"it arrived in {chunks} piece(s)"),
    ("the speech service returned audio", turn.get("speech") == "discarded" and audio >= 10_000, f"{audio} bytes"),
]

lines = ["| Check | Result | Detail |", "| --- | --- | --- |"]
ok = True
for name, passed, detail in checks:
    print(f"smoke: {'ok  ' if passed else 'FAIL'} {name} ({detail})")
    lines.append(f"| {name} | {'pass' if passed else '**FAIL**'} | {detail} |")
    ok = ok and passed
lines.append("")
lines.append(
    "Timings: transcribe {transcribe} ms, chat {chat} ms, speech {speech} ms.".format(
        transcribe=timings.get("transcribe"), chat=timings.get("chat"), speech=timings.get("speech")
    )
)
print("smoke: timings: transcribe {transcribe} ms, chat {chat} ms, speech {speech} ms".format(
    transcribe=timings.get("transcribe"), chat=timings.get("chat"), speech=timings.get("speech")))
open(sys.argv[2], "w").write("\n".join(lines) + "\n")
sys.exit(0 if ok else 1)
PY

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  { echo "### Smoke test"; echo; cat "$OUT/summary.md"; } >>"$GITHUB_STEP_SUMMARY"
fi
say "passed: transcribed, answered in a streamed reply, and spoke it"
