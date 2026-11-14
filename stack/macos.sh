#!/usr/bin/env bash
# The voice-notes service stack on macOS, without containers.
#
#   stack/macos.sh install            install or build everything (once; safe to repeat)
#   stack/macos.sh start PROFILE      download the models of a profile and start the services
#   stack/macos.sh status             show whether the services are up
#   stack/macos.sh logs [SERVICE]     show the log of llama-swap or kokoro
#   stack/macos.sh stop               stop the services
#
# It runs the same services on the same ports as stack/compose.yaml: llama-swap (chat and speech
# to text) on 127.0.0.1:8080 and Kokoro (text to speech) on 127.0.0.1:8880, with the same model
# sets: PROFILE is one of minimal, light, standard, power, or ci (see README.md). The llama-swap
# configuration is the very file the container uses, with the models folder path rewritten.
# On Apple Silicon, llama.cpp and whisper.cpp use the GPU (Metal) by default, and Kokoro uses it
# through PyTorch's MPS.
#
# Everything listens on 127.0.0.1 only; there is deliberately no setting to change that.
#
# Settings, all optional (environment variables):
#   VOICE_NOTES_STACK_HOME   where everything is kept            (default: ~/.local/share/voice-notes-stack)
#   LLAMA_SWAP_PORT, KOKORO_PORT   ports to use                  (defaults: 8080, 8880)
#   STACK_SERVICES           which services to manage            (default: "llama-swap kokoro")
#   START_TIMEOUT            seconds to wait for each service    (default: 300)
#   KOKORO_DEVICE            mps (Apple GPU) or cpu for Kokoro   (default: mps)
#
# Written for the bash 3.2 that ships with macOS: no associative arrays, mapfile or ${var,,}.
# `install` only runs on macOS. The other commands work on any system that has llama-server,
# whisper-server and llama-swap on PATH (or in $VOICE_NOTES_STACK_HOME/bin), which is how this
# script is tested on Linux.

set -euo pipefail

# ---- what we install, pinned like the container images are (see README.md, "Images")

LLAMA_SWAP_VERSION="272"
LLAMA_SWAP_SHA256_ARM64="ee16e99873d76e0560000742d2524d7b97932c77c9b57a402283015839ef9d25"
LLAMA_SWAP_SHA256_AMD64="959885a60348076322629924fe06c90af37b66e5b8d69bafe3e33aa045032d44"
WHISPER_CPP_TAG="v1.9.5"
WHISPER_CPP_COMMIT="d2a764c648779feb36d44363f212a29b00aad1af"
KOKORO_TAG="v0.10.0"
KOKORO_COMMIT="f2a5270d2ccd9590bc87e2393c774814f23fe7a1"
# llama.cpp comes from Homebrew, which cannot pin a version: it is whatever Homebrew has today.
# The macOS smoke test (.github/workflows/macos.yml) is what notices when a new one breaks us.

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="${VOICE_NOTES_STACK_HOME:-$HOME/.local/share/voice-notes-stack}"
BIN="$HOME_DIR/bin"
MODELS="$HOME_DIR/models"
RUN="$HOME_DIR/run"
SRC="$HOME_DIR/src"
LLAMA_SWAP_PORT="${LLAMA_SWAP_PORT:-8080}"
KOKORO_PORT="${KOKORO_PORT:-8880}"
SERVICES="${STACK_SERVICES:-llama-swap kokoro}"
START_TIMEOUT="${START_TIMEOUT:-300}"
KOKORO_DEVICE="${KOKORO_DEVICE:-mps}"
LOOPBACK="127.0.0.1"

say() { printf 'macos-stack: %s\n' "$*"; }
fail() { printf 'macos-stack: error: %s\n' "$*" >&2; exit 1; }

# llama-swap splits the commands in its configuration on spaces, so the folders must not have any.
case "$HOME_DIR" in
  *[[:space:]]*) fail "VOICE_NOTES_STACK_HOME must not contain spaces: $HOME_DIR" ;;
esac

wants() {
  case " $SERVICES " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

need() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is needed but not installed. $2"
}

# ---- helpers

port_in_use() {
  (exec 3<>"/dev/tcp/$LOOPBACK/$1") 2>/dev/null
}

pid_file() { printf '%s/%s.pid' "$RUN" "$1"; }
log_file() { printf '%s/%s.log' "$RUN" "$1"; }

# Is the process in the pid file alive, and is it really ours (not a reused pid)?
running_pid() {
  local file pid
  file="$(pid_file "$1")"
  [[ -f "$file" ]] || return 1
  pid="$(cat "$file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  ps -p "$pid" -o command= 2>/dev/null | grep -q -E "$2" || return 1
  printf '%s' "$pid"
}

healthy() {
  curl -fsS -o /dev/null --max-time 3 "http://$LOOPBACK:$1/health" 2>/dev/null
}

wait_healthy() {
  local name="$1" port="$2" waited=0
  while ! healthy "$port"; do
    running_pid "$name" '.' >/dev/null || fail "$name stopped while starting; see $(log_file "$name")"
    if [[ "$waited" -ge "$START_TIMEOUT" ]]; then
      fail "$name did not become healthy within ${START_TIMEOUT}s; see $(log_file "$name")"
    fi
    sleep 1
    waited=$((waited + 1))
  done
  say "$name is healthy after ${waited}s"
}

# The files a profile uses, relative to this folder, as "models-file swap-config".
profile_files() {
  case "$1" in
    ci) printf '%s %s' "ci-models.toml" "llama-swap.yaml" ;;
    minimal | light | standard | power) printf '%s %s' "profiles/$1.toml" "profiles/$1.yaml" ;;
    *) fail "unknown profile '$1'; use one of: minimal, light, standard, power, ci" ;;
  esac
}

# fetch_models.py needs Python 3.11 (tomllib). macOS ships an older one, so use uv's if needed.
run_python() {
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    python3 "$@"
  elif command -v uv >/dev/null 2>&1; then
    uv run --quiet --python 3.12 "$@"
  else
    fail "Python 3.11 or newer is needed (or uv, which can fetch it). Run: stack/macos.sh install"
  fi
}

# ---- install

sha256_of() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1; else sha256sum "$1" | cut -d' ' -f1; fi
}

install_llama_swap() {
  if [[ -x "$BIN/llama-swap" ]] && "$BIN/llama-swap" -version 2>&1 | grep -q "$LLAMA_SWAP_VERSION"; then
    say "llama-swap $LLAMA_SWAP_VERSION is already installed"
    return
  fi
  local arch expected tmp url
  case "$(uname -m)" in
    arm64 | aarch64) arch="arm64"; expected="$LLAMA_SWAP_SHA256_ARM64" ;;
    x86_64) arch="amd64"; expected="$LLAMA_SWAP_SHA256_AMD64" ;;
    *) fail "no llama-swap build for $(uname -m)" ;;
  esac
  url="https://github.com/mostlygeek/llama-swap/releases/download/v${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION}_darwin_${arch}.tar.gz"
  tmp="$(mktemp -d)"
  say "downloading llama-swap $LLAMA_SWAP_VERSION"
  curl -fsSL -o "$tmp/llama-swap.tar.gz" "$url"
  [[ "$(sha256_of "$tmp/llama-swap.tar.gz")" == "$expected" ]] || fail "llama-swap download does not match its pinned sha256"
  tar -xzf "$tmp/llama-swap.tar.gz" -C "$tmp"
  mkdir -p "$BIN"
  install -m 0755 "$tmp/llama-swap" "$BIN/llama-swap"
  rm -rf "$tmp"
}

# Fetch exactly one commit of a repository into $2 (a shallow clone of a tag, then a check).
checkout_pinned() {
  local url="$1" dir="$2" tag="$3" commit="$4"
  if [[ -d "$dir/.git" ]] && [[ "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]]; then
    return
  fi
  rm -rf "$dir"
  mkdir -p "$(dirname "$dir")"
  git clone --quiet --depth 1 --branch "$tag" "$url" "$dir"
  [[ "$(git -C "$dir" rev-parse HEAD)" == "$commit" ]] || fail "$url at $tag is not the pinned commit $commit"
}

install_whisper() {
  if [[ -x "$BIN/whisper-server" ]] && [[ -f "$SRC/whisper.cpp/.pinned-$WHISPER_CPP_COMMIT" ]]; then
    say "whisper-server $WHISPER_CPP_TAG is already built"
    return
  fi
  # Homebrew's whisper.cpp is built without the server, so it has to be built from source.
  say "building whisper.cpp $WHISPER_CPP_TAG (a few minutes; Metal is used automatically on Apple Silicon)"
  checkout_pinned "https://github.com/ggml-org/whisper.cpp" "$SRC/whisper.cpp" "$WHISPER_CPP_TAG" "$WHISPER_CPP_COMMIT"
  cmake -S "$SRC/whisper.cpp" -B "$SRC/whisper.cpp/build" -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_SERVER=ON \
    -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF >/dev/null
  cmake --build "$SRC/whisper.cpp/build" --config Release --target whisper-server -j "$(sysctl -n hw.ncpu)"
  mkdir -p "$BIN"
  install -m 0755 "$SRC/whisper.cpp/build/bin/whisper-server" "$BIN/whisper-server"
  : >"$SRC/whisper.cpp/.pinned-$WHISPER_CPP_COMMIT"
}

install_kokoro() {
  local dir="$SRC/kokoro"
  if [[ -x "$dir/.venv/bin/uvicorn" ]] && [[ -f "$dir/.pinned-$KOKORO_COMMIT" ]]; then
    say "Kokoro $KOKORO_TAG is already installed"
    return
  fi
  say "installing Kokoro $KOKORO_TAG (downloads PyTorch and the voice model; a few minutes)"
  checkout_pinned "https://github.com/remsky/Kokoro-FastAPI" "$dir" "$KOKORO_TAG" "$KOKORO_COMMIT"
  (
    cd "$dir"
    uv venv --python 3.12 --quiet .venv
    uv pip install --quiet --python .venv/bin/python -e .
    .venv/bin/python docker/scripts/download_model.py --output api/src/models/v1_0
  )
  : >"$dir/.pinned-$KOKORO_COMMIT"
}

cmd_install() {
  [[ "$(uname -s)" == "Darwin" ]] || fail "install only runs on macOS. On Linux, use stack/compose.yaml."
  need brew "Install Homebrew first: https://brew.sh"
  need git "Run: xcode-select --install"
  say "installing from Homebrew: llama.cpp, cmake, ffmpeg, espeak-ng, uv"
  # ffmpeg: whisper-server --convert needs it. espeak-ng: Kokoro uses it for unknown words.
  brew install llama.cpp cmake ffmpeg espeak-ng uv
  mkdir -p "$BIN" "$MODELS" "$RUN" "$SRC"
  install_llama_swap
  install_whisper
  install_kokoro
  say "installed in $HOME_DIR. Next: stack/macos.sh start light   (or minimal, standard, power)"
}

# ---- start, stop, status

start_llama_swap() {
  local profile="$1" config
  config="$RUN/llama-swap.yaml"
  # The container sees the models at /models; here they are in $MODELS.
  sed "s| /models/| $MODELS/|g" "$here/$(profile_files "$profile" | cut -d' ' -f2)" >"$config"
  if running_pid llama-swap 'llama-swap' >/dev/null; then
    say "llama-swap is already running"
    return
  fi
  port_in_use "$LLAMA_SWAP_PORT" && fail "port $LLAMA_SWAP_PORT is already in use"
  say "starting llama-swap on $LOOPBACK:$LLAMA_SWAP_PORT"
  # PATH: our whisper-server first, then Homebrew's llama-server.
  PATH="$BIN:$PATH" nohup "$(command -v llama-swap || echo "$BIN/llama-swap")" -config "$config" \
    -listen "$LOOPBACK:$LLAMA_SWAP_PORT" >"$(log_file llama-swap)" 2>&1 &
  echo $! >"$(pid_file llama-swap)"
  wait_healthy llama-swap "$LLAMA_SWAP_PORT"
}

start_kokoro() {
  local dir="$SRC/kokoro"
  if running_pid kokoro 'uvicorn' >/dev/null; then
    say "kokoro is already running"
    return
  fi
  [[ -x "$dir/.venv/bin/uvicorn" ]] || fail "Kokoro is not installed. Run: stack/macos.sh install"
  port_in_use "$KOKORO_PORT" && fail "port $KOKORO_PORT is already in use"
  say "starting kokoro on $LOOPBACK:$KOKORO_PORT"
  # The environment is that of Kokoro's own start-gpu_mac.sh, except for the address: its script
  # listens on all interfaces by default, and this stack must not.
  (
    cd "$dir"
    USE_GPU="$([[ "$KOKORO_DEVICE" == mps ]] && echo true || echo false)" PYTHONPATH="$dir:$dir/api" \
      MODEL_DIR=src/models VOICES_DIR=src/voices/v1_0 WEB_PLAYER_PATH="$dir/web" \
      DEVICE_TYPE="$KOKORO_DEVICE" PYTORCH_ENABLE_MPS_FALLBACK=1 \
      nohup .venv/bin/uvicorn api.src.main:app --host "$LOOPBACK" --port "$KOKORO_PORT" \
      >"$(log_file kokoro)" 2>&1 &
    echo $! >"$(pid_file kokoro)"
  )
  wait_healthy kokoro "$KOKORO_PORT"
}

cmd_start() {
  [[ $# -eq 1 ]] || fail "usage: stack/macos.sh start PROFILE   (minimal, light, standard, power or ci)"
  local profile="$1" files models_file
  files="$(profile_files "$profile")"
  models_file="${files%% *}"
  need curl "curl is part of macOS; is PATH set?"
  mkdir -p "$MODELS" "$RUN"

  say "checking the models of profile '$profile' (downloads what is missing)"
  run_python -u "$here/fetch_models.py" --models-file "$here/$models_file" --dest "$MODELS"

  if wants llama-swap; then
    [[ -x "$BIN/llama-swap" ]] || command -v llama-swap >/dev/null 2>&1 || fail "llama-swap is not installed. Run: stack/macos.sh install"
    command -v llama-server >/dev/null 2>&1 || fail "llama-server is not installed. Run: stack/macos.sh install"
    [[ -x "$BIN/whisper-server" ]] || PATH="$BIN:$PATH" command -v whisper-server >/dev/null 2>&1 || fail "whisper-server is not installed. Run: stack/macos.sh install"
    start_llama_swap "$profile"
  fi
  if wants kokoro; then
    start_kokoro
  fi
  wants llama-swap && say "chat and speech to text: http://$LOOPBACK:$LLAMA_SWAP_PORT"
  wants kokoro && say "text to speech: http://$LOOPBACK:$KOKORO_PORT"
  say "up"
}

stop_one() {
  local name="$1" pattern="$2" pid waited=0
  if ! pid="$(running_pid "$name" "$pattern")"; then
    rm -f "$(pid_file "$name")"
    say "$name is not running"
    return
  fi
  kill "$pid" 2>/dev/null || true
  while kill -0 "$pid" 2>/dev/null; do
    if [[ "$waited" -ge 15 ]]; then
      kill -9 "$pid" 2>/dev/null || true
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done
  rm -f "$(pid_file "$name")"
  say "stopped $name"
}

cmd_stop() {
  # llama-swap stops the llama.cpp and whisper.cpp servers it started when it exits.
  stop_one llama-swap 'llama-swap'
  stop_one kokoro 'uvicorn'
}

cmd_status() {
  local down=0
  for pair in "llama-swap:$LLAMA_SWAP_PORT" "kokoro:$KOKORO_PORT"; do
    local name="${pair%%:*}" port="${pair##*:}"
    if healthy "$port"; then
      say "$name: healthy on $LOOPBACK:$port"
    else
      say "$name: not reachable on $LOOPBACK:$port"
      down=1
    fi
  done
  return "$down"
}

cmd_logs() {
  local name="${1:-llama-swap}"
  case "$name" in llama-swap | kokoro) ;; *) fail "usage: stack/macos.sh logs [llama-swap|kokoro]" ;; esac
  [[ -f "$(log_file "$name")" ]] || fail "no log for $name yet"
  cat "$(log_file "$name")"
}

command="${1:-}"
[[ $# -gt 0 ]] && shift
case "$command" in
  install) cmd_install "$@" ;;
  start) cmd_start "$@" ;;
  stop) cmd_stop "$@" ;;
  status) cmd_status "$@" ;;
  logs) cmd_logs "$@" ;;
  *) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; [[ -z "$command" ]] || exit 2 ;;
esac
