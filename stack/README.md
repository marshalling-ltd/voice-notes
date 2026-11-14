# The voice-notes service stack

Everything the client talks to, started with one command:

```sh
docker compose -f stack/compose.yaml up --wait      # start and wait until every service is healthy
docker compose -f stack/compose.yaml down           # stop (keeps the downloaded models)
docker compose -f stack/compose.yaml down -v        # stop and delete the downloaded models too
```

`--wait` returns when the model download has finished and both services report healthy, so it can be used as a step in a script or in CI.

## What runs

| Service | Image | Address | Serves |
| --- | --- | --- | --- |
| `models` | the llama-swap image | none | One-shot job: downloads the models in `ci-models.toml` into the `models` volume and checks their size and sha256. Does nothing if they are already there. |
| `llama-swap` | `ghcr.io/mostlygeek/llama-swap:unified-vulkan` (pinned) | `127.0.0.1:8080` | Chat (`/v1/chat/completions`), the model list (`/v1/models`) and speech to text (`/v1/audio/transcriptions`). It starts llama.cpp or whisper.cpp when a request names the model, and stops them when idle. Both models stay loaded together (see `groups` in `llama-swap.yaml`). |
| `kokoro` | `ghcr.io/remsky/kokoro-fastapi-cpu:v0.10.0` (pinned) | `127.0.0.1:8880` | Text to speech (`/v1/audio/speech`). The model weights are part of the image. |

These are the addresses the client uses by default. The default model set is the small one chosen for CI (`ci-models.toml`): Whisper tiny.en, Qwen3-0.6B and Kokoro.

## Using it with the client

The speech-to-text model is called `whisper` (the client's default), but the chat model is called `qwen3-0.6b`, so name it:

```sh
voice-notes --chat-model qwen3-0.6b                    # interactive
voice-notes turn --input-wav question.wav --no-audio-output --chat-model qwen3-0.6b
```

## Settings

Set these in the environment or in a `stack/.env` file.

| Variable | Default | Meaning |
| --- | --- | --- |
| `STACK_BIND` | `127.0.0.1` | Address the ports are published on. See the security notes before changing it. |
| `LLAMA_SWAP_PORT` | `8080` | Host port for chat and speech to text. |
| `KOKORO_PORT` | `8880` | Host port for text to speech. |
| `KOKORO_IMAGE` | the pinned Kokoro image | Kokoro image, as `name:tag@sha256:digest`. |
| `LLAMA_SWAP_IMAGE` | the pinned llama-swap image | llama-swap image, as `name:tag@sha256:digest`. |
| `MODELS_FILE` | `./ci-models.toml` | The models to download. |
| `LLAMA_SWAP_CONFIG_FILE` | `./llama-swap.yaml` | The llama-swap configuration. It must match the models file, since it names the files to load. |

Paths for `MODELS_FILE` and `LLAMA_SWAP_CONFIG_FILE` are relative to the `stack/` folder.

To run next to an existing stack that already uses 8080 and 8880, change the ports and use a separate project name: `LLAMA_SWAP_PORT=18080 KOKORO_PORT=18880 docker compose -p test -f stack/compose.yaml up --wait`.

## What to expect

Measured on a development machine (24 threads, no GPU passed to the containers; GPU numbers are in [GPU.md](GPU.md)), with the CI models:

- First `up --wait`, including downloading the 430 MB of models: about 90 seconds. Starting again with the models already in the volume: about 2 seconds.
- A voice turn (`voice-notes turn`) once the models are loaded: about 0.65 seconds in all (transcribe 0.25 s, chat 0.07 s, speech 0.33 s). The first turn after a start also loads the models and takes a few seconds longer.

Two settings in `llama-swap.yaml` exist because the stack hit the problems they fix when it was first tried: whisper.cpp needs `--tmp-dir /tmp` because `/models` is read-only, and the `voice` group keeps both models loaded, because by default llama-swap swaps them on every request and adds a model load to every turn.

## Health checks

- `llama-swap`: `GET /health` on the llama-swap server. This shows llama-swap is up, not that a model is loaded: models load on first use, so the first request after a start is slow.
- `kokoro`: `GET /health` on the Kokoro server.
- `models`: not a health check; the other services wait for it to finish successfully.

## Security

- Every port is published on `127.0.0.1` only, so nothing is reachable from other machines.
- There is no authentication. If you set `STACK_BIND` to a reachable address, anyone who can connect can use your models and your CPU or GPU. Put something with authentication in front of it first.
- The llama-swap image runs as root inside its container, which is how the image is built.

## Limits for now

- CPU by default. A GPU can be used through Vulkan with `compose.gpu.yaml`; see [GPU.md](GPU.md) for how, what it measured, and when it is worth it.
- To use other models, point `MODELS_FILE` at another file and edit `llama-swap.yaml` to match. Ready-made sets per machine size are ticket #10.

## Images

Both images are pinned by digest in `compose.yaml`, written as `name:tag@sha256:digest`. A tag such as `unified-vulkan` can be moved to a different image at any time, which would change what the stack and CI run without anyone changing this repository. A digest names exactly one image, and Docker refuses a digest that does not exist instead of falling back to the tag. The tag stays in the reference so people can read which version it is.

**Mirror, not build.** We use the upstream images as they are. Upstream already publishes them on `ghcr.io`, and building our own would mean maintaining builds of llama.cpp, whisper.cpp and Kokoro for no benefit. When we want independence from upstream (it deleting or rewriting an image), we copy the pinned images to our own namespace instead, see below.

**Policy: changing an image is a pull request that must pass the smoke test.**

1. Run `python3 stack/pin_images.py --update`. It asks the registry for the digest each tag points to now and rewrites `compose.yaml`. To move to a new version, change the tag first (for example Kokoro's `v0.10.0`), then run it.
2. Review `git diff` and open a pull request. The smoke test runs on every pull request, pulls the new images and runs a full voice turn, and it must pass before the change is merged.
3. A pull request that adds an image without a digest fails the first step of the smoke workflow (`pin_images.py --check`, which also runs in the unit tests).

`pin_images.py --list` shows every image and whether it is pinned.

**Mirroring.** The `Mirror images` workflow (`.github/workflows/mirror-images.yml`, started by hand from the Actions tab) copies each pinned image to `ghcr.io/<owner>/voice-notes-images/<name>:<tag>`, registry to registry, so the multi-architecture manifest and the digest stay the same. It fails if a copy ends up with a different digest. To run the stack from the mirror:

```sh
LLAMA_SWAP_IMAGE=ghcr.io/<owner>/voice-notes-images/llama-swap@sha256:<digest from compose.yaml> \
KOKORO_IMAGE=ghcr.io/<owner>/voice-notes-images/kokoro-fastapi-cpu@sha256:<digest from compose.yaml> \
docker compose -f stack/compose.yaml up --wait
```

## Smoke test

`stack/smoke.sh` starts the stack, runs one full voice turn with `test-clip.wav` (a recording of "What is the capital of France?"), checks the result, and removes the stack again:

```sh
cargo build && stack/smoke.sh
```

It passes only if all of these hold, and prints one line for each:

- the recording was transcribed, and the transcript mentions the capital or France;
- the chat model answered;
- the reply was streamed (it arrived in more than one piece);
- the speech service returned audio.

If anything fails it exits non-zero, says which stage failed, and saves the result of the turn and the logs of every service in `smoke-out/` (change it with `OUT_DIR`). Settings, such as another port, a model set, or `STACK_GPU=1` to pass the GPU through, are listed at the top of the script. A run with the CI models takes about 20 seconds on a machine that already has the images.

CI runs it two ways:

- `.github/workflows/smoke.yml` on every pull request and every push to `master`, with the small CI models. On a failure the saved logs are attached to the run as the `smoke-logs` artifact.
- `.github/workflows/nightly.yml` every night with bigger models (`nightly-models.toml` and `llama-swap.nightly.yaml`: Whisper small and Qwen3-4B, about 3 GB), which catches changes in the upstream images and models early. Its logs are the `nightly-logs` artifact.

`test-clip.wav` is 1.8 seconds of speech made with Kokoro (Apache-2.0) and converted to 16 kHz mono.

## Checks

```sh
python3 -m unittest discover -s stack     # the model downloader, the image pinning, and that the llama-swap configs match their catalogues
python3 stack/pin_images.py --check       # every image is pinned by digest
```
