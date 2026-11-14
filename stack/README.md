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
| `llama-swap` | `ghcr.io/mostlygeek/llama-swap:unified-vulkan` | `127.0.0.1:8080` | Chat (`/v1/chat/completions`), the model list (`/v1/models`) and speech to text (`/v1/audio/transcriptions`). It starts llama.cpp or whisper.cpp when a request names the model, and stops them when idle. Both models stay loaded together (see `groups` in `llama-swap.yaml`). |
| `kokoro` | `ghcr.io/remsky/kokoro-fastapi-cpu:v0.10.0` | `127.0.0.1:8880` | Text to speech (`/v1/audio/speech`). The model weights are part of the image. |

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
| `KOKORO_VERSION` | `v0.10.0` | Kokoro image tag. |
| `LLAMA_SWAP_IMAGE` | `ghcr.io/mostlygeek/llama-swap:unified-vulkan` | llama-swap image. |
| `MODELS_FILE` | `./ci-models.toml` | The models to download. |
| `LLAMA_SWAP_CONFIG_FILE` | `./llama-swap.yaml` | The llama-swap configuration. It must match the models file, since it names the files to load. |

Paths for `MODELS_FILE` and `LLAMA_SWAP_CONFIG_FILE` are relative to the `stack/` folder.

To run next to an existing stack that already uses 8080 and 8880, change the ports and use a separate project name: `LLAMA_SWAP_PORT=18080 KOKORO_PORT=18880 docker compose -p test -f stack/compose.yaml up --wait`.

## What to expect

Measured on a development machine (24 threads, no GPU passed to the containers), with the CI models:

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

- CPU only. The `unified-vulkan` image can use a GPU through Vulkan, but passing the device through is not set up here.
- Images are pinned by tag, not by digest, so they can change under the same tag. Digest pinning is ticket #12.
- To use other models, point `MODELS_FILE` at another file and edit `llama-swap.yaml` to match. Ready-made sets per machine size are ticket #10.

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

If anything fails it exits non-zero, says which stage failed, and saves the result of the turn and the logs of every service in `smoke-out/` (change it with `OUT_DIR`). Settings, such as another port or model set, are listed at the top of the script. A run with the CI models takes about 20 seconds on a machine that already has the images.

CI runs it two ways:

- `.github/workflows/smoke.yml` on every pull request and every push to `master`, with the small CI models. On a failure the saved logs are attached to the run as the `smoke-logs` artifact.
- `.github/workflows/nightly.yml` every night with bigger models (`nightly-models.toml` and `llama-swap.nightly.yaml`: Whisper small and Qwen3-4B, about 3 GB), which catches changes in the upstream images and models early. Its logs are the `nightly-logs` artifact.

`test-clip.wav` is 1.8 seconds of speech made with Kokoro (Apache-2.0) and converted to 16 kHz mono.

## Checks

```sh
python3 -m unittest discover -s stack     # the model downloader, and that the llama-swap configs match their catalogues
```
