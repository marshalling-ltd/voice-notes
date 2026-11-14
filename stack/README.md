# The voice-notes service stack

Everything the client talks to, started with one command:

```sh
cd stack
docker compose --profile light up --wait      # start and wait until every service is healthy
docker compose --profile light down           # stop (keeps the downloaded models)
docker compose --profile light down -v        # stop and delete the downloaded models too
```

Pick the profile that fits your machine (see below). From the repository root, add `-f stack/compose.yaml`: `docker compose -f stack/compose.yaml --profile light up --wait`. Run one profile at a time, because they all use port 8080. Without a profile only Kokoro starts.

`--wait` returns when the model download has finished and the services report healthy, so it can be used as a step in a script or in CI.

## Profiles

The profiles are the four size classes the client's hardware check (`voice-notes setup`) sorts a machine into, by the memory available to models. Each downloads its own models and starts a llama-swap configured for them. `ci` is the tiny set used by the smoke test.

| Profile | Memory for models | Speech to text | Chat model(s), by the name to give `--chat-model` | Download | Loaded at once |
| --- | --- | --- | --- | ---: | ---: |
| `minimal` | under 8 GB | Whisper small (488 MB) | `qwen3-4b` (Qwen3-4B Q4_K_M, 2.5 GB) | 3.0 GB | about 3.5 GB |
| `light` | 8 to 16 GB | Whisper small | `qwen3-8b` (Qwen3-8B Q4_K_M, 5.0 GB) | 5.5 GB | about 6.5 GB |
| `standard` | 16 to 24 GB | Whisper small | `qwen3-14b` (Qwen3-14B Q4_K_M, 9.0 GB) | 9.5 GB | about 11 GB |
| `power` | 24 GB or more | Whisper large-v3-turbo q5_0 (574 MB) | `qwen3-coder-30b` (Qwen3-Coder-30B-A3B UD-Q4_K_XL, 17.7 GB) and `qwen-reasoner` (Qwen3-14B Q4_K_M, 9.0 GB) | 27.2 GB | about 19 GB |
| `ci` | any | Whisper tiny.en q5_1 (32 MB) | `qwen3-0.6b` (Qwen3-0.6B Q4_K_M, 397 MB) | 430 MB | about 0.6 GB |

- **Download** is the size of the model files. **Loaded at once** is an estimate: the files in use plus room for the conversation (the context size is 4096 tokens for `minimal` and 8192 for the others). It was not measured, except that the `minimal` and `ci` profiles were started and used.
- **Whisper large on a CPU is slow** (about 11 seconds for a 2 second clip, see [GPU.md](GPU.md)), so only `power` uses it, and `power` is meant to run with the GPU (`compose.gpu.yaml`). Without a GPU, use `standard`.
- **In `power` the two chat models take turns**, because they are too big to keep together: asking for the other one unloads the first and loads the second, which takes a while. Whisper stays loaded. The other profiles keep both models loaded.
- The chat model names match the client's defaults where they exist: `qwen3-coder-30b` is what the `fast` profile asks for and `qwen-reasoner` is what `smart` and `think` ask for.
- Where the files are downloaded from, at which commit, and whether we may mirror them: [MODEL_HOSTING.md](MODEL_HOSTING.md).
- The model choice is ours: Qwen3 models for chat, because they are open (Apache-2.0), run on llama.cpp and cover every size, and Whisper for speech. The files, sizes and checksums are in `profiles/<name>.toml`.
- Only `minimal` and `ci` have been run end to end by the smoke test. The other profiles were checked for configuration and for the size of every download, but their models have not been run here.

## What runs

| Service | Image | Address | Serves |
| --- | --- | --- | --- |
| `models-<profile>` | the llama-swap image | none | One-shot job: downloads the models of the profile into the `models` volume and checks their size and sha256. Does nothing if they are already there. |
| `llama-swap-<profile>` | `ghcr.io/mostlygeek/llama-swap:unified-vulkan` (pinned) | `127.0.0.1:8080` | Chat (`/v1/chat/completions`), the model list (`/v1/models`) and speech to text (`/v1/audio/transcriptions`). It starts llama.cpp or whisper.cpp when a request names the model, and stops them when idle. Whisper stays loaded (see `groups` in the profile's `.yaml`). |
| `kokoro` | `ghcr.io/remsky/kokoro-fastapi-cpu:v0.10.0` (pinned) | `127.0.0.1:8880` | Text to speech (`/v1/audio/speech`). The model weights are part of the image. |

These are the addresses the client uses by default.

## Using it with the client

The speech-to-text model is called `whisper` (the client's default), but the chat model's name depends on the profile (see the table), so name it, for example for `light`:

```sh
voice-notes --chat-model qwen3-8b                    # interactive
voice-notes turn --input-wav question.wav --no-audio-output --chat-model qwen3-8b
```

With `power` the client's defaults already match (`qwen3-coder-30b` and `qwen-reasoner`). `voice-notes setup` lists the models the stack offers and picks from them.

## Settings

Set these in the environment or in a `stack/.env` file.

| Variable | Default | Meaning |
| --- | --- | --- |
| `STACK_BIND` | `127.0.0.1` | Address the ports are published on. See the security notes before changing it. |
| `LLAMA_SWAP_PORT` | `8080` | Host port for chat and speech to text. |
| `KOKORO_PORT` | `8880` | Host port for text to speech. |
| `KOKORO_IMAGE` | the pinned Kokoro image | Kokoro image, as `name:tag@sha256:digest`. |
| `LLAMA_SWAP_IMAGE` | the pinned llama-swap image | llama-swap image, as `name:tag@sha256:digest`. |
| `MODELS_FILE` | `./ci-models.toml` | Only for the `ci` profile: the models to download. |
| `LLAMA_SWAP_CONFIG_FILE` | `./llama-swap.yaml` | Only for the `ci` profile: the llama-swap configuration. It must match the models file, since it names the files to load. |

Paths for `MODELS_FILE` and `LLAMA_SWAP_CONFIG_FILE` are relative to the `stack/` folder. The other profiles use `profiles/<name>.toml` and `profiles/<name>.yaml`.

To run next to an existing stack that already uses 8080 and 8880, change the ports and use a separate project name: `LLAMA_SWAP_PORT=18080 KOKORO_PORT=18880 docker compose -p test -f stack/compose.yaml --profile light up --wait`.

## What to expect

Measured on a development machine (24 threads, no GPU passed to the containers; GPU numbers are in [GPU.md](GPU.md)), with the CI models:

- With the `ci` profile, the first `up --wait`, including downloading the 430 MB of models: about 90 seconds (`minimal`, 3 GB: about 70 seconds on a fast connection). Starting again with the models already in the volume: about 2 seconds.
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
- To use other models, copy a profile's `.toml` and `.yaml` in `profiles/`, change them, and point the `ci` profile at them with `MODELS_FILE` and `LLAMA_SWAP_CONFIG_FILE`. The catalogue of models with checksums for every size is ticket #22.

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
docker compose -f stack/compose.yaml --profile light up --wait
```

## Smoke test

`stack/smoke.sh` starts the stack, runs one full voice turn with `test-clip.wav` (a recording of "What is the capital of France?"), checks the result, and removes the stack again:

```sh
cargo build && stack/smoke.sh                                   # the tiny ci models, about 20 seconds
STACK_PROFILE=minimal CHAT_MODEL=qwen3-4b stack/smoke.sh        # a real size class, about 70 seconds
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
