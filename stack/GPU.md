# Using the GPU in the stack

Ticket #13: can llama.cpp and whisper.cpp use an AMD GPU through Vulkan inside the stack's container, how fast is it compared with running natively, and what is the fallback?

**Short answer: yes, it works, and it is worth it for bigger models and for speech to text.** It is opt-in, and the stack runs on the CPU by default.

```sh
docker compose -f stack/compose.yaml -f stack/compose.gpu.yaml --profile power up --wait
```

`compose.gpu.yaml` only passes `/dev/dri` into the `llama-swap-<profile>` containers; use it with any profile. Nothing else changes, and Kokoro stays on the CPU because its image is a CPU build. The smoke test accepts `STACK_GPU=1` to run with it.

## What was tested

One machine: AMD Radeon 890M (RADV, an APU that shares system memory, 80 GB visible to the GPU), Ubuntu with Mesa 26.1.6, Docker 29. The container image was the pinned `llama-swap:unified-vulkan`.

- **Without the device**, llama.cpp in the container sees no devices and runs on the CPU. This is what CI and the default stack get.
- **With `--device /dev/dri`**, it sees `Vulkan0: AMD Radeon 890M Graphics (RADV GFX1150)` with the whole 80 GB pool. The container runs as root, so no `render` or `video` group setup was needed.
- Our exact `llama-server` command (no GPU flags) puts the model on `Vulkan0` by itself.
- The full stack with the GPU override passed the smoke test.

## Speed

llama-bench, tokens per second (higher is better). `pp512` is prompt processing, which decides how long you wait for the first word of a reply; `tg128` is generation, the speed after that.

| Model | Test | Native GPU | Container GPU | Container CPU |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B (Q4_K_M) | prompt | 4656 | 4151 | 873 |
| | generation | 158 | 158 | 136 |
| Qwen2.5-VL-7B (Q4_K_M) | prompt | 565 | 422 | 77 |
| | generation | 16.5 | 17.5 | 16.0 |
| Qwen3-Coder-30B-A3B (Q4_K_XL) | prompt | 572 | 503 | 126 |
| | generation | 37.7 | 37.7 | 30.3 |

Whisper large-v3-turbo (q5_0), a 1.8 second clip, encoder time:

| Native GPU | Container GPU | Container CPU (default 4 threads) |
| ---: | ---: | ---: |
| 0.55 s | 0.75 s | 10.8 s |

The whole stack with the small CI models, one warm voice turn: about **0.43 s on the GPU** against 0.65 s on the CPU (transcribe 0.09 s against 0.25 s).

## What this means

- **Prompt processing is 4 to 6 times faster on the GPU, and Whisper's encoder about 15 times** (about 9 times for the whole transcription). For the `power` profile (a 30B model, large Whisper), that is the difference between a first word after 0.2 s and after 0.8 s for a 100-token prompt, and a transcript in about a second instead of eleven.
- **Generation barely improves** (up to about 1.25 times). On an APU the GPU and the CPU share the same memory, and generation is limited by memory bandwidth, so there is little to gain. A machine with a dedicated GPU would differ; that was not tested.
- **The container matches native for generation and is 10 to 35 percent slower for prompt and audio processing.** That gap should not be blamed on the container alone: the image has an older graphics driver (Mesa 25.2.8 against 26.1.6 on the host) and a different llama.cpp build. It was not separated out.
- **The first load of a model on the GPU is slower** (the first chat request took 3.4 s against 1.4 s on the CPU with the small model), after which requests are fast. Models stay loaded (see the `voice` group in `llama-swap.yaml`), so this is paid once per start.
- llama.cpp prints a warning that flash attention for some layers runs on the CPU. It is harmless here.

Caveats: single runs on one machine that was also running its own services; the CPU speech-to-text number used whisper.cpp's default of 4 threads, which is also what the stack uses; NVIDIA, Intel, discrete AMD cards and ROCm were not tested.

## Decision for the `power` profile

The `power` profile can use the GPU in containers, on machines with Vulkan and a `/dev/dri` device. The default stays CPU, so that:

- CI (GitHub runners have no GPU) and every other machine work unchanged;
- a missing or wrong device never stops the stack from starting.

The `power` profile is meant to be started with `compose.gpu.yaml`; the profile table in `README.md` says so.

## What it needs from the host

- A GPU with a working Vulkan driver, and `/dev/dri` present (`ls /dev/dri` shows `card*` and `renderD*`).
- Docker that can pass devices through. Rootless Docker or Podman may need the user to be in the `render` and `video` groups and `group_add` in the override.
- Docker Desktop on macOS and Windows does not pass a GPU into containers, so the stack stays on the CPU there. On macOS, run the services natively instead (ticket #30).

## Fallback: run it natively

If the container route does not work on a machine (an unusual driver, no device passthrough), run llama-swap natively with Vulkan, as the development machine does today, and point the client at it (`--server http://127.0.0.1:8080`). Native was the fastest setup measured. Kokoro can still come from the stack: `docker compose -f stack/compose.yaml up --wait kokoro`.
