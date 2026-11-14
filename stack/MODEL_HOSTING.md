# Where the models come from

Ticket #29 (VN-23): where do the catalogue's download URLs point for the first weeks, what do we do later, and may we redistribute each model?

## Decision

**For now, every model is downloaded from Hugging Face, from the repository that publishes it, at a fixed commit.** No mirror of our own yet.

- It costs nothing and needs nothing from us: no bucket, no billing, no upload step, no access keys to guard.
- The files are already there, public and not gated: nobody has to accept a licence or log in to download them.
- Our downloader (`fetch_models.py`) checks the size and sha256 of every file against the catalogue. So the host does not have to be trusted: a changed, truncated or substituted file is rejected, and nothing is put in place.
- The URLs name a **commit** (`.../resolve/<40 hex>/file`), not `main`. A branch can be rewritten or a file replaced; a commit cannot. This is the same idea as pinning container images by digest (see `README.md`). A test keeps `resolve/main` out of the catalogues.

| Repository | Commit |
| --- | --- |
| `ggerganov/whisper.cpp` | `5359861c739e` |
| `unsloth/Qwen3-0.6B-GGUF` | `50968a4468ef` |
| `unsloth/Qwen3-4B-GGUF` | `22c9fc8a8c76` |
| `unsloth/Qwen3-8B-GGUF` | `a6adef130ffb` |
| `unsloth/Qwen3-14B-GGUF` | `a04a82c4739b` |
| `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` | `b17cb02dd882` |
| `hexgrad/Kokoro-82M` | `f3ff3571791e` |

To move to a newer version of a model, change the commit and the size and sha256 together in the catalogue (the sha256 is on the file's page on Hugging Face), and let the smoke test pass, like any other change.

## What this does not protect against

- **The repository is deleted, renamed or made private.** Then the download fails with a clear error, because there is nothing to fetch. Commits survive normal edits, but not a deletion. This is the risk a mirror would remove.
- **Hugging Face is down or slow.** Downloads fail or take long; models already downloaded are unaffected, because the volume keeps them.
- **Whole-catalogue download volume.** The files for all profiles are about 35 GB, and one `power` install is 27 GB. Hugging Face serves these without limits that we have hit, but we did not test heavy parallel use, for example many CI runners at once. The CI models are small (430 MB) for that reason.

## When to move to our own storage

Move when one of these happens, not before:

1. a model we depend on disappears or is replaced upstream, or
2. downloads from Hugging Face become a problem for users (rate limits, regions where it is slow or blocked), or
3. we want to ship models that upstream does not offer in the form we need (for example a re-quantised file).

**Where to:** an S3-compatible bucket, with Cloudflare R2 or Backblaze B2 as the candidates. Both speak the S3 protocol, so the catalogue only changes its `url` values and nothing in the client or the downloader does. The deciding factor is the cost of data leaving the bucket, since every install downloads several gigabytes: compare current prices for storage and for downloads on both before choosing; they were not looked up for this note and change over time. Put the files behind a custom domain we control, so moving again later does not change the URLs. Keep the sha256 values as they are: they stay valid wherever the same file is hosted, which is what makes a move safe.

## May we redistribute them?

This decides whether we may mirror a model, which we would do in the cases above. Licences are those stated on each model's Hugging Face card, checked on 2026-11-14, including the original model that each file was converted from. They were **not independently audited and this is not legal advice**; read the licence text before the first real mirror.

| Model | Used in | Licence | Redistribution |
| --- | --- | --- | --- |
| Whisper (tiny.en, base.en, small, large-v3-turbo) from `ggerganov/whisper.cpp`, converted from OpenAI Whisper | all profiles | MIT | Yes, keep the licence notice. |
| Qwen3-0.6B, 4B, 8B, 14B (`unsloth/...-GGUF`, converted from `Qwen/Qwen3-*`) | `ci`, `minimal`, `light`, `standard`, `power` | Apache-2.0 | Yes, keep the licence and state changes (a conversion to GGUF is one). |
| Qwen3-Coder-30B-A3B (`unsloth/...-GGUF`, from `Qwen/Qwen3-Coder-30B-A3B-Instruct`) | `power` | Apache-2.0 | Yes, same terms. |
| Kokoro-82M (weights inside the Kokoro container image; listed in `ci-models.toml` for reference) | all | Apache-2.0 (built on StyleTTS 2) | Yes, same terms. Not downloaded by us: the image is mirrored instead (`README.md`). |

None of these repositories is gated, and none needs the user to accept terms before download. Consequences for a mirror:

- Include each model's licence file and a note of where it came from and what was changed (the file was quantised or converted by a third party, named in the catalogue URL).
- The quantised files are third-party conversions (Unsloth, whisper.cpp's author). Their cards state the same licence as the original; keep that attribution.
- Adding a model later means doing this check again for that model. A licence such as Llama's or Gemma's has conditions (use policy, naming, size thresholds) that Apache-2.0 and MIT do not, and would need a closer read before mirroring.
