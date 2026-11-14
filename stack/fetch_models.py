#!/usr/bin/env python3
"""Download the models listed in a models TOML file into a folder, and check them.

Every file is downloaded to a temporary name, its size and sha256 are checked against the
catalogue, and only then is it moved into place, so a half-finished or corrupt download is never
left where a service would load it. Files that are already present and correct are left alone.

Only the Python standard library is used, so this runs inside the llama-swap image.
"""

import argparse
import hashlib
import os
import sys
import time
import tomllib
import urllib.error
import urllib.request
from urllib.parse import unquote, urlparse

# Kokoro's weights are part of its container image, so they are not downloaded here.
DEFAULT_ROLES = "speech-to-text,chat"
RETRIES = 3


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_name(url):
    return os.path.basename(unquote(urlparse(url).path))


def is_present_and_correct(path, model):
    return (
        os.path.isfile(path)
        and os.path.getsize(path) == model["size_bytes"]
        and sha256_of(path) == model["sha256"]
    )


def download(url, part_path, expected_size, log):
    """Stream `url` into `part_path`, returning the sha256 of what was written."""
    request = urllib.request.Request(url, headers={"User-Agent": "voice-notes-stack"})
    digest = hashlib.sha256()
    written = 0
    next_report = 0.25
    with urllib.request.urlopen(request, timeout=60) as response, open(part_path, "wb") as out:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            out.write(block)
            digest.update(block)
            written += len(block)
            if expected_size and written / expected_size >= next_report:
                log(f"    {int(written / expected_size * 100)}%")
                next_report += 0.25
    return digest.hexdigest(), written


def fetch(model, dest, log=print, sleep=time.sleep):
    """Make sure `model` is in `dest`. Returns "present" or "downloaded"; raises on failure."""
    name = file_name(model["url"])
    final = os.path.join(dest, name)
    if is_present_and_correct(final, model):
        log(f"  {name}: already present and correct")
        return "present"

    part = final + ".part"
    last_error = None
    for attempt in range(1, RETRIES + 1):
        log(f"  {name}: downloading ({model['size_bytes'] / 1e6:.0f} MB), attempt {attempt}")
        try:
            got_sha, got_size = download(model["url"], part, model["size_bytes"], log)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last_error = f"download failed: {e}"
        else:
            if got_size != model["size_bytes"]:
                last_error = f"wrong size: got {got_size} bytes, expected {model['size_bytes']}"
            elif got_sha != model["sha256"]:
                last_error = f"wrong sha256: got {got_sha}, expected {model['sha256']}"
            else:
                os.replace(part, final)
                log(f"  {name}: downloaded and verified")
                return "downloaded"
        log(f"  {name}: {last_error}")
        if os.path.exists(part):
            os.remove(part)
        if attempt < RETRIES:
            sleep(2 * attempt)
    raise RuntimeError(f"{name}: {last_error}")


def chosen_models(config, roles):
    wanted = {r.strip() for r in roles.split(",") if r.strip()}
    return [m for m in config.get("model", []) if m.get("chosen") and m.get("role") in wanted]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models-file", required=True, help="the models TOML file")
    parser.add_argument("--dest", required=True, help="folder to put the models in")
    parser.add_argument("--roles", default=DEFAULT_ROLES, help="comma separated roles to fetch")
    args = parser.parse_args(argv)

    with open(args.models_file, "rb") as f:
        config = tomllib.load(f)
    models = chosen_models(config, args.roles)
    if not models:
        print(f"no chosen models with roles {args.roles!r} in {args.models_file}", file=sys.stderr)
        return 1

    os.makedirs(args.dest, exist_ok=True)
    print(f"models for roles {args.roles}: {len(models)} file(s) in {args.dest}")
    try:
        results = [fetch(m, args.dest) for m in models]
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"done: {results.count('present')} already present, {results.count('downloaded')} downloaded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
