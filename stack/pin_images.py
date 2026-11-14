#!/usr/bin/env python3
"""Pin the container images in compose.yaml by digest, and mirror them to our own registry.

A tag such as `unified-vulkan` can be moved to a different image at any time, which would change
what CI runs without anyone changing the repository. A digest (`@sha256:...`) names exactly one
image, so the stack only changes when this repository does.

    pin_images.py --list                  show every image and whether it is pinned
    pin_images.py --check                 exit 1 if an image is not pinned (works offline)
    pin_images.py --update                look up the current digest of every tag and rewrite
                                          compose.yaml; review the change, and let the smoke
                                          test pass, before merging it
    pin_images.py --mirror PREFIX         copy each pinned image to PREFIX/NAME, keeping its digest
    pin_images.py --mirror PREFIX --dry-run    print what would be copied

Only the Python standard library is used. Looking up digests needs network access to the registry;
mirroring also needs the `docker buildx` plugin and a login that may push to PREFIX.
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_COMPOSE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compose.yaml")

ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)

DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
IMAGE_LINE = re.compile(r"^(?P<indent>\s+)image:\s*(?P<value>\S+)\s*$")
VARIABLE_DEFAULT = re.compile(r"^\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*):-(?P<default>[^}]*)\}$")


class PinError(Exception):
    """Something that stops pinning or mirroring, with a message for the user."""


class Image:
    """One image reference found in the compose file."""

    def __init__(self, line_no, variable, ref):
        self.line_no = line_no  # 0-based index into the file's lines
        self.variable = variable  # e.g. LLAMA_SWAP_IMAGE, or None for a literal
        self.ref = ref
        self.registry, self.repo, self.tag, self.digest = split_ref(ref)

    @property
    def pinned(self):
        return self.digest is not None

    @property
    def mirror_name(self):
        return self.repo.split("/")[-1]


def split_ref(ref):
    """Split `registry/repo/name:tag@sha256:...` into (registry, repo, tag, digest).

    The registry is `docker.io` when the first part is not a host name, and the tag is `latest`
    when none is given.
    """
    digest = None
    if "@" in ref:
        ref, digest = ref.split("@", 1)
        if not DIGEST.fullmatch(digest):
            raise PinError(f"not a valid sha256 digest: {digest}")
    first, _, rest = ref.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry, path = first, rest
    else:
        registry, path = "docker.io", ref
    name, colon, tag = path.rpartition(":")
    if colon and "/" not in tag:
        repo = name
    else:
        repo, tag = path, "latest"
    return registry, repo, tag, digest


def image_refs(text):
    """Every `image:` line of a compose file, with its default value if it uses a variable."""
    images = []
    for index, line in enumerate(text.splitlines()):
        match = IMAGE_LINE.match(line)
        if not match:
            continue
        value = match.group("value")
        variable = None
        default = VARIABLE_DEFAULT.match(value)
        if default:
            variable, value = default.group("name"), default.group("default")
        elif "$" in value:
            raise PinError(f"line {index + 1}: cannot pin an image that is built from a variable: {value}")
        images.append(Image(index, variable, value))
    return images


def check(images):
    """Messages for every image that is not pinned by digest."""
    return [
        f"line {i.line_no + 1}: {i.ref} is not pinned by digest"
        for i in images
        if not i.pinned
    ]


def _challenge(header):
    return dict(re.findall(r'(\w+)="([^"]*)"', header))


def _token(challenge, repo, opener):
    realm = challenge.get("realm")
    if not realm:
        raise PinError("the registry asked for authentication in a way that is not understood")
    service = challenge.get("service", "")
    scope = challenge.get("scope") or f"repository:{repo}:pull"
    url = f"{realm}?service={urllib.request.quote(service)}&scope={urllib.request.quote(scope)}"
    request = urllib.request.Request(url, headers={"User-Agent": "voice-notes-stack"})
    with opener(request, timeout=30) as response:
        import json

        body = json.load(response)
    return body.get("token") or body.get("access_token")


def resolve_digest(registry, repo, tag, scheme="https", opener=urllib.request.urlopen):
    """The digest that `registry/repo:tag` points to right now (of the index, for multi-arch)."""
    url = f"{scheme}://{registry}/v2/{repo}/manifests/{tag}"
    headers = {"Accept": ACCEPT, "User-Agent": "voice-notes-stack"}
    token = None
    for _ in range(2):
        request = urllib.request.Request(url, headers=dict(headers))
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with opener(request, timeout=30) as response:
                digest = response.headers.get("Docker-Content-Digest")
                body = response.read()
        except urllib.error.HTTPError as e:
            if e.code == 401 and token is None:
                token = _token(_challenge(e.headers.get("WWW-Authenticate", "")), repo, opener)
                continue
            if e.code == 404:
                raise PinError(f"{registry}/{repo}:{tag} was not found") from e
            raise PinError(f"{registry}/{repo}:{tag}: the registry answered {e.code}") from e
        except urllib.error.URLError as e:
            raise PinError(f"could not reach {registry}: {e.reason}") from e
        return digest or "sha256:" + hashlib.sha256(body).hexdigest()
    raise PinError(f"{registry}/{repo}:{tag}: authentication failed")


def update(text, resolve=resolve_digest):
    """Return `text` with every image pinned to the digest its tag currently points to."""
    lines = text.split("\n")
    for image in image_refs(text):
        digest = resolve(image.registry, image.repo, image.tag)
        base = image.ref.split("@", 1)[0]
        new_ref = f"{base}@{digest}"
        line = lines[image.line_no]
        lines[image.line_no] = line.replace(image.ref, new_ref, 1)
    return "\n".join(lines)


def mirror_commands(images, prefix):
    """`docker buildx imagetools` commands that copy each pinned image to `prefix/name:tag`,
    as (destination, digest, create_command, inspect_command)."""
    seen = {}
    plan = []
    for image in images:
        if not image.pinned:
            raise PinError(f"{image.ref} is not pinned by digest; run --update first")
        name = image.mirror_name
        if name in seen and seen[name] != image.ref:
            raise PinError(f"two different images would both be mirrored as {name}")
        seen[name] = image.ref
        destination = f"{prefix.rstrip('/')}/{name}:{image.tag}"
        source = f"{image.registry}/{image.repo}@{image.digest}"
        create = ["docker", "buildx", "imagetools", "create", "--tag", destination, source]
        inspect = ["docker", "buildx", "imagetools", "inspect", destination, "--format", "{{.Manifest.Digest}}"]
        plan.append((destination, image.digest, create, inspect))
    return plan


def mirror(images, prefix, dry_run=False, run=subprocess.run, log=print):
    """Copy every image to our registry and check that the copy has the very same digest."""
    for destination, digest, create, inspect in mirror_commands(images, prefix):
        log(f"{destination}: copying {digest}")
        if dry_run:
            log("  " + " ".join(create))
            continue
        run(create, check=True)
        copied = run(inspect, check=True, capture_output=True, text=True).stdout.strip()
        if copied != digest:
            raise PinError(f"{destination}: the copy has digest {copied}, expected {digest}")
        log(f"  ok: {destination}@{digest}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--compose", default=DEFAULT_COMPOSE, help="the compose file to read")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="show every image")
    group.add_argument("--check", action="store_true", help="fail if an image is not pinned")
    group.add_argument("--update", action="store_true", help="re-pin every image to its tag's current digest")
    group.add_argument("--mirror", metavar="PREFIX", help="copy the pinned images to PREFIX/NAME")
    parser.add_argument("--dry-run", action="store_true", help="with --mirror: only print the commands")
    args = parser.parse_args(argv)

    try:
        with open(args.compose) as f:
            text = f.read()
        images = image_refs(text)
        if not images:
            raise PinError(f"no images found in {args.compose}")

        if args.list:
            for image in images:
                state = "pinned " if image.pinned else "UNPINNED"
                print(f"{state} {image.ref}")
            return 0
        if args.check:
            problems = check(images)
            for problem in problems:
                print(problem, file=sys.stderr)
            print(f"{len(images) - len(problems)} of {len(images)} images are pinned by digest")
            return 1 if problems else 0
        if args.update:
            new_text = update(text)
            if new_text == text:
                print("every image is already pinned to its tag's current digest")
            else:
                with open(args.compose, "w") as f:
                    f.write(new_text)
                print(f"updated {args.compose}; review the change with git diff")
            return 0
        mirror(images, args.mirror, dry_run=args.dry_run)
        return 0
    except PinError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"error: {' '.join(e.cmd)} failed with exit code {e.returncode}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
