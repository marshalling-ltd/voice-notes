"""Tests for pin_images.py. Run with: python3 -m unittest discover -s stack"""

import hashlib
import http.server
import json
import os
import re
import threading
import unittest

import pin_images

D1 = "sha256:" + "a" * 64
D2 = "sha256:" + "b" * 64
HERE = os.path.dirname(os.path.abspath(__file__))

COMPOSE = f"""services:
  one:
    image: ${{ONE_IMAGE:-ghcr.io/acme/one:v1@{D1}}}
  two:
    image: ghcr.io/acme/two:stable
  three:
    image: ${{ONE_IMAGE:-ghcr.io/acme/one:v1@{D1}}}
"""


class FakeRegistry(http.server.BaseHTTPRequestHandler):
    """A registry that wants a token first, like ghcr.io, and knows two tags."""

    manifests = {("acme/one", "v1"): b'{"manifests": []}', ("acme/two", "stable"): b'{"layers": []}'}
    hits = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        FakeRegistry.hits.append(self.path)
        if self.path.startswith("/token"):
            body = json.dumps({"token": "secret"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.headers.get("Authorization") != "Bearer secret":
            self.send_response(401)
            port = self.server.server_address[1]
            self.send_header("WWW-Authenticate", f'Bearer realm="http://127.0.0.1:{port}/token",service="fake"')
            self.end_headers()
            return
        match = re.fullmatch(r"/v2/(.+)/manifests/(.+)", self.path)
        body = FakeRegistry.manifests.get((match.group(1), match.group(2))) if match else None
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        if match.group(2) == "v1":  # one tag answers with the digest header, the other does not
            self.send_header("Docker-Content-Digest", "sha256:" + hashlib.sha256(body).hexdigest())
        self.end_headers()
        self.wfile.write(body)


class PinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeRegistry)
        cls.registry = f"127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def resolve(self, repo, tag):
        return pin_images.resolve_digest(self.registry, repo, tag, scheme="http")

    # ---- reading references

    def test_split_ref(self):
        self.assertEqual(pin_images.split_ref("ghcr.io/a/b:v1"), ("ghcr.io", "a/b", "v1", None))
        self.assertEqual(pin_images.split_ref(f"ghcr.io/a/b:v1@{D1}"), ("ghcr.io", "a/b", "v1", D1))
        self.assertEqual(pin_images.split_ref("ghcr.io/a/b"), ("ghcr.io", "a/b", "latest", None))
        self.assertEqual(pin_images.split_ref("alpine"), ("docker.io", "alpine", "latest", None))
        self.assertEqual(pin_images.split_ref("library/alpine:3"), ("docker.io", "library/alpine", "3", None))
        self.assertEqual(pin_images.split_ref("localhost:5000/x/y:t"), ("localhost:5000", "x/y", "t", None))
        with self.assertRaises(pin_images.PinError):
            pin_images.split_ref("ghcr.io/a/b@sha256:short")

    def test_image_lines_with_and_without_variables(self):
        images = pin_images.image_refs(COMPOSE)
        self.assertEqual([i.variable for i in images], ["ONE_IMAGE", None, "ONE_IMAGE"])
        self.assertEqual([i.ref for i in images], [f"ghcr.io/acme/one:v1@{D1}", "ghcr.io/acme/two:stable", f"ghcr.io/acme/one:v1@{D1}"])
        self.assertEqual([i.line_no for i in images], [2, 4, 6])

    def test_an_image_built_from_a_variable_cannot_be_pinned(self):
        with self.assertRaises(pin_images.PinError) as ctx:
            pin_images.image_refs("services:\n  a:\n    image: ghcr.io/a/b:${VERSION}\n")
        self.assertIn("variable", str(ctx.exception))

    # ---- the check

    def test_check_reports_only_unpinned_images(self):
        problems = pin_images.check(pin_images.image_refs(COMPOSE))
        self.assertEqual(len(problems), 1)
        self.assertIn("ghcr.io/acme/two:stable", problems[0])
        self.assertIn("line 5", problems[0])

    def test_every_image_in_the_real_compose_file_is_pinned_by_digest(self):
        """The ticket's rule: nothing in compose.yaml may be referenced by a tag alone."""
        with open(os.path.join(HERE, "compose.yaml")) as f:
            images = pin_images.image_refs(f.read())
        self.assertGreaterEqual(len(images), 3)
        self.assertEqual(pin_images.check(images), [])

    def test_the_two_llama_swap_services_use_the_same_image(self):
        with open(os.path.join(HERE, "compose.yaml")) as f:
            images = pin_images.image_refs(f.read())
        swap = {i.ref for i in images if i.variable == "LLAMA_SWAP_IMAGE"}
        self.assertEqual(len(swap), 1, "the models job and llama-swap must run the same image")

    # ---- looking digests up

    def test_resolve_follows_the_token_challenge_and_returns_the_digest_header(self):
        FakeRegistry.hits.clear()
        digest = self.resolve("acme/one", "v1")
        self.assertEqual(digest, "sha256:" + hashlib.sha256(b'{"manifests": []}').hexdigest())
        self.assertTrue(any(h.startswith("/token") for h in FakeRegistry.hits), "it asked for a token")

    def test_resolve_falls_back_to_hashing_the_body(self):
        digest = self.resolve("acme/two", "stable")
        self.assertEqual(digest, "sha256:" + hashlib.sha256(b'{"layers": []}').hexdigest())

    def test_resolve_reports_a_missing_tag(self):
        with self.assertRaises(pin_images.PinError) as ctx:
            self.resolve("acme/one", "nope")
        self.assertIn("was not found", str(ctx.exception))

    def test_resolve_reports_an_unreachable_registry(self):
        with self.assertRaises(pin_images.PinError) as ctx:
            pin_images.resolve_digest("127.0.0.1:1", "a/b", "t", scheme="http")
        self.assertIn("could not reach", str(ctx.exception))

    # ---- updating

    def test_update_pins_every_image_and_keeps_tags_and_layout(self):
        fake = lambda registry, repo, tag: D2 if repo == "acme/two" else D1
        new = pin_images.update(COMPOSE, resolve=fake)
        refs = [i.ref for i in pin_images.image_refs(new)]
        self.assertEqual(refs, [f"ghcr.io/acme/one:v1@{D1}", f"ghcr.io/acme/two:stable@{D2}", f"ghcr.io/acme/one:v1@{D1}"])
        self.assertIn("${ONE_IMAGE:-ghcr.io/acme/one:v1@", new, "the variable and its default syntax are kept")
        self.assertEqual(new.count("\n"), COMPOSE.count("\n"))

    def test_update_replaces_an_old_digest_with_the_current_one(self):
        fake = lambda registry, repo, tag: D2
        new = pin_images.update(COMPOSE, resolve=fake)
        self.assertNotIn(D1, new)
        self.assertEqual(new.count(D2), 3)

    def test_update_is_stable_when_nothing_moved(self):
        fake = lambda registry, repo, tag: D1
        once = pin_images.update(COMPOSE.replace("acme/two:stable", f"acme/two:stable@{D1}"), resolve=fake)
        self.assertEqual(pin_images.update(once, resolve=fake), once)

    # ---- mirroring

    def pinned_images(self):
        fake = lambda registry, repo, tag: D1
        return pin_images.image_refs(pin_images.update(COMPOSE, resolve=fake))

    def test_mirror_commands_copy_by_digest_to_named_destinations(self):
        plan = pin_images.mirror_commands(self.pinned_images(), "ghcr.io/org/images/")
        destinations = [p[0] for p in plan]
        self.assertEqual(destinations, ["ghcr.io/org/images/one:v1", "ghcr.io/org/images/two:stable", "ghcr.io/org/images/one:v1"])
        create = plan[0][2]
        self.assertEqual(create[:4], ["docker", "buildx", "imagetools", "create"])
        self.assertEqual(create[-1], f"ghcr.io/acme/one@{D1}", "the source is named by digest, not by tag")

    def test_mirror_refuses_unpinned_images_and_name_clashes(self):
        with self.assertRaises(pin_images.PinError):
            pin_images.mirror_commands(pin_images.image_refs(COMPOSE), "x")
        clash = pin_images.image_refs(f"a:\n  image: ghcr.io/p/same@{D1}\nb:\n  image: ghcr.io/q/same@{D2}\n")
        with self.assertRaises(pin_images.PinError) as ctx:
            pin_images.mirror_commands(clash, "x")
        self.assertIn("same", str(ctx.exception))

    def run_mirror(self, copied_digest, dry_run=False):
        calls, log = [], []

        def fake_run(command, **kwargs):
            calls.append(command)

            class Result:
                stdout = copied_digest + "\n"

            return Result()

        pin_images.mirror(self.pinned_images()[:1], "reg/x", dry_run=dry_run, run=fake_run, log=log.append)
        return calls, log

    def test_mirror_runs_the_copy_and_verifies_the_digest(self):
        calls, log = self.run_mirror(D1)
        self.assertEqual([c[3] for c in calls], ["create", "inspect"])
        self.assertTrue(any("ok:" in line for line in log))

    def test_mirror_fails_if_the_copy_has_a_different_digest(self):
        with self.assertRaises(pin_images.PinError) as ctx:
            self.run_mirror(D2)
        self.assertIn("expected", str(ctx.exception))

    def test_dry_run_copies_nothing(self):
        calls, log = self.run_mirror(D1, dry_run=True)
        self.assertEqual(calls, [])
        self.assertTrue(any("imagetools create" in line for line in log))

    # ---- the command line

    def test_main_check_list_and_missing_images(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "compose.yaml")
            with open(path, "w") as f:
                f.write(COMPOSE)
            self.assertEqual(pin_images.main(["--compose", path, "--check"]), 1)
            self.assertEqual(pin_images.main(["--compose", path, "--list"]), 0)
            with open(path, "w") as f:
                f.write("services: {}\n")
            self.assertEqual(pin_images.main(["--compose", path, "--check"]), 1, "no images at all is an error")


if __name__ == "__main__":
    unittest.main()
