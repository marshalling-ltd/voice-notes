"""Tests for fetch_models.py. Run with: python3 -m unittest discover -s stack"""

import hashlib
import http.server
import os
import tempfile
import threading
import unittest

import fetch_models

PAYLOAD = b"pretend model weights " * 5000


class Handler(http.server.BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self):
        Handler.hits += 1
        body = PAYLOAD if self.path == "/good.bin" else PAYLOAD[:-7] if self.path == "/short.bin" else None
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class FetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.dest = self.dir.name
        Handler.hits = 0
        self.log = []

    def tearDown(self):
        self.dir.cleanup()

    def model(self, path="/good.bin", sha=None, size=None):
        return {
            "url": self.base + path,
            "sha256": sha or hashlib.sha256(PAYLOAD).hexdigest(),
            "size_bytes": size if size is not None else len(PAYLOAD),
        }

    def run_fetch(self, model):
        return fetch_models.fetch(model, self.dest, log=self.log.append, sleep=lambda _: None)

    def test_downloads_and_verifies(self):
        self.assertEqual(self.run_fetch(self.model()), "downloaded")
        with open(os.path.join(self.dest, "good.bin"), "rb") as f:
            self.assertEqual(f.read(), PAYLOAD)
        self.assertEqual(os.listdir(self.dest), ["good.bin"], "no temporary file is left behind")

    def test_second_run_downloads_nothing(self):
        self.run_fetch(self.model())
        hits = Handler.hits
        self.assertEqual(self.run_fetch(self.model()), "present")
        self.assertEqual(Handler.hits, hits, "an intact file is not downloaded again")

    def test_corrupt_existing_file_is_replaced(self):
        with open(os.path.join(self.dest, "good.bin"), "wb") as f:
            f.write(b"corrupt")
        self.assertEqual(self.run_fetch(self.model()), "downloaded")
        with open(os.path.join(self.dest, "good.bin"), "rb") as f:
            self.assertEqual(f.read(), PAYLOAD)

    def test_wrong_checksum_is_rejected_and_nothing_is_kept(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.run_fetch(self.model(sha="0" * 64))
        self.assertIn("wrong sha256", str(ctx.exception))
        self.assertEqual(os.listdir(self.dest), [], "a bad file is never left in place")
        self.assertEqual(Handler.hits, fetch_models.RETRIES, "it retried before giving up")

    def test_truncated_download_is_rejected(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.run_fetch(self.model(path="/short.bin"))
        self.assertIn("wrong size", str(ctx.exception))
        self.assertEqual(os.listdir(self.dest), [])

    def test_missing_file_is_an_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            self.run_fetch(self.model(path="/nope.bin"))
        self.assertIn("download failed", str(ctx.exception))

    def test_unreachable_server_is_an_error(self):
        model = self.model()
        model["url"] = "http://127.0.0.1:1/good.bin"
        with self.assertRaises(RuntimeError):
            self.run_fetch(model)

    def test_only_chosen_models_with_wanted_roles_are_fetched(self):
        config = {
            "model": [
                {"id": "a", "role": "speech-to-text", "chosen": True},
                {"id": "b", "role": "speech-to-text", "chosen": False},
                {"id": "c", "role": "chat", "chosen": True},
                {"id": "d", "role": "text-to-speech", "chosen": True},
            ]
        }
        ids = [m["id"] for m in fetch_models.chosen_models(config, fetch_models.DEFAULT_ROLES)]
        self.assertEqual(ids, ["a", "c"], "unchosen models and baked-in Kokoro are skipped")
        self.assertEqual([m["id"] for m in fetch_models.chosen_models(config, "text-to-speech")], ["d"])

    def test_file_name_comes_from_the_url_path(self):
        self.assertEqual(fetch_models.file_name("https://h/a/b/ggml-tiny.en-q5_1.bin?download=true"), "ggml-tiny.en-q5_1.bin")
        self.assertEqual(fetch_models.file_name("https://h/a/My%20Model.gguf"), "My Model.gguf")

    PAIRS = [
        ("ci-models.toml", "llama-swap.yaml"),
        ("nightly-models.toml", "llama-swap.nightly.yaml"),
    ]

    def test_real_catalogue_names_match_the_llama_swap_config(self):
        """The files llama-swap loads must be the ones the catalogue downloads."""
        import tomllib

        here = os.path.dirname(os.path.abspath(__file__))
        for models_file, swap_file in self.PAIRS:
            with open(os.path.join(here, models_file), "rb") as f:
                config = tomllib.load(f)
            with open(os.path.join(here, swap_file)) as f:
                swap = f.read()
            chosen = fetch_models.chosen_models(config, fetch_models.DEFAULT_ROLES)
            self.assertEqual(len(chosen), 2, f"{models_file}: one speech-to-text and one chat model")
            for model in chosen:
                self.assertIn("/models/" + fetch_models.file_name(model["url"]), swap, f"{models_file}: {model['id']}")

    def test_every_catalogue_entry_is_complete(self):
        """Each model needs what the downloader checks, with a well-formed checksum."""
        import re
        import tomllib

        here = os.path.dirname(os.path.abspath(__file__))
        for models_file, _ in self.PAIRS:
            with open(os.path.join(here, models_file), "rb") as f:
                config = tomllib.load(f)
            for model in config["model"]:
                where = f"{models_file}: {model.get('id')}"
                for key in ("id", "role", "class", "url", "sha256", "size_bytes", "licence"):
                    self.assertIn(key, model, where)
                self.assertRegex(model["sha256"], r"^[0-9a-f]{64}$", where)
                self.assertGreater(model["size_bytes"], 1_000_000, where)
                self.assertTrue(model["url"].startswith("https://"), where)

    def test_whisper_writes_its_temporary_files_outside_the_read_only_models_folder(self):
        """--convert writes files to --tmp-dir, which defaults to the working directory (/models),
        and compose mounts /models read-only. Without --tmp-dir every transcription returns 500."""
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "compose.yaml")) as f:
            compose = f.read()
        self.assertIn("models:/models:ro", compose, "the services mount the models read-only")
        for _, swap_file in self.PAIRS:
            with open(os.path.join(here, swap_file)) as f:
                swap = f.read()
            if "--convert" in swap:
                self.assertIn("--tmp-dir /tmp", swap, swap_file)

    def test_every_model_stays_loaded_together(self):
        """Without a group llama-swap swaps models on every request, adding the load time of one
        model to every turn. All models in each file must be members of the keep-loaded group."""
        import re

        here = os.path.dirname(os.path.abspath(__file__))
        for _, swap_file in self.PAIRS:
            with open(os.path.join(here, swap_file)) as f:
                swap = f.read()
            models_block = swap.split("\nmodels:", 1)[1].split("\ngroups:", 1)[0]
            defined = set(re.findall(r"(?m)^  ([A-Za-z0-9._-]+):\s*$", models_block))
            members = set(re.findall(r'(?m)^      - "([^"]+)"', swap.split("\ngroups:", 1)[1]))
            self.assertEqual(defined, members, swap_file)
            self.assertIn("swap: false", swap, swap_file)

    def test_main_reports_missing_models(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
            f.write('[[model]]\nid="x"\nrole="chat"\nchosen=false\n')
        try:
            self.assertEqual(fetch_models.main(["--models-file", f.name, "--dest", self.dest]), 1)
        finally:
            os.remove(f.name)


if __name__ == "__main__":
    unittest.main()
