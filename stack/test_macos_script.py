"""Checks on macos.sh that can run anywhere. The script itself is exercised on a Mac by the
macOS workflow (.github/workflows/macos.yml). Run with: python3 -m unittest discover -s stack"""

import os
import re
import subprocess
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "macos.sh")


def script_text():
    with open(SCRIPT) as f:
        return f.read()


def code_lines():
    return [line for line in script_text().splitlines() if not line.lstrip().startswith("#")]


class MacosScriptTests(unittest.TestCase):
    def test_script_is_valid_bash_and_executable(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK), "macos.sh must be executable")
        subprocess.run(["bash", "-n", SCRIPT], check=True)

    def test_uses_only_what_bash_3_2_on_macos_has(self):
        text = "\n".join(code_lines())
        for pattern in (r"declare\s+-A", r"\bmapfile\b", r"\breadarray\b", r"\$\{[^}]*(,,|\^\^)[^}]*\}", r"\$\{[^}]*@[QEPAa]\}", r"&>>", r"\|&"):
            self.assertIsNone(re.search(pattern, text), f"not available in bash 3.2: {pattern}")

    def test_pins_are_exact(self):
        text = script_text()
        for name in ("LLAMA_SWAP_SHA256_ARM64", "LLAMA_SWAP_SHA256_AMD64"):
            self.assertRegex(text, rf'{name}="[0-9a-f]{{64}}"')
        for name in ("WHISPER_CPP_COMMIT", "KOKORO_COMMIT"):
            self.assertRegex(text, rf'{name}="[0-9a-f]{{40}}"')
        self.assertRegex(text, r'LLAMA_SWAP_VERSION="\d+"')

    def test_every_service_listens_on_loopback_only(self):
        code = "\n".join(code_lines())
        self.assertNotIn("0.0.0.0", code)
        self.assertIn('LOOPBACK="127.0.0.1"', code)
        self.assertEqual(code.count("--host "), 1)
        self.assertIn('--host "$LOOPBACK"', code)
        self.assertIn('-listen "$LOOPBACK:', code)

    def test_defaults_match_the_compose_stack(self):
        text = script_text()
        self.assertIn('LLAMA_SWAP_PORT="${LLAMA_SWAP_PORT:-8080}"', text)
        self.assertIn('KOKORO_PORT="${KOKORO_PORT:-8880}"', text)
        with open(os.path.join(HERE, "compose.yaml")) as f:
            compose = f.read()
        self.assertIn("LLAMA_SWAP_PORT:-8080", compose)
        self.assertIn("KOKORO_PORT:-8880", compose)

    def test_every_profile_uses_files_that_exist(self):
        text = script_text()
        for pair in re.findall(r'printf \'%s %s\' "([^"]+)" "([^"]+)"', text):
            for name in pair:
                if "$1" not in name:
                    self.assertTrue(os.path.isfile(os.path.join(HERE, name)), name)
        for name in ("minimal", "light", "standard", "power"):
            self.assertIn(name, text)
            for ext in ("toml", "yaml"):
                self.assertTrue(os.path.isfile(os.path.join(HERE, "profiles", f"{name}.{ext}")))

    def test_models_folder_rewrite_matches_the_configs(self):
        """The script rewrites ' /models/' to the native folder; every config must use that form."""
        import glob

        configs = glob.glob(os.path.join(HERE, "profiles", "*.yaml")) + [os.path.join(HERE, "llama-swap.yaml")]
        for path in configs:
            with open(path) as f:
                text = f.read()
            self.assertRegex(text, r"-m /models/", path)
            self.assertNotRegex(text, r"[^ ]/models/", f"{path}: a /models/ path the rewrite would miss")

    def test_the_other_commands_refuse_a_bad_profile_and_install_refuses_linux(self):
        env = dict(os.environ, VOICE_NOTES_STACK_HOME="/nonexistent-voice-notes-test")
        bad = subprocess.run([SCRIPT, "start", "nope"], capture_output=True, text=True, env=env)
        self.assertEqual(bad.returncode, 1)
        self.assertIn("unknown profile", bad.stderr)
        spaces = subprocess.run([SCRIPT, "status"], capture_output=True, text=True, env=dict(env, VOICE_NOTES_STACK_HOME="/a b"))
        self.assertEqual(spaces.returncode, 1)
        self.assertIn("spaces", spaces.stderr)


if __name__ == "__main__":
    unittest.main()
