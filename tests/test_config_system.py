import os
import tempfile
import unittest
from pathlib import Path

from src.config import ConfigError, ConfigLoader


class ConfigSystemTests(unittest.TestCase):
    def _write_config_dir(self, root: Path) -> None:
        (root / "models.yaml").write_text(
            "model_root: ./models\n"
            "asr:\n"
            "  provider: whisper\n"
            "  local_path: ./models/asr\n"
            "  device: cuda\n"
            "llm:\n"
            "  provider: llama_cpp\n"
            "  local_path: ./models/llm\n"
            "tts:\n"
            "  provider: cosyvoice\n"
            "  local_path: ./models/tts\n",
            encoding="utf-8",
        )
        (root / "audio.yaml").write_text(
            "audio:\n"
            "  sample_rate: 16000\n"
            "  input:\n"
            "    provider: sounddevice\n",
            encoding="utf-8",
        )
        (root / "demo.yaml").write_text(
            "audio:\n"
            "  output_provider: sounddevice\n"
            "models:\n"
            "  asr: whisper\n",
            encoding="utf-8",
        )
        profiles = root / "profiles"
        profiles.mkdir()
        (profiles / "dev.yaml").write_text(
            "audio:\n"
            "  input:\n"
            "    provider: fake\n"
            "models:\n"
            "  asr:\n"
            "    provider: fake\n"
            "runtime:\n"
            "  environment: development\n",
            encoding="utf-8",
        )
        (profiles / "production.yaml").write_text(
            "audio:\n"
            "  input:\n"
            "    provider: sounddevice\n"
            "runtime:\n"
            "  environment: production\n",
            encoding="utf-8",
        )

    def test_loads_base_files_and_deep_merges_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_config_dir(root)

            config = ConfigLoader(root).load("dev")

        self.assertEqual(config["audio"]["sample_rate"], 16000)
        self.assertEqual(config["audio"]["input"]["provider"], "fake")
        self.assertEqual(config["audio"]["output_provider"], "sounddevice")
        self.assertEqual(config["models"]["asr"]["provider"], "fake")
        self.assertEqual(config["models"]["asr"]["local_path"], "./models/asr")
        self.assertEqual(config["models"]["llm"]["provider"], "llama_cpp")
        self.assertEqual(config["runtime"]["environment"], "development")

    def test_load_from_env_selects_profile_and_defaults_to_dev(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_config_dir(root)
            loader = ConfigLoader(root)
            old_value = os.environ.get("VOICE_AGENT_PROFILE")
            try:
                os.environ.pop("VOICE_AGENT_PROFILE", None)
                self.assertEqual(loader.load_from_env()["runtime"]["environment"], "development")
                os.environ["VOICE_AGENT_PROFILE"] = "production"
                self.assertEqual(loader.load_from_env()["runtime"]["environment"], "production")
            finally:
                if old_value is None:
                    os.environ.pop("VOICE_AGENT_PROFILE", None)
                else:
                    os.environ["VOICE_AGENT_PROFILE"] = old_value

    def test_unknown_profile_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_config_dir(root)
            with self.assertRaisesRegex(ConfigError, "unknown configuration profile"):
                ConfigLoader(root).load("missing")

    def test_missing_required_section_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_config_dir(root)
            (root / "profiles" / "dev.yaml").write_text(
                "audio:\n  input_provider: fake\nmodels: null\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "missing required section: models"):
                ConfigLoader(root).load("dev")

    def test_missing_model_provider_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_config_dir(root)
            (root / "profiles" / "dev.yaml").write_text(
                "models:\n  llm:\n    provider: ''\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "models.llm.provider"):
                ConfigLoader(root).load("dev")

    def test_malformed_yaml_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_config_dir(root)
            (root / "profiles" / "dev.yaml").write_text(
                "audio:\n  - malformed-list-entry\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "invalid YAML entry"):
                ConfigLoader(root).load("dev")


if __name__ == "__main__":
    unittest.main()
