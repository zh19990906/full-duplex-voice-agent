import os
import tempfile
import unittest
from pathlib import Path

from src.model_runtime.resolver import ModelResolver, load_model_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "models.yaml"


class ModelDeploymentTests(unittest.TestCase):
    def test_yaml_config_parses_profiles(self):
        config = load_model_config(CONFIG_PATH)
        self.assertEqual(set(config.profiles), {"asr", "llm", "tts", "translation"})
        self.assertEqual(config.profiles["asr"].provider, "whisper")
        self.assertEqual(config.profiles["tts"].provider, "modelscope")

    def test_local_path_resolution(self):
        config = load_model_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(config, model_home=temp_dir)
            self.assertEqual(resolver.resolve("asr"), Path(temp_dir).resolve() / "asr")

    def test_model_home_environment_override(self):
        config = load_model_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            old_value = os.environ.get("MODEL_HOME")
            os.environ["MODEL_HOME"] = temp_dir
            try:
                resolver = ModelResolver(config)
                self.assertEqual(resolver.resolve("translation"), Path(temp_dir).resolve() / "translation")
            finally:
                if old_value is None:
                    os.environ.pop("MODEL_HOME", None)
                else:
                    os.environ["MODEL_HOME"] = old_value

    def test_missing_model_detection(self):
        config = load_model_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(config, model_home=temp_dir)
            with self.assertRaises(FileNotFoundError):
                resolver.require_available("llm")

    def test_provider_profiles(self):
        config = load_model_config(CONFIG_PATH)
        resolver = ModelResolver(config, model_home=tempfile.gettempdir())
        self.assertEqual({p.name for p in resolver.profiles(provider="huggingface")}, {"translation"})
        self.assertEqual([p.name for p in resolver.profiles(provider="modelscope")], ["tts"])
        self.assertEqual([p.name for p in resolver.profiles(provider="whisper")], ["asr"])
        self.assertEqual([p.name for p in resolver.profiles(provider="llama_cpp")], ["llm"])

    def test_offline_prepare_only_creates_directories(self):
        config = load_model_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = ModelResolver(config, model_home=temp_dir)
            target = resolver.prepare("asr")
            self.assertTrue(target.is_dir())
            self.assertTrue(resolver.is_available("asr"))


if __name__ == "__main__":
    unittest.main()
