import unittest
from pathlib import Path

from src.model_runtime.resolver import load_model_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "models.yaml"
SCRIPT_NAMES = (
    "download_hf_models.sh",
    "download_modelscope_models.sh",
    "download_models_hf.sh",
    "download_models_modelscope.sh",
    "check_environment.sh",
)
MODEL_SECTIONS = ("asr", "llm", "tts", "translation")


class DeploymentConfigTests(unittest.TestCase):
    def test_config_file_exists(self):
        self.assertTrue(CONFIG_PATH.is_file())

    def test_required_model_sections_and_fields_exist(self):
        config = load_model_config(CONFIG_PATH)
        self.assertEqual(config.model_root, "./models")
        self.assertEqual(set(config.profiles), set(MODEL_SECTIONS))
        for profile in config.profiles.values():
            self.assertIn(profile.provider, {"huggingface", "modelscope", "local", "whisper"})
            self.assertTrue(profile.model_id)
            self.assertTrue(profile.local_path)

    def test_config_paths_are_relative(self):
        config = load_model_config(CONFIG_PATH)
        self.assertFalse(Path(config.model_root).is_absolute())
        self.assertTrue(all(not Path(profile.local_path).is_absolute() for profile in config.profiles.values()))

    def test_deployment_scripts_exist(self):
        for script_name in SCRIPT_NAMES:
            script = REPOSITORY_ROOT / "scripts" / script_name
            self.assertTrue(script.is_file(), script_name)
            self.assertTrue(script.stat().st_mode & 0o111, script_name)

    def test_models_documentation_exists(self):
        self.assertTrue((REPOSITORY_ROOT / "models" / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
