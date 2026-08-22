import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "models.yaml"
SCRIPT_NAMES = (
    "download_models_hf.sh",
    "download_models_modelscope.sh",
    "check_environment.sh",
)
MODEL_SECTIONS = ("turn", "asr", "llm", "tts", "translation")


class DeploymentConfigTests(unittest.TestCase):
    def test_config_file_exists(self):
        self.assertTrue(CONFIG_PATH.is_file())

    def test_required_model_sections_and_fields_exist(self):
        config = CONFIG_PATH.read_text(encoding="utf-8")
        self.assertIn("model_root: ./models", config)
        for section in MODEL_SECTIONS:
            self.assertRegex(config, rf"(?m)^  {section}:\s*$")
            self.assertRegex(config, rf"(?m)^    backend:\s*\S+")
            self.assertRegex(config, rf"(?m)^    name:\s*")
            self.assertRegex(config, rf"(?m)^    path:\s*\./models/\S+")

    def test_config_paths_are_relative(self):
        config = CONFIG_PATH.read_text(encoding="utf-8")
        paths = re.findall(r"(?m)^\s+(?:model_root|path):\s*(\S+)", config)
        self.assertTrue(paths)
        self.assertTrue(all(not Path(path).is_absolute() for path in paths))
        self.assertTrue(all(path.startswith("./models") for path in paths))

    def test_deployment_scripts_exist(self):
        for script_name in SCRIPT_NAMES:
            script = REPOSITORY_ROOT / "scripts" / script_name
            self.assertTrue(script.is_file(), script_name)
            self.assertTrue(script.stat().st_mode & 0o111, script_name)

    def test_models_documentation_exists(self):
        self.assertTrue((REPOSITORY_ROOT / "models" / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
