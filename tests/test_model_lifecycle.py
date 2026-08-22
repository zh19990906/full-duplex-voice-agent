import asyncio
import unittest

from src.model_runtime.lifecycle import ModelLifecycleManager, ModelState
from src.model_runtime.resolver import load_model_config


class LifecycleProvider:
    def __init__(self):
        self.calls = []

    async def load(self):
        self.calls.append("load")

    async def warmup(self):
        self.calls.append("warmup")

    async def unload(self):
        self.calls.append("unload")


class NoLifecycleProvider:
    pass


class ModelLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_registration_and_duplicate_handling(self):
        manager = ModelLifecycleManager()
        record = manager.register("asr", "whisper", "models/asr")
        self.assertEqual(record.status, ModelState.REGISTERED)
        with self.assertRaises(ValueError):
            manager.register("asr", "whisper", "models/asr")

    async def test_load_warmup_unload_lifecycle(self):
        provider = LifecycleProvider()
        manager = ModelLifecycleManager()
        manager.register("llm", "llama_cpp", "models/llm", resource=provider)
        await manager.load("llm")
        self.assertEqual(manager.status("llm"), ModelState.READY)
        await manager.warmup("llm")
        self.assertEqual(manager.status("llm"), ModelState.ACTIVE)
        await manager.unload("llm")
        self.assertEqual(manager.status("llm"), ModelState.UNLOADED)
        self.assertEqual(provider.calls, ["load", "warmup", "unload"])

    async def test_provider_without_lifecycle_methods_gracefully_ready(self):
        manager = ModelLifecycleManager()
        manager.register("tts", "cosyvoice", "models/tts", resource=NoLifecycleProvider())
        await manager.load("tts")
        self.assertEqual(manager.status("tts"), ModelState.READY)
        await manager.warmup("tts")
        self.assertEqual(manager.status("tts"), ModelState.ACTIVE)
        await manager.unload("tts")
        self.assertEqual(manager.status("tts"), ModelState.UNLOADED)

    async def test_invalid_transitions_raise_clear_errors(self):
        manager = ModelLifecycleManager()
        manager.register("asr", "whisper", "models/asr")
        with self.assertRaises(RuntimeError):
            await manager.warmup("asr")
        await manager.load("asr")
        with self.assertRaises(RuntimeError):
            await manager.load("asr")
        await manager.warmup("asr")
        with self.assertRaises(RuntimeError):
            await manager.warmup("asr")

    async def test_missing_model_and_failed_provider_are_reported(self):
        manager = ModelLifecycleManager()
        with self.assertRaises(KeyError):
            manager.status("missing")

        class BrokenProvider:
            async def load(self):
                raise RuntimeError("load failed")

        manager.register("broken", "fake", "models/broken", resource=BrokenProvider())
        with self.assertRaises(RuntimeError):
            await manager.load("broken")
        self.assertEqual(manager.status("broken"), ModelState.ERROR)

    def test_models_config_contains_lifecycle_profiles(self):
        config = load_model_config("configs/models.yaml")
        self.assertTrue(config.profiles["asr"].auto_load)
        self.assertTrue(config.profiles["llm"].auto_load)
        self.assertFalse(config.profiles["tts"].auto_load)


if __name__ == "__main__":
    unittest.main()
