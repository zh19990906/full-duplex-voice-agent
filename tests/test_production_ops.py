import asyncio
import io
import json
import logging
import unittest

from src.application.entrypoint import ProductionRuntime
from src.model_runtime.lifecycle import ModelLifecycleManager, ModelState
from src.ops.health import HealthChecker
from src.ops.logging_config import configure_logging, log_event
from src.ops.status import RuntimeStatus


def config_for(profile="test"):
    return {
        "audio": {"input_provider": "fake", "output_provider": "fake"},
        "models": {
            "asr": {"provider": "fake", "local_path": "models/asr", "auto_load": True},
            "llm": {"provider": "fake", "local_path": "models/llm", "auto_load": True},
            "tts": {"provider": "fake", "local_path": "models/tts", "auto_load": True},
        },
        "runtime": {"environment": profile},
    }


class StaticLoader:
    def __init__(self, config):
        self.config = config

    def load(self, profile):
        return self.config


class LifecycleResource:
    async def load(self):
        return None

    async def warmup(self):
        return None

    async def unload(self):
        return None


class ProductionOperationsTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_and_shutdown_sequence(self):
        tts_stopped = []

        async def stop_tts():
            tts_stopped.append(True)

        runtime = ProductionRuntime(
            profile="test",
            config_loader=StaticLoader(config_for()),
            tts_stop=stop_tts,
            model_resources={name: LifecycleResource() for name in ("asr", "llm", "tts")},
        )

        await runtime.initialize()

        self.assertTrue(runtime.application.initialized)
        self.assertEqual(runtime.lifecycle.status("asr"), ModelState.ACTIVE)
        self.assertEqual(runtime.lifecycle.status("llm"), ModelState.ACTIVE)
        self.assertIsNotNone(runtime.status().started_at)
        await runtime.application.generation_manager.start_generation("response-1")

        await runtime.shutdown()

        self.assertFalse(runtime.application.initialized)
        self.assertEqual(runtime.lifecycle.status("tts"), ModelState.UNLOADED)
        self.assertEqual(tts_stopped, [True])
        self.assertIsNone(runtime.application.generation_manager.active_session)

    async def test_signal_request_performs_graceful_shutdown(self):
        runtime = ProductionRuntime(
            profile="test",
            config_loader=StaticLoader(config_for()),
            model_resources={name: LifecycleResource() for name in ("asr", "llm", "tts")},
        )
        await runtime.initialize()
        runtime.request_shutdown()
        await runtime.wait_for_shutdown()

        self.assertTrue(runtime.shutdown_requested)
        self.assertFalse(runtime.application.initialized)

    def test_health_reports_healthy_and_failed_model(self):
        manager = ModelLifecycleManager()
        for name in ("asr", "llm", "tts"):
            manager.register(name, "fake", f"models/{name}")
            manager.record(name).state = ModelState.ACTIVE
        checker = HealthChecker(application_state=True, lifecycle_manager=manager)
        self.assertEqual(checker.health()["status"], "healthy")

        manager.record("llm").state = ModelState.ERROR
        health = checker.health()
        self.assertEqual(health["status"], "unhealthy")
        self.assertEqual(health["models"]["llm"], "error")

    def test_runtime_status_is_serializable(self):
        status = RuntimeStatus(
            profile="production",
            runtime_state="running",
            loaded_models={"asr": "active"},
            started_at=10.0,
            now=12.5,
        )

        payload = status.to_dict()

        self.assertEqual(payload["profile"], "production")
        self.assertEqual(payload["uptime_seconds"], 2.5)
        self.assertEqual(json.loads(status.to_json()), payload)

    def test_structured_logging_emits_json_event(self):
        stream = io.StringIO()
        logger = configure_logging(stream=stream, name="test-ops")

        log_event(logger, "model_loaded", model="llm", status="ready")

        record = json.loads(stream.getvalue())
        self.assertEqual(record["event"], "model_loaded")
        self.assertEqual(record["model"], "llm")
        self.assertEqual(record["status"], "ready")


if __name__ == "__main__":
    unittest.main()
