import asyncio
import queue
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

from src.adapters.llm.providers.qwen_transformers import (
    TransformersQwenProvider,
    _TransformersRuntime,
)
from src.llm_runtime.stream import TokenChunk
from src.model_runtime.factory import create_llm_adapter


class FakeQwenRuntime:
    def __init__(self):
        self.prompts = []
        self.cancelled = False

    async def stream_tokens(self, prompt, **options):
        self.prompts.append(prompt)
        yield TokenChunk("one", "你好", 1.0, False)
        if not self.cancelled:
            yield TokenChunk("two", "，世界", 2.0, True)

    async def generate(self, prompt, **options):
        self.prompts.append(prompt)
        return "完整回答"

    async def cancel(self):
        self.cancelled = True


async def _collect_async(iterable):
    return [item async for item in iterable]


class TransformersQwenProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_transformers_cancel_waits_for_background_generate_to_stop(self):
        started = threading.Event()
        stopped = threading.Event()

        class FakeTensor:
            def to(self, _device):
                return self

        class FakeTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                return {"input_ids": FakeTensor()}

        class FakeModel:
            def parameters(self):
                yield types.SimpleNamespace(device="cpu")

            def generate(self, **kwargs):
                criteria = kwargs.get("stopping_criteria")
                started.set()
                deadline = time.monotonic() + 0.3
                while time.monotonic() < deadline:
                    if criteria is not None and criteria(None, None):
                        break
                    time.sleep(0.001)
                stopped.set()

        class FakeStreamer:
            def __init__(self, *_args, **_kwargs):
                pass

            def __iter__(self):
                return self

            def __next__(self):
                if stopped.is_set():
                    raise StopIteration
                raise queue.Empty

        class FakeStoppingCriteria:
            pass

        class FakeStoppingCriteriaList(list):
            def __call__(self, input_ids, scores, **kwargs):
                return any(item(input_ids, scores, **kwargs) for item in self)

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.TextIteratorStreamer = FakeStreamer
        fake_transformers.StoppingCriteria = FakeStoppingCriteria
        fake_transformers.StoppingCriteriaList = FakeStoppingCriteriaList
        fake_torch = types.ModuleType("torch")
        runtime = _TransformersRuntime(
            FakeTokenizer(),
            FakeModel(),
            max_new_tokens=128,
            temperature=0.0,
        )

        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "transformers": fake_transformers},
        ):
            consumer = asyncio.create_task(
                _collect_async(runtime.stream_tokens("keep generating"))
            )
            self.assertTrue(await asyncio.to_thread(started.wait, 0.2))

            await runtime.cancel()
            stopped_before_cancel_returned = stopped.is_set()
            await asyncio.wait_for(consumer, timeout=0.5)

            self.assertTrue(stopped_before_cancel_returned)

    async def test_injected_runtime_streams_tokens_without_transformers_dependency(self):
        runtime = FakeQwenRuntime()
        provider = TransformersQwenProvider("models/qwen", runtime=runtime)

        result = [chunk async for chunk in provider.stream_tokens("你好")]

        self.assertEqual([chunk.text for chunk in result], ["你好", "，世界"])
        self.assertTrue(result[-1].is_final)
        self.assertEqual(runtime.prompts, ["你好"])

    async def test_injected_runtime_supports_complete_generation_and_cancel(self):
        runtime = FakeQwenRuntime()
        provider = TransformersQwenProvider("models/qwen", runtime=runtime)

        self.assertEqual(await provider.generate("问题"), "完整回答")
        await provider.cancel()

        self.assertTrue(runtime.cancelled)

    def test_missing_runtime_requires_transformers_runtime(self):
        with self.assertRaises(RuntimeError):
            TransformersQwenProvider("models/qwen", import_runtime=False)

    def test_factory_creates_transformers_provider(self):
        adapter = create_llm_adapter(
            {
                "provider": "transformers",
                "model_path": "models/qwen",
                "provider_runtime": FakeQwenRuntime(),
                "device": "cuda",
            }
        )

        self.assertIsInstance(adapter.provider, TransformersQwenProvider)

    def test_transformers_runtime_flattens_model_options(self):
        runtime = _TransformersRuntime(
            None,
            None,
            model_options={"max_new_tokens": 64},
            temperature=0.0,
        )

        self.assertEqual(runtime.generation_options["max_new_tokens"], 64)
        self.assertEqual(runtime.generation_options["temperature"], 0.0)
        self.assertNotIn("model_options", runtime.generation_options)


if __name__ == "__main__":
    unittest.main()
