import unittest

from src.adapters.llm.providers.qwen_transformers import TransformersQwenProvider
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


class TransformersQwenProviderTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
