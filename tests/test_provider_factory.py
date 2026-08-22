import asyncio
import unittest

from src.adapters.asr.backend import StreamingASRBackend
from src.adapters.llm.backend import StreamingLLMBackend
from src.adapters.tts.backend import StreamingTTSBackend
from src.model_runtime.factory import (
    create_asr_adapter,
    create_llm_adapter,
    create_tts_adapter,
)


class ProviderFactoryTests(unittest.TestCase):
    def test_factory_creates_adapters_and_injects_model_path(self):
        provider = object()
        profile = {
            "provider": "local",
            "model_path": "models/asr",
            "provider_instance": provider,
        }
        adapter = create_asr_adapter(profile)
        self.assertIsInstance(adapter, StreamingASRBackend)
        self.assertIs(adapter.provider, provider)
        self.assertEqual(adapter.model_path, "models/asr")

    def test_factory_supports_all_streaming_adapter_kinds(self):
        provider = object()
        profile = {"provider": "fake", "local_path": "models/model", "provider_instance": provider}
        self.assertIsInstance(create_asr_adapter(profile), StreamingASRBackend)
        self.assertIsInstance(create_llm_adapter(profile), StreamingLLMBackend)
        self.assertIsInstance(create_tts_adapter(profile), StreamingTTSBackend)

    def test_invalid_provider_profile_fails_clearly(self):
        with self.assertRaises(ValueError):
            create_asr_adapter({"model_path": "models/asr"})
        with self.assertRaises(ValueError):
            create_asr_adapter({"provider": "local"})
        with self.assertRaises(ValueError):
            create_asr_adapter(
                {"provider": "not-a-provider", "model_path": "models/asr"},
                provider=object(),
            )


class FakeASRProvider:
    async def stream_audio(self, audio):
        return {"text": audio.decode(), "is_final": True}


class FakeLLMProvider:
    async def generate(self, prompt):
        return prompt + "!"

    async def stream_tokens(self, prompt):
        async def stream():
            yield {"text": "hello", "is_final": False}
            yield {"text": "", "is_final": True}

        return stream()


class FakeTTSProvider:
    async def stream_audio(self, text):
        return [text.encode()]


class RealAdapterContractTests(unittest.TestCase):
    def test_fake_provider_streams_work_behind_existing_contracts(self):
        async def run():
            asr = create_asr_adapter({"provider": "fake", "model_path": "models/asr", "provider_instance": FakeASRProvider()})
            llm = create_llm_adapter({"provider": "fake", "model_path": "models/llm", "provider_instance": FakeLLMProvider()})
            tts = create_tts_adapter({"provider": "fake", "model_path": "models/tts", "provider_instance": FakeTTSProvider()})
            asr_result = await asr.stream_audio(b"hello")
            llm_result = await llm.stream_tokens("prompt")
            llm_items = [item async for item in llm_result]
            tts_result = await tts.stream_audio("hello")
            return asr_result, llm_items, tts_result

        asr_result, llm_items, tts_result = asyncio.run(run())
        self.assertTrue(asr_result["is_final"])
        self.assertEqual(llm_items[0]["text"], "hello")
        self.assertEqual(tts_result, [b"hello"])


if __name__ == "__main__":
    unittest.main()
