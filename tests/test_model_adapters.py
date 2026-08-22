import inspect
import unittest

from src.adapters.asr.base import BaseASRAdapter
from src.adapters.llm.base import BaseLLMAdapter
from src.adapters.tts.base import BaseTTSAdapter
from src.adapters.translation.base import BaseTranslationAdapter


class TestASRAdapter(BaseASRAdapter):
    async def stream_audio(self, audio_chunk: bytes):
        return await super().stream_audio(audio_chunk)


class TestLLMAdapter(BaseLLMAdapter):
    async def generate(self, prompt: str):
        return await super().generate(prompt)

    async def stream_tokens(self, prompt: str):
        return await super().stream_tokens(prompt)

    async def cancel(self):
        return await super().cancel()


class TestTTSAdapter(BaseTTSAdapter):
    async def synthesize(self, text: str):
        return await super().synthesize(text)

    async def stream_audio(self, text: str):
        return await super().stream_audio(text)

    async def interrupt(self):
        return await super().interrupt()


class TestTranslationAdapter(BaseTranslationAdapter):
    async def translate_stream(self, text: str):
        return await super().translate_stream(text)


class ModelAdapterTest(unittest.IsolatedAsyncioTestCase):
    def test_all_adapter_modules_import_as_abstract_classes(self):
        adapters = (
            BaseASRAdapter,
            BaseLLMAdapter,
            BaseTTSAdapter,
            BaseTranslationAdapter,
        )
        for adapter in adapters:
            self.assertTrue(inspect.isabstract(adapter))

    def test_required_methods_are_async_abstract_methods(self):
        methods = (
            (BaseASRAdapter, "stream_audio"),
            (BaseLLMAdapter, "generate"),
            (BaseLLMAdapter, "stream_tokens"),
            (BaseLLMAdapter, "cancel"),
            (BaseTTSAdapter, "synthesize"),
            (BaseTTSAdapter, "stream_audio"),
            (BaseTTSAdapter, "interrupt"),
            (BaseTranslationAdapter, "translate_stream"),
        )
        for adapter, method_name in methods:
            method = getattr(adapter, method_name)
            self.assertTrue(inspect.iscoroutinefunction(method))
            self.assertTrue(getattr(method, "__isabstractmethod__", False))

    async def test_unimplemented_methods_raise_not_implemented_error(self):
        calls = (
            (TestASRAdapter().stream_audio, (b"audio",)),
            (TestLLMAdapter().generate, ("prompt",)),
            (TestLLMAdapter().stream_tokens, ("prompt",)),
            (TestLLMAdapter().cancel, ()),
            (TestTTSAdapter().synthesize, ("text",)),
            (TestTTSAdapter().stream_audio, ("text",)),
            (TestTTSAdapter().interrupt, ()),
            (TestTranslationAdapter().translate_stream, ("text",)),
        )
        for method, args in calls:
            with self.assertRaises(NotImplementedError):
                await method(*args)

    def test_adapters_have_no_model_specific_constructor_requirements(self):
        self.assertEqual(inspect.signature(BaseASRAdapter), inspect.Signature())
        self.assertEqual(inspect.signature(BaseLLMAdapter), inspect.Signature())
        self.assertEqual(inspect.signature(BaseTTSAdapter), inspect.Signature())
        self.assertEqual(inspect.signature(BaseTranslationAdapter), inspect.Signature())
