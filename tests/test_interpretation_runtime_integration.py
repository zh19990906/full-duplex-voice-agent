import asyncio
import unittest

from src.adapters.asr.base import BaseASRAdapter
from src.asr.pipeline import ASRPipeline
from src.asr.stream import TranscriptChunk
from src.controller.actions import ActionType, ControllerAction
from src.realtime.session_runtime import RealtimeSessionRuntime
from src.realtime.text_segmenter import TextSegment
from src.tts_runtime.stream import AudioChunk


class FakeASRAdapter(BaseASRAdapter):
    def __init__(self, outputs):
        self.outputs = list(outputs)

    async def stream_audio(self, audio_chunk):
        if self.outputs:
            return self.outputs.pop(0)
        return None


class RecordingTranslator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.started = asyncio.Event()
        self.release = None
        self.prompts = []

    async def translate_stream(self, prompt):
        self.prompts.append(prompt)
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if not self.responses:
            raise AssertionError("unexpected translation request")
        return self.responses.pop(0)


class RecordingInterpretationSink:
    def __init__(self):
        self.items = []

    async def publish(self, segment):
        self.items.append(segment)


class RuntimeBackedInterpretationSink:
    def __init__(self, runtime):
        self.runtime = runtime
        self.recorded_segments = []
        self.recorded_audio = []

    async def publish(self, segment):
        text_segment = TextSegment(
            segment.translation_segment_id,
            segment.translated_text,
            False,
            segment.response_id,
            segment.generation_epoch,
        )
        audio_chunk = AudioChunk(
            chunk_id=f"{segment.response_id}:{segment.generation_epoch}:{segment.translation_segment_id}",
            audio_data=b"\x00\x00" * 2,
            timestamp=segment.translation_segment_id + 1.0,
            is_final=True,
            request_id=f"{segment.response_id}:{segment.generation_epoch}:{segment.translation_segment_id}",
            response_id=segment.response_id,
            generation_epoch=segment.generation_epoch,
            segment_id=segment.translation_segment_id,
        )
        self.recorded_segments.append(self.runtime.record_response_segment(text_segment))
        self.recorded_audio.append(self.runtime.record_audio_chunk(audio_chunk))


class InterpretationRuntimeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_asr_events_flow_through_runtime_interpretation_pipeline(self):
        translator = RecordingTranslator(["Hello"])
        sink = RecordingInterpretationSink()
        runtime = RealtimeSessionRuntime(
            "session-interpret",
            interpretation_translator=translator,
            interpretation_sink=sink,
        )
        await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "INTERPRETATION",
                        "source_language": None,
                        "target_language": "English",
                    },
                ),
            )
        )
        asr = ASRPipeline(
            FakeASRAdapter([TranscriptChunk("chunk-1", "你好", 1.0, False, revision_id=1)]),
            event_handler=runtime.accept_asr_event,
        )
        await asr.start_session("session-1", "zh")

        await asr.push_audio(b"audio")

        self.assertEqual([item.source_text for item in sink.items], ["你好"])
        self.assertEqual([item.translated_text for item in sink.items], ["Hello"])

    async def test_epoch_advance_cancels_pending_runtime_interpretation_work(self):
        translator = RecordingTranslator(["late"])
        translator.release = asyncio.Event()
        sink = RecordingInterpretationSink()
        runtime = RealtimeSessionRuntime(
            "session-interpret",
            interpretation_translator=translator,
            interpretation_sink=sink,
        )
        await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "INTERPRETATION",
                        "source_language": None,
                        "target_language": "English",
                    },
                ),
            )
        )
        asr_event = TranscriptChunk("chunk-1", "你好", 1.0, False, revision_id=1)

        task = asyncio.create_task(runtime.accept_transcript_chunk(asr_event))
        await translator.started.wait()
        runtime.advance_generation()
        translator.release.set()
        result = await task

        self.assertIsNone(result)
        self.assertEqual(sink.items, [])

    async def test_runtime_activates_interpretation_response_identity_for_runtime_backed_sink(self):
        translator = RecordingTranslator(["Hello"])
        runtime = RealtimeSessionRuntime("session-interpret", interpretation_translator=translator)
        sink = RuntimeBackedInterpretationSink(runtime)
        runtime.interpretation_sink = sink
        await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "INTERPRETATION",
                        "source_language": None,
                        "target_language": "English",
                    },
                ),
            )
        )

        result = await runtime.accept_transcript_chunk(
            TranscriptChunk("chunk-1", "你好", 1.0, False, revision_id=1)
        )

        self.assertIsNotNone(result)
        self.assertEqual(runtime.current_response_id, result.response_id)
        self.assertEqual(sink.recorded_segments, [True])
        self.assertEqual(sink.recorded_audio, [True])
        checkpoint = runtime.checkpoints.get(result.response_id)
        self.assertEqual(checkpoint.generated_text, "")
        self.assertEqual(checkpoint.segments[0].text, "Hello")
        self.assertEqual(checkpoint.segments[0].audio, b"\x00\x00" * 2)

    async def test_epoch_advance_rotates_interpretation_response_identity(self):
        translator = RecordingTranslator(["Hello", "Again"])
        sink = RecordingInterpretationSink()
        runtime = RealtimeSessionRuntime(
            "session-interpret",
            interpretation_translator=translator,
            interpretation_sink=sink,
        )
        await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "INTERPRETATION",
                        "source_language": None,
                        "target_language": "English",
                    },
                ),
            )
        )
        first_response_id = runtime.current_response_id

        runtime.advance_generation()
        second_response_id = runtime.current_response_id
        result = await runtime.accept_transcript_chunk(
            TranscriptChunk("chunk-2", "你好", 2.0, False, revision_id=2)
        )

        self.assertNotEqual(first_response_id, second_response_id)
        self.assertEqual(second_response_id, runtime.interpretation_session.response_id)
        self.assertEqual(result.response_id, second_response_id)
        self.assertEqual(result.generation_epoch, runtime.generation_epoch)

    async def test_mode_exit_cancels_pending_runtime_interpretation_work(self):
        translator = RecordingTranslator(["late"])
        translator.release = asyncio.Event()
        sink = RecordingInterpretationSink()
        runtime = RealtimeSessionRuntime(
            "session-interpret",
            interpretation_translator=translator,
            interpretation_sink=sink,
        )
        await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "INTERPRETATION",
                        "source_language": None,
                        "target_language": "English",
                    },
                ),
            )
        )
        asr_event = TranscriptChunk("chunk-1", "你好", 1.0, False, revision_id=1)

        task = asyncio.create_task(runtime.accept_transcript_chunk(asr_event))
        await translator.started.wait()
        await runtime.apply_controller_actions(
            (
                ControllerAction(
                    ActionType.SWITCH_MODE,
                    {
                        "target_mode": "CHAT",
                        "source_language": None,
                        "target_language": None,
                    },
                ),
            )
        )
        translator.release.set()
        result = await task

        self.assertIsNone(result)
        self.assertEqual(sink.items, [])
