import unittest

from src.controller.states import ControllerState
from src.integration.pipeline_demo import FullDuplexIntegrationDemo
from src.integration.scenarios import ScenarioResult


class FullDuplexIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.demo = FullDuplexIntegrationDemo()
        await self.demo.start()

    async def asyncTearDown(self):
        await self.demo.shutdown()

    async def test_complete_pipeline_initialization(self):
        self.assertTrue(self.demo.application.initialized)
        self.assertTrue(self.demo.audio_processor.running)
        self.assertEqual(self.demo.asr_pipeline.session.status.value, "RUNNING")
        self.assertEqual(self.demo.tts_pipeline.session.status.value, "RUNNING")

    async def test_backchannel_keeps_generation_alive(self):
        result = await self.demo.run_backchannel_scenario()

        self.assertIsInstance(result, ScenarioResult)
        self.assertTrue(result.passed, result.details)
        self.assertTrue(result.details["generation_active_during_backchannel"])
        self.assertFalse(result.details["tts_interrupted"])

    async def test_interrupt_cancels_generation_and_tts_without_stale_audio(self):
        result = await self.demo.run_interrupt_scenario()

        self.assertTrue(result.passed, result.details)
        self.assertTrue(result.details["generation_cancelled"])
        self.assertTrue(result.details["tts_interrupted"])
        self.assertEqual(result.details["stale_audio_frames"], 0)

    async def test_translation_contract(self):
        result = await self.demo.run_translation_scenario()

        self.assertTrue(result.passed, result.details)
        self.assertEqual(result.details["translated_text"], "Hello")
        self.assertGreater(result.details["audio_frames"], 0)

    async def test_task_state_resume(self):
        result = await self.demo.run_task_resume_scenario()

        self.assertTrue(result.passed, result.details)
        self.assertEqual(result.details["resumed_state"], {"current_number": 5})
        self.assertEqual(result.details["next_number_contract"], 6)

    async def test_audio_input_reaches_asr_event_boundary(self):
        await self.demo.feed_audio(b"hello")

        self.assertEqual(self.demo.asr_adapter.received_audio, [b"hello"])
        self.assertEqual(self.demo.last_asr_event.payload["text"], "hello")
        self.assertEqual(self.demo.application.controller.state, ControllerState.THINKING)


if __name__ == "__main__":
    unittest.main()
