import asyncio
import math
import threading
import unittest

from src.realtime.audio_ingress import (
    AudioActivityCandidate,
    AudioIngress,
    AudioIngressConsumerError,
)
from src.realtime.protocol import AudioFrameHeader, encode_audio_frame
from src.realtime.session_runtime import RealtimeSessionRuntime


def wire_frame(sequence, timestamp=None, pcm=None):
    """Build one hand-controlled V1 browser frame."""
    if timestamp is None:
        timestamp = sequence * 0.02
    if pcm is None:
        pcm = b"\x00\x00" * 320
    return encode_audio_frame(AudioFrameHeader(sequence, timestamp), pcm)


class RealtimeAudioIngressTests(unittest.IsolatedAsyncioTestCase):
    async def test_out_of_order_frames_fan_out_in_contiguous_order(self):
        """Catches releasing a buffered future frame before its missing predecessor."""
        received = {"asr": [], "turn": []}
        ingress = AudioIngress(
            asr_consumer=lambda frame: received["asr"].append(frame.sequence),
            turn_consumer=lambda frame: received["turn"].append(frame.sequence),
        )

        await ingress.push(wire_frame(1))
        await ingress.push(wire_frame(0))
        await ingress.flush()

        self.assertEqual(received, {"asr": [0, 1], "turn": [0, 1]})
        self.assertEqual(ingress.next_sequence, 2)
        await ingress.close()

    async def test_sync_and_async_consumers_receive_same_frame_identity(self):
        """Catches copying or adapting decoded frames differently per consumer."""
        seen = []

        def sync_consumer(frame):
            seen.append(("sync", frame))

        async def async_consumer(frame):
            await asyncio.sleep(0)
            seen.append(("async", frame))

        ingress = AudioIngress(
            asr_consumer=sync_consumer,
            turn_consumer=async_consumer,
        )

        await ingress.push(wire_frame(0, timestamp=3.25))
        await ingress.flush()

        self.assertEqual([kind for kind, _ in seen], ["sync", "async"])
        self.assertIs(seen[0][1], seen[1][1])
        self.assertEqual(seen[0][1].sequence, 0)
        self.assertEqual(seen[0][1].capture_timestamp, 3.25)
        self.assertEqual(seen[0][1].pcm, b"\x00\x00" * 320)
        await ingress.close()

    async def test_slow_asr_consumer_does_not_block_turn_delivery(self):
        """Catches a shared worker or queue letting slow ASR stall turn detection."""
        asr_started = asyncio.Event()
        release_asr = asyncio.Event()
        turns = []

        async def slow_asr(frame):
            asr_started.set()
            await release_asr.wait()

        ingress = AudioIngress(
            asr_consumer=slow_asr,
            turn_consumer=lambda frame: turns.append(frame.sequence),
            consumer_queue_size=4,
        )

        await ingress.push(wire_frame(0))
        await asyncio.wait_for(asr_started.wait(), timeout=0.1)
        await ingress.push(wire_frame(1))
        await ingress.push(wire_frame(2))

        for _ in range(20):
            if turns == [0, 1, 2]:
                break
            await asyncio.sleep(0)
        self.assertEqual(turns, [0, 1, 2])

        release_asr.set()
        await ingress.close()

    async def test_blocking_sync_asr_does_not_block_turn_delivery(self):
        """Catches invoking a blocking synchronous ASR consumer on the event loop."""
        started = threading.Event()
        release = threading.Event()
        turns = []
        turns_complete = asyncio.Event()

        def blocking_asr(frame):
            started.set()
            release.wait(timeout=0.5)

        async def turn_consumer(frame):
            turns.append(frame.sequence)
            if len(turns) == 2:
                turns_complete.set()

        ingress = AudioIngress(
            asr_consumer=blocking_asr,
            turn_consumer=turn_consumer,
            consumer_queue_size=4,
        )
        try:
            began = asyncio.get_running_loop().time()
            await ingress.push(wire_frame(0))
            self.assertLess(asyncio.get_running_loop().time() - began, 0.1)
            await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=0.1)

            await ingress.push(wire_frame(1))
            await asyncio.wait_for(turns_complete.wait(), timeout=0.1)
            self.assertEqual(turns, [0, 1])
            self.assertFalse(release.is_set())
        finally:
            release.set()
            await ingress.close()

    async def test_sync_consumer_returning_awaitable_runs_call_off_loop_then_awaits_result(self):
        """Catches dropping or running a sync consumer's returned awaitable on a worker thread."""
        loop_thread = threading.get_ident()
        call_threads = []
        await_threads = []

        async def returned_work():
            await_threads.append(threading.get_ident())

        def sync_consumer(frame):
            call_threads.append(threading.get_ident())
            return returned_work()

        ingress = AudioIngress(asr_consumer=sync_consumer)
        await ingress.push(wire_frame(0))
        await ingress.flush()

        self.assertEqual(len(call_threads), 1)
        self.assertNotEqual(call_threads, [loop_thread])
        self.assertEqual(await_threads, [loop_thread])
        await ingress.close()

    async def test_duplicate_frames_are_rejected_and_counted(self):
        """Catches accepting duplicate delivered or buffered browser sequence numbers."""
        ingress = AudioIngress()

        await ingress.push(wire_frame(1))
        with self.assertRaises(ValueError):
            await ingress.push(wire_frame(1))
        await ingress.push(wire_frame(0))
        with self.assertRaises(ValueError):
            await ingress.push(wire_frame(0))

        self.assertEqual(ingress.metrics.duplicate_frames, 2)
        await ingress.close()

    async def test_flush_records_missing_ranges_then_drains_sorted_remainder(self):
        """Catches flush dropping buffered gaps or failing to surface lost sequences."""
        received = []
        ingress = AudioIngress(turn_consumer=lambda frame: received.append(frame.sequence))

        await ingress.push(wire_frame(1))
        await ingress.push(wire_frame(3))
        await ingress.flush()

        self.assertEqual(received, [1, 3])
        self.assertEqual(ingress.metrics.missing_sequence_ranges, [(0, 0), (2, 2)])
        self.assertEqual(ingress.next_sequence, 4)
        await ingress.close()

    async def test_bounded_queues_drop_newest_frame_without_stalling_other_consumers(self):
        """Catches queue overflow silently blocking or dropping every consumer's audio."""
        asr_started = asyncio.Event()
        release_asr = asyncio.Event()
        turns = []
        turns_complete = asyncio.Event()

        async def slow_asr(frame):
            asr_started.set()
            await release_asr.wait()

        async def turn_consumer(frame):
            turns.append(frame.sequence)
            if len(turns) == 4:
                turns_complete.set()

        ingress = AudioIngress(
            asr_consumer=slow_asr,
            turn_consumer=turn_consumer,
            consumer_queue_size=1,
        )

        await ingress.push(wire_frame(0))
        await asyncio.wait_for(asr_started.wait(), timeout=0.1)
        await ingress.push(wire_frame(1))
        await ingress.push(wire_frame(2))
        await ingress.push(wire_frame(3))

        await asyncio.wait_for(turns_complete.wait(), timeout=0.5)
        self.assertEqual(turns, [0, 1, 2, 3])
        self.assertEqual(ingress.metrics.consumer_overflow_frames["asr"], 2)
        self.assertEqual(ingress.metrics.consumer_overflow_frames["turn"], 0)

        release_asr.set()
        await ingress.close()

    async def test_bounded_jitter_buffer_rejects_newest_future_frame_and_counts_it(self):
        """Catches unbounded future-frame buffering under a missing initial frame."""
        ingress = AudioIngress(jitter_buffer_size=1)

        await ingress.push(wire_frame(2))
        with self.assertRaises(OverflowError):
            await ingress.push(wire_frame(3))

        self.assertEqual(ingress.metrics.jitter_buffer_overflow_frames, 1)
        await ingress.close()

    async def test_activity_candidate_uses_pcm16_rms_and_original_frame_identity(self):
        """Catches activity candidates that lose sequence/timestamp or use byte-level energy."""
        candidates = []
        high_pcm = (1000).to_bytes(2, "little", signed=True) * 320
        ingress = AudioIngress(
            activity_consumer=lambda candidate: candidates.append(candidate),
            activity_threshold=500.0,
        )

        await ingress.push(wire_frame(0, timestamp=7.5, pcm=high_pcm))
        await ingress.flush()

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertIsInstance(candidate, AudioActivityCandidate)
        self.assertEqual(candidate.sequence, 0)
        self.assertEqual(candidate.capture_timestamp, 7.5)
        self.assertTrue(candidate.active)
        self.assertTrue(math.isclose(candidate.rms, 1000.0))
        self.assertIs(ingress.last_activity_candidate, candidate)
        await ingress.close()

    async def test_flush_raises_consumer_failure_without_deadlocking(self):
        """Catches worker exceptions that leave queue.join waiting forever or disappear."""
        async def broken_consumer(frame):
            raise RuntimeError("turn failed")

        ingress = AudioIngress(turn_consumer=broken_consumer)
        await ingress.push(wire_frame(0))

        with self.assertRaisesRegex(AudioIngressConsumerError, "turn failed"):
            await asyncio.wait_for(ingress.flush(), timeout=0.1)

        self.assertIsInstance(ingress.worker_errors["turn"], RuntimeError)
        with self.assertRaisesRegex(AudioIngressConsumerError, "turn failed"):
            await ingress.close()
        with self.assertRaises(RuntimeError):
            await ingress.push(wire_frame(1))

    async def test_direct_close_propagates_consumer_failure_after_cleanup(self):
        """Catches close swallowing a worker failure after it releases resources."""
        async def broken_consumer(frame):
            raise RuntimeError("direct close failed")

        ingress = AudioIngress(turn_consumer=broken_consumer)
        await ingress.push(wire_frame(0))

        with self.assertRaisesRegex(AudioIngressConsumerError, "direct close failed"):
            await ingress.close()

        self.assertTrue(ingress.closed)
        with self.assertRaisesRegex(AudioIngressConsumerError, "direct close failed"):
            await asyncio.wait_for(ingress.flush(), timeout=0.1)

    async def test_runtime_close_propagates_ingress_failure_after_cleanup(self):
        """Catches RealtimeSessionRuntime masking its ingress close failure."""
        async def broken_consumer(frame):
            raise RuntimeError("runtime close failed")

        runtime = RealtimeSessionRuntime(
            "session-close-failure",
            ingress=AudioIngress(turn_consumer=broken_consumer),
        )
        await runtime.push_audio(wire_frame(0))

        with self.assertRaisesRegex(AudioIngressConsumerError, "runtime close failed"):
            await runtime.close()

        self.assertTrue(runtime.closed)
        self.assertTrue(runtime.ingress.closed)

    async def test_empty_close_with_configured_consumer_keeps_flush_and_close_idempotent(self):
        """Catches enqueuing a stop marker for a worker that was never started."""
        ingress = AudioIngress(turn_consumer=lambda frame: None)

        await asyncio.wait_for(ingress.close(), timeout=0.1)
        await asyncio.wait_for(ingress.flush(), timeout=0.1)
        await asyncio.wait_for(ingress.close(), timeout=0.1)

    async def test_close_rejects_late_push_and_drains_previously_accepted_buffer(self):
        """Catches accepting a frame after close starts and then losing it at shutdown."""
        class CloseRaceIngress(AudioIngress):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.flush_completed = asyncio.Event()
                self.release_close = asyncio.Event()

            async def flush(self):
                await super().flush()
                self.flush_completed.set()
                await self.release_close.wait()

        received = []
        ingress = CloseRaceIngress(turn_consumer=lambda frame: received.append(frame.sequence))
        try:
            await ingress.push(wire_frame(0))
            for _ in range(20):
                if received == [0]:
                    break
                await asyncio.sleep(0)
            self.assertEqual(received, [0])
            await ingress.push(wire_frame(2))

            close_task = asyncio.create_task(ingress.close())
            await asyncio.wait_for(ingress.flush_completed.wait(), timeout=0.1)

            with self.assertRaises(RuntimeError):
                await ingress.push(wire_frame(3))
            self.assertEqual(received, [0, 2])
        finally:
            ingress.release_close.set()
            await asyncio.wait_for(close_task, timeout=0.1)

    async def test_minimal_session_runtime_composes_epoch_and_ingress_lifecycle(self):
        """Catches a runtime skeleton that does not delegate input or expose its epoch."""
        received = []
        ingress = AudioIngress(turn_consumer=lambda frame: received.append(frame.sequence))
        runtime = RealtimeSessionRuntime("session-5", ingress=ingress)

        self.assertEqual(runtime.session_id, "session-5")
        self.assertEqual(runtime.generation_epoch, 0)
        self.assertEqual(runtime.advance_generation(), 1)
        await runtime.push_audio(wire_frame(0))
        await runtime.flush()

        self.assertEqual(received, [0])
        await runtime.close()
        self.assertTrue(runtime.closed)


if __name__ == "__main__":
    unittest.main()
