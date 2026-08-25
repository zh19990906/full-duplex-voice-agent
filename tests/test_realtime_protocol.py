import struct
import unittest

from src.realtime.protocol import (
    AudioFrameHeader,
    RealtimeEnvelope,
    decode_audio_frame,
    encode_audio_frame,
)


class RealtimeProtocolTests(unittest.TestCase):
    def test_envelope_contains_ordering_and_generation_identity(self):
        envelope = RealtimeEnvelope(
            event="BACKCHANNEL_CONFIRMED",
            event_id="evt-1",
            session_id="session-1",
            sequence=4,
            capture_timestamp=10.0,
            server_timestamp=10.1,
            response_id="response-2",
            generation_epoch=3,
            segment_id=5,
            payload={"confidence": 0.9},
        )

        self.assertEqual(envelope.to_dict()["protocol_version"], 1)
        self.assertEqual(envelope.to_dict()["generation_epoch"], 3)
        self.assertEqual(envelope.to_dict()["session_id"], "session-1")
        self.assertEqual(envelope.to_dict()["sequence"], 4)
        self.assertEqual(envelope.to_dict()["segment_id"], 5)

    def test_existing_positional_protocol_version_slot_remains_unchanged(self):
        positional_arguments = (
            "BACKCHANNEL_CONFIRMED",
            "evt-1",
            "session-1",
            4,
            10.0,
            10.1,
            "response-2",
            3,
            {"confidence": 0.9},
        )

        envelope = RealtimeEnvelope(*positional_arguments, 1)

        self.assertEqual(envelope.protocol_version, 1)
        self.assertIsNone(envelope.segment_id)
        with self.assertRaises(ValueError):
            RealtimeEnvelope(*positional_arguments, 2)

    def test_envelope_rejects_non_integer_segment_id(self):
        with self.assertRaises(ValueError):
            RealtimeEnvelope(
                event="BACKCHANNEL_CONFIRMED",
                event_id="evt-1",
                session_id="session-1",
                sequence=4,
                capture_timestamp=10.0,
                server_timestamp=10.1,
                response_id="response-2",
                generation_epoch=3,
                payload={"confidence": 0.9},
                segment_id="segment-5",
            )

    def test_audio_frame_round_trip_preserves_pcm_and_header(self):
        header = AudioFrameHeader(
            sequence=8,
            capture_timestamp=4.5,
            sample_rate=16000,
            channels=1,
        )

        pcm = b"\x01\x00" * 320
        encoded = encode_audio_frame(header, pcm)
        decoded_header, pcm = decode_audio_frame(encoded)

        self.assertEqual(decoded_header, header)
        self.assertEqual(pcm, b"\x01\x00" * 320)

    def test_encode_rejects_non_pcm16_audio_contract(self):
        with self.assertRaises(ValueError):
            encode_audio_frame(AudioFrameHeader(1, 0.0, sample_rate=8000), b"\x00\x00")
        with self.assertRaises(ValueError):
            encode_audio_frame(AudioFrameHeader(1, 0.0, channels=2), b"\x00\x00")
        with self.assertRaises(ValueError):
            encode_audio_frame(AudioFrameHeader(1, 0.0), b"\x00")
        with self.assertRaises(ValueError):
            encode_audio_frame(AudioFrameHeader(1, 0.0), b"\x00\x00" * 319)

    def test_decode_rejects_invalid_header_or_truncated_frame(self):
        valid = encode_audio_frame(AudioFrameHeader(1, 0.0), b"\x00\x00" * 320)
        unsupported_version = bytes([2]) + valid[1:]
        non_16khz = struct.pack("!BIdIB", 1, 1, 0.0, 8000, 1) + b"\x00\x00" * 320
        stereo = struct.pack("!BIdIB", 1, 1, 0.0, 16000, 2) + b"\x00\x00" * 320
        short_payload = struct.pack("!BIdIB", 1, 1, 0.0, 16000, 1) + b"\x00\x00" * 319

        for frame in (
            unsupported_version,
            non_16khz,
            stereo,
            short_payload,
            valid[:-1],
            b"\x01",
        ):
            with self.subTest(frame=frame):
                with self.assertRaises(ValueError):
                    decode_audio_frame(frame)


if __name__ == "__main__":
    unittest.main()
