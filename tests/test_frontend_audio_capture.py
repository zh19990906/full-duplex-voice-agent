import base64
import shutil
import subprocess
import unittest
from pathlib import Path

from src.realtime.protocol import decode_audio_frame


ROOT = Path(__file__).parents[1]


class FrontendAudioCaptureTests(unittest.TestCase):
    def test_javascript_pcm_frame_decodes_with_python_v1_contract(self):
        if shutil.which("node") is None:
            self.skipTest("Node.js is required for the browser module behavior test")

        result = subprocess.run(
            [
                "node",
                "--input-type=module",
                "--eval",
                """
                    import { PcmFrameEncoder } from './frontend/src/capture-worklet.js';
                    const samples = new Float32Array(320);
                    samples[0] = 0;
                    samples[1] = 1;
                    samples[2] = -1;
                    const [frame] = new PcmFrameEncoder().push(samples, 12.5);
                    process.stdout.write(Buffer.from(frame).toString('base64'));
                """,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        header, pcm = decode_audio_frame(base64.b64decode(result.stdout))
        self.assertEqual((header.sequence, header.capture_timestamp), (0, 12.5))
        self.assertEqual((header.sample_rate, header.channels), (16000, 1))
        self.assertEqual(pcm[:6], b"\x00\x00\xff\x7f\x00\x80")


if __name__ == "__main__":
    unittest.main()
