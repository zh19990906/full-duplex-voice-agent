import assert from "node:assert/strict";
import test from "node:test";

const { PcmFrameEncoder, StreamingResampler } = await import("../frontend/src/capture-worklet.js");
const { PcmMicrophoneInput } = await import("../frontend/src/audio.js");

const PCM_HEADER_BYTES = 18;

const makeLiteralFrame = () => {
  const samples = new Float32Array(320);
  samples[0] = 0;
  samples[1] = 1;
  samples[2] = -1;
  return samples;
};

test("encoder holds partial samples and emits a complete V1 PCM16 frame", () => {
  const encoder = new PcmFrameEncoder();
  const partial = new Float32Array(319);

  assert.deepEqual(encoder.push(partial, 10), []);

  const [binary] = encoder.push(new Float32Array([0]), 10 + (319 / 16000));
  assert.ok(binary instanceof ArrayBuffer);
  assert.equal(binary.byteLength, 658);
  const header = new DataView(binary, 0, PCM_HEADER_BYTES);
  assert.equal(header.getUint8(0), 1);
  assert.equal(header.getUint32(1, false), 0);
  assert.equal(header.getFloat64(5, false), 10);
  assert.equal(header.getUint32(13, false), 16000);
  assert.equal(header.getUint8(17), 1);
  const pcm = new Int16Array(binary, PCM_HEADER_BYTES, 320);
  assert.equal(pcm[0], 0);
  assert.equal(pcm[1], 0);
  assert.equal(pcm[2], 0);
  assert.equal(pcm[319], 0);

  const [second] = encoder.push(makeLiteralFrame(), 10.02);
  assert.equal(new DataView(second).getUint32(1, false), 1);
  const secondPcm = new Int16Array(second, PCM_HEADER_BYTES, 320);
  assert.equal(secondPcm[0], 0);
  assert.equal(secondPcm[1], 32767);
  assert.equal(secondPcm[2], -32768);
});

test("resampler retains source-rate phase across worklet process boundaries", () => {
  const resampler = new StreamingResampler({ sourceSampleRate: 32000, targetSampleRate: 16000 });
  const first = resampler.push(new Float32Array(320).fill(0.25), 3);
  const second = resampler.push(new Float32Array(322).fill(-0.5), 3.01);

  assert.equal(first.samples.length, 160);
  assert.equal(second.samples.length, 161);
  assert.equal(first.captureTimestamp, 3);
  assert.equal(second.captureTimestamp, 3.01);
  assert.equal(first.samples[0], 0.25);
  assert.equal(second.samples[0], -0.5);
});

test("resampler does not skip a source sample when a 48kHz process block ends mid-phase", () => {
  const resampler = new StreamingResampler({ sourceSampleRate: 48000, targetSampleRate: 16000 });
  const firstBlock = Float32Array.from({ length: 128 }, (_, index) => index);
  const secondBlock = Float32Array.from({ length: 128 }, (_, index) => index + 128);

  const first = resampler.push(firstBlock, 5);
  const second = resampler.push(secondBlock, 5 + (128 / 48000));

  assert.equal(first.samples.at(-1), 126);
  assert.equal(second.samples[0], 129);
  assert.ok(Math.abs(second.captureTimestamp - (5 + (129 / 48000))) < 1e-12);
});

test("PcmMicrophoneInput sends transferred binary buffers and stops every owned track", async () => {
  const events = [];
  const track = { stop: () => events.push("track-stopped") };
  const stream = { getTracks: () => [track] };
  const source = {
    connect: (node) => events.push(["source-connect", node]),
    disconnect: () => events.push("source-disconnected"),
  };
  let getUserMediaCalls = 0;
  class FakeAudioContext {
    constructor() {
      this.audioWorklet = { addModule: async (url) => events.push(["worklet-module", url]) };
      this.destination = { name: "destination" };
      this.state = "running";
    }

    createMediaStreamSource(receivedStream) {
      assert.equal(receivedStream, stream);
      return source;
    }

    close() {
      events.push("context-closed");
      return Promise.resolve();
    }
  }
  class FakeAudioWorkletNode {
    constructor(context, processorName) {
      assert.ok(context instanceof FakeAudioContext);
      assert.equal(processorName, "pcm-capture-processor");
      this.port = { onmessage: null, close: () => events.push("port-closed") };
    }

    connect(destination) {
      assert.equal(destination.name, "destination");
      events.push("worklet-connected");
    }

    disconnect() {
      events.push("worklet-disconnected");
    }
  }
  const previousNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  const previousAudioContext = globalThis.AudioContext;
  const previousAudioWorkletNode = globalThis.AudioWorkletNode;
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: {
    mediaDevices: {
      getUserMedia: async (constraints) => {
        getUserMediaCalls += 1;
        assert.deepEqual(constraints, {
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        return stream;
      },
    },
  } });
  globalThis.AudioContext = FakeAudioContext;
  globalThis.AudioWorkletNode = FakeAudioWorkletNode;

  try {
    const chunks = [];
    const input = new PcmMicrophoneInput({
      onChunk: (chunk) => chunks.push(chunk),
      onStateChange: (state) => events.push(state),
    });
    await input.start();
    await input.start();
    const emitted = new ArrayBuffer(658);
    input.worklet.port.onmessage({ data: emitted });
    await input.stop();
    await input.stop();

    assert.equal(getUserMediaCalls, 1);
    assert.deepEqual(chunks, [emitted]);
    assert.deepEqual(events.filter((value) => typeof value === "string"), [
      "worklet-connected", "录音中", "port-closed", "worklet-disconnected", "source-disconnected", "track-stopped", "context-closed", "已停止",
    ]);
  } finally {
    Object.defineProperty(globalThis, "navigator", previousNavigator);
    globalThis.AudioContext = previousAudioContext;
    globalThis.AudioWorkletNode = previousAudioWorkletNode;
  }
});
