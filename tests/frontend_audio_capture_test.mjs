import assert from "node:assert/strict";
import test from "node:test";

globalThis.window ||= {
  location: {
    protocol: "http:",
    host: "localhost:8001",
    origin: "http://localhost:8001",
  },
};

const { PcmFrameEncoder, StreamingResampler } = await import("../frontend/src/capture-worklet.js");
const { PcmMicrophoneInput } = await import("../frontend/src/audio.js");
const { bootResearchConsole } = await import("../frontend/src/app.js");

const PCM_HEADER_BYTES = 18;

const makeLiteralFrame = () => {
  const samples = new Float32Array(320);
  samples[0] = 0;
  samples[1] = 1;
  samples[2] = -1;
  return samples;
};

const replaceGlobal = (name, value) => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  return () => {
    if (previous) Object.defineProperty(globalThis, name, previous);
    else delete globalThis[name];
  };
};

const deferred = () => {
  let resolve;
  const promise = new Promise((resolvePromise) => { resolve = resolvePromise; });
  return { promise, resolve };
};

const waitFor = async (condition) => {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (condition()) return;
    await Promise.resolve();
  }
  assert.fail("timed out waiting for deterministic microphone action");
};

const createConsoleDocument = () => {
  const elements = new Map();
  const element = () => ({
    textContent: "",
    value: "",
    append: () => {},
    prepend: () => {},
    addEventListener(type, listener) { this.listeners ||= new Map(); this.listeners.set(type, listener); },
    async trigger(type) { await this.listeners.get(type)({ preventDefault: () => {} }); },
  });
  for (const id of ["timeline", "session-status", "messages", "agent-state", "tool-output", "session-id", "create-session", "close-session", "message-form", "message", "start-mic", "stop-mic", "mic-status"]) {
    elements.set(id, element());
  }
  return {
    elements,
    documentRef: {
      getElementById: (id) => elements.get(id),
      createElement: () => element(),
    },
  };
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

test("encoder writes PCM16 payload bytes in little-endian order", () => {
  const [binary] = new PcmFrameEncoder().push(makeLiteralFrame(), 1);

  assert.deepEqual([...new Uint8Array(binary, PCM_HEADER_BYTES, 6)], [0x00, 0x00, 0xff, 0x7f, 0x00, 0x80]);
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

test("PcmMicrophoneInput resumes a suspended context before capture", async () => {
  const events = [];
  const stream = { getTracks: () => [{ stop: () => events.push("track-stopped") }] };
  class SuspendedContext {
    constructor() {
      this.state = "suspended";
      this.destination = {};
      this.audioWorklet = { addModule: async () => {} };
    }

    async resume() {
      events.push("resumed");
      this.state = "running";
    }

    createMediaStreamSource() {
      return { connect: () => {}, disconnect: () => events.push("source-disconnected") };
    }

    close() { events.push("context-closed"); }
  }
  class WorkletNode {
    constructor() {
      this.port = { close: () => events.push("port-closed") };
    }

    connect() {}
    disconnect() { events.push("worklet-disconnected"); }
  }
  const restoreNavigator = replaceGlobal("navigator", { mediaDevices: { getUserMedia: async () => stream } });
  const restoreContext = replaceGlobal("AudioContext", SuspendedContext);
  const restoreWorklet = replaceGlobal("AudioWorkletNode", WorkletNode);

  try {
    const input = new PcmMicrophoneInput();
    await input.start();
    assert.deepEqual(events, ["resumed"]);
    await input.stop();
  } finally {
    restoreNavigator();
    restoreContext();
    restoreWorklet();
  }
});

test("PcmMicrophoneInput clears every owned resource when recording-state notification fails", async () => {
  const events = [];
  const stream = { getTracks: () => [{ stop: () => events.push("track-stopped") }] };
  class Context {
    constructor() {
      this.state = "running";
      this.destination = {};
      this.audioWorklet = { addModule: async () => {} };
    }

    createMediaStreamSource() {
      return {
        connect: () => {},
        disconnect: () => events.push("source-disconnected"),
      };
    }

    close() { events.push("context-closed"); }
  }
  class WorkletNode {
    constructor() {
      this.port = { close: () => events.push("port-closed") };
    }

    connect() {}
    disconnect() { events.push("worklet-disconnected"); }
  }
  const restoreNavigator = replaceGlobal("navigator", { mediaDevices: { getUserMedia: async () => stream } });
  const restoreContext = replaceGlobal("AudioContext", Context);
  const restoreWorklet = replaceGlobal("AudioWorkletNode", WorkletNode);

  try {
    const input = new PcmMicrophoneInput({
      onStateChange: () => { throw new Error("recording state failed"); },
    });
    await assert.rejects(input.start(), /recording state failed/);
    assert.deepEqual(events, ["port-closed", "worklet-disconnected", "source-disconnected", "track-stopped", "context-closed"]);
    assert.deepEqual([input.stream, input.context, input.source, input.worklet], [null, null, null, null]);
    await input.stop();
    assert.deepEqual(events, ["port-closed", "worklet-disconnected", "source-disconnected", "track-stopped", "context-closed"]);
  } finally {
    restoreNavigator();
    restoreContext();
    restoreWorklet();
  }
});

test("a second Start click stops the active microphone before replacement", async () => {
  const { elements, documentRef } = createConsoleDocument();
  const restoreDocument = replaceGlobal("document", documentRef);
  const lifecycle = [];
  const microphones = ["first", "second"].map((name) => ({
    start: async () => lifecycle.push(`${name}-start`),
    stop: async () => lifecycle.push(`${name}-stop`),
  }));
  const [, second] = microphones;

  try {
    const { state } = bootResearchConsole(documentRef, {
      microphoneFactory: () => microphones.shift(),
    });
    state.socket = { sendAudio: () => {} };
    await elements.get("start-mic").trigger("click");
    await elements.get("start-mic").trigger("click");

    assert.deepEqual(lifecycle, ["first-start", "first-stop", "second-start"]);
    assert.equal(state.microphone, second);
  } finally {
    restoreDocument();
  }
});

test("queued Start then Stop shuts down the final microphone", async () => {
  const { elements, documentRef } = createConsoleDocument();
  const restoreDocument = replaceGlobal("document", documentRef);
  const events = [];
  const firstGate = deferred();
  const secondGate = deferred();
  const gates = [firstGate, secondGate];
  const microphones = ["first", "second"].map((name, index) => ({
    start: async () => {
      events.push(`${name}-start`);
      await gates[index].promise;
      events.push(`${name}-ready`);
    },
    stop: async () => events.push(`${name}-stop`),
  }));

  try {
    const { state } = bootResearchConsole(documentRef, { microphoneFactory: () => microphones.shift() });
    state.socket = { sendAudio: () => {} };
    const firstStart = elements.get("start-mic").trigger("click");
    await waitFor(() => events.includes("first-start"));
    const secondStart = elements.get("start-mic").trigger("click");
    const stop = elements.get("stop-mic").trigger("click");
    firstGate.resolve();
    await waitFor(() => events.includes("second-start"));
    secondGate.resolve();
    await Promise.all([firstStart, secondStart, stop]);

    assert.deepEqual(events, ["first-start", "first-ready", "first-stop", "second-start", "second-ready", "second-stop"]);
    assert.equal(state.microphone, null);
  } finally {
    restoreDocument();
  }
});

test("queued Start then Close shuts down the final microphone before session state clears", async () => {
  const { elements, documentRef } = createConsoleDocument();
  const restoreDocument = replaceGlobal("document", documentRef);
  const events = [];
  const firstGate = deferred();
  const secondGate = deferred();
  const gates = [firstGate, secondGate];
  const microphones = ["first", "second"].map((name, index) => ({
    start: async () => {
      events.push(`${name}-start`);
      await gates[index].promise;
      events.push(`${name}-ready`);
    },
    stop: async () => events.push(`${name}-stop`),
  }));

  try {
    const { state, api } = bootResearchConsole(documentRef, { microphoneFactory: () => microphones.shift() });
    state.sessionId = "session-1";
    state.socket = { close: () => events.push("socket-closed"), sendAudio: () => {} };
    api.deleteSession = async () => events.push("session-deleted");
    const firstStart = elements.get("start-mic").trigger("click");
    await waitFor(() => events.includes("first-start"));
    const secondStart = elements.get("start-mic").trigger("click");
    const close = elements.get("close-session").trigger("click");
    firstGate.resolve();
    await waitFor(() => events.includes("second-start"));
    secondGate.resolve();
    await Promise.all([firstStart, secondStart, close]);

    assert.deepEqual(events, ["first-start", "first-ready", "first-stop", "second-start", "second-ready", "second-stop", "session-deleted", "socket-closed"]);
    assert.equal(state.microphone, null);
    assert.equal(state.sessionId, null);
  } finally {
    restoreDocument();
  }
});
