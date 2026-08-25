import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const { BrowserPlaybackCoordinator, PlaybackQueue } = await import("../frontend/src/audio.js");
const { PlaybackRenderer } = await import("../frontend/src/playback-worklet.js");
const { AgentWebSocket } = await import("../frontend/src/websocket.js");

class FakePort {
  constructor() {
    this.messages = [];
    this.onmessage = null;
  }

  postMessage(message) {
    this.messages.push(message);
  }

  emit(data) {
    this.onmessage?.({ data });
  }
}

const deferred = () => {
  let resolve;
  const promise = new Promise((resolvePromise) => { resolve = resolvePromise; });
  return { promise, resolve };
};

const replaceGlobal = (name, value) => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  return () => {
    if (previous) Object.defineProperty(globalThis, name, previous);
    else delete globalThis[name];
  };
};

class FakeAudioParam {
  constructor(value = 1) {
    this.value = value;
    this.calls = [];
  }

  cancelScheduledValues(time) { this.calls.push(["cancel", time]); }
  setValueAtTime(value, time) { this.value = value; this.calls.push(["set", value, time]); }
  linearRampToValueAtTime(value, time) { this.value = value; this.calls.push(["ramp", value, time]); }
}

const createCoordinator = () => {
  const port = new FakePort();
  const context = { currentTime: 8, closeCalls: 0, close() { this.closeCalls += 1; } };
  const gain = { gain: new FakeAudioParam(), connect() {} };
  const worklet = { port, connect() {} };
  const acknowledgements = [];
  return {
    acknowledgements,
    context,
    gain,
    port,
    coordinator: new BrowserPlaybackCoordinator({
      context,
      gainNode: gain,
      workletNode: worklet,
      onPlaybackAck: (ack) => acknowledgements.push(ack),
    }),
  };
};

const item = (responseId, epoch = 4, segmentId = 1) => ({
  response_id: responseId,
  generation_epoch: epoch,
  segment_id: segmentId,
  pcm16: new Int16Array([0, 16384, -16384, 32767]),
  sample_rate: 24000,
  channels: 1,
});

test("queue rejects stale epochs and purges stale pending and active audio", () => {
  const queue = new PlaybackQueue();

  assert.equal(queue.setEpoch(4), true);
  assert.equal(queue.enqueue(item("old", 3)), false);
  assert.equal(queue.enqueue(item("pending", 4)), true);
  queue.markActive("pending", 4, 1);
  assert.equal(queue.enqueue(item("fresh", 5)), true);

  assert.equal(queue.setEpoch(5), true);
  assert.equal(queue.currentEpoch, 5);
  assert.equal(queue.active.length, 0);
  assert.deepEqual(queue.pending.map(({ response_id }) => response_id), ["fresh"]);
  assert.equal(queue.staleAudioCount, 0);
  assert.equal(queue.setEpoch(4), false);
});

test("coordinator stops only the named response without closing its AudioContext", () => {
  const { coordinator, context, port } = createCoordinator();
  coordinator.setEpoch(4);
  coordinator.enqueue(item("first"));
  coordinator.enqueue(item("second", 4, 2));

  coordinator.stopResponse("first");

  assert.deepEqual(coordinator.queue.active.map(({ response_id }) => response_id), ["second"]);
  assert.deepEqual(port.messages.at(-1), { type: "stop_response", response_id: "first" });
  assert.equal(context.closeCalls, 0);
});

test("coordinator pauses only the named response and leaves unrelated output queued", () => {
  const { coordinator, context, port } = createCoordinator();
  coordinator.setEpoch(4);
  coordinator.enqueue(item("first"));
  coordinator.enqueue(item("second", 4, 2));

  coordinator.pauseResponse("first");

  assert.deepEqual(coordinator.queue.active.map(({ response_id }) => response_id), ["second"]);
  assert.deepEqual(coordinator.queue.paused.map(({ response_id }) => response_id), ["first"]);
  assert.deepEqual(port.messages.at(-1), { type: "pause_response", response_id: "first" });
  assert.equal(context.closeCalls, 0);
});

test("coordinator forwards a newer epoch to the worklet after purging stale active output", () => {
  const { coordinator, port } = createCoordinator();
  coordinator.setEpoch(4);
  coordinator.enqueue(item("old"));

  assert.equal(coordinator.setEpoch(5), true);

  assert.equal(coordinator.queue.staleAudioCount, 0);
  assert.equal(coordinator.queue.active.length, 0);
  assert.deepEqual(port.messages.at(-1), { type: "set_epoch", generation_epoch: 5 });
});

test("coordinator ducks using a gain ramp no longer than 100ms and restores smoothly", () => {
  const { coordinator, gain, context } = createCoordinator();

  coordinator.duck();
  context.currentTime = 9;
  coordinator.restore();

  assert.deepEqual(gain.gain.calls, [
    ["cancel", 8], ["set", 1, 8], ["ramp", 0.25, 8.1],
    ["cancel", 9], ["set", 0.25, 9], ["ramp", 1, 9.1],
  ]);
});

test("coordinator emits exact playback ACK payloads for progress and terminal stop", () => {
  const { coordinator, port, acknowledgements } = createCoordinator();
  coordinator.setEpoch(4);
  coordinator.enqueue(item("r1", 4, 7));

  port.emit({ type: "progress", response_id: "r1", generation_epoch: 4, segment_id: 7, sample_offset: 2, audio_time: 10.125 });
  port.emit({ type: "stopped", response_id: "r1", generation_epoch: 4, segment_id: 7, sample_offset: 4, audio_time: 10.25 });

  assert.deepEqual(acknowledgements, [
    { response_id: "r1", generation_epoch: 4, segment_id: 7, sample_offset: 2, audio_time: 10.125 },
    { response_id: "r1", generation_epoch: 4, segment_id: 7, sample_offset: 4, audio_time: 10.25 },
  ]);
  assert.equal(coordinator.queue.active.length, 0);
});

test("coordinator decodes base64 PCM16 and forwards the source sample rate to the used worklet", () => {
  const { coordinator, port } = createCoordinator();
  coordinator.setEpoch(1);
  const source = new Int16Array([0, 16384, -16384]);
  const audio_data = Buffer.from(source.buffer).toString("base64");

  assert.equal(coordinator.enqueue({
    response_id: "base64", generation_epoch: 1, segment_id: 3, audio_data, sample_rate: 24000, channels: 1,
  }), true);

  const message = port.messages.at(-1);
  assert.equal(message.type, "enqueue");
  assert.equal(message.item.sample_rate, 24000);
  assert.deepEqual([...new Int16Array(message.item.pcm16)], [0, 16384, -16384]);
});

test("concurrent unlock calls create one worklet and gain chain", async () => {
  const moduleGate = deferred();
  const contexts = [];
  const nodes = [];
  class Context {
    constructor() {
      this.state = "running";
      this.destination = {};
      this.gains = [];
      this.audioWorklet = { addModule: () => { this.moduleCalls = (this.moduleCalls || 0) + 1; return moduleGate.promise; } };
      contexts.push(this);
    }
    createGain() { const gain = { gain: new FakeAudioParam(), connect() {} }; this.gains.push(gain); return gain; }
  }
  class WorkletNode {
    constructor() { this.port = new FakePort(); nodes.push(this); }
    connect() {}
  }
  const restoreContext = replaceGlobal("AudioContext", Context);
  const restoreWorklet = replaceGlobal("AudioWorkletNode", WorkletNode);

  try {
    const coordinator = new BrowserPlaybackCoordinator();
    const first = coordinator.unlock();
    const second = coordinator.unlock();
    assert.equal(contexts.length, 1);
    assert.equal(contexts[0].moduleCalls, 1);

    moduleGate.resolve();
    await Promise.all([first, second]);

    assert.equal(nodes.length, 1);
    assert.equal(contexts[0].gains.length, 1);
    assert.equal(coordinator.workletNode, nodes[0]);
  } finally {
    restoreContext();
    restoreWorklet();
  }
});

test("close during initialization invalidates the old chain and reopens at epoch zero", async () => {
  const firstModule = deferred();
  const contexts = [];
  class Context {
    constructor() {
      this.state = "running";
      this.destination = {};
      this.closeCalls = 0;
      this.audioWorklet = { addModule: () => (contexts.length === 1 ? firstModule.promise : Promise.resolve()) };
      contexts.push(this);
    }
    createGain() { return { gain: new FakeAudioParam(), connect() {}, disconnect() {} }; }
    close() { this.closeCalls += 1; }
  }
  class WorkletNode {
    constructor() { this.port = new FakePort(); }
    connect() {}
    disconnect() {}
  }
  const restoreContext = replaceGlobal("AudioContext", Context);
  const restoreWorklet = replaceGlobal("AudioWorkletNode", WorkletNode);

  try {
    const coordinator = new BrowserPlaybackCoordinator();
    const opening = coordinator.unlock();
    coordinator.setEpoch(6);
    coordinator.enqueue(item("discard-on-close", 6, 1));
    const closing = coordinator.close();
    firstModule.resolve();
    await Promise.all([opening, closing]);

    assert.equal(contexts[0].closeCalls, 1);
    assert.equal(coordinator.context, null);
    assert.equal(coordinator.workletNode, null);
    assert.equal(coordinator.queue.currentEpoch, 0);
    assert.deepEqual([coordinator.queue.pending, coordinator.queue.active, coordinator.queue.paused], [[], [], []]);

    await coordinator.unlock();
    assert.equal(contexts.length, 2);
    assert.equal(coordinator.queue.currentEpoch, 0);
  } finally {
    restoreContext();
    restoreWorklet();
  }
});

test("close detaches stale port handlers before a former response can acknowledge", async () => {
  const { coordinator, port, acknowledgements } = createCoordinator();
  coordinator.setEpoch(4);
  coordinator.enqueue(item("old", 4, 2));
  const oldHandler = port.onmessage;

  await coordinator.close();
  oldHandler({ data: { type: "progress", response_id: "old", generation_epoch: 4, segment_id: 2, sample_offset: 1, audio_time: 9 } });

  assert.equal(port.onmessage, null);
  assert.deepEqual(acknowledgements, []);
  assert.equal(coordinator.queue.currentEpoch, 0);
});

test("coordinator rejects invalid sample rates without blocking a following valid segment", () => {
  const { coordinator, port } = createCoordinator();
  coordinator.setEpoch(4);

  assert.equal(coordinator.enqueue({ ...item("zero"), sample_rate: 0 }), false);
  assert.equal(coordinator.enqueue({ ...item("negative", 4, 2), sample_rate: -1 }), false);
  assert.equal(coordinator.enqueue({ ...item("infinite", 4, 3), sample_rate: Infinity }), false);
  assert.equal(coordinator.enqueue({ ...item("nan", 4, 4), sample_rate: Number.NaN }), false);
  assert.equal(coordinator.enqueue({ ...item("valid", 4, 5), sample_rate: 24000 }), true);

  assert.deepEqual(coordinator.queue.active.map(({ response_id }) => response_id), ["valid"]);
  assert.equal(port.messages.at(-1).item.sample_rate, 24000);
});

test("coordinator limits progress ACKs to one per 50ms while sending terminal ACKs immediately", () => {
  const { coordinator, port, acknowledgements } = createCoordinator();
  coordinator.setEpoch(4);
  coordinator.enqueue(item("cadence", 4, 5));
  const event = (type, sample_offset, audio_time) => port.emit({
    type, response_id: "cadence", generation_epoch: 4, segment_id: 5, sample_offset, audio_time,
  });

  event("progress", 1, 1);
  event("progress", 2, 1.01);
  event("progress", 3, 1.049);
  event("progress", 4, 1.05);
  event("stopped", 5, 1.051);

  assert.deepEqual(acknowledgements.map(({ sample_offset, audio_time }) => [sample_offset, audio_time]), [
    [1, 1], [4, 1.05], [5, 1.051],
  ]);
});

test("pre-init pause and stop emit terminal zero-offset acknowledgements", () => {
  const acknowledgements = [];
  const coordinator = new BrowserPlaybackCoordinator({
    context: { currentTime: 12.5 },
    onPlaybackAck: (ack) => acknowledgements.push(ack),
  });
  coordinator.setEpoch(4);
  coordinator.enqueue(item("pause-before-init", 4, 1));
  coordinator.pauseResponse("pause-before-init");
  coordinator.enqueue(item("stop-before-init", 4, 2));
  coordinator.stopResponse("stop-before-init");

  assert.deepEqual(acknowledgements, [
    { response_id: "pause-before-init", generation_epoch: 4, segment_id: 1, sample_offset: 0, audio_time: 12.5 },
    { response_id: "stop-before-init", generation_epoch: 4, segment_id: 2, sample_offset: 0, audio_time: 12.5 },
  ]);
  assert.deepEqual(coordinator.queue.pending, []);
  assert.deepEqual(coordinator.queue.paused, []);
});

test("worklet renderer plays 24kHz PCM at the correct 48kHz AudioContext speed and reports source offsets", () => {
  const events = [];
  const renderer = new PlaybackRenderer({ outputSampleRate: 48000, onEvent: (event) => events.push(event) });
  renderer.enqueue({ ...item("rate", 1, 2), pcm16: new Int16Array([0, 16384, 32767]) });
  const output = new Float32Array(6);

  renderer.render(output, 3);

  assert.deepEqual([...output], [0, 0, 0.5, 0.5, 0.999969482421875, 0.999969482421875]);
  assert.deepEqual(events.map(({ type, sample_offset }) => [type, sample_offset]), [["progress", 3], ["completed", 3]]);
});

test("worklet emits terminal zero-offset ACK events when a pending response is paused or stopped", () => {
  const events = [];
  const renderer = new PlaybackRenderer({ outputSampleRate: 48000, onEvent: (event) => events.push(event) });
  renderer.enqueue(item("pause-pending", 1, 1));
  renderer.enqueue(item("stop-pending", 1, 2));

  renderer.pauseResponse("pause-pending", 4.5);
  renderer.stopResponse("stop-pending", 4.75);

  assert.deepEqual(events.map(({ type, response_id, sample_offset, audio_time }) => ({ type, response_id, sample_offset, audio_time })), [
    { type: "paused", response_id: "pause-pending", sample_offset: 0, audio_time: 4.5 },
    { type: "stopped", response_id: "stop-pending", sample_offset: 0, audio_time: 4.75 },
  ]);
});

test("playback worklet is loaded by the coordinator rather than left as an unused module", async () => {
  const audio = await readFile(new URL("../frontend/src/audio.js", import.meta.url), "utf8");
  const worklet = await readFile(new URL("../frontend/src/playback-worklet.js", import.meta.url), "utf8");

  assert.match(audio, /addModule\(new URL\("\.\/playback-worklet\.js", import\.meta\.url\)\)/);
  assert.match(audio, /new AudioWorkletNode\([^,]+, "pcm-playback-processor"\)/);
  assert.match(worklet, /registerProcessor\("pcm-playback-processor"/);
});

test("websocket sends only the exact playback ACK payload while open", () => {
  const sent = [];
  const socket = { readyState: 0, send: (message) => sent.push(message) };
  const client = new AgentWebSocket("ws://example.test/ws");
  client.socket = socket;
  const acknowledgement = { response_id: "r1", generation_epoch: 3, segment_id: 4, sample_offset: 22, audio_time: 8.5 };

  assert.equal(client.sendPlaybackAck(acknowledgement), false);
  socket.readyState = 1;
  assert.equal(client.sendPlaybackAck(acknowledgement), true);
  assert.deepEqual(sent.map(JSON.parse), [{ type: "playback_ack", payload: acknowledgement }]);
});

test("app routes identity-aware audio and playback controls without closing new output", async () => {
  const previousWindow = globalThis.window;
  const previousDocument = globalThis.document;
  const previousFetch = globalThis.fetch;
  const previousWebSocket = globalThis.WebSocket;
  const elements = new Map();
  const element = () => ({
    textContent: "", value: "", append() {}, prepend() {},
    addEventListener(type, listener) { this.listeners ||= new Map(); this.listeners.set(type, listener); },
    async trigger(type) { await this.listeners.get(type)({ preventDefault() {} }); },
  });
  for (const id of ["timeline", "session-status", "messages", "agent-state", "tool-output", "session-id", "create-session", "close-session", "message-form", "message", "start-mic", "stop-mic", "mic-status"]) {
    elements.set(id, element());
  }
  class FakeSocket {
    static OPEN = 1;
    constructor() { this.readyState = 1; this.listeners = new Map(); this.sent = []; FakeSocket.current = this; }
    addEventListener(type, listener) { this.listeners.set(type, listener); }
    send(message) { this.sent.push(message); }
    close() { this.readyState = 3; this.listeners.get("close")?.(); }
    emit(type, data) { this.listeners.get(type)?.(data); }
  }
  const calls = [];
  let playbackOptions;
  const playback = {
    unlock: async () => calls.push(["unlock"]), close: async () => calls.push(["close"]),
    enqueue: (value) => calls.push(["enqueue", value]), duck: () => calls.push(["duck"]),
    restore: () => calls.push(["restore"]), pauseResponse: (value) => calls.push(["pause", value]),
    stopResponse: (value) => calls.push(["stop", value]), setEpoch: (value) => calls.push(["epoch", value]),
  };

  try {
    globalThis.window = { location: { protocol: "http:", host: "localhost:8001", origin: "http://localhost:8001" } };
    globalThis.document = { getElementById: (id) => elements.get(id), createElement: () => element() };
    globalThis.fetch = async () => ({ ok: true, json: async () => ({ session_id: "session-1" }) });
    globalThis.WebSocket = FakeSocket;
    const { bootResearchConsole } = await import("../frontend/src/app.js?playback-control-test");
    const { state } = bootResearchConsole(globalThis.document, {
      playbackFactory: (options) => { playbackOptions = options; return playback; },
    });
    await elements.get("create-session").trigger("click");

    FakeSocket.current.emit("message", { data: JSON.stringify({
      event: "audio", response_id: "root-response", generation_epoch: 8, segment_id: 9,
      payload: { audio_data: "AAABAA==", sample_rate: 24000, channels: 1 },
    }) });
    for (const event of [
      { event: "DUCK" }, { event: "RESTORE" },
      { event: "PAUSE_RESPONSE", payload: { response_id: "pause-me" } },
      { event: "STOP_RESPONSE", response_id: "stop-me" },
      { event: "SET_EPOCH", payload: { generation_epoch: 11 } },
    ]) FakeSocket.current.emit("message", { data: JSON.stringify(event) });
    playbackOptions.onPlaybackAck({ response_id: "root-response", generation_epoch: 8, segment_id: 9, sample_offset: 2, audio_time: 1.5 });

    assert.deepEqual(calls.slice(0, 7), [
      ["unlock"],
      ["enqueue", { response_id: "root-response", generation_epoch: 8, segment_id: 9, audio_data: "AAABAA==", sample_rate: 24000, channels: 1 }],
      ["duck"], ["restore"], ["pause", "pause-me"], ["stop", "stop-me"], ["epoch", 11],
    ]);
    assert.deepEqual(FakeSocket.current.sent.map(JSON.parse), [{
      type: "playback_ack",
      payload: { response_id: "root-response", generation_epoch: 8, segment_id: 9, sample_offset: 2, audio_time: 1.5 },
    }]);
    await elements.get("close-session").trigger("click");
    assert.equal(state.socket, null);
    assert.deepEqual(calls.at(-1), ["close"]);
  } finally {
    globalThis.window = previousWindow;
    globalThis.document = previousDocument;
    globalThis.fetch = previousFetch;
    globalThis.WebSocket = previousWebSocket;
  }
});
