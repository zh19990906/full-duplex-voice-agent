export class PcmMicrophoneInput {
  constructor({ onChunk, onStateChange = () => {} } = {}) {
    this.onChunk = onChunk;
    this.onStateChange = onStateChange;
    this.stream = null;
    this.context = null;
    this.source = null;
    this.worklet = null;
    this.starting = null;
    this.stopping = null;
    this.cancelStart = false;
  }

  start() {
    if (this.starting) return this.starting;
    if (this.stopping) return this.stopping.then(() => this.start());
    if (this.context) return Promise.resolve();
    this.cancelStart = false;
    this.starting = this.#start().finally(() => { this.starting = null; });
    return this.starting;
  }

  async #start() {
    let stream = null;
    let context = null;
    let source = null;
    let worklet = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      if (this.cancelStart) return this.#release({ stream });
      context = new AudioContext();
      await context.audioWorklet.addModule(new URL("./capture-worklet.js", import.meta.url));
      if (this.cancelStart) return this.#release({ stream, context });
      if (context.state === "suspended") await context.resume();
      if (this.cancelStart) return this.#release({ stream, context });
      source = context.createMediaStreamSource(stream);
      worklet = new AudioWorkletNode(context, "pcm-capture-processor");
      worklet.port.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) this.onChunk?.(event.data);
      };
      source.connect(worklet);
      worklet.connect(context.destination);
      if (this.cancelStart) return this.#release({ stream, context, source, worklet });
      this.stream = stream;
      this.context = context;
      this.source = source;
      this.worklet = worklet;
      this.onStateChange("录音中");
    } catch (error) {
      const ownsResources = this.stream === stream || this.context === context
        || this.source === source || this.worklet === worklet;
      await this.#release(ownsResources ? this : { stream, context, source, worklet });
      throw error;
    }
  }

  stop() {
    if (this.stopping) return this.stopping;
    this.cancelStart = true;
    const start = this.starting;
    this.stopping = (async () => {
      if (start) await start;
      const released = await this.#release();
      if (released) this.onStateChange("已停止");
    })().finally(() => { this.stopping = null; });
    return this.stopping;
  }

  async #release(resources = this) {
    const { stream, context, source, worklet } = resources;
    const hasResources = Boolean(stream || context || source || worklet);
    if (resources === this) {
      this.stream = null;
      this.context = null;
      this.source = null;
      this.worklet = null;
    }
    const safely = async (operation) => {
      try {
        await operation();
      } catch {
        // Continue releasing the remaining browser resources.
      }
    };
    if (worklet) {
      worklet.port.onmessage = null;
      await safely(() => worklet.port.close?.());
      await safely(() => worklet.disconnect?.());
    }
    await safely(() => source?.disconnect?.());
    for (const track of stream?.getTracks() || []) await safely(() => track.stop());
    await safely(() => context?.close?.());
    return hasResources;
  }
}

export const MicrophoneInput = PcmMicrophoneInput;

const asInt16 = (audio) => {
  if (audio instanceof Int16Array) return new Int16Array(audio);
  if (audio instanceof ArrayBuffer) return new Int16Array(audio.slice(0));
  if (typeof audio !== "string" || !audio) return null;
  const binary = atob(audio);
  const pcm = new Int16Array(binary.length / 2);
  for (let index = 0; index < pcm.length; index += 1) {
    const low = binary.charCodeAt(index * 2);
    const high = binary.charCodeAt(index * 2 + 1);
    pcm[index] = (high << 8) | low;
  }
  return pcm;
};

const identityMatches = (item, responseId, epoch, segmentId) => (
  item.response_id === responseId
  && item.generation_epoch === epoch
  && item.segment_id === segmentId
);

export class PlaybackQueue {
  constructor() {
    this.currentEpoch = 0;
    this.pending = [];
    this.active = [];
    this.paused = [];
  }

  get staleAudioCount() {
    return [...this.pending, ...this.active, ...this.paused]
      .filter((item) => item.generation_epoch < this.currentEpoch).length;
  }

  setEpoch(epoch) {
    if (!Number.isInteger(epoch) || epoch < this.currentEpoch) return false;
    this.currentEpoch = epoch;
    const keepCurrent = (item) => item.generation_epoch >= epoch;
    this.pending = this.pending.filter(keepCurrent);
    this.active = this.active.filter(keepCurrent);
    this.paused = this.paused.filter(keepCurrent);
    return true;
  }

  enqueue(item) {
    if (!item || item.generation_epoch < this.currentEpoch) return false;
    if (item.generation_epoch > this.currentEpoch) this.setEpoch(item.generation_epoch);
    this.pending.push(item);
    return true;
  }

  markActive(responseId, epoch, segmentId) {
    const index = this.pending.findIndex((item) => identityMatches(item, responseId, epoch, segmentId));
    if (index < 0) return false;
    this.active.push(this.pending.splice(index, 1)[0]);
    return true;
  }

  pauseResponse(responseId) {
    const matches = (item) => item.response_id === responseId;
    const paused = [...this.pending.filter(matches), ...this.active.filter(matches)];
    this.pending = this.pending.filter((item) => !matches(item));
    this.active = this.active.filter((item) => !matches(item));
    this.paused.push(...paused);
    return paused;
  }

  stopResponse(responseId) {
    const matches = (item) => item.response_id === responseId;
    const stopped = [...this.pending.filter(matches), ...this.active.filter(matches), ...this.paused.filter(matches)];
    this.pending = this.pending.filter((item) => !matches(item));
    this.active = this.active.filter((item) => !matches(item));
    this.paused = this.paused.filter((item) => !matches(item));
    return stopped;
  }

  complete(responseId, epoch, segmentId) {
    const matches = (item) => identityMatches(item, responseId, epoch, segmentId);
    this.pending = this.pending.filter((item) => !matches(item));
    this.active = this.active.filter((item) => !matches(item));
    this.paused = this.paused.filter((item) => !matches(item));
  }

  clear() {
    this.pending = [];
    this.active = [];
    this.paused = [];
  }
}

export class BrowserPlaybackCoordinator {
  constructor({
    context = null,
    workletNode = null,
    gainNode = null,
    onPlaybackAck = () => {},
    duckLevel = 0.25,
    duckRampSeconds = 0.1,
  } = {}) {
    this.context = context;
    this.workletNode = workletNode;
    this.gainNode = gainNode;
    this.queue = new PlaybackQueue();
    this.onPlaybackAck = onPlaybackAck;
    this.duckLevel = duckLevel;
    this.duckRampSeconds = Math.min(duckRampSeconds, 0.1);
    this.nextLegacySegmentId = 0;
    this.#attachWorklet();
  }

  async unlock() {
    this.context ||= new AudioContext();
    if (this.context.state === "suspended") await this.context.resume();
    if (!this.workletNode) {
      await this.context.audioWorklet.addModule(new URL("./playback-worklet.js", import.meta.url));
      this.gainNode = this.context.createGain();
      this.gainNode.gain.value = 1;
      this.workletNode = new AudioWorkletNode(this.context, "pcm-playback-processor");
      this.workletNode.connect(this.gainNode);
      this.gainNode.connect(this.context.destination);
      this.#attachWorklet();
    }
    this.#flushPending();
  }

  enqueue(payload) {
    const item = this.#normaliseItem(payload);
    if (!item || !this.queue.enqueue(item)) return false;
    if (this.workletNode) this.#sendItem(item);
    return true;
  }

  setEpoch(epoch) {
    if (!this.queue.setEpoch(epoch)) return false;
    this.workletNode?.port.postMessage({ type: "set_epoch", generation_epoch: epoch });
    return true;
  }

  duck() { this.#rampGain(this.duckLevel); }
  restore() { this.#rampGain(1); }

  pauseResponse(responseId) {
    const paused = this.queue.pauseResponse(responseId);
    if (paused.length) this.workletNode?.port.postMessage({ type: "pause_response", response_id: responseId });
    return paused.length > 0;
  }

  stopResponse(responseId) {
    const stopped = this.queue.stopResponse(responseId);
    if (stopped.length) this.workletNode?.port.postMessage({ type: "stop_response", response_id: responseId });
    return stopped.length > 0;
  }

  async close() {
    this.queue.clear();
    this.workletNode?.port.postMessage({ type: "stop_all" });
    this.workletNode?.disconnect?.();
    this.gainNode?.disconnect?.();
    const context = this.context;
    this.context = null;
    this.workletNode = null;
    this.gainNode = null;
    await context?.close?.();
  }

  #normaliseItem(payload) {
    const pcm16 = asInt16(payload?.pcm16 ?? payload?.audio_data);
    if (!pcm16 || payload?.channels && payload.channels !== 1) return null;
    const generation_epoch = Number.isInteger(payload?.generation_epoch ?? payload?.epoch)
      ? (payload.generation_epoch ?? payload.epoch) : 0;
    const segment_id = Number.isInteger(payload?.segment_id ?? payload?.segmentId)
      ? (payload.segment_id ?? payload.segmentId) : this.nextLegacySegmentId++;
    return {
      response_id: payload?.response_id ?? payload?.responseId ?? "legacy-response",
      generation_epoch,
      segment_id,
      pcm16,
      sample_rate: payload?.sample_rate ?? payload?.sampleRate ?? 24000,
      channels: 1,
    };
  }

  #attachWorklet() {
    if (!this.workletNode) return;
    this.workletNode.port.onmessage = ({ data }) => this.#handleWorkletEvent(data);
    this.workletNode.port.postMessage({ type: "set_epoch", generation_epoch: this.queue.currentEpoch });
  }

  #flushPending() {
    for (const item of [...this.queue.pending]) this.#sendItem(item);
  }

  #sendItem(item) {
    if (!this.queue.markActive(item.response_id, item.generation_epoch, item.segment_id)) return;
    this.workletNode.port.postMessage({
      type: "enqueue",
      item: { ...item, pcm16: item.pcm16.buffer },
    }, [item.pcm16.buffer]);
  }

  #handleWorkletEvent(event) {
    if (!event || event.generation_epoch < this.queue.currentEpoch) return;
    if (!["progress", "completed", "stopped", "paused"].includes(event.type)) return;
    const acknowledgement = {
      response_id: event.response_id,
      generation_epoch: event.generation_epoch,
      segment_id: event.segment_id,
      sample_offset: Math.floor(event.sample_offset),
      audio_time: event.audio_time,
    };
    this.onPlaybackAck(acknowledgement);
    if (["completed", "stopped", "paused"].includes(event.type)) {
      this.queue.complete(event.response_id, event.generation_epoch, event.segment_id);
    }
  }

  #rampGain(target) {
    const gain = this.gainNode?.gain;
    if (!gain) return;
    const now = this.context?.currentTime ?? 0;
    gain.cancelScheduledValues(now);
    gain.setValueAtTime(gain.value, now);
    gain.linearRampToValueAtTime(target, now + this.duckRampSeconds);
  }
}

export class BrowserAudioOutput {
  constructor() {
    this.coordinator = new BrowserPlaybackCoordinator();
    this.segmentId = 0;
  }

  async unlock() {
    await this.coordinator.unlock();
  }

  async playBase64(audioData, sampleRate = 24000) {
    await this.unlock();
    this.coordinator.enqueue({
      response_id: "legacy-response",
      generation_epoch: 0,
      segment_id: this.segmentId++,
      audio_data: audioData,
      sample_rate: sampleRate,
      channels: 1,
    });
  }

  stop() {
    return this.coordinator.close();
  }
}
