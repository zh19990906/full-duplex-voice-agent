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

export class BrowserAudioOutput {
  constructor() {
    this.context = null;
    this.nextStart = 0;
  }

  async unlock() {
    this.context ||= new AudioContext();
    if (this.context.state === "suspended") await this.context.resume();
  }

  async playBase64(audioData, sampleRate = 24000) {
    if (!audioData) return;
    this.context ||= new AudioContext();
    if (this.context.state === "suspended") await this.context.resume();
    const binary = atob(audioData);
    const pcm = new Int16Array(binary.length / 2);
    for (let index = 0; index < pcm.length; index += 1) {
      const low = binary.charCodeAt(index * 2);
      const high = binary.charCodeAt(index * 2 + 1);
      const value = (high << 8) | low;
      pcm[index] = value & 0x8000 ? value - 0x10000 : value;
    }
    const buffer = this.context.createBuffer(1, pcm.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let index = 0; index < pcm.length; index += 1) channel[index] = pcm[index] / 32768;
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    this.nextStart = Math.max(this.nextStart, this.context.currentTime);
    source.start(this.nextStart);
    this.nextStart += buffer.duration;
  }

  stop() {
    this.nextStart = 0;
    this.context?.close();
    this.context = null;
  }
}
