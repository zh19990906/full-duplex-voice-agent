export class MicrophoneInput {
  constructor({ onChunk, onStateChange = () => {} } = {}) {
    this.onChunk = onChunk;
    this.onStateChange = onStateChange;
    this.recorder = null;
    this.stream = null;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.recorder = new MediaRecorder(this.stream);
    this.recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) this.onChunk?.(event.data);
    });
    this.recorder.addEventListener("start", () => this.onStateChange("recording"));
    this.recorder.addEventListener("stop", () => this.onStateChange("stopped"));
    this.recorder.start(250);
  }

  stop() {
    this.recorder?.stop();
    this.stream?.getTracks().forEach((track) => track.stop());
    this.recorder = null;
    this.stream = null;
  }
}

export class BrowserAudioOutput {
  constructor() {
    this.context = null;
    this.nextStart = 0;
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
