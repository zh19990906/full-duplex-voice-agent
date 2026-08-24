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

  async playBase64(audioData) {
    if (!audioData) return;
    this.context ||= new AudioContext();
    const binary = atob(audioData);
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    // Browser-native decoding is the transport boundary; the UI does not inspect codecs.
    const buffer = await this.context.decodeAudioData(bytes.buffer);
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
