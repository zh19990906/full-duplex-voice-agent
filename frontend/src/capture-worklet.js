const PROTOCOL_VERSION = 1;
const TARGET_SAMPLE_RATE = 16000;
const MONO_CHANNELS = 1;
const FRAME_SAMPLES = 320;
const HEADER_BYTES = 18;

const clampPcm16 = (sample) => {
  const clamped = Math.max(-1, Math.min(1, sample));
  return Math.round(clamped < 0 ? clamped * 32768 : clamped * 32767);
};

export class PcmFrameEncoder {
  constructor({ sampleRate = TARGET_SAMPLE_RATE, frameSamples = FRAME_SAMPLES } = {}) {
    if (sampleRate !== TARGET_SAMPLE_RATE || frameSamples !== FRAME_SAMPLES) {
      throw new RangeError("V1 PCM frames must be 16kHz mono with 320 samples");
    }
    this.sampleRate = sampleRate;
    this.frameSamples = frameSamples;
    this.sequence = 0;
    this.pendingSamples = [];
    this.frameTimestamp = null;
  }

  push(samples, captureTimestamp) {
    const frames = [];
    for (let index = 0; index < samples.length; index += 1) {
      if (this.pendingSamples.length === 0) {
        this.frameTimestamp = captureTimestamp + (index / this.sampleRate);
      }
      this.pendingSamples.push(samples[index]);
      if (this.pendingSamples.length === this.frameSamples) {
        frames.push(this.#encodeFrame());
      }
    }
    return frames;
  }

  #encodeFrame() {
    const frame = new ArrayBuffer(HEADER_BYTES + (this.frameSamples * Int16Array.BYTES_PER_ELEMENT));
    const header = new DataView(frame);
    header.setUint8(0, PROTOCOL_VERSION);
    header.setUint32(1, this.sequence, false);
    header.setFloat64(5, this.frameTimestamp, false);
    header.setUint32(13, this.sampleRate, false);
    header.setUint8(17, MONO_CHANNELS);
    for (let index = 0; index < this.frameSamples; index += 1) {
      header.setInt16(HEADER_BYTES + (index * Int16Array.BYTES_PER_ELEMENT), clampPcm16(this.pendingSamples[index]), true);
    }
    this.pendingSamples = [];
    this.frameTimestamp = null;
    this.sequence += 1;
    return frame;
  }
}

export class StreamingResampler {
  constructor({ sourceSampleRate, targetSampleRate = TARGET_SAMPLE_RATE }) {
    this.sourceSampleRate = sourceSampleRate;
    this.targetSampleRate = targetSampleRate;
    this.sourceSamples = [];
    this.sourceStartTime = null;
    this.nextSourcePosition = 0;
    this.sourceSamplesPerTargetSample = sourceSampleRate / targetSampleRate;
  }

  push(samples, captureTimestamp) {
    if (this.sourceSamples.length === 0) this.sourceStartTime = captureTimestamp;
    this.sourceSamples.push(...samples);
    const output = [];
    let firstTimestamp = null;
    while (this.nextSourcePosition + 1 < this.sourceSamples.length) {
      const lowerIndex = Math.floor(this.nextSourcePosition);
      const fraction = this.nextSourcePosition - lowerIndex;
      const lower = this.sourceSamples[lowerIndex];
      const upper = this.sourceSamples[lowerIndex + 1];
      if (firstTimestamp === null) {
        firstTimestamp = this.sourceStartTime + (this.nextSourcePosition / this.sourceSampleRate);
      }
      output.push(lower + ((upper - lower) * fraction));
      this.nextSourcePosition += this.sourceSamplesPerTargetSample;
    }
    const consumed = Math.min(Math.floor(this.nextSourcePosition), this.sourceSamples.length);
    if (consumed > 0) {
      this.sourceSamples.splice(0, consumed);
      this.sourceStartTime += consumed / this.sourceSampleRate;
      this.nextSourcePosition -= consumed;
    }
    return {
      samples: Float32Array.from(output),
      captureTimestamp: firstTimestamp ?? captureTimestamp,
    };
  }
}

const WorkletProcessorBase = globalThis.AudioWorkletProcessor || class {};

export class PcmCaptureProcessor extends WorkletProcessorBase {
  constructor() {
    super();
    const sourceSampleRate = typeof sampleRate === "number" ? sampleRate : TARGET_SAMPLE_RATE;
    this.encoder = new PcmFrameEncoder();
    this.resampler = new StreamingResampler({ sourceSampleRate });
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input?.length) return true;
    const output = this.resampler.push(input, currentTime);
    for (const frame of this.encoder.push(output.samples, output.captureTimestamp)) {
      this.port.postMessage(frame, [frame]);
    }
    return true;
  }
}

if (typeof globalThis.registerProcessor === "function") {
  globalThis.registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
}
