export class PlaybackRenderer {
  constructor({ outputSampleRate, onEvent = () => {} } = {}) {
    this.outputSampleRate = outputSampleRate || 48000;
    this.onEvent = onEvent;
    this.currentEpoch = 0;
    this.pending = [];
    this.active = null;
  }

  setEpoch(epoch) {
    if (!Number.isInteger(epoch) || epoch < this.currentEpoch) return false;
    this.currentEpoch = epoch;
    this.pending = this.pending.filter((item) => item.generation_epoch >= epoch);
    if (this.active?.generation_epoch < epoch) this.active = null;
    return true;
  }

  enqueue(item) {
    if (item.generation_epoch < this.currentEpoch) return false;
    if (item.generation_epoch > this.currentEpoch) this.setEpoch(item.generation_epoch);
    this.pending.push({ ...item, pcm16: new Int16Array(item.pcm16), position: 0 });
    return true;
  }

  pauseResponse(responseId, audioTime) {
    this.pending = this.pending.filter((item) => {
      if (item.response_id !== responseId) return true;
      this.#emit("paused", item, audioTime);
      return false;
    });
    if (this.active?.response_id === responseId) {
      this.#emit("paused", this.active, audioTime);
      this.active = null;
    }
  }

  stopResponse(responseId, audioTime) {
    this.pending = this.pending.filter((item) => {
      if (item.response_id !== responseId) return true;
      this.#emit("stopped", item, audioTime);
      return false;
    });
    if (this.active?.response_id === responseId) {
      this.#emit("stopped", this.active, audioTime);
      this.active = null;
    }
  }

  render(output, audioTime = 0) {
    output.fill(0);
    for (let frame = 0; frame < output.length; frame += 1) {
      this.active ||= this.pending.shift() || null;
      if (!this.active) break;
      const item = this.active;
      if (item.generation_epoch < this.currentEpoch) {
        this.active = null;
        frame -= 1;
        continue;
      }
      const index = Math.floor(item.position);
      if (index >= item.pcm16.length) {
        this.#emit("completed", item, audioTime);
        this.active = null;
        frame -= 1;
        continue;
      }
      output[frame] = item.pcm16[index] / 32768;
      item.position += item.sample_rate / this.outputSampleRate;
    }
    if (this.active) this.#emit("progress", this.active, audioTime);
    if (this.active && Math.floor(this.active.position) >= this.active.pcm16.length) {
      this.#emit("completed", this.active, audioTime);
      this.active = null;
    }
  }

  #emit(type, item, audioTime) {
    this.onEvent({
      type,
      response_id: item.response_id,
      generation_epoch: item.generation_epoch,
      segment_id: item.segment_id,
      sample_offset: Math.min(item.pcm16.length, Math.floor(item.position)),
      audio_time: audioTime,
    });
  }
}

if (typeof AudioWorkletProcessor !== "undefined") {
  class PcmPlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.renderer = new PlaybackRenderer({
        outputSampleRate: sampleRate,
        onEvent: (event) => this.port.postMessage(event),
      });
      this.port.onmessage = ({ data }) => {
        if (data.type === "enqueue") this.renderer.enqueue(data.item);
        if (data.type === "set_epoch") this.renderer.setEpoch(data.generation_epoch ?? data.epoch);
        if (data.type === "pause_response") this.renderer.pauseResponse(data.response_id, currentTime);
        if (data.type === "stop_response") this.renderer.stopResponse(data.response_id, currentTime);
      };
    }

    process(_inputs, outputs) {
      this.renderer.render(outputs[0][0], currentTime);
      return true;
    }
  }

  registerProcessor("pcm-playback-processor", PcmPlaybackProcessor);
}
