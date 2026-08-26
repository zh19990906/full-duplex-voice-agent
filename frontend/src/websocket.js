export const EVENT_TYPES = Object.freeze([
  "transcript",
  "token",
  "audio",
  "tool_call",
  "tool_result",
  "agent_state",
  "duck",
  "restore",
  "pause_response",
  "resume_response",
  "stop_response",
  "set_epoch",
]);

const normalizeEventName = (name) => {
  const normalized = String(name || "message").toLowerCase();
  return normalized === "audio_chunk" ? "audio" : normalized;
};

export class AgentWebSocket extends EventTarget {
  constructor(url) {
    super();
    this.url = url;
    this.socket = null;
  }

  connect(sessionId) {
    const separator = this.url.endsWith("/") ? "" : "/";
    this.socket = new WebSocket(`${this.url}${separator}${encodeURIComponent(sessionId)}`);
    this.socket.addEventListener("message", (message) => {
      const event = typeof message.data === "string" ? JSON.parse(message.data) : message.data;
      const type = normalizeEventName(event.event || event.type || "message");
      this.dispatchEvent(new CustomEvent(type, { detail: event }));
    });
    this.socket.addEventListener("open", () => this.dispatchEvent(new Event("open")));
    this.socket.addEventListener("close", () => this.dispatchEvent(new Event("close")));
    this.socket.addEventListener("error", (error) => this.dispatchEvent(new CustomEvent("error", { detail: error })));
    return this.socket;
  }

  sendText(text) {
    return this.#sendJson({ type: "text", text });
  }

  sendAudio(frame) {
    this.socket?.send(frame);
  }

  sendPlaybackAck(payload) {
    const acknowledgement = {
      response_id: payload.response_id,
      generation_epoch: payload.generation_epoch,
      segment_id: payload.segment_id,
      sample_offset: payload.sample_offset,
      audio_time: payload.audio_time,
    };
    if (payload.playback_attempt_id !== undefined) {
      acknowledgement.playback_attempt_id = payload.playback_attempt_id;
    }
    return this.#sendJson({ type: "playback_ack", payload: acknowledgement });
  }

  sendPlaybackControl(type, payload = {}) {
    return this.#sendJson({ type, payload });
  }

  close() {
    this.socket?.close();
    this.socket = null;
  }

  #sendJson(message) {
    const openState = typeof WebSocket === "undefined" ? 1 : WebSocket.OPEN;
    if (!this.socket || this.socket.readyState !== openState) return false;
    this.socket.send(JSON.stringify(message));
    return true;
  }
}
