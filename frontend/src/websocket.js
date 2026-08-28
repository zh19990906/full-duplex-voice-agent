export const EVENT_TYPES = Object.freeze([
  "transcript",
  "translation",
  "token",
  "audio",
  "session_snapshot",
  "tool_call",
  "tool_result",
  "agent_state",
  "duck",
  "restore",
  "pause_response",
  "policy_uncertain",
  "request_clarification",
  "request_repeat",
  "resume_response",
  "stop_response",
  "set_epoch",
  "playback_failed",
  "llm_failed",
  "tts_failed",
  "worker_terminal",
]);

const normalizeEventName = (name) => {
  const normalized = String(name || "message").toLowerCase();
  return normalized === "audio_chunk" ? "audio" : normalized;
};

export class AgentWebSocket extends EventTarget {
  constructor(url, {
    maxReconnectAttempts = 5,
    reconnectDelayMs = 250,
    maxReconnectDelayMs = 4000,
    schedule = (callback, delay) => setTimeout(callback, delay),
    cancelSchedule = (handle) => clearTimeout(handle),
  } = {}) {
    super();
    this.url = url;
    this.socket = null;
    this.sessionId = null;
    this.maxReconnectAttempts = maxReconnectAttempts;
    this.reconnectDelayMs = reconnectDelayMs;
    this.maxReconnectDelayMs = maxReconnectDelayMs;
    this.schedule = schedule;
    this.cancelSchedule = cancelSchedule;
    this.reconnectAttempts = 0;
    this.reconnectHandle = null;
    this.closedExplicitly = false;
  }

  connect(sessionId) {
    this.sessionId = sessionId;
    this.closedExplicitly = false;
    this.reconnectAttempts = 0;
    this.#cancelReconnect();
    return this.#open();
  }

  #open() {
    const separator = this.url.endsWith("/") ? "" : "/";
    const sessionId = this.sessionId;
    const socket = new WebSocket(`${this.url}${separator}${encodeURIComponent(sessionId)}`);
    this.socket = socket;
    socket.addEventListener("message", (message) => {
      const event = typeof message.data === "string" ? JSON.parse(message.data) : message.data;
      const type = normalizeEventName(event.event || event.type || "message");
      this.dispatchEvent(new CustomEvent(type, { detail: event }));
    });
    socket.addEventListener("open", () => {
      if (this.socket !== socket) return;
      this.reconnectAttempts = 0;
      this.dispatchEvent(new Event("open"));
    });
    socket.addEventListener("close", () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.dispatchEvent(new Event("close"));
      this.#scheduleReconnect();
    });
    socket.addEventListener("error", (error) => this.dispatchEvent(new CustomEvent("error", { detail: error })));
    return socket;
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
    this.closedExplicitly = true;
    this.#cancelReconnect();
    const socket = this.socket;
    this.socket = null;
    socket?.close();
  }

  #scheduleReconnect() {
    if (this.closedExplicitly || !this.sessionId) return;
    if (this.reconnectAttempts >= this.maxReconnectAttempts) return;
    const delay = Math.min(
      this.reconnectDelayMs * (2 ** this.reconnectAttempts),
      this.maxReconnectDelayMs,
    );
    this.reconnectAttempts += 1;
    this.reconnectHandle = this.schedule(() => {
      this.reconnectHandle = null;
      if (!this.closedExplicitly) this.#open();
    }, delay);
    this.dispatchEvent(new CustomEvent("reconnecting", {
      detail: { attempt: this.reconnectAttempts, delay },
    }));
  }

  #cancelReconnect() {
    if (this.reconnectHandle === null) return;
    this.cancelSchedule(this.reconnectHandle);
    this.reconnectHandle = null;
  }

  #sendJson(message) {
    const openState = typeof WebSocket === "undefined" ? 1 : WebSocket.OPEN;
    if (!this.socket || this.socket.readyState !== openState) return false;
    this.socket.send(JSON.stringify(message));
    return true;
  }
}
