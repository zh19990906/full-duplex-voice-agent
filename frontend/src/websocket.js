export const EVENT_TYPES = Object.freeze([
  "transcript",
  "token",
  "audio",
  "tool_call",
  "tool_result",
  "agent_state",
]);

const normalizeEventName = (name) => (name === "audio_chunk" ? "audio" : name);

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
    this.socket?.send(JSON.stringify({ type: "text", text }));
  }

  sendAudio(chunk) {
    this.socket?.send(chunk);
  }

  close() {
    this.socket?.close();
    this.socket = null;
  }
}
