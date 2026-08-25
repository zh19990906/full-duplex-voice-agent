import { appConfig } from "../config.js";
import { createApiClient } from "./api.js";
import { MicrophoneInput, BrowserAudioOutput } from "./audio.js";
import { EventTimeline } from "./timeline.js";
import { AgentWebSocket, EVENT_TYPES } from "./websocket.js";

const byId = (id) => document.getElementById(id);

export function bootResearchConsole(documentRef = document) {
  const api = createApiClient(appConfig);
  const timeline = new EventTimeline(byId("timeline"));
  const audioOutput = new BrowserAudioOutput();
  const state = { sessionId: null, socket: null, microphone: null };

  const setStatus = (text) => { byId("session-status").textContent = text; };
  const appendChat = (role, text) => {
    const item = documentRef.createElement("div");
    item.className = `chat-message ${role}`;
    item.textContent = text;
    byId("messages").append(item);
  };

  const connect = (sessionId) => {
    state.socket = new AgentWebSocket(appConfig.websocket);
    EVENT_TYPES.forEach((type) => state.socket.addEventListener(type, (event) => {
      const envelope = event.detail;
      timeline.add(envelope);
      if (type === "token" || type === "transcript") appendChat(type, envelope.payload?.text || "");
      if (type === "audio") audioOutput.playBase64(
        envelope.payload?.audio_data,
        envelope.payload?.sample_rate || 24000,
      ).catch(console.error);
      if (type === "agent_state") byId("agent-state").textContent = JSON.stringify(envelope.payload, null, 2);
      if (type === "tool_call" || type === "tool_result") byId("tool-output").textContent = JSON.stringify(envelope.payload, null, 2);
    }));
    state.socket.connect(sessionId);
  };

  byId("create-session").addEventListener("click", async () => {
    await audioOutput.unlock();
    const session = await api.createSession();
    state.sessionId = session.session_id;
    byId("session-id").textContent = state.sessionId;
    setStatus("active");
    connect(state.sessionId);
  });

  byId("close-session").addEventListener("click", async () => {
    if (!state.sessionId) return;
    await api.deleteSession(state.sessionId);
    state.socket?.close();
    state.microphone?.stop();
    audioOutput.stop();
    setStatus("closed");
    state.sessionId = null;
  });

  byId("message-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    await audioOutput.unlock();
    if (!state.sessionId) return setStatus("请先创建会话");
    const input = byId("message");
    const message = input.value.trim();
    if (!message) return;
    appendChat("user", message);
    input.value = "";
    const response = await api.sendMessage(state.sessionId, message);
    if (response.response) appendChat("assistant", response.response);
  });

  byId("start-mic").addEventListener("click", async () => {
    if (!state.socket) return setStatus("请先创建会话");
    state.microphone = new MicrophoneInput({
      onChunk: (chunk) => chunk.arrayBuffer().then((buffer) => state.socket.sendAudio(buffer)),
      onStateChange: (value) => { byId("mic-status").textContent = value; },
    });
    await state.microphone.start();
  });
  byId("stop-mic").addEventListener("click", () => state.microphone?.stop());

  return { state, api, timeline };
}

if (typeof document !== "undefined") bootResearchConsole();
