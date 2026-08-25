import { appConfig } from "../config.js";
import { createApiClient } from "./api.js";
import { PcmMicrophoneInput, BrowserAudioOutput } from "./audio.js";
import { EventTimeline } from "./timeline.js";
import { AgentWebSocket, EVENT_TYPES } from "./websocket.js";

const byId = (id) => document.getElementById(id);

export class StreamingAssistantMessage {
  constructor(createMessage) {
    this.createMessage = createMessage;
    this.item = null;
    this.text = "";
  }

  begin() {
    this.item = null;
    this.text = "";
  }

  append(chunk) {
    if (!chunk) return;
    this.item ||= this.createMessage("assistant", "");
    this.text += chunk;
    this.item.textContent = this.text;
  }

  finalize(text) {
    if (text) {
      this.item ||= this.createMessage("assistant", "");
      this.item.textContent = text;
    }
    this.item = null;
    this.text = "";
  }
}

export function bootResearchConsole(
  documentRef = document,
  { microphoneFactory = (options) => new PcmMicrophoneInput(options) } = {},
) {
  const api = createApiClient(appConfig);
  const timeline = new EventTimeline(byId("timeline"));
  const audioOutput = new BrowserAudioOutput();
  const state = { sessionId: null, socket: null, microphone: null, microphoneAction: Promise.resolve() };

  const setStatus = (text) => { byId("session-status").textContent = text; };
  const queueMicrophoneAction = (action) => {
    const pending = state.microphoneAction.then(action, action);
    state.microphoneAction = pending.catch(() => {});
    return pending;
  };
  const stopActiveMicrophone = async () => {
    const microphone = state.microphone;
    state.microphone = null;
    await microphone?.stop();
  };
  const appendChat = (role, text) => {
    const item = documentRef.createElement("div");
    item.className = `chat-message ${role}`;
    item.textContent = text;
    byId("messages").append(item);
    return item;
  };
  const assistantMessage = new StreamingAssistantMessage(appendChat);

  const connect = (sessionId) => {
    state.socket = new AgentWebSocket(appConfig.websocket);
    EVENT_TYPES.forEach((type) => state.socket.addEventListener(type, (event) => {
      const envelope = event.detail;
      timeline.add(envelope);
      if (type === "token") assistantMessage.append(envelope.payload?.text || "");
      if (type === "transcript") appendChat(type, envelope.payload?.text || "");
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
    await queueMicrophoneAction(stopActiveMicrophone);
    await api.deleteSession(state.sessionId);
    state.socket?.close();
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
    assistantMessage.begin();
    const response = await api.sendMessage(state.sessionId, message);
    assistantMessage.finalize(response.response);
  });

  byId("start-mic").addEventListener("click", async () => {
    if (!state.socket) return setStatus("请先创建会话");
    const startMicrophone = async () => {
      await stopActiveMicrophone();
      const microphone = microphoneFactory({
        onChunk: (buffer) => state.socket.sendAudio(buffer),
        onStateChange: (value) => { byId("mic-status").textContent = value; },
      });
      state.microphone = microphone;
      try {
        await microphone.start();
      } catch (error) {
        if (state.microphone === microphone) state.microphone = null;
        throw error;
      }
    };
    await queueMicrophoneAction(startMicrophone);
  });
  byId("stop-mic").addEventListener("click", () => queueMicrophoneAction(stopActiveMicrophone));

  return { state, api, timeline };
}

if (typeof document !== "undefined") bootResearchConsole();
