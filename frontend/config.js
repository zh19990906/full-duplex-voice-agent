const defaultWebSocketUrl = () => {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
};

const defaults = {
  api: window.location.origin,
  websocket: defaultWebSocketUrl(),
};

// Deployments may replace this object before app.js loads without rebuilding the UI.
export const appConfig = Object.freeze({
  ...defaults,
  ...(window.VOICE_AGENT_CONFIG || {}),
});
