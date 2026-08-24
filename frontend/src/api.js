import { appConfig } from "../config.js";

export function createApiClient(config = appConfig) {
  const request = async (method, path, body) => {
    const options = { method, headers: {} };
    if (body !== undefined) {
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(body);
    }
    const response = await fetch(`${config.api}${path}`, options);
    const payload = response.status === 204 ? null : await response.json();
    if (!response.ok) {
      throw new Error(payload?.error || `API request failed (${response.status})`);
    }
    return payload;
  };

  return {
    createSession: (metadata = {}) => request("POST", "/sessions", { metadata }),
    getSession: (sessionId) => request("GET", `/sessions/${encodeURIComponent(sessionId)}`),
    deleteSession: (sessionId) => request("DELETE", `/sessions/${encodeURIComponent(sessionId)}`),
    sendMessage: (sessionId, message) => request(
      "POST",
      `/sessions/${encodeURIComponent(sessionId)}/message`,
      { message },
    ),
  };
}
