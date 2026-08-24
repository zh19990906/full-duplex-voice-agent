# Full Duplex Voice Agent Research Console

This is a dependency-free browser client for the existing API and WebSocket
service. It is intentionally kept separate from the Python backend and does
not implement sessions, memory, tools, ASR, LLM, or TTS logic.

## Start locally

For a quick static preview, open `frontend/index.html` directly in a browser.
If browser module or microphone permissions require HTTP, serve the repository
with the standard library:

```bash
python -m http.server 8000 --directory frontend
```

Then visit <http://localhost:8000>.

## Configure the API

The default configuration uses the current browser origin for HTTP and the
current host for WebSocket connections. To point the console at another
development service, define `window.VOICE_AGENT_CONFIG` before loading
`src/app.js`, for example:

```html
<script>
  window.VOICE_AGENT_CONFIG = {
    api: "http://localhost:8000",
    websocket: "ws://localhost:8000"
  };
</script>
```

The browser forwards microphone chunks through the existing audio boundary;
it does not inspect or transform their codec. Events from the backend are
shown in the conversation and timeline panels for research and debugging.
