import assert from "node:assert/strict";
import test from "node:test";

globalThis.window = {
  location: {
    protocol: "http:",
    host: "localhost:8001",
    origin: "http://localhost:8001",
  },
};

const { StreamingAssistantMessage } = await import("../frontend/src/app.js");

const createHarness = () => {
  const messages = [];
  const stream = new StreamingAssistantMessage((role, text) => {
    const item = { role, textContent: text };
    messages.push(item);
    return item;
  });
  return { messages, stream };
};

test("token chunks update one assistant message and final response reconciles it", () => {
  const { messages, stream } = createHarness();

  stream.begin();
  stream.append("你");
  stream.append("好");

  assert.equal(messages.length, 1);
  assert.deepEqual(messages[0], { role: "assistant", textContent: "你好" });

  stream.finalize("你好！");

  assert.equal(messages.length, 1);
  assert.equal(messages[0].textContent, "你好！");
});

test("complete response creates one assistant message when no tokens arrive", () => {
  const { messages, stream } = createHarness();

  stream.begin();
  stream.finalize("完整回复");

  assert.deepEqual(messages, [{ role: "assistant", textContent: "完整回复" }]);
});
