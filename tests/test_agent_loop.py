import asyncio
import unittest

from src.agent.loop import AgentLoop, AgentLoopError
from src.agent.models import AgentState, AgentStatus, ToolCallRequest
from src.application.voice_agent import VoiceAgent
from src.config import load_yaml
from src.controller.controller import ConversationController
from src.generation.manager import GenerationManager
from src.memory.manager import MemoryManager
from src.runtime.event_bus import EventBus
from src.tools.builtins import calculator_definition
from src.tools.registry import ToolRegistry
from src.tools.router import ToolRouter


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    def _loop(self, decisions, *, max_iterations=5, memory=None):
        index = 0

        async def decide(context):
            nonlocal index
            decision = decisions[index]
            index += 1
            return decision

        memory = memory or MemoryManager()
        memory.create_session("s1")
        registry = ToolRegistry()
        registry.register_tool(calculator_definition())
        router = ToolRouter(registry, memory_manager=memory, session_id="s1")
        loop = AgentLoop(
            decide,
            memory_manager=memory,
            tool_router=router,
            session_id="s1",
            max_iterations=max_iterations,
        )
        return loop, memory

    def test_agent_state_is_serializable(self):
        state = AgentState("s1", status=AgentStatus.THINKING, iteration=1)

        self.assertEqual(state.to_dict()["status"], "THINKING")
        self.assertEqual(state.to_dict()["session_id"], "s1")

    def test_tool_call_request_accepts_provider_neutral_mapping(self):
        request = ToolCallRequest.from_mapping(
            {"tool": "calculator", "arguments": {"expression": "2 + 2"}, "call_id": "c1"}
        )

        self.assertEqual(request.name, "calculator")
        self.assertEqual(request.call_id, "c1")

    async def test_tool_call_continues_and_returns_final_response(self):
        loop, memory = self._loop(
            [ToolCallRequest("calculator", {"expression": "25 * 4"}, "call-1"), "The answer is 100."]
        )

        result = await loop.run("What is 25 * 4?")

        self.assertEqual(result, "The answer is 100.")
        self.assertEqual(loop.state.status, AgentStatus.COMPLETED)
        self.assertEqual(loop.state.iteration, 2)
        self.assertEqual(
            loop.state_history,
            [
                AgentStatus.IDLE,
                AgentStatus.THINKING,
                AgentStatus.EXECUTING_TOOL,
                AgentStatus.OBSERVING,
                AgentStatus.THINKING,
                AgentStatus.COMPLETED,
            ],
        )
        contents = [message.content for message in memory.get_context("s1")]
        self.assertIn("What is 25 * 4?", contents)
        self.assertTrue(any(message.metadata.get("kind") == "tool_call" for message in memory.get_context("s1")))
        self.assertTrue(any(message.metadata.get("kind") == "observation" for message in memory.get_context("s1")))
        self.assertIn("The answer is 100.", contents)

    async def test_invalid_tool_result_is_observed_and_loop_can_finish(self):
        loop, memory = self._loop(
            [ToolCallRequest("missing", {}, "call-1"), "I could not run that tool."]
        )

        result = await loop.run("Use the missing tool")

        self.assertEqual(result, "I could not run that tool.")
        observations = [message for message in memory.get_context("s1") if message.metadata.get("kind") == "observation"]
        self.assertEqual(len(observations), 1)
        self.assertIn('"success": false', observations[0].content)

    async def test_max_iterations_prevents_infinite_tool_loop(self):
        loop, _ = self._loop(
            [ToolCallRequest("calculator", {"expression": "1 + 1"}, "call-1")] * 3,
            max_iterations=2,
        )

        with self.assertRaises(AgentLoopError):
            await loop.run("Keep calculating")
        self.assertEqual(loop.state.status, AgentStatus.FAILED)

    async def test_tool_execution_timeout_is_recoverable(self):
        async def slow(_expression):
            await asyncio.sleep(1)

        memory = MemoryManager()
        memory.create_session("s1")
        registry = ToolRegistry()
        from src.tools.models import ToolDefinition

        registry.register_tool(ToolDefinition("slow", "slow", {"type": "object"}, slow))
        router = ToolRouter(registry)

        decisions = [ToolCallRequest("slow", {"_expression": "x"}, "call-1"), "done"]
        index = 0

        async def decide(_context):
            nonlocal index
            value = decisions[index]
            index += 1
            return value

        loop = AgentLoop(decide, memory_manager=memory, tool_router=router, session_id="s1", tool_timeout=0.01)
        self.assertEqual(await loop.run("run slow"), "done")

    async def test_voice_agent_can_delegate_to_agent_loop(self):
        loop, _ = self._loop(["final"])
        voice_agent = VoiceAgent(
            EventBus(),
            ConversationController(),
            GenerationManager(),
            agent_loop=loop,
        )

        self.assertEqual(await voice_agent.run_agent("hello"), "final")

    def test_agent_config_defines_bounded_tool_loop(self):
        config = load_yaml("configs/agent.yaml")

        self.assertEqual(config["agent"]["max_iterations"], 5)
        self.assertTrue(config["agent"]["enable_tools"])


if __name__ == "__main__":
    unittest.main()
