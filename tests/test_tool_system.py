import asyncio
import json
import unittest
from pathlib import Path

from src.application.voice_agent import VoiceAgent
from src.controller.controller import ConversationController
from src.generation.manager import GenerationManager
from src.memory.manager import MemoryManager
from src.runtime.event_bus import EventBus
from src.tools.builtins import calculator_definition
from src.tools.executor import ToolExecutor
from src.tools.models import ToolDefinition, ToolResult
from src.tools.registry import ToolRegistry
from src.tools.router import ToolRouter


class ToolSystemTests(unittest.IsolatedAsyncioTestCase):
    def test_registry_register_lookup_list_and_unregister(self):
        registry = ToolRegistry()
        definition = ToolDefinition("echo", "Echo input", {"type": "object"}, lambda value: value)

        registry.register_tool(definition)

        self.assertIs(registry.get_tool("echo"), definition)
        self.assertEqual([tool.name for tool in registry.list_tools()], ["echo"])
        self.assertEqual(registry.discover_tools()[0]["name"], "echo")
        registry.unregister_tool("echo")
        with self.assertRaises(KeyError):
            registry.get_tool("echo")

    async def test_router_executes_valid_tool_call(self):
        registry = ToolRegistry()
        registry.register_tool(calculator_definition())
        router = ToolRouter(registry)

        result = await router.route({"name": "calculator", "arguments": {"expression": "25 * 4"}})

        self.assertEqual(result, ToolResult(success=True, output="100", tool_name="calculator"))

    async def test_router_reports_unknown_and_invalid_calls(self):
        router = ToolRouter(ToolRegistry())

        unknown = await router.route({"name": "missing", "arguments": {}})

        self.assertFalse(unknown.success)
        self.assertIn("unknown tool", unknown.error)

        registry = ToolRegistry()
        registry.register_tool(calculator_definition())
        invalid = await ToolRouter(registry).route(
            {"name": "calculator", "arguments": {"unexpected": "value"}}
        )
        self.assertFalse(invalid.success)
        self.assertIn("unexpected", invalid.error)

    async def test_executor_isolates_failures_and_enforces_timeout(self):
        async def failure():
            raise RuntimeError("broken")

        async def slow():
            await asyncio.sleep(1)

        executor = ToolExecutor(default_timeout=0.01)
        failure_result = await executor.execute(
            ToolDefinition("failure", "fails", {"type": "object"}, failure), {}
        )
        timeout_result = await executor.execute(
            ToolDefinition("slow", "slow", {"type": "object"}, slow), {}
        )

        self.assertEqual(failure_result.success, False)
        self.assertIn("broken", failure_result.error)
        self.assertEqual(timeout_result.success, False)
        self.assertIn("timed out", timeout_result.error)

    async def test_tool_call_and_result_are_stored_in_memory(self):
        memory = MemoryManager()
        memory.create_session("s1")
        registry = ToolRegistry()
        registry.register_tool(calculator_definition())
        router = ToolRouter(registry, memory_manager=memory, session_id="s1")

        await router.route({"name": "calculator", "arguments": {"expression": "2 + 3"}})

        messages = memory.get_context("s1")
        self.assertEqual([message.role for message in messages], ["system", "system"])
        self.assertEqual(messages[0].metadata["kind"], "tool_call")
        self.assertEqual(messages[1].metadata["kind"], "tool_result")
        self.assertEqual(json.loads(messages[1].content)["output"], "5")

    async def test_voice_agent_executes_tool_flow(self):
        memory = MemoryManager()
        memory.create_session("s1")
        registry = ToolRegistry()
        registry.register_tool(calculator_definition())
        agent = VoiceAgent(
            EventBus(),
            ConversationController(),
            GenerationManager(),
            memory_manager=memory,
            memory_session_id="s1",
            tool_router=ToolRouter(registry, memory_manager=memory, session_id="s1"),
        )

        result = await agent.execute_tool(
            {"name": "calculator", "arguments": {"expression": "10 / 2"}}
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output, "5")
        self.assertEqual(len(memory.get_context("s1")), 2)

    def test_tool_config_is_dependency_free(self):
        config = Path("configs/tools.yaml").read_text(encoding="utf-8")
        self.assertIn("enabled:", config)
        self.assertIn("- calculator", config)


if __name__ == "__main__":
    unittest.main()
