import json
import unittest

from src.agent.loop import AgentLoop
from src.agent.models import ToolCallRequest
from src.memory.manager import MemoryManager
from src.tools.builtins import calculator_definition
from src.tools.parser import ToolCallParser
from src.tools.registry import ToolRegistry
from src.tools.router import ToolRouter
from src.config import load_yaml


class NativeToolCallingTests(unittest.IsolatedAsyncioTestCase):
    def test_parser_reads_nested_json_tool_call(self):
        parser = ToolCallParser()

        request = parser.parse(
            json.dumps(
                {
                    "tool_call": {
                        "name": "calculator",
                        "arguments": {"expression": "10 + 20"},
                    }
                }
            )
        )

        self.assertIsInstance(request, ToolCallRequest)
        self.assertTrue(request.call_id)
        self.assertEqual(request.tool_name, "calculator")
        self.assertEqual(request.arguments["expression"], "10 + 20")
        self.assertEqual(request.to_dict()["tool_name"], "calculator")

    def test_parser_accepts_structured_mapping_and_plain_text(self):
        parser = ToolCallParser()

        request = parser.parse({"tool": "calculator", "arguments": {"expression": "2 + 2"}})

        self.assertEqual(request.tool_name, "calculator")
        self.assertIsNone(parser.parse("ordinary final answer"))

    def test_parser_rejects_malformed_json(self):
        with self.assertRaises(ValueError):
            ToolCallParser().parse('{"tool_call":')

    def test_parser_reports_unknown_tool_when_registry_is_supplied(self):
        registry = ToolRegistry([calculator_definition()])

        with self.assertRaises(ValueError):
            ToolCallParser(registry).parse(
                {"tool_call": {"name": "missing", "arguments": {}}}
            )

    async def test_agent_loop_parses_native_json_and_continues(self):
        memory = MemoryManager()
        memory.create_session("s1")
        registry = ToolRegistry([calculator_definition()])
        router = ToolRouter(registry, memory_manager=memory, session_id="s1")
        decisions = [
            json.dumps(
                {
                    "tool_call": {
                        "name": "calculator",
                        "arguments": {"expression": "25 * 4"},
                    }
                }
            ),
            "The answer is 100.",
        ]
        calls = []

        async def generate(context):
            calls.append(context)
            return decisions.pop(0)

        result = await AgentLoop(
            generate,
            memory_manager=memory,
            tool_router=router,
            session_id="s1",
        ).run("What is 25 * 4?")

        self.assertEqual(result, "The answer is 100.")
        self.assertEqual(len(calls), 2)
        self.assertTrue(any(message.metadata.get("kind") == "tool_call" for message in memory.get_context("s1")))
        self.assertTrue(any(message.metadata.get("kind") == "observation" for message in memory.get_context("s1")))

    async def test_existing_injected_tool_call_remains_compatible(self):
        memory = MemoryManager()
        memory.create_session("s1")
        router = ToolRouter(
            ToolRegistry([calculator_definition()]),
            memory_manager=memory,
            session_id="s1",
        )
        decisions = [ToolCallRequest("calculator", {"expression": "1 + 1"}, "legacy"), "2"]

        async def generate(_context):
            return decisions.pop(0)

        self.assertEqual(
            await AgentLoop(
                generate,
                memory_manager=memory,
                tool_router=router,
                session_id="s1",
            ).run("calculate"),
            "2",
        )

    def test_agent_config_enables_native_tools(self):
        config = load_yaml("configs/agent.yaml")

        self.assertTrue(config["agent"]["enable_native_tools"])


if __name__ == "__main__":
    unittest.main()
