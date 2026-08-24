import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.timeline import BenchmarkTimeline
from experiments.config import ExperimentConfig, load_experiment_config
from experiments.models import ExperimentResult
from experiments.reports import ExperimentReport
from experiments.runner import ExperimentRunner
from experiments.scenarios import ExperimentScenario


class FakeAgent:
    def __init__(self):
        self.state = type("State", (), {"iteration": 1})()
        self.tool_calls = 0
        self.interrupted = False

    async def run(self, text):
        if "calculate" in text:
            self.tool_calls += 1
        return f"response: {text}"

    async def interrupt(self):
        self.interrupted = True


class ExperimentFrameworkTests(unittest.IsolatedAsyncioTestCase):
    def test_config_loads_and_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiments.yaml"
            path.write_text(
                """experiments:\n  smoke:\n    name: Smoke Test\n    description: Basic conversation\n    profile: dev\n    scenario:\n      type: conversation\n    metrics: latency,agent\n""",
                encoding="utf-8",
            )

            config = load_experiment_config(path, "smoke")

        self.assertEqual(config.experiment_id, "smoke")
        self.assertEqual(config.scenario, "conversation")
        self.assertEqual(config.metrics, ("latency", "agent"))

    def test_repository_experiment_config_is_loadable(self):
        config = load_experiment_config("configs/experiments.yaml", "interrupt_test")

        self.assertEqual(config.profile, "hardware_benchmark")
        self.assertEqual(config.scenario, "interruption")

    def test_config_rejects_unknown_scenario(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(
                experiment_id="bad",
                name="Bad",
                description="",
                profile="dev",
                scenario="unsupported",
                parameters={},
                metrics=(),
            )

    async def test_runner_executes_conversation_and_collects_metrics(self):
        agent = FakeAgent()
        values = iter(range(20))
        runner = ExperimentRunner(clock=lambda: float(next(values)))
        config = ExperimentConfig(
            "conversation_test", "Conversation", "", "dev", "conversation", {"input": "hello"}, ("latency",)
        )

        report = await runner.run(config, agent)

        self.assertTrue(report.success)
        self.assertEqual(report.scenario, "conversation")
        self.assertIn("startup_latency_ms", report.metrics)
        self.assertEqual(report.agent["iterations"], 1)

    async def test_interruption_scenario_calls_agent_interrupt(self):
        agent = FakeAgent()
        runner = ExperimentRunner(clock=iter(range(20)).__next__)
        config = ExperimentConfig("interrupt", "Interrupt", "", "dev", "interruption", {}, ("cancellation",))

        report = await runner.run(config, agent)

        self.assertTrue(agent.interrupted)
        self.assertTrue(report.success)
        self.assertIn("interrupt_latency_ms", report.metrics)
        self.assertIn("cancellation_latency_ms", report.metrics)

    async def test_tool_use_scenario_collects_agent_tool_metrics(self):
        agent = FakeAgent()
        runner = ExperimentRunner(clock=iter(range(20)).__next__)
        config = ExperimentConfig("tools", "Tools", "", "dev", "tool_use", {"input": "calculate 2+2"}, ("agent",))

        report = await runner.run(config, agent)

        self.assertTrue(report.success)
        self.assertEqual(agent.tool_calls, 1)
        self.assertEqual(report.agent["tool_calls"], 1)

    def test_report_has_stable_json_structure_and_text(self):
        report = ExperimentReport(
            experiment_id="exp",
            name="Example",
            scenario="conversation",
            success=True,
            metrics={"first_token_latency_ms": 10.0},
            agent={"iterations": 1},
        )

        document = json.loads(report.to_json())

        self.assertEqual(
            list(document), ["agent", "experiment_id", "metrics", "name", "scenario", "success"]
        )
        self.assertIn("Experiment Report", report.to_text())

    async def test_same_config_produces_same_report_structure(self):
        config = ExperimentConfig("same", "Same", "", "dev", "conversation", {"input": "hello"}, ())
        first = await ExperimentRunner(clock=iter(range(20)).__next__).run(config, FakeAgent())
        second = await ExperimentRunner(clock=iter(range(20)).__next__).run(config, FakeAgent())

        self.assertEqual(list(first.to_dict()), list(second.to_dict()))
        self.assertEqual(list(first.metrics), list(second.metrics))

    def test_scenario_model_wraps_existing_timeline_boundary(self):
        scenario = ExperimentScenario("conversation", {"input": "hello"})
        timeline = BenchmarkTimeline()

        scenario.record_start(timeline, timestamp=1.0)
        scenario.record_end(timeline, timestamp=2.0)

        self.assertEqual(timeline.names(), ["experiment_started", "experiment_completed"])


if __name__ == "__main__":
    unittest.main()
