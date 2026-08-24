import asyncio
import json
import unittest

from experiments.comparison import ExperimentComparison
from experiments.config import ExperimentConfig
from experiments.profiles.loader import ExperimentProfileBundle
from experiments.profiles.models import ExperimentVariant
from experiments.reports import ComparisonReport
from experiments.runner import ExperimentRunner


class ProfileAgent:
    def __init__(self, response="ok"):
        self.response = response
        self.state = type("State", (), {"iteration": 1})()
        self.tool_calls = 0

    async def run(self, _text):
        return self.response


class ExperimentProfilesTests(unittest.IsolatedAsyncioTestCase):
    def test_variant_loads_from_repository_config(self):
        bundle = ExperimentProfileBundle.from_files(
            "configs/experiment_profiles.yaml",
            "experiments/profiles/models.yaml",
            "experiments/profiles/prompts.yaml",
            "experiments/profiles/strategies.yaml",
        )

        variant = bundle.variant("baseline")

        self.assertEqual(variant.variant_id, "baseline")
        self.assertEqual(variant.model_profile, "baseline")
        self.assertEqual(variant.prompt_profile, "v1")
        self.assertEqual(variant.memory_profile, "recent_20")
        self.assertEqual(variant.tool_profile, "calculator")

    def test_model_profile_reuses_deployment_profiles(self):
        bundle = ExperimentProfileBundle.from_files(
            "configs/experiment_profiles.yaml",
            "experiments/profiles/models.yaml",
            "experiments/profiles/prompts.yaml",
            "experiments/profiles/strategies.yaml",
        )

        model = bundle.model("baseline")

        self.assertEqual(model.llm.provider, "llama_cpp")
        self.assertEqual(model.asr.provider, "whisper")
        self.assertEqual(model.tts.provider, "cosyvoice")

    def test_prompt_memory_and_tool_profiles_are_selectable(self):
        bundle = ExperimentProfileBundle.from_files(
            "configs/experiment_profiles.yaml",
            "experiments/profiles/models.yaml",
            "experiments/profiles/prompts.yaml",
            "experiments/profiles/strategies.yaml",
        )

        self.assertEqual(bundle.prompt("v2").system_prompt, "You are a concise assistant.")
        self.assertEqual(bundle.memory("recent_50").max_messages, 50)
        self.assertEqual(bundle.tools("disabled").enabled_tools, ())
        self.assertEqual(bundle.tools("calculator").enabled_tools, ("calculator",))

    async def test_runner_injects_selected_variant_context(self):
        bundle = ExperimentProfileBundle.from_files(
            "configs/experiment_profiles.yaml",
            "experiments/profiles/models.yaml",
            "experiments/profiles/prompts.yaml",
            "experiments/profiles/strategies.yaml",
        )
        contexts = []

        def factory(context):
            contexts.append(context)
            return ProfileAgent()

        runner = ExperimentRunner(
            clock=iter(range(20)).__next__,
            agent_factory=factory,
            profile_bundle=bundle,
        )
        config = ExperimentConfig("exp", "Experiment", "", "dev", "conversation", {"input": "hi"}, ())

        report = await runner.run(config, variant="baseline")

        self.assertTrue(report.success)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0].variant.variant_id, "baseline")
        self.assertEqual(report.variant_id, "baseline")

    async def test_comparison_runs_variants_and_reports_each_result(self):
        runner = ExperimentRunner(clock=iter(range(40)).__next__)
        comparison = ExperimentComparison(runner)
        config = ExperimentConfig("exp", "Comparison", "", "dev", "conversation", {"input": "hi"}, ())
        bundle = ExperimentProfileBundle.from_files(
            "configs/experiment_profiles.yaml",
            "experiments/profiles/models.yaml",
            "experiments/profiles/prompts.yaml",
            "experiments/profiles/strategies.yaml",
        )

        report = await comparison.run(
            config,
            {
                "baseline": (bundle.variant("baseline"), ProfileAgent("one")),
                "concise": (bundle.variant("concise"), ProfileAgent("two")),
            },
        )

        self.assertEqual([item["variant_id"] for item in report.to_dict()["variants"]], ["baseline", "concise"])
        self.assertIn("Model Comparison", report.to_text())

    def test_comparison_report_is_json_serializable(self):
        report = ComparisonReport("exp", "Comparison", ())

        self.assertEqual(json.loads(report.to_json())["experiment_id"], "exp")

    async def test_existing_experiment_runner_call_remains_compatible(self):
        runner = ExperimentRunner(clock=iter(range(20)).__next__)
        config = ExperimentConfig("legacy", "Legacy", "", "dev", "conversation", {"input": "hi"}, ())

        report = await runner.run(config, ProfileAgent())

        self.assertTrue(report.success)
        self.assertIsNone(report.variant_id)


if __name__ == "__main__":
    unittest.main()
