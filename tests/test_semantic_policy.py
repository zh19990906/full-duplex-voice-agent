import asyncio
import json
import math
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

from src.adapters.llm.providers.qwen_policy import QwenPolicyProvider
from src.asr.stream import TranscriptChunk
from src.realtime.policy import (
    PolicyAction,
    PolicyDecision,
    PolicyRequest,
    SemanticPolicyEngine,
)
from src.realtime.session_state import ConversationMode, FloorState, ResponseState, SessionState
from src.realtime.speech_fusion import SpeechCandidateEvent


ROOT = Path(__file__).resolve().parents[1]


def request_for(
    text="嗯",
    *,
    assistant_act=None,
    is_final=True,
    replaces_committed=False,
    committed_text=None,
):
    transcript = TranscriptChunk(
        "chunk-1",
        text,
        2.0,
        is_final,
        revision_id=3,
        unstable_text="" if is_final else text,
        committed_text=committed_text,
        replaces_committed=replaces_committed,
    )
    candidate = SpeechCandidateEvent(
        event="USER_BACKCHANNEL_CANDIDATE",
        event_id="candidate-1",
        timestamp=2.1,
        source="x2_turn",
        payload={"label": "backchannel", "confidence": 0.91, "evidence": {"frames": 3}},
    )
    return PolicyRequest(
        state=SessionState(
            mode=ConversationMode.CHAT,
            floor=FloorState.OVERLAP,
            response=ResponseState.DUCKED,
            assistant_act=assistant_act,
        ),
        assistant_last_text="你确认目的地是上海吗？",
        unplayed_text_summary="后续行程建议",
        user_transcript=transcript,
        candidate=candidate,
        task_checkpoint={"city": "上海", "step": 2},
        source_language="zh",
        target_language="en",
    )


class PolicySchemaTests(unittest.TestCase):
    def test_policy_action_has_exact_closed_set(self):
        self.assertEqual(
            [item.value for item in PolicyAction],
            [
                "BACKCHANNEL",
                "ANSWER",
                "PAUSE",
                "RESUME",
                "REVISE",
                "NEW_REQUEST",
                "MODE_SWITCH",
                "UNCERTAIN",
            ],
        )

    def test_policy_rejects_unknown_action(self):
        with self.assertRaises(ValueError):
            PolicyDecision.from_mapping(
                {"action": "GUESS", "confidence": 1.0, "rationale": "not in schema"}
            )

    def test_mode_switch_requires_known_intent_and_target_for_interpretation(self):
        invalid = [
            {"action": "MODE_SWITCH", "confidence": 0.9, "rationale": "missing"},
            {
                "action": "MODE_SWITCH",
                "confidence": 0.9,
                "rationale": "one shot",
                "intent": "one_shot_translation",
                "target_language": "English",
            },
            {
                "action": "MODE_SWITCH",
                "confidence": 0.9,
                "rationale": "no target",
                "intent": "continuous_interpretation",
            },
            {
                "action": "MODE_SWITCH",
                "confidence": 0.9,
                "rationale": "blank target",
                "intent": "continuous_interpretation",
                "target_language": "   ",
            },
        ]
        for mapping in invalid:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                PolicyDecision.from_mapping(mapping)

        enter = PolicyDecision(
            PolicyAction.MODE_SWITCH,
            0.9,
            "enter",
            intent="continuous_interpretation",
            source_language=None,
            target_language="English",
        )
        leave = PolicyDecision(
            PolicyAction.MODE_SWITCH,
            0.9,
            "leave",
            intent="chat",
        )
        self.assertEqual(enter.intent, "continuous_interpretation")
        self.assertEqual(leave.intent, "chat")

    def test_policy_schema_rejects_unknown_missing_and_wrong_fields(self):
        invalid = [
            {"action": "ANSWER", "confidence": 0.9, "rationale": "ok", "extra": 1},
            {"action": "ANSWER", "confidence": 0.9},
            {"action": "ANSWER", "confidence": "0.9", "rationale": "ok"},
            {"action": "ANSWER", "confidence": True, "rationale": "ok"},
            {"action": "ANSWER", "confidence": math.nan, "rationale": "ok"},
            {"action": "ANSWER", "confidence": 1.1, "rationale": "ok"},
            {"action": "ANSWER", "confidence": 0.9, "rationale": 3},
            {"action": "ANSWER", "confidence": 0.9, "rationale": "ok", "intent": 3},
        ]
        for mapping in invalid:
            with self.subTest(mapping=mapping), self.assertRaises((TypeError, ValueError)):
                PolicyDecision.from_mapping(mapping)

    def test_policy_decision_is_frozen_and_transport_safe(self):
        decision = PolicyDecision.from_mapping(
            {
                "action": "MODE_SWITCH",
                "confidence": 0.97,
                "rationale": "continuous interpretation requested",
                "intent": "continuous_interpretation",
                "source_language": "Chinese",
                "target_language": "English",
            }
        )
        self.assertEqual(decision.to_dict()["action"], "MODE_SWITCH")
        with self.assertRaises(Exception):
            decision.confidence = 0.1

    def test_policy_request_serializes_all_context_deterministically(self):
        request = request_for(
            "不对，我说的是北京，不是上海",
            replaces_committed=True,
            committed_text="不对，我说的是北京，不是上海",
        )

        first = request.to_prompt_json()
        second = request.to_prompt_json()
        payload = json.loads(first)

        self.assertEqual(first, second)
        self.assertEqual(payload["state"], {
            "mode": "CHAT",
            "floor": "OVERLAP",
            "response": "DUCKED",
            "assistant_act": None,
        })
        self.assertEqual(payload["user_transcript"]["revision_id"], 3)
        self.assertTrue(payload["user_transcript"]["replaces_committed"])
        self.assertEqual(payload["candidate"]["label"], "backchannel")
        self.assertEqual(payload["candidate"]["confidence"], 0.91)
        self.assertEqual(payload["candidate"]["evidence"], {"frames": 3})
        self.assertEqual(payload["task_checkpoint"], {"city": "上海", "step": 2})
        self.assertEqual(
            payload["policy_contract"]["allowed_actions"],
            [item.value for item in PolicyAction],
        )
        self.assertTrue(payload["policy_contract"]["strict_json_only"])
        self.assertEqual(
            payload["policy_contract"]["mode_switch_intents"],
            ["chat", "continuous_interpretation"],
        )
        self.assertEqual(
            payload["policy_contract"]["continuous_interpretation_requires"],
            ["target_language"],
        )
        self.assertNotIn("translation_keywords", payload)

    def test_policy_request_snapshots_mutable_state_and_candidate_context(self):
        original_state = SessionState(
            ConversationMode.CHAT,
            FloorState.OVERLAP,
            ResponseState.DUCKED,
            "ASKING",
        )
        original_payload = {"label": "backchannel", "confidence": 0.9, "evidence": {"frames": 2}}
        request = PolicyRequest(
            state=original_state,
            candidate=SpeechCandidateEvent(
                "USER_BACKCHANNEL_CANDIDATE", "event-1", 1.0, "x2", original_payload
            ),
        )
        before = request.to_prompt_json()

        original_state.mode = ConversationMode.INTERPRETATION
        original_payload["label"] = "turn_end"

        self.assertEqual(request.to_prompt_json(), before)


class SemanticPolicyEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_semantics_are_supplied_by_fake_llm_not_keywords(self):
        cases = [
            ("嗯，对，继续", None, "BACKCHANNEL"),
            ("对", "ASKING", "ANSWER"),
            ("对，不过我说的是北京", None, "REVISE"),
            ("嗯……等一下", None, "PAUSE"),
            ("把这句话翻译成英文", None, "NEW_REQUEST"),
            ("接下来一直把中文实时翻译成英文", None, "MODE_SWITCH"),
        ]
        decisions = {
            text: {
                "action": action,
                "confidence": 0.96,
                "rationale": "fake semantic decision",
                **(
                    {
                        "intent": "continuous_interpretation",
                        "source_language": "Chinese",
                        "target_language": "English",
                    }
                    if action == "MODE_SWITCH"
                    else {}
                ),
            }
            for text, _, action in cases
        }

        class FakePolicyRuntime:
            async def generate(self, prompt):
                text = json.loads(prompt)["user_transcript"]["text"]
                return json.dumps(decisions[text], ensure_ascii=False)

        engine = SemanticPolicyEngine(FakePolicyRuntime())
        for text, assistant_act, expected in cases:
            with self.subTest(text=text):
                decision = await engine.decide(request_for(text, assistant_act=assistant_act))
                self.assertEqual(decision.action.value, expected)
        self.assertEqual(
            (await engine.decide(request_for("把这句话翻译成英文"))).action,
            PolicyAction.NEW_REQUEST,
        )

    async def test_failures_and_low_confidence_become_safe_uncertain(self):
        outputs = [
            "```json\n{}\n```",
            '{"action":"ANSWER","confidence":0.9,"rationale":"ok"} trailing',
            '{"action":"ANSWER","confidence":0.4,"rationale":"weak"}',
            '{"action":"GUESS","confidence":0.9,"rationale":"bad"}',
        ]
        for output in outputs:
            class Runtime:
                async def generate(self, prompt, value=output):
                    return value

            with self.subTest(output=output):
                decision = await SemanticPolicyEngine(Runtime(), confidence_threshold=0.7).decide(
                    request_for()
                )
                self.assertEqual(decision.action, PolicyAction.UNCERTAIN)
                self.assertIsNone(decision.intent)
                self.assertIsNone(decision.source_language)
                self.assertIsNone(decision.target_language)

    async def test_invalid_mode_switch_schema_becomes_uncertain(self):
        class Runtime:
            async def generate(self, prompt):
                return json.dumps(
                    {
                        "action": "MODE_SWITCH",
                        "confidence": 0.99,
                        "rationale": "single translation is not a mode",
                        "intent": "one_shot_translation",
                        "target_language": "English",
                    }
                )

        decision = await SemanticPolicyEngine(Runtime()).decide(request_for("翻译这一句"))

        self.assertEqual(decision.action, PolicyAction.UNCERTAIN)
        self.assertIsNone(decision.intent)

    async def test_arbitrary_runtime_failure_becomes_safe_uncertain(self):
        class Runtime:
            async def generate(self, prompt):
                raise OSError("provider process disappeared")

        decision = await SemanticPolicyEngine(Runtime()).decide(request_for())

        self.assertEqual(decision.action, PolicyAction.UNCERTAIN)
        self.assertEqual(decision.rationale, "policy runtime failure")

    async def test_timeout_is_bounded_and_late_result_cannot_replace_decision(self):
        published = []

        class Runtime:
            async def generate(self, prompt):
                try:
                    await asyncio.sleep(0.2)
                except asyncio.CancelledError:
                    await asyncio.sleep(0.03)
                    return '{"action":"ANSWER","confidence":1,"rationale":"late"}'

        engine = SemanticPolicyEngine(Runtime(), timeout_ms=20, on_decision=published.append)
        started = time.perf_counter()
        decision = await engine.decide(request_for())
        elapsed = time.perf_counter() - started
        await asyncio.sleep(0.05)

        self.assertLess(elapsed, 0.1)
        self.assertEqual(decision.action, PolicyAction.UNCERTAIN)
        self.assertEqual(published, [decision])

    async def test_sequential_calls_keep_requests_isolated(self):
        class Runtime:
            async def generate(self, prompt):
                text = json.loads(prompt)["user_transcript"]["text"]
                await asyncio.sleep(0.01 if text == "first" else 0)
                action = "PAUSE" if text == "first" else "RESUME"
                return json.dumps({"action": action, "confidence": 0.99, "rationale": text})

        engine = SemanticPolicyEngine(Runtime())
        first = await engine.decide(request_for("first"))
        second = await engine.decide(request_for("second"))
        self.assertEqual((first.action, first.rationale), (PolicyAction.PAUSE, "first"))
        self.assertEqual((second.action, second.rationale), (PolicyAction.RESUME, "second"))

    async def test_noncooperative_sync_timeout_stays_tracked_and_rejects_accumulation(self):
        started = threading.Event()
        release = threading.Event()
        published = []

        class Runtime:
            def __init__(self):
                self.call_count = 0

            def generate(self, prompt):
                self.call_count += 1
                if self.call_count == 1:
                    started.set()
                    release.wait(timeout=2.0)
                    return '{"action":"ANSWER","confidence":1,"rationale":"late first"}'
                return '{"action":"RESUME","confidence":1,"rationale":"next request"}'

        runtime = Runtime()
        engine = SemanticPolicyEngine(runtime, timeout_ms=20, on_decision=published.append)
        heartbeat = asyncio.Event()

        async def beat():
            await asyncio.sleep(0.005)
            heartbeat.set()

        beat_task = asyncio.create_task(beat())
        first = await engine.decide(request_for("first"))
        await beat_task

        self.assertTrue(started.is_set())
        self.assertTrue(heartbeat.is_set())
        self.assertTrue(engine.inflight)
        self.assertEqual(first.rationale, "policy timeout")

        busy = await asyncio.gather(
            *(engine.decide(request_for(f"busy-{index}")) for index in range(8))
        )
        self.assertEqual(runtime.call_count, 1)
        self.assertTrue(all(item.action is PolicyAction.UNCERTAIN for item in busy))
        self.assertTrue(all(item.rationale == "policy busy" for item in busy))

        release.set()
        for _ in range(100):
            if not engine.inflight:
                break
            await asyncio.sleep(0.005)
        self.assertFalse(engine.inflight)

        next_decision = await engine.decide(request_for("next"))
        self.assertEqual(runtime.call_count, 2)
        self.assertEqual(next_decision.action, PolicyAction.RESUME)
        self.assertNotIn(PolicyAction.ANSWER, [item.action for item in published])


class QwenPolicyProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_runtime_and_output_normalization_stay_off_event_loop(self):
        heartbeat = asyncio.Event()

        class BlockingRuntime:
            def generate(self, prompt, **options):
                time.sleep(0.04)
                return {"generated_text": "{\"action\":\"ANSWER\",\"confidence\":1,\"rationale\":\"ok\"}"}

        provider = QwenPolicyProvider(runtime=BlockingRuntime())

        async def beat():
            await asyncio.sleep(0.005)
            heartbeat.set()

        generation = asyncio.create_task(provider.generate("prompt"))
        await beat()
        self.assertTrue(heartbeat.is_set())
        self.assertFalse(generation.done())
        self.assertIn('"ANSWER"', await generation)

    async def test_async_runtime_is_supported(self):
        class Runtime:
            async def generate(self, prompt, **options):
                return '{"action":"PAUSE","confidence":1,"rationale":"ok"}'

        provider = QwenPolicyProvider(runtime=Runtime())
        self.assertIn('"PAUSE"', await provider.generate("prompt"))

    async def test_sync_callable_returning_awaitable_is_supported(self):
        class Runtime:
            def generate(self, prompt, **options):
                async def result():
                    return '{"action":"RESUME","confidence":1,"rationale":"ok"}'

                return result()

        provider = QwenPolicyProvider(runtime=Runtime())
        self.assertIn('"RESUME"', await provider.generate("prompt"))


class PolicyBenchmarkTests(unittest.TestCase):
    def test_help_works_from_absolute_path_without_loading_model(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "benchmark_policy_llm.py"), "--help"],
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--placement", result.stdout)
        self.assertIn("--quantization", result.stdout)
        self.assertIn("--prompt-case", result.stdout)

    def test_percentiles_and_gate_are_model_independent(self):
        from scripts.benchmark_policy_llm import summarize_latencies

        summary = summarize_latencies([100.0, 200.0, 300.0, 400.0], gate_ms=300.0)
        self.assertEqual(summary["p50_ms"], 250.0)
        self.assertEqual(summary["p95_ms"], 385.0)
        self.assertFalse(summary["meets_300ms_gate"])

    def test_multiple_placements_are_reported_and_only_passing_fastest_is_selected(self):
        from scripts.benchmark_policy_llm import analyze_placements, parse_placements

        placements = parse_placements(["cuda:none", "cpu:8bit", "cuda:4bit"])
        report = analyze_placements(
            placements,
            {
                "cuda:none": [250.0, 290.0, 310.0],
                "cpu:8bit": [320.0, 350.0, 400.0],
                "cuda:4bit": [120.0, 140.0, 160.0],
            },
            gate_ms=300.0,
        )

        self.assertEqual(
            [(item.device, item.quantization) for item in placements],
            [("cuda", "none"), ("cpu", "8bit"), ("cuda", "4bit")],
        )
        self.assertEqual(set(report["placements"]), {"cuda:none", "cpu:8bit", "cuda:4bit"})
        self.assertFalse(report["placements"]["cuda:none"]["meets_gate"])
        self.assertFalse(report["placements"]["cpu:8bit"]["meets_gate"])
        self.assertTrue(report["placements"]["cuda:4bit"]["meets_gate"])
        self.assertEqual(report["selected_placement"], "cuda:4bit")

    def test_no_placement_is_selected_when_all_fail_gate(self):
        from scripts.benchmark_policy_llm import analyze_placements, parse_placements

        placements = parse_placements(["cuda:none", "cpu:none"])
        report = analyze_placements(
            placements,
            {"cuda:none": [301.0], "cpu:none": [450.0]},
            gate_ms=300.0,
        )

        self.assertIsNone(report["selected_placement"])

    def test_representative_benchmark_prompt_uses_full_policy_request(self):
        from scripts.benchmark_policy_llm import representative_policy_prompt

        payload = json.loads(representative_policy_prompt("revise"))

        self.assertEqual(payload["state"]["response"], "DUCKED")
        self.assertEqual(payload["state"]["assistant_act"], "STATEMENT")
        self.assertEqual(payload["user_transcript"]["revision_id"], 7)
        self.assertEqual(payload["candidate"]["label"], "turn_end")
        self.assertEqual(payload["task_checkpoint"], {"city": "上海", "step": 2})
        self.assertEqual(payload["policy_contract"]["allowed_actions"][0], "BACKCHANNEL")


if __name__ == "__main__":
    unittest.main()
