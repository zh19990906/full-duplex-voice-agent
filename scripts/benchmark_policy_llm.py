#!/usr/bin/env python3
"""Benchmark local semantic-policy latency without loading on import/help."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import gc
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PROMPT_CASES = {
    "backchannel": "嗯，对，继续",
    "answer": "对",
    "revise": "对，不过我说的是北京",
    "pause": "嗯……等一下",
    "interpretation": "接下来一直把中文实时翻译成英文",
}


@dataclass(frozen=True)
class PlacementSpec:
    device: str
    quantization: str

    @property
    def key(self) -> str:
        return f"{self.device}:{self.quantization}"


def parse_placements(values: Sequence[str]) -> list[PlacementSpec]:
    placements = []
    seen = set()
    for value in values:
        parts = value.split(":")
        if len(parts) != 2 or parts[0] not in {"cuda", "cpu"} or parts[1] not in {
            "none",
            "4bit",
            "8bit",
        }:
            raise ValueError(
                f"invalid placement {value!r}; expected DEVICE:QUANTIZATION"
            )
        placement = PlacementSpec(parts[0], parts[1])
        if placement.key in seen:
            raise ValueError(f"duplicate placement: {placement.key}")
        seen.add(placement.key)
        placements.append(placement)
    if not placements:
        raise ValueError("at least one placement is required")
    return placements


def percentile(values: list[float], percent: float) -> float:
    if not values:
        raise ValueError("latencies must not be empty")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_latencies(values: list[float], *, gate_ms: float = 300.0) -> dict[str, Any]:
    p50 = percentile(values, 0.50)
    p95 = percentile(values, 0.95)
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": p50,
        "p95_ms": p95,
        "gate_ms": gate_ms,
        "meets_gate": p95 <= gate_ms,
        "meets_300ms_gate": p95 <= gate_ms,
    }


def analyze_placements(
    placements: Sequence[PlacementSpec],
    measurements: Mapping[str, list[float]],
    *,
    gate_ms: float = 300.0,
) -> dict[str, Any]:
    summaries = {}
    for placement in placements:
        if placement.key not in measurements:
            raise ValueError(f"missing measurements for {placement.key}")
        summaries[placement.key] = summarize_latencies(
            measurements[placement.key], gate_ms=gate_ms
        )
    passing = [
        (name, summary)
        for name, summary in summaries.items()
        if summary["meets_gate"]
    ]
    selected = min(
        passing,
        key=lambda item: (item[1]["p95_ms"], item[1]["p50_ms"], item[0]),
        default=None,
    )
    return {
        "gate_ms": gate_ms,
        "placements": summaries,
        "selected_placement": selected[0] if selected is not None else None,
    }


def representative_policy_prompt(prompt_case: str) -> str:
    """Build the same complete deterministic request shape used by the engine."""
    from src.asr.stream import TranscriptChunk
    from src.realtime.policy import PolicyRequest
    from src.realtime.session_state import (
        ConversationMode,
        FloorState,
        ResponseState,
        SessionState,
    )
    from src.realtime.speech_fusion import SpeechCandidateEvent

    if prompt_case not in PROMPT_CASES:
        raise ValueError(f"unknown prompt case: {prompt_case}")
    text = PROMPT_CASES[prompt_case]
    request = PolicyRequest(
        state=SessionState(
            ConversationMode.CHAT,
            FloorState.OVERLAP,
            ResponseState.DUCKED,
            "STATEMENT",
        ),
        assistant_last_text="我先按上海为目的地继续介绍行程。",
        unplayed_text_summary="上海后续景点和交通建议",
        user_transcript=TranscriptChunk(
            "benchmark-transcript",
            text,
            1.0,
            True,
            revision_id=7,
            committed_text=text,
        ),
        candidate=SpeechCandidateEvent(
            "USER_TURN_END_CANDIDATE",
            "benchmark-candidate",
            1.1,
            "x2_turn",
            {
                "label": "turn_end",
                "confidence": 0.94,
                "evidence": {"frames": 2, "sequence": 81},
            },
        ),
        task_checkpoint={"city": "上海", "step": 2},
        source_language="Chinese",
        target_language="English",
    )
    return request.to_prompt_json()


async def _release_provider(provider: Any) -> None:
    close = getattr(provider, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result
    if hasattr(provider, "runtime"):
        provider.runtime = None
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from src.adapters.llm.providers.qwen_policy import QwenPolicyProvider

    placement_values = args.placement or [f"{args.device}:{args.quantization}"]
    placements = parse_placements(placement_values)
    prompt = representative_policy_prompt(args.prompt_case)
    measurements = {}
    errors = {}
    for placement in placements:
        provider = QwenPolicyProvider(
            args.model,
            device=placement.device,
            quantization=placement.quantization,
            load_local=True,
        )
        try:
            for _ in range(args.warmup):
                await provider.generate(prompt)
            latencies = []
            for _ in range(args.runs):
                started = time.perf_counter()
                await provider.generate(prompt)
                latencies.append((time.perf_counter() - started) * 1000.0)
            measurements[placement.key] = latencies
        except Exception as exc:
            errors[placement.key] = f"{type(exc).__name__}: {exc}"
        finally:
            await _release_provider(provider)

    successful = [item for item in placements if item.key in measurements]
    report = (
        analyze_placements(successful, measurements, gate_ms=args.gate_ms)
        if successful
        else {"gate_ms": args.gate_ms, "placements": {}, "selected_placement": None}
    )
    for placement in placements:
        if placement.key in errors:
            report["placements"][placement.key] = {
                "error": errors[placement.key],
                "meets_gate": False,
                "meets_300ms_gate": False,
            }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local Qwen semantic policy latency")
    parser.add_argument("--model", required=True, help="Local Qwen model directory")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--quantization", default="none", choices=("none", "4bit", "8bit"))
    parser.add_argument(
        "--placement",
        action="append",
        metavar="DEVICE:QUANTIZATION",
        help="Repeat to compare multiple explicit placements in one run",
    )
    parser.add_argument("--prompt-case", default="backchannel", choices=tuple(PROMPT_CASES))
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--gate-ms", type=float, default=300.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.runs < 1 or args.warmup < 0:
        raise SystemExit("--runs must be positive and --warmup must be nonnegative")
    summary = asyncio.run(run(args))
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["selected_placement"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
