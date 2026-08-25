#!/usr/bin/env python3
"""Benchmark local semantic-policy latency without loading on import/help."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import statistics
import sys
import time
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
        "meets_300ms_gate": p95 <= gate_ms,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from src.adapters.llm.providers.qwen_policy import QwenPolicyProvider

    provider = QwenPolicyProvider(
        args.model,
        device=args.device,
        quantization=args.quantization,
        load_local=True,
    )
    prompt = json.dumps(
        {"user_transcript": {"text": PROMPT_CASES[args.prompt_case]}},
        ensure_ascii=False,
    )
    latencies = []
    for _ in range(args.warmup):
        await provider.generate(prompt)
    for _ in range(args.runs):
        started = time.perf_counter()
        await provider.generate(prompt)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return summarize_latencies(latencies, gate_ms=args.gate_ms)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark local Qwen semantic policy latency")
    parser.add_argument("--model", required=True, help="Local Qwen model directory")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--quantization", default="none", choices=("none", "4bit", "8bit"))
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
    return 0 if summary["meets_300ms_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
