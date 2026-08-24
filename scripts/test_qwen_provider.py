"""Run a real local Qwen Transformers streaming smoke test."""

from __future__ import annotations

import argparse
import asyncio
import time

from src.adapters.llm.providers.qwen_transformers import TransformersQwenProvider


async def main(model_path: str, prompt: str) -> None:
    started = time.perf_counter()
    provider = TransformersQwenProvider(model_path, device="cuda")
    loaded = time.perf_counter()
    chunks = []
    async for chunk in provider.stream_tokens(prompt):
        chunks.append(chunk.text)
        print(chunk.text, end="", flush=True)
    print()
    print(f"model_load_seconds={loaded - started:.3f}")
    print(f"token_chunks={len(chunks)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.prompt))
