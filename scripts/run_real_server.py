"""Serve the realtime runtime stack and bundled frontend on port 8001."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_app(args: argparse.Namespace):
    from src.config.loader import load_yaml
    from src.runtime_app.bootstrap import RealtimeServerSettings, build_realtime_app
    from src.runtime_app.container import build_production_runtime_factory

    model_config = load_yaml(getattr(args, "config", ROOT / "configs" / "models.yaml"))
    loaded_audio_config = load_yaml(getattr(args, "audio_config", ROOT / "configs" / "audio.yaml"))
    audio_config = loaded_audio_config.get("audio", loaded_audio_config)
    overrides = {
        "asr_model": getattr(args, "asr_model", None),
        "turn_model": getattr(args, "turn_model", None),
        "policy_model": getattr(args, "policy_model", None),
        "llm_model": getattr(args, "llm_model", None),
        "tts_model": getattr(args, "tts_model", None),
        "worker_python": getattr(args, "worker_python", None),
        "worker_script": getattr(args, "worker_script", None),
        "cosy_root": getattr(args, "cosy_root", None),
        "prompt_audio": getattr(args, "prompt_audio", None),
        "prompt_text": getattr(args, "prompt_text", None),
        "worker_startup_timeout": getattr(args, "worker_startup_timeout", None),
    }
    runtime_factory = build_production_runtime_factory(
        model_config=model_config,
        overrides=overrides,
        audio_config=audio_config,
    )
    settings = RealtimeServerSettings(
        static_dir=ROOT / "frontend",
        host=getattr(args, "host", "0.0.0.0"),
        port=getattr(args, "port", 8001),
    )
    # build_realtime_app still constructs FastAPI(..., lifespan=lifespan).
    return build_realtime_app(settings, runtime_factory=runtime_factory)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the realtime frontend backend on port 8001")
    parser.add_argument(
        "--config",
        default=os.environ.get("VOICE_AGENT_MODEL_CONFIG", str(ROOT / "configs" / "models.yaml")),
    )
    parser.add_argument(
        "--audio-config",
        default=os.environ.get("VOICE_AGENT_AUDIO_CONFIG", str(ROOT / "configs" / "audio.yaml")),
    )
    parser.add_argument("--asr-model", default=os.environ.get("VOICE_AGENT_ASR_MODEL"))
    parser.add_argument("--turn-model", default=os.environ.get("VOICE_AGENT_TURN_MODEL"))
    parser.add_argument("--policy-model", default=os.environ.get("VOICE_AGENT_POLICY_MODEL"))
    parser.add_argument("--llm-model", default=os.environ.get("VOICE_AGENT_LLM_MODEL"))
    parser.add_argument("--tts-model", default=os.environ.get("VOICE_AGENT_TTS_MODEL"))
    parser.add_argument("--prompt-audio", default=os.environ.get("VOICE_AGENT_PROMPT_AUDIO"))
    parser.add_argument("--prompt-text", default=os.environ.get("VOICE_AGENT_PROMPT_TEXT"))
    parser.add_argument("--host", default=os.environ.get("VOICE_AGENT_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("VOICE_AGENT_PORT", "8001")))
    parser.add_argument(
        "--worker-startup-timeout",
        type=float,
        default=float(os.environ.get("VOICE_AGENT_WORKER_STARTUP_TIMEOUT", "180.0")),
    )
    parser.add_argument(
        "--worker-python",
        default=os.environ.get("VOICE_AGENT_WORKER_PYTHON", "/home/CosyVoice/.venv/bin/python"),
    )
    parser.add_argument(
        "--cosy-root",
        default=os.environ.get("VOICE_AGENT_COSY_ROOT", "/home/CosyVoice"),
    )
    parser.add_argument(
        "--worker-script",
        default=os.environ.get(
            "VOICE_AGENT_WORKER_SCRIPT",
            str(Path(__file__).with_name("cosyvoice_worker.py")),
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
