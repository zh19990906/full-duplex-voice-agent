"""Application-level autonomous tool-use orchestration.

The public classes are loaded lazily so importing ``src.agent.models`` does
not eagerly import ``src.agent.loop`` and its tool-parser dependency.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loop import AgentLoop, AgentLoopError
    from .models import AgentState, AgentStatus, ToolCallRequest

__all__ = ["AgentLoop", "AgentLoopError", "AgentState", "AgentStatus", "ToolCallRequest"]


def __getattr__(name: str):
    """Resolve package-level exports only when they are requested."""

    if name in {"AgentLoop", "AgentLoopError"}:
        from .loop import AgentLoop, AgentLoopError

        return {"AgentLoop": AgentLoop, "AgentLoopError": AgentLoopError}[name]
    if name in {"AgentState", "AgentStatus", "ToolCallRequest"}:
        from .models import AgentState, AgentStatus, ToolCallRequest

        return {
            "AgentState": AgentState,
            "AgentStatus": AgentStatus,
            "ToolCallRequest": ToolCallRequest,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
