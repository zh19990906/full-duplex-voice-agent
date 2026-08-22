"""Minimal runtime lifecycle hooks."""


class RuntimeLifecycle:
    """Overridable initialization and shutdown hooks for a runtime."""

    async def initialize(self) -> None:
        """Initialize runtime infrastructure."""
        return None

    async def start(self) -> None:
        """Start runtime infrastructure."""
        return None

    async def stop(self) -> None:
        """Stop runtime infrastructure."""
        return None

    async def shutdown(self) -> None:
        """Release runtime infrastructure."""
        return None


Lifecycle = RuntimeLifecycle
