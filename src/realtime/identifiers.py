"""Identifiers and generation epochs for one realtime session."""


class IdentifierAllocator:
    """Allocate Task 1-compatible event and response string identities."""

    def __init__(self) -> None:
        self._event_sequence = 0
        self._response_sequence = 0

    def next_event_id(self) -> str:
        """Return the next unique event identity for this allocator."""
        self._event_sequence += 1
        return f"evt-{self._event_sequence}"

    def next_response_id(self) -> str:
        """Return the next unique response identity for this allocator."""
        self._response_sequence += 1
        return f"response-{self._response_sequence}"


class GenerationClock:
    """Monotonically advance and validate the active response generation."""

    def __init__(self) -> None:
        self._current = 0

    @property
    def current(self) -> int:
        """Return the active generation epoch."""
        return self._current

    def advance(self) -> int:
        """Invalidate the current generation and return its replacement epoch."""
        self._current += 1
        return self._current

    def is_current(self, epoch: int) -> bool:
        """Return whether ``epoch`` still identifies the active generation."""
        return epoch == self._current
