"""Commit only ASR text that survived two consecutive hypotheses."""

from __future__ import annotations


class StablePrefixCommitter:
    """Maintain one turn's committed prefix and replaceable hypothesis tail.

    A prefix becomes externally visible only after it belongs to the longest
    common prefix of two consecutive decoder hypotheses.  Committed text is
    intentionally append-only: downstream speech may already have consumed it.
    """

    def __init__(self) -> None:
        self.reset()

    def update(self, hypothesis: str) -> str:
        """Apply a non-final hypothesis and return newly stable text only."""
        _require_text(hypothesis)
        common = _longest_common_prefix(self.previous_hypothesis, hypothesis)
        addition = ""
        if common.startswith(self.committed_text):
            addition = common[len(self.committed_text) :]
            self.committed_text += addition

        self.previous_hypothesis = hypothesis
        self.unstable_text = (
            hypothesis[len(self.committed_text) :]
            if hypothesis.startswith(self.committed_text)
            else ""
        )
        self._finalized = False
        return addition

    def finalize(self, hypothesis: str | None = None) -> str:
        """Commit a final hypothesis's remaining suffix exactly once."""
        if self._finalized:
            return ""
        final_hypothesis = self.previous_hypothesis if hypothesis is None else hypothesis
        _require_text(final_hypothesis)
        if final_hypothesis.startswith(self.committed_text):
            addition = final_hypothesis[len(self.committed_text) :]
        elif not self.committed_text:
            addition = final_hypothesis
        else:
            # A decoder correction cannot retract text already sent downstream.
            # The provider publishes an authoritative replacement chunk for
            # this case. Keep the committer's internal state exact as well.
            addition = ""
            self.committed_text = final_hypothesis
        if final_hypothesis.startswith(self.committed_text):
            self.committed_text += addition
        self.previous_hypothesis = final_hypothesis
        self.unstable_text = ""
        self._finalized = True
        return addition

    def reset(self) -> None:
        """Start a new turn without retaining any prior hypothesis state."""
        self.committed_text = ""
        self.previous_hypothesis = ""
        self.unstable_text = ""
        self._finalized = False


def _longest_common_prefix(left: str, right: str) -> str:
    index = 0
    limit = min(len(left), len(right))
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


def _require_text(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("ASR hypothesis must be a string")
