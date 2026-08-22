"""Priority levels used by the realtime runtime scheduler."""

from enum import IntEnum


class Priority(IntEnum):
    """Scheduling priority, ordered from most to least time-sensitive."""

    INTERRUPT = 5
    AUDIO = 4
    TURN_EVENT = 3
    GENERATION = 2
    BACKGROUND = 1
