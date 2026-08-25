"""Priority levels used by the realtime runtime scheduler."""

from enum import IntEnum


class Priority(IntEnum):
    """Scheduling priority, ordered from most to least time-sensitive."""

    PLAYBACK_STOP_DUCK = 6
    SPEECH_START_AND_X2 = 5
    SEMANTIC_POLICY = 4
    FIRST_TTS_SEGMENT = 3
    MAIN_LLM_GENERATION = 2
    LATER_TTS_BACKGROUND = 1

    # Established names remain valid where their original semantics match.
    INTERRUPT = PLAYBACK_STOP_DUCK
    AUDIO = SPEECH_START_AND_X2
    TURN_EVENT = SPEECH_START_AND_X2
    GENERATION = MAIN_LLM_GENERATION
    BACKGROUND = LATER_TTS_BACKGROUND
