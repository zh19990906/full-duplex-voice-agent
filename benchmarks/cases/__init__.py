"""The four full-duplex capability benchmark cases."""

from .case01_backchannel import SCENARIO as CASE01_BACKCHANNEL
from .case02_interrupt import SCENARIO as CASE02_INTERRUPT
from .case03_translation import SCENARIO as CASE03_TRANSLATION
from .case04_resume_task import SCENARIO as CASE04_RESUME_TASK

__all__ = [
    "CASE01_BACKCHANNEL",
    "CASE02_INTERRUPT",
    "CASE03_TRANSLATION",
    "CASE04_RESUME_TASK",
]
