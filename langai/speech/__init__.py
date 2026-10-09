"""Speech out (speaker) and speech in (ear).

speaker is cheap to import. ear pulls in vosk and sounddevice, so it is imported directly by whoever
needs it rather than here.
"""

from . import speaker

__all__ = ["speaker"]
