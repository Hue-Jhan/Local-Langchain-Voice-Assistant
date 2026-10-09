"""Tools the model may call, and the registry that binds them.

Kept deliberately few: every tool's schema is re-sent on every turn, and on a CPU box each extra one
costs seconds of prompt evaluation.
"""

from .clock import now
from .web import (BACKEND, FORMATTER, FILLER, Result, contentless, numbered_text, search, search_web, set_turn)

TOOLS = [now, search_web]

__all__ = ["TOOLS", "BACKEND", "FORMATTER", "FILLER", "Result", "contentless", "now", "numbered_text", "search", "search_web", "set_turn"]
