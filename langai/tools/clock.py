"""The clock tool, and the wording that asks for it. Models have no clock and will confidently invent a date."""

import re
from datetime import datetime

from langchain_core.tools import tool

# "whats the time", "what day is it today", "the full date", "che ore sono"
ASK = re.compile(
    r"\b(?:what|which|whats|what'?s)\s+(?:is\s+)?(?:the\s+)?(?:time|day|date)\b"
    r"|\btime\s+is\s+it\b|\b(?:current|todays?|today'?s)\s+(?:time|date|day)\b"
    r"|\b(?:full|complete|exact)\s+(?:date|time)\b|^\s*(?:the\s+)?(?:time|date)\s*\?*$"
    r"|\bche\s+(?:ore|ora|giorno|data)\b|\bdata\s+(?:di\s+oggi|completa)\b", re.I)
# a named place or zone is the one case the model has to work out rather than read out
QUALIFIED = re.compile( r"\b(?:in|at|for|a|per|timezone|fuso|utc|gmt)\s+(?!the\s+moment|now|present)\w", re.I)


@tool
def now() -> str:
    """Current date and time. Use for anything about today, now, or the date."""
    return datetime.now().astimezone().strftime("%A %d %B %Y, %H:%M %Z")


def asked(text, last_user=""):
    """Answer straight from the clock unless a place is named. The model only adds risk: it has
    mangled the date, refused outright, and cut a full date down to "Thursday".
    """
    if not ASK.search(text):
        return None
    return {"rule": "asked the time", "tools": (), "prefetch": "now", "direct": not QUALIFIED.search(text)}
