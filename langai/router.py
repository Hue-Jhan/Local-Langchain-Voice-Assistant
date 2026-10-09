"""What a turn needs, decided before the model sees it. Full reasoning in CLAUDE.md.

Each tool file owns the wording that asks for it and returns a plain dict, which becomes a
Decision here; this file holds the rule order and the rules belonging to no tool.

No tool is ever bound, for four measured reasons on qwen2.5:3b: handed tools it calls them for
no reason ("ok" searched "current political developments"); handed both it picks the wrong one
("btc price now" fetched the clock); told outright to search it refuses, and ollama ignores
tool_choice; and worst, binding schemas on one turn and not the next invalidates ollama's prompt
cache, costing 4.5s a turn against 1.2s. So the rules decide, we run the tool, and the prompt
prefix never changes shape. `[tools] mode = "model"` restores model-chosen tools, slower.

To add a layer, write a rule returning a dict and put it in RULES.
"""

import re
from dataclasses import dataclass

from .tools import clock, contentless, web


@dataclass(frozen=True)
class Decision:
    rule: str
    tools: tuple | None = None  # None = every tool, () = none, else these names
    prefetch: str = ""          # a tool we run ourselves before the model
    query: str = ""             # argument for the prefetch
    direct: bool = False        # the tool result IS the answer; skip the model


ALL = Decision("tools bound", None)   # [tools] mode = "model"
NOTHING = Decision("default", ())     # the usual case: answer, no tool

# arithmetic anywhere in the message: "2+2", "whats 2*3?", "quanto fa 5-1"
MATH = re.compile(r"\d+(?:\.\d+)?\s*[-+*/x×÷^]\s*\d+(?:\.\d+)?")
WORDY_MATH = re.compile( r"\b\d+\s*(?:times|plus|minus|divided by|per|più|meno|diviso)\s*\d+\b", re.I)
# dates and year ranges look like arithmetic but still deserve their tools
DATEISH = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}\s*-\s*\d{4}")


def filler(text, last_user=""):
    """Greetings and acknowledgements need no tool at all."""
    return {"rule": "filler", "tools": ()} if contentless(text) else None


def arithmetic(text, last_user=""):
    """Maths is knowledge, not a lookup. The "whats 2*3?" case."""
    if DATEISH.search(text):
        return None  # a date is not a sum
    if MATH.search(text) or WORDY_MATH.search(text):
        return {"rule": "arithmetic", "tools": ()}
    return None


# earliest decision wins; add layers here
RULES = [filler, web.refused, arithmetic, web.asked, clock.asked, web.live]


def decide(text, last_user="", rules=None):
    """What this turn should be given. last_user is the previous thing you asked."""
    for rule in RULES if rules is None else rules:
        if (found := rule(text, last_user)) is not None:
            return Decision(**found)
    return NOTHING
