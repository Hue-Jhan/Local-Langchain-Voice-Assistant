"""Web search: the DuckDuckGo backend, the result formatting, the tool itself, the guard that stops it
firing on a message with nothing to search for, and the wording that asks for a search.
"""

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from ddgs import DDGS
from langchain_core.tools import tool

MAX_RESULTS, SNIPPET_CHARS, TIMEOUT, ATTEMPTS = 7, 200, 20, 3


@dataclass
class Result:
    title: str
    url: str
    snippet: str


def ddg_text(query, max_results=MAX_RESULTS):
    """DuckDuckGo text search. Retries - the public backend times out intermittently."""
    last = None
    for _ in range(ATTEMPTS):
        try:
            hits = DDGS(timeout=TIMEOUT).text(query, max_results=max_results)
        except Exception as exc:
            last = exc
            continue
        return [Result(h.get("title", ""), h.get("href", ""), h.get("body", "")) for h in hits]
    raise RuntimeError(last)


def numbered_text(results):
    """Flat numbered lines - small models parse this far more reliably than JSON."""
    if not results:
        return "No results."
    return "\n".join(
        f"[{i}] {r.title} ({urlsplit(r.url).netloc.removeprefix('www.')})\n"
        f"    {' '.join(r.snippet.split())[:SNIPPET_CHARS]}"
        for i, r in enumerate(results, 1))


# the seam: reassign either name to swap provider or output shape
BACKEND, FORMATTER = ddg_text, numbered_text


def search(text):
    try:
        results = BACKEND(text)
    except Exception as exc:
        return f"Search unavailable: {exc}"
    return FORMATTER(results)


MIN_WORD = 4  # shorter words carry no searchable content
# greetings and acknowledgements, Italian and English
FILLER = {
    "ok", "okay", "si", "no", "pronto", "ciao", "salve", "allora", "boh", "beh",
    "bene", "va", "ecco", "grazie", "prego", "scusa", "senti", "dai", "mmm", "ehm",
    "eh", "ah", "oh", "hey", "hi", "hello", "yo", "sup", "thanks", "cool", "nice",
    "sure", "yes", "yeah", "nope", "right",
}

_turn = ""  # the user's current message


def set_turn(text):
    """Record the user's message so search_web can refuse a pointless search."""
    global _turn
    _turn = text or ""


def contentless(text):
    """True when a message is only a greeting or acknowledgement, which small models answer by
    inventing a search. Judges only what the user said: judging the query would wrongly block
    "meteo roma" answered by "weather rome".
    """
    words = re.findall(r"\w+", text.lower())
    if not words or len(words) > 3:
        return False
    return not any(w not in FILLER and len(w) >= MIN_WORD for w in words)


@tool
def search_web(query: str) -> str:
    """Search the web for current facts. Input: a short search query."""
    if contentless(_turn):
        return "Search skipped: nothing in the message to search for. Just reply."
    try:
        return search(query)
    except Exception as exc:
        return f"Search failed: {exc}"


# --- when to search -------------------------------------------------------------
# Checked before anything offers a tool: "dont use the search tool" searched for "dont use the
# tool". The negation must sit next to the trigger, so "dont be lazy, look up the price" searches.
REFUSED = re.compile(
    r"\b(?:don'?t|do\s+not|dont|never|stop|without|no\s+need\s+to|avoid|"
    r"non|senza|smetti\s+di|evita\s+di)\b[\w\s'-]{0,14}?"
    r"(?:cerca\w*|ricerca\w*|google\w*|search\w*|lookup|look\s+\w+\s+up|internet|"
    r"\bweb\b|tool)", re.I)
# an outright instruction to look something up
ASK = re.compile(r"\b(?:cerca\w*|ricerca\w*|informati|google\w*|search(?:\s+for)?|" r"lookup|look(?:\s+\w+){0,2}\s+up)\b", re.I)
# wording only live data can answer, so search is the only sensible tool
LIVE = re.compile(
    r"\b(?:news|notizie|weather|meteo|forecast|previsioni|price|prezzo|quotazione|"
    r"score|risultato|classifica|stock|borsa|latest|ultima|ultimo|recent|recente|"
    r"release[ds]?|uscito|version|versione|who won|chi ha vinto|trending)\b", re.I)
# trigger words are instructions to us, not part of what to search for
STRIP = re.compile(
    r"\b(?:cerca\w*|ricerca\w*|informati\s+su|informati|google\w*|search(?:\s+for)?|"
    r"lookup|look(?:\s+\w+){0,2}\s+up|per\s+favore|please|dimmi|mi\s+dici|tell\s+me|"
    r"puoi|can\s+you|yes\s+you\s+can)\b", re.I)
# "look it up", "cercalo": the subject is the previous turn, not this message
PRONOUN = re.compile(r"\b(?:it|that|this|them|those|these)\b|cercal[oai]\b|" r"trovalo\b|guardalo\b", re.I)


def _clean(text):
    bare = re.sub(r"[^\w\s'àèéìòùáéíóú-]", " ", STRIP.sub(" ", text))
    return " ".join(bare.split())


def _weak(query):
    """Nothing to go on - "it", "-lo", "godamnit"."""
    words = query.split()
    return len(words) < 2 and not any(len(w) >= 5 for w in words)


def query(text, last_user=""):
    """"cerca il prezzo del btc" -> "il prezzo del btc". "look it up" and "cercalo" carry no subject of
    their own, so they fall back to the previous message.
    """
    found = _clean(text)
    if last_user and (_weak(found) or PRONOUN.search(text)):
        if not _weak(fallback := _clean(last_user)):
            return fallback
    return found or text.strip()


def refused(text, last_user=""):
    """Told not to look it up."""
    return {"rule": "told not to search", "tools": ()} if REFUSED.search(text) else None


def asked(text, last_user=""):
    """Told to look it up, so look it up - do not leave it to the model."""
    if not ASK.search(text):
        return None
    return {"rule": "asked to search", "tools": (), "prefetch": "search_web", "query": query(text, last_user)}


def live(text, last_user=""):
    """Prices, news, results: only a search can answer."""
    if not LIVE.search(text):
        return None
    return {"rule": "live data", "tools": (), "prefetch": "search_web", "query": query(text, last_user)}
