"""One assistant session. Both the CLI and the TUI drive this, so turn handling, commands and model
switching exist once.
"""

import subprocess
import time
from dataclasses import dataclass
from datetime import datetime

from langchain_core.messages import (AIMessage, AIMessageChunk, HumanMessage, ToolMessage)

from .speech import speaker
from .agent import build_agent
from . import router
from .config import ROOT
from .tools import set_turn
from .ui.term import ThinkFilter

MODEL_MAP = {
    "qwen25": "qwen2.5:3b",
    "qwen4": "qwen3:4b",
    "granite": "granite3.3:2b",
    "lama3": "llama3.2:3b",
    "lama3ab": "huihui_ai/llama3.2-abliterate:3b",
    "qwen3ab": "huihui_ai/qwen2.5-coder-abliterate:3b",
    "qwen7ab": "huihui_ai/qwen2.5-coder-abliterate:7b",
}
# measured here - surfaced at startup so broken tool calling is not a mystery
MODEL_NOTES = {
    "qwen3ab": "abliterated: emits tool calls as plain text, so tools never fire",
    "qwen7ab": "abliterated: emits tool calls as plain text, so tools never fire",
    "lama3ab": "abliterated: tool arguments are often malformed",
    "lama3": "erratic: skips searches it needs and sometimes malforms arguments",
    "qwen4": "picks tools correctly but reasons at length - very slow on CPU",
    "granite": "declares tool support but emits no structured calls",
}
COMMANDS = [
    ("/help", "this list"),
    ("/menu", "current settings"),
    ("/v", "toggle voice"),
    ("/voice [on|off|auto|<id>]", "voice state; auto matches the language, or 0 IT 1 US 2 UK"),
    ("/s [text]", "speak the last answer, or the given text"),
    ("/stop", "stop playback"),
    ("/model [alias]", "show or switch model"),
    ("/exit", "quit"),
]
RECURSION_LIMIT, PREVIEW = 15, 100


@dataclass
class Stats:
    elapsed: float = 0.0
    tool_time: float = 0.0
    tokens: int = 0

    def __str__(self):
        model_time = max(self.elapsed - self.tool_time, 0.0)
        tools = f" ({self.tool_time:.1f}s tools)" if self.tool_time else ""
        # a direct answer costs no model time, and dividing by it claimed 660 tok/s
        rate = (f" · {self.tokens / model_time:.1f} tok/s"
                if model_time > 0.05 and self.tokens else "")
        return f"{self.elapsed:.1f}s{tools} · {self.tokens} tok{rate}"


def unload_models(keep=""):
    """Keep at most one model resident - this machine cannot hold two at once."""
    try:
        listed = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=15)
    except Exception:
        return
    for line in listed.stdout.splitlines()[1:]:
        name = line.split()[0] if line.split() else ""
        if name and name != keep:
            subprocess.run(["ollama", "stop", name], capture_output=True, timeout=7)


class Session:
    """Model, memory, voice and ear for one run of the program."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.alias = cfg["model"]["alias"]
        self.model = MODEL_MAP[self.alias]
        self.voice = speaker.VoiceState(bool(cfg["voice"]["enabled"]),
                                        int(cfg["voice"]["voice_id"]),
                                        bool(cfg["voice"]["auto"]))
        self.thread = cfg["memory"]["thread"]
        self.last_answer = ""
        self.last_user = ""
        self.gate = None
        self.agent, self._warm = build_agent(self.model, cfg)
        self._log = ROOT / cfg["log"]["path"] if cfg["log"]["path"] else None

    # --- plumbing ----------------------------------------------------------------

    @property
    def note(self):
        return MODEL_NOTES.get(self.alias, "")

    def run_config(self, tools=None, context=""):
        return {"configurable": {"thread_id": self.thread, "tools": tools, "context": context}, "recursion_limit": RECURSION_LIMIT}

    def log(self, kind, text):
        if not self._log:
            return
        try:
            with self._log.open("a") as handle:
                handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} [{kind}] {text}\n")
        except OSError:
            pass  # logging must never break a turn

    def warm(self):
        """Pay the cost of reading the stored conversation during startup instead of inside the first
        answer. Failing is harmless - it only loses the head start.
        """
        try:
            state = self.agent.get_state(self.run_config())
            self._warm((state.values or {}).get("messages") or [])
        except Exception as exc:
            self.log("error", f"warm: {exc}")

    def open_ear(self):
        """Build the speech gate. Returns a status line, or an error description."""
        settings = self.cfg["ear"]
        try:
            from .speech import ear
            if models := self.cfg["paths"].get("models"):
                ear.MODELS_DIR = (ROOT / models).resolve()
            self.gate = ear.build_gate(settings["gate"], settings["model"] or None, settings["device"], settings["language"])
        except Exception as exc:
            self.log("error", f"ear: {exc}")
            return f"ear disabled: {exc}"
        return f"ear: {self.gate.name} gate, vosk {self.gate.transcriber.path.name}"

    def switch_model(self, alias):
        if alias not in MODEL_MAP:
            return f"unknown model {alias!r}. known: {', '.join(MODEL_MAP)}"
        if alias == self.alias:
            return f"already on {alias} ({self.model})"
        speaker.stop()
        self.alias, self.model = alias, MODEL_MAP[alias]
        unload_models(keep=self.model)  # only one model fits in RAM
        self.cfg["model"]["alias"] = alias
        self.agent, self._warm = build_agent(self.model, self.cfg)
        self.log("model", self.model)
        extra = f" - {self.note}" if self.note else ""
        return f"switched to {alias} ({self.model}){extra}"

    def settings_text(self):
        ear = self.cfg["ear"]
        return "\n".join([
            f"model    {self.alias} ({self.model})",
            f"temp     {self.cfg['model']['temperature']}   "
            f"max tok {self.cfg['model']['max_tokens'] or 'off'}",
            f"{self.voice.status}",
            f"ear      {'on' if self.gate else 'off'}"
            f"{f', {self.gate.name} gate' if self.gate else ''}, lang {ear['language']}",
            f"memory   thread {self.thread}, "
            f"{'sqlite' if self.cfg['memory']['persist'] else 'in-process'}",
            f"router   {'on' if self.cfg['router']['enabled'] else 'off'}, "
            f"tools {self.cfg['tools']['mode']}",
            f"log      {self._log or 'off'}",
        ])

    def models_text(self):
        width = max(len(a) for a in MODEL_MAP)
        lines = []
        for alias, model in MODEL_MAP.items():
            mark = ">" if alias == self.alias else " "
            note = MODEL_NOTES.get(alias, "")
            lines.append(f"{mark} {alias:<{width}}  {model}" + (f"   - {note}" if note else ""))
        return "\n".join(lines) + "\n/model <alias> to switch"

    @staticmethod
    def help_text():
        width = max(len(name) for name, _ in COMMANDS)
        return "\n".join(f"{name:<{width}}  {what}" for name, what in COMMANDS)

    # --- a turn ------------------------------------------------------------------

    def run(self, text):
        """Stream one turn, yielding ("tool", line), ("token", text), ("stats", Stats)."""
        set_turn(text)  # lets search_web refuse a query invented out of nothing
        self.log("you", text)
        stats, think, answer = Stats(), ThinkFilter(), []
        began = time.perf_counter()
        tool_began = None

        if self.cfg["tools"]["mode"] == "model" or not self.cfg["router"]["enabled"]:
            decision = router.ALL  # bind every tool and let the model choose
        else:
            decision = router.decide(text, self.last_user)
        self.last_user = text

        context = ""
        if decision.prefetch:
            yield "tool", f"[router] {decision.rule} -> {decision.prefetch}"
            started = time.perf_counter()
            result = self.prefetch(decision)
            stats.tool_time += time.perf_counter() - started
            yield "tool", f"<- {decision.prefetch}: {' '.join(result.split())[:PREVIEW]}"
            if decision.direct:
                # the tool result is the whole answer; the model would only garble it
                yield "token", result
                self.last_answer = result
                self.log("bot", result)
                self.agent.update_state( self.run_config(), {"messages": [HumanMessage(text), AIMessage(result)]})
                stats.elapsed = time.perf_counter() - began
                stats.tokens = len(result.split())
                yield "stats", stats
                return
            context = self.context_for(decision.prefetch, result)
        elif decision.tools:
            yield "tool", f"[router] {decision.rule}: {', '.join(decision.tools)}"

        for mode, payload in self.agent.stream(
                {"messages": [HumanMessage(text)]},
                config=self.run_config(decision.tools, context),
                stream_mode=["messages", "updates"]):
            if mode == "messages":
                chunk, _meta = payload
                if not isinstance(chunk, AIMessageChunk) or not chunk.content:
                    continue
                if visible := think.feed(chunk.content):
                    answer.append(visible)
                    yield "token", visible
                continue
            for update in payload.values():
                for msg in (update or {}).get("messages") or []:
                    if isinstance(msg, AIMessage):
                        stats.tokens += (msg.usage_metadata or {}).get("output_tokens", 0)
                        if msg.tool_calls:
                            tool_began = time.perf_counter()
                        for call in msg.tool_calls:
                            self.log("tool", f"{call['name']}({call['args']})")
                            yield "tool", f"-> {call['name']}({call['args']})"
                    elif isinstance(msg, ToolMessage):
                        if tool_began is not None:
                            stats.tool_time += time.perf_counter() - tool_began
                            tool_began = None
                        preview = " ".join(msg.content.split())[:PREVIEW]
                        self.log("tool", f"{msg.name} -> {preview}")
                        yield "tool", f"<- {msg.name}: {preview}"

        if tail := think.flush():
            answer.append(tail)
            yield "token", tail

        stats.elapsed = time.perf_counter() - began
        stats.tokens = stats.tokens or sum(len(part.split()) for part in answer)
        final = "".join(answer).strip()
        if final:
            self.last_answer = final
            self.log("bot", final)
        yield "stats", stats

    @staticmethod
    def context_for(tool, result):
        """How a pre-run tool result is handed over. Only reached when the result is not the whole
        answer, so the clock arrives as a reference a named place can be converted from.
        """
        if tool == "now":
            return f"The current local time here is {result}. Use it to answer."
        return f"Here are current web search results. Answer using only these.\n\n{result}"

    def prefetch(self, decision):
        """Run a tool ourselves. Ollama ignores tool_choice, so when the user asked outright for a
        search this is the only way to guarantee one happens.
        """
        from .tools import now as clock
        from .tools import search
        try:
            if decision.prefetch == "search_web":
                return search(decision.query)
            return clock.invoke({})
        except Exception as exc:
            return f"({decision.prefetch} failed: {exc})"

    def maybe_speak(self, user_text):
        """Speak the last answer if the voice policy says so. Returns an error, if any."""
        if not self.last_answer:
            return None
        if not (spoken := speaker.resolve(self.voice, user_text, self.last_answer)):
            return None
        try:
            speaker.say(self.voice, spoken, user_text)
        except Exception as exc:
            return f"playback error: {exc}"
        return None

    def command(self, line):
        """Handle a slash command. Returns text to show, "" if silent, None if not ours."""
        name, _, rest = line.partition(" ")
        name, rest = name.lower(), rest.strip()
        if name == "/help":
            return self.help_text()
        if name == "/menu":
            return self.settings_text()
        if name == "/model":
            return self.switch_model(rest) if rest else self.models_text()
        return speaker.voice_command(self.voice, line, self.last_answer)
