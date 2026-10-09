"""LangGraph agent: Ollama model, tool loop, conversation memory."""

import sqlite3

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .config import ROOT
from .tools import TOOLS

SYSTEM_PROMPT = SystemMessage(
    "Never act like a polite AI or offer help. "
    "Reply with short, blunt, sarcastic remarks, max 2 sentences unless asked to. "
    "Always answer in the language the user wrote in."
)


class Window:
    """Which part of the thread is sent to the model, anchored rather than sliding.

    A last-N window moved its first message every turn, which invalidated ollama's prompt cache
    every turn: prefix reuse 16/16 -> 1/16, prompt eval 0.24s -> 3.4s, permanently. So it anchors
    once and only grows, which is append-only and cheap. Re-anchoring reads the whole window again,
    so it waits for `limit` or for `due` - a turn already paying for a long prompt. See CLAUDE.md.
    """

    def __init__(self, keep):
        self.keep = keep
        self.limit = keep * 3
        self.anchor = None

    def of(self, messages, due=False):
        if not self.keep:
            return list(messages)
        size = len(messages) - (self.anchor or 0)
        if self.anchor is None or size > self.limit or (due and size > self.keep):
            self.anchor = self._start(messages, max(0, len(messages) - self.keep))
        return list(messages[self.anchor:])

    @staticmethod
    def _start(messages, first):
        """A window opens on a user message, so a tool result is never sent without the call it answers
        - the model rejects that.
        """
        for index in range(first, len(messages)):
            if isinstance(messages[index], HumanMessage):
                return index
        return first


def checkpointer(cfg):
    """Sqlite keeps conversations across restarts; memory forgets on exit."""
    settings = cfg["memory"]
    if not settings.get("persist"):
        return MemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver
    path = ROOT / settings["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def build_agent(model_name, cfg):
    """Compile the agent graph for an Ollama model. Returns the graph and a `warm(messages)` that
    pre-reads the thread into ollama's prompt cache.
    """
    settings = cfg["model"]
    options = {"model": model_name, "temperature": settings["temperature"], "num_predict": settings["max_tokens"] or None}
    if settings.get("keep_alive"):
        options["keep_alive"] = settings["keep_alive"]
    plain = ChatOllama(**options)
    # same prompt, no answer: used to fill the cache before the first question
    warmer = ChatOllama(**{**options, "num_predict": 1})
    window = Window(cfg["memory"].get("window", 0))
    bound = {}

    def warm(messages):
        """Fill ollama's prompt cache before the first question, so it is not the slow one: 3.9s
        against 1.0s. Shares `window`, so this is exactly the prefix the first turn will send.
        """
        warmer.invoke([SYSTEM_PROMPT] + window.of(list(messages)))

    def pick(names):
        """The model with just these tools bound. None = all, () = none."""
        if names is not None and not names:
            return plain
        key = None if names is None else tuple(sorted(names))
        if key not in bound:
            chosen = TOOLS if key is None else [t for t in TOOLS if t.name in key]
            bound[key] = plain.bind_tools(chosen)
        return bound[key]

    def call_model(state: MessagesState, config):
        # the router narrows this per turn; unset means every tool is offered
        settings = config.get("configurable", {})
        names = settings.get("tools")
        context = settings.get("context")
        prompt = [SYSTEM_PROMPT] + window.of(state["messages"], due=bool(context))
        # Router-fetched results: used for this answer only, never written to the thread. They go in a
        # user turn before the question because as a system message the model ignored them.
        if context and prompt:
            prompt.insert(len(prompt) - 1, HumanMessage(context))
        return {"messages": [pick(names).invoke(prompt)]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    # tools_condition routes to "tools" on a tool call, otherwise to END
    builder.add_conditional_edges("agent", tools_condition)
    # back to the agent so it synthesizes tool output instead of ending on it
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer(cfg)), warm
