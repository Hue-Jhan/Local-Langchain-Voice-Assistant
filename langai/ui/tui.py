"""Textual TUI: scrolling chat, live streaming, an activity line, and key bindings."""

import threading

from rich.markup import escape
from textual import work
from textual.app import App
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Static

from ..app import Session
from ..speech import speaker

SPIN = "|/-\\"
WAVES = ("<)", "<))", "<)))", "<))")
TOOL_NAMES = {"search_web": "search tool", "now": "clock tool"}


class LangaiApp(App):
    """The default frontend. `-cli` selects the plain REPL instead."""

    CSS = """
    Screen { layout: vertical; }
    #chat { height: 1fr; padding: 0 1; }
    #activity { height: 1; padding: 0 1; color: $text-muted; }
    #status { height: 1; background: $panel; color: $text-muted; padding: 0 1; }
    Input { border: none; }
    .dim { color: $text-muted; }
    .err { color: $error; }
    """
    # priority so Ctrl+C reaches us instead of killing the app mid-answer
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "hush", "Hush", priority=True),
        Binding("f2", "toggle_voice", "Voice"),
        Binding("f3", "talk", "Talk"),
        Binding("ctrl+l", "clear", "Clear"),
    ]

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.verbose = cfg["ui"]["verbose"]
        self.session = None
        self._answer = None
        self._buffer = ""
        self._recording = None
        self._cancel = threading.Event()
        self._busy = "booting up model"
        self._frame = 0
        self._last_activity = None

    def compose(self):
        yield VerticalScroll(id="chat")
        yield Static("", id="activity")
        yield Static("", id="status")
        yield Input(placeholder="message, or /help", id="prompt")
        yield Footer()

    # --- chat pane ----------------------------------------------------------------

    def _at_bottom(self):
        chat = self.query_one("#chat", VerticalScroll)
        return chat.scroll_offset.y >= chat.max_scroll_y - 2

    def add_line(self, css, text):
        """Mount a line. Never yanks the view down if the user scrolled up to read."""
        chat = self.query_one("#chat", VerticalScroll)
        follow = self._at_bottom()
        line = Static(text, classes=css)
        chat.mount(line)
        if follow:
            chat.scroll_end(animate=False)
        return line

    def reply(self, text):
        """Command output, with a blank line so successive outputs stay separated."""
        self.add_line("dim", escape(text))
        self.add_line("dim", "")

    # --- the activity line --------------------------------------------------------

    def _tick(self):
        parts = []
        self._frame = (self._frame + 1) % 4
        if self._busy:
            parts.append(f"{SPIN[self._frame]} {self._busy}")
        if speaker.is_playing():
            parts.append(f"[cyan]{WAVES[self._frame]}[/cyan] speaking")
        text = "   ".join(parts)
        if text != self._last_activity:  # only touch the widget when it changed
            self.query_one("#activity", Static).update(text)
            self._last_activity = text

    def refresh_status(self):
        if not self.session:
            return
        s = self.session
        ear = f"{s.gate.name} gate" if s.gate else "off"
        router = "on" if self.cfg["router"]["enabled"] else "off"
        self.query_one("#status", Static).update(
            f" [green]{s.model}[/green] · {s.voice.status} · ear {ear} · "
            f"router {router} · thread {s.thread}")

    def on_mount(self):
        self.query_one("#prompt", Input).focus()
        self.set_interval(0.12, self._tick)
        self.boot()

    @work(thread=True)
    def boot(self):
        """Building the agent touches Ollama, so keep it off the UI thread."""
        try:
            session = Session(self.cfg)
            session.warm()
            ear_line = session.open_ear() if self.cfg["ear"]["enabled"] else ""
        except Exception as exc:
            self.call_from_thread(self._failed, str(exc))
            return
        self.call_from_thread(self._session_ready, session, ear_line)

    def _failed(self, message):
        self._busy = ""
        self.add_line("err", f"startup failed: {escape(message)}")

    def _session_ready(self, session, ear_line):
        self.session = session
        self._busy = ""
        if session.note:
            self.add_line("dim", f"note: {escape(session.note)}")
        if ear_line:
            self.add_line("dim", escape(ear_line))
        if session.gate is not None and session.gate.name == "wake":
            self.add_line("dim", "wake gate runs in -cli mode; here press F3 to talk")
        self.add_line("dim", "/help for commands · F3 talk · F2 voice · Ctrl+C hush " "· Ctrl+Q quit")
        self.add_line("dim", "")
        self.refresh_status()

    # --- a turn ------------------------------------------------------------------

    def on_input_submitted(self, event):
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text in {"/exit", "/quit"}:
            self.exit()
            return
        if not self.session:
            self.reply("still loading...")
            return
        if text.startswith("/"):
            answer = self.session.command(text)
            if answer is None:
                answer = self.session.help_text()
            if answer:
                self.reply(answer)
            self.refresh_status()
            return
        self.add_line("", f"[cyan]You:[/cyan] {escape(text)}")
        self.turn(text)

    @work(thread=True, exclusive=True)
    def turn(self, text):
        self._answer, self._buffer = None, ""
        self._cancel.clear()
        self._busy = "thinking"
        try:
            for kind, value in self.session.run(text):
                if self._cancel.is_set():
                    self.call_from_thread(self.add_line, "dim", "[cancelled]")
                    break
                if kind == "token":
                    self._buffer += value
                    self.call_from_thread(self._show, self._buffer)
                elif kind == "tool":
                    self._busy = self._tool_status(value)
                    # always shown: you asked to be able to see tool use
                    line = value if self.verbose else " ".join(value.split())[:88]
                    self.call_from_thread(self.add_line, "dim", f"  {escape(line)}")
                else:
                    if not self._buffer.strip():
                        self.call_from_thread(self._show, "(no output)")
                    self.call_from_thread(self.add_line, "dim", escape(str(value)))
                    self.call_from_thread(self.add_line, "dim", "")
        except Exception as exc:
            self.call_from_thread(self.add_line, "err", f"error: {escape(str(exc))}")
            return
        finally:
            self._busy = ""
        if problem := self.session.maybe_speak(text):
            self.call_from_thread(self.add_line, "err", escape(problem))

    @staticmethod
    def _tool_status(event):
        """Turn a tool event into something worth showing on the activity line."""
        if event.startswith("[router]"):
            if " -> " in event:  # the router is running a tool itself
                name = event.rsplit(" -> ", 1)[1].strip()
                return f"calling {TOOL_NAMES.get(name, name)}"
            return "thinking"
        if event.startswith("-> "):
            name = event[3:].split("(", 1)[0]
            return f"calling {TOOL_NAMES.get(name, name)}"
        return "reading the result"

    def _show(self, buffer):
        follow = self._at_bottom()
        if self._answer is None:
            self._answer = self.add_line("", "")
        self._answer.update(f"[green]{self.session.alias}:[/green] {escape(buffer)}")
        if follow:
            self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

    # --- key actions -------------------------------------------------------------

    def action_hush(self):
        """Ctrl+C stops what is happening, Ctrl+Q quits. A worker blocked on an HTTP call cannot be
        killed, so the stream is abandoned rather than interrupted.
        """
        did = []
        if speaker.is_playing():
            speaker.stop()
            did.append("playback")
        if self._recording:
            self._recording.set()
            self._recording = None
            self._busy = ""
            did.append("recording")
        if self._busy:
            self._cancel.set()
            did.append("answer")
        self.reply(f"stopped {', '.join(did)}" if did else "nothing to stop (Ctrl+Q quits)")

    def action_toggle_voice(self):
        if self.session:
            self.reply(self.session.command("/v"))
            self.refresh_status()

    def action_clear(self):
        self.query_one("#chat", VerticalScroll).remove_children()

    def action_talk(self):
        """F3 starts recording, F3 again stops it: a terminal cannot see a key being released. The
        transcript lands in the input box so it can be edited before sending.
        """
        if not self.session or not self.session.gate:
            self.reply("ear is off - start with -ear")
            return
        if self._recording:
            self._recording.set()
            self._recording = None
            return
        self._recording = threading.Event()
        self._busy = "listening - F3 to stop"
        self.record(self._recording)

    @work(thread=True)
    def record(self, stop_event):
        try:
            heard = self.session.gate.listen_once(stop_event=stop_event)
        except Exception as exc:
            self.call_from_thread(self.add_line, "err", f"mic error: {escape(str(exc))}")
            return
        finally:
            self._busy = ""
        if not heard:
            self.call_from_thread(self.reply, "(heard nothing)")
            return
        self.call_from_thread(self._prefill, heard)

    def _prefill(self, heard):
        """Put the transcript in the box so it can be checked or edited, then Enter."""
        box = self.query_one("#prompt", Input)
        box.value = heard
        box.cursor_position = len(heard)
        box.focus()


def run(cfg):
    LangaiApp(cfg).run()
    speaker.stop()
