"""Terminal helpers for the plain CLI: colors, a spinner, and <think> stripping."""

import itertools
import sys
import threading

CYAN, GREEN, RED, DIM, BOLD, RESET = ( "\033[96m", "\033[92m", "\033[91m", "\033[2m", "\033[1m", "\033[0m")
FRAMES, INTERVAL = "|/-\\", 0.08


class Spinner:
    """One spinning character, erased completely on stop so only the answer remains."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.enabled = self.stream.isatty()  # piped output must stay clean
        self._stop, self._thread = threading.Event(), None

    def start(self):
        if not self.enabled or self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        for frame in itertools.cycle(FRAMES):
            if self._stop.is_set():
                return
            self.stream.write(f"\r{DIM}{frame}{RESET}")
            self.stream.flush()
            self._stop.wait(INTERVAL)

    def stop(self):
        if not self._thread:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        self.stream.write("\r \r")
        self.stream.flush()


class ThinkFilter:
    """Strips <think>...</think> blocks from streamed text, across chunk boundaries."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self._inside, self._held = False, ""

    @staticmethod
    def _split(text, tag):
        """Hold back a trailing fragment that might be the start of tag."""
        for size in range(len(tag) - 1, 0, -1):
            if text.endswith(tag[:size]):
                return text[:-size], text[-size:]
        return text, ""

    def feed(self, text):
        self._held += text
        out = []
        while self._held:
            if self._inside:
                at = self._held.find(self.CLOSE)
                if at == -1:
                    _, self._held = self._split(self._held, self.CLOSE)
                    break
                self._held = self._held[at + len(self.CLOSE):]
                self._inside = False
                continue
            opens, closes = self._held.find(self.OPEN), self._held.find(self.CLOSE)
            # a stray close tag means reasoning began before streaming - drop it
            if closes != -1 and (opens == -1 or closes < opens):
                out.append(self._held[:closes])
                self._held = self._held[closes + len(self.CLOSE):]
                continue
            if opens == -1:
                # hold back a fragment that could begin either tag
                emit_o, hold_o = self._split(self._held, self.OPEN)
                emit_c, hold_c = self._split(self._held, self.CLOSE)
                longer = len(hold_c) > len(hold_o)
                out.append(emit_c if longer else emit_o)
                self._held = hold_c if longer else hold_o
                break
            out.append(self._held[:opens])
            self._held = self._held[opens + len(self.OPEN):]
            self._inside = True
        return "".join(out)

    def flush(self):
        rest = "" if self._inside else self._held
        self._held = ""
        return rest
