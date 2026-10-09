"""Speech input. Capture is fixed; the gate and the transcriber are swappable.

A gate decides whether what you said was meant for the assistant, and yields plain text. Gates hold
state, which is what lets a wake word stay armed - and what a future speaker-authorization gate will
need.
"""

import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

from . import speaker
from ..config import ROOT

SetLogLevel(-1)  # vosk is noisy on stderr
SAMPLE_RATE, BLOCK = 16000, 8000

# resolved when a model is actually needed, so config.toml can move the directory
MODELS_DIR = ROOT / "models"
MODEL_NAMES = {
    "it": ["vosk-model-small-it-0.22", "vosk-model-it-0.22"],
    "en": ["vosk-model-en-us-0.22-lgraph", "vosk-model-small-en-us-0.15"],
}

# Measured, not guessed - see `tester.py ear`. "assistente" scores 8/8 where "ciao robot" manages
# 5/8, and a wake word run into a command collapses ("hey assist in turn on the light"), so the
# collapsed forms are listed literally, longest first so the command splits off cleanly.
WAKE = {
    "it": ("ciao assistente",
           ("ciao assistente", "ehi assistente", "senti assistente",
            "assistente", "assistent", "ciao robot", "ai robot", "senti robot",
            "robot", "assist")),
    "en": ("hey assistant",
           ("hey assistant", "hello assistant", "ok assistant", "assistance",
            "assist in", "assistant", "hey computer", "computer", "assist")),
}
CLOSING = {"it": ("basta", "chiudi", "riposo", "stop"), "en": ("stop", "enough", "dismissed", "never mind")}
IDLE_TIMEOUT = 15.0


def model_candidates(language):
    """A model living elsewhere is symlinked into models/, like voices/ is."""
    return [MODELS_DIR / name for name in MODEL_NAMES.get(language, [])]


def find_model(explicit=None, language="it"):
    """Locate a Vosk model: explicit path, then $VOSK_MODEL, then the known places."""
    known = model_candidates(language)
    for candidate in [explicit, os.environ.get("VOSK_MODEL"), *known]:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    tried = ", ".join(str(p) for p in known)
    raise FileNotFoundError(f"No Vosk model for {language!r}. Tried: {tried}")


class VoskTranscriber:
    """Offline speech to text, in batch or streaming mode."""

    def __init__(self, model_path=None, language="it"):
        self.language = language
        self.path = find_model(model_path, language)
        self.model = Model(str(self.path))

    def _recognizer(self):
        rec = KaldiRecognizer(self.model, SAMPLE_RATE)
        rec.SetWords(False)
        return rec

    def transcribe(self, frames):
        rec = self._recognizer()
        rec.AcceptWaveform(frames)
        return json.loads(rec.FinalResult()).get("text", "").strip()

    def stream(self):
        """Returns feed(block) -> (final, partial). Vosk finalises on silence."""
        rec = self._recognizer()

        def feed(block):
            if rec.AcceptWaveform(block):
                return json.loads(rec.Result()).get("text", "").strip(), ""
            return "", json.loads(rec.PartialResult()).get("partial", "").strip()

        return feed


def _mic(q, device=None):
    def callback(indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(bytes(indata))

    return sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=BLOCK, dtype="int16", channels=1, callback=callback, device=device)


class PushToTalk:
    """Record between two Enter presses. Nothing is picked up by accident, which makes it the safe default."""

    name = "push"

    def __init__(self, transcriber, device=None):
        self.transcriber, self.device = transcriber, device

    def listen_once(self, stop_event=None, prompt=None):
        """Record until Enter (or stop_event is set), then transcribe."""
        speaker.stop()  # never record the assistant's own voice
        frames, q = bytearray(), queue.Queue()
        done = stop_event or threading.Event()
        with _mic(q, self.device):
            if stop_event is None:
                if prompt:
                    print(prompt, end="", flush=True)
                threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
            while not done.is_set():
                try:
                    frames += q.get(timeout=0.1)
                except queue.Empty:
                    pass
        return self.transcriber.transcribe(bytes(frames))

    def utterances(self):
        while True:
            try:
                input("[Enter] to talk, Ctrl+C to quit")
            except (KeyboardInterrupt, EOFError):
                return
            if text := self.listen_once(prompt="recording... [Enter] to stop"):
                yield text


class WakeWord:
    """Always-on listening armed by a wake word - the listener5 behaviour. Stays awake until a closing
    word or IDLE_TIMEOUT of silence.
    """

    name = "wake"

    def __init__(self, transcriber, wake=None, device=None, idle_timeout=IDLE_TIMEOUT, variants=None):
        language = getattr(transcriber, "language", "it")
        default_wake, default_variants = WAKE.get(language, WAKE["it"])
        self.transcriber, self.device = transcriber, device
        self.wake = (wake or default_wake).lower()
        self.idle_timeout = idle_timeout
        self.closing = CLOSING.get(language, CLOSING["it"])
        # recognition is imperfect, so accept the near misses it really makes
        self.variants = tuple(dict.fromkeys((self.wake, *(variants or default_variants))))

    def matched(self, text):
        """Return the wake variant heard in text, if any."""
        return next((v for v in self.variants if v in text), None)

    def utterances(self):
        feed = self.transcriber.stream()
        q, awake_until = queue.Queue(), 0.0
        standby = f"standby - say '{self.wake}' to wake me"
        with _mic(q, self.device):
            print(f"idle - say '{self.wake}' to wake me")
            while True:
                try:
                    block = q.get(timeout=0.5)
                except queue.Empty:
                    if awake_until and time.monotonic() > awake_until:
                        awake_until = 0.0
                        print(f"\n{standby}")
                    continue
                if speaker.is_playing():
                    continue  # do not transcribe our own output
                final, partial = feed(block)
                if partial:
                    sys.stdout.write(f"\r  ...{partial}\033[K")
                    sys.stdout.flush()
                if not final:
                    continue
                sys.stdout.write("\r\033[K")

                if not (awake_until and time.monotonic() <= awake_until):
                    if heard := self.matched(final):
                        awake_until = time.monotonic() + self.idle_timeout
                        print("awake")
                        if rest := final.split(heard, 1)[-1].strip():
                            yield rest  # a command spoken in the same breath
                    continue
                if any(w in final for w in self.closing):
                    awake_until = 0.0
                    print(standby)
                    continue
                yield final
                awake_until = time.monotonic() + self.idle_timeout


GATES = {"push": PushToTalk, "wake": WakeWord}


def build_gate(kind="push", model_path=None, device=None, language="it"):
    """Create a gate. 'push' is the default, 'wake' is the always-on wake word."""
    if kind not in GATES:
        raise ValueError(f"unknown gate {kind!r}; choose from {', '.join(GATES)}")
    if device is not None and int(device) < 0:
        device = None
    return GATES[kind](VoskTranscriber(model_path, language), device=device)
