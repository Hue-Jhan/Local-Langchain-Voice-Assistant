"""Speech output: the Piper engine, session state, and what to speak.

speak() blocks, speak_async() returns at once, and new speech cancels whatever is still playing.
Voice is deliberately not a tool - making the model copy its answer into a tool argument made it
speak the wrong text - so a policy decides instead.
"""

import re
import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from ..config import ROOT

VOICES_DIR = ROOT / "voices"
# voice_id -> (onnx file, length_scale, pitch) - same tuning as v1
VOICES = {
    0: ("it_IT-riccardo-x_low.onnx", 1.0, 1.0),
    1: ("en_US-ryan-medium.onnx", 0.80, 1.06),
    2: ("en_GB-jenny_dioco-medium.onnx", 1.0, 1.0),
}
VOICE_LABELS = {0: "Riccardo (Italian)", 1: "Ryan (US)", 2: "Jenny (UK)"}
DEFAULT_VOICE = 1
VOICE_FOR = {"it": 0, "en": 1}  # which voice reads which language when auto is on

# Word overlap, not a language library: answers are short, and a dependency to tell two known languages apart would
# cost more than it is worth. Ties go to English.
IT_WORDS = frozenset((
    "il lo la i gli le un uno una di del della a da in con su per tra fra che non e ed è "
    "sono ho hai ha abbiamo come cosa dove quando perche piu anche ma se cosi questo "
    "questa quello si no grazie prego ciao ore ora giorno oggi domani bene molto tutto "
    "niente essere fare dire vorrei puoi mi ti ci vi lui lei loro").split())
EN_WORDS = frozenset((
    "the a an of to in is are was were am i you he she it we they and or but not this "
    "that these those what where when why how have has had do does did can could will "
    "would should there here with for on at from about more very all no yes thanks "
    "hello today time day your my me").split())
ACCENTED = set("àèéìòùáíóúÀÈÉÌÒÙ")

_lock = threading.Lock()
_playing = None


def _spawn(text, voice_id):
    name, length, pitch = VOICES.get(voice_id, VOICES[DEFAULT_VOICE])
    model = VOICES_DIR / name
    config = Path(f"{model}.json")
    for path in (model, config):
        if not path.is_file():
            raise FileNotFoundError(f"Voice file missing: {path}")
    rate = json.loads(config.read_text())["audio"]["sample_rate"]
    try:
        piper = subprocess.Popen(
            ["piper-tts", "--model", str(model), "--output-raw",
             "--length_scale", str(length)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # asetrate shifts pitch, aresample restores the playback rate
        af = ["-af", f"asetrate={int(rate * pitch)},aresample={rate}"] if pitch != 1 else []
        ffplay = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-f", "s16le", "-ar", str(rate),
             "-ch_layout", "mono", *af, "-i", "pipe:0"],
            stdin=piper.stdout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise FileNotFoundError( f"{exc.filename} not found - install piper-tts and ffmpeg") from exc
    piper.stdout.close()
    # feed from a thread so long text cannot block the caller on a full pipe
    threading.Thread(target=_feed, args=(piper, text), daemon=True).start()
    return piper, ffplay


def _feed(piper, text):
    try:
        piper.stdin.write(text.encode())
        piper.stdin.close()
    except (BrokenPipeError, ValueError):
        pass  # cancelled


def _kill(procs):
    for proc in reversed(procs):
        if proc and proc.poll() is None:
            proc.terminate()
    for proc in procs:
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()


def stop():
    global _playing
    with _lock:
        procs, _playing = _playing, None
    if procs:
        _kill(procs)


def is_playing():
    with _lock:
        return bool(_playing) and _playing[1].poll() is None


def _start(text, voice_id):
    global _playing
    stop()
    procs = _spawn(text, voice_id)
    with _lock:
        _playing = procs
    return procs


def _release(procs):
    global _playing
    with _lock:
        if _playing is procs:
            _playing = None


def speak_async(text, voice_id=DEFAULT_VOICE):
    """Start speaking in the background, cancelling any current playback."""
    if not (text := text.strip()):
        return
    procs = _start(text, voice_id)

    def reap():
        procs[1].wait()
        procs[0].wait()
        _release(procs)

    threading.Thread(target=reap, daemon=True).start()


def speak(text, voice_id=DEFAULT_VOICE):
    """Speak and block until playback ends."""
    if not (text := text.strip()):
        return
    procs = _start(text, voice_id)
    try:
        procs[1].wait()
        procs[0].wait()
    finally:
        _release(procs)
        _kill(procs)


# --- session state and the "what to speak" policy ---------------------------------

def language(text):
    """Guess "it" or "en" from how many known words of each appear."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    italian = sum(w in IT_WORDS for w in words) + sum(c in ACCENTED for c in text)
    return "it" if italian > sum(w in EN_WORDS for w in words) else "en"


@dataclass
class VoiceState:
    enabled: bool = False
    voice_id: int = DEFAULT_VOICE
    auto: bool = True  # read each answer in the voice matching its language

    def voice_for(self, text, asked=""):
        """The voice this text should be read in. The question counts too, because a clock answer reads
        as English whatever language it was asked in.
        """
        if not self.auto:
            return self.voice_id
        return VOICE_FOR.get(language(f"{asked} {text}"), self.voice_id)

    @property
    def label(self):
        return VOICE_LABELS.get(self.voice_id, str(self.voice_id))

    @property
    def status(self):
        voice = "auto" if self.auto else self.label
        return f"voice {'on' if self.enabled else 'off'} ({voice})"


def manual(state, user_text, answer):
    """Speak the whole answer, and only while voice is switched on."""
    return answer.strip() or None if state.enabled else None


def never(state, user_text, answer):
    return None


# The seam. A policy takes (state, user_text, answer) and returns text to speak or None. Swap for a phrase matcher, a
# classifier, or a summarizer that shortens first.
POLICY = manual


def resolve(state, user_text, answer):
    return POLICY(state, user_text, answer)


def say(state, text, asked=""):
    """Start speaking in the voice matching the language. Returns at once."""
    speak_async(text, state.voice_for(text, asked))


def voice_command(state, line, last_answer):
    """Handle /v /voice /s /stop. Returns text to show, "" if silent, None if not ours."""
    name, _, rest = line.partition(" ")
    name, rest = name.lower(), rest.strip()
    if name == "/stop":
        stop()
        return "playback stopped"
    if name == "/s":
        if not (text := rest or last_answer):
            return "nothing to speak yet"
        say(state, text)
        return ""
    if name == "/v":
        state.enabled = not state.enabled
        if not state.enabled:
            stop()
        return state.status
    if name != "/voice":
        return None
    if not rest:
        return state.status
    if rest in {"on", "off"}:
        state.enabled = rest == "on"
        if not state.enabled:
            stop()
        return state.status
    if rest == "auto":
        state.auto = True
        return "voice picked from the language of each answer"
    if rest.isdigit() and int(rest) in VOICES:
        state.voice_id, state.auto = int(rest), False  # an explicit choice overrides auto
        return f"voice set to {state.label}"
    ids = ", ".join(f"{i} {VOICE_LABELS[i]}" for i in sorted(VOICES))
    return f"usage: /voice on|off|auto|<id>   ids: {ids}"


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    try:
        if argv and argv[0].isdigit():
            speak(" ".join(argv[1:]), int(argv[0]))
        elif argv:
            speak(" ".join(argv))
        else:
            speak("System initialized. Ready.")
    except (KeyboardInterrupt, EOFError):
        stop()
        sys.exit(130)
    except FileNotFoundError as exc:
        sys.exit(f"error: {exc}")
