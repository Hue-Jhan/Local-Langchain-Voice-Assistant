"""Test harness.

  python tester.py logic                      deterministic checks, no LLM needed
  python tester.py ear [it|en]                 speech round trip, no microphone needed
  python tester.py tui                         boot the TUI headless and drive it
  python tester.py models [alias...] [-n N]    tool-routing benchmark
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

VOICE_MODELS = {"it": ("it_IT-riccardo-x_low.onnx", 16000), "en": ("en_US-ryan-medium.onnx", 22050)}


class Checks:
    """check(label, got, want=True) - prints a line and tracks failures."""

    def __init__(self):
        self.failed = self.total = 0

    def __call__(self, label, got, want=True):
        self.total += 1
        ok = got == want
        self.failed += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
        if not ok:
            print(f"        got  {got!r}\n        want {want!r}")

    def report(self):
        print(f"\n{self.total - self.failed}/{self.total} passed")
        return 1 if self.failed else 0


def test_logic():
    from langai import config, router, tools
    from langai.speech import speaker
    from langai.tools import Result, numbered_text
    from langai.speech.speaker import (VOICES, VOICES_DIR, VoiceState, speak, voice_command)
    from langai.tools import TOOLS, now, search_web, set_turn
    from langai.ui.term import ThinkFilter

    check = Checks()

    def filtered(chunks):
        f = ThinkFilter()
        return "".join(f.feed(c) for c in chunks) + f.flush()

    print("ThinkFilter")
    for chunks, want, label in [
        (["hello"], "hello", "plain text"),
        (["<think>x</think>answer"], "answer", "block stripped"),
        (["<th", "ink>x</th", "ink>answer"], "answer", "tags split across chunks"),
        (["a<think>x</think>b<think>y</think>c"], "abc", "two blocks"),
        (["<think>", "reasoning"], "", "unterminated block"),
        (["</think>answer"], "answer", "stray close tag dropped"),
        (["a<", "b"], "a<b", "lone < is content"),
        (["1<thi"], "1<thi", "partial tag at EOF is content"),
    ]:
        check(label, filtered(chunks), want)

    print("\nconfig")
    cfg = config.load()
    check("has every section", sorted(cfg) == sorted(config.DEFAULTS))
    check("tui is the default mode", cfg["ui"]["mode"], "tui")
    check("unreadable file falls back to defaults", config.load("/nonexistent/x.toml")["model"]["alias"], config.DEFAULTS["model"]["alias"])

    class Args:
        model, cli, v = "lama3", True, True
    flagged = config.apply_flags(config.load(), Args())
    check("flags override the file", flagged["model"]["alias"], "lama3")
    check("-cli switches mode", flagged["ui"]["mode"], "cli")

    print("\nsearch formatting")
    out = numbered_text([Result("One", "https://www.example.com/a", " snip  one "), Result("Two", "https://kernel.org/b", "snip two")])
    check("numbers results", out.startswith("[1] One"))
    check("strips www", "(example.com)" in out)
    check("collapses whitespace", "snip one" in out)
    check("includes second result", "[2] Two" in out)
    check("empty results", numbered_text([]), "No results.")

    print("\ntools")
    check("registered tools", sorted(t.name for t in TOOLS), ["now", "search_web"])
    check("search takes one flat string", search_web.args, {"query": {"title": "Query", "type": "string"}})
    check("now takes no arguments", now.args, {})
    check("now returns a year", str(time.localtime().tm_year) in now.invoke({}))

    print("\ncontentless-message guard")
    for turn, blocked, label in [
        ("ok", True, "english filler"),
        ("pronto", True, "italian filler"),
        ("si", True, "one-word ack"),
        ("va bene", True, "two filler words"),
        ("", False, "empty turn is not judged"),
        ("meteo roma", False, "short but real request"),
        ("ciao assistente", False, "wake phrase carries a content word"),
        ("what is the latest linux kernel?", False, "real question"),
        ("chi ha vinto il gran premio ieri", False, "long question, other language"),
    ]:
        check(f"{label}: {turn!r}", tools.contentless(turn), blocked)
    set_turn("ok")
    check("guard blocks the search", "skipped" in search_web.invoke({"query": "news"}))
    set_turn("")

    print("\nrouter: no tool schemas are bound by default")
    # binding them on one turn and not the next invalidates ollama's prompt cache, which measured 4.5s a turn against
    # 1.2s when nothing is ever bound
    for text, label in [("ok", "filler"), ("whats 2*3?", "arithmetic"),
                        ("tell me about rust", "ordinary question"),
                        ("btc price now", "live data"),
                        ("cerca il prezzo", "explicit search"),
                        ("what time is it", "clock")]:
        check(f"{label}: {text!r} binds nothing", router.decide(text).tools, ())
    check("tools mode 'model' binds everything", router.ALL.tools, None)

    print("\nrouter: tools we run ourselves")
    for text, tool, label in [
        ("cerca il prezzo del btc", "search_web", "italian search verb"),
        ("look up btc price", "search_web", "english search verb"),
        ("btc price now", "search_web", "live data runs a search"),
        ("whats bitcoin price now?", "search_web", "live data, question form"),
        ("what time is it", "now", "clock question"),
        ("che ore sono", "now", "clock question, italian"),
        ("aight bet, now whats the time", "now", "clock question buried in chatter"),
        ("whats the time", "now", "'whats the' phrasing"),
        ("what day is it today", "now", "day question"),
        ("the full date", "now", "follow-up asking for the whole date"),
        ("the best day of my life", "", "'day' in passing is not a question"),
        ("tell me about rust", "", "ordinary question runs nothing"),
        ("ok", "", "filler runs nothing"),
    ]:
        check(f"{label}: {text!r}", router.decide(text).prefetch, tool)

    print("\nrouter: told not to search")
    # "dont use the search tool" ran a search for "dont use the tool" and came back with printable do-not-use signs
    for text, label in [
        ("dont use the search tool", "english refusal"),
        ("don't search for that", "contracted refusal"),
        ("no need to look it up", "polite refusal"),
        ("stop searching", "an order to stop"),
        ("non cercare su google", "italian refusal"),
        ("senza cercare, che ne pensi?", "italian 'without searching'"),
        ("answer without the web", "refusing the web"),
    ]:
        decision = router.decide(text, "tell me btc price now")
        check(f"{label}: {text!r} runs nothing", decision.prefetch, "")
    for text, label in [("search for the btc price", "a plain request still searches"),
                        ("dont be lazy, look up the btc price", "refusal of something else"),
                        ("whats the weather, dont tell me a joke", "unrelated 'dont'")]:
        check(f"{label}: {text!r}", router.decide(text).prefetch, "search_web")

    print("\nrouter: answers that skip the model entirely")
    for text, want, label in [
        ("che ore sono", True, "bare clock question is answered from the tool"),
        ("what day is it", True, "bare clock question, english"),
        ("what day is it today", True, "a longer bare question is still answered direct"),
        ("aight bet, now whats the time", True, "chatter around it does not matter"),
        ("the full date", True, "the whole date comes from the tool, not a summary"),
        ("what time is it at the moment", True, "'at the moment' is not a place"),
        ("what time is it in tokyo", False, "a qualified question still needs the model"),
        ("che ore sono a tokyo", False, "a place, in italian"),
        ("btc price now", False, "a search needs the model to summarise"),
    ]:
        check(f"{label}: {text!r}", router.decide(text).direct, want)

    print("\nrouter: query extraction")
    previous = "tell me btc price now"
    for text, want, label in [
        ("cerca il prezzo del btc", "il prezzo del btc", "trigger word stripped"),
        ("look up the weather in rome", "the weather in rome", "subject kept"),
        ("look it up godamnit", "btc price now", "'it' falls back to the last question"),
        ("cercalo su google", "btc price now", "'-lo' falls back too"),
    ]:
        check(f"{label}: {text!r}", router.decide(text, previous).query, want)
    check("rules are swappable", router.decide("ok", rules=[]).rule, "default")

    print("\nprompt window keeps ollama's prompt cache")
    # a last-N window moved its first message every turn, which re-read the whole prompt every
    # turn: prefix reuse 1 of 16, prompt eval 0.24s -> 3.4s. It is anchored instead.
    from langchain_core.messages import AIMessage, HumanMessage

    from langai.agent import Window
    history, firsts, sizes = [], [], []
    window = Window(16)
    for turn in range(40):
        history += [HumanMessage(f"question {turn}"), AIMessage(f"answer {turn}")]
        sent = window.of(history)
        firsts.append(str(sent[0].content) if sent else "")
        sizes.append(len(sent))
    moves = sum(a != b for a, b in zip(firsts, firsts[1:]))
    check(f"window start moved {moves} times in 40 turns, not 40", moves <= 6)
    check(f"window stays bounded (max {max(sizes)} messages)", max(sizes) <= 49)
    check("window keeps recent turns", len(window.of(history)) >= 16)
    check("window starts on a user message", isinstance(window.of(history)[0], HumanMessage))
    sliding = [str(history[max(0, n * 2 + 2 - 16)].content) for n in range(40)]
    check("the old sliding window really did move every turn", sum(a != b for a, b in zip(sliding, sliding[1:])), 32)
    check("window 0 sends the whole thread", len(Window(0).of(history)), len(history))
    check("empty thread is fine", Window(16).of([]), [])
    due = Window(16)
    due.of(history[:40])
    before = due.anchor
    check("a tool turn absorbs the re-anchor", len(due.of(history, due=True)), 16)
    check("the re-anchor moved the window", due.anchor != before)

    print("\nvoice policy")
    state = VoiceState()
    check("silent while off", speaker.resolve(state, "hi", "answer"), None)
    state.enabled = True
    check("speaks the answer when on", speaker.resolve(state, "hi", "answer"), "answer")
    check("blank answer not spoken", speaker.resolve(state, "hi", "   "), None)
    check("never policy", speaker.never(state, "hi", "answer"), None)

    print("\nspoken language picks the voice")
    for text, want, label in [
        ("ciao, come stai? che ore sono", "it", "italian question"),
        ("il prezzo del bitcoin oggi e molto alto", "it", "italian sentence"),
        ("perche non funziona", "it", "accent-free italian"),
        ("perché non funziona", "it", "accented italian"),
        ("hey, what time is it today", "en", "english question"),
        ("Rust is a systems programming language", "en", "english sentence"),
        ("ok", "en", "nothing to go on falls back to english"),
    ]:
        check(f"{label}: {text[:34]!r}", speaker.language(text), want)
    auto = VoiceState(True)
    check("italian answer gets Riccardo", auto.voice_for("ciao come stai"), 0)
    check("english answer gets Ryan", auto.voice_for("hello how are you"), 1)
    fixed = VoiceState(True, 2, auto=False)
    check("a fixed voice ignores the language", fixed.voice_for("ciao come stai"), 2)
    check("the question decides when the answer is a bare date",
          auto.voice_for("Friday 09 October 2026, 12:37 CEST", asked="che ore sono"), 0)
    check("auto shows as auto", "auto" in auto.status)

    print("\nvoice commands")
    state = VoiceState()
    check("non-command ignored", voice_command(state, "hello there", ""), None)
    check("/v toggles on", "on" in (voice_command(state, "/v", "") or ""))
    check("/v set the state", state.enabled, True)
    check("/v toggles off", "off" in (voice_command(state, "/v", "") or ""))
    voice_command(state, "/voice 2", "")
    check("/voice <id> sets voice", state.voice_id, 2)
    check("/voice <id> turns auto off", state.auto, False)
    check("/voice auto turns it back on", "language" in (voice_command(state, "/voice auto", "") or ""))
    check("/voice auto set the state", state.auto, True)
    check("/voice bad id shows usage", "usage" in (voice_command(state, "/voice 9", "") or ""))
    check("/s with nothing buffered", voice_command(state, "/s", ""), "nothing to speak yet")
    spoken, real_say = [], speaker.say
    speaker.say = lambda st, text: spoken.append(text)
    try:
        check("/s returns handled-sentinel", voice_command(state, "/s", "buffered"), "")
        check("/s speaks the buffer", spoken, ["buffered"])
    finally:
        speaker.say = real_say

    print("\nspeaker guards")
    check("empty text is a no-op", speak("   "), None)
    real_voice = VOICES[1]
    VOICES[1] = ("missing.onnx", 1.0, 1.0)
    try:
        speak("x", 1)
        check("missing voice raises", "no error", "FileNotFoundError")
    except FileNotFoundError:
        check("missing voice raises", True)
    finally:
        VOICES[1] = real_voice
    # length_scale is yours to tune (you set Ryan to 0.80), so bound-check it rather than pin it; the pitch ratio and
    # the file names do get pinned
    for vid, (name, length, pitch) in VOICES.items():
        check(f"voice {vid} file exists", (VOICES_DIR / name).is_file())
        check(f"voice {vid} length_scale {length} is sane", 0.5 <= length <= 1.5)
    check("ryan keeps his pitch lift", VOICES[1][2], 1.06)
    check("italian is unpitched", VOICES[0][1:], (1.0, 1.0))

    print("\nsession wiring")
    from langai.app import MODEL_MAP, Session
    check("default alias is known", config.DEFAULTS["model"]["alias"] in MODEL_MAP)
    check("help lists /model", "/model" in Session.help_text())
    return check.report()


def test_ear(language="it"):
    """Round-trip Piper speech back through Vosk. No microphone required."""
    from langai.speech import ear
    from langai.speech.speaker import VOICES_DIR

    check = Checks()
    name, rate = VOICE_MODELS[language]
    silence = b"\x00" * 48000  # 1.5s; vosk only finalises once silence fills blocks

    def say(phrase):
        """Synthesize a phrase at 16 kHz. Raises if the toolchain produced no audio - otherwise an
        ffmpeg or piper hiccup silently looks like a recognition miss.
        """
        done = subprocess.run( ["piper-tts", "--model", str(VOICES_DIR / name), "--output-raw"], input=phrase.encode(), capture_output=True)
        raw = done.stdout
        if not raw:
            raise RuntimeError(f"piper produced no audio for {phrase!r}: " f"{done.stderr.decode('utf-8', 'replace')[:200]}")
        if rate != ear.SAMPLE_RATE:  # vosk needs 16 kHz, so resample
            done = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le",
                 "-ar", str(rate), "-ac", "1", "-i", "pipe:0", "-f", "s16le",
                 "-ar", str(ear.SAMPLE_RATE), "-ac", "1", "pipe:1"],
                input=raw, capture_output=True)
            raw = done.stdout
            if not raw:
                raise RuntimeError(f"ffmpeg resample failed for {phrase!r}: " f"{done.stderr.decode('utf-8', 'replace')[:200]}")
        return raw + silence

    try:
        stt = ear.VoskTranscriber(language=language)
    except FileNotFoundError as exc:
        print(f"  [FAIL] no vosk model: {exc}")
        return 1
    print(f"language {language}, model {stt.path.name}, tts {name}\n\ngates")
    check("registry", sorted(ear.GATES), ["push", "wake"])
    check("push is the default", ear.build_gate("push", str(stt.path)).name, "push")
    try:
        ear.build_gate("nope")
        check("unknown gate rejected", "accepted", "ValueError")
    except ValueError:
        check("unknown gate rejected", True)

    # these phrases check that the pipeline works, not how good the model is, so they are ones that measured stable.
    # "che tempo fa oggi a roma" was dropped: it comes back as "che tempo fara oggi roma" often enough to make the
    # suite flaky.
    phrases = {"it": ["accendi la luce in cucina", "spegni tutto"],
               "en": ["turn on the kitchen light", "what is the weather today"]}[language]
    # a recogniser is statistical, so require most words back, not all of them
    print("\nbatch transcription (push-to-talk path, >=80% of words)")
    for phrase in phrases:
        heard = stt.transcribe(say(phrase))
        words = phrase.split()
        ratio = len(set(words) & set(heard.split())) / len(words)
        check(f"{phrase!r} -> {heard!r} ({ratio:.0%})", ratio >= 0.8)

    print("\nstreaming transcription (wake gate path)")
    feed, raw, finals = stt.stream(), say(phrases[0]), []
    for i in range(0, len(raw), ear.BLOCK):
        final, _partial = feed(raw[i:i + ear.BLOCK])
        if final:
            finals.append(final)
    joined = " ".join(finals)
    ratio = len(set(phrases[0].split()) & set(joined.split())) / len(phrases[0].split())
    check(f"stream produced {finals!r} ({ratio:.0%})", bool(finals) and ratio >= 0.8)

    print("\nwake word")
    gate = ear.WakeWord(stt)
    wake = gate.wake
    # spoken alone this is a short, quiet clip and the recogniser is statistical, so measure reliability over a few
    # attempts instead of demanding a perfect hit
    hits = sum(bool(gate.matched(stt.transcribe(say(wake)))) for _ in range(3))
    check(f"wake word {wake!r} recognised {hits}/3 alone", hits >= 2)
    check("unrelated speech does not wake it", gate.matched(stt.transcribe(say(phrases[0]))), None)
    heard = stt.transcribe(say(f"{wake} {phrases[0]}"))
    if matched := gate.matched(heard):
        check(f"command split from {heard!r}", len(set(phrases[0].split()) & set(heard.split(matched, 1)[-1].split())) > 0)
    else:
        check(f"wake word found in {heard!r}", False)
    return check.report()


def test_tui():
    """Boot the TUI headless and drive it, so the frontend is covered too."""
    import asyncio

    from langai import config
    from langai.ui.tui import LangaiApp

    check = Checks()

    async def drive():
        cfg = config.load()
        app = LangaiApp(cfg)
        async with app.run_test() as pilot:
            check("app mounted", app.is_running)
            check("chat pane exists", bool(app.query("#chat")))
            check("input focused", app.query_one("#prompt").has_focus)
            for _ in range(60):  # the model loads in a worker
                if app.session:
                    break
                await pilot.pause(0.25)
            check("session built in the worker", bool(app.session))
            if app.session:
                app.query_one("#prompt").value = "/help"
                await pilot.press("enter")
                await pilot.pause(0.2)
                lines = [str(w.render()) for w in app.query("#chat Static")]
                check("/help rendered", any("/model" in line for line in lines))
                app.query_one("#prompt").value = "/menu"
                await pilot.press("enter")
                await pilot.pause(0.2)
                lines = [str(w.render()) for w in app.query("#chat Static")]
                check("/menu rendered", any("thread" in line for line in lines))
                app.query_one("#prompt").value = "/model"
                await pilot.press("enter")
                await pilot.pause(0.2)
                lines = [str(w.render()) for w in app.query("#chat Static")]
                check("/model lists every alias", any("qwen25" in ln and ">" in ln for ln in lines))
                await pilot.press("f2")
                await pilot.pause(0.2)
                check("F2 toggled voice on", app.session.voice.enabled)
                await pilot.press("f2")
                await pilot.pause(0.2)
                check("F2 toggled voice off", not app.session.voice.enabled)
                check("activity line exists", bool(app.query("#activity")))
                await pilot.press("ctrl+c")
                await pilot.pause(0.2)
                check("ctrl+c hushes instead of quitting", app.is_running)
            await pilot.press("ctrl+q")

    asyncio.run(drive())
    return check.report()


# label, prompt, expected tool (None = should answer without tools)
CASES = [
    ("search", "what is the latest stable version of the linux kernel", "search_web"),
    ("time", "what day is it today", "now"),
    ("banter", "you're kinda mid ngl", None),
    ("math", "what is 17 times 23", None),
    ("filler", "ok", None),
]


def test_models(aliases, samples):
    from langchain_core.messages import HumanMessage
    from langchain_ollama import ChatOllama

    from langai.agent import SYSTEM_PROMPT
    from langai.app import MODEL_MAP, unload_models
    from langai.tools import TOOLS

    rows = []
    for alias in aliases:
        model = MODEL_MAP[alias]
        print(f"testing {alias} ({model})...", flush=True)
        unload_models(keep=model)
        llm = ChatOllama(model=model, temperature=0.7).bind_tools(TOOLS)
        scores, bad, structured, times = {}, 0, 0, []
        for label, prompt, want in CASES:
            hits = 0
            for _ in range(samples):
                began = time.perf_counter()
                try:
                    reply = llm.invoke([SYSTEM_PROMPT, HumanMessage(prompt)])
                except Exception as exc:
                    print(f"  {label}: ERROR {str(exc)[:70]}")
                    hits = -1
                    break
                times.append(time.perf_counter() - began)
                names = [c["name"] for c in reply.tool_calls]
                structured += bool(names)
                # a correct call passes plain strings, not schema fragments
                bad += sum(not all(isinstance(v, str) for v in c["args"].values()) for c in reply.tool_calls)
                hits += (want in names) if want else (not names)
            scores[label] = hits
        rows.append(dict(alias=alias, model=model, scores=scores, bad=bad,
                         structured=structured,
                         avg=sum(times) / len(times) if times else 0.0))
    unload_models()

    width = max(len(r["model"]) for r in rows) + 2
    header = f"{'alias':<9} {'model':<{width}}" + "".join(f"{c[0]:<9}" for c in CASES)
    print("\n" + header + f"{'args':<7}{'s/turn':>7}\n" + "-" * (len(header) + 14))
    for row in rows:
        cells = ""
        for label, _prompt, _want in CASES:
            hits = row["scores"][label]
            cells += f"{('err' if hits < 0 else f'{hits}/{samples}'):<9}"
        args = "n/a" if not row["structured"] else ("ok" if not row["bad"] else "BAD")
        print(f"{row['alias']:<9} {row['model']:<{width}}{cells}{args:<7}{row['avg']:>6.1f}s")
    print(f"\n{len(CASES)} cases x {samples} sample(s). 'search' and 'time' want that "
          "tool; the rest want none.\nargs = arguments were plain strings; "
          "n/a = no structured calls at all.")
    return 0


def main():
    p = argparse.ArgumentParser(description="Test harness for langai v2")
    p.add_argument("mode", choices=["logic", "ear", "tui", "models"])
    p.add_argument("args", nargs="*", help="ear: it|en   models: aliases")
    p.add_argument("-n", type=int, default=1, help="Samples per case")
    args = p.parse_args()

    if args.mode == "logic":
        return test_logic()
    if args.mode == "tui":
        return test_tui()
    if args.mode == "ear":
        return test_ear(args.args[0] if args.args else "it")

    from langai.app import MODEL_MAP
    aliases = args.args or list(MODEL_MAP)
    if unknown := [a for a in aliases if a not in MODEL_MAP]:
        p.error(f"unknown alias(es): {', '.join(unknown)}")
    return test_models(aliases, args.n)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
