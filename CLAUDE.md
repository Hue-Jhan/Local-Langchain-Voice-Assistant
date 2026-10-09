# CLAUDE.md, orientation for a fresh chat

Read this before touching anything. It records decisions that were made from
measurements, so that they are not quietly undone.

## What this is

A local terminal assistant at `~/code_projects/langai/v2`. LangChain + LangGraph wire a
local Ollama model to a web search, a clock, Piper text-to-speech and Vosk speech
recognition. No cloud, no API keys. `v1/` holds the original prototypes and **must stay
untouched**, `v2/voices` is a symlink into it.

## The hardware is the main constraint

i7-13620H, 16 threads, **15 GiB RAM with ~11 GiB usually already in use**, Intel UHD
integrated graphics that Ollama **cannot** use. Everything runs on the CPU at roughly
8-18 tokens/second.

**Only ever have one Ollama model resident.** Ollama keeps a model loaded for five
minutes after use, so running two in sequence leaves both in RAM and swamps the machine.
`ollama stop <name>` before loading another. This was learned by annoying the user; do
not repeat it.

## Architecture

```
langai/__main__.py  flags -> config -> picks a frontend
langai/app.py       Session: one turn, commands, model switching. Both frontends use it.
                    Session.run is a generator yielding ("token"|"tool"|"stats", value)
langai/router.py    rules deciding what a message needs, BEFORE the model sees it
langai/agent.py     LangGraph StateGraph: agent node + ToolNode, SqliteSaver checkpointer
langai/tools/       clock.py, web.py (ddgs search + formatter + guard), rag.py (stub)
langai/speech/      speaker.py (Piper, voice policy), ear.py (Vosk, entry gates)
langai/ui/          tui.py (Textual, default), cli.py, term.py
analysis/tester.py  logic | ear it|en | tui | models
config.toml         all settings, in the project root
data/sessions.db    conversation history per thread
```

## Decisions that must not be casually reversed

**The model is never given tool definitions.** `[tools] mode = "router"`. Three reasons,
all measured on `qwen2.5:3b`:

- It calls tools pointlessly, answered `"ok"` by searching *"current political
  developments"*; answered `"whats 2*3?"` by fetching the date, 16 seconds.
- It picks the wrong tool, `"btc price now"` fetched the clock. Bound to search alone,
  the same prompt searches correctly.
- Told outright to look something up, it refused four times. **Ollama ignores
  `tool_choice`** (tested: both `"any"` and a tool name produced no call), so forcing is
  not available.

And the decisive one: **switching tool definitions on and off between messages was the
main source of latency.** Ollama caches the evaluated prompt prefix; tool schemas live in
that prefix. Over one 8-turn conversation: never bound **1.2s** per turn, always bound
1.5s, alternating **4.5s with spikes to 9s**. Keep the prompt prefix stable.

So the rules decide and `Session.prefetch` runs the tool. **Each tool file owns the
wording that calls for it** - `tools/clock.py` has `ASK`/`QUALIFIED` next to `now()`,
`tools/web.py` has `REFUSED`/`ASK`/`LIVE` next to `search()`. A rule returns a plain dict,
which `router.decide` turns into a `Decision`; that is what keeps `tools/` from importing
`router`. `router.py` holds only the rule order (`RULES`) and the rules belonging to no
tool (`filler`, `arithmetic`). Consequences to preserve:

- Pre-fetched results go into a **user** message for that one answer, and are never
  written to history. As a *system* message this model ignored them and quoted a 2023
  bitcoin price instead of the one just handed to it.
- A clock question is answered **straight from the tool, skipping the model**, and
  recorded with `agent.update_state`. The model turned "Wednesday 07 October" into
  "Thursday, October 8th", replied "it's time for your AI coffee break", answered "Check
  your device for the time. I can't fetch it here", and cut a full date to "Thursday".
  The test for "direct" used to be `≤4` words, which missed "what day is it today" and
  "aight bet, now whats the time"; it is now `clock.QUALIFIED` - direct unless a place or
  zone is named. For a qualified question the time is handed over as a *reference* ("The
  current local time here is ..."), not as "answer with exactly this".
- `router.decide(text, last_user)` falls back to the previous message for the query when
  the current one is a pronoun, "look it up", "cercalo".

**Speech is one language at a time, and the voice follows it.** A Vosk model knows one
language, so `[ear] language` / `-lang` picks what you speak and there is no detection to
do there. `speaker.language(text)` does detect, by word overlap against two small word
lists, and `VoiceState.voice_for(text, asked)` maps it to a voice. It takes the *question*
too: a clock answer is English whatever language it was asked in, so judging the answer
alone read Italian conversations in Ryan. `/voice <id>` pins a voice and clears `auto`.
The system prompt's third line ("answer in the language the user wrote in") exists for
this; do not drop it. The first two lines are the user's own persona wording - leave them.

**Voice is an output channel, not a tool.** As a tool the model must copy its answer into
an argument and copies the wrong thing: asked to sing it called `speak("Sing me a song")`
and wrote the song as text. `speech/speaker.py` holds a `POLICY` instead.

**The history window must be anchored, never sliding.** `agent.Window` keeps messages
from a fixed anchor and only appends; it re-anchors at `window * 3`, or on a turn that
already carries a pre-fetched tool result (`due=True`), so the trim hides inside a turn
that is slow anyway. A last-N sliding window was the second big latency bug: it changed
the message right after the system prompt every turn, so Ollama re-read the whole prompt
every turn. Measured the moment a thread passed 16 messages: prefix reuse **16/16 -> 1/16**
and prompt eval **0.24s -> 3.4s per turn, permanently**. This is what made every message
slow after one search - the search did not cause it, it just pushed the thread over the
window and filled it with long messages. Symptoms if it regresses: chat turns at 7-20s
with `tok/s` around 2, and worse the longer the answers in the window.

**`Session.warm()` runs at startup** in both UIs, inside the existing "booting up model"
phase. It sends the stored window with `num_predict=1` so the first real question is not
the one that pays for reading the thread: 3.9s -> 0.8s. It shares the same `Window`
object as `call_model`, which is what makes the prefix match - a separate `Window` would
compute an anchor one message off and warm the wrong prefix.

**Prompt length dominates latency.** Bare question 34 tokens, +system prompt 25, tool
schemas used to add 160 *per message*. Keep the system prompt short. Prompt eval runs at
roughly 90 tok/s, so the ~600-token search block costs ~7s on its own - that is why a
search turn is ~15-19s. `[tools] web.py MAX_RESULTS`/`SNIPPET_CHARS` are the lever, and
7 results is the user's own setting.

**`keep_alive` stays empty** (Ollama's 5-minute default). The user asked for this
explicitly after I set it to 30m.

**No calculator tool, no RAG yet.** The user declined both. `tools/rag.py` is a
deliberate stub. Every tool added costs prompt tokens on every message.

## Model notes

`qwen25` (`qwen2.5:3b`) is the default and the only good tool-caller here. `qwen3:4b`
reasons at length, 190s for one line of banter. `granite3.3:2b` claims tool support but
emits none. The abliterated builds (`*ab`) emit tool calls as **plain text**, so the graph
never routes to a tool; abliteration damaged their structured output. Fine for plain chat.

## Speech notes

Both Vosk Italian models are Italian-only; English uses
`models/vosk-model-en-us-0.22-lgraph` (205 MB). Wake words were chosen **by measurement**:
`ciao assistente` (it), `hey assistant` (en). Spoken alone, "assistente" scored 8/8 while
"ciao robot" managed 5/8 ("ciao rodo", "ciao rubo", "ciao paolo") and bare "robot" became
"prodotti". When a wake word runs into a command the recogniser collapses it, "hey
assistant turn on the light" → "hey assist in turn on the light", so `WAKE` lists the
collapsed forms explicitly. **Re-measure any new wake word with `tester.py ear`.**

Hold-to-talk is impossible: terminals have no key-release event. Press to start, press to
stop.

## Testing

```bash
python analysis/tester.py logic       # ~81 deterministic checks, no model
python analysis/tester.py ear it|en   # Piper -> Vosk round trip, no microphone
python analysis/tester.py tui         # Textual headless pilot
python analysis/tester.py models ...  # tool-routing benchmark, unloads between models
```

Speech and model checks are **statistical**, score ratios and best-of-N, never exact
equality, or the suite turns flaky. The TUI test has already caught a startup crash (a
method named `_ready` collided with a Textual hook) and a broken Ctrl+C.

## How this user works

- **Measure, do not guess.** Every latency or quality claim here came from a benchmark.
  Ollama reports `prompt_eval_duration`/`eval_duration` in `response_metadata` - that
  split is how both latency bugs were found. Comparing the prompt against the previous
  prompt message-by-message shows how much prefix was reused.
  They notice hand-waving and ask for numbers.
- **They edit the code themselves.** `MAX_RESULTS = 7` and Ryan's `length_scale = 0.80`
  are theirs, check before overwriting, and do not pin their tuning in a test.
- They write in English and Italian and expect replies in the language they used.
- They want short answers and short documents, and will say when something is too long.
- They put instructions and questions in `TODO.md` and expect it read at the start of a
  chat; keep their text and add status above it.
- Report failures plainly, including your own. Partial credit for honest limits beats a
  confident overstatement.
