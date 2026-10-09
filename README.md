# Local LangChain Voice Assistant

A chat assistant that runs entirely on your own machine. **LangChain** and **LangGraph**
wire a local Ollama model to a few abilities: web search, the clock, speaking its answers,
and listening through the microphone. Nothing leaves the machine and there are no API keys.

# LLM

### LangChain

LangChain supplies the model client (`ChatOllama`) and the message types. LangGraph
supplies the shape of a turn and the memory. The graph is small on purpose:

```
START -> agent -> (tool call?) -> tools -> agent -> ... -> END
                  (no call?)  -> END
```

`agent.py` builds it. The agent node sends the system prompt plus the recent messages to
Ollama; `tools_condition` sends the answer to the tool node if it contains a tool call,
and the tool node loops back so the model writes the final reply itself rather than
leaving raw tool output on screen. A `SqliteSaver` checkpointer stores every thread in
`data/sessions.db`, so conversations survive quitting.

The system prompt is three lines: never act like a polite AI, keep replies short and blunt
(two sentences unless more is asked for), and answer in the language the question came in.
That last line matters because the microphone and the voice both work in one language at a
time, and the reply has to match. Keeping the whole thing short matters too, since it is
re-read on every message.

| Name | Model | Notes |
|---|---|---|
| `qwen25` | `qwen2.5:3b` | **The default, and the best of these.** |
| `qwen4` | `qwen3:4b` | Thinks before answering, which costs minutes here |
| `granite` | `granite3.3:2b` | Says it supports tools but never calls them |
| `lama3` | `llama3.2:3b` | Unreliable: skips searches, malformed arguments |
| `lama3ab` `qwen3ab` `qwen7ab` | uncensored builds | Tool calls come out as plain text |

The uncensored ("abliterated") builds lost the structured output that tool calling needs.
Fine for plain chat; anything with a known problem warns you at startup. Only one model
fits in memory at a time, so `/model` and the benchmark stop the previous one for you.

### Tools

Two tools, `tools/clock.py` and `tools/web.py`, plus `tools/rag.py` as an empty
placeholder for searching your own documents later. Each tool file holds both the tool and
the words that call for it, so everything about the clock is in one place.

Speaking is deliberately **not** a tool. As a tool the model has to copy its own answer
into an argument, and small models copy the wrong thing: asked to sing, it called
`speak("Sing me a song")` and then wrote the song as text. Instead `speech/speaker.py`
reads out whatever the answer turned out to be.

How a tool gets chosen is set by one line, `[tools] mode` in `config.toml`:

- **`"router"`** (the default) matches your wording against patterns, then runs the tool
  itself and hands the result to the model. Deterministic, and nothing is added to the
  prompt, so chat stays fast.
- **`"model"`** gives every tool to the model on every message and lets it decide. This is
  the option to switch to once a smarter model fits on the machine.

`"model"` works, and `qwen2.5:3b` is not good enough for it. Tested just now: told
literally "use your search_web tool", it searches correctly, and told "call the now tool",
it reads the clock. Asked "what time is it", it answered *"It's about time you told me what
to do"*, and asked for the bitcoin price it replied *"I don't handle cryptocurrency data
directly, try searching online"*. It also calls tools nobody wanted: it once answered
`"ok"` by searching *"current political developments"*, and spent 16 seconds on
`"whats 2*3?"` by fetching the date first. Hence the patterns, for now.

### Efficiency

**What you get.** A short answer in under a second. A longer one in two or three, because
writing it out is the slow part. A clock question instantly, since the clock answers it and
the model never runs. A web search in 15 to 20 seconds, nearly all of that the computer
reading the results. On a Raspberry Pi 5, maybe three to five times slower, so a few
seconds for a short reply; that last figure is a guess, not a measurement.

**Why it is quick.** One small model, nothing in the cloud, three lines of instructions.
Tool descriptions, which would add a paragraph to every message, are never sent. Only the
last 16 messages are shown to it. Most of all, they always arrive in the same order: Ollama
remembers what it read last time and only reads the new part, so a long chat is no slower
than a fresh one. Search results are used once and thrown away. Startup does the first read
while it says "booting up model", so your first question is not the slow one.

**Why it is simple, in the bad sense.** Tools are chosen by matching your words against
lists, written out twice, once per language, so unusual phrasing is missed. Time zones are
left to the model, which gets them wrong. Wake words are matched by listing the mistakes the
microphone actually makes. The 16-message limit counts messages, not their length. The
voices and the Italian speech model are shortcuts to folders elsewhere on the machine, so a
fresh copy must put them back. `data/sessions.db` saves the whole chat again after every
message, so it grows fast: 120 messages is already 4.9MB.

**What would fix it.** Mostly a better model: set `[tools] mode = "model"`, let it pick its
own tools, and delete the word lists. You pay in speed, because the tool descriptions go
back into every message. In between, a small trained classifier could decide which tools to
offer, doing the word lists' job without being written by hand. Time zones need a few lines
of `zoneinfo`.

# Code

### Start & config

```bash
pip install --user -e .        # pyproject.toml is the only list of dependencies
ollama pull qwen2.5:3b
python main.py                # the full-screen interface
python main.py -cli           # a plain scrolling prompt instead
```

Needs `ollama` running, plus `piper-tts` and `ffplay` on your `PATH` for speech.
`config.toml` sits in the top folder and holds every setting. Flags override it:
`-model -temp -max -v -voice -ear -gate push|wake -lang it|en -thread -log -noroute`.

`models/` and `voices/` are not in the repository, being half a gigabyte between them. Put
a Vosk model in `models/` for speech recognition and Piper `.onnx` voices in `voices/`, or
point `[ear] model` and `[paths] voices` somewhere else.

| | |
|---|---|
| `/help` · `/menu` | the commands · your current settings |
| `/model [name]` | list the models, or switch to one |
| `/v` · `/voice [on\|off\|auto\|<id>]` | talking on or off; `auto` matches the language, or ids `0` Italian, `1` US, `2` UK |
| `/s [text]` · `/stop` | say the last answer, or this text · stop talking |
| `/exit` | quit |

Full-screen interface: **F2** voice, **F3** talk, **Ctrl+C** stop whatever is happening,
**Ctrl+Q** quit, **Ctrl+L** clear. A spinner shows what it is doing (*booting up model*,
*thinking*, *calling search tool*, *listening*) and an indicator moves while it speaks.
Talking does not block typing, and a new answer cuts off the previous one.

`-ear` turns on the microphone; recognition is Vosk, offline, `-lang it` (default) or
`-lang en`. **One language at a time**, because a Vosk model only knows one, so `-lang`
picks which language you are speaking. The reply comes back in the same language, and the
voice follows it: `[voice] auto` reads Italian answers in Riccardo and English ones in
Ryan, working it out from the words. The question counts as well as the answer, so a clock
reply, which is English whatever you asked in, is still read in the language you asked.
`/voice <id>` fixes one voice and turns that off, `/voice auto` turns it back on. **Press to talk** is the default: Enter on an empty line in the plain
interface, **F3** in the full-screen one, once to start and again to stop, because a
terminal cannot tell when you release a key. What you said lands in the input box so you
can correct it. **`-gate wake`** listens constantly for a wake word instead, plain
interface only. The wake words were chosen by testing, not taste: `ciao assistente` and
`hey assistant`. It never listens while the assistant is talking.

### Structure

```
config.toml  every setting, in the top folder so it is easy to find
main.py  what you run · pyproject.toml  dependencies · data/  your history
models/  speech recognition · analysis/tester.py  the test suite
langai/ __main__.py  options, then one of the two interfaces
        app.py  Session: a turn, the commands, switching model
        agent.py  the graph, the system prompt, the history window
        router.py  rule order, and the rules belonging to no tool
        config.py · tools/ (clock, web, rag) · speech/ (speaker, ear)
        ui/  tui.py (full screen) · cli.py (plain) · term.py (colours)
```

`langai/` is a package rather than a bare `src/` because Python needs a name to import by,
and because `langai` installs as a command.

Four seams are meant to be swapped. A **rule** looks at your message and either stays quiet
or returns a decision, first one wins; a tool's rules live in its own file and `RULES` in
`router.py` sets the order. In **search**, `BACKEND` fetches and `FORMATTER` lays out.
In **voice**, `POLICY` decides what gets spoken. In **ear**, `GATES` decide whether speech
was meant for the assistant; they are objects because they remember things, and they see
the recording as well as the words, so a future gate could recognise *who* is speaking.
That is where "only these people may give commands" would go.

### Workflow of a message

```
you type or speak
  ├─ starts with "/" ─────────────> a command; the model never sees it
  └─ router.decide(text)
       ├─ greeting, arithmetic, "dont search" ──> the model, no tools
       ├─ "what time is it" ──────────────────> the clock, no model
       ├─ "cerca ...", "look it up", a price  ──> we search, then hand the
       │  or news question                        results to the model
       └─ anything else ─────────────────────> the model
            └─ streams out word by word, read aloud if voice is on
```

`Session.run` yields `token`, `tool` and `stats` events, so turn logic is written once and
both interfaces only display it. History is saved per thread in `data/sessions.db` and
survives quitting (`-thread <name>` for a separate one). The grey line after an answer,
`1.2s · 46 tok · 14.4 tok/s`, times it, counting waits for the web separately.

### Testing

```bash
python analysis/tester.py logic              # fast, no model needed
python analysis/tester.py ear it|en          # speech, no microphone needed
python analysis/tester.py tui                # starts the interface invisibly and uses it
python analysis/tester.py models qwen25 -n 4 # which model picks tools correctly
```

The speech test speaks with Piper and listens back through Vosk, scoring words returned
rather than demanding a perfect match. The interface test has caught a startup crash and a
broken Ctrl+C. One known rough edge the tests do not cover: `qwen2.5:3b` still ends some
answers with a question however firmly it is told not to.

`TODO.md` has the open questions. `CLAUDE.md` is a summary for starting a fresh chat.
