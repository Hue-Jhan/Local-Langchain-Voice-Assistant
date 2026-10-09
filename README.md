# Local LangChain Voice Assistant

Lightweight local LLM voice assistant that uses **LangChain** and **LangGraph** to connect a local Ollama model to a few skills: web search, time/clock, speaking its answers, and listening through the microphone. ~90% Vibecoded with opus5.

- [🤖 LLM](#-llm)
  - [1 — LangChain](#1---langchain)
  - [2 — Tools](#2---tools)
  - [3 — Efficiency](#3---efficiency)
- [🖥️ Code](#️-code)
  - [0 — Start & Config](#0---start--config)
  - [1 — Structure & Workflow](#1---structure--workflow)
  - [2 — Testing](#2---testing)

<a id="llm"></a>
# 🤖 LLM

<a id="langchain"></a>
### 1 - LangChain

LangChain supplies the model client (`ChatOllama`) and the message types, LangGraph supplies the shape of a turn and the memory. The graph is small on purpose:

```
START -> agent -> (tool call?) -> tools -> agent -> ... -> END
                  (no call?)  -> END
```

The agent node sends the system prompt plus the recent messages to Ollama; `tools_condition` sends the answer to the tool node if it contains a tool call, and the tool node loops back so the model writes the final reply itself rather than leaving raw tool output on screen. A `SqliteSaver` checkpointer stores every thread in `data/sessions.db`, so conversations survive quitting.

The system prompt says to never act like a polite AI (funny hehe) and to keep replies short and blunt. Keeping the whole thing short matters since previous messages are re-read on every new message and this project is meant to be fast and lightweight.

| Name | Model | Notes |
|---|---|---|
| `qwen25` | `qwen2.5:3b` | **The default, and the best of these.** |
| `qwen4` | `qwen3:4b` | Thinks before answering, which costs minutes here |
| `granite` | `granite3.3:2b` | Says it supports tools but never calls them (stupid)|
| `lama3` | `llama3.2:3b` | Unreliable: skips searches, malformed arguments, but good at conversation |
| `lama3ab` `qwen3ab` `qwen7ab` | uncensored builds | Tool calls come out as plain text and don't work properly|


<a id="tools"></a>
### 2 - Tools

Two tools, `tools/clock.py` and `tools/web.py`, plus `tools/rag.py` as an empty placeholder for searching your own documents later, each tool file holds both the tool and the words that call for it, so everything about the clock is in one place.

Speaking is deliberately **not** a tool cuz as a tool the model has to copy its own answer into an argument and small models copy the wrong thing: asked to sing, it called
`speak("Sing me a song")` and then wrote the song as text. Instead `speech/speaker.py`reads out whatever the answer turned out to be.

How a tool gets chosen is set by one line, `[tools] mode` in `config.toml`:

- **`"router"`** (the default) matches your wording against patterns, then runs the tool itself and hands the result to the model, nothing is added to the prompt, so chat stays fast, however it's not as precise as a custom machine learnign script or a mini llm that only checks which tools to call.
- **`"model"`** gives every tool to the model on every message and lets it decide. This is the option to switch if you have a powerful machine and can fit a smarter model.

<a id="efficiency"></a>
### 3 - Efficiency

Answers in <1s, however a web search requires 15 to 20 seconds, nearly all of that the computer reading the results. On a Raspberry Pi 5, maybe three to five times slower.

Tools are chosen by matching your words against lists, written out twice, once per language, so unusual phrasing is missed. Time zones are left to the model, which gets them wrong. Wake words are matched by listing the mistakes the microphone actually makes. The 16-message limit counts messages, not their length. The voices and the Italian speech model are shortcuts to folders elsewhere on the machine, so a fresh copy must put them back. `data/sessions.db` saves the whole chat again after every message, so it grows fast: 120 messages is already 4.9MB.

A better model would fix most of this projects' issues by: set `[tools] mode = "model"`, which lets it pick its own tools and delete the word lists. You pay in speed, because the tool descriptions go back into every message. In between, a small trained classifier could decide which tools to offer, doing the word lists' job without being written by hand. Time zones need a few lines of `zoneinfo`.

I built this project just because Hermes agent couldn't connect to local ollama models as it injects into the system prompt thousends of lines of tools, instructions, and other stuff (even if all plugins and skills are disabled). This project instead only injects the bare necessary but it's way dumber and less capable in terms of agentics skills.

<a id="code"></a>
# 🖥️ Code

<a id="start-config"></a>
### 0 - Start & config

```bash
pip install --user -e .        # pyproject.toml is the only list of dependencies
ollama pull qwen2.5:3b
python main.py                # the full-screen interface
python main.py -cli           # a plain scrolling prompt instead
```

Needs `ollama` running, plus `piper-tts` and `ffplay` on your `PATH` for speech. `config.toml` sits in the top folder and holds every setting. Flags override it: -model -temp -max -v -voice -ear -gate push|wake -lang it|en -thread -log -noroute`.

`models/` and `voices/` are not in the repository, being half a gigabyte between them. Put a Vosk model in `models/` for speech recognition and Piper `.onnx` voices in `voices/`, or point `[ear] model` and `[paths] voices` somewhere else.

| | |
|---|---|
| `/help` · `/menu` | the commands · your current settings |
| `/model [name]` | list the models, or switch to one |
| `/v` · `/voice [on\|off\|auto\|<id>]` | talking on or off; `auto` matches the language, or ids `0` Italian, `1` US, `2` UK |
| `/s [text]` · `/stop` | say the last answer, or this text · stop talking |
| `/exit` | quit |

Full-screen interface: **F2** voice, **F3** talk (doesnt work), **Ctrl+C** stop whatever is happening, **Ctrl+Q** quit, **Ctrl+L** clear. A spinner shows what it is doing (*booting up model*,
*thinking*, *calling search tool*, *listening*) and an indicator moves while it speaks. Talking does not block typing, and a new answer cuts off the previous one.

`-ear` turns on the microphone; recognition is Vosk, offline, `-lang it` (default) or `-lang en`. The reply comes back in the same language, and the voice follows it: `[voice] auto` reads Italian answers in Riccardo and English ones in Ryan, working it out from the words. 

`-gate wake` listens constantly for a wake word instead, plain cli interface only, the wake words were chosen by testing, not taste: `ciao assistente` and `hey assistant`, but you can modify them. I suggest using simple ones like "Open". 

<a id="structure-workflow"></a>
### 1 - Structure & Workflow

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

Workflow of the message:

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

History is saved per thread in `data/sessions.db` and survives quitting (`-thread <name>` for a separate one), also the grey line after an answer, `1.2s · 46 tok · 14.4 tok/s`, times the answer itself, waiting for the web search separately.

<a id="testing"></a>
### 2 - Testing

```bash
python analysis/tester.py logic              # fast, no model needed
python analysis/tester.py ear it|en          # speech, no microphone needed
python analysis/tester.py tui                # starts the interface invisibly and uses it
python analysis/tester.py models qwen25 -n 4 # which model picks tools correctly
```

The speech test speaks with Piper and listens back through Vosk, scoring words returned rather than demanding a perfect match. 
