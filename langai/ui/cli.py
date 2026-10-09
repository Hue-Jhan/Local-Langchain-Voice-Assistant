"""Plain CLI frontend: a streaming REPL with a spinner."""

import sys

from ..speech import speaker
from ..app import Session
from .term import BOLD, CYAN, DIM, GREEN, RED, RESET, Spinner

BAR = DIM + "─" * 56 + RESET


def banner(session, ear_line):
    print(f"{BOLD}langai{RESET} {DIM}v2{RESET}   {GREEN}{session.model}{RESET}" f"  {DIM}thread {session.thread}{RESET}")
    if session.note:
        print(f"{DIM}note: {session.note}{RESET}")
    if ear_line:
        print(f"{DIM}{ear_line}{RESET}")
    print(f"{DIM}/help for commands · Ctrl+C to quit{RESET}\n")


def turn(session, text, verbose, spinner):
    started = False
    spinner.start()
    try:
        for kind, value in session.run(text):
            if kind == "tool":
                # always shown: you asked to be able to see tool use
                spinner.stop()
                line = value if verbose else " ".join(value.split())[:88]
                print(f"{DIM}  {line}{RESET}")
                spinner.start()
            elif kind == "token":
                if not started:
                    spinner.stop()
                    print(f"\n{GREEN}{session.alias}:{RESET} ", end="", flush=True)
                    started = True
                print(value, end="", flush=True)
            else:
                spinner.stop()
                if started:
                    print()
                else:
                    print(f"\n{GREEN}{session.alias}:{RESET} {DIM}(no output){RESET}")
                print(f"{DIM}{value}{RESET}\n")
    finally:
        spinner.stop()
    if problem := session.maybe_speak(text):
        print(f"{RED}{problem}{RESET}")
    print(BAR)


def run(cfg):
    loud = sys.stdout.isatty()  # piped output stays clean
    if loud:
        print(f"{DIM}booting up model...{RESET}", end="", flush=True)
    session = Session(cfg)
    session.warm()
    if loud:
        print("\r" + " " * 20 + "\r", end="")
    ear_line = session.open_ear() if cfg["ear"]["enabled"] else ""
    verbose, spinner = cfg["ui"]["verbose"], Spinner()
    banner(session, ear_line)

    # the wake gate is always listening, so it owns the loop and there is no prompt
    if session.gate is not None and session.gate.name == "wake":
        try:
            for heard in session.gate.utterances():
                print(f"{CYAN}You:{RESET} {heard}")
                turn(session, heard, verbose, spinner)
        except (KeyboardInterrupt, EOFError):
            pass
        speaker.stop()
        print("\nExiting...")
        return

    talk = session.gate is not None
    prompt = f"{CYAN}You{DIM}(Enter=talk):{RESET} " if talk else f"{CYAN}You:{RESET} "
    while True:
        try:
            line = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            speaker.stop()
            print("\nExiting...")
            return

        if not line:
            if not talk:
                continue
            try:
                heard = session.gate.listen_once(prompt="recording... [Enter] to stop")
            except KeyboardInterrupt:
                print(f"\n{DIM}[recording cancelled]{RESET}")
                continue
            except Exception as exc:
                print(f"{RED}mic error:{RESET} {exc}")
                continue
            if not heard:
                print(f"{DIM}(heard nothing){RESET}")
                continue
            print(f"{CYAN}heard:{RESET} {heard}")
            turn(session, heard, verbose, spinner)
        elif line.startswith("/"):
            if line in {"/exit", "/quit"}:
                speaker.stop()
                print("Exiting...")
                return
            reply = session.command(line)
            if reply is None:
                reply = session.help_text()
            if reply:
                print(f"{DIM}{reply}{RESET}")
        else:
            try:
                turn(session, line, verbose, spinner)
            except Exception as exc:
                print(f"\n{RED}error:{RESET} {exc}\n{BAR}")
