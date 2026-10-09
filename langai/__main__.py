"""Argument parsing and frontend dispatch."""

import warnings

# must precede the langchain imports - this warning fires at import time
warnings.filterwarnings("ignore", message=".*Pydantic V1.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse  # noqa: E402
import sys  # noqa: E402

from . import config  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="langai", description="Local AI assistant")
    p.add_argument("-cli", action="store_true", help="Plain REPL instead of the TUI")
    p.add_argument("-tui", action="store_true", help="Force the TUI (the default)")
    p.add_argument("-model", help="Model alias (see /model)")
    p.add_argument("-temp", type=float, help="Temperature")
    p.add_argument("-max", type=int, help="Max tokens per answer (0 = off)")
    p.add_argument("-v", "-verbose", action="store_true", help="Show tool activity")
    p.add_argument("-voice", action="store_true", help="Start with voice on")
    p.add_argument("-ear", action="store_true", help="Enable speech input")
    p.add_argument("-gate", choices=["push", "wake"], help="Speech entry point")
    p.add_argument("-lang", choices=["it", "en"], help="Speech recognition language")
    p.add_argument("-vosk", help="Path to a Vosk model directory")
    p.add_argument("-mic", type=int, help="Input device index")
    p.add_argument("-thread", help="Conversation thread name")
    p.add_argument("-log", help="Append a transcript to this file")
    p.add_argument("-noroute", action="store_true", help="Always offer tools; skip the rule layer")
    p.add_argument("-config", help="Path to a config.toml")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = config.apply_flags(config.load(args.config), args)
    config.apply_paths(cfg)
    try:
        if cfg["ui"]["mode"] == "cli":
            from .ui.cli import run
        else:
            from .ui.tui import run
        run(cfg)
    except KeyboardInterrupt:
        from .speech import speaker
        speaker.stop()
        print("\nExiting...")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
