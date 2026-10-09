"""Settings from config.toml. Command-line flags override them."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the project directory
CONFIG_FILE = ROOT / "config.toml"

DEFAULTS = {
    "ui": {"mode": "tui", "verbose": False},
    "model": {"alias": "qwen25", "temperature": 0.7, "max_tokens": 500,
              "keep_alive": ""},
    "voice": {"enabled": False, "voice_id": 1, "auto": True},
    "ear": {"enabled": False, "gate": "push", "language": "it", "model": "", "device": -1},
    "memory": {"persist": True, "path": "data/sessions.db", "thread": "main",
               "window": 16},
    "router": {"enabled": True},
    "tools": {"mode": "router"},
    "paths": {"voices": "voices", "models": "models"},
    "log": {"path": ""},
}


def load(path=None):
    """Read config.toml over the defaults. Missing file or bad TOML is not fatal."""
    merged = {name: dict(values) for name, values in DEFAULTS.items()}
    source = Path(path) if path else CONFIG_FILE
    try:
        found = tomllib.loads(source.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return merged
    for section, values in found.items():
        if isinstance(values, dict):
            merged.setdefault(section, {}).update(values)
    return merged


def apply_flags(cfg, args):
    """Overlay the command-line flags that were actually given."""
    for flag, (section, key) in {
        "model": ("model", "alias"), "temp": ("model", "temperature"),
        "max": ("model", "max_tokens"), "gate": ("ear", "gate"),
        "vosk": ("ear", "model"), "lang": ("ear", "language"),
        "mic": ("ear", "device"), "log": ("log", "path"),
        "thread": ("memory", "thread"),
    }.items():
        value = getattr(args, flag, None)
        if value is not None:
            cfg[section][key] = value
    if getattr(args, "ear", False):
        cfg["ear"]["enabled"] = True
    if getattr(args, "cli", False):
        cfg["ui"]["mode"] = "cli"
    if getattr(args, "tui", False):
        cfg["ui"]["mode"] = "tui"
    if getattr(args, "voice", False):
        cfg["voice"]["enabled"] = True
    if getattr(args, "v", False):
        cfg["ui"]["verbose"] = True
    if getattr(args, "noroute", False):
        cfg["router"]["enabled"] = False
    return cfg


def apply_paths(cfg):
    """Point the speech modules at the configured directories, so a moved copy only needs
    config.toml changed. ear is touched lazily because importing it pulls in vosk.
    """
    paths = cfg.get("paths", {})
    if voices := paths.get("voices"):
        from .speech import speaker
        speaker.VOICES_DIR = (ROOT / voices).resolve()
