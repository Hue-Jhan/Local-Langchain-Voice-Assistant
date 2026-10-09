#!/usr/bin/env python3
"""Entry point, so `python main.py` works without installing the package."""

import sys

from langai.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
