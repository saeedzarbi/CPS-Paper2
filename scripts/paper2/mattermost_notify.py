#!/usr/bin/env python3
"""CLI wrapper for src.utils.mattermost_notify."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.mattermost_notify import notify


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: mattermost_notify.py <message>", file=sys.stderr)
        return 1
    return 0 if notify(" ".join(sys.argv[1:])) else 2


if __name__ == "__main__":
    raise SystemExit(main())
