"""Mattermost incoming webhook helper (see mattermost_webhook.url at repo root)."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEBHOOK_ENV = "MATTERMOST_WEBHOOK_URL"
WEBHOOK_FILE = ROOT / "mattermost_webhook.url"


def resolve_webhook_url() -> str | None:
    url = os.environ.get(WEBHOOK_ENV, "").strip()
    if url:
        return url
    if WEBHOOK_FILE.is_file():
        return WEBHOOK_FILE.read_text(encoding="utf-8").strip().strip("\r")
    return None


def notify(text: str, *, username: str = "OACS") -> bool:
    url = resolve_webhook_url()
    if not url:
        return False
    payload = json.dumps({"text": text, "username": username}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except urllib.error.URLError as exc:
        print(f"mattermost notify failed: {exc}", file=sys.stderr)
        return False
