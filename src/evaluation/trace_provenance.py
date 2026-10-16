"""Canonical trace manifests (sha256) and verification for Tube-v2 campaigns."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from configs.paths import PROJECT_ROOT

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_MANIFEST = ROOT / "configs" / "trace_manifests" / "canonical_traces.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath_from_repo(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_manifest_directory(entry: dict[str, Any]) -> Path:
    """Resolve a manifest directory entry to an absolute path."""
    relpath = entry.get("relpath") or entry.get("path")
    if not relpath:
        raise KeyError(f"manifest entry missing relpath: {entry.get('logical_name')}")
    p = Path(relpath)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def manifest_entries_for_dir(
    directory: Path,
    *,
    logical_name: str,
    generator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not directory.is_dir():
        return {
            "logical_name": logical_name,
            "relpath": relpath_from_repo(directory),
            "missing": True,
            "files": {},
        }
    files: dict[str, str] = {}
    for p in sorted(directory.glob("*.json")):
        if p.name.startswith("trace_stats"):
            continue
        files[p.name] = sha256_file(p)
    return {
        "logical_name": logical_name,
        "relpath": relpath_from_repo(directory),
        "generator": generator or {},
        "files": files,
    }


def load_canonical_manifest() -> dict[str, Any]:
    if not CANONICAL_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Missing canonical trace manifest: {CANONICAL_MANIFEST}. "
            "Run: python scripts/paper2/build_trace_manifests.py --write"
        )
    return json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))


def trace_directories_absent(manifest: dict[str, Any] | None = None) -> bool:
    """True when no manifest directory has any JSON trace files on disk."""
    manifest = manifest or load_canonical_manifest()
    for entry in manifest.get("directories", []):
        if entry.get("missing"):
            continue
        dir_path = resolve_manifest_directory(entry)
        if dir_path.is_dir() and any(dir_path.glob("*.json")):
            return False
    return True


def verify_canonical_manifest(
    *,
    strict: bool = True,
    manifest: dict[str, Any] | None = None,
) -> list[str]:
    """Return list of mismatch descriptions; abort via ``verify_or_exit``."""
    manifest = manifest or load_canonical_manifest()
    errors: list[str] = []
    for entry in manifest.get("directories", []):
        logical = entry.get("logical_name", "?")
        if entry.get("missing"):
            if strict:
                errors.append(f"{logical}: directory missing ({entry.get('relpath')})")
            continue
        dir_path = resolve_manifest_directory(entry)
        expected: dict[str, str] = entry.get("files", {})
        if not dir_path.is_dir():
            errors.append(f"{logical}: path not found: {dir_path}")
            continue
        actual: dict[str, str] = {}
        for p in sorted(dir_path.glob("*.json")):
            if p.name.startswith("trace_stats"):
                continue
            actual[p.name] = sha256_file(p)
        exp_names = set(expected)
        act_names = set(actual)
        if exp_names != act_names:
            missing = sorted(exp_names - act_names)
            extra = sorted(act_names - exp_names)
            if missing:
                errors.append(f"{logical}: missing files {missing[:5]}")
            if extra:
                errors.append(f"{logical}: extra files {extra[:5]}")
        for name, digest in expected.items():
            if name in actual and actual[name] != digest:
                errors.append(
                    f"{logical}/{name}: sha256 mismatch "
                    f"(expected {digest[:12]}… got {actual[name][:12]}…)"
                )
    return errors


def verify_or_exit(*, strict: bool = True) -> None:
    errors = verify_canonical_manifest(strict=strict)
    if errors:
        msg = (
            "Trace provenance verification FAILED (canonical manifest mismatch).\n"
            + "\n".join(f"  - {e}" for e in errors[:20])
            + "\nSync trace data with the canonical set or run "
            "`python scripts/paper2/build_trace_manifests.py --write` after intentional updates."
        )
        print(msg, file=sys.stderr)
        raise SystemExit(2)


__all__ = [
    "CANONICAL_MANIFEST",
    "load_canonical_manifest",
    "manifest_entries_for_dir",
    "relpath_from_repo",
    "resolve_manifest_directory",
    "sha256_file",
    "trace_directories_absent",
    "verify_canonical_manifest",
    "verify_or_exit",
]
