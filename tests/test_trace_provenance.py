"""Canonical trace manifest verification."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.evaluation.trace_provenance import (
    load_canonical_manifest,
    sha256_file,
    trace_directories_absent,
    verify_canonical_manifest,
    verify_or_exit,
)


def test_canonical_manifest_loads() -> None:
    m = load_canonical_manifest()
    assert m.get("version") == 2
    assert "standardized_root" not in m
    assert len(m.get("directories", [])) >= 1
    for entry in m["directories"]:
        assert "relpath" in entry
        assert "logical_name" in entry


def test_canonical_manifest_matches_local_traces() -> None:
    if trace_directories_absent():
        pytest.skip("trace directories absent on this machine; skipping hash verification")
    errors = verify_canonical_manifest()
    assert not errors, "\n".join(errors[:10])


def test_verify_or_exit_raises_on_tampered_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.evaluation import trace_provenance as tp

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    name = "probe.json"
    (trace_dir / name).write_text('{"throughput_kbps":[1000.0,1000.0]}', encoding="utf-8")
    wrong_digest = "0" * 64
    assert sha256_file(trace_dir / name) != wrong_digest
    manifest = {
        "version": 2,
        "directories": [
            {"logical_name": "tamper_test", "relpath": "traces", "files": {name: wrong_digest}},
        ],
    }
    monkeypatch.setattr(tp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tp, "load_canonical_manifest", lambda: manifest)
    with pytest.raises(SystemExit) as exc:
        verify_or_exit()
    assert exc.value.code == 2


def test_paired_align_requires_equal_keys() -> None:
    from src.evaluation.paired_bootstrap import align_paired_metric

    a = [{"trace_file": "t.json", "episode": 0, "episode_seed": 1, "episodic_qoe": 1.0}]
    b = [{"trace_file": "t.json", "episode": 1, "episode_seed": 1, "episodic_qoe": 2.0}]
    with pytest.raises(ValueError, match="paired episode keys differ"):
        align_paired_metric(a, b, "episodic_qoe")

    x, y, n = align_paired_metric(a, a, "episodic_qoe")
    assert n == 1
    assert float(x[0]) == 1.0
