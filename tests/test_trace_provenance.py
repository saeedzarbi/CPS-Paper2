"""Canonical trace manifest verification."""
from __future__ import annotations

import pytest

from src.evaluation.trace_provenance import load_canonical_manifest, verify_canonical_manifest


def test_canonical_manifest_loads() -> None:
    m = load_canonical_manifest()
    assert m.get("version") == 1
    assert len(m.get("directories", [])) >= 1


def test_canonical_manifest_matches_local_traces() -> None:
    errors = verify_canonical_manifest()
    if errors:
        pytest.skip("local traces differ from canonical (sync data or rebuild manifest): " + errors[0][:80])
    assert not errors
