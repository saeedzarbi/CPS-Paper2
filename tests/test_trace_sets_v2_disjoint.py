"""Train/val and harsh calib/test disjointness for Tube-v2."""
from __future__ import annotations

import pytest

from configs.trace_sets_v2 import (
    assert_disjoint_dirs,
    harsh_family_calib_test_dirs,
    markov_train_available,
    train_minus_val_dir,
    val_trace_dir,
)


def test_markov_train_minus_val_disjoint_from_val() -> None:
    if not markov_train_available():
        pytest.skip("markov train missing")
    tmv = train_minus_val_dir("5g_markov")
    vd = val_trace_dir("5g_markov")
    assert_disjoint_dirs(tmv, vd)


def test_harsh_family_calib_test_disjoint() -> None:
    calib, test = harsh_family_calib_test_dirs()
    assert_disjoint_dirs(calib, test)
