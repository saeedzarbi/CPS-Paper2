"""Train/val/test disjointness for Tube-v2 trace sets."""
from __future__ import annotations

import pytest

from configs.paths import TRAIN_TRACES_5G_IRISH_MMSYS20_DIR
from configs.trace_sets_v2 import (
    assert_disjoint_dirs,
    harsh_family_three_way_dirs,
    markov_train_available,
    resolve_trace_dir,
    split_dirs_for_trace_set,
    train_minus_val_dir,
    val_trace_dir,
)


def test_markov_train_minus_val_disjoint_from_val() -> None:
    if not markov_train_available():
        pytest.skip("markov train missing")
    tmv = train_minus_val_dir("5g_markov")
    vd = val_trace_dir("5g_markov")
    assert_disjoint_dirs(tmv, vd)


def test_harsh_family_calib_val_test_disjoint() -> None:
    calib, val, test = harsh_family_three_way_dirs()
    assert_disjoint_dirs(calib, val)
    assert_disjoint_dirs(calib, test)
    assert_disjoint_dirs(val, test)


def test_harsh_val_not_same_as_test_dir() -> None:
    val_dir = resolve_trace_dir("5g_harsh", "val")
    test_dir = resolve_trace_dir("5g_harsh", "test")
    assert val_dir != test_dir
    assert_disjoint_dirs(val_dir, test_dir)


@pytest.mark.parametrize("trace_set", ["5g_markov", "5g_harsh", "bb", "5g_irish"])
def test_calib_val_test_pairwise_disjoint(trace_set: str) -> None:
    if trace_set == "5g_irish" and not TRAIN_TRACES_5G_IRISH_MMSYS20_DIR.is_dir():
        pytest.skip("irish train missing")
    if trace_set == "5g_markov" and not markov_train_available():
        pytest.skip("markov train missing")
    try:
        calib, val, test = split_dirs_for_trace_set(trace_set)
    except FileNotFoundError as e:
        pytest.skip(str(e))
    assert_disjoint_dirs(calib, val)
    assert_disjoint_dirs(calib, test)
    assert_disjoint_dirs(val, test)
