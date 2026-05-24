"""Tests for early-stopping by metric threshold."""

from __future__ import annotations

from drp_warm.callbacks import EarlyStoppingByMetric


def test_below_mode_triggers_when_value_reaches_target():
    stop = EarlyStoppingByMetric(target=0.05, mode="below")
    assert not stop.should_stop(0.10)
    assert not stop.should_stop(0.051)
    assert stop.should_stop(0.05)
    assert stop.should_stop(0.04)


def test_above_mode_triggers_when_value_reaches_target():
    stop = EarlyStoppingByMetric(target=0.70, mode="above")
    assert not stop.should_stop(0.50)
    assert not stop.should_stop(0.69)
    assert stop.should_stop(0.70)
    assert stop.should_stop(0.75)


def test_default_mode_is_below():
    stop = EarlyStoppingByMetric(target=1.0)
    assert stop.should_stop(0.5)
    assert not stop.should_stop(2.0)
