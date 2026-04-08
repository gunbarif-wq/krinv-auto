from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ml_signal_common import PolicyConfig


def load_feature_dataset(path: Path, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    x: List[List[float]] = []
    open_: List[float] = []
    close: List[float] = []
    dates: List[str] = []
    vwap_gap_day: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                x.append([float(r[c]) for c in feature_cols])
                open_.append(float(r["open"]))
                close.append(float(r["close"]))
                dates.append(r["date"])
                vwap_gap_day.append(float(r.get("vwap_gap_day", 0.0)))
            except Exception:
                continue
    if not x:
        raise RuntimeError(f"empty dataset: {path}")
    return (
        np.asarray(x, dtype=float),
        np.asarray(open_, dtype=float),
        np.asarray(close, dtype=float),
        dates,
        np.asarray(vwap_gap_day, dtype=float),
    )


def build_policy_config(
    policy: Dict[str, object],
    *,
    threshold: float,
    fee_roundtrip: float,
    min_hold_bars: int,
    exit_threshold: float,
    entry_start_hhmm: int,
    entry_end_hhmm: int,
    skip_open_min: int,
    skip_close_min: int,
    loss_streak_for_cooldown: int,
    cooldown_bars: int,
    trailing_stop_pct: float,
    trailing_activate_pct: float,
    vwap_exit_min_hold_bars: int,
    vwap_exit_max_profit_pct: float,
    max_concurrent_positions: int,
    position_size_pct: float,
    min_entry_gap_bars: int,
) -> PolicyConfig:
    return PolicyConfig(
        threshold=threshold,
        fee_roundtrip=fee_roundtrip,
        hold_bars=0,
        min_hold_bars=max(1, int(min_hold_bars)),
        exit_threshold=float(policy.get("exit_threshold", exit_threshold)),
        entry_start_hhmm=int(policy.get("entry_start_hhmm", entry_start_hhmm)),
        entry_end_hhmm=int(policy.get("entry_end_hhmm", entry_end_hhmm)),
        skip_open_min=max(0, int(policy.get("skip_open_min", skip_open_min))),
        skip_close_min=max(0, int(policy.get("skip_close_min", skip_close_min))),
        loss_streak_for_cooldown=max(0, int(policy.get("loss_streak_for_cooldown", loss_streak_for_cooldown))),
        cooldown_bars=max(0, int(policy.get("cooldown_bars", cooldown_bars))),
        trailing_stop_pct=max(0.0, float(policy.get("trailing_stop_pct", trailing_stop_pct))),
        trailing_activate_pct=max(0.0, float(policy.get("trailing_activate_pct", trailing_activate_pct))),
        vwap_exit_min_hold_bars=max(0, int(policy.get("vwap_exit_min_hold_bars", vwap_exit_min_hold_bars))),
        vwap_exit_max_profit_pct=float(policy.get("vwap_exit_max_profit_pct", vwap_exit_max_profit_pct)),
        max_concurrent_positions=max(1, int(policy.get("max_concurrent_positions", max_concurrent_positions))),
        position_size_pct=min(1.0, max(0.01, float(policy.get("position_size_pct", position_size_pct)))),
        min_entry_gap_bars=max(0, int(min_entry_gap_bars)),
    )
