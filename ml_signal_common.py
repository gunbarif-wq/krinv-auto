from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class PolicyConfig:
    threshold: float
    fee_roundtrip: float
    hold_bars: int
    entry_confirm_bars: int = 1
    size_mid_threshold: float = 0.0
    size_high_threshold: float = 0.0
    size_low_pct: float = 1.0
    size_mid_pct: float = 1.0
    size_high_pct: float = 1.0
    min_score_delta: float = 0.0
    min_hold_bars: int = 1
    exit_threshold: float = 0.0
    entry_start_hhmm: int = 900
    entry_end_hhmm: int = 1530
    skip_open_min: int = 0
    skip_close_min: int = 0
    loss_streak_for_cooldown: int = 0
    cooldown_bars: int = 0
    take_profit_pct: float = 0.0
    stop_loss_pct: float = 0.0
    trailing_stop_pct: float = 0.0
    trailing_activate_pct: float = 0.0
    vwap_exit_min_hold_bars: int = 0
    vwap_exit_max_profit_pct: float = 0.0
    max_concurrent_positions: int = 1
    position_size_pct: float = 1.0
    min_entry_gap_bars: int = 0


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_policy(path: Path) -> Dict[str, Any]:
    return load_json(path)


def load_model_bundle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def smoothstep(t: float) -> float:
    t = clamp01(t)
    return t * t * (3.0 - 2.0 * t)


def size_pct_for_signal(prob: float, cfg: PolicyConfig) -> float:
    low = max(0.01, min(1.0, float(cfg.size_low_pct)))
    high = max(low, min(1.0, float(cfg.size_high_pct)))
    lo_thr = max(0.0, float(cfg.threshold))
    hi_thr = max(lo_thr + 1e-6, float(cfg.size_high_threshold))
    if prob <= lo_thr:
        return 0.0
    if prob >= hi_thr:
        return high
    t = (prob - lo_thr) / (hi_thr - lo_thr)
    return low + (high - low) * smoothstep(t)


def hhmm_from_date_str(dt: str) -> int:
    if len(dt) < 16:
        return -1
    return int(dt[11:13]) * 100 + int(dt[14:16])


def normalize_indicator(arr: np.ndarray, mode: str, window: int) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if mode == "local_rank01":
        n = x.shape[0]
        w = max(10, int(window))
        out = np.full(n, 0.5, dtype=float)
        for i in range(n):
            s = max(0, i - w + 1)
            seg = x[s : i + 1]
            if seg.size <= 1:
                out[i] = 0.5
                continue
            v = x[i]
            out[i] = float(np.sum(seg <= v)) / float(seg.size)
        return np.clip(out, 0.0, 1.0)
    lo = float(np.nanpercentile(x, 1))
    hi = float(np.nanpercentile(x, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full(x.shape[0], 0.5, dtype=float)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def ema_smooth(arr: np.ndarray, span: int = 3) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return x
    span = max(1, int(span))
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out
