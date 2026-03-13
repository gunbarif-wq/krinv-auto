from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SymbolState:
    has_position: bool = False
    held_qty: int = 0
    entry_price: float = 0.0
    entry_total_cost: float = 0.0
    held_bars: int = 0
    cooldown_left: int = 0
    entry_streak: int = 0
    exit_streak: int = 0


def required_bars_for_signal(args) -> int:
    if args.strategy_mode == "ma_cross_level":
        return max(max(args.ma_a, args.ma_b) + 1, args.cross_level_window)
    return max(args.long, args.mom_window + 1, args.stoch_window + args.stoch_smooth)


def _stoch_k(highs: List[float], lows: List[float], closes: List[float], window: int) -> float | None:
    if len(closes) < window:
        return None
    hh = max(highs[-window:])
    ll = min(lows[-window:])
    if hh == ll:
        return 50.0
    return (closes[-1] - ll) / (hh - ll) * 100.0


def multi_factor_signal(
    ohlc: List[Dict[str, float]],
    short: int,
    long: int,
    mom_window: int,
    stoch_window: int,
    stoch_smooth: int,
    entry_threshold: float,
    exit_threshold: float,
) -> Tuple[int, Dict[str, float]]:
    closes = [x["close"] for x in ohlc]
    highs = [x["high"] for x in ohlc]
    lows = [x["low"] for x in ohlc]
    need = max(long, mom_window + 1, stoch_window + stoch_smooth)
    if len(closes) < need:
        return 0, {"score": 0.0}

    short_sma = sum(closes[-short:]) / short
    long_sma = sum(closes[-long:]) / long
    trend = 1.0 if short_sma > long_sma else -1.0

    mom = (closes[-1] / closes[-1 - mom_window]) - 1.0
    mom_score = max(-1.0, min(1.0, mom / 0.01))

    k_vals: List[float] = []
    for i in range(stoch_smooth):
        end = len(closes) - i
        k = _stoch_k(highs[:end], lows[:end], closes[:end], stoch_window)
        if k is not None:
            k_vals.append(k)
    if not k_vals:
        return 0, {"score": 0.0}
    k_now = k_vals[0]
    d_now = sum(k_vals) / len(k_vals)
    stoch_bias = (50.0 - k_now) / 50.0
    cross = 0.3 if k_now > d_now else -0.3

    score = 0.45 * trend + 0.35 * mom_score + 0.15 * stoch_bias + 0.05 * cross
    if score >= entry_threshold:
        signal = 1
    elif score <= exit_threshold:
        signal = -1
    else:
        signal = 0
    return signal, {
        "score": score,
        "mom": mom,
        "k": k_now,
        "d": d_now,
        "short_sma": short_sma,
        "long_sma": long_sma,
    }


def ma_cross_level_signal(
    ohlc: List[Dict[str, float]],
    ma_a: int,
    ma_b: int,
    level_window: int,
    buy_level: float,
    sell_level: float,
) -> Tuple[int, Dict[str, float]]:
    if ma_a <= 0 or ma_b <= 0:
        return 0, {"score": 0.0}
    short = min(ma_a, ma_b)
    long = max(ma_a, ma_b)
    need = max(long + 1, level_window)

    closes = [x["close"] for x in ohlc]
    if len(closes) < need:
        return 0, {"score": 0.0}

    short_now = sum(closes[-short:]) / short
    long_now = sum(closes[-long:]) / long
    short_prev = sum(closes[-1 - short : -1]) / short
    long_prev = sum(closes[-1 - long : -1]) / long
    spread_prev = short_prev - long_prev
    spread_now = short_now - long_now

    recent = closes[-level_window:]
    lo = min(recent)
    hi = max(recent)
    if hi == lo:
        level = 0.5
    else:
        level = (closes[-1] - lo) / (hi - lo)
    level = max(0.0, min(1.0, level))

    cross_up = spread_prev <= 0 and spread_now > 0
    cross_down = spread_prev >= 0 and spread_now < 0
    if cross_up and level >= buy_level:
        signal = 1
    elif cross_down and level <= sell_level:
        signal = -1
    else:
        signal = 0
    spread_pct = 0.0 if long_now == 0 else (spread_now / long_now) * 100.0
    return signal, {"score": level, "ma_short": short_now, "ma_long": long_now, "spread_pct": spread_pct}


def evaluate_state_transition(
    state: SymbolState,
    signal: int,
    last_px: float,
    strategy_mode: str,
    stop_loss_pct: float,
    take_profit_pct: float,
    entry_confirm_bars: int,
    exit_confirm_bars: int,
    min_hold_bars: int,
    hard_liquidation_window: bool,
    ease_sell_window: bool = False,
    ease_ratio: float = 0.0,
) -> Dict[str, object]:
    sell_reason = "signal"

    if state.cooldown_left > 0:
        state.cooldown_left -= 1
        if signal == 1:
            signal = 0

    if state.has_position and state.entry_price > 0:
        state.held_bars += 1
        pnl_pct = (last_px / state.entry_price) - 1.0
        if pnl_pct <= -stop_loss_pct:
            signal = -1
            sell_reason = "stop_loss"
        elif pnl_pct >= take_profit_pct:
            signal = -1
            sell_reason = "take_profit"
        if hard_liquidation_window:
            signal = -1
            sell_reason = "hard_close"

    if signal == 1:
        state.entry_streak += 1
        state.exit_streak = 0
    elif signal == -1:
        state.exit_streak += 1
        state.entry_streak = 0
    else:
        state.entry_streak = 0
        state.exit_streak = 0

    entry_confirm_req = entry_confirm_bars if strategy_mode == "multi_factor" else 1
    exit_confirm_req = exit_confirm_bars if strategy_mode == "multi_factor" else 1
    min_hold_req = min_hold_bars
    if ease_sell_window:
        min_hold_req = max(0, int(round(min_hold_bars * (1.0 - ease_ratio))))
        if strategy_mode == "multi_factor":
            exit_confirm_req = max(1, int(round(exit_confirm_bars * (1.0 - ease_ratio))))
    if hard_liquidation_window:
        min_hold_req = 0
        exit_confirm_req = 1

    buy_ready = False
    sell_ready = False
    entry_wait: Tuple[int, int] | None = None
    hold_wait: Tuple[int, int] | None = None
    exit_wait: Tuple[int, int] | None = None

    if signal == 1 and not state.has_position:
        if state.entry_streak >= entry_confirm_req:
            buy_ready = True
        else:
            entry_wait = (state.entry_streak, entry_confirm_req)
    elif signal == -1 and state.has_position:
        if state.held_bars < min_hold_req:
            hold_wait = (state.held_bars, min_hold_req)
        elif state.exit_streak < exit_confirm_req:
            exit_wait = (state.exit_streak, exit_confirm_req)
        else:
            sell_ready = True

    return {
        "signal": signal,
        "sell_reason": sell_reason,
        "buy_ready": buy_ready,
        "sell_ready": sell_ready,
        "entry_wait": entry_wait,
        "hold_wait": hold_wait,
        "exit_wait": exit_wait,
    }
