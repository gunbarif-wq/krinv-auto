from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ml_signal_common import PolicyConfig, hhmm_from_date_str, size_pct_for_signal


@dataclass
class OpenPosition:
    entry_i: int
    entry_price: float
    entry_open: float
    notional: float
    peak_price: float
    trough_price: float


def is_eod_flatten_bar(i: int, dates: List[str], cfg: PolicyConfig) -> bool:
    dt = dates[i]
    hhmm = hhmm_from_date_str(dt)
    if hhmm >= 0:
        close_total = 15 * 60 + 30
        cut_total = max(9 * 60, close_total - max(0, int(cfg.skip_close_min)))
        cut_hhmm = (cut_total // 60) * 100 + (cut_total % 60)
        if hhmm >= cut_hhmm:
            return True
    if i + 1 < len(dates):
        d0 = dt[:10] if len(dt) >= 10 else ""
        d1 = dates[i + 1][:10] if len(dates[i + 1]) >= 10 else ""
        if d0 and d1 and d0 != d1:
            return True
    return False


def run_policy(
    prob: np.ndarray,
    open_: np.ndarray,
    close: np.ndarray,
    vwap_gap_day: np.ndarray,
    dates: List[str],
    cfg: PolicyConfig,
    initial_cash: float,
    return_trades: bool = False,
) -> Dict[str, float | int | str | None]:
    cash = float(initial_cash)
    equity_curve: List[float] = [cash]
    trades = 0
    wins = 0
    trade_rets: List[float] = []
    gross_profit = 0.0
    gross_loss = 0.0
    first_ts: str | None = None
    last_ts: str | None = None
    loss_streak = 0
    cooldown_left = 0
    open_positions: List[OpenPosition] = []
    trade_logs: List[Dict[str, float | int | str]] = []
    last_entry_i = -10**9

    i = 0
    n = prob.shape[0]
    while i < n:
        if cooldown_left > 0:
            cooldown_left -= 1
            i += 1
            continue

        still_open: List[OpenPosition] = []
        for pos in open_positions:
            px = float(close[i])
            if px > pos.peak_price:
                pos.peak_price = px
            if px < pos.trough_price:
                pos.trough_price = px
            gross_ret = px / pos.entry_price - 1.0
            dd_from_peak = (px / pos.peak_price - 1.0) if pos.peak_price > 0 else 0.0
            exit_reason: str | None = None
            held = i - pos.entry_i
            if held >= max(1, int(cfg.min_hold_bars)):
                if cfg.exit_threshold > 0 and float(prob[i]) <= float(cfg.exit_threshold):
                    exit_reason = "score_drop"
                elif (
                    cfg.trailing_stop_pct > 0
                    and gross_ret >= max(float(cfg.trailing_activate_pct), float(cfg.trailing_stop_pct) * 1.5)
                    and dd_from_peak <= -cfg.trailing_stop_pct
                ):
                    exit_reason = "trailing_stop"
                elif (
                    i < vwap_gap_day.shape[0]
                    and float(vwap_gap_day[i]) <= 0.0
                    and held >= max(1, int(cfg.vwap_exit_min_hold_bars))
                    and gross_ret <= float(cfg.vwap_exit_max_profit_pct)
                ):
                    exit_reason = "vwap_break"
            if exit_reason is None and is_eod_flatten_bar(i, dates, cfg):
                exit_reason = "eod_flatten"
            if exit_reason is None:
                still_open.append(pos)
                continue

            r = gross_ret - cfg.fee_roundtrip
            cash += pos.notional * (1.0 + r)
            trades += 1
            if r > 0:
                wins += 1
                loss_streak = 0
                gross_profit += r
            else:
                loss_streak += 1
                gross_loss += -r
            if cfg.loss_streak_for_cooldown > 0 and cfg.cooldown_bars > 0 and loss_streak >= cfg.loss_streak_for_cooldown:
                cooldown_left = cfg.cooldown_bars
                loss_streak = 0
            trade_rets.append(r)
            first_ts = first_ts or dates[pos.entry_i]
            last_ts = dates[i]
            if return_trades:
                trade_logs.append(
                    {
                        "entry_i": pos.entry_i,
                        "exit_i": i,
                        "entry_time": dates[pos.entry_i],
                        "exit_time": dates[i],
                        "entry_price": pos.entry_price,
                        "entry_open": pos.entry_open,
                        "exit_price": px,
                        "return_pct": r * 100.0,
                        "bars_held": held,
                        "mfe_pct": (pos.peak_price / pos.entry_price - 1.0) * 100.0,
                        "mae_pct": (pos.trough_price / pos.entry_price - 1.0) * 100.0,
                        "reason": exit_reason,
                    }
                )
        open_positions = still_open

        can_enter = (
            prob[i] >= cfg.threshold
            and hhmm_from_date_str(dates[i]) >= 0
            and len(open_positions) < max(1, int(cfg.max_concurrent_positions))
            and (i - last_entry_i) >= max(0, int(cfg.min_entry_gap_bars))
            and i + 1 < n
        )
        if can_enter:
            hhmm = hhmm_from_date_str(dates[i])
            if hhmm >= 0:
                if hhmm < cfg.entry_start_hhmm or hhmm > cfg.entry_end_hhmm:
                    can_enter = False
                if cfg.skip_open_min > 0 and hhmm < 900 + cfg.skip_open_min:
                    can_enter = False
                if cfg.skip_close_min > 0:
                    close_total = 15 * 60 + 30
                    cut_total = max(9 * 60, close_total - cfg.skip_close_min)
                    cut_hhmm = (cut_total // 60) * 100 + (cut_total % 60)
                    if hhmm >= cut_hhmm:
                        can_enter = False
        if can_enter:
            size_pct = size_pct_for_signal(float(prob[i]), cfg)
            if size_pct <= 0.0:
                size_pct = min(1.0, max(0.01, float(cfg.position_size_pct)))
            notional = cash * size_pct
            if notional > 0:
                cash -= notional
                open_positions.append(
                    OpenPosition(
                        entry_i=i,
                        entry_price=float(close[i]),
                        entry_open=float(open_[i]) if i < open_.shape[0] else float(close[i]),
                        notional=notional,
                        peak_price=float(close[i]),
                        trough_price=float(close[i]),
                    )
                )
                last_entry_i = i

        open_value = 0.0
        px_now = float(close[i])
        for pos in open_positions:
            open_value += pos.notional * (px_now / pos.entry_price)
        equity_curve.append(cash + open_value)
        i += 1

    if n > 0 and open_positions:
        i = n - 1
        px = float(close[i])
        for pos in open_positions:
            r = (px / pos.entry_price - 1.0) - cfg.fee_roundtrip
            cash += pos.notional * (1.0 + r)
            trades += 1
            if r > 0:
                wins += 1
                gross_profit += r
            else:
                gross_loss += -r
            trade_rets.append(r)
            first_ts = first_ts or dates[pos.entry_i]
            last_ts = dates[i]
            if return_trades:
                trade_logs.append(
                    {
                        "entry_i": pos.entry_i,
                        "exit_i": i,
                        "entry_time": dates[pos.entry_i],
                        "exit_time": dates[i],
                        "entry_price": pos.entry_price,
                        "entry_open": pos.entry_open,
                        "exit_price": px,
                        "return_pct": r * 100.0,
                        "bars_held": i - pos.entry_i,
                        "mfe_pct": (pos.peak_price / pos.entry_price - 1.0) * 100.0,
                        "mae_pct": (pos.trough_price / pos.entry_price - 1.0) * 100.0,
                        "reason": "forced_eod",
                    }
                )
        equity_curve.append(cash)

    total_ret = cash / initial_cash - 1.0
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    avg_trade_ret = (float(np.mean(trade_rets)) * 100.0) if trade_rets else 0.0
    peak = -1.0
    mdd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = v / peak - 1.0
            if dd < mdd:
                mdd = dd
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else (999.0 if gross_profit > 0 else 0.0)
    out: Dict[str, float | int | str | None] = {
        "final_equity": cash,
        "total_return_pct": total_ret * 100.0,
        "trades": trades,
        "wins": wins,
        "win_rate_pct": win_rate,
        "avg_trade_return_pct": avg_trade_ret,
        "max_drawdown_pct": mdd * 100.0,
        "profit_factor": profit_factor,
        "first_signal_time": first_ts,
        "last_signal_time": last_ts,
    }
    if return_trades:
        out["trade_logs"] = json.dumps(trade_logs, ensure_ascii=False)
    return out
