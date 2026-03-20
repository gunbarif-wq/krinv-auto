from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest ML signal on prepared feature dataset")
    p.add_argument("--dataset-csv", default="data/ml/047810/047810_full_ml.csv")
    p.add_argument("--model-path", default="data/ml/047810/047810_model.pkl")
    p.add_argument("--threshold", type=float, default=0.80, help="override model threshold")
    p.add_argument("--fee-roundtrip", type=float, default=0.0004, help="override model fee")
    p.add_argument("--hold-bars", type=int, default=16, help="non-overlap holding bars")
    p.add_argument("--entry-start-hhmm", type=int, default=900)
    p.add_argument("--entry-end-hhmm", type=int, default=1530)
    p.add_argument("--skip-open-min", type=int, default=0, help="skip first N minutes after 09:00")
    p.add_argument("--skip-close-min", type=int, default=0, help="skip last N minutes before 15:30")
    p.add_argument("--loss-streak-for-cooldown", type=int, default=0, help="activate cooldown after N consecutive losses")
    p.add_argument("--cooldown-bars", type=int, default=0, help="cooldown bars after loss streak trigger")
    p.add_argument("--take-profit-pct", type=float, default=0.020)
    p.add_argument("--stop-loss-pct", type=float, default=0.005)
    p.add_argument("--trailing-stop-pct", type=float, default=0.004)
    p.add_argument("--max-concurrent-positions", type=int, default=1)
    p.add_argument("--position-size-pct", type=float, default=0.25)
    p.add_argument("--min-entry-gap-bars", type=int, default=2)
    p.add_argument("--initial-cash", type=float, default=10_000_000)
    p.add_argument("--report-out", default="data/ml/047810/047810_test_backtest.json")
    return p.parse_args()


def load_dataset(path: Path, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X: List[List[float]] = []
    close: List[float] = []
    dates: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                X.append([float(r[c]) for c in feature_cols])
                close.append(float(r["close"]))
                dates.append(r["date"])
            except Exception:
                continue
    if not X:
        raise RuntimeError(f"empty dataset: {path}")
    return np.asarray(X, dtype=float), np.asarray(close, dtype=float), dates


def max_drawdown(equity_curve: List[float]) -> float:
    peak = -1.0
    mdd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = v / peak - 1.0
            if dd < mdd:
                mdd = dd
    return mdd


@dataclass
class PolicyConfig:
    threshold: float
    fee_roundtrip: float
    hold_bars: int
    entry_start_hhmm: int = 900
    entry_end_hhmm: int = 1530
    skip_open_min: int = 0
    skip_close_min: int = 0
    loss_streak_for_cooldown: int = 0
    cooldown_bars: int = 0
    take_profit_pct: float = 0.0
    stop_loss_pct: float = 0.0
    trailing_stop_pct: float = 0.0
    max_concurrent_positions: int = 1
    position_size_pct: float = 1.0
    min_entry_gap_bars: int = 1


@dataclass
class OpenPosition:
    entry_i: int
    entry_price: float
    notional: float
    planned_exit_i: int
    peak_price: float


def hhmm_from_date_str(dt: str) -> int:
    if len(dt) < 16:
        return -1
    return int(dt[11:13]) * 100 + int(dt[14:16])


def in_entry_window(dt: str, cfg: PolicyConfig) -> bool:
    hhmm = hhmm_from_date_str(dt)
    if hhmm < 0:
        return False
    if hhmm < cfg.entry_start_hhmm or hhmm > cfg.entry_end_hhmm:
        return False
    if cfg.skip_open_min > 0:
        start_h = 9
        start_m = 0 + cfg.skip_open_min
        start_hhmm = start_h * 100 + start_m
        if hhmm < start_hhmm:
            return False
    if cfg.skip_close_min > 0:
        close_total = 15 * 60 + 30
        cut_total = max(9 * 60, close_total - cfg.skip_close_min)
        cut_hhmm = (cut_total // 60) * 100 + (cut_total % 60)
        if hhmm >= cut_hhmm:
            return False
    return True


def run_policy(
    prob: np.ndarray,
    close: np.ndarray,
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
    hold = max(1, int(cfg.hold_bars))
    n = prob.shape[0]
    while i < n:
        if cooldown_left > 0:
            cooldown_left -= 1
            i += 1
            continue

        # Exit evaluation for open positions
        still_open: List[OpenPosition] = []
        for pos in open_positions:
            px = float(close[i])
            if px > pos.peak_price:
                pos.peak_price = px
            gross_ret = px / pos.entry_price - 1.0
            dd_from_peak = (px / pos.peak_price - 1.0) if pos.peak_price > 0 else 0.0
            exit_reason: str | None = None
            if cfg.stop_loss_pct > 0 and gross_ret <= -cfg.stop_loss_pct:
                exit_reason = "stop_loss"
            elif cfg.take_profit_pct > 0 and gross_ret >= cfg.take_profit_pct:
                exit_reason = "take_profit"
            elif cfg.trailing_stop_pct > 0 and dd_from_peak <= -cfg.trailing_stop_pct:
                exit_reason = "trailing_stop"
            elif i >= pos.planned_exit_i:
                exit_reason = "timeout"

            if exit_reason is None:
                still_open.append(pos)
                continue

            r = gross_ret - cfg.fee_roundtrip
            proceeds = pos.notional * (1.0 + r)
            cash += proceeds
            trades += 1
            if r > 0:
                wins += 1
                loss_streak = 0
                gross_profit += r
            else:
                loss_streak += 1
                gross_loss += -r
            if (
                cfg.loss_streak_for_cooldown > 0
                and cfg.cooldown_bars > 0
                and loss_streak >= cfg.loss_streak_for_cooldown
            ):
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
                        "exit_price": px,
                        "return_pct": r * 100.0,
                        "reason": exit_reason,
                    }
                )
        open_positions = still_open

        # Entry evaluation (overlapping positions allowed)
        can_enter = (
            prob[i] >= cfg.threshold
            and in_entry_window(dates[i], cfg)
            and len(open_positions) < max(1, int(cfg.max_concurrent_positions))
            and (i - last_entry_i) >= max(1, int(cfg.min_entry_gap_bars))
            and i + 1 < n
        )
        if can_enter:
            size_pct = min(1.0, max(0.01, float(cfg.position_size_pct)))
            notional = cash * size_pct
            if notional > 0:
                cash -= notional
                entry_price = float(close[i])
                open_positions.append(
                    OpenPosition(
                        entry_i=i,
                        entry_price=entry_price,
                        notional=notional,
                        planned_exit_i=min(i + hold, n - 1),
                        peak_price=entry_price,
                    )
                )
                last_entry_i = i

        # Mark-to-market equity
        open_value = 0.0
        px_now = float(close[i])
        for pos in open_positions:
            open_value += pos.notional * (px_now / pos.entry_price)
        equity_curve.append(cash + open_value)
        i += 1

    # Force close remaining positions at last bar
    if n > 0 and open_positions:
        i = n - 1
        px = float(close[i])
        for pos in open_positions:
            r = (px / pos.entry_price - 1.0) - cfg.fee_roundtrip
            proceeds = pos.notional * (1.0 + r)
            cash += proceeds
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
                        "exit_price": px,
                        "return_pct": r * 100.0,
                        "reason": "forced_eod",
                    }
                )
        equity_curve.append(cash)

    total_ret = cash / initial_cash - 1.0
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    avg_trade_ret = (float(np.mean(trade_rets)) * 100.0) if trade_rets else 0.0
    mdd = max_drawdown(equity_curve) * 100.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-12 else (999.0 if gross_profit > 0 else 0.0)
    out: Dict[str, float | int | str | None] = {
        "final_equity": cash,
        "total_return_pct": total_ret * 100.0,
        "trades": trades,
        "wins": wins,
        "win_rate_pct": win_rate,
        "avg_trade_return_pct": avg_trade_ret,
        "max_drawdown_pct": mdd,
        "profit_factor": profit_factor,
        "first_signal_time": first_ts,
        "last_signal_time": last_ts,
    }
    if return_trades:
        out["trade_logs"] = json.dumps(trade_logs, ensure_ascii=False)
    return out


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    with model_path.open("rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    feature_cols: List[str] = bundle["feature_columns"]
    threshold = float(args.threshold if args.threshold is not None else bundle.get("threshold", 0.60))
    fee = float(args.fee_roundtrip if args.fee_roundtrip is not None else bundle.get("fee_roundtrip", 0.0004))

    X, close, dates = load_dataset(Path(args.dataset_csv), feature_cols)
    prob = model.predict_proba(X)[:, 1]

    cfg = PolicyConfig(
        threshold=threshold,
        fee_roundtrip=fee,
        hold_bars=args.hold_bars,
        entry_start_hhmm=args.entry_start_hhmm,
        entry_end_hhmm=args.entry_end_hhmm,
        skip_open_min=max(0, int(args.skip_open_min)),
        skip_close_min=max(0, int(args.skip_close_min)),
        loss_streak_for_cooldown=max(0, int(args.loss_streak_for_cooldown)),
        cooldown_bars=max(0, int(args.cooldown_bars)),
        take_profit_pct=max(0.0, float(args.take_profit_pct)),
        stop_loss_pct=max(0.0, float(args.stop_loss_pct)),
        trailing_stop_pct=max(0.0, float(args.trailing_stop_pct)),
        max_concurrent_positions=max(1, int(args.max_concurrent_positions)),
        position_size_pct=min(1.0, max(0.01, float(args.position_size_pct))),
        min_entry_gap_bars=max(1, int(args.min_entry_gap_bars)),
    )
    result = run_policy(prob=prob, close=close, dates=dates, cfg=cfg, initial_cash=float(args.initial_cash))

    report = {
        "dataset_csv": str(args.dataset_csv),
        "model_path": str(model_path),
        "threshold": threshold,
        "fee_roundtrip": fee,
        "hold_bars": int(cfg.hold_bars),
        "entry_start_hhmm": cfg.entry_start_hhmm,
        "entry_end_hhmm": cfg.entry_end_hhmm,
        "skip_open_min": cfg.skip_open_min,
        "skip_close_min": cfg.skip_close_min,
        "loss_streak_for_cooldown": cfg.loss_streak_for_cooldown,
        "cooldown_bars": cfg.cooldown_bars,
        "take_profit_pct": cfg.take_profit_pct,
        "stop_loss_pct": cfg.stop_loss_pct,
        "trailing_stop_pct": cfg.trailing_stop_pct,
        "max_concurrent_positions": cfg.max_concurrent_positions,
        "position_size_pct": cfg.position_size_pct,
        "min_entry_gap_bars": cfg.min_entry_gap_bars,
        "initial_cash": args.initial_cash,
        **result,
    }

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"report_saved={report_out}")
    print(
        f"return={report['total_return_pct']:.2f}% trades={int(report['trades'])} "
        f"win_rate={report['win_rate_pct']:.2f}% avg_trade={report['avg_trade_return_pct']:.4f}% "
        f"mdd={report['max_drawdown_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
