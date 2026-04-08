from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ml_backtest_common import run_policy
from ml_signal_common import PolicyConfig, ema_smooth, load_json, load_model_bundle
from ml_trade_common import build_policy_config, load_feature_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest ML signal on prepared feature dataset")
    p.add_argument("--dataset-csv", default="data/ml/225190_1y/225190_test_ml.csv")
    p.add_argument("--model-path", default="data/ml/225190_1y/225190_model_fast.pkl")
    p.add_argument("--policy-path", default="data/ml/225190_1y/225190_fast_policy.json")
    p.add_argument("--threshold", type=float, default=0.75, help="entry threshold (relprice default)")
    p.add_argument("--size-low-pct", type=float, default=0.25)
    p.add_argument("--size-mid-pct", type=float, default=0.50)
    p.add_argument("--size-high-pct", type=float, default=1.00)
    p.add_argument("--size-mid-threshold", type=float, default=0.82)
    p.add_argument("--size-high-threshold", type=float, default=0.87)
    p.add_argument("--fee-roundtrip", type=float, default=0.004, help="roundtrip fee (relprice default)")
    p.add_argument("--min-hold-bars", type=int, default=5, help="minimum holding bars before any rule-based exit can happen")
    p.add_argument("--exit-threshold", type=float, default=0.55, help="score-drop exit threshold")
    p.add_argument("--entry-start-hhmm", type=int, default=900)
    p.add_argument("--entry-end-hhmm", type=int, default=1530)
    p.add_argument("--skip-open-min", type=int, default=9, help="skip first N minutes after 09:00")
    p.add_argument("--skip-close-min", type=int, default=19, help="skip last N minutes before 15:30")
    p.add_argument("--loss-streak-for-cooldown", type=int, default=3, help="activate cooldown after N consecutive losses")
    p.add_argument("--cooldown-bars", type=int, default=24, help="cooldown bars after loss streak trigger")
    p.add_argument("--trailing-stop-pct", type=float, default=0.0073)
    p.add_argument("--trailing-activate-pct", type=float, default=0.0155, help="activate trailing stop only after this profit pct")
    p.add_argument("--vwap-exit-min-hold-bars", type=int, default=19, help="minimum bars held before vwap_break can exit")
    p.add_argument("--vwap-exit-max-profit-pct", type=float, default=-0.0036, help="only allow vwap_break exit when profit is at or below this pct")
    p.add_argument("--max-concurrent-positions", type=int, default=1)
    p.add_argument("--position-size-pct", type=float, default=1.0, help="fallback fixed size if banded sizing is not configured")
    p.add_argument("--min-entry-gap-bars", type=int, default=0)
    p.add_argument("--no-benchmark-simple", action="store_true", help="disable indicator-only benchmarks")
    p.add_argument("--initial-cash", type=float, default=10_000_000)
    p.add_argument("--report-out", default="data/ml/225190_1y/225190_test_backtest_fast.json")
    p.add_argument("--trades-out", default="", help="optional json path to save detailed trade logs/analysis")
    return p.parse_args()


def load_dataset(path: Path, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X: List[List[float]] = []
    opens: List[float] = []
    close: List[float] = []
    dates: List[str] = []
    vwap_gap_day: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                X.append([float(r[c]) for c in feature_cols])
                opens.append(float(r["open"]))
                close.append(float(r["close"]))
                dates.append(r["date"])
                vwap_gap_day.append(float(r["vwap_gap_day"]) if "vwap_gap_day" in r else 0.0)
            except Exception:
                continue
    if not X:
        raise RuntimeError(f"empty dataset: {path}")
    return np.asarray(X, dtype=float), np.asarray(opens, dtype=float), np.asarray(close, dtype=float), dates, np.asarray(vwap_gap_day, dtype=float)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    bundle = load_model_bundle(model_path)

    model = bundle["model"]
    feature_cols: List[str] = bundle["feature_columns"]
    policy_path = Path(args.policy_path)
    policy = load_json(policy_path)
    threshold = float(policy.get("threshold", args.threshold if args.threshold is not None else bundle.get("threshold", 0.90)))
    fee = float(policy.get("fee_roundtrip", args.fee_roundtrip if args.fee_roundtrip is not None else bundle.get("fee_roundtrip", 0.004)))

    X, open_, close, dates, vwap_gap_day = load_feature_dataset(Path(args.dataset_csv), feature_cols)
    prob = model.predict_proba(X)[:, 1]
    prob = ema_smooth(prob, 3)

    cfg = build_policy_config(
        policy,
        threshold=threshold,
        fee_roundtrip=fee,
        min_hold_bars=args.min_hold_bars,
        exit_threshold=args.exit_threshold,
        entry_start_hhmm=args.entry_start_hhmm,
        entry_end_hhmm=args.entry_end_hhmm,
        skip_open_min=args.skip_open_min,
        skip_close_min=args.skip_close_min,
        loss_streak_for_cooldown=args.loss_streak_for_cooldown,
        cooldown_bars=args.cooldown_bars,
        trailing_stop_pct=args.trailing_stop_pct,
        trailing_activate_pct=args.trailing_activate_pct,
        vwap_exit_min_hold_bars=args.vwap_exit_min_hold_bars,
        vwap_exit_max_profit_pct=args.vwap_exit_max_profit_pct,
        max_concurrent_positions=args.max_concurrent_positions,
        position_size_pct=args.position_size_pct,
        min_entry_gap_bars=args.min_entry_gap_bars,
    )
    need_trade_logs = bool(str(args.trades_out).strip())
    result = run_policy(
        prob=prob,
        open_=open_,
        close=close,
        vwap_gap_day=vwap_gap_day,
        dates=dates,
        cfg=cfg,
        initial_cash=float(args.initial_cash),
        return_trades=need_trade_logs,
    )
    report = {
        "dataset_csv": str(args.dataset_csv),
        "model_path": str(model_path),
        "policy_path": str(policy_path),
        "threshold": threshold,
        "fee_roundtrip": fee,
        "size_low_pct": float(cfg.size_low_pct),
        "size_mid_pct": float(cfg.size_mid_pct),
        "size_high_pct": float(cfg.size_high_pct),
        "size_mid_threshold": float(cfg.size_mid_threshold),
        "size_high_threshold": float(cfg.size_high_threshold),
        "min_hold_bars": int(cfg.min_hold_bars),
        "exit_threshold": float(cfg.exit_threshold),
        "entry_start_hhmm": cfg.entry_start_hhmm,
        "entry_end_hhmm": cfg.entry_end_hhmm,
        "skip_open_min": cfg.skip_open_min,
        "skip_close_min": cfg.skip_close_min,
        "loss_streak_for_cooldown": cfg.loss_streak_for_cooldown,
        "cooldown_bars": cfg.cooldown_bars,
        "trailing_stop_pct": cfg.trailing_stop_pct,
        "trailing_activate_pct": cfg.trailing_activate_pct,
        "vwap_exit_min_hold_bars": cfg.vwap_exit_min_hold_bars,
        "vwap_exit_max_profit_pct": cfg.vwap_exit_max_profit_pct,
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
    if need_trade_logs:
        trade_logs = json.loads(str(result.get("trade_logs", "[]")))
        by_reason: Dict[str, int] = {}
        hold_vals: List[float] = []
        ret_vals: List[float] = []
        mfe_vals: List[float] = []
        for t in trade_logs:
            reason = str(t.get("reason", "unknown"))
            by_reason[reason] = by_reason.get(reason, 0) + 1
            hold_vals.append(float(t.get("bars_held", 0)))
            ret_vals.append(float(t.get("return_pct", 0.0)))
            mfe_vals.append(float(t.get("mfe_pct", 0.0)))
        avg_missed_peak = 0.0
        if ret_vals and mfe_vals:
            avg_missed_peak = float(np.mean(np.asarray(mfe_vals) - np.asarray(ret_vals)))
        trade_dump = {
            "summary": {
                "count": len(trade_logs),
                "avg_bars_held": float(np.mean(hold_vals)) if hold_vals else 0.0,
                "avg_missed_peak_pct": avg_missed_peak,
                "exit_reason_counts": by_reason,
            },
            "trades": trade_logs,
        }
        trades_out = Path(str(args.trades_out))
        trades_out.parent.mkdir(parents=True, exist_ok=True)
        trades_out.write_text(json.dumps(trade_dump, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"trades_saved={trades_out}")


if __name__ == "__main__":
    main()
