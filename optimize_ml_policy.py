from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from backtest_ml_signal import PolicyConfig, run_policy
from ml_signal_common import ema_smooth, load_model_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-optimize ML policy parameters on validation set")
    p.add_argument("--model-path", default="data/ml/225190_1y/225190_model_fast.pkl")
    p.add_argument("--val-csv", default="data/ml/225190_1y/225190_val_ml.csv")
    p.add_argument("--test-csv", default="data/ml/225190_1y/225190_test_ml.csv")
    p.add_argument("--symbol", default="225190")
    p.add_argument("--initial-cash", type=float, default=10_000_000)
    p.add_argument("--mdd-penalty", type=float, default=0.12, help="score = return_pct - penalty*abs(mdd_pct)")
    p.add_argument("--min-trades", type=int, default=10)
    p.add_argument("--max-trades", type=int, default=10**9)
    p.add_argument("--max-evals", type=int, default=5000, help="randomly sample combos if grid is bigger")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold-range", default="0.50,0.95")
    p.add_argument("--skip-open-range", default="0,30")
    p.add_argument("--skip-close-range", default="0,20")
    p.add_argument("--loss-streak-range", default="0,3")
    p.add_argument("--cooldown-range", default="0,60")
    p.add_argument("--trailing-stop-range", default="0.004,0.010", help="min,max trailing stop pct")
    p.add_argument("--trailing-activate-range", default="0.004,0.020", help="min,max profit pct before trailing stop activates")
    p.add_argument("--vwap-exit-hold-range", default="0,20", help="min,max bars held before vwap exit is allowed")
    p.add_argument("--vwap-exit-profit-range", default="-0.010,0.005", help="min,max profit pct allowed for vwap exit")
    p.add_argument("--entry-start-hhmm", type=int, default=900)
    p.add_argument("--entry-end-hhmm", type=int, default=1530)
    p.add_argument("--min-hold-bars", type=int, default=5)
    p.add_argument("--position-size-pct", type=float, default=0.5, help="position size fraction stored into policy")
    p.add_argument("--report-out", default="data/ml/225190_1y/225190_policy_opt_report.json")
    p.add_argument("--policy-out", default="data/ml/225190_1y/225190_fast_policy.json")
    return p.parse_args()


def parse_float_spec(s: str) -> Dict[str, object]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if len(vals) != 2:
        raise RuntimeError("float range must contain exactly two values: lo,hi")
    a, b = float(vals[0]), float(vals[1])
    lo, hi = (a, b) if a <= b else (b, a)
    return {"kind": "range", "lo": lo, "hi": hi}


def parse_int_spec(s: str) -> Dict[str, object]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if len(vals) != 2:
        raise RuntimeError("int range must contain exactly two values: lo,hi")
    a, b = int(vals[0]), int(vals[1])
    lo, hi = (a, b) if a <= b else (b, a)
    return {"kind": "range", "lo": lo, "hi": hi}


def load_dataset(path: Path, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
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
                vwap_gap_day.append(float(r.get("vwap_gap_day", 0.0)))
            except Exception:
                continue
    if not X:
        raise RuntimeError(f"empty dataset: {path}")
    return np.asarray(X, dtype=float), np.asarray(opens, dtype=float), np.asarray(close, dtype=float), dates, np.asarray(vwap_gap_day, dtype=float)


def policy_score(result: Dict[str, float | int | str | None], mdd_penalty: float) -> float:
    total_return = float(result["total_return_pct"])  # type: ignore[arg-type]
    max_drawdown = abs(float(result["max_drawdown_pct"]))  # type: ignore[arg-type]
    return total_return - float(mdd_penalty) * max_drawdown


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    bundle = load_model_bundle(Path(args.model_path))
    model = bundle["model"]
    feature_cols: List[str] = bundle["feature_columns"]
    fee = float(bundle.get("fee_roundtrip", 0.001))

    X_val, open_val, close_val, dates_val, vwap_val = load_dataset(Path(args.val_csv), feature_cols)
    X_test, open_test, close_test, dates_test, vwap_test = load_dataset(Path(args.test_csv), feature_cols)
    p_val = ema_smooth(model.predict_proba(X_val)[:, 1], 3)
    p_test = ema_smooth(model.predict_proba(X_test)[:, 1], 3)

    thr_spec = parse_float_spec(args.threshold_range)
    open_spec = parse_int_spec(args.skip_open_range)
    close_spec = parse_int_spec(args.skip_close_range)
    streak_spec = parse_int_spec(args.loss_streak_range)
    cooldown_spec = parse_int_spec(args.cooldown_range)
    trail_spec = parse_float_spec(args.trailing_stop_range)
    trail_act_spec = parse_float_spec(args.trailing_activate_range)
    vwap_hold_spec = parse_int_spec(args.vwap_exit_hold_range)
    vwap_profit_spec = parse_float_spec(args.vwap_exit_profit_range)

    def sample_float_spec(spec: Dict[str, object]) -> float:
        if spec["kind"] == "range":
            lo = float(spec["lo"])
            hi = float(spec["hi"])
            return rng.uniform(lo, hi)
        return float(rng.choice(list(spec["values"])))  # type: ignore[arg-type]

    def sample_int_spec(spec: Dict[str, object]) -> int:
        if spec["kind"] == "range":
            lo = int(spec["lo"])
            hi = int(spec["hi"])
            return rng.randint(lo, hi)
        return int(rng.choice(list(spec["values"])))  # type: ignore[arg-type]

    use_sampling = any(
        spec["kind"] == "range"
        for spec in [thr_spec, open_spec, close_spec, streak_spec, cooldown_spec, trail_spec, trail_act_spec, vwap_hold_spec, vwap_profit_spec]
    )
    combos: List[Tuple[float, int, int, int, int, float, float, int, float]] = []
    if use_sampling:
        sample_n = args.max_evals if args.max_evals > 0 else 1500
        for _ in range(sample_n):
            combos.append(
                (
                    sample_float_spec(thr_spec),
                    sample_int_spec(open_spec),
                    sample_int_spec(close_spec),
                    sample_int_spec(streak_spec),
                    sample_int_spec(cooldown_spec),
                    sample_float_spec(trail_spec),
                    sample_float_spec(trail_act_spec),
                    sample_int_spec(vwap_hold_spec),
                    sample_float_spec(vwap_profit_spec),
                )
            )
    else:
        thr_grid = list(thr_spec["values"])  # type: ignore[index]
        open_grid = list(open_spec["values"])  # type: ignore[index]
        close_grid = list(close_spec["values"])  # type: ignore[index]
        streak_grid = list(streak_spec["values"])  # type: ignore[index]
        cooldown_grid = list(cooldown_spec["values"])  # type: ignore[index]
        trail_grid = list(trail_spec["values"])  # type: ignore[index]
        trail_act_grid = list(trail_act_spec["values"])  # type: ignore[index]
        vwap_hold_grid = list(vwap_hold_spec["values"])  # type: ignore[index]
        vwap_profit_grid = list(vwap_profit_spec["values"])  # type: ignore[index]
        combos = list(itertools.product(thr_grid, open_grid, close_grid, streak_grid, cooldown_grid, trail_grid, trail_act_grid, vwap_hold_grid, vwap_profit_grid))
        if len(combos) > args.max_evals > 0:
            combos = rng.sample(combos, k=args.max_evals)

    best_score = -10**18
    best_cfg: PolicyConfig | None = None
    best_val: Dict[str, float | int | str | None] | None = None
    best_test: Dict[str, float | int | str | None] | None = None

    checked = 0
    for thr, skip_open, skip_close, streak, cooldown, trail, trail_act, vwap_hold, vwap_profit in combos:
        if streak <= 0:
            cooldown = 0
        if streak > 0 and cooldown <= 0:
            continue

        cfg = PolicyConfig(
            threshold=float(thr),
            fee_roundtrip=fee,
            hold_bars=0,
            min_hold_bars=max(5, int(args.min_hold_bars)),
            entry_start_hhmm=int(args.entry_start_hhmm),
            entry_end_hhmm=int(args.entry_end_hhmm),
            skip_open_min=int(skip_open),
            skip_close_min=int(skip_close),
            loss_streak_for_cooldown=int(streak),
            cooldown_bars=int(cooldown),
            trailing_stop_pct=max(0.0, float(trail)),
            trailing_activate_pct=max(0.0, float(trail_act)),
            vwap_exit_min_hold_bars=max(0, int(vwap_hold)),
            vwap_exit_max_profit_pct=float(vwap_profit),
        )
        val_result = run_policy(
            prob=p_val,
            open_=open_val,
            close=close_val,
            vwap_gap_day=vwap_val,
            dates=dates_val,
            cfg=cfg,
            initial_cash=float(args.initial_cash),
        )
        trade_count = int(val_result["trades"])  # type: ignore[arg-type]
        if trade_count < args.min_trades or trade_count > args.max_trades:
            continue

        s = policy_score(result=val_result, mdd_penalty=float(args.mdd_penalty))
        checked += 1
        if s > best_score:
            best_score = s
            best_cfg = cfg
            best_val = val_result
            best_test = run_policy(
                prob=p_test,
                open_=open_test,
                close=close_test,
                vwap_gap_day=vwap_test,
                dates=dates_test,
                cfg=cfg,
                initial_cash=float(args.initial_cash),
            )

    if best_cfg is None or best_val is None or best_test is None:
        raise RuntimeError("no valid policy candidate evaluated")

    report = {
        "symbol": args.symbol,
        "model_path": args.model_path,
        "val_csv": args.val_csv,
        "test_csv": args.test_csv,
        "search": {
            "checked": checked,
            "max_evals": args.max_evals,
            "seed": args.seed,
            "mdd_penalty": args.mdd_penalty,
            "min_trades": args.min_trades,
            "max_trades": args.max_trades,
        },
        "best_policy": {
            "threshold": best_cfg.threshold,
            "entry_start_hhmm": best_cfg.entry_start_hhmm,
            "entry_end_hhmm": best_cfg.entry_end_hhmm,
            "skip_open_min": best_cfg.skip_open_min,
            "skip_close_min": best_cfg.skip_close_min,
            "loss_streak_for_cooldown": best_cfg.loss_streak_for_cooldown,
            "cooldown_bars": best_cfg.cooldown_bars,
            "fee_roundtrip": best_cfg.fee_roundtrip,
            "position_size_pct": float(args.position_size_pct),
            "trailing_stop_pct": best_cfg.trailing_stop_pct,
            "trailing_activate_pct": best_cfg.trailing_activate_pct,
            "vwap_exit_min_hold_bars": best_cfg.vwap_exit_min_hold_bars,
            "vwap_exit_max_profit_pct": best_cfg.vwap_exit_max_profit_pct,
        },
        "best_val": best_val,
        "best_test": best_test,
        "best_val_score": best_score,
    }

    policy = {
        "symbol": args.symbol,
        "model_path": args.model_path,
        "selected_on": "validation",
        "threshold": best_cfg.threshold,
        "entry_start_hhmm": best_cfg.entry_start_hhmm,
        "entry_end_hhmm": best_cfg.entry_end_hhmm,
        "skip_open_min": best_cfg.skip_open_min,
        "skip_close_min": best_cfg.skip_close_min,
        "loss_streak_for_cooldown": best_cfg.loss_streak_for_cooldown,
        "cooldown_bars": best_cfg.cooldown_bars,
        "fee_roundtrip": best_cfg.fee_roundtrip,
        "position_size_pct": float(args.position_size_pct),
        "trailing_stop_pct": best_cfg.trailing_stop_pct,
        "trailing_activate_pct": best_cfg.trailing_activate_pct,
        "vwap_exit_min_hold_bars": best_cfg.vwap_exit_min_hold_bars,
        "vwap_exit_max_profit_pct": best_cfg.vwap_exit_max_profit_pct,
    }

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    policy_out = Path(args.policy_out)
    policy_out.parent.mkdir(parents=True, exist_ok=True)
    policy_out.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"report_saved={out}")
    print(f"policy_saved={policy_out}")
    print(
        f"best_val_score={best_score:.4f} "
        f"val_ret={float(best_val['total_return_pct']):.2f}% "
        f"val_mdd={float(best_val['max_drawdown_pct']):.2f}% "
        f"val_trades={int(best_val['trades'])}"
    )
    print(
        f"test_ret={float(best_test['total_return_pct']):.2f}% "
        f"test_mdd={float(best_test['max_drawdown_pct']):.2f}% "
        f"test_trades={int(best_test['trades'])}"
    )


if __name__ == "__main__":
    main()
