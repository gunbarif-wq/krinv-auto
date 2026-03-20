from __future__ import annotations

import argparse
import csv
import itertools
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from backtest_ml_signal import PolicyConfig, run_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-optimize ML policy parameters on validation set")
    p.add_argument("--model-path", default="data/ml/225190/225190_model.pkl")
    p.add_argument("--val-csv", default="data/ml/225190/225190_val_ml.csv")
    p.add_argument("--test-csv", default="data/ml/225190/225190_test_ml.csv")
    p.add_argument("--symbol", default="225190")
    p.add_argument("--initial-cash", type=float, default=10_000_000)
    p.add_argument("--mdd-penalty", type=float, default=0.12, help="score = return_pct - penalty*abs(mdd_pct)")
    p.add_argument("--min-trades", type=int, default=30)
    p.add_argument("--max-trades", type=int, default=500)
    p.add_argument("--max-evals", type=int, default=1500, help="randomly sample combos if grid is bigger")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold-grid", default="0.60,0.65,0.70,0.75,0.80,0.85")
    p.add_argument("--hold-grid", default="10,20,30,40")
    p.add_argument("--skip-open-grid", default="0,10,20")
    p.add_argument("--skip-close-grid", default="0,10,20")
    p.add_argument("--loss-streak-grid", default="0,2,3")
    p.add_argument("--cooldown-grid", default="0,30,60")
    p.add_argument("--entry-start-hhmm", type=int, default=900)
    p.add_argument("--entry-end-hhmm", type=int, default=1530)
    p.add_argument("--report-out", default="data/ml/225190/225190_policy_opt_report.json")
    return p.parse_args()


def parse_float_grid(s: str) -> List[float]:
    out: List[float] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    if not out:
        raise RuntimeError("empty float grid")
    return out


def parse_int_grid(s: str) -> List[int]:
    out: List[int] = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise RuntimeError("empty int grid")
    return out


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


def policy_score(result: Dict[str, float | int | str | None], min_trades: int, max_trades: int, mdd_penalty: float) -> float:
    ret = float(result["total_return_pct"])  # type: ignore[arg-type]
    mdd = float(result["max_drawdown_pct"])  # negative
    trades = int(result["trades"])  # type: ignore[arg-type]
    s = ret - mdd_penalty * abs(min(0.0, mdd))
    if trades < min_trades:
        s -= (min_trades - trades) * 0.05
    if trades > max_trades:
        s -= (trades - max_trades) * 0.02
    return s


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    with Path(args.model_path).open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols: List[str] = bundle["feature_columns"]
    fee = float(bundle.get("fee_roundtrip", 0.001))

    X_val, close_val, dates_val = load_dataset(Path(args.val_csv), feature_cols)
    X_test, close_test, dates_test = load_dataset(Path(args.test_csv), feature_cols)
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    thr_grid = parse_float_grid(args.threshold_grid)
    hold_grid = parse_int_grid(args.hold_grid)
    open_grid = parse_int_grid(args.skip_open_grid)
    close_grid = parse_int_grid(args.skip_close_grid)
    streak_grid = parse_int_grid(args.loss_streak_grid)
    cooldown_grid = parse_int_grid(args.cooldown_grid)

    combos = list(itertools.product(thr_grid, hold_grid, open_grid, close_grid, streak_grid, cooldown_grid))
    if len(combos) > args.max_evals > 0:
        combos = rng.sample(combos, k=args.max_evals)

    best_score = -10**18
    best_cfg: PolicyConfig | None = None
    best_val: Dict[str, float | int | str | None] | None = None
    best_test: Dict[str, float | int | str | None] | None = None

    checked = 0
    for thr, hold, skip_open, skip_close, streak, cooldown in combos:
        if streak <= 0:
            cooldown = 0
        if streak > 0 and cooldown <= 0:
            continue

        cfg = PolicyConfig(
            threshold=float(thr),
            fee_roundtrip=fee,
            hold_bars=int(hold),
            entry_start_hhmm=int(args.entry_start_hhmm),
            entry_end_hhmm=int(args.entry_end_hhmm),
            skip_open_min=int(skip_open),
            skip_close_min=int(skip_close),
            loss_streak_for_cooldown=int(streak),
            cooldown_bars=int(cooldown),
        )
        val_result = run_policy(
            prob=p_val,
            close=close_val,
            dates=dates_val,
            cfg=cfg,
            initial_cash=float(args.initial_cash),
        )
        s = policy_score(
            result=val_result,
            min_trades=int(args.min_trades),
            max_trades=int(args.max_trades),
            mdd_penalty=float(args.mdd_penalty),
        )
        checked += 1
        if s > best_score:
            best_score = s
            best_cfg = cfg
            best_val = val_result
            best_test = run_policy(
                prob=p_test,
                close=close_test,
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
            "hold_bars": best_cfg.hold_bars,
            "entry_start_hhmm": best_cfg.entry_start_hhmm,
            "entry_end_hhmm": best_cfg.entry_end_hhmm,
            "skip_open_min": best_cfg.skip_open_min,
            "skip_close_min": best_cfg.skip_close_min,
            "loss_streak_for_cooldown": best_cfg.loss_streak_for_cooldown,
            "cooldown_bars": best_cfg.cooldown_bars,
            "fee_roundtrip": best_cfg.fee_roundtrip,
        },
        "best_val": best_val,
        "best_test": best_test,
        "best_val_score": best_score,
    }

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"report_saved={out}")
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
