from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


RET_RE = re.compile(r"total_return\s*:\s*([\-0-9.]+)%")
REALIZED_RE = re.compile(r"realized_pnl\s*:\s*([\-0-9,]+)\s*KRW")
TRADES_RE = re.compile(r"trades\s*:\s*([0-9]+)")
WIN_RE = re.compile(r"win_rate\s*:\s*([\-0-9.]+)%\s*\(([0-9]+)/([0-9]+)\)")


@dataclass
class FoldSpec:
    fold_idx: int
    train_days: List[str]
    test_days: List[str]
    train_dir: Path
    test_dir: Path


BASE_PARAMS_COMMON: Dict[str, str] = {
    # Mirrors backtest_realtime_from_csv.py defaults.
    "--cash-buffer-pct": "0.12",
    "--max-invested-pct": "0.90",
    "--max-positions": "4",
    "--min-order-krw": "150000",
    "--no-buy-before-close-min": "25",
    "--no-buy-morning-start-hhmm": "900",
    "--no-buy-morning-end-hhmm": "1000",
    "--entry-confirm-bars": "5",
    "--exit-confirm-bars": "4",
    "--min-hold-bars": "4",
    "--cooldown-bars": "40",
    "--stop-loss-pct": "0.008",
    "--take-profit-pct": "0.020",
}

BASE_PARAMS_BY_MODE: Dict[str, Dict[str, str]] = {
    "ma_cross_level": {
        "--ma-a": "6",
        "--ma-b": "26",
        "--cross-level-window": "120",
        "--cross-buy-level": "0.93",
        "--cross-sell-level": "0.55",
    },
    "multi_factor": {
        "--short": "5",
        "--long": "20",
        "--mom-window": "12",
        "--stoch-window": "14",
        "--stoch-smooth": "3",
        "--entry-threshold": "0.45",
        "--exit-threshold": "-0.25",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walk-forward random search for backtest_realtime_from_csv.py")
    p.add_argument("--source-data-dir", default="data/backtest_sets/full_1m")
    p.add_argument("--n-trials", type=int, default=30, help="random-search trials per fold (including default trial)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep-sec", type=float, default=0.0, help="sleep between trials")
    p.add_argument("--min-trades", type=int, default=120, help="penalize too-few-trades candidates")
    p.add_argument("--max-trades", type=int, default=2200, help="penalize too-many-trades candidates")
    p.add_argument("--strategy-mode", default="ma_cross_level", choices=["ma_cross_level", "multi_factor"])
    p.add_argument("--wf-train-days", type=int, default=30)
    p.add_argument("--wf-test-days", type=int, default=10)
    p.add_argument("--wf-step-days", type=int, default=10)
    p.add_argument("--wf-max-folds", type=int, default=0, help="0 means no fold limit")
    p.add_argument("--out-dir", default="data/runs")
    p.add_argument("--save-stdout", action="store_true")
    p.add_argument("--keep-fold-data", action="store_true")
    return p.parse_args()


def default_params(strategy_mode: str) -> Dict[str, str]:
    params: Dict[str, str] = {"--strategy-mode": strategy_mode}
    params.update(BASE_PARAMS_COMMON)
    params.update(BASE_PARAMS_BY_MODE[strategy_mode])
    return params


def sample_params(rng: random.Random, strategy_mode: str) -> Dict[str, str]:
    params: Dict[str, str] = {
        "--strategy-mode": strategy_mode,
        # Wider ranges around backtest_realtime defaults.
        "--cash-buffer-pct": f"{rng.choice([0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.25]):.2f}",
        "--max-invested-pct": f"{rng.choice([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]):.2f}",
        "--max-positions": str(rng.choice([2, 3, 4, 5, 6])),
        "--min-order-krw": str(rng.choice([100_000, 150_000, 200_000, 250_000, 300_000, 400_000])),
        "--no-buy-before-close-min": str(rng.choice([10, 15, 20, 25, 30, 35, 40, 50])),
        "--no-buy-morning-start-hhmm": "900",
        "--no-buy-morning-end-hhmm": str(rng.choice([930, 945, 1000, 1015, 1030, 1100])),
        "--entry-confirm-bars": str(rng.choice([1, 2, 3, 4, 5, 6, 8])),
        "--exit-confirm-bars": str(rng.choice([1, 2, 3, 4, 5, 6, 8])),
        "--cooldown-bars": str(rng.choice([10, 20, 30, 40, 50, 60, 80])),
        "--min-hold-bars": str(rng.choice([1, 2, 3, 4, 5, 6, 8])),
        "--stop-loss-pct": f"{rng.choice([0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012, 0.015]):.3f}",
        "--take-profit-pct": f"{rng.choice([0.012, 0.015, 0.018, 0.020, 0.022, 0.025, 0.030, 0.035]):.3f}",
    }
    if strategy_mode == "ma_cross_level":
        ma_a = rng.choice([4, 5, 6, 7, 8, 9, 10, 12])
        ma_b = rng.choice([16, 20, 24, 26, 30, 36, 42, 50, 60])
        if ma_b <= ma_a:
            ma_b = ma_a + 2
        buy_level = rng.choice([0.88, 0.90, 0.93, 0.95, 0.97, 0.99])
        sell_level = rng.choice([0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70])
        if sell_level >= buy_level:
            sell_level = max(0.20, buy_level - 0.15)
        params.update(
            {
                "--ma-a": str(ma_a),
                "--ma-b": str(ma_b),
                "--cross-level-window": str(rng.choice([45, 60, 75, 90, 120, 150, 180, 240])),
                "--cross-buy-level": f"{buy_level:.2f}",
                "--cross-sell-level": f"{sell_level:.2f}",
            }
        )
    else:
        short = rng.choice([3, 4, 5, 6, 7, 8, 10])
        long = rng.choice([12, 16, 20, 24, 28, 32, 40])
        if long <= short:
            long = short + 2
        params.update(
            {
                "--short": str(short),
                "--long": str(long),
                "--mom-window": str(rng.choice([6, 8, 10, 12, 14, 16, 20])),
                "--stoch-window": str(rng.choice([8, 10, 12, 14, 16, 20])),
                "--stoch-smooth": str(rng.choice([2, 3, 4, 5, 6])),
                "--entry-threshold": f"{rng.choice([0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]):.2f}",
                "--exit-threshold": f"{rng.choice([-0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10]):.2f}",
            }
        )
    return params


def parse_metrics(stdout: str) -> Dict[str, float]:
    m_ret = RET_RE.search(stdout)
    m_real = REALIZED_RE.search(stdout)
    m_tr = TRADES_RE.search(stdout)
    m_win = WIN_RE.search(stdout)
    if not (m_ret and m_real and m_tr and m_win):
        raise RuntimeError("failed to parse backtest output")
    return {
        "total_return_pct": float(m_ret.group(1)),
        "realized_pnl": float(m_real.group(1).replace(",", "")),
        "trades": float(m_tr.group(1)),
        "win_rate_pct": float(m_win.group(1)),
        "wins": float(m_win.group(2)),
        "sells": float(m_win.group(3)),
    }


def score(metrics: Dict[str, float], min_trades: int, max_trades: int) -> float:
    s = metrics["total_return_pct"]
    trades = metrics["trades"]
    if trades < min_trades:
        s -= (min_trades - trades) * 0.002
    if trades > max_trades:
        s -= (trades - max_trades) * 0.001
    return s


def run_backtest(data_dir: Path, params: Dict[str, str]) -> Tuple[Dict[str, float], str, float]:
    cmd: List[str] = [sys.executable, "backtest_realtime_from_csv.py", "--data-dir", str(data_dir)]
    for k, v in params.items():
        cmd.extend([k, v])
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    stdout = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(f"backtest failed rc={proc.returncode}\n{stdout[:2000]}")
    metrics = parse_metrics(stdout)
    return metrics, stdout, elapsed


def load_rows_by_file(data_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    files = sorted(data_dir.glob("*1m*.csv"))
    if not files:
        raise RuntimeError(f"no csv files in {data_dir}")

    rows_by_file: Dict[str, List[Dict[str, str]]] = {}
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows if r.get("date")]
        rows.sort(key=lambda x: x["date"])
        if rows:
            rows_by_file[path.name] = rows
    if not rows_by_file:
        raise RuntimeError("all csv files are empty or invalid")
    return rows_by_file


def unique_days(rows_by_file: Dict[str, List[Dict[str, str]]]) -> List[str]:
    days = {r["date"][:10] for rows in rows_by_file.values() for r in rows if len(r["date"]) >= 10}
    out = sorted(days)
    if not out:
        raise RuntimeError("no valid day values found in source rows")
    return out


def save_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["date", "open", "high", "low", "close", "volume"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "date": r.get("date", ""),
                    "open": r.get("open", ""),
                    "high": r.get("high", ""),
                    "low": r.get("low", ""),
                    "close": r.get("close", ""),
                    "volume": r.get("volume", "0"),
                }
            )


def build_folds(
    rows_by_file: Dict[str, List[Dict[str, str]]],
    days: List[str],
    train_days: int,
    test_days: int,
    step_days: int,
    max_folds: int,
    fold_root: Path,
) -> List[FoldSpec]:
    if train_days <= 0 or test_days <= 0:
        raise RuntimeError("wf-train-days and wf-test-days must be positive")
    if step_days <= 0:
        raise RuntimeError("wf-step-days must be positive")
    if len(days) < train_days + test_days:
        raise RuntimeError(
            f"not enough day buckets for walk-forward: days={len(days)} train={train_days} test={test_days}"
        )

    folds: List[FoldSpec] = []
    start = 0
    fold_idx = 1
    while start + train_days + test_days <= len(days):
        train_slice = days[start : start + train_days]
        test_slice = days[start + train_days : start + train_days + test_days]
        train_set = set(train_slice)
        test_set = set(test_slice)
        this_fold_dir = fold_root / f"fold_{fold_idx:02d}"
        train_dir = this_fold_dir / "train"
        test_dir = this_fold_dir / "test"

        train_file_count = 0
        test_file_count = 0
        for name, rows in rows_by_file.items():
            tr = [r for r in rows if r["date"][:10] in train_set]
            te = [r for r in rows if r["date"][:10] in test_set]
            if tr:
                save_rows(train_dir / name, tr)
                train_file_count += 1
            if te:
                save_rows(test_dir / name, te)
                test_file_count += 1

        if train_file_count > 0 and test_file_count > 0:
            folds.append(
                FoldSpec(
                    fold_idx=fold_idx,
                    train_days=train_slice,
                    test_days=test_slice,
                    train_dir=train_dir,
                    test_dir=test_dir,
                )
            )
        fold_idx += 1
        start += step_days
        if max_folds > 0 and len(folds) >= max_folds:
            break

    if not folds:
        raise RuntimeError("failed to build any non-empty fold")
    return folds


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = out_dir / f"tune_wf_{stamp}.csv"
    summary_path = out_dir / f"tune_wf_summary_{stamp}.json"
    fold_root = out_dir / f"tune_wf_folds_{stamp}"
    stdout_dir = out_dir / f"tune_wf_stdout_{stamp}"
    if args.save_stdout:
        stdout_dir.mkdir(parents=True, exist_ok=True)

    rows_by_file = load_rows_by_file(Path(args.source_data_dir))
    days = unique_days(rows_by_file)
    folds = build_folds(
        rows_by_file=rows_by_file,
        days=days,
        train_days=args.wf_train_days,
        test_days=args.wf_test_days,
        step_days=args.wf_step_days,
        max_folds=args.wf_max_folds,
        fold_root=fold_root,
    )
    print(
        f"walk_forward_folds={len(folds)} source_days={len(days)} "
        f"train_days={args.wf_train_days} test_days={args.wf_test_days} step_days={args.wf_step_days}"
    )

    fieldnames = [
        "fold",
        "phase",
        "trial",
        "elapsed_sec",
        "total_return_pct",
        "realized_pnl",
        "trades",
        "win_rate_pct",
        "wins",
        "sells",
        "score",
        "params_json",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    ]

    fold_summaries: List[Dict[str, object]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for fold in folds:
            print(
                f"[fold {fold.fold_idx}] train={fold.train_days[0]}..{fold.train_days[-1]} "
                f"test={fold.test_days[0]}..{fold.test_days[-1]}"
            )
            candidate_count = max(1, args.n_trials)
            candidates = [default_params(args.strategy_mode)] + [
                sample_params(rng, args.strategy_mode) for _ in range(max(0, candidate_count - 1))
            ]

            best_train_score = -10**9
            best_train_metrics: Dict[str, float] | None = None
            best_train_params: Dict[str, str] | None = None
            best_train_trial = 1

            for trial_idx, params in enumerate(candidates, start=1):
                train_metrics, train_stdout, elapsed = run_backtest(fold.train_dir, params)
                train_score = score(train_metrics, args.min_trades, args.max_trades)
                row = {
                    "fold": str(fold.fold_idx),
                    "phase": "train",
                    "trial": str(trial_idx),
                    "elapsed_sec": f"{elapsed:.2f}",
                    "total_return_pct": f"{train_metrics['total_return_pct']:.4f}",
                    "realized_pnl": f"{int(train_metrics['realized_pnl'])}",
                    "trades": f"{int(train_metrics['trades'])}",
                    "win_rate_pct": f"{train_metrics['win_rate_pct']:.4f}",
                    "wins": f"{int(train_metrics['wins'])}",
                    "sells": f"{int(train_metrics['sells'])}",
                    "score": f"{train_score:.6f}",
                    "params_json": json.dumps(params, ensure_ascii=False),
                    "train_start": fold.train_days[0],
                    "train_end": fold.train_days[-1],
                    "test_start": fold.test_days[0],
                    "test_end": fold.test_days[-1],
                }
                w.writerow(row)
                f.flush()

                if args.save_stdout:
                    (stdout_dir / f"fold_{fold.fold_idx:02d}_train_trial_{trial_idx:03d}.txt").write_text(
                        train_stdout, encoding="utf-8"
                    )

                if train_score > best_train_score:
                    best_train_score = train_score
                    best_train_metrics = train_metrics
                    best_train_params = params
                    best_train_trial = trial_idx
                    print(
                        f"[fold {fold.fold_idx}] train trial {trial_idx}/{candidate_count} NEW BEST "
                        f"ret={train_metrics['total_return_pct']:.2f}% score={train_score:.4f}"
                    )
                else:
                    print(
                        f"[fold {fold.fold_idx}] train trial {trial_idx}/{candidate_count} "
                        f"ret={train_metrics['total_return_pct']:.2f}% score={train_score:.4f}"
                    )

                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

            if best_train_params is None or best_train_metrics is None:
                raise RuntimeError(f"fold {fold.fold_idx}: best params not found")

            test_metrics, test_stdout, test_elapsed = run_backtest(fold.test_dir, best_train_params)
            test_score = score(test_metrics, args.min_trades, args.max_trades)
            test_row = {
                "fold": str(fold.fold_idx),
                "phase": "test_best",
                "trial": str(best_train_trial),
                "elapsed_sec": f"{test_elapsed:.2f}",
                "total_return_pct": f"{test_metrics['total_return_pct']:.4f}",
                "realized_pnl": f"{int(test_metrics['realized_pnl'])}",
                "trades": f"{int(test_metrics['trades'])}",
                "win_rate_pct": f"{test_metrics['win_rate_pct']:.4f}",
                "wins": f"{int(test_metrics['wins'])}",
                "sells": f"{int(test_metrics['sells'])}",
                "score": f"{test_score:.6f}",
                "params_json": json.dumps(best_train_params, ensure_ascii=False),
                "train_start": fold.train_days[0],
                "train_end": fold.train_days[-1],
                "test_start": fold.test_days[0],
                "test_end": fold.test_days[-1],
            }
            w.writerow(test_row)
            f.flush()
            if args.save_stdout:
                (stdout_dir / f"fold_{fold.fold_idx:02d}_test_best.txt").write_text(test_stdout, encoding="utf-8")

            print(
                f"[fold {fold.fold_idx}] TEST ret={test_metrics['total_return_pct']:.2f}% "
                f"trades={int(test_metrics['trades'])} win={test_metrics['win_rate_pct']:.2f}% "
                f"(train_best_trial={best_train_trial})"
            )

            fold_summaries.append(
                {
                    "fold": fold.fold_idx,
                    "train_start": fold.train_days[0],
                    "train_end": fold.train_days[-1],
                    "test_start": fold.test_days[0],
                    "test_end": fold.test_days[-1],
                    "train_best_trial": best_train_trial,
                    "train_best_score": best_train_score,
                    "train_best_metrics": best_train_metrics,
                    "test_metrics": test_metrics,
                    "test_score": test_score,
                    "params": best_train_params,
                }
            )

    oos_returns = [float(x["test_metrics"]["total_return_pct"]) for x in fold_summaries]
    oos_realized = [float(x["test_metrics"]["realized_pnl"]) for x in fold_summaries]
    oos_trades = [int(x["test_metrics"]["trades"]) for x in fold_summaries]
    oos_wins = [int(x["test_metrics"]["wins"]) for x in fold_summaries]
    oos_sells = [int(x["test_metrics"]["sells"]) for x in fold_summaries]

    aggregate = {
        "fold_count": len(fold_summaries),
        "test_return_mean_pct": statistics.mean(oos_returns),
        "test_return_median_pct": statistics.median(oos_returns),
        "test_return_min_pct": min(oos_returns),
        "test_return_max_pct": max(oos_returns),
        "test_realized_pnl_sum": int(sum(oos_realized)),
        "test_trades_sum": int(sum(oos_trades)),
        "test_win_rate_weighted_pct": (sum(oos_wins) / sum(oos_sells) * 100.0) if sum(oos_sells) > 0 else 0.0,
        "test_wins_sum": int(sum(oos_wins)),
        "test_sells_sum": int(sum(oos_sells)),
    }
    summary = {
        "timestamp": stamp,
        "args": vars(args),
        "source_days": {"start": days[0], "end": days[-1], "count": len(days)},
        "aggregate_oos": aggregate,
        "folds": fold_summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_csv={csv_path}")
    print(f"saved_summary={summary_path}")
    print(
        "oos_summary: "
        f"mean_ret={aggregate['test_return_mean_pct']:.2f}% "
        f"median_ret={aggregate['test_return_median_pct']:.2f}% "
        f"sum_realized={aggregate['test_realized_pnl_sum']:,} "
        f"weighted_win={aggregate['test_win_rate_weighted_pct']:.2f}%"
    )

    if not args.keep_fold_data:
        shutil.rmtree(fold_root, ignore_errors=True)
    else:
        print(f"fold_data_kept={fold_root}")


if __name__ == "__main__":
    main()
