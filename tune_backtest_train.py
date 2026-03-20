from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


RET_RE = re.compile(r"total_return\s*:\s*([\-0-9.]+)%")
REALIZED_RE = re.compile(r"realized_pnl\s*:\s*([\-0-9,]+)\s*KRW")
TRADES_RE = re.compile(r"trades\s*:\s*([0-9]+)")
WIN_RE = re.compile(r"win_rate\s*:\s*([\-0-9.]+)%\s*\(([0-9]+)/([0-9]+)\)")


@dataclass
class TrialResult:
    idx: int
    elapsed_sec: float
    total_return_pct: float
    realized_pnl: int
    trades: int
    win_rate_pct: float
    wins: int
    sells: int
    params: Dict[str, str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-search train-only params for backtest_realtime_from_csv.py")
    p.add_argument("--data-dir", default="data/backtest_sets/train_1m")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep-sec", type=float, default=0.0, help="sleep between trials")
    p.add_argument("--min-trades", type=int, default=120, help="penalize too-few-trades candidates")
    p.add_argument("--max-trades", type=int, default=2200, help="penalize too-many-trades candidates")
    p.add_argument("--strategy-mode", default="ma_cross_level", choices=["ma_cross_level", "multi_factor"])
    p.add_argument("--out-dir", default="data/runs")
    p.add_argument("--save-stdout", action="store_true", help="save each trial stdout to a text file")
    return p.parse_args()


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
    # Main objective: return. Light penalty for very low/high trade counts.
    s = metrics["total_return_pct"]
    trades = metrics["trades"]
    if trades < min_trades:
        s -= (min_trades - trades) * 0.002
    if trades > max_trades:
        s -= (trades - max_trades) * 0.001
    return s


def run_trial(idx: int, args: argparse.Namespace, params: Dict[str, str]) -> tuple[TrialResult, str]:
    cmd: List[str] = [
        sys.executable,
        "backtest_realtime_from_csv.py",
        "--data-dir",
        args.data_dir,
    ]
    for k, v in params.items():
        cmd.extend([k, v])
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    stdout = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(f"trial {idx} failed rc={proc.returncode}\n{stdout[:2000]}")
    m = parse_metrics(stdout)
    result = TrialResult(
        idx=idx,
        elapsed_sec=elapsed,
        total_return_pct=float(m["total_return_pct"]),
        realized_pnl=int(m["realized_pnl"]),
        trades=int(m["trades"]),
        win_rate_pct=float(m["win_rate_pct"]),
        wins=int(m["wins"]),
        sells=int(m["sells"]),
        params=params,
    )
    return result, stdout


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"tune_train_{stamp}.csv"
    best_path = out_dir / f"tune_train_best_{stamp}.json"
    log_dir = out_dir / f"tune_train_stdout_{stamp}"
    if args.save_stdout:
        log_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
    ]
    best_score = -10**9
    best_row: Dict[str, str] | None = None
    elapsed_hist: List[float] = []

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        # Trial 1 uses the explicit baseline aligned with backtest_realtime defaults.
        default_trial_params = default_params(args.strategy_mode)
        candidates = [default_trial_params] + [
            sample_params(rng, args.strategy_mode) for _ in range(max(0, args.n_trials - 1))
        ]
        total = len(candidates)
        for i, params in enumerate(candidates, start=1):
            result, stdout = run_trial(i, args, params)
            sc = score(
                {
                    "total_return_pct": result.total_return_pct,
                    "trades": float(result.trades),
                },
                min_trades=args.min_trades,
                max_trades=args.max_trades,
            )
            row = {
                "trial": str(result.idx),
                "elapsed_sec": f"{result.elapsed_sec:.2f}",
                "total_return_pct": f"{result.total_return_pct:.4f}",
                "realized_pnl": str(result.realized_pnl),
                "trades": str(result.trades),
                "win_rate_pct": f"{result.win_rate_pct:.4f}",
                "wins": str(result.wins),
                "sells": str(result.sells),
                "score": f"{sc:.6f}",
                "params_json": json.dumps(result.params, ensure_ascii=False),
            }
            w.writerow(row)
            f.flush()

            if args.save_stdout:
                (log_dir / f"trial_{i:03d}.txt").write_text(stdout, encoding="utf-8")

            elapsed_hist.append(result.elapsed_sec)
            eta = (sum(elapsed_hist) / len(elapsed_hist)) * (total - i)
            if sc > best_score:
                best_score = sc
                best_row = row
                best_path.write_text(json.dumps(best_row, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    f"[{i}/{total}] NEW BEST return={result.total_return_pct:.2f}% trades={result.trades} "
                    f"win={result.win_rate_pct:.2f}% score={sc:.4f} elapsed={result.elapsed_sec:.1f}s ETA={eta/60:.1f}m"
                )
            else:
                print(
                    f"[{i}/{total}] return={result.total_return_pct:.2f}% trades={result.trades} "
                    f"win={result.win_rate_pct:.2f}% score={sc:.4f} elapsed={result.elapsed_sec:.1f}s ETA={eta/60:.1f}m"
                )
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    print(f"saved_csv={csv_path}")
    if best_row is not None:
        print(f"saved_best={best_path}")


if __name__ == "__main__":
    main()
