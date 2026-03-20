from __future__ import annotations

import argparse
from pathlib import Path

from build_ml_dataset import build_rows, load_split, save_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build full-period ML feature dataset from full_1m csv")
    p.add_argument("--data-root", default="data/backtest_sets_225190_1y")
    p.add_argument("--symbol", default="225190")
    p.add_argument("--horizon-bars", type=int, default=30)
    p.add_argument("--label-mode", default="fixed", choices=["fixed", "atr"])
    p.add_argument("--up-threshold", type=float, default=0.012)
    p.add_argument("--down-threshold", type=float, default=0.007)
    p.add_argument("--atr-up-mult", type=float, default=2.0)
    p.add_argument("--atr-down-mult", type=float, default=1.2)
    p.add_argument("--atr-floor-pct", type=float, default=0.003)
    p.add_argument("--out-csv", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.data_root) / "full_1m" / f"{args.symbol}_1m_full.csv"
    if not src.exists():
        raise RuntimeError(f"missing source csv: {src}")
    out_csv = Path(args.out_csv) if str(args.out_csv).strip() else Path("data/ml") / args.symbol / f"{args.symbol}_full_ml.csv"
    data = load_split(src)
    rows = build_rows(
        data=data,
        horizon=args.horizon_bars,
        label_mode=args.label_mode,
        up_threshold=args.up_threshold,
        down_threshold=args.down_threshold,
        atr_up_mult=args.atr_up_mult,
        atr_down_mult=args.atr_down_mult,
        atr_floor_pct=args.atr_floor_pct,
    )
    save_rows(out_csv, rows)
    print(f"rows={len(rows)}")
    print(f"saved={out_csv}")


if __name__ == "__main__":
    main()
