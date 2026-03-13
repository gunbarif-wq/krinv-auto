from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from fetch_kis_daily import DEFAULT_BASE_URL, get_access_token
from fetch_kis_minute import fetch_one_day_1m, iter_business_days


DEFAULT_SYMBOLS = [
    # Defense (5)
    "012450",  # Hanwha Aerospace
    "079550",  # LIG Nex1
    "047810",  # KAI
    "272210",  # Hanwha Systems
    "103140",  # Poongsan
    # Space (5)
    "099320",  # SATREC INITIATIVE
    "211270",  # AP Satellite
    "274090",  # Kenko Aerospace
    "214270",  # Genohco
    "271940",  # ILJIN Hysolus (aerospace supply-chain proxy)
]


def load_dotenv(dotenv_path: str = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch 1m data and prepare train/test backtest datasets in one run"
    )
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma-separated symbols")
    p.add_argument("--business-days", type=int, default=15, help="number of business days to fetch")
    p.add_argument("--include-today", action="store_true", help="include today's data")
    p.add_argument("--train-ratio", type=float, default=0.7, help="train split ratio by time")
    p.add_argument("--out-dir", default="data/backtest_sets", help="output root directory")
    p.add_argument("--max-bars-per-day", type=int, default=450)
    p.add_argument("--pause-ms", type=int, default=500)
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", DEFAULT_BASE_URL))
    return p.parse_args()


def save_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(rows)


def split_rows(rows: List[Dict[str, str]], train_ratio: float) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not rows:
        return [], []
    ratio = max(0.1, min(0.9, train_ratio))
    cut = int(len(rows) * ratio)
    cut = max(1, min(len(rows) - 1, cut))
    return rows[:cut], rows[cut:]


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET required")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise RuntimeError("No symbols provided")

    today = datetime.now().date()
    end_d = today if args.include_today else (today - timedelta(days=1))
    # Build a wide calendar window first, then take the most recent N business days.
    cal_lookback = max(30, args.business_days * 4)
    start_d = end_d - timedelta(days=cal_lookback)
    start = start_d.strftime("%Y%m%d")
    end = end_d.strftime("%Y%m%d")
    days_all = iter_business_days(start, end)
    days = days_all[-max(1, args.business_days) :]
    if not days:
        raise RuntimeError("No business days in requested range")

    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
    root = Path(args.out_dir)
    train_dir = root / "train_1m"
    test_dir = root / "test_1m"
    merged_dir = root / "full_1m"

    print(
        f"period={days[0].strftime('%Y%m%d')}~{days[-1].strftime('%Y%m%d')} "
        f"business_days={len(days)} symbols={len(symbols)} include_today={args.include_today}"
    )
    print(f"output={root}")

    for sym in symbols:
        merged: Dict[str, Dict[str, str]] = {}
        for d in days:
            ymd = d.strftime("%Y%m%d")
            try:
                rows = fetch_one_day_1m(
                    app_key=args.app_key,
                    app_secret=args.app_secret,
                    token=token,
                    symbol=sym,
                    yyyymmdd=ymd,
                    max_bars_per_day=args.max_bars_per_day,
                    pause_ms=args.pause_ms,
                    base_url=args.base_url,
                )
                for r in rows:
                    merged[r["date"]] = r
                time.sleep(max(0, args.pause_ms) / 1000.0)
            except Exception as e:
                print(f"[WARN] {sym} {ymd} skipped: {e}")

        rows_all = [merged[k] for k in sorted(merged.keys())]
        if len(rows_all) < 200:
            print(f"[WARN] {sym} low rows={len(rows_all)} (may be insufficient)")
        train_rows, test_rows = split_rows(rows_all, args.train_ratio)

        save_csv(merged_dir / f"{sym}_1m_full.csv", rows_all)
        save_csv(train_dir / f"{sym}_1m_train.csv", train_rows)
        save_csv(test_dir / f"{sym}_1m_test.csv", test_rows)
        print(
            f"{sym}: full={len(rows_all)} train={len(train_rows)} test={len(test_rows)} "
            f"train_end={train_rows[-1]['date'] if train_rows else '-'}"
        )

    print("done")


if __name__ == "__main__":
    main()
