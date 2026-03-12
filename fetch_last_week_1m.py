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


DEFAULT_SYMBOLS = ["012450", "079550", "047810", "272210", "064350"]


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
    p = argparse.ArgumentParser(description="Fetch last N days of KIS 1m bars per symbol")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma-separated symbols")
    p.add_argument("--days", type=int, default=7, help="lookback calendar days")
    p.add_argument("--out-dir", default="data/intraday_1w", help="output directory")
    p.add_argument("--max-bars-per-day", type=int, default=450)
    p.add_argument("--pause-ms", type=int, default=380)
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


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET required")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise RuntimeError("No symbols provided")

    end_d = datetime.now().date()
    start_d = end_d - timedelta(days=max(1, args.days) - 1)
    start = start_d.strftime("%Y%m%d")
    end = end_d.strftime("%Y%m%d")
    days = iter_business_days(start, end)

    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
    out_dir = Path(args.out_dir)
    print(f"period={start}~{end} days={len(days)} out_dir={out_dir}")

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

        ordered = [merged[k] for k in sorted(merged.keys())]
        out_path = out_dir / f"{sym}_1m_last{args.days}d.csv"
        if ordered:
            save_csv(out_path, ordered)
            print(f"saved {sym}: {out_path} ({len(ordered)} rows)")
        else:
            print(f"[WARN] no rows for {sym}")


if __name__ == "__main__":
    main()
