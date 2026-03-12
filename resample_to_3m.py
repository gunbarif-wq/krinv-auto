from __future__ import annotations

import argparse
import csv
from datetime import datetime
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="1분봉 CSV를 3분봉으로 리샘플")
    p.add_argument("--in", dest="input_path", required=True, help="입력 1분봉 CSV")
    p.add_argument("--out", dest="output_path", required=True, help="출력 3분봉 CSV")
    return p.parse_args()


def floor_to_3m(dt: datetime) -> datetime:
    m = (dt.minute // 3) * 3
    return dt.replace(minute=m, second=0, microsecond=0)


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resample_3m(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    buckets: Dict[datetime, List[Dict[str, str]]] = {}
    for r in rows:
        dt = datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S")
        key = floor_to_3m(dt)
        buckets.setdefault(key, []).append(r)

    out: List[Dict[str, str]] = []
    for key in sorted(buckets.keys()):
        grp = buckets[key]
        open_p = float(grp[0]["open"])
        high_p = max(float(x["high"]) for x in grp)
        low_p = min(float(x["low"]) for x in grp)
        close_p = float(grp[-1]["close"])
        volume = sum(float(x.get("volume", 0)) for x in grp)
        out.append(
            {
                "date": key.strftime("%Y-%m-%d %H:%M:%S"),
                "open": f"{open_p:.2f}",
                "high": f"{high_p:.2f}",
                "low": f"{low_p:.2f}",
                "close": f"{close_p:.2f}",
                "volume": f"{volume:.0f}",
            }
        )
    return out


def save_rows(path: str, rows: List[Dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    rows_1m = load_rows(args.input_path)
    rows_3m = resample_3m(rows_1m)
    save_rows(args.output_path, rows_3m)
    print(f"saved: {args.output_path} ({len(rows_3m)} rows)")


if __name__ == "__main__":
    main()
