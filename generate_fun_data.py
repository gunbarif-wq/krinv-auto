from __future__ import annotations

import argparse
import csv
import math
import random
from datetime import date, timedelta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="재미있는 가상 주가 데이터 생성기")
    p.add_argument("--out", default="data/fun_prices.csv", help="출력 CSV 경로")
    p.add_argument("--days", type=int, default=260, help="생성할 거래일 수")
    p.add_argument("--start-price", type=float, default=100.0, help="시작 가격")
    p.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return p.parse_args()


def iter_business_days(n: int, start: date) -> list[date]:
    d = start
    out: list[date] = []
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    start = date(2025, 1, 2)
    dates = iter_business_days(args.days, start)

    price = args.start_price
    rows: list[dict[str, str]] = []

    for i, d in enumerate(dates):
        # 구간별 분위기: 상승 -> 박스권 -> 급락 -> 회복 -> 과열 -> 조정
        if i < args.days * 0.20:
            drift = 0.0015
            vol = 0.010
        elif i < args.days * 0.40:
            drift = 0.0001
            vol = 0.008
        elif i < args.days * 0.55:
            drift = -0.0022
            vol = 0.018
        elif i < args.days * 0.75:
            drift = 0.0020
            vol = 0.013
        elif i < args.days * 0.90:
            drift = 0.0028
            vol = 0.020
        else:
            drift = -0.0012
            vol = 0.015

        # 주기성(약한 사이클) + 랜덤 쇼크
        cycle = 0.0015 * math.sin(i / 11.0)
        shock = random.gauss(0.0, vol)
        ret = drift + cycle + shock

        open_p = price
        close_p = max(1.0, open_p * (1.0 + ret))
        intraday = abs(random.gauss(0.0, vol * 0.9))
        high_p = max(open_p, close_p) * (1.0 + intraday)
        low_p = min(open_p, close_p) * (1.0 - intraday * 0.9)
        volume = int(150_000 * (1.0 + abs(ret) * 14 + random.random() * 0.5))

        rows.append(
            {
                "date": d.isoformat(),
                "open": f"{open_p:.2f}",
                "high": f"{high_p:.2f}",
                "low": f"{low_p:.2f}",
                "close": f"{close_p:.2f}",
                "volume": str(volume),
            }
        )
        price = close_p

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved: {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
