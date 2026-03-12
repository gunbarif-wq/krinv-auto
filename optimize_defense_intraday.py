from __future__ import annotations

import argparse
import itertools
import os
import time
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

from fetch_kis_daily import DEFAULT_BASE_URL, get_access_token
from fetch_kis_minute import fetch_one_day_1m, iter_business_days, resample_nm
from src.backtester import BacktestConfig, Backtester
from src.models import PriceBar
from src.strategy import MeanReversionStrategy, MomentumStrategy, SmaCrossStrategy


DEFENSE_SYMBOLS = ["012450", "079550", "047810", "272210", "064350"]
SYMBOL_NAMES = {
    "012450": "한화에어로스페이스",
    "079550": "LIG넥스원",
    "047810": "한국항공우주",
    "272210": "한화시스템",
    "064350": "현대로템",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="방산 단기봉(3/5분) 전략 조합 최적화")
    p.add_argument("--start", required=True, help="YYYYMMDD")
    p.add_argument("--end", required=True, help="YYYYMMDD")
    p.add_argument("--intervals", default="3,5", help="comma-separated intervals")
    p.add_argument("--symbols", default=",".join(DEFENSE_SYMBOLS))
    p.add_argument("--cash", type=float, default=2_000_000)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--pause-ms", type=int, default=450)
    p.add_argument("--max-bars-per-day", type=int, default=450)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", DEFAULT_BASE_URL))
    return p.parse_args()


def fmt_symbol(s: str) -> str:
    return f"{SYMBOL_NAMES.get(s, 'Unknown')}({s})"


def to_pricebars(rows: List[Dict[str, str]]) -> List[PriceBar]:
    out: List[PriceBar] = []
    for r in rows:
        out.append(
            PriceBar(
                date=datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S"),
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r.get("volume", 0)),
            )
        )
    out.sort(key=lambda x: x.date)
    return out


def generate_configs() -> Iterable[Tuple[str, Dict]]:
    for short, long in [(5, 20), (8, 30), (12, 40)]:
        yield "sma", {"short": short, "long": long}
    for lookback, buy, sell in itertools.product([6, 10, 15], [0.002, 0.004, 0.006], [-0.002, -0.004]):
        if buy <= abs(sell):
            continue
        yield "momentum", {"lookback": lookback, "buy": buy, "sell": sell}
    for window, z_buy, z_sell in itertools.product([12, 20], [1.0, 1.5, 2.0], [0.0, 0.3]):
        yield "mean_reversion", {"window": window, "z_buy": z_buy, "z_sell": z_sell}


def run_single(bars: List[PriceBar], strategy_name: str, params: Dict, cash: float, fee: float) -> float:
    if strategy_name == "sma":
        strategy = SmaCrossStrategy(short_window=params["short"], long_window=params["long"])
    elif strategy_name == "momentum":
        strategy = MomentumStrategy(
            lookback=params["lookback"],
            buy_threshold=params["buy"],
            sell_threshold=params["sell"],
        )
    else:
        strategy = MeanReversionStrategy(
            window=params["window"],
            z_buy=params["z_buy"],
            z_sell=params["z_sell"],
        )
    bt = Backtester(BacktestConfig(initial_cash=cash, fee_rate=fee), strategy)
    return bt.run(bars).total_return_pct


def main() -> None:
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET 필요")

    intervals = [int(x.strip()) for x in args.intervals.split(",") if x.strip()]
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
    days = iter_business_days(args.start, args.end)

    raw_1m: Dict[str, List[Dict[str, str]]] = {}
    failed = []
    for s in symbols:
        one_symbol_rows: List[Dict[str, str]] = []
        for d in days:
            ymd = d.strftime("%Y%m%d")
            try:
                rows = fetch_one_day_1m(
                    app_key=args.app_key,
                    app_secret=args.app_secret,
                    token=token,
                    symbol=s,
                    yyyymmdd=ymd,
                    max_bars_per_day=args.max_bars_per_day,
                    pause_ms=args.pause_ms,
                    base_url=args.base_url,
                )
                if rows:
                    one_symbol_rows.extend(rows)
                time.sleep(args.pause_ms / 1000.0)
            except Exception:
                continue
        if one_symbol_rows:
            dedup = {r["date"]: r for r in one_symbol_rows}
            raw_1m[s] = [dedup[k] for k in sorted(dedup.keys())]
        else:
            failed.append(s)
            print(f"[WARN] no intraday data: {fmt_symbol(s)}")

    if not raw_1m:
        raise RuntimeError("수집된 분봉 데이터가 없습니다.")

    bars_by_interval: Dict[int, Dict[str, List[PriceBar]]] = {}
    for iv in intervals:
        bars_by_interval[iv] = {}
        for s, rows in raw_1m.items():
            if iv == 1:
                rr = rows
            else:
                rr = resample_nm(rows, iv)
            bars_by_interval[iv][s] = to_pricebars(rr)

    results = []
    for iv, by_sym in bars_by_interval.items():
        for strategy_name, params in generate_configs():
            per = []
            for s, bars in by_sym.items():
                if len(bars) < 60:
                    continue
                try:
                    per.append(run_single(bars, strategy_name, params, args.cash, args.fee))
                except Exception:
                    continue
            if not per:
                continue
            avg_ret = sum(per) / len(per)
            win_rate = sum(1 for x in per if x > 0) / len(per) * 100
            results.append(
                {
                    "interval": iv,
                    "strategy": strategy_name,
                    "params": params,
                    "avg_return": avg_ret,
                    "win_rate": win_rate,
                    "n_symbols": len(per),
                }
            )

    results.sort(key=lambda x: x["avg_return"], reverse=True)
    top_n = min(args.top, len(results))

    print("=== Defense Intraday Optimization ===")
    print(f"Period      : {args.start} ~ {args.end}")
    print(f"Intervals   : {intervals}")
    print(f"Symbols     : {','.join(fmt_symbol(s) for s in raw_1m.keys())}")
    if failed:
        print(f"Skipped     : {','.join(fmt_symbol(s) for s in failed)}")
    print(f"Combos run  : {len(results)}")
    print(f"Top {top_n}:")
    for i, r in enumerate(results[:top_n], start=1):
        print(
            f"{i}. {r['interval']}m | {r['strategy']} | {r['params']} | "
            f"avg_return={r['avg_return']:.2f}% | win_rate={r['win_rate']:.1f}% | n={r['n_symbols']}"
        )


if __name__ == "__main__":
    main()
