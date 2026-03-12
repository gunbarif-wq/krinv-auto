from __future__ import annotations

import argparse
import itertools
import os
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

from fetch_kis_daily import DEFAULT_BASE_URL, fetch_daily_prices, get_access_token
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
    p = argparse.ArgumentParser(description="방산 테마 봉/지표/모델 조합 최적화")
    p.add_argument("--start", required=True, help="YYYYMMDD")
    p.add_argument("--end", required=True, help="YYYYMMDD")
    p.add_argument("--symbols", default=",".join(DEFENSE_SYMBOLS))
    p.add_argument("--cash", type=float, default=2_000_000)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--top", type=int, default=10, help="상위 결과 출력 개수")
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", DEFAULT_BASE_URL))
    return p.parse_args()


def fmt_symbol(s: str) -> str:
    return f"{SYMBOL_NAMES.get(s, 'Unknown')}({s})"


def to_pricebars(rows: List[Dict[str, str]]) -> List[PriceBar]:
    out: List[PriceBar] = []
    for r in rows:
        d = datetime.strptime(r["date"], "%Y-%m-%d")
        out.append(
            PriceBar(
                date=d,
                open=float(r["open"]),
                high=float(r["high"]),
                low=float(r["low"]),
                close=float(r["close"]),
                volume=float(r.get("volume", 0)),
            )
        )
    out.sort(key=lambda x: x.date)
    return out


def resample_3day(bars: List[PriceBar]) -> List[PriceBar]:
    out: List[PriceBar] = []
    for i in range(0, len(bars), 3):
        grp = bars[i : i + 3]
        if not grp:
            continue
        out.append(
            PriceBar(
                date=grp[-1].date,
                open=grp[0].open,
                high=max(x.high for x in grp),
                low=min(x.low for x in grp),
                close=grp[-1].close,
                volume=sum(x.volume for x in grp),
            )
        )
    return out


def resample_weekly(bars: List[PriceBar]) -> List[PriceBar]:
    out: List[PriceBar] = []
    bucket: List[PriceBar] = []
    current_key = None
    for b in bars:
        key = (b.date.isocalendar().year, b.date.isocalendar().week)
        if current_key is None:
            current_key = key
        if key != current_key:
            out.append(_merge_bucket(bucket))
            bucket = []
            current_key = key
        bucket.append(b)
    if bucket:
        out.append(_merge_bucket(bucket))
    return out


def _merge_bucket(grp: List[PriceBar]) -> PriceBar:
    return PriceBar(
        date=grp[-1].date,
        open=grp[0].open,
        high=max(x.high for x in grp),
        low=min(x.low for x in grp),
        close=grp[-1].close,
        volume=sum(x.volume for x in grp),
    )


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
    rep = bt.run(bars)
    return rep.total_return_pct


def generate_configs() -> Iterable[Tuple[str, Dict]]:
    for short, long in [(5, 20), (10, 30), (20, 60)]:
        yield "sma", {"short": short, "long": long}
    for lookback, buy, sell in itertools.product([5, 10, 20], [0.01, 0.02, 0.03], [-0.01, -0.02]):
        if buy <= abs(sell):
            continue
        yield "momentum", {"lookback": lookback, "buy": buy, "sell": sell}
    for window, z_buy, z_sell in itertools.product([10, 20], [1.0, 1.5, 2.0], [0.0, 0.3, 0.5]):
        yield "mean_reversion", {"window": window, "z_buy": z_buy, "z_sell": z_sell}


def main() -> None:
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET 필요")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)

    raw_by_symbol: Dict[str, List[PriceBar]] = {}
    failed = []
    for s in symbols:
        try:
            rows = fetch_daily_prices(
                app_key=args.app_key,
                app_secret=args.app_secret,
                symbol=s,
                start_yyyymmdd=args.start,
                end_yyyymmdd=args.end,
                access_token=token,
                base_url=args.base_url,
            )
            raw_by_symbol[s] = to_pricebars(rows)
        except Exception as e:
            failed.append(s)
            print(f"[WARN] {fmt_symbol(s)} skipped: {e}")

    if not raw_by_symbol:
        raise RuntimeError("사용 가능한 종목 데이터가 없습니다.")

    bars_by_type: Dict[str, Dict[str, List[PriceBar]]] = {"daily": {}, "3day": {}, "weekly": {}}
    for s, bars in raw_by_symbol.items():
        bars_by_type["daily"][s] = bars
        bars_by_type["3day"][s] = resample_3day(bars)
        bars_by_type["weekly"][s] = resample_weekly(bars)

    results = []
    for bar_type, by_sym in bars_by_type.items():
        for strategy_name, params in generate_configs():
            per_symbol_returns = []
            for s, bars in by_sym.items():
                if len(bars) < 30:
                    continue
                try:
                    r = run_single(bars, strategy_name, params, args.cash, args.fee)
                    per_symbol_returns.append(r)
                except Exception:
                    continue
            if not per_symbol_returns:
                continue
            avg_ret = sum(per_symbol_returns) / len(per_symbol_returns)
            win_rate = sum(1 for x in per_symbol_returns if x > 0) / len(per_symbol_returns) * 100
            results.append(
                {
                    "bar_type": bar_type,
                    "strategy": strategy_name,
                    "params": params,
                    "avg_return": avg_ret,
                    "win_rate": win_rate,
                    "n_symbols": len(per_symbol_returns),
                }
            )

    results.sort(key=lambda x: x["avg_return"], reverse=True)
    top_n = min(args.top, len(results))

    print("=== Defense Strategy Optimization ===")
    print(f"Period      : {args.start} ~ {args.end}")
    print(f"Symbols     : {','.join(fmt_symbol(s) for s in raw_by_symbol.keys())}")
    if failed:
        print(f"Skipped     : {','.join(fmt_symbol(s) for s in failed)}")
    print(f"Combos run  : {len(results)}")
    print(f"Top {top_n}:")
    for i, r in enumerate(results[:top_n], start=1):
        print(
            f"{i}. bar={r['bar_type']} strategy={r['strategy']} params={r['params']} "
            f"avg_return={r['avg_return']:.2f}% win_rate={r['win_rate']:.1f}% n={r['n_symbols']}"
        )


if __name__ == "__main__":
    main()
