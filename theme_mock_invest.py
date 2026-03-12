from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from fetch_kis_daily import DEFAULT_BASE_URL, fetch_daily_prices, get_access_token

SYMBOL_NAMES = {
    "012450": "한화에어로스페이스",
    "079550": "LIG넥스원",
    "047810": "한국항공우주",
    "272210": "한화시스템",
    "064350": "현대로템",
}


@dataclass
class PortfolioResult:
    final_equity: float
    total_return_pct: float
    rebalance_count: int
    equity_curve: List[Dict[str, str]]
    logs: List[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Defense theme mock investing with KIS daily data")
    p.add_argument(
        "--symbols",
        default="012450,079550,047810,272210,064350",
        help="comma-separated symbols",
    )
    p.add_argument("--start", required=True, help="YYYYMMDD")
    p.add_argument("--end", required=True, help="YYYYMMDD")
    p.add_argument("--cash", type=float, default=10_000_000, help="initial cash")
    p.add_argument("--fee", type=float, default=0.0005, help="fee rate")
    p.add_argument("--ma-window", type=int, default=20, help="SMA window")
    p.add_argument("--rebalance-every", type=int, default=20, help="rebalance interval in bars")
    p.add_argument("--curve-out", default="data/defense_equity_curve.csv")
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--verbose", action="store_true", help="상세 로그 출력")
    return p.parse_args()


def fmt_symbol(sym: str) -> str:
    return f"{SYMBOL_NAMES.get(sym, 'Unknown')}({sym})"


def fmt_symbol_list(symbols: List[str]) -> str:
    return ",".join(fmt_symbol(s) for s in symbols)


def run_theme_backtest(
    symbol_rows: Dict[str, List[Dict[str, str]]],
    initial_cash: float,
    fee_rate: float,
    ma_window: int,
    rebalance_every: int,
    verbose: bool = False,
) -> PortfolioResult:
    close_by_symbol_date: Dict[str, Dict[str, float]] = {}
    all_dates = set()
    for sym, rows in symbol_rows.items():
        table: Dict[str, float] = {}
        for r in rows:
            d = r["date"]
            table[d] = float(r["close"])
            all_dates.add(d)
        close_by_symbol_date[sym] = table

    ordered_dates = sorted(all_dates)
    symbols = sorted(symbol_rows.keys())
    cash = initial_cash
    qty: Dict[str, float] = {s: 0.0 for s in symbols}
    rebalance_count = 0
    logs: List[str] = []
    equity_curve: List[Dict[str, str]] = []
    history: Dict[str, List[float]] = defaultdict(list)

    for i, d in enumerate(ordered_dates):
        prices_today: Dict[str, float] = {}
        for s in symbols:
            p = close_by_symbol_date[s].get(d)
            if p is not None:
                prices_today[s] = p
                history[s].append(p)

        if not prices_today:
            continue

        do_rebalance = (i % rebalance_every == 0)
        if do_rebalance:
            equity_before = cash + sum(qty[s] * prices_today.get(s, 0.0) for s in symbols)
            investable = []
            signal_details: List[str] = []
            for s, p in prices_today.items():
                hist = history[s]
                if len(hist) < ma_window:
                    if verbose:
                        signal_details.append(
                            f"{fmt_symbol(s)} price={p:.2f} signal=HOLD reason=insufficient_history({len(hist)}/{ma_window})"
                        )
                    continue
                sma = sum(hist[-ma_window:]) / ma_window
                if p > sma:
                    investable.append(s)
                    if verbose:
                        signal_details.append(
                            f"{fmt_symbol(s)} price={p:.2f} sma{ma_window}={sma:.2f} signal=BUY_CANDIDATE"
                        )
                elif verbose:
                    signal_details.append(
                        f"{fmt_symbol(s)} price={p:.2f} sma{ma_window}={sma:.2f} signal=OUT"
                    )

            equity = equity_before
            target_value: Dict[str, float] = {s: 0.0 for s in symbols}
            if investable:
                per = equity / len(investable)
                for s in investable:
                    target_value[s] = per

            # Fractional-share rebalancing for simple simulation.
            trade_details: List[str] = []
            for s in symbols:
                p = prices_today.get(s)
                if p is None:
                    continue
                current_value = qty[s] * p
                delta_value = target_value[s] - current_value
                fee = abs(delta_value) * fee_rate
                side = "BUY" if delta_value > 0 else ("SELL" if delta_value < 0 else "HOLD")
                delta_qty = delta_value / p if p else 0.0
                cash -= delta_value + fee
                qty[s] += delta_value / p
                if verbose and side != "HOLD":
                    trade_details.append(
                        f"{fmt_symbol(s)} {side} qty={delta_qty:.4f} px={p:.2f} value={delta_value:.2f} fee={fee:.2f}"
                    )

            rebalance_count += 1
            equity_after = cash + sum(qty[s] * prices_today.get(s, 0.0) for s in symbols)
            logs.append(
                f"{d} REBALANCE investable={fmt_symbol_list(investable) if investable else 'NONE'} "
                f"equity_before={equity_before:.2f} equity_after={equity_after:.2f} cash={cash:.2f}"
            )
            if verbose:
                logs.extend(signal_details)
                if trade_details:
                    logs.extend(trade_details)
                else:
                    logs.append("NO_TRADES")

        equity_now = cash + sum(qty[s] * prices_today.get(s, 0.0) for s in symbols)
        equity_curve.append({"date": d, "equity": f"{equity_now:.2f}"})

    final_equity = float(equity_curve[-1]["equity"]) if equity_curve else initial_cash
    ret = ((final_equity / initial_cash) - 1.0) * 100.0
    return PortfolioResult(
        final_equity=final_equity,
        total_return_pct=ret,
        rebalance_count=rebalance_count,
        equity_curve=equity_curve,
        logs=logs,
    )


def save_curve(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "equity"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("Set KIS_APP_KEY and KIS_APP_SECRET first.")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise RuntimeError("No symbols provided.")

    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
    symbol_rows: Dict[str, List[Dict[str, str]]] = {}
    failed_symbols: List[str] = []
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
            if rows:
                symbol_rows[s] = rows
            else:
                failed_symbols.append(s)
        except Exception as e:
            failed_symbols.append(s)
            print(f"[WARN] symbol={fmt_symbol(s)} skipped: {e}")

    if not symbol_rows:
        raise RuntimeError("모든 종목 조회에 실패했습니다. 키 권한/시장시간/종목코드를 확인하세요.")

    result = run_theme_backtest(
        symbol_rows=symbol_rows,
        initial_cash=args.cash,
        fee_rate=args.fee,
        ma_window=args.ma_window,
        rebalance_every=args.rebalance_every,
        verbose=args.verbose,
    )
    save_curve(args.curve_out, result.equity_curve)

    print("=== Defense Theme Mock Investing ===")
    print(f"Symbols        : {fmt_symbol_list(symbols)}")
    if failed_symbols:
        print(f"Skipped        : {fmt_symbol_list(failed_symbols)}")
    print(f"Initial Cash   : {args.cash:,.2f}")
    print(f"Final Equity   : {result.final_equity:,.2f}")
    print(f"Total Return   : {result.total_return_pct:.2f}%")
    print(f"Rebalances     : {result.rebalance_count}")
    print(f"Equity Curve   : {args.curve_out}")
    print("Recent Logs    :")
    tail_n = 50 if args.verbose else 10
    for line in result.logs[-tail_n:]:
        print(f"- {line}")


if __name__ == "__main__":
    main()
