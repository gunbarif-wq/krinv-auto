from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from strategy_runtime import SymbolState, evaluate_state_transition, ma_cross_level_signal, multi_factor_signal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay-test realtime strategy on historical 1m csv data")
    p.add_argument("--data-dir", default="data/intraday_1w", help="directory with *1m*.csv files")
    p.add_argument("--cash", type=float, default=10_000_000)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--cash-buffer-pct", type=float, default=0.15)
    p.add_argument("--position-size-pct", type=float, default=0.08)
    p.add_argument("--min-order-krw", type=float, default=200_000)

    p.add_argument("--short", type=int, default=5)
    p.add_argument("--long", type=int, default=20)
    p.add_argument("--strategy-mode", choices=["multi_factor", "ma_cross_level"], default="multi_factor")
    p.add_argument("--mom-window", type=int, default=12)
    p.add_argument("--stoch-window", type=int, default=14)
    p.add_argument("--stoch-smooth", type=int, default=3)
    p.add_argument("--entry-threshold", type=float, default=0.45)
    p.add_argument("--exit-threshold", type=float, default=-0.25)
    p.add_argument("--ma-a", type=int, default=5)
    p.add_argument("--ma-b", type=int, default=20)
    p.add_argument("--cross-level-window", type=int, default=60)
    p.add_argument("--cross-buy-level", type=float, default=0.8)
    p.add_argument("--cross-sell-level", type=float, default=0.2)
    p.add_argument("--entry-confirm-bars", type=int, default=5)
    p.add_argument("--exit-confirm-bars", type=int, default=4)
    p.add_argument("--min-hold-bars", type=int, default=8)
    p.add_argument("--cooldown-bars", type=int, default=8)
    p.add_argument("--stop-loss-pct", type=float, default=0.012)
    p.add_argument("--take-profit-pct", type=float, default=0.025)
    return p.parse_args()


def load_symbol_csv(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                dt = datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            rows.append(
                {
                    "date": dt,
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                }
            )
    rows.sort(key=lambda x: x["date"])
    return rows


def extract_symbol(path: Path) -> str:
    return path.stem.split("_")[0]


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*1m*.csv"))
    if not files:
        raise RuntimeError(f"no csv files in {data_dir}")

    series_by_symbol: Dict[str, List[Dict]] = {}
    for f in files:
        sym = extract_symbol(f)
        rows = load_symbol_csv(f)
        if rows:
            series_by_symbol[sym] = rows
    if not series_by_symbol:
        raise RuntimeError("all csv files are empty or invalid")

    all_timestamps = sorted({r["date"] for rows in series_by_symbol.values() for r in rows})
    idx_map: Dict[str, int] = {s: 0 for s in series_by_symbol}
    ohlc_hist: Dict[str, List[Dict[str, float]]] = {s: [] for s in series_by_symbol}
    state: Dict[str, SymbolState] = {s: SymbolState() for s in series_by_symbol}
    last_price: Dict[str, float] = {s: 0.0 for s in series_by_symbol}

    cash = args.cash
    realized = 0.0
    trades = 0

    for ts in all_timestamps:
        for sym, rows in series_by_symbol.items():
            i = idx_map[sym]
            if i >= len(rows) or rows[i]["date"] != ts:
                continue
            row = rows[i]
            idx_map[sym] += 1

            ohlc_hist[sym].append(
                {"open": row["open"], "high": row["high"], "low": row["low"], "close": row["close"]}
            )
            last_px = row["close"]
            last_price[sym] = last_px
            st = state[sym]

            if args.strategy_mode == "ma_cross_level":
                signal, _ = ma_cross_level_signal(
                    ohlc=ohlc_hist[sym],
                    ma_a=args.ma_a,
                    ma_b=args.ma_b,
                    level_window=args.cross_level_window,
                    buy_level=args.cross_buy_level,
                    sell_level=args.cross_sell_level,
                )
            else:
                signal, _ = multi_factor_signal(
                    ohlc=ohlc_hist[sym],
                    short=args.short,
                    long=args.long,
                    mom_window=args.mom_window,
                    stoch_window=args.stoch_window,
                    stoch_smooth=args.stoch_smooth,
                    entry_threshold=args.entry_threshold,
                    exit_threshold=args.exit_threshold,
                )

            transition = evaluate_state_transition(
                state=st,
                signal=signal,
                last_px=last_px,
                strategy_mode=args.strategy_mode,
                stop_loss_pct=args.stop_loss_pct,
                take_profit_pct=args.take_profit_pct,
                entry_confirm_bars=args.entry_confirm_bars,
                exit_confirm_bars=args.exit_confirm_bars,
                min_hold_bars=args.min_hold_bars,
                hard_liquidation_window=False,
                ease_sell_window=False,
                ease_ratio=0.0,
            )
            signal = int(transition["signal"])

            if signal == 1 and not st.has_position:
                if transition.get("entry_wait"):
                    continue

                mtm_positions = sum(
                    state[s].held_qty * (last_price[s] if last_price[s] > 0 else state[s].entry_price)
                    for s in state
                    if state[s].has_position and state[s].held_qty > 0
                )
                portfolio_equity_est = cash + mtm_positions
                order_budget = min(
                    max(0.0, portfolio_equity_est * args.position_size_pct),
                    max(0.0, cash * (1.0 - args.cash_buffer_pct)),
                )
                if order_budget < args.min_order_krw:
                    continue

                qty = int(order_budget // (last_px * (1.0 + args.fee_rate)))
                if qty <= 0:
                    continue
                buy_cost = qty * last_px
                buy_fee = buy_cost * args.fee_rate
                total_buy = buy_cost + buy_fee
                if total_buy > cash:
                    qty = int(cash // (last_px * (1.0 + args.fee_rate)))
                    if qty <= 0:
                        continue
                    buy_cost = qty * last_px
                    buy_fee = buy_cost * args.fee_rate
                    total_buy = buy_cost + buy_fee

                cash -= total_buy
                st.has_position = True
                st.held_qty = qty
                st.entry_price = last_px
                st.entry_total_cost = total_buy
                st.held_bars = 0
                st.entry_streak = 0
                trades += 1

            elif signal == -1 and st.has_position:
                if transition.get("hold_wait"):
                    continue
                if transition.get("exit_wait"):
                    continue

                sell_gross = last_px * st.held_qty
                sell_fee = sell_gross * args.fee_rate
                sell_net = sell_gross - sell_fee
                pnl = sell_net - st.entry_total_cost
                realized += pnl
                cash += sell_net
                trades += 1

                st.has_position = False
                st.held_qty = 0
                st.entry_price = 0.0
                st.entry_total_cost = 0.0
                st.held_bars = 0
                st.exit_streak = 0
                st.cooldown_left = args.cooldown_bars

    unrealized = sum(
        state[s].held_qty * max(0.0, last_price[s]) - state[s].entry_total_cost
        for s in state
        if state[s].has_position and state[s].held_qty > 0
    )
    final_equity = cash + sum(
        state[s].held_qty * max(0.0, last_price[s]) for s in state if state[s].has_position and state[s].held_qty > 0
    )
    ret_pct = (final_equity / args.cash - 1.0) * 100.0

    print("=== Replay Backtest Result ===")
    print(f"strategy       : {args.strategy_mode}")
    print(f"symbols        : {','.join(sorted(series_by_symbol.keys()))}")
    print(f"initial_cash   : {args.cash:,.0f} KRW")
    print(f"final_equity   : {final_equity:,.0f} KRW")
    print(f"total_return   : {ret_pct:.2f}%")
    print(f"realized_pnl   : {realized:,.0f} KRW")
    print(f"unrealized_pnl : {unrealized:,.0f} KRW")
    print(f"trades         : {trades}")


if __name__ == "__main__":
    main()
