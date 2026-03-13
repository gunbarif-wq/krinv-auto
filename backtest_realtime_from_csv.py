from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from strategy_runtime import SymbolState, evaluate_state_transition, ma_cross_level_signal, multi_factor_signal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay-test realtime strategy on historical 1m csv data")
    p.add_argument("--data-dir", default="data/backtest_sets/train_1m", help="directory with *1m*.csv files")
    p.add_argument("--cash", type=float, default=10_000_000)
    p.add_argument("--fee-rate", type=float, default=0.0005)
    p.add_argument("--cash-buffer-pct", type=float, default=0.12)
    p.add_argument("--max-invested-pct", type=float, default=0.30)
    p.add_argument("--max-positions", type=int, default=2)
    p.add_argument("--min-order-krw", type=float, default=250_000)
    p.add_argument("--no-buy-before-close-min", type=int, default=25)
    p.add_argument("--no-buy-morning-start-hhmm", type=int, default=900)
    p.add_argument("--no-buy-morning-end-hhmm", type=int, default=1000)

    p.add_argument("--short", type=int, default=5)
    p.add_argument("--long", type=int, default=20)
    p.add_argument("--strategy-mode", choices=["multi_factor", "ma_cross_level"], default="ma_cross_level")
    p.add_argument("--mom-window", type=int, default=12)
    p.add_argument("--stoch-window", type=int, default=14)
    p.add_argument("--stoch-smooth", type=int, default=3)
    p.add_argument("--entry-threshold", type=float, default=0.45)
    p.add_argument("--exit-threshold", type=float, default=-0.25)
    p.add_argument("--ma-a", type=int, default=6)
    p.add_argument("--ma-b", type=int, default=26)
    p.add_argument("--cross-level-window", type=int, default=120)
    p.add_argument("--cross-buy-level", type=float, default=0.93)
    p.add_argument("--cross-sell-level", type=float, default=0.55)
    p.add_argument("--entry-confirm-bars", type=int, default=5)
    p.add_argument("--exit-confirm-bars", type=int, default=4)
    p.add_argument("--min-hold-bars", type=int, default=3)
    p.add_argument("--cooldown-bars", type=int, default=50)
    p.add_argument("--stop-loss-pct", type=float, default=0.008)
    p.add_argument("--take-profit-pct", type=float, default=0.020)
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
    trade_events: List[str] = []
    sell_count = 0
    win_count = 0
    reason_stats: Dict[str, Dict[str, float]] = {}

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
            hhmm = ts.hour * 100 + ts.minute
            is_weekday = ts.weekday() < 5
            in_session = is_weekday and (900 <= hhmm <= 1530)
            close_dt = ts.replace(hour=15, minute=30, second=0, microsecond=0)
            mins_to_close = (close_dt - ts).total_seconds() / 60.0
            ease_sell_window = is_weekday and in_session and (3.0 < mins_to_close <= 30.0)
            hard_liquidation_window = is_weekday and in_session and (0.0 <= mins_to_close <= 3.0)
            no_new_buy_window = is_weekday and in_session and (0.0 <= mins_to_close <= max(0, args.no_buy_before_close_min))
            morning_no_buy_window = args.no_buy_morning_start_hhmm <= hhmm < args.no_buy_morning_end_hhmm
            ease_ratio = 0.0
            if ease_sell_window:
                ease_ratio = (30.0 - mins_to_close) / 27.0
                ease_ratio = max(0.0, min(1.0, ease_ratio))

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
            # Ignore noisy close-auction minutes (15:20~15:29) for strategy signals.
            if 1520 <= hhmm <= 1529:
                signal = 0

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
                hard_liquidation_window=hard_liquidation_window,
                ease_sell_window=ease_sell_window,
                ease_ratio=ease_ratio,
            )
            signal = int(transition["signal"])

            if signal == 1 and not st.has_position:
                if transition.get("entry_wait"):
                    continue
                if no_new_buy_window or morning_no_buy_window:
                    continue
                open_positions = sum(1 for s in state if state[s].has_position and state[s].held_qty > 0)
                if open_positions >= args.max_positions:
                    continue

                mtm_positions = sum(
                    state[s].held_qty * (last_price[s] if last_price[s] > 0 else state[s].entry_price)
                    for s in state
                    if state[s].has_position and state[s].held_qty > 0
                )
                portfolio_equity_est = cash + mtm_positions
                per_slot_cap = max(0.0, portfolio_equity_est * args.max_invested_pct / max(1, args.max_positions))
                capped_by_cash = max(0.0, cash * (1.0 - args.cash_buffer_pct))
                invested_cap = max(0.0, portfolio_equity_est * args.max_invested_pct)
                remaining_investable = max(0.0, invested_cap - mtm_positions)
                order_budget = min(per_slot_cap, capped_by_cash, remaining_investable)
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
                trade_events.append(
                    f"{ts.strftime('%Y-%m-%d %H:%M:%S')} BUY  {sym} qty={qty} px={last_px:.0f} "
                    f"gross={buy_cost:,.0f} fee={buy_fee:,.0f} total={total_buy:,.0f} cash_after={cash:,.0f}"
                )

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
                sell_count += 1
                if pnl > 0:
                    win_count += 1
                reason = str(transition.get("sell_reason", "signal"))
                stat = reason_stats.setdefault(reason, {"count": 0.0, "wins": 0.0, "pnl_sum": 0.0})
                stat["count"] += 1
                if pnl > 0:
                    stat["wins"] += 1
                stat["pnl_sum"] += pnl
                trade_events.append(
                    f"{ts.strftime('%Y-%m-%d %H:%M:%S')} SELL {sym} qty={st.held_qty} px={last_px:.0f} "
                    f"gross={sell_gross:,.0f} fee={sell_fee:,.0f} net={sell_net:,.0f} pnl={pnl:,.0f} "
                    f"reason={reason} cash_after={cash:,.0f}"
                )

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
    win_rate = (win_count / sell_count * 100.0) if sell_count > 0 else 0.0

    print("=== Replay Backtest Result ===")
    print(f"strategy       : {args.strategy_mode}")
    print(f"symbols        : {','.join(sorted(series_by_symbol.keys()))}")
    print(f"initial_cash   : {args.cash:,.0f} KRW")
    print(f"final_equity   : {final_equity:,.0f} KRW")
    print(f"total_return   : {ret_pct:.2f}%")
    print(f"realized_pnl   : {realized:,.0f} KRW")
    print(f"unrealized_pnl : {unrealized:,.0f} KRW")
    print(f"trades         : {trades}")
    print(f"win_rate       : {win_rate:.2f}% ({win_count}/{sell_count})")
    if reason_stats:
        for r in sorted(reason_stats.keys()):
            c = int(reason_stats[r]["count"])
            w = int(reason_stats[r]["wins"])
            wr = (w / c * 100.0) if c > 0 else 0.0
            print(f"win_rate_{r:<10}: {wr:.2f}% ({w}/{c})")

    runs_dir = Path("data/runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with run_path.open("w", encoding="utf-8") as f:
        f.write("=== Backtest Run Args ===\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v}\n")
        f.write("\n=== Trade Events ===\n")
        if trade_events:
            for line in trade_events:
                f.write(line + "\n")
        else:
            f.write("NO_TRADES\n")
        f.write("\n=== Summary ===\n")
        f.write(f"strategy={args.strategy_mode}\n")
        f.write(f"symbols={','.join(sorted(series_by_symbol.keys()))}\n")
        f.write(f"initial_cash={args.cash:,.0f}\n")
        f.write(f"final_equity={final_equity:,.0f}\n")
        f.write(f"total_return_pct={ret_pct:.4f}\n")
        f.write(f"realized_pnl={realized:,.0f}\n")
        f.write(f"unrealized_pnl={unrealized:,.0f}\n")
        f.write(f"trades={trades}\n")
        f.write(f"win_rate={win_rate:.4f}% ({win_count}/{sell_count})\n")
        if reason_stats:
            f.write("win_rate_by_reason:\n")
            for r in sorted(reason_stats.keys()):
                c = int(reason_stats[r]["count"])
                w = int(reason_stats[r]["wins"])
                wr = (w / c * 100.0) if c > 0 else 0.0
                pnl_sum = reason_stats[r]["pnl_sum"]
                f.write(f"  {r}: {wr:.4f}% ({w}/{c}), pnl_sum={pnl_sum:,.0f}\n")
    print(f"run_saved      : {run_path}")


if __name__ == "__main__":
    main()
