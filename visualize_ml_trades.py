from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

import numpy as np

from backtest_ml_signal import run_policy
from build_ml_dataset import build_rows, load_split
from ml_signal_common import PolicyConfig, ema_smooth, load_json, load_model_bundle, normalize_indicator
from ml_trade_common import build_policy_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize ML trades on price chart")
    p.add_argument("--raw-csv", default="data/backtest_sets_225190_1y/full_1m/225190_1m_full.csv")
    p.add_argument("--feature-csv", default="data/ml/225190_1y/225190_test_ml.csv", help="prebuilt ML feature csv (date,close,feature...,label,fwd_close_ret)")
    p.add_argument("--price-csv", default="", help="OHLCV csv for candlestick (default: --raw-csv)")
    p.add_argument("--model-path", default="data/ml/225190_1y/225190_model_fast.pkl")
    p.add_argument("--policy-path", default="data/ml/225190_1y/225190_fast_policy.json")
    p.add_argument("--interactive-html", default="data/ml/225190_1y/225190_trade_plot_fast.html")
    p.add_argument("--horizon-bars", type=int, default=30)
    p.add_argument("--label-mode", default="fixed", choices=["fixed", "atr"])
    p.add_argument("--up-threshold", type=float, default=0.03)
    p.add_argument("--down-threshold", type=float, default=0.015)
    p.add_argument("--atr-up-mult", type=float, default=2.0)
    p.add_argument("--atr-down-mult", type=float, default=1.2)
    p.add_argument("--atr-floor-pct", type=float, default=0.003)
    p.add_argument("--threshold", type=float, default=0.75, help="entry threshold (relprice default)")
    p.add_argument("--fee-roundtrip", type=float, default=0.004, help="roundtrip fee (relprice default)")
    p.add_argument(
        "--indicator-cols",
        default="price_z20,price_z60,atr14_pct,ma_gap_10_20,vwap_gap_day",
        help="comma separated feature columns to plot below price",
    )
    p.add_argument("--indicator-mode", default="raw", choices=["raw"])
    p.add_argument("--indicator-window", type=int, default=120, help="window for local_rank01")
    p.add_argument("--score-threshold", type=float, default=None, help="0..1 threshold line (default: use strategy threshold)")
    p.add_argument("--min-hold-bars", type=int, default=5)
    p.add_argument("--trailing-stop-pct", type=float, default=0.0)
    p.add_argument("--max-concurrent-positions", type=int, default=1)
    p.add_argument("--position-size-pct", type=float, default=1.0)
    p.add_argument("--min-entry-gap-bars", type=int, default=0)
    p.add_argument("--entry-start-hhmm", type=int, default=900)
    p.add_argument("--entry-end-hhmm", type=int, default=1530)
    p.add_argument("--skip-open-min", type=int, default=20)
    p.add_argument("--skip-close-min", type=int, default=10)
    p.add_argument("--loss-streak-for-cooldown", type=int, default=3)
    p.add_argument("--cooldown-bars", type=int, default=60)
    p.add_argument("--initial-cash", type=float, default=10_000_000)
    p.add_argument("--start-datetime", default="", help="inclusive, e.g. 2026-03-01 or 2026-03-01 09:00:00")
    p.add_argument("--end-datetime", default="", help="inclusive, e.g. 2026-03-14 or 2026-03-14 15:30:00")
    return p.parse_args()


def rows_to_arrays(rows: List[Dict[str, str]], feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
    x: List[List[float]] = []
    opens: List[float] = []
    close: List[float] = []
    dates: List[str] = []
    extra_cols: Dict[str, List[float]] = {c: [] for c in feature_cols}
    for r in rows:
        try:
            row_vals = [float(r[c]) for c in feature_cols]
            x.append(row_vals)
            opens.append(float(r["open"]))
            close.append(float(r["close"]))
            dates.append(r["date"])
            for c, v in zip(feature_cols, row_vals):
                extra_cols[c].append(v)
        except Exception:
            continue
    if not x:
        raise RuntimeError("empty rows after raw->feature build")
    extra_np = {k: np.asarray(v, dtype=float) for k, v in extra_cols.items()}
    return np.asarray(x, dtype=float), np.asarray(opens, dtype=float), np.asarray(close, dtype=float), dates, extra_np


def load_feature_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        raise RuntimeError(f"empty feature csv: {path}")
    return rows


def load_price_ohlc(path: Path) -> Dict[str, Tuple[float, float, float, float]]:
    out: Dict[str, Tuple[float, float, float, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                dt = str(r["date"])
                o = float(r["open"])
                h = float(r["high"])
                l = float(r["low"])
                c = float(r["close"])
                out[dt] = (o, h, l, c)
            except Exception:
                continue
    return out


def parse_dt_opt(s: str) -> datetime | None:
    t = str(s).strip()
    if not t:
        return None
    fmts = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")
    for f in fmts:
        try:
            return datetime.strptime(t, f)
        except ValueError:
            continue
    raise RuntimeError(f"invalid datetime format: {s}")


def filter_rows_by_range(
    rows: List[Dict[str, str]],
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> List[Dict[str, str]]:
    if not rows:
        return rows
    if start_dt is None and end_dt is None:
        return rows
    out: List[Dict[str, str]] = []
    for r in rows:
        dt_s = str(r.get("date", ""))
        if len(dt_s) < 19:
            continue
        try:
            dt = datetime.strptime(dt_s[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if start_dt is not None and dt < start_dt:
            continue
        if end_dt is not None and dt > end_dt:
            continue
        out.append(r)
    return out


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(Path(args.model_path))
    model = bundle["model"]
    feature_cols: List[str] = bundle["feature_columns"]
    policy_path = Path(args.policy_path)
    policy: Dict[str, object] = load_json(policy_path)
    threshold = float(policy.get("threshold", args.threshold if args.threshold is not None else bundle.get("threshold", 0.9)))
    fee = float(policy.get("fee_roundtrip", args.fee_roundtrip if args.fee_roundtrip is not None else bundle.get("fee_roundtrip", 0.001)))

    if str(args.feature_csv).strip():
        rows = load_feature_rows(Path(args.feature_csv))
    else:
        raw = load_split(Path(args.raw_csv))
        rows = build_rows(
            data=raw,
            horizon=args.horizon_bars,
            label_mode=args.label_mode,
            up_threshold=args.up_threshold,
            down_threshold=args.down_threshold,
            atr_up_mult=args.atr_up_mult,
            atr_down_mult=args.atr_down_mult,
            atr_floor_pct=args.atr_floor_pct,
        )
    if not rows:
        raise RuntimeError("no rows in feature dataset")

    x, open_, close, dates, feature_series = rows_to_arrays(rows, feature_cols)
    vwap_gap_day = feature_series.get("vwap_gap_day", np.zeros_like(close))
    prob_raw = model.predict_proba(x)[:, 1]  # type: ignore[attr-defined]
    prob_for_policy = ema_smooth(prob_raw, 3)
    cfg = build_policy_config(
        policy,
        threshold=threshold,
        fee_roundtrip=fee,
        min_hold_bars=args.min_hold_bars,
        entry_start_hhmm=args.entry_start_hhmm,
        entry_end_hhmm=args.entry_end_hhmm,
        skip_open_min=args.skip_open_min,
        skip_close_min=args.skip_close_min,
        loss_streak_for_cooldown=args.loss_streak_for_cooldown,
        cooldown_bars=args.cooldown_bars,
        trailing_stop_pct=args.trailing_stop_pct,
        trailing_activate_pct=0.0,
        vwap_exit_min_hold_bars=0,
        vwap_exit_max_profit_pct=0.0,
        max_concurrent_positions=args.max_concurrent_positions,
        position_size_pct=args.position_size_pct,
        min_entry_gap_bars=args.min_entry_gap_bars,
    )
    result = run_policy(
        prob=prob_for_policy,
        open_=open_,
        close=close,
        vwap_gap_day=vwap_gap_day,
        dates=dates,
        cfg=cfg,
        initial_cash=float(args.initial_cash),
        return_trades=True,
    )
    trade_logs = json.loads(str(result.get("trade_logs", "[]")))
    score_threshold = float(args.score_threshold) if args.score_threshold is not None else threshold
    score_threshold = float(np.clip(score_threshold, 0.0, 1.0))


    n = close.shape[0]
    # Keep full-range simulation, but render only a recent window in HTML by default.
    cli_start = parse_dt_opt(args.start_datetime)
    cli_end = parse_dt_opt(args.end_datetime)
    if cli_start is None and cli_end is None and dates:
        last_dt = datetime.strptime(dates[-1][:19], "%Y-%m-%d %H:%M:%S")
        cli_end = last_dt
        cli_start = last_dt - timedelta(days=30)
    plot_rows = [{"date": d} for d in dates]
    plot_rows = filter_rows_by_range(plot_rows, cli_start, cli_end)
    if plot_rows:
        keep = {r["date"] for r in plot_rows}
        idx = np.asarray([i for i, d in enumerate(dates) if d in keep], dtype=int)
    else:
        idx = np.arange(0, n, 1, dtype=int)
    close_ds = close[idx]

    indicator_cols = [c.strip() for c in str(args.indicator_cols).split(",") if c.strip()]
    indicator_cols = [c for c in indicator_cols if c in feature_series]

    title = (
        f"ret={float(result['total_return_pct']):.2f}% "
        f"mdd={float(result['max_drawdown_pct']):.2f}% "
        f"trades={int(result['trades'])} win={float(result['win_rate_pct']):.1f}%"
    )

    prob_ds = prob_for_policy[idx]
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        raise RuntimeError(f"plotly import failed: {e}")

    ohlc_map: Dict[str, Tuple[float, float, float, float]] = {}
    price_csv = str(args.price_csv).strip() or str(args.raw_csv).strip()
    if price_csv:
        ohlc_map = load_price_ohlc(Path(price_csv))
    x_all = [dates[i] for i in idx]
    x_all_set = set(x_all)
    has_candle = bool(ohlc_map)
    if has_candle:
        o_vals: List[float] = []
        h_vals: List[float] = []
        l_vals: List[float] = []
        c_vals: List[float] = []
        valid_x: List[str] = []
        for dt in x_all:
            if dt not in ohlc_map:
                continue
            o, h, l, c = ohlc_map[dt]
            valid_x.append(dt)
            o_vals.append(o)
            h_vals.append(h)
            l_vals.append(l)
            c_vals.append(c)
        x_price = valid_x
    else:
        x_price = x_all

    pfig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.62, 0.38],
        subplot_titles=("Price + Trades", "Model/Indicators"),
    )
    if has_candle:
        pfig.add_trace(
            go.Candlestick(
                x=x_price,
                open=o_vals,
                high=h_vals,
                low=l_vals,
                close=c_vals,
                name="candles",
                increasing_line_color="#d62728",
                increasing_fillcolor="#d62728",
                decreasing_line_color="#1f77b4",
                decreasing_fillcolor="#1f77b4",
            ),
            row=1,
            col=1,
        )
    else:
        pfig.add_trace(
            go.Scatter(
                x=x_all,
                y=close_ds,
                mode="lines",
                name="close",
                line=dict(width=1.2),
            ),
            row=1,
            col=1,
        )
    if trade_logs:
        entry_i = np.asarray([int(t["entry_i"]) for t in trade_logs], dtype=int)
        exit_i = np.asarray([int(t["exit_i"]) for t in trade_logs], dtype=int)
        entry_dt = [dates[min(max(0, i), len(dates) - 1)] for i in entry_i]
        exit_dt = [dates[min(max(0, i), len(dates) - 1)] for i in exit_i]
        for xdt in entry_dt:
            if xdt not in x_all_set:
                continue
            pfig.add_vline(x=xdt, line_width=1, line_color="#2ca02c", opacity=0.5, row=1, col=1)
        for xdt in exit_dt:
            if xdt not in x_all_set:
                continue
            pfig.add_vline(x=xdt, line_width=1, line_color="#d62728", opacity=0.5, row=1, col=1)
        pfig.add_trace(
            go.Scatter(x=[None], y=[None], mode="lines", name="buy", line=dict(color="#2ca02c", width=1)),
            row=1,
            col=1,
        )
        pfig.add_trace(
            go.Scatter(x=[None], y=[None], mode="lines", name="sell", line=dict(color="#d62728", width=1)),
            row=1,
            col=1,
        )

    pfig.add_trace(
        go.Scatter(
            x=x_all,
            y=prob_ds,
            mode="lines",
            name="model_score",
            line=dict(width=1.2),
        ),
        row=2,
        col=1,
    )
    pfig.add_trace(
        go.Scatter(
            x=x_all,
            y=np.full(prob_ds.shape[0], score_threshold, dtype=float),
            mode="lines",
            name=f"threshold={score_threshold:.2f}",
            line=dict(width=1.0, dash="dash"),
            visible="legendonly",
        ),
        row=2,
        col=1,
    )
    for c in indicator_cols:
        vals = feature_series[c][idx]
        vals = normalize_indicator(vals, args.indicator_mode, args.indicator_window)
        pfig.add_trace(
            go.Scatter(
                x=x_all,
                y=vals,
                mode="lines",
                name=c,
                line=dict(width=1.0),
                opacity=0.9,
                visible="legendonly",
            ),
            row=2,
            col=1,
        )
    pfig.update_layout(
        title=title,
        template="plotly_white",
        height=860,
        legend=dict(orientation="h"),
        xaxis_rangeslider_visible=False,
    )
    pfig.update_xaxes(title_text="datetime", row=2, col=1)
    pfig.update_yaxes(title_text="price", row=1, col=1)
    pfig.update_yaxes(title_text="indicators", row=2, col=1)
    pfig.update_yaxes(range=[0.0, 1.0], row=2, col=1)

    html_out = Path(args.interactive_html)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    pfig.write_html(str(html_out), include_plotlyjs="cdn")
    print(f"interactive_saved={html_out}")
    print(
        f"return={float(result['total_return_pct']):.2f}% trades={int(result['trades'])} "
        f"win_rate={float(result['win_rate_pct']):.2f}% mdd={float(result['max_drawdown_pct']):.2f}%"
    )
if __name__ == "__main__":
    main()
