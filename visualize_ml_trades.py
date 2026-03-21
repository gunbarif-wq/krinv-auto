from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

import numpy as np

from backtest_ml_signal import PolicyConfig, run_policy
from build_ml_dataset import build_rows, load_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize ML trades on price chart")
    p.add_argument("--raw-csv", default="data/backtest_sets_225190_1y/full_1m/225190_1m_full.csv")
    p.add_argument("--feature-csv", default="", help="prebuilt ML feature csv (date,close,feature...,label,fwd_close_ret)")
    p.add_argument("--price-csv", default="", help="OHLCV csv for candlestick (default: --raw-csv)")
    p.add_argument("--model-path", default="data/ml/225190/225190_model.pkl")
    p.add_argument("--interactive-html", default="data/ml/225190/225190_trade_plot_full.html")
    p.add_argument("--horizon-bars", type=int, default=30)
    p.add_argument("--label-mode", default="fixed", choices=["fixed", "atr"])
    p.add_argument("--up-threshold", type=float, default=0.012)
    p.add_argument("--down-threshold", type=float, default=0.007)
    p.add_argument("--atr-up-mult", type=float, default=2.0)
    p.add_argument("--atr-down-mult", type=float, default=1.2)
    p.add_argument("--atr-floor-pct", type=float, default=0.003)
    p.add_argument("--threshold", type=float, default=0.80, help="entry threshold (default tuned policy: 0.80)")
    p.add_argument("--fee-roundtrip", type=float, default=0.001, help="roundtrip fee (default tuned policy: 0.001)")
    p.add_argument(
        "--indicator-cols",
        default="ema_spread_3_8,rsi_5,distance_to_vwap",
        help="comma separated feature columns to plot below price",
    )
    p.add_argument("--indicator-mode", default="local_rank01", choices=["raw01", "local_rank01"])
    p.add_argument("--indicator-window", type=int, default=120, help="window for local_rank01")
    p.add_argument("--score-threshold", type=float, default=None, help="0..1 threshold line (default: use strategy threshold)")
    p.add_argument("--bench-macd-threshold", type=float, default=0.4, help="benchmark threshold for macd_plus (0..1)")
    p.add_argument("--bench-slowk-threshold", type=float, default=0.35, help="benchmark threshold for slow_k14_3 (0..1)")
    p.add_argument("--hold-bars", type=int, default=20)
    p.add_argument("--take-profit-pct", type=float, default=0.0)
    p.add_argument("--stop-loss-pct", type=float, default=0.0)
    p.add_argument("--trailing-stop-pct", type=float, default=0.0)
    p.add_argument("--max-concurrent-positions", type=int, default=1)
    p.add_argument("--position-size-pct", type=float, default=1.0)
    p.add_argument("--min-entry-gap-bars", type=int, default=1)
    p.add_argument("--entry-start-hhmm", type=int, default=900)
    p.add_argument("--entry-end-hhmm", type=int, default=1530)
    p.add_argument("--skip-open-min", type=int, default=0)
    p.add_argument("--skip-close-min", type=int, default=10)
    p.add_argument("--loss-streak-for-cooldown", type=int, default=1)
    p.add_argument("--cooldown-bars", type=int, default=30)
    p.add_argument("--initial-cash", type=float, default=10_000_000)
    p.add_argument("--start-datetime", default="", help="inclusive, e.g. 2026-03-01 or 2026-03-01 09:00:00")
    p.add_argument("--end-datetime", default="", help="inclusive, e.g. 2026-03-14 or 2026-03-14 15:30:00")
    return p.parse_args()


def rows_to_arrays(rows: List[Dict[str, str]], feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
    x: List[List[float]] = []
    close: List[float] = []
    dates: List[str] = []
    extra_cols: Dict[str, List[float]] = {c: [] for c in feature_cols}
    for r in rows:
        try:
            row_vals = [float(r[c]) for c in feature_cols]
            x.append(row_vals)
            close.append(float(r["close"]))
            dates.append(r["date"])
            for c, v in zip(feature_cols, row_vals):
                extra_cols[c].append(v)
        except Exception:
            continue
    if not x:
        raise RuntimeError("empty rows after raw->feature build")
    extra_np = {k: np.asarray(v, dtype=float) for k, v in extra_cols.items()}
    return np.asarray(x, dtype=float), np.asarray(close, dtype=float), dates, extra_np


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


def robust_minmax_01(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float).copy()
    if x.size == 0:
        return x
    lo = float(np.nanpercentile(x, 1))
    hi = float(np.nanpercentile(x, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full(x.shape[0], 0.5, dtype=float)
    x = (x - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0)


def local_rank_01(arr: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    n = x.shape[0]
    w = max(10, int(window))
    out = np.full(n, 0.5, dtype=float)
    for i in range(n):
        s = max(0, i - w + 1)
        seg = x[s : i + 1]
        if seg.size <= 1:
            out[i] = 0.5
            continue
        v = x[i]
        out[i] = float(np.sum(seg <= v)) / float(seg.size)
    return np.clip(out, 0.0, 1.0)


def normalize_indicator(arr: np.ndarray, mode: str, window: int) -> np.ndarray:
    if mode == "local_rank01":
        return local_rank_01(arr, window)
    return robust_minmax_01(arr)


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
    with Path(args.model_path).open("rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols: List[str] = bundle["feature_columns"]
    threshold = float(args.threshold if args.threshold is not None else bundle.get("threshold", 0.8))
    fee = float(args.fee_roundtrip if args.fee_roundtrip is not None else bundle.get("fee_roundtrip", 0.001))

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

    x, close, dates, feature_series = rows_to_arrays(rows, feature_cols)
    prob = model.predict_proba(x)[:, 1]  # type: ignore[attr-defined]
    cfg = PolicyConfig(
        threshold=threshold,
        fee_roundtrip=fee,
        hold_bars=max(1, int(args.hold_bars)),
        entry_start_hhmm=int(args.entry_start_hhmm),
        entry_end_hhmm=int(args.entry_end_hhmm),
        skip_open_min=max(0, int(args.skip_open_min)),
        skip_close_min=max(0, int(args.skip_close_min)),
        loss_streak_for_cooldown=max(0, int(args.loss_streak_for_cooldown)),
        cooldown_bars=max(0, int(args.cooldown_bars)),
        take_profit_pct=max(0.0, float(args.take_profit_pct)),
        stop_loss_pct=max(0.0, float(args.stop_loss_pct)),
        trailing_stop_pct=max(0.0, float(args.trailing_stop_pct)),
        max_concurrent_positions=max(1, int(args.max_concurrent_positions)),
        position_size_pct=min(1.0, max(0.01, float(args.position_size_pct))),
        min_entry_gap_bars=max(1, int(args.min_entry_gap_bars)),
    )
    result = run_policy(prob=prob, close=close, dates=dates, cfg=cfg, initial_cash=float(args.initial_cash), return_trades=True)
    trade_logs = json.loads(str(result.get("trade_logs", "[]")))
    score_threshold = float(args.score_threshold) if args.score_threshold is not None else threshold
    score_threshold = float(np.clip(score_threshold, 0.0, 1.0))

    bench_results: Dict[str, Dict[str, float | int | str | None]] = {}
    bench_thresholds = {
        "macd_plus": float(np.clip(args.bench_macd_threshold, 0.0, 1.0))
        if args.bench_macd_threshold is not None
        else score_threshold,
        "slow_k14_3": float(np.clip(args.bench_slowk_threshold, 0.0, 1.0))
        if args.bench_slowk_threshold is not None
        else score_threshold,
    }
    for bench_col in ("macd_plus", "slow_k14_3"):
        if bench_col not in feature_series:
            continue
        bench_score = normalize_indicator(feature_series[bench_col], args.indicator_mode, args.indicator_window)
        bench_thr = bench_thresholds.get(bench_col, score_threshold)
        bench_prob = np.where(bench_score >= bench_thr, 1.0, 0.0)
        bench_results[bench_col] = run_policy(
            prob=bench_prob,
            close=close,
            dates=dates,
            cfg=cfg,
            initial_cash=float(args.initial_cash),
            return_trades=False,
        )

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

    prob_ds = prob[idx]
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        raise RuntimeError(f"plotly import failed: {e}")

    ohlc_map: Dict[str, Tuple[float, float, float, float]] = {}
    price_csv = str(args.price_csv).strip() or (str(args.raw_csv) if not str(args.feature_csv).strip() else "")
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
            name="model_prob",
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
    for bench_col in ("macd_plus", "slow_k14_3"):
        bench = bench_results.get(bench_col)
        if bench is None:
            print(f"benchmark_{bench_col}: skipped (column not found)")
            continue
        print(
            f"benchmark_{bench_col}: return={float(bench['total_return_pct']):.2f}% "
            f"trades={int(bench['trades'])} "
            f"win_rate={float(bench['win_rate_pct']):.2f}% "
            f"mdd={float(bench['max_drawdown_pct']):.2f}% "
            f"thr={bench_thresholds.get(bench_col, score_threshold):.2f}"
        )


if __name__ == "__main__":
    main()
