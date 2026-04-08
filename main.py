from __future__ import annotations

import argparse
import json
import os
import time
import math
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Any

import requests
import numpy as np

from fetch_kis_daily import get_access_token
from build_ml_dataset import build_rows
from ml_signal_common import load_json


VTS_BASE_URL = "https://openapivts.koreainvestment.com:29443"
DEFAULT_POLICY_PATH = Path("data/ml/225190_1y/225190_fast_policy.json")
DEFAULT_POLICY = load_json(DEFAULT_POLICY_PATH)
DEFAULT_SYMBOLS = [str(DEFAULT_POLICY.get("symbol", "225190"))]
KST = ZoneInfo("Asia/Seoul")


def seconds_until_next_open(now_kst: datetime) -> int:
    target = now_kst.replace(hour=9, minute=0, second=0, microsecond=0)
    if now_kst.weekday() < 5 and now_kst < target:
        return max(1, int((target - now_kst).total_seconds()))

    # Move to next weekday 09:00.
    d = now_kst + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    next_open = d.replace(hour=9, minute=0, second=0, microsecond=0)
    return max(1, int((next_open - now_kst).total_seconds()))


def required_bars_for_signal(args: argparse.Namespace) -> int:
    return max(80, int(args.ml_feature_warmup_bars))


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
    p = argparse.ArgumentParser(description="KIS realtime paper trader (ML-only)")
    p.add_argument("--symbols", default="225190", help="comma-separated symbols")
    p.add_argument("--interval-sec", type=int, default=0, help="main loop interval in seconds")
    p.add_argument("--bar-minutes", type=int, choices=[1, 3, 5], default=1, help="signal timeframe")
    p.add_argument("--startup-lookback-minutes", type=int, default=25, help="extra warmup minutes for startup/immediate trading")
    p.add_argument("--startup-history-bars", type=int, default=120, help="max number of historical 1m bars to prefill at startup")
    p.add_argument("--live-dashboard-html", default="data/ml/225190_1y/225190_live_dashboard.html", help="realtime html dashboard path")
    p.add_argument("--live-dashboard-refresh-sec", type=int, default=5, help="browser refresh interval for the live dashboard")
    p.add_argument("--live-dashboard-history-bars", type=int, default=300, help="max bars shown in the live dashboard")
    p.add_argument(
        "--live-dashboard-indicator-cols",
        default="price_z20,price_z60,atr14_pct,ma_gap_10_20,vwap_gap_day",
        help="comma separated feature columns to plot in the live dashboard",
    )
    p.add_argument("--after-close-action", choices=["wait", "exit"], default="wait", help="behavior after market close")
    p.add_argument(
        "--strategy-mode",
        choices=["ml_alpha"],
        default="ml_alpha",
        help="signal strategy mode",
    )

    # ML signal parameters
    p.add_argument(
        "--ml-model-path",
        default=str(DEFAULT_POLICY.get("model_path", "data/ml/225190_1y/225190_model_fast.pkl")),
        help="trained ML model bundle path",
    )
    p.add_argument(
        "--ml-threshold",
        type=float,
        default=float(DEFAULT_POLICY.get("threshold", 0.8665617667145)),
        help="entry threshold for ML alpha/prob score",
    )
    p.add_argument("--ml-signal-mode", choices=["alpha", "prob"], default="prob", help="ml signal score mode")
    p.add_argument("--ml-alpha-ret-scale", type=float, default=0.004, help="sigmoid scale for expected return in alpha mode")
    p.add_argument("--ml-alpha-rank-window", type=int, default=180, help="rolling rank window for alpha mode")
    p.add_argument("--ml-hold-bars", type=int, default=5, help="timeout exit bars for ML mode")
    p.add_argument(
        "--exit-threshold",
        type=float,
        default=float(DEFAULT_POLICY.get("exit_threshold", 0.5312432671588239)),
        help="score-drop exit threshold",
    )
    p.add_argument(
        "--vwap-exit-min-hold-bars",
        type=int,
        default=int(DEFAULT_POLICY.get("vwap_exit_min_hold_bars", 4)),
        help="minimum bars held before VWAP-break exit is allowed",
    )
    p.add_argument(
        "--vwap-exit-max-profit-pct",
        type=float,
        default=float(DEFAULT_POLICY.get("vwap_exit_max_profit_pct", -0.002506087866765109)),
        help="only allow VWAP-break exit when profit is at or below this pct",
    )
    p.add_argument("--ml-feature-warmup-bars", type=int, default=120, help="minimum bars for ML feature extraction")
    p.add_argument("--ml-min-history-bars", type=int, default=30, help="minimum same-day history bars for feature rows")
    p.add_argument("--ml-entry-gap-bars", type=int, default=0, help="minimum bars between entries in ML mode")
    p.add_argument(
        "--trailing-stop-pct",
        type=float,
        default=float(DEFAULT_POLICY.get("trailing_stop_pct", 0.004293934920684575)),
        help="trailing stop ratio",
    )

    # Risk and execution
    p.add_argument("--cash-buffer-pct", type=float, default=0.0, help="keep this cash ratio unused")
    p.add_argument("--fee-rate", type=float, default=0.0005, help="fee rate for sizing/pnl")
    p.add_argument("--paper-cash", type=float, default=10_000_000, help="fallback paper cash when balance inquiry fails")
    p.add_argument(
        "--max-invested-pct",
        type=float,
        default=float(DEFAULT_POLICY.get("position_size_pct", 0.5)),
        help="max total invested fraction of portfolio equity",
    )
    p.add_argument(
        "--max-positions",
        type=int,
        default=int(DEFAULT_POLICY.get("max_concurrent_positions", 1)),
        help="max number of concurrent positions",
    )
    p.add_argument("--min-order-krw", type=float, default=150_000, help="skip buy when target order value is below this amount")
    p.add_argument("--retry", type=int, default=3, help="quote retry count")
    p.add_argument("--throttle-ms", type=int, default=800, help="delay between symbols in milliseconds")
    p.add_argument(
        "--no-buy-before-close-min",
        type=int,
        default=int(DEFAULT_POLICY.get("skip_close_min", 20)),
        help="block new buys during last N minutes before market close",
    )
    p.add_argument("--no-buy-morning-start-hhmm", type=int, default=900, help="morning no-buy start (HHMM)")
    p.add_argument(
        "--no-buy-morning-end-hhmm",
        type=int,
        default=int(DEFAULT_POLICY.get("entry_start_hhmm", 900)) + 2,
        help="morning no-buy end (HHMM, exclusive)",
    )
    p.add_argument("--dry-run", action="store_true", default=False, help="print only, no order requests")
    p.add_argument("--log-file", default="logs/realtime_events.txt", help="important event log path")
    p.add_argument("--log-rotate-minutes", type=int, default=1440, help="create a new log file every N minutes")

    # Credentials
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", VTS_BASE_URL))
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--cano", default=os.getenv("KIS_CANO", ""))
    p.add_argument("--acnt-prdt-cd", default=os.getenv("KIS_ACNT_PRDT_CD", "01"))
    return p.parse_args()


def get_minute_bars(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    symbol: str,
    count_hint: int,
    retry: int,
) -> List[Dict]:
    url = f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
    headers = {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST03010200",
    }
    now_kst = datetime.now(KST)
    ymd = now_kst.strftime("%Y%m%d")
    cursor_time = now_kst.strftime("%H%M%S")
    rows_all: List[Dict] = []
    seen = set()

    while len(rows_all) < max(80, count_hint * 2):
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": ymd,
            "FID_INPUT_HOUR_1": cursor_time,
            "FID_PW_DATA_INCU_YN": "N",
        }
        last_err = None
        resp = None
        for i in range(retry + 1):
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=15)
                resp.raise_for_status()
                last_err = None
                break
            except requests.RequestException as e:
                last_err = e
                time.sleep(0.4 * (i + 1))
        if last_err is not None:
            raise last_err

        data = resp.json()
        rows = data.get("output2", [])
        if not rows:
            break
        added = 0
        for row in rows:
            d = row.get("stck_bsop_date", "")
            t = row.get("stck_cntg_hour", "")
            if d != ymd or len(t) != 6:
                continue
            key = f"{d}{t}"
            if key in seen:
                continue
            seen.add(key)
            rows_all.append(row)
            added += 1
        if added == 0:
            break
        last = rows[-1].get("stck_cntg_hour", "")
        if not last or last == cursor_time:
            break
        cursor_time = last
        time.sleep(0.2)

    rows_all.sort(key=lambda x: x.get("stck_cntg_hour", ""))
    return rows_all[-count_hint:]


def load_startup_history_bars(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    symbol: str,
    max_bars: int,
) -> List[Dict[str, str]]:
    limit = max(0, int(max_bars))
    if limit <= 0:
        return []
    return get_minute_bars(
        base_url=base_url,
        token=token,
        app_key=app_key,
        app_secret=app_secret,
        symbol=symbol,
        count_hint=limit,
        retry=3,
    )


def merge_unique_bars(existing: List[Dict[str, str]], fresh: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for r in existing + fresh:
        d = str(r.get("stck_bsop_date", ""))
        t = str(r.get("stck_cntg_hour", ""))
        if d and t:
            merged[f"{d}{t}"] = r
    return [merged[k] for k in sorted(merged.keys())]


def build_live_dashboard_state(indicator_cols: List[str]) -> Dict[str, object]:
    return {
        "dates": [],
        "open": [],
        "high": [],
        "low": [],
        "close": [],
        "score": [],
        "prob": [],
        "alpha_raw": [],
        "ret_pred": [],
        "indicators": {c: [] for c in indicator_cols},
        "alpha_raw_hist": [],
        "events": [],
        "summary": {},
    }


def sync_live_dashboard_state(
    state: Dict[str, object],
    ohlc: List[Dict[str, float]],
    *,
    feature_columns: List[str],
    clf_model: Any,
    reg_model: Any,
    signal_mode: str,
    alpha_ret_scale: float,
    alpha_rank_window: int,
    min_history_bars: int,
    indicator_cols: List[str],
) -> None:
    dates: List[str] = state["dates"]  # type: ignore[assignment]
    opens: List[float] = state["open"]  # type: ignore[assignment]
    highs: List[float] = state["high"]  # type: ignore[assignment]
    lows: List[float] = state["low"]  # type: ignore[assignment]
    closes: List[float] = state["close"]  # type: ignore[assignment]
    scores: List[float] = state["score"]  # type: ignore[assignment]
    probs: List[float] = state["prob"]  # type: ignore[assignment]
    alpha_raws: List[float] = state["alpha_raw"]  # type: ignore[assignment]
    ret_preds: List[float] = state["ret_pred"]  # type: ignore[assignment]
    alpha_hist: List[float] = state["alpha_raw_hist"]  # type: ignore[assignment]
    indicators: Dict[str, List[float]] = state["indicators"]  # type: ignore[assignment]

    start_i = len(dates)
    if start_i > len(ohlc):
        state["dates"] = []
        state["open"] = []
        state["high"] = []
        state["low"] = []
        state["close"] = []
        state["score"] = []
        state["prob"] = []
        state["alpha_raw"] = []
        state["ret_pred"] = []
        state["alpha_raw_hist"] = []
        state["indicators"] = {c: [] for c in indicator_cols}
        return sync_live_dashboard_state(
            state,
            ohlc,
            feature_columns=feature_columns,
            clf_model=clf_model,
            reg_model=reg_model,
            signal_mode=signal_mode,
            alpha_ret_scale=alpha_ret_scale,
            alpha_rank_window=alpha_rank_window,
            min_history_bars=min_history_bars,
            indicator_cols=indicator_cols,
        )

    for i in range(start_i, len(ohlc)):
        score, m = ml_signal_from_ohlc(
            ohlc=ohlc[: i + 1],
            feature_columns=feature_columns,
            clf_model=clf_model,
            reg_model=reg_model,
            signal_mode=signal_mode,
            alpha_ret_scale=alpha_ret_scale,
            alpha_rank_window=alpha_rank_window,
            alpha_raw_hist=alpha_hist,
            min_history_bars=min_history_bars,
        )
        bar = ohlc[i]
        dates.append(str(bar.get("date", "")))
        opens.append(float(bar.get("open", 0.0)))
        highs.append(float(bar.get("high", 0.0)))
        lows.append(float(bar.get("low", 0.0)))
        closes.append(float(bar.get("close", 0.0)))
        scores.append(float(score) if score is not None else float("nan"))
        probs.append(float(m.get("prob", float("nan"))))
        alpha_raws.append(float(m.get("alpha_raw", float("nan"))))
        ret_preds.append(float(m.get("ret_pred", float("nan"))))
        snapshot = m.get("feature_snapshot", {})
        if isinstance(snapshot, dict):
            for c in indicator_cols:
                indicators[c].append(float(snapshot.get(c, float("nan"))))
        else:
            for c in indicator_cols:
                indicators[c].append(float("nan"))


def update_live_dashboard_summary(
    state: Dict[str, object],
    *,
    symbol: str,
    price: float,
    has_position: bool,
    qty: int,
    entry_price: float,
    cash: float,
    cooldown: int,
    timestamp: str,
) -> None:
    state["summary"] = {
        "symbol": symbol,
        "price": float(price),
        "position": "long" if has_position else "flat",
        "qty": int(qty),
        "entry_price": float(entry_price),
        "cash": float(cash),
        "cooldown": int(cooldown),
        "timestamp": timestamp,
        "unrealized_pct": ((float(price) / float(entry_price) - 1.0) if has_position and entry_price > 0 else float("nan")),
    }


def write_live_dashboard_html(
    html_path: Path,
    *,
    symbol: str,
    state: Dict[str, object],
    policy_path: str,
    score_threshold: float,
    indicator_cols: List[str],
    refresh_sec: int,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        raise RuntimeError(f"plotly import failed: {e}")

    dates: List[str] = state["dates"]  # type: ignore[assignment]
    opens: List[float] = state["open"]  # type: ignore[assignment]
    highs: List[float] = state["high"]  # type: ignore[assignment]
    lows: List[float] = state["low"]  # type: ignore[assignment]
    closes: List[float] = state["close"]  # type: ignore[assignment]
    scores: List[float] = state["score"]  # type: ignore[assignment]
    probs: List[float] = state["prob"]  # type: ignore[assignment]
    alpha_raws: List[float] = state["alpha_raw"]  # type: ignore[assignment]
    ret_preds: List[float] = state["ret_pred"]  # type: ignore[assignment]
    indicators: Dict[str, List[float]] = state["indicators"]  # type: ignore[assignment]
    events: List[Dict[str, object]] = state["events"]  # type: ignore[assignment]
    summary: Dict[str, object] = state.get("summary", {})  # type: ignore[assignment]

    n = len(dates)
    if n == 0:
        return
    x_all = dates
    pfig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.62, 0.38],
        subplot_titles=(f"{symbol} Price + Trades", "Model / Indicators"),
    )
    pfig.add_trace(
        go.Candlestick(
            x=x_all,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="candles",
            increasing_line_color="#d62728",
            increasing_fillcolor="#d62728",
            decreasing_line_color="#1f77b4",
            decreasing_fillcolor="#1f77b4",
        ),
        row=1,
        col=1,
    )

    buy_x: List[str] = []
    buy_y: List[float] = []
    sell_x: List[str] = []
    sell_y: List[float] = []
    for ev in events:
        dt = str(ev.get("date", ""))
        px = float(ev.get("price", float("nan")))
        if dt not in x_all:
            continue
        if str(ev.get("kind", "")) == "buy":
            buy_x.append(dt)
            buy_y.append(px)
        elif str(ev.get("kind", "")) == "sell":
            sell_x.append(dt)
            sell_y.append(px)
    if buy_x:
        pfig.add_trace(
            go.Scatter(
                x=buy_x,
                y=buy_y,
                mode="markers",
                name="buy",
                marker=dict(symbol="triangle-up", size=11, color="#2ca02c"),
            ),
            row=1,
            col=1,
        )
    if sell_x:
        pfig.add_trace(
            go.Scatter(
                x=sell_x,
                y=sell_y,
                mode="markers",
                name="sell",
                marker=dict(symbol="triangle-down", size=11, color="#d62728"),
            ),
            row=1,
            col=1,
        )
    pfig.add_trace(
        go.Scatter(
            x=x_all,
            y=scores,
            mode="lines",
            name="model_score",
            line=dict(width=1.2, color="#111111"),
        ),
        row=2,
        col=1,
    )
    pfig.add_trace(
        go.Scatter(
            x=x_all,
            y=probs,
            mode="lines",
            name="prob",
            line=dict(width=1.0, color="#17becf"),
        ),
        row=2,
        col=1,
    )
    pfig.add_trace(
        go.Scatter(
            x=x_all,
            y=alpha_raws,
            mode="lines",
            name="alpha_raw",
            line=dict(width=1.0, color="#9467bd"),
        ),
        row=2,
        col=1,
    )
    pfig.add_trace(
        go.Scatter(
            x=x_all,
            y=ret_preds,
            mode="lines",
            name="ret_pred",
            line=dict(width=1.0, color="#8c564b"),
        ),
        row=2,
        col=1,
    )
    pfig.add_trace(
        go.Scatter(
            x=x_all,
            y=np.full(n, score_threshold, dtype=float),
            mode="lines",
            name=f"threshold={score_threshold:.2f}",
            line=dict(width=1.3, dash="dash", color="#ff7f0e"),
        ),
        row=2,
        col=1,
    )
    for c in indicator_cols:
        vals = indicators.get(c, [])
        if len(vals) != n:
            continue
        pfig.add_trace(
            go.Scatter(
                x=x_all,
                y=safe_normalize_indicator(np.asarray(vals, dtype=float)),
                mode="lines",
                name=c,
                line=dict(width=1.0),
                opacity=0.85,
                visible="legendonly",
            ),
            row=2,
            col=1,
        )

    last_score = scores[-1] if scores else float("nan")
    last_prob = probs[-1] if probs else float("nan")
    last_alpha = alpha_raws[-1] if alpha_raws else float("nan")
    last_close = closes[-1] if closes else float("nan")
    latest_event = events[-1] if events else {}
    summary_price = summary.get("price")
    summary_position = summary.get("position", "flat")
    summary_cash = summary.get("cash")
    summary_qty = summary.get("qty")
    summary_entry = summary.get("entry_price")
    summary_cd = summary.get("cooldown")
    summary_pnl = summary.get("unrealized_pct")
    summary_ts = summary.get("timestamp", "")
    summary_line = (
        f"price={float(summary_price):.2f} " if isinstance(summary_price, (int, float)) and np.isfinite(float(summary_price)) else f"price={last_close:.2f} "
    )
    if isinstance(summary_cash, (int, float)) and np.isfinite(float(summary_cash)):
        summary_line += f"cash={float(summary_cash):,.0f} "
    if isinstance(summary_qty, (int, float)) and float(summary_qty) > 0:
        summary_line += f"qty={int(summary_qty)} "
    if isinstance(summary_entry, (int, float)) and np.isfinite(float(summary_entry)):
        summary_line += f"entry={float(summary_entry):.2f} "
    if isinstance(summary_pnl, (int, float)) and np.isfinite(float(summary_pnl)):
        summary_line += f"unrealized={float(summary_pnl):+.2%} "
    if isinstance(summary_cd, (int, float)):
        summary_line += f"cd={int(summary_cd)} "
    cash_text = "n/a"
    if isinstance(summary_cash, (int, float)) and np.isfinite(float(summary_cash)):
        cash_text = f"{float(summary_cash):,.0f} KRW"
    entry_text = "n/a"
    if isinstance(summary_entry, (int, float)) and np.isfinite(float(summary_entry)):
        entry_text = f"{float(summary_entry):.2f}"
    title = (
        f"{symbol} live | score={last_score:.3f} prob={last_prob:.3f} alpha={last_alpha:.3f} "
        f"thr={score_threshold:.2f} pos={summary_position} {summary_line} "
        f"events={len(events)} last={latest_event.get('kind', '-')}"
    )
    pfig.update_layout(
        title=title,
        template="plotly_white",
        height=900,
        legend=dict(orientation="h"),
        xaxis_rangeslider_visible=False,
    )
    pfig.update_xaxes(title_text="datetime", row=2, col=1)
    pfig.update_yaxes(title_text="price", row=1, col=1)
    pfig.update_yaxes(title_text="signals", row=2, col=1)
    pfig.update_yaxes(range=[0.0, 1.0], row=2, col=1)

    body = pfig.to_html(include_plotlyjs="cdn", full_html=False, config={"displayModeBar": True})
    refresh = max(1, int(refresh_sec))
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta http-equiv='refresh' content='{refresh}'>"
        f"<title>{symbol} live dashboard</title>"
        "<style>"
        "body{margin:0;background:#fafafa;font-family:Arial,sans-serif;}"
        ".wrap{padding:10px 12px;}"
        ".meta{display:flex;flex-wrap:wrap;gap:10px;margin:6px 0 12px 0;}"
        ".card{background:#fff;border:1px solid #e6e6e6;border-radius:10px;padding:8px 10px;min-width:120px;box-shadow:0 1px 2px rgba(0,0,0,.03);}"
        ".label{font-size:11px;color:#777;text-transform:uppercase;letter-spacing:.04em;}"
        ".value{font-size:15px;color:#111;font-weight:700;margin-top:2px;}"
        ".sub{font-size:12px;color:#666;margin-top:2px;}"
        "</style>"
        "</head><body><div class='wrap'>"
        f"<div class='meta'>"
        f"<div class='card'><div class='label'>Symbol</div><div class='value'>{symbol}</div><div class='sub'>{policy_path}</div></div>"
        f"<div class='card'><div class='label'>Price</div><div class='value'>{last_close:.2f}</div><div class='sub'>threshold {score_threshold:.2f}</div></div>"
        f"<div class='card'><div class='label'>Score</div><div class='value'>{last_score:.3f}</div><div class='sub'>prob {last_prob:.3f} / alpha {last_alpha:.3f}</div></div>"
        f"<div class='card'><div class='label'>Position</div><div class='value'>{summary_position}</div><div class='sub'>qty {summary_qty if summary_qty is not None else 0} / cd {summary_cd if summary_cd is not None else 0}</div></div>"
        f"<div class='card'><div class='label'>Cash</div><div class='value'>{cash_text}</div><div class='sub'>entry {entry_text}</div></div>"
        f"<div class='card'><div class='label'>Last Event</div><div class='value'>{latest_event.get('kind', '-')}</div><div class='sub'>{summary_ts}</div></div>"
        f"</div>"
        f"{body}</div></body></html>"
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")


def safe_normalize_indicator(vals: np.ndarray) -> np.ndarray:
    arr = np.asarray(vals, dtype=float)
    out = np.full(arr.shape, 0.5, dtype=float)
    finite_mask = np.isfinite(arr)
    finite = arr[finite_mask]
    if finite.size == 0:
        return out
    lo = float(np.nanpercentile(finite, 1))
    hi = float(np.nanpercentile(finite, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return out
    out[finite_mask] = np.clip((arr[finite_mask] - lo) / (hi - lo), 0.0, 1.0)
    return out


def resample_ohlc(rows: List[Dict], bar_minutes: int) -> List[Dict[str, float]]:
    if bar_minutes == 1:
        out = []
        for r in rows:
            d = r.get("stck_bsop_date", "")
            t = r.get("stck_cntg_hour", "")
            o = r.get("stck_oprc")
            h = r.get("stck_hgpr")
            l = r.get("stck_lwpr")
            c = r.get("stck_prpr")
            v = r.get("cntg_vol") or r.get("acml_vol") or "0"
            if not (o and h and l and c) or len(d) != 8 or len(t) != 6:
                continue
            out.append(
                {
                    "date": f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}",
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v),
                }
            )
        return out

    buckets: Dict[str, Dict[str, float]] = {}
    for r in rows:
        d = r.get("stck_bsop_date", "")
        t = r.get("stck_cntg_hour", "")
        o = r.get("stck_oprc", "")
        h = r.get("stck_hgpr", "")
        l = r.get("stck_lwpr", "")
        c = r.get("stck_prpr", "")
        if len(d) != 8 or len(t) != 6 or not (o and h and l and c):
            continue
        floored = (int(t[2:4]) // bar_minutes) * bar_minutes
        key = f"{d}{t[:2]}{floored:02d}"
        v = float(r.get("cntg_vol") or r.get("acml_vol") or 0.0)
        if key not in buckets:
            buckets[key] = {
                "date": f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{floored:02d}:00",
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": v,
            }
        else:
            b = buckets[key]
            b["high"] = max(b["high"], float(h))
            b["low"] = min(b["low"], float(l))
            b["close"] = float(c)
            b["volume"] += v
    return [buckets[k] for k in sorted(buckets.keys())]


def _sigmoid(x: float) -> float:
    x = max(-40.0, min(40.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _rolling_rank_01(values: List[float], x: float) -> float:
    if not values:
        return 0.5
    n = len(values)
    le = 0
    for v in values:
        if v <= x:
            le += 1
    return max(0.0, min(1.0, le / n))


def ml_signal_from_ohlc(
    ohlc: List[Dict[str, float]],
    feature_columns: List[str],
    clf_model: Any,
    reg_model: Any,
    signal_mode: str,
    alpha_ret_scale: float,
    alpha_rank_window: int,
    alpha_raw_hist: List[float],
    min_history_bars: int,
) -> tuple[float | None, Dict[str, float]]:
    if len(ohlc) < 70:
        return None, {"score": 0.0}
    data = {
        "date": [str(x.get("date", "")) for x in ohlc],
        "open": np.asarray([float(x["open"]) for x in ohlc], dtype=float),
        "high": np.asarray([float(x["high"]) for x in ohlc], dtype=float),
        "low": np.asarray([float(x["low"]) for x in ohlc], dtype=float),
        "close": np.asarray([float(x["close"]) for x in ohlc], dtype=float),
        "volume": np.asarray([float(x.get("volume", 0.0)) for x in ohlc], dtype=float),
    }
    rows = build_rows(
        data=data,
        horizon=1,
        min_history_bars=max(30, int(min_history_bars)),
        label_mode="fixed",
        up_threshold=0.01,
        down_threshold=0.01,
        atr_up_mult=2.0,
        atr_down_mult=1.2,
        atr_floor_pct=0.003,
    )
    if not rows:
        return None, {"score": 0.0}
    last = rows[-1]
    try:
        x = np.asarray([[float(last[c]) for c in feature_columns]], dtype=float)
    except Exception:
        return None, {"score": 0.0}

    prob = float(clf_model.predict_proba(x)[0, 1])  # type: ignore[attr-defined]
    if signal_mode == "prob":
        return prob, {
            "score": prob,
            "prob": prob,
            "alpha_raw": prob,
            "ret_pred": 0.0,
            "vwap_gap_day": float(last.get("vwap_gap_day", 0.0)),
        }

    ret_pred = float(reg_model.predict(x)[0]) if reg_model is not None else 0.0
    if reg_model is None:
        alpha_raw = prob
    else:
        ret_score = _sigmoid(ret_pred / max(1e-6, float(alpha_ret_scale)))
        alpha_raw = prob * ret_score
    alpha_raw_hist.append(alpha_raw)
    w = max(10, int(alpha_rank_window))
    if len(alpha_raw_hist) > w * 3:
        del alpha_raw_hist[: len(alpha_raw_hist) - (w * 3)]
    recent = alpha_raw_hist[-w:]
    score = _rolling_rank_01(recent, alpha_raw)
    return score, {
        "score": score,
        "prob": prob,
        "alpha_raw": alpha_raw,
        "ret_pred": ret_pred,
        "vwap_gap_day": float(last.get("vwap_gap_day", 0.0)),
        "feature_snapshot": {
            "price_z20": float(last.get("price_z20", 0.0)),
            "price_z60": float(last.get("price_z60", 0.0)),
            "atr14_pct": float(last.get("atr14_pct", 0.0)),
            "ma_gap_10_20": float(last.get("ma_gap_10_20", 0.0)),
            "vwap_gap_day": float(last.get("vwap_gap_day", 0.0)),
        },
    }


def get_hashkey(base_url: str, app_key: str, app_secret: str, body: Dict) -> str:
    url = f"{base_url}/uapi/hashkey"
    headers = {"appKey": app_key, "appSecret": app_secret, "content-type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=15)
    r.raise_for_status()
    return r.json().get("HASH", "")


def request_json_with_retry(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    params: Dict[str, str],
    timeout: int = 15,
    retries: int = 3,
    sleep_base: float = 0.6,
) -> Dict:
    last_err: Exception | None = None
    for i in range(max(1, retries)):
        try:
            if method.lower() == "get":
                r = requests.get(url, headers=headers, params=params, timeout=timeout)
            else:
                r = requests.post(url, headers=headers, data=json.dumps(params), timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            if i + 1 < retries:
                time.sleep(sleep_base * (i + 1))
    if last_err is not None:
        raise last_err
    raise RuntimeError("request failed")


def is_demo_base_url(base_url: str) -> bool:
    return "openapivts" in base_url.lower()


def order_tr_id(base_url: str, side: str) -> str:
    demo = is_demo_base_url(base_url)
    if side.lower() == "buy":
        return "VTTC0012U" if demo else "TTTC0012U"
    return "VTTC0011U" if demo else "TTTC0011U"


def place_order(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    cano: str,
    acnt_prdt_cd: str,
    symbol: str,
    qty: int,
    side: str,
) -> Dict:
    if qty <= 0:
        return {"rt_cd": "1", "msg1": "qty<=0"}
    url = f"{base_url}/uapi/domestic-stock/v1/trading/order-cash"
    tr_id = order_tr_id(base_url, side)
    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": acnt_prdt_cd,
        "PDNO": symbol,
        "ORD_DVSN": "01",
        "ORD_QTY": str(qty),
        "ORD_UNPR": "0",
        "EXCG_ID_DVSN_CD": "KRX",
    }
    hashkey = get_hashkey(base_url, app_key, app_secret, body)
    headers = {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": tr_id,
        "custtype": "P",
        "hashkey": hashkey,
        "content-type": "application/json",
    }
    last_err: Exception | None = None
    for i in range(4):  # first try + 3 retries
        try:
            r = requests.post(url, headers=headers, data=json.dumps(body), timeout=15)
            if r.status_code >= 400:
                snippet = r.text[:500].replace("\n", " ").strip()
                raise RuntimeError(f"order failed status={r.status_code} body={snippet}")
            return r.json()
        except requests.RequestException as e:
            last_err = e
            # Retry only for transient classes: 5xx or network/timeout.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status < 500:
                raise
            if i < 3:
                time.sleep(0.6 * (i + 1))
        except RuntimeError as e:
            last_err = e
            if i < 3 and "status=5" in str(e):
                time.sleep(0.6 * (i + 1))
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("unknown order error")


def get_orderable_cash(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    cano: str,
    acnt_prdt_cd: str,
) -> float:
    # Orderable cash inquiry. Parse defensively because response fields vary by account type.
    url = f"{base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
    headers = {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC8908R" if is_demo_base_url(base_url) else "TTTC8908R",
        "custtype": "P",
    }
    params = {
        "CANO": cano,
        "ACNT_PRDT_CD": acnt_prdt_cd,
        "PDNO": "005930",
        "ORD_DVSN": "01",
        "ORD_QTY": "1",
        "ORD_UNPR": "0",
    }
    data = request_json_with_retry("get", url, headers=headers, params=params, timeout=15, retries=3)
    if data.get("rt_cd") != "0":
        raise RuntimeError(f"balance inquiry failed: {data.get('msg1', '')}")
    candidates: List[Dict] = []
    out = data.get("output", {})
    if isinstance(out, dict):
        candidates = [out]
    elif isinstance(out, list):
        candidates = [x for x in out if isinstance(x, dict)]
    # Keep fallback parsing for older/variant responses.
    out1 = data.get("output1", {})
    out2 = data.get("output2", {})
    for extra in (out1, out2):
        if isinstance(extra, dict):
            candidates.append(extra)
        elif isinstance(extra, list):
            candidates.extend([x for x in extra if isinstance(x, dict)])

    # Only use cash-like fields. Do not use total valuation fields.
    for obj in candidates:
        for key in ("ord_psbl_cash", "ord_psbl_cash_amt", "dnca_tot_amt", "prvs_rcdl_excc_amt"):
            v = obj.get(key)
            if v is not None and str(v).strip() != "":
                try:
                    return float(str(v).replace(",", ""))
                except ValueError:
                    pass
    raise RuntimeError("cannot parse orderable cash (ord_psbl_cash/dnca_tot_amt/prvs_rcdl_excc_amt)")


def get_positions(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    cano: str,
    acnt_prdt_cd: str,
) -> Dict[str, Dict[str, float]]:
    url = f"{base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "VTTC8434R",
        "custtype": "P",
    }
    params = {
        "CANO": cano,
        "ACNT_PRDT_CD": acnt_prdt_cd,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    data = request_json_with_retry("get", url, headers=headers, params=params, timeout=15, retries=3)
    if data.get("rt_cd") != "0":
        raise RuntimeError(f"position inquiry failed: {data.get('msg1', '')}")

    output1 = data.get("output1", [])
    rows = output1 if isinstance(output1, list) else []
    positions: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("pdno", "")).strip()
        if not symbol:
            continue
        qty_raw = row.get("hldg_qty", row.get("hold_qty", "0"))
        avg_raw = row.get("pchs_avg_pric", row.get("avg_prvs", row.get("pchs_avg", "0")))
        try:
            qty = int(float(str(qty_raw).replace(",", "")))
        except ValueError:
            qty = 0
        try:
            avg = float(str(avg_raw).replace(",", ""))
        except ValueError:
            avg = 0.0
        if qty > 0:
            positions[symbol] = {"qty": qty, "avg_price": avg}
    return positions


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET required")
    if not args.dry_run and (not args.cano or not args.acnt_prdt_cd):
        raise RuntimeError("KIS_CANO and KIS_ACNT_PRDT_CD required for order placement")

    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise RuntimeError("No symbols provided")

    has_position: Dict[str, bool] = {s: False for s in symbols}
    held_qty: Dict[str, int] = {s: 0 for s in symbols}
    entry_price: Dict[str, float] = {s: 0.0 for s in symbols}
    entry_total_cost: Dict[str, float] = {s: 0.0 for s in symbols}
    held_bars: Dict[str, int] = {s: 0 for s in symbols}
    cooldown_left: Dict[str, int] = {s: 0 for s in symbols}
    entry_bar_index: Dict[str, int] = {s: -10**9 for s in symbols}
    last_entry_bar_index: Dict[str, int] = {s: -10**9 for s in symbols}
    bar_index: Dict[str, int] = {s: 0 for s in symbols}
    peak_price: Dict[str, float] = {s: 0.0 for s in symbols}
    alpha_raw_hist: Dict[str, List[float]] = {s: [] for s in symbols}
    realized_pnl_krw: Dict[str, float] = {s: 0.0 for s in symbols}
    last_price: Dict[str, float] = {s: 0.0 for s in symbols}
    last_processed_bar: Dict[str, str] = {s: "" for s in symbols}
    paper_cash = args.paper_cash
    last_known_cash: float | None = None
    predicted_cash: float | None = None
    cash_fail_streak = 0
    base_log_path = Path(args.log_file)
    base_log_path.parent.mkdir(parents=True, exist_ok=True)
    signal_need = required_bars_for_signal(args)
    warmup_bars = max(1, math.ceil(args.startup_lookback_minutes / max(1, args.bar_minutes)))
    count_hint = max(signal_need + warmup_bars, signal_need + 5, 120)

    model_path = Path(args.ml_model_path)
    if not model_path.exists():
        raise RuntimeError(f"ML model not found: {model_path}")
    with model_path.open("rb") as f:
        ml_bundle = pickle.load(f)
    ml_model = ml_bundle["model"]
    ml_feature_columns: List[str] = list(ml_bundle["feature_columns"])
    ml_reg_model = ml_bundle.get("reg_model")

    def current_log_path(now: datetime) -> Path:
        rotate = max(1, args.log_rotate_minutes)
        if rotate >= 1440:
            bucket = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            bucket_minute = (now.minute // rotate) * rotate
            bucket = now.replace(minute=bucket_minute, second=0, microsecond=0)
        return base_log_path.with_name(
            f"{base_log_path.stem}_{bucket.strftime('%Y%m%d')}{base_log_path.suffix or '.txt'}"
        )

    def emit(message: str, *, save: bool = False) -> None:
        print(message)
        if save:
            is_order_event = (">>> BUY <<<" in message) or ("<<< SELL >>>" in message)
            if not is_order_event:
                return
            path = current_log_path(datetime.now())
            with path.open("a", encoding="utf-8") as f:
                f.write(message + "\n")

    emit("start realtime paper trader", save=True)
    emit(
        f"symbols={','.join(symbols)} dry_run={args.dry_run} bar={args.bar_minutes}m "
        f"model={args.strategy_mode}(signal={args.ml_signal_mode},thr={args.ml_threshold:.2f},"
        f"hold={args.ml_hold_bars},exit_thr={args.exit_threshold:.2f},trail={args.trailing_stop_pct:.4f},"
        f"vwap_hold={args.vwap_exit_min_hold_bars},vwap_max_profit={args.vwap_exit_max_profit_pct:.4f}) "
        f"count_hint={count_hint}",
        save=True,
    )
    dashboard_symbol = symbols[0]
    dashboard_indicator_cols = [c.strip() for c in str(args.live_dashboard_indicator_cols).split(",") if c.strip()]
    dashboard_state = build_live_dashboard_state(dashboard_indicator_cols)
    startup_bars: Dict[str, List[Dict[str, str]]] = {}
    for symbol in symbols:
        try:
            cached = load_startup_history_bars(
                base_url=args.base_url,
                token=token,
                app_key=args.app_key,
                app_secret=args.app_secret,
                symbol=symbol,
                max_bars=max(count_hint, args.startup_history_bars),
            )
            startup_bars[symbol] = cached
            emit(f"startup_history={symbol} bars={len(cached)}", save=True)
        except Exception as e:
            startup_bars[symbol] = []
            emit(f"[WARN] startup_history={symbol} skipped: {str(e)[:180]}", save=True)
    startup_dashboard_ohlc = resample_ohlc(startup_bars.get(dashboard_symbol, []), args.bar_minutes)
    if startup_dashboard_ohlc:
        sync_live_dashboard_state(
            dashboard_state,
            startup_dashboard_ohlc,
            feature_columns=ml_feature_columns,
            clf_model=ml_model,
            reg_model=ml_reg_model,
            signal_mode=args.ml_signal_mode,
            alpha_ret_scale=args.ml_alpha_ret_scale,
            alpha_rank_window=args.ml_alpha_rank_window,
            min_history_bars=args.ml_min_history_bars,
            indicator_cols=dashboard_indicator_cols,
        )
        startup_last_px = float(startup_dashboard_ohlc[-1]["close"])
        update_live_dashboard_summary(
            dashboard_state,
            symbol=dashboard_symbol,
            price=startup_last_px,
            has_position=has_position[dashboard_symbol],
            qty=held_qty[dashboard_symbol],
            entry_price=entry_price[dashboard_symbol],
            cash=last_known_cash,
            cooldown=cooldown_left[dashboard_symbol],
            timestamp=str(startup_dashboard_ohlc[-1].get("date", "")),
        )
        write_live_dashboard_html(
            Path(args.live_dashboard_html),
            symbol=dashboard_symbol,
            state=dashboard_state,
            policy_path=str(DEFAULT_POLICY_PATH),
            score_threshold=float(args.ml_threshold),
            indicator_cols=dashboard_indicator_cols,
            refresh_sec=args.live_dashboard_refresh_sec,
        )
        emit(f"dashboard_saved={args.live_dashboard_html}", save=True)
    if args.dry_run:
        emit(f"paper_cash={paper_cash:,.0f} KRW", save=True)
    else:
        try:
            initial_cash = get_orderable_cash(
                base_url=args.base_url,
                token=token,
                app_key=args.app_key,
                app_secret=args.app_secret,
                cano=args.cano,
                acnt_prdt_cd=args.acnt_prdt_cd,
            )
            last_known_cash = initial_cash
            predicted_cash = initial_cash
            emit(f"initial_orderable_cash={initial_cash:,.0f} KRW", save=True)
            positions = get_positions(
                base_url=args.base_url,
                token=token,
                app_key=args.app_key,
                app_secret=args.app_secret,
                cano=args.cano,
                acnt_prdt_cd=args.acnt_prdt_cd,
            )
            synced = []
            for s in symbols:
                pos = positions.get(s)
                if not pos:
                    continue
                qty = int(pos["qty"])
                avg = float(pos["avg_price"])
                has_position[s] = True
                held_qty[s] = qty
                entry_price[s] = avg
                entry_total_cost[s] = qty * avg * (1.0 + args.fee_rate)
                synced.append(f"{s}:{qty}@{avg:.0f}")
            if synced:
                emit("initial_positions=" + ",".join(synced), save=True)
            else:
                emit("initial_positions=NONE", save=True)
        except Exception as e:
            emit(f"initial_orderable_cash=WARN ({str(e)[:200]})", save=True)
            # Keep running in live mode with a conservative fallback when balance API is unstable.
            last_known_cash = float(args.paper_cash)
            predicted_cash = float(args.paper_cash)
            emit(f"initial_orderable_cash_fallback={args.paper_cash:,.0f} KRW", save=True)
            try:
                positions = get_positions(
                    base_url=args.base_url,
                    token=token,
                    app_key=args.app_key,
                    app_secret=args.app_secret,
                    cano=args.cano,
                    acnt_prdt_cd=args.acnt_prdt_cd,
                )
                synced = []
                for s in symbols:
                    pos = positions.get(s)
                    if not pos:
                        continue
                    qty = int(pos["qty"])
                    avg = float(pos["avg_price"])
                    has_position[s] = True
                    held_qty[s] = qty
                    entry_price[s] = avg
                    entry_total_cost[s] = qty * avg * (1.0 + args.fee_rate)
                    synced.append(f"{s}:{qty}@{avg:.0f}")
                if synced:
                    emit("initial_positions=" + ",".join(synced), save=True)
                else:
                    emit("initial_positions=NONE", save=True)
            except Exception as pos_e:
                emit(f"initial_positions=WARN ({str(pos_e)[:120]})", save=True)

    while True:
        try:
            now_kst = datetime.now(KST)
            hhmm = now_kst.hour * 100 + now_kst.minute
            is_weekday = now_kst.weekday() < 5
            in_session = is_weekday and (900 <= hhmm <= 1530)
            close_dt = now_kst.replace(hour=15, minute=30, second=0, microsecond=0)
            mins_to_close = (close_dt - now_kst).total_seconds() / 60.0
            ease_sell_window = is_weekday and in_session and (3.0 < mins_to_close <= 30.0)
            hard_liquidation_window = is_weekday and in_session and (0.0 <= mins_to_close <= 3.0)
            no_new_buy_window = is_weekday and in_session and (0.0 <= mins_to_close <= max(0, args.no_buy_before_close_min))
            # 0.0 at 30min left -> 1.0 at 3min left
            ease_ratio = 0.0
            if ease_sell_window:
                ease_ratio = (30.0 - mins_to_close) / 27.0
                ease_ratio = max(0.0, min(1.0, ease_ratio))
            if not in_session:
                if args.after_close_action == "exit":
                    emit(f"[{datetime.now().strftime('%H:%M:%S')}] market closed (KST) -> exit", save=True)
                    break
                emit(f"[{datetime.now().strftime('%H:%M:%S')}] market closed (KST) -> waiting", save=False)
                sleep_sec = min(900, seconds_until_next_open(now_kst))
                time.sleep(sleep_sec)
                continue

            cycle_summary: List[str] = []
            cycle_bar_hms = "--:--:--"
            for symbol in symbols:
                ts = datetime.now().strftime("%H:%M:%S")
                try:
                    bars = get_minute_bars(
                        base_url=args.base_url,
                        token=token,
                        app_key=args.app_key,
                        app_secret=args.app_secret,
                        symbol=symbol,
                        count_hint=count_hint,
                        retry=args.retry,
                    )
                except Exception as e:
                    emit(f"[{ts}] {symbol} fetch error: {str(e)[:180]}", save=True)
                    time.sleep(max(0, args.throttle_ms) / 1000.0)
                    continue

                if not bars:
                    emit(f"[{ts}] {symbol} no live bars")
                    time.sleep(max(0, args.throttle_ms) / 1000.0)
                    continue

                combined_bars = merge_unique_bars(startup_bars.get(symbol, []), bars)
                ohlc = resample_ohlc(combined_bars, args.bar_minutes)
                if not ohlc:
                    emit(f"[{ts}] {symbol} no data")
                    continue
                buy_reason = "ml_score"
                sell_reason = "signal"
                cross_info = ""
                last_px = ohlc[-1]["close"]
                last_price[symbol] = last_px
                bar_ts = str(ohlc[-1].get("date", ""))
                if len(bar_ts) >= 19:
                    cycle_bar_hms = bar_ts[11:19]
                if bar_ts and last_processed_bar[symbol] == bar_ts:
                    time.sleep(max(0, args.throttle_ms) / 1000.0)
                    continue
                if bar_ts:
                    last_processed_bar[symbol] = bar_ts
                bar_index[symbol] += 1
                if cooldown_left[symbol] > 0:
                    cooldown_left[symbol] -= 1

                score, m = ml_signal_from_ohlc(
                    ohlc=ohlc,
                    feature_columns=ml_feature_columns,
                    clf_model=ml_model,
                    reg_model=ml_reg_model,
                    signal_mode=args.ml_signal_mode,
                    alpha_ret_scale=args.ml_alpha_ret_scale,
                    alpha_rank_window=args.ml_alpha_rank_window,
                    alpha_raw_hist=alpha_raw_hist[symbol],
                    min_history_bars=args.ml_min_history_bars,
                )
                if score is None or len(ohlc) < max(70, int(args.ml_feature_warmup_bars)):
                    cycle_summary.append(f"{symbol} score=NA warmup")
                    time.sleep(max(0, args.throttle_ms) / 1000.0)
                    continue

                if has_position[symbol]:
                    held_bars[symbol] = max(0, bar_index[symbol] - entry_bar_index[symbol])
                    peak_price[symbol] = max(peak_price[symbol], last_px)
                else:
                    held_bars[symbol] = 0

                signal = 0
                if has_position[symbol]:
                    gross_ret = (last_px / entry_price[symbol] - 1.0) if entry_price[symbol] > 0 else 0.0
                    dd_from_peak = (
                        (last_px / peak_price[symbol] - 1.0) if peak_price[symbol] > 0 else 0.0
                    )
                    vwap_gap_day = float(m.get("vwap_gap_day", 0.0))
                    if hard_liquidation_window:
                        signal = -1
                        sell_reason = "hard_close"
                    elif score is not None and float(score) <= float(args.exit_threshold):
                        signal = -1
                        sell_reason = "score_drop"
                    elif (
                        args.trailing_stop_pct > 0
                        and gross_ret >= max(0.0, args.trailing_stop_pct * 1.5)
                        and dd_from_peak <= -args.trailing_stop_pct
                    ):
                        signal = -1
                        sell_reason = "trailing_stop"
                    elif (
                        args.vwap_exit_min_hold_bars > 0
                        and held_bars[symbol] >= max(1, int(args.vwap_exit_min_hold_bars))
                        and vwap_gap_day <= 0.0
                        and gross_ret <= float(args.vwap_exit_max_profit_pct)
                    ):
                        signal = -1
                        sell_reason = "vwap_break"
                    elif held_bars[symbol] >= max(1, int(args.ml_hold_bars)):
                        signal = -1
                        sell_reason = "timeout"
                else:
                    allow_entry_gap = (bar_index[symbol] - last_entry_bar_index[symbol]) >= max(1, int(args.ml_entry_gap_bars))
                    if cooldown_left[symbol] <= 0 and allow_entry_gap and score >= float(args.ml_threshold):
                        signal = 1

                if signal == 1 and not has_position[symbol]:
                    now_kst_symbol = datetime.now(KST)
                    hhmm_symbol = now_kst_symbol.hour * 100 + now_kst_symbol.minute
                    in_session_symbol = (now_kst_symbol.weekday() < 5) and (900 <= hhmm_symbol <= 1530)
                    bar_hhmm = hhmm_symbol
                    if bars:
                        t_last = str(bars[-1].get("stck_cntg_hour", ""))
                        if len(t_last) == 6:
                            bar_hhmm = int(t_last[:4])
                    morning_no_buy_window = (
                        args.no_buy_morning_start_hhmm <= bar_hhmm < args.no_buy_morning_end_hhmm
                    )
                    mins_to_close_symbol = (
                        now_kst_symbol.replace(hour=15, minute=30, second=0, microsecond=0) - now_kst_symbol
                    ).total_seconds() / 60.0
                    no_new_buy_window_symbol = in_session_symbol and (
                        0.0 <= mins_to_close_symbol <= max(0, args.no_buy_before_close_min)
                    )
                    if (not in_session_symbol) or no_new_buy_window_symbol or morning_no_buy_window:
                        cycle_summary.append(
                            f"{symbol} score={m.get('score', 0):.2f}{cross_info} cd={cooldown_left[symbol]} no_new_buy"
                        )
                        time.sleep(max(0, args.throttle_ms) / 1000.0)
                        continue
                    if no_new_buy_window:
                        cycle_summary.append(
                            f"{symbol} score={m.get('score', 0):.2f}{cross_info} cd={cooldown_left[symbol]} no_new_buy"
                        )
                        time.sleep(max(0, args.throttle_ms) / 1000.0)
                        continue

                    if args.dry_run:
                        orderable_cash = paper_cash
                    else:
                        try:
                            orderable_cash = get_orderable_cash(
                                base_url=args.base_url,
                                token=token,
                                app_key=args.app_key,
                                app_secret=args.app_secret,
                                cano=args.cano,
                                acnt_prdt_cd=args.acnt_prdt_cd,
                            )
                            last_known_cash = orderable_cash
                            predicted_cash = orderable_cash
                            cash_fail_streak = 0
                        except Exception as e:
                            cash_fail_streak += 1
                            fallback_cash = predicted_cash if predicted_cash is not None else last_known_cash
                            if fallback_cash is None:
                                emit(f"[{ts}] {symbol} cash inquiry error: {str(e)[:180]} (no fallback)", save=True)
                                time.sleep(max(0, args.throttle_ms) / 1000.0)
                                continue
                            orderable_cash = fallback_cash
                            emit(
                                f"[{ts}] {symbol} cash inquiry error -> using predicted cash "
                                f"{orderable_cash:,.0f} KRW (fail_streak={cash_fail_streak})",
                                save=True,
                            )
                    open_positions = sum(1 for s in symbols if has_position[s] and held_qty[s] > 0)
                    if open_positions >= args.max_positions:
                        cycle_summary.append(
                            f"{symbol} score={m.get('score', 0):.2f}{cross_info} cd={cooldown_left[symbol]} max_pos"
                        )
                        time.sleep(max(0, args.throttle_ms) / 1000.0)
                        continue

                    mtm_positions = sum(
                        held_qty[s] * (last_price[s] if last_price[s] > 0 else entry_price[s])
                        for s in symbols
                        if has_position[s] and held_qty[s] > 0
                    )
                    portfolio_equity_est = orderable_cash + mtm_positions
                    per_slot_cap = max(0.0, portfolio_equity_est * args.max_invested_pct / max(1, args.max_positions))
                    capped_by_cash = max(0.0, orderable_cash * (1.0 - args.cash_buffer_pct))
                    invested_cap = max(0.0, portfolio_equity_est * args.max_invested_pct)
                    remaining_investable = max(0.0, invested_cap - mtm_positions)
                    order_budget = min(per_slot_cap, capped_by_cash, remaining_investable)
                    if order_budget < args.min_order_krw:
                        cycle_summary.append(
                            f"{symbol} score={m.get('score', 0):.2f}{cross_info} cd={cooldown_left[symbol]}"
                        )
                        time.sleep(max(0, args.throttle_ms) / 1000.0)
                        continue

                    qty = int(order_budget // (last_px * (1.0 + args.fee_rate)))
                    buy_cost = qty * last_px
                    buy_fee = buy_cost * args.fee_rate
                    total_buy = buy_cost + buy_fee
                    if args.dry_run and total_buy > paper_cash:
                        qty = int(paper_cash // (last_px * (1.0 + args.fee_rate)))
                        buy_cost = qty * last_px
                        buy_fee = buy_cost * args.fee_rate
                        total_buy = buy_cost + buy_fee
                    if qty <= 0:
                        cycle_summary.append(
                            f"{symbol} score={m.get('score', 0):.2f}{cross_info} cd={cooldown_left[symbol]} qty=0"
                        )
                        time.sleep(max(0, args.throttle_ms) / 1000.0)
                        continue
                    if args.dry_run:
                        emit(
                            f"[{ts}] >>> BUY <<< {symbol} bar={bar_ts or '-'} qty={qty} "
                            f"score={m.get('score', 0):.2f} prob={m.get('prob', 0):.3f} "
                            f"alpha={m.get('alpha_raw', 0):.4f} "
                            f"cash={orderable_cash:.0f} budget={order_budget:.0f} fee={buy_fee:,.0f} "
                            f"reason={buy_reason} (dry-run)",
                            save=True,
                        )
                        buy_filled = True
                    else:
                        res = place_order(
                            base_url=args.base_url,
                            token=token,
                            app_key=args.app_key,
                            app_secret=args.app_secret,
                            cano=args.cano,
                            acnt_prdt_cd=args.acnt_prdt_cd,
                            symbol=symbol,
                            qty=qty,
                            side="buy",
                        )
                        buy_filled = str(res.get("rt_cd", "")) == "0"
                        if buy_filled:
                            emit(
                                f"[{ts}] >>> BUY <<< {symbol} bar={bar_ts or '-'} qty={qty} px={last_px:.0f} "
                                f"reason={buy_reason} order -> {res.get('msg1', '')} / rt_cd={res.get('rt_cd')}",
                                save=True,
                            )
                        else:
                            emit(
                                f"[{ts}] [BUY_REJECTED] {symbol} bar={bar_ts or '-'} qty={qty} px={last_px:.0f} "
                                f"reason={buy_reason} order -> {res.get('msg1', '')} / rt_cd={res.get('rt_cd')}",
                                save=True,
                            )
                    if qty > 0 and buy_filled:
                        if symbol == dashboard_symbol:
                            dashboard_state["events"].append(
                                {
                                    "date": bar_ts or "",
                                    "kind": "buy",
                                    "price": float(last_px),
                                    "score": float(m.get("score", 0.0)),
                                    "reason": buy_reason,
                                }
                            )
                        if args.dry_run:
                            paper_cash -= total_buy
                        else:
                            if predicted_cash is None:
                                predicted_cash = orderable_cash
                            predicted_cash = max(0.0, predicted_cash - total_buy)
                        has_position[symbol] = True
                        held_qty[symbol] = qty
                        entry_price[symbol] = last_px
                        entry_total_cost[symbol] = total_buy
                        held_bars[symbol] = 0
                        entry_bar_index[symbol] = bar_index[symbol]
                        last_entry_bar_index[symbol] = bar_index[symbol]
                        peak_price[symbol] = last_px
                        total_realized = sum(realized_pnl_krw.values())
                        total_unrealized = sum(
                            (last_price[s] - entry_price[s]) * held_qty[s]
                            for s in symbols
                            if has_position[s] and entry_price[s] > 0 and held_qty[s] > 0
                        )
                        emit(
                            f"[{ts}] portfolio realized={total_realized:,.0f} KRW "
                            f"unrealized={total_unrealized:,.0f} KRW",
                            save=True,
                        )
                elif signal == -1 and has_position[symbol]:
                    qty = held_qty[symbol]
                    sell_gross = last_px * qty
                    sell_fee = sell_gross * args.fee_rate
                    sell_net = sell_gross - sell_fee
                    trade_pnl_krw = sell_net - entry_total_cost[symbol]
                    pnl_pct = (trade_pnl_krw / entry_total_cost[symbol] * 100.0) if entry_total_cost[symbol] > 0 else 0.0
                    if args.dry_run:
                        emit(
                            f"[{ts}] <<< SELL >>> {symbol} bar={bar_ts or '-'} qty={qty} "
                            f"pnl={pnl_pct:.2f}% ({trade_pnl_krw:,.0f} KRW) "
                            f"score={m.get('score', 0):.2f} fee={sell_fee:,.0f} reason={sell_reason} (dry-run)",
                            save=True,
                        )
                        sell_filled = True
                    else:
                        res = place_order(
                            base_url=args.base_url,
                            token=token,
                            app_key=args.app_key,
                            app_secret=args.app_secret,
                            cano=args.cano,
                            acnt_prdt_cd=args.acnt_prdt_cd,
                            symbol=symbol,
                            qty=qty,
                            side="sell",
                        )
                        sell_filled = str(res.get("rt_cd", "")) == "0"
                        if sell_filled:
                            emit(
                                f"[{ts}] <<< SELL >>> {symbol} bar={bar_ts or '-'} qty={qty} px={last_px:.0f} "
                                f"reason={sell_reason} order -> {res.get('msg1', '')} / rt_cd={res.get('rt_cd')}",
                                save=True,
                            )
                        else:
                            emit(
                                f"[{ts}] [SELL_REJECTED] {symbol} bar={bar_ts or '-'} qty={qty} px={last_px:.0f} "
                                f"reason={sell_reason} order -> {res.get('msg1', '')} / rt_cd={res.get('rt_cd')}",
                                save=True,
                            )
                    if not sell_filled:
                        time.sleep(max(0, args.throttle_ms) / 1000.0)
                        continue
                    if symbol == dashboard_symbol:
                        dashboard_state["events"].append(
                            {
                                "date": bar_ts or "",
                                "kind": "sell",
                                "price": float(last_px),
                                "score": float(m.get("score", 0.0)),
                                "reason": sell_reason,
                            }
                        )
                    if args.dry_run:
                        paper_cash += sell_net
                    else:
                        if predicted_cash is None:
                            predicted_cash = 0.0
                        predicted_cash += sell_net
                    realized_pnl_krw[symbol] += trade_pnl_krw
                    has_position[symbol] = False
                    held_qty[symbol] = 0
                    entry_price[symbol] = 0.0
                    entry_total_cost[symbol] = 0.0
                    held_bars[symbol] = 0
                    entry_bar_index[symbol] = -10**9
                    peak_price[symbol] = 0.0
                    cooldown_left[symbol] = 0
                    total_realized = sum(realized_pnl_krw.values())
                    total_unrealized = sum(
                        (last_price[s] - entry_price[s]) * held_qty[s]
                        for s in symbols
                        if has_position[s] and entry_price[s] > 0 and held_qty[s] > 0
                    )
                    emit(
                        f"[{ts}] portfolio realized={total_realized:,.0f} KRW "
                        f"unrealized={total_unrealized:,.0f} KRW",
                        save=True,
                    )
                else:
                    if hard_liquidation_window and has_position[symbol]:
                        qty = held_qty[symbol]
                        sell_gross = last_px * qty
                        sell_fee = sell_gross * args.fee_rate
                        sell_net = sell_gross - sell_fee
                        trade_pnl_krw = sell_net - entry_total_cost[symbol]
                        pnl_pct = (trade_pnl_krw / entry_total_cost[symbol] * 100.0) if entry_total_cost[symbol] > 0 else 0.0
                        if args.dry_run:
                            emit(
                                f"[{ts}] <<< SELL >>> {symbol} bar={bar_ts or '-'} qty={qty} "
                                f"pnl={pnl_pct:.2f}% ({trade_pnl_krw:,.0f} KRW) "
                                f"score={m.get('score', 0):.2f} fee={sell_fee:,.0f} reason=hard_close (dry-run)",
                                save=True,
                            )
                            sell_filled = True
                        else:
                            res = place_order(
                                base_url=args.base_url,
                                token=token,
                                app_key=args.app_key,
                                app_secret=args.app_secret,
                                cano=args.cano,
                                acnt_prdt_cd=args.acnt_prdt_cd,
                                symbol=symbol,
                                qty=qty,
                                side="sell",
                            )
                            sell_filled = str(res.get("rt_cd", "")) == "0"
                            if sell_filled:
                                emit(
                                    f"[{ts}] <<< SELL >>> {symbol} bar={bar_ts or '-'} qty={qty} px={last_px:.0f} "
                                    f"reason=hard_close order -> {res.get('msg1', '')} / rt_cd={res.get('rt_cd')}",
                                    save=True,
                                )
                            else:
                                emit(
                                    f"[{ts}] [SELL_REJECTED] {symbol} bar={bar_ts or '-'} qty={qty} px={last_px:.0f} "
                                    f"reason=hard_close order -> {res.get('msg1', '')} / rt_cd={res.get('rt_cd')}",
                                    save=True,
                                )
                        if not sell_filled:
                            time.sleep(max(0, args.throttle_ms) / 1000.0)
                            continue
                        if args.dry_run:
                            paper_cash += sell_net
                        else:
                            if predicted_cash is None:
                                predicted_cash = 0.0
                            predicted_cash += sell_net
                        realized_pnl_krw[symbol] += trade_pnl_krw
                        has_position[symbol] = False
                        held_qty[symbol] = 0
                        entry_price[symbol] = 0.0
                        entry_total_cost[symbol] = 0.0
                        held_bars[symbol] = 0
                        entry_bar_index[symbol] = -10**9
                        peak_price[symbol] = 0.0
                        cooldown_left[symbol] = 0
                        total_realized = sum(realized_pnl_krw.values())
                        total_unrealized = sum(
                            (last_price[s] - entry_price[s]) * held_qty[s]
                            for s in symbols
                            if has_position[s] and entry_price[s] > 0 and held_qty[s] > 0
                        )
                        emit(
                            f"[{ts}] portfolio realized={total_realized:,.0f} KRW "
                            f"unrealized={total_unrealized:,.0f} KRW",
                            save=True,
                        )
                    else:
                        cycle_summary.append(f"{symbol} score={m.get('score', 0):.2f}{cross_info} cd={cooldown_left[symbol]}")

                if symbol == dashboard_symbol:
                    update_live_dashboard_summary(
                        dashboard_state,
                        symbol=dashboard_symbol,
                        price=last_px,
                        has_position=has_position[symbol],
                        qty=held_qty[symbol],
                        entry_price=entry_price[symbol],
                        cash=last_known_cash,
                        cooldown=cooldown_left[symbol],
                        timestamp=bar_ts,
                    )
                    sync_live_dashboard_state(
                        dashboard_state,
                        ohlc,
                        feature_columns=ml_feature_columns,
                        clf_model=ml_model,
                        reg_model=ml_reg_model,
                        signal_mode=args.ml_signal_mode,
                        alpha_ret_scale=args.ml_alpha_ret_scale,
                        alpha_rank_window=args.ml_alpha_rank_window,
                        min_history_bars=args.ml_min_history_bars,
                        indicator_cols=dashboard_indicator_cols,
                    )
                    write_live_dashboard_html(
                        Path(args.live_dashboard_html),
                        symbol=dashboard_symbol,
                        state=dashboard_state,
                        policy_path=str(DEFAULT_POLICY_PATH),
                        score_threshold=float(args.ml_threshold),
                        indicator_cols=dashboard_indicator_cols,
                        refresh_sec=args.live_dashboard_refresh_sec,
                    )

                time.sleep(max(0, args.throttle_ms) / 1000.0)
            if cycle_summary:
                emit(f"[{datetime.now().strftime('%H:%M:%S')}] {cycle_bar_hms} | " + " | ".join(cycle_summary))
        except KeyboardInterrupt:
            emit("stopped by user", save=True)
            break
        except Exception as e:
            emit(f"[ERR] {e}", save=True)
            if "Unauthorized" in str(e) or "access token" in str(e).lower():
                token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
            time.sleep(max(2, args.interval_sec))
            continue

        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
