from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "logs" / "mplconfig").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from build_ml_dataset import build_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build chart image dataset from OHLCV CSV")
    p.add_argument("--csv", required=True, help="input OHLCV csv (date,open,high,low,close,volume)")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--symbol", default="", help="symbol override")
    p.add_argument("--bar-minutes", type=int, default=3, choices=[1, 3, 5], help="bar size for chart images")
    p.add_argument("--window-bars", type=int, default=40, help="bars per image window")
    p.add_argument("--stride", type=int, default=3, help="window step")
    p.add_argument("--limit", type=int, default=0, help="max images (0 = all)")
    p.add_argument("--horizon-bars", type=int, default=15)
    p.add_argument("--label-mode", default="fixed", choices=["fixed", "atr", "bearish2"])
    p.add_argument("--up-threshold", type=float, default=0.025)
    p.add_argument("--down-threshold", type=float, default=0.012)
    p.add_argument("--atr-up-mult", type=float, default=2.5)
    p.add_argument("--atr-down-mult", type=float, default=1.2)
    p.add_argument("--atr-floor-pct", type=float, default=0.003)
    p.add_argument("--min-history-bars", type=int, default=20)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    return p.parse_args()


def load_ohlcv_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "date": str(row["date"]).strip(),
                        "open": f"{float(row['open']):.6f}",
                        "high": f"{float(row['high']):.6f}",
                        "low": f"{float(row['low']):.6f}",
                        "close": f"{float(row['close']):.6f}",
                        "volume": f"{float(row.get('volume', 0) or 0):.0f}",
                    }
                )
            except Exception:
                continue
    return rows


def resample_rows(rows_1m: List[Dict[str, str]], minutes: int) -> List[Dict[str, str]]:
    if minutes <= 1:
        return list(rows_1m)
    buckets: Dict[str, List[Dict[str, str]]] = {}
    for row in rows_1m:
        dt = row["date"]
        day = dt[:10]
        hh = int(dt[11:13])
        mm = int(dt[14:16])
        floor_mm = (mm // minutes) * minutes
        key = f"{day} {hh:02d}:{floor_mm:02d}:00"
        buckets.setdefault(key, []).append(row)
    out: List[Dict[str, str]] = []
    for key in sorted(buckets.keys()):
        grp = buckets[key]
        open_p = float(grp[0]["open"])
        high_p = max(float(x["high"]) for x in grp)
        low_p = min(float(x["low"]) for x in grp)
        close_p = float(grp[-1]["close"])
        volume = sum(float(x.get("volume", 0) or 0) for x in grp)
        out.append(
            {
                "date": key,
                "open": f"{open_p:.6f}",
                "high": f"{high_p:.6f}",
                "low": f"{low_p:.6f}",
                "close": f"{close_p:.6f}",
                "volume": f"{volume:.0f}",
            }
        )
    return out


def rows_to_data(rows: List[Dict[str, str]]) -> Dict[str, np.ndarray | List[str]]:
    return {
        "date": [r["date"] for r in rows],
        "open": np.array([float(r["open"]) for r in rows], dtype=float),
        "high": np.array([float(r["high"]) for r in rows], dtype=float),
        "low": np.array([float(r["low"]) for r in rows], dtype=float),
        "close": np.array([float(r["close"]) for r in rows], dtype=float),
        "volume": np.array([float(r.get("volume", 0) or 0) for r in rows], dtype=float),
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=float)
    if window <= 0 or values.shape[0] < window:
        return out
    prefix = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    out[window - 1 :] = (prefix[window:] - prefix[:-window]) / window
    return out


def exponential_moving_average(values: np.ndarray, span: int) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=float)
    if values.size == 0 or span <= 0:
        return out
    alpha = 2.0 / (span + 1.0)
    out[0] = values[0]
    for i in range(1, values.shape[0]):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def stochastic_kd(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14, smooth: int = 3) -> tuple[np.ndarray, np.ndarray]:
    raw_k = np.full(close.shape[0], np.nan, dtype=float)
    if close.shape[0] < period:
        return raw_k, raw_k.copy()
    for i in range(period - 1, close.shape[0]):
        hh = np.max(high[i - period + 1 : i + 1])
        ll = np.min(low[i - period + 1 : i + 1])
        denom = hh - ll
        raw_k[i] = 50.0 if denom <= 0 else ((close[i] - ll) / denom) * 100.0
    slow_k = moving_average(np.nan_to_num(raw_k, nan=50.0), smooth)
    slow_d = moving_average(np.nan_to_num(slow_k, nan=50.0), smooth)
    return slow_k, slow_d


def adx_dmi(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    plus_di = np.full(n, np.nan, dtype=float)
    minus_di = np.full(n, np.nan, dtype=float)
    adx = np.full(n, np.nan, dtype=float)
    if n <= period:
        return plus_di, minus_di, adx

    tr = np.zeros(n, dtype=float)
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    tr_smooth = moving_average(tr, period)
    plus_dm_smooth = moving_average(plus_dm, period)
    minus_dm_smooth = moving_average(minus_dm, period)
    valid = tr_smooth > 0
    plus_di[valid] = (plus_dm_smooth[valid] / tr_smooth[valid]) * 100.0
    minus_di[valid] = (minus_dm_smooth[valid] / tr_smooth[valid]) * 100.0

    dx = np.full(n, np.nan, dtype=float)
    denom = plus_di + minus_di
    valid_dx = denom > 0
    dx[valid_dx] = np.abs(plus_di[valid_dx] - minus_di[valid_dx]) / denom[valid_dx] * 100.0
    adx = moving_average(np.nan_to_num(dx, nan=0.0), period)
    return plus_di, minus_di, adx


def same_day_window(rows: List[Dict[str, str]]) -> bool:
    if not rows:
        return False
    start_day = rows[0]["date"][:10]
    end_day = rows[-1]["date"][:10]
    return start_day == end_day


def render_chart_png(rows: List[Dict[str, str]], out_path: Path, width: int, height: int) -> None:
    open_ = np.array([float(r["open"]) for r in rows], dtype=float)
    high = np.array([float(r["high"]) for r in rows], dtype=float)
    low = np.array([float(r["low"]) for r in rows], dtype=float)
    close = np.array([float(r["close"]) for r in rows], dtype=float)
    volume = np.array([float(r.get("volume", 0) or 0) for r in rows], dtype=float)
    ma3 = moving_average(close, 3)
    ma5 = moving_average(close, 5)
    ma10 = moving_average(close, 10)
    ma20 = moving_average(close, 20)
    slow_k, slow_d = stochastic_kd(high, low, close)
    plus_di, minus_di, adx = adx_dmi(high, low, close)

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    gs = fig.add_gridspec(6, 1, hspace=0.04, height_ratios=[3.4, 1.0, 1.0, 1.0, 0.01, 0.01])
    ax_price = fig.add_subplot(gs[0, 0])
    ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_stoch = fig.add_subplot(gs[2, 0], sharex=ax_price)
    ax_adx = fig.add_subplot(gs[3, 0], sharex=ax_price)

    xs = np.arange(len(rows), dtype=float)
    body_w = 0.65
    up_color = "#d32f2f"
    down_color = "#1976d2"

    for i, row in enumerate(rows):
        o = open_[i]
        h = high[i]
        l = low[i]
        c = close[i]
        color = up_color if c >= o else down_color
        ax_price.vlines(xs[i], l, h, color=color, linewidth=1.0, zorder=2)
        low_body = min(o, c)
        body_h = max(abs(c - o), max(c, o) * 0.0002)
        ax_price.add_patch(
            Rectangle(
                (xs[i] - body_w / 2.0, low_body),
                body_w,
                body_h,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                zorder=3,
            )
        )
        ax_vol.bar(xs[i], volume[i], color=color, width=body_w, alpha=0.9)

    ax_price.plot(xs, ma3, color="#2e7d32", linewidth=1.0)
    ax_price.plot(xs, ma5, color="#c62828", linewidth=1.0)
    ax_price.plot(xs, ma10, color="#6a1b9a", linewidth=1.0)
    ax_price.plot(xs, ma20, color="#ef6c00", linewidth=1.0)
    ax_stoch.plot(xs, slow_k, color="#c62828", linewidth=1.0)
    ax_stoch.plot(xs, slow_d, color="#2e7d32", linewidth=1.0)
    ax_stoch.axhline(80.0, color="#bdbdbd", linewidth=0.8, linestyle="--")
    ax_stoch.axhline(20.0, color="#bdbdbd", linewidth=0.8, linestyle="--")
    ax_adx.plot(xs, plus_di, color="#2e7d32", linewidth=1.0)
    ax_adx.plot(xs, minus_di, color="#c62828", linewidth=1.0)
    ax_adx.plot(xs, adx, color="#424242", linewidth=1.1)
    ax_adx.axhline(20.0, color="#bdbdbd", linewidth=0.8, linestyle="--")

    ax_price.set_facecolor("white")
    ax_vol.set_facecolor("white")
    ax_stoch.set_facecolor("white")
    ax_adx.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax_price.grid(True, axis="y", color="#e8e8e8", linewidth=0.6)
    ax_vol.grid(False)
    ax_stoch.grid(True, axis="y", color="#efefef", linewidth=0.5)
    ax_adx.grid(True, axis="y", color="#efefef", linewidth=0.5)

    price_min = float(np.min(low))
    price_max = float(np.max(high))
    pad = max((price_max - price_min) * 0.06, max(price_max, 1.0) * 0.002)
    ax_price.set_ylim(price_min - pad, price_max + pad)
    ax_price.set_xlim(-0.8, len(rows) - 0.2)
    ax_stoch.set_ylim(0, 100)
    adx_max = float(np.nanmax(np.nan_to_num(np.concatenate([plus_di, minus_di, adx]), nan=0.0)))
    ax_adx.set_ylim(0, max(40.0, adx_max * 1.15))

    ax_price.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_vol.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_stoch.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_price.tick_params(axis="y", labelsize=8)
    ax_vol.tick_params(axis="x", labelsize=7, rotation=0)
    ax_vol.tick_params(axis="y", labelsize=7)
    ax_stoch.tick_params(axis="y", labelsize=7)
    ax_adx.tick_params(axis="y", labelsize=7)

    labels = []
    for idx, row in enumerate(rows):
        if idx == 0 or idx == len(rows) - 1 or idx % max(1, len(rows) // 4) == 0:
            labels.append(row["date"][11:16])
        else:
            labels.append("")
    ax_vol.set_xticks(xs)
    ax_vol.set_xticklabels(labels)

    for ax in (ax_price, ax_vol, ax_stoch, ax_adx):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def build_label_map(args: argparse.Namespace, rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    data = rows_to_data(rows)
    labeled_rows = build_rows(
        data=data,
        horizon=args.horizon_bars,
        min_history_bars=args.min_history_bars,
        label_mode=args.label_mode,
        up_threshold=args.up_threshold,
        down_threshold=args.down_threshold,
        atr_up_mult=args.atr_up_mult,
        atr_down_mult=args.atr_down_mult,
        atr_floor_pct=args.atr_floor_pct,
    )
    return {str(row["date"]): row for row in labeled_rows}


def infer_symbol(args: argparse.Namespace, csv_path: Path) -> str:
    if args.symbol:
        return str(args.symbol).strip()
    stem = csv_path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if 4 <= len(digits) <= 6:
        return digits.zfill(6)
    return "unknown"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)

    rows_src = load_ohlcv_csv(csv_path)
    if not rows_src:
        raise RuntimeError("input csv has no usable rows")
    rows = resample_rows(rows_src, args.bar_minutes)
    label_map = build_label_map(args, rows)
    symbol = infer_symbol(args, csv_path)

    image_root = out_dir / "images"
    meta_rows: List[Dict[str, str | int]] = []
    generated = 0

    for end_idx in range(max(0, args.window_bars - 1), len(rows), max(1, args.stride)):
        if args.limit > 0 and generated >= int(args.limit):
            break
        end_date = rows[end_idx]["date"]
        label_row = label_map.get(end_date)
        if label_row is None:
            continue
        window = rows[end_idx - args.window_bars + 1 : end_idx + 1]
        if len(window) < args.window_bars or not same_day_window(window):
            continue

        label = int(label_row["label"])
        day = end_date[:10].replace("-", "")
        t = end_date[11:19].replace(":", "")
        filename = f"{symbol}_{day}_{t}_w{args.window_bars}_b{args.bar_minutes}.png"
        rel_path = Path(str(label)) / filename
        out_path = image_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        render_chart_png(window, out_path, args.width, args.height)

        meta_rows.append(
            {
                "symbol": symbol,
                "date": end_date,
                "label": label,
                "fwd_close_ret": label_row["fwd_close_ret"],
                "image_path": str((Path("images") / rel_path).as_posix()),
                "window_bars": args.window_bars,
                "bar_minutes": args.bar_minutes,
            }
        )
        generated += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = out_dir / "metadata.csv"
    with meta_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["symbol", "date", "label", "fwd_close_ret", "image_path", "window_bars", "bar_minutes"],
        )
        writer.writeheader()
        writer.writerows(meta_rows)

    summary = {
        "csv": str(csv_path),
        "symbol": symbol,
        "bar_minutes": args.bar_minutes,
        "window_bars": args.window_bars,
        "stride": args.stride,
        "generated_images": generated,
        "label_mode": args.label_mode,
        "out_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
