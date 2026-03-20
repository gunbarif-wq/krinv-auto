from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_30",
    "ret_60",
    "hl_pct",
    "oc_pct",
    "candle_body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "gap_prev_close",
    "high_prev_close",
    "low_prev_close",
    "vol_ratio20",
    "vol_ratio60",
    "vol_z20",
    "vol_z60",
    "vol_ret_1",
    "vol_ret_5",
    "turnover_z20",
    "volatility20",
    "volatility60",
    "ma_gap_5_20",
    "ma_gap_10_20",
    "ma_gap_20_60",
    "macd_plus",
    "slow_k14_3",
    "price_z20",
    "price_z60",
    "momentum_20",
    "rsi14",
    "atr14_pct",
    "boll_width20",
    "day_ret_open",
    "day_range_pos",
    "vwap_gap_day",
    "minute_norm",
]

TRIO_FEATURE_COLUMNS = [
    "ma_gap_5_20",
    "ma_gap_10_20",
    "ma_gap_20_60",
    "macd_plus",
    "slow_k14_3",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ML dataset (features + labels) from 1m OHLCV CSV splits")
    p.add_argument("--data-root", default="data/backtest_sets_047810_5y")
    p.add_argument("--symbol", default="047810")
    p.add_argument("--horizon-bars", type=int, default=30)
    p.add_argument("--label-mode", default="fixed", choices=["fixed", "atr"])
    p.add_argument("--up-threshold", type=float, default=0.012)
    p.add_argument("--down-threshold", type=float, default=0.007)
    p.add_argument("--atr-up-mult", type=float, default=2.0)
    p.add_argument("--atr-down-mult", type=float, default=1.2)
    p.add_argument("--atr-floor-pct", type=float, default=0.003)
    p.add_argument("--out-dir", default="data/ml")
    return p.parse_args()


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if window <= 0 or arr.shape[0] < window:
        return out
    prefix = np.concatenate(([0.0], np.cumsum(arr, dtype=float)))
    out[window - 1 :] = (prefix[window:] - prefix[:-window]) / window
    return out


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if window <= 1 or arr.shape[0] < window:
        return out
    prefix = np.concatenate(([0.0], np.cumsum(arr, dtype=float)))
    prefix2 = np.concatenate(([0.0], np.cumsum(arr * arr, dtype=float)))
    s = prefix[window:] - prefix[:-window]
    s2 = prefix2[window:] - prefix2[:-window]
    mean = s / window
    var = np.maximum(0.0, s2 / window - mean * mean)
    out[window - 1 :] = np.sqrt(var)
    return out


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape[0], np.nan, dtype=float)
    ok = np.isfinite(den) & (np.abs(den) > 1e-12)
    out[ok] = num[ok] / den[ok]
    return out


def ema_vec(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if window <= 0 or arr.shape[0] == 0:
        return out
    alpha = 2.0 / (window + 1.0)
    out[0] = arr[0]
    for i in range(1, arr.shape[0]):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if window <= 0 or arr.shape[0] < window:
        return out
    for i in range(window - 1, arr.shape[0]):
        out[i] = float(np.min(arr[i - window + 1 : i + 1]))
    return out


def rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if window <= 0 or arr.shape[0] < window:
        return out
    for i in range(window - 1, arr.shape[0]):
        out[i] = float(np.max(arr[i - window + 1 : i + 1]))
    return out


def rolling_minmax01(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(arr.shape[0], np.nan, dtype=float)
    if window <= 1 or arr.shape[0] < window:
        return out
    lo = rolling_min(arr, window)
    hi = rolling_max(arr, window)
    den = hi - lo
    ok = np.isfinite(arr) & np.isfinite(lo) & np.isfinite(hi) & (den > 1e-12)
    out[ok] = (arr[ok] - lo[ok]) / den[ok]
    flat = np.isfinite(arr) & np.isfinite(lo) & np.isfinite(hi) & (den <= 1e-12)
    out[flat] = 0.5
    return out


def rsi_vec(close: np.ndarray, window: int) -> np.ndarray:
    diff = close - np.roll(close, 1)
    gain = np.where(diff > 0.0, diff, 0.0)
    loss = np.where(diff < 0.0, -diff, 0.0)
    avg_gain = rolling_mean(gain, window)
    avg_loss = rolling_mean(loss, window)
    out = np.full(close.shape[0], np.nan, dtype=float)
    ok = np.isfinite(avg_gain) & np.isfinite(avg_loss)
    if not np.any(ok):
        return out
    rs = np.full(close.shape[0], np.nan, dtype=float)
    good_den = ok & (avg_loss > 1e-12)
    rs[good_den] = avg_gain[good_den] / avg_loss[good_den]
    out[good_den] = 100.0 - (100.0 / (1.0 + rs[good_den]))
    only_gain = ok & (avg_loss <= 1e-12) & (avg_gain > 1e-12)
    out[only_gain] = 100.0
    flat = ok & (avg_loss <= 1e-12) & (avg_gain <= 1e-12)
    out[flat] = 50.0
    return out


def intraday_vectors(
    dates: List[str], o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, v: np.ndarray
) -> Dict[str, np.ndarray]:
    n = c.shape[0]
    day_open = np.full(n, np.nan, dtype=float)
    day_high_so_far = np.full(n, np.nan, dtype=float)
    day_low_so_far = np.full(n, np.nan, dtype=float)
    vwap_day = np.full(n, np.nan, dtype=float)
    cur_day = ""
    cur_open = np.nan
    cur_high = np.nan
    cur_low = np.nan
    pv = 0.0
    vv = 0.0
    for i, dt in enumerate(dates):
        day = dt[:10] if len(dt) >= 10 else ""
        if day != cur_day:
            cur_day = day
            cur_open = o[i]
            cur_high = h[i]
            cur_low = l[i]
            pv = 0.0
            vv = 0.0
        cur_high = max(cur_high, h[i])
        cur_low = min(cur_low, l[i])
        typ = (h[i] + l[i] + c[i]) / 3.0
        pv += typ * max(0.0, v[i])
        vv += max(0.0, v[i])
        day_open[i] = cur_open
        day_high_so_far[i] = cur_high
        day_low_so_far[i] = cur_low
        vwap_day[i] = (pv / vv) if vv > 1e-12 else c[i]
    return {
        "day_open": day_open,
        "day_high_so_far": day_high_so_far,
        "day_low_so_far": day_low_so_far,
        "vwap_day": vwap_day,
    }


def load_split(path: Path) -> Dict[str, np.ndarray | List[str]]:
    dates: List[str] = []
    opens: List[float] = []
    highs: List[float] = []
    lows: List[float] = []
    closes: List[float] = []
    volumes: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                dates.append(r["date"])
                opens.append(float(r["open"]))
                highs.append(float(r["high"]))
                lows.append(float(r["low"]))
                closes.append(float(r["close"]))
                volumes.append(float(r.get("volume", 0) or 0))
            except Exception:
                continue
    return {
        "date": dates,
        "open": np.asarray(opens, dtype=float),
        "high": np.asarray(highs, dtype=float),
        "low": np.asarray(lows, dtype=float),
        "close": np.asarray(closes, dtype=float),
        "volume": np.asarray(volumes, dtype=float),
    }


def minute_norm_vec(dates: List[str]) -> np.ndarray:
    out = np.full(len(dates), np.nan, dtype=float)
    for i, dt in enumerate(dates):
        if len(dt) < 16:
            continue
        hh = int(dt[11:13])
        mm = int(dt[14:16])
        minute = hh * 60 + mm
        out[i] = (minute - 540.0) / 390.0
    return out


def build_rows(
    data: Dict[str, np.ndarray | List[str]],
    horizon: int,
    label_mode: str,
    up_threshold: float,
    down_threshold: float,
    atr_up_mult: float,
    atr_down_mult: float,
    atr_floor_pct: float,
) -> List[Dict[str, str]]:
    dates = data["date"]  # type: ignore[assignment]
    o = data["open"]  # type: ignore[assignment]
    h = data["high"]  # type: ignore[assignment]
    l = data["low"]  # type: ignore[assignment]
    c = data["close"]  # type: ignore[assignment]
    v = data["volume"]  # type: ignore[assignment]

    n = c.shape[0]
    if n <= horizon + 30:
        return []

    ret_1 = safe_div(c - np.roll(c, 1), np.roll(c, 1))
    ret_3 = safe_div(c - np.roll(c, 3), np.roll(c, 3))
    ret_5 = safe_div(c - np.roll(c, 5), np.roll(c, 5))
    ret_10 = safe_div(c - np.roll(c, 10), np.roll(c, 10))
    ret_20 = safe_div(c - np.roll(c, 20), np.roll(c, 20))
    ret_30 = safe_div(c - np.roll(c, 30), np.roll(c, 30))
    ret_60 = safe_div(c - np.roll(c, 60), np.roll(c, 60))
    hl_pct = safe_div(h - l, c)
    oc_pct = safe_div(c - o, o)
    candle_body_pct = safe_div(np.abs(c - o), o)
    upper_wick_pct = safe_div(h - np.maximum(o, c), c)
    lower_wick_pct = safe_div(np.minimum(o, c) - l, c)
    gap_prev_close = safe_div(o - np.roll(c, 1), np.roll(c, 1))
    high_prev_close = safe_div(h - np.roll(c, 1), np.roll(c, 1))
    low_prev_close = safe_div(l - np.roll(c, 1), np.roll(c, 1))

    vol_mean20 = rolling_mean(v, 20)
    vol_mean60 = rolling_mean(v, 60)
    vol_std20 = rolling_std(v, 20)
    vol_std60 = rolling_std(v, 60)
    vol_ratio20 = safe_div(v, vol_mean20)
    vol_ratio60 = safe_div(v, vol_mean60)
    vol_z20 = safe_div(v - vol_mean20, vol_std20)
    vol_z60 = safe_div(v - vol_mean60, vol_std60)
    vol_ret_1 = safe_div(v - np.roll(v, 1), np.roll(v, 1))
    vol_ret_5 = safe_div(v - np.roll(v, 5), np.roll(v, 5))
    turnover = c * v
    turnover_mean20 = rolling_mean(turnover, 20)
    turnover_std20 = rolling_std(turnover, 20)
    turnover_z20 = safe_div(turnover - turnover_mean20, turnover_std20)
    volatility20 = rolling_std(ret_1, 20)
    volatility60 = rolling_std(ret_1, 60)

    ma5 = rolling_mean(c, 5)
    ma10 = rolling_mean(c, 10)
    ma20 = rolling_mean(c, 20)
    ma60 = rolling_mean(c, 60)
    ma_gap_5_20 = safe_div(ma5 - ma20, ma20)
    ma_gap_10_20 = safe_div(ma10 - ma20, ma20)
    ma_gap_20_60 = safe_div(ma20 - ma60, ma60)
    ema12 = ema_vec(c, 12)
    ema26 = ema_vec(c, 26)
    macd_line = ema12 - ema26
    macd_signal = ema_vec(macd_line, 9)
    macd_hist = macd_line - macd_signal
    macd_plus = rolling_minmax01(macd_hist, 120)
    ll14 = rolling_min(l, 14)
    hh14 = rolling_max(h, 14)
    fast_k14 = np.full(n, 0.5, dtype=float)
    den_k14 = hh14 - ll14
    ok_k14 = np.isfinite(c) & np.isfinite(ll14) & np.isfinite(hh14) & (den_k14 > 1e-12)
    fast_k14[ok_k14] = (c[ok_k14] - ll14[ok_k14]) / den_k14[ok_k14]
    slow_k14_3 = rolling_mean(fast_k14, 3)
    close_std20 = rolling_std(c, 20)
    close_std60 = rolling_std(c, 60)
    price_z20 = safe_div(c - ma20, close_std20)
    price_z60 = safe_div(c - ma60, close_std60)
    momentum_20 = ret_20.copy()
    rsi14 = safe_div(rsi_vec(c, 14), np.full(n, 100.0))  # normalize to 0..1
    prev_close = np.roll(c, 1)
    tr = np.maximum.reduce([h - l, np.abs(h - prev_close), np.abs(l - prev_close)])
    atr14 = rolling_mean(tr, 14)
    atr14_pct = safe_div(atr14, c)
    boll_width20 = safe_div(4.0 * close_std20, ma20)
    intra = intraday_vectors(dates, o, h, l, c, v)  # type: ignore[arg-type]
    day_ret_open = safe_div(c - intra["day_open"], intra["day_open"])
    day_range = intra["day_high_so_far"] - intra["day_low_so_far"]
    day_range_pos = safe_div(c - intra["day_low_so_far"], day_range)
    vwap_gap_day = safe_div(c - intra["vwap_day"], intra["vwap_day"])
    minute_norm = minute_norm_vec(dates)  # type: ignore[arg-type]

    start = 60
    end = n - horizon
    out: List[Dict[str, str]] = []
    for i in range(start, end):
        feats = np.array(
            [
                ret_1[i],
                ret_3[i],
                ret_5[i],
                ret_10[i],
                ret_20[i],
                ret_30[i],
                ret_60[i],
                hl_pct[i],
                oc_pct[i],
                candle_body_pct[i],
                upper_wick_pct[i],
                lower_wick_pct[i],
                gap_prev_close[i],
                high_prev_close[i],
                low_prev_close[i],
                vol_ratio20[i],
                vol_ratio60[i],
                vol_z20[i],
                vol_z60[i],
                vol_ret_1[i],
                vol_ret_5[i],
                turnover_z20[i],
                volatility20[i],
                volatility60[i],
                ma_gap_5_20[i],
                ma_gap_10_20[i],
                ma_gap_20_60[i],
                macd_plus[i],
                slow_k14_3[i],
                price_z20[i],
                price_z60[i],
                momentum_20[i],
                rsi14[i],
                atr14_pct[i],
                boll_width20[i],
                day_ret_open[i],
                day_range_pos[i],
                vwap_gap_day[i],
                minute_norm[i],
            ],
            dtype=float,
        )
        if not np.isfinite(feats).all():
            continue

        entry = c[i]
        if label_mode == "atr":
            atr_here = float(atr14[i]) if np.isfinite(atr14[i]) and atr14[i] > 0 else 0.0
            atr_floor = entry * max(0.0, float(atr_floor_pct))
            atr_used = max(atr_here, atr_floor)
            up_level = entry + atr_used * max(0.0, float(atr_up_mult))
            down_level = max(0.0, entry - atr_used * max(0.0, float(atr_down_mult)))
        else:
            up_level = entry * (1.0 + up_threshold)
            down_level = entry * (1.0 - down_threshold)
        fut_h = h[i + 1 : i + 1 + horizon]
        fut_l = l[i + 1 : i + 1 + horizon]

        up_hits = np.where(fut_h >= up_level)[0]
        down_hits = np.where(fut_l <= down_level)[0]
        up_idx = int(up_hits[0]) if up_hits.size else None
        down_idx = int(down_hits[0]) if down_hits.size else None
        label = 1 if (up_idx is not None and (down_idx is None or up_idx < down_idx)) else 0
        fwd_close_ret = c[i + horizon] / entry - 1.0

        row = {
            "date": str(dates[i]),
            "close": f"{entry:.6f}",
            "label": str(label),
            "fwd_close_ret": f"{fwd_close_ret:.8f}",
        }
        for col, val in zip(FEATURE_COLUMNS, feats):
            row[col] = f"{val:.8f}"
        out.append(row)
    return out


def save_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["date", "close", *FEATURE_COLUMNS, "label", "fwd_close_ret"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def resolve_split_path(data_root: Path, symbol: str, split: str) -> Path:
    if split == "val":
        return data_root / "val_1m" / f"{symbol}_1m_val.csv"
    return data_root / f"{split}_1m" / f"{symbol}_1m_{split}.csv"


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) / args.symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        src = resolve_split_path(data_root, args.symbol, split)
        if not src.exists():
            raise RuntimeError(f"missing split file: {src}")
        data = load_split(src)
        rows = build_rows(
            data=data,
            horizon=args.horizon_bars,
            label_mode=args.label_mode,
            up_threshold=args.up_threshold,
            down_threshold=args.down_threshold,
            atr_up_mult=args.atr_up_mult,
            atr_down_mult=args.atr_down_mult,
            atr_floor_pct=args.atr_floor_pct,
        )
        dst = out_dir / f"{args.symbol}_{split}_ml.csv"
        save_rows(dst, rows)
        print(f"{split}: {len(rows)} rows -> {dst}")

    meta = {
        "symbol": args.symbol,
        "horizon_bars": args.horizon_bars,
        "label_mode": args.label_mode,
        "up_threshold": args.up_threshold,
        "down_threshold": args.down_threshold,
        "atr_up_mult": args.atr_up_mult,
        "atr_down_mult": args.atr_down_mult,
        "atr_floor_pct": args.atr_floor_pct,
        "features": FEATURE_COLUMNS,
    }
    (out_dir / f"{args.symbol}_ml_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"meta: {out_dir / f'{args.symbol}_ml_meta.json'}")


if __name__ == "__main__":
    main()
