from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from fetch_kis_daily import get_access_token
from fetch_kis_minute import fetch_one_day_1m, iter_business_days
from user_only_strategy.build_chart_image_dataset import build_label_map, render_chart_png, resample_rows, same_day_window
from user_only_strategy.monday_custom_timing_bot import fetch_candidate_universe, load_dotenv
from user_only_strategy.train_chart_image_classifier import main as train_classifier_main  # type: ignore


ETF_KEYWORDS = (
    "KODEX",
    "TIGER",
    "KOSEF",
    "KBSTAR",
    "HANARO",
    "ARIRANG",
    "SOL",
    "ACE",
    "PLUS",
    "ETN",
    "레버리지",
    "인버스",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch live candidate symbols, build chart-image dataset, and retrain classifier")
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", "https://openapivts.koreainvestment.com:29443"))
    p.add_argument("--quote-base-url", default=os.getenv("KIS_QUOTE_BASE_URL", "https://openapi.koreainvestment.com:9443"))
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--symbols", type=int, default=50, help="number of candidate symbols to select (ignored when --use-selected-summary is set)")
    # NOTE: Historically this was 30 business days. For "6 months" on KRX, a reasonable default is ~126 business days.
    p.add_argument("--fetch-days", type=int, default=30, help="business days of minute data per symbol (alias of --fetch-bdays)")
    p.add_argument("--fetch-bdays", type=int, default=0, help="business days of minute data per symbol (overrides --fetch-days when >0)")
    p.add_argument("--exclude-recent-bdays", type=int, default=0, help="exclude the most recent N business days (e.g., to avoid overlapping an existing training window)")
    p.add_argument(
        "--use-selected-summary",
        default="",
        help="path to a previous retrain summary.json to reuse its selected_symbols list (recommended for 'same symbols, different period')",
    )
    p.add_argument("--bar-minutes", type=int, default=3, choices=[1, 3, 5])
    p.add_argument("--window-bars", type=int, default=40)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--limit-per-symbol", type=int, default=60)
    p.add_argument("--label-mode", default="fixed", choices=["fixed", "atr", "bearish2"])
    p.add_argument("--target-side", default="long", choices=["long", "short"])
    p.add_argument("--up-threshold", type=float, default=0.015)
    p.add_argument("--down-threshold", type=float, default=0.01)
    p.add_argument("--horizon-bars", type=int, default=15)
    p.add_argument("--out-root", default="data/chart_retrain")
    p.add_argument("--model-name", default="", help="override model basename")
    p.add_argument(
        "--preferred-model",
        default="",
        choices=["", "logistic_raw", "pca_logistic", "pca_random_forest", "pca_gradient_boosting", "pca_catboost"],
        help="forward to train_chart_image_classifier.py",
    )
    p.add_argument(
        "--preferred-threshold",
        type=float,
        default=-1.0,
        help="forward to train_chart_image_classifier.py",
    )
    p.add_argument("--pause-ms", type=int, default=380)
    p.add_argument("--max-bars-per-day", type=int, default=450)
    p.add_argument("--retry-count", type=int, default=2, help="retries per business day fetch")
    p.add_argument("--max-day-errors-per-symbol", type=int, default=18, help="stop a symbol early after too many failed days")
    p.add_argument("--max-consecutive-day-errors", type=int, default=8, help="stop a symbol early after consecutive failed days")
    p.add_argument("--skip-etf", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fetch-only", action="store_true", help="only fetch raw minute data and summary, skip image build/train")
    p.add_argument("--symbol-offset", type=int, default=0, help="start index within selected symbols (for parallel shard runs)")
    p.add_argument("--symbol-limit", type=int, default=0, help="max number of selected symbols to process from offset (0=all)")
    p.add_argument("--reuse-raw-root", default="", help="reuse an existing raw_1m directory instead of fetching again")
    p.add_argument("--offline-raw", action=argparse.BooleanOptionalAction, default=False, help="use existing raw CSVs only and do not fetch missing days")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="reuse existing per-symbol datasets when present")
    return p.parse_args()


def fetch_one_day_1m_10230(
    app_key: str,
    app_secret: str,
    token: str,
    symbol: str,
    yyyymmdd: str,
    pause_ms: int,
    max_rows: int,
    base_url: str,
    market_code: str = "J",
) -> List[Dict[str, str]]:
    if "openapivts" in str(base_url).lower():
        raise RuntimeError("10230 historical minute API is not supported on mock(VTS) domain")

    url = f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
    out: Dict[str, Dict[str, str]] = {}
    cursor_time = "153000"
    last_cursor = ""
    req_count = 0

    while len(out) < max_rows:
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": app_key,
            "appsecret": app_secret,
            "tr_id": "FHKST03010230",
            "custtype": "P",
        }
        params = {
            "FID_COND_MRKT_DIV_CODE": market_code,
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_HOUR_1": cursor_time,
            "FID_INPUT_DATE_1": yyyymmdd,
            "FID_PW_DATA_INCU_YN": "Y",
            "FID_FAKE_TICK_INCU_YN": "",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if str(data.get("rt_cd")) != "0":
            raise RuntimeError(f"rt_cd={data.get('rt_cd')} msg1={data.get('msg1')}")

        rows = data.get("output2", []) or []
        if not rows:
            break

        min_time = None
        for row in rows:
            d = str(row.get("stck_bsop_date", "")).strip()
            t = str(row.get("stck_cntg_hour", "")).strip()
            if len(t) != 6 or len(d) != 8:
                continue
            if min_time is None or t < min_time:
                min_time = t
            dt = f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
            out[dt] = {
                "date": dt,
                "open": str(row.get("stck_oprc", "0")),
                "high": str(row.get("stck_hgpr", "0")),
                "low": str(row.get("stck_lwpr", "0")),
                "close": str(row.get("stck_prpr", "0")),
                "volume": str(row.get("cntg_vol", "0")),
            }

        req_count += 1
        if len(out) >= max_rows or not min_time:
            break
        if min_time == last_cursor or min_time >= cursor_time:
            break
        last_cursor = cursor_time
        cursor_time = min_time
        if cursor_time <= "090000" or req_count > 20:
            break
        time.sleep(max(0, pause_ms) / 1000.0)

    return [out[k] for k in sorted(out.keys())]


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return "EGW00201" in msg or "초당 거래건수를 초과" in msg


def _is_token_expired_error(exc: Exception) -> bool:
    msg = str(exc)
    return "EGW00123" in msg or "기간이 만료된 token" in msg


def _is_transient_network_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(
        token in msg
        for token in (
            "Read timed out",
            "RemoteDisconnected",
            "Connection aborted",
            "Max retries exceeded",
            "Connection reset",
            "temporarily unavailable",
        )
    )


def should_skip_name(name: str) -> bool:
    nm = str(name or "").strip().upper()
    if not nm:
        return False
    return any(token.upper() in nm for token in ETF_KEYWORDS)


def recent_business_range(days: int) -> Tuple[str, str]:
    today = datetime.now().date()
    got = []
    d = today
    while len(got) < max(1, days):
        if d.weekday() < 5:
            got.append(d)
        d -= timedelta(days=1)
    got.sort()
    return got[0].strftime("%Y%m%d"), got[-1].strftime("%Y%m%d")


def recent_business_days_list(*, days: int, exclude_recent_bdays: int) -> List[str]:
    """
    Return a list of YYYYMMDD business days, ending at (today - exclude_recent_bdays business days).
    Example:
      days=30, exclude_recent_bdays=0 -> last 30 business days up to today.
      days=126, exclude_recent_bdays=30 -> the 126 business days ending 30 business days ago.
    """
    today = datetime.now().date()
    want = max(1, int(days))
    skip = max(0, int(exclude_recent_bdays))

    picked: List[str] = []
    d = today
    # First skip the most recent N business days.
    while skip > 0:
        if d.weekday() < 5:
            skip -= 1
        d -= timedelta(days=1)

    # Now collect N business days.
    while len(picked) < want:
        if d.weekday() < 5:
            picked.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)

    picked.sort()
    return picked


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(rows)


def csv_unique_days(path: Path) -> int:
    if not path.exists():
        return 0
    days = set()
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dt = str(row.get("date", "")).strip()
            if len(dt) >= 10:
                days.add(dt[:10])
    return len(days)


def load_existing_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def csv_day_set(rows: List[Dict[str, str]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        dt = str(row.get("date", "")).strip()
        if len(dt) >= 10:
            out.add(dt[:10].replace("-", ""))
    return out


def build_symbol_dataset(
    rows_1m: List[Dict[str, str]],
    symbol: str,
    out_dir: Path,
    *,
    bar_minutes: int,
    window_bars: int,
    stride: int,
    limit: int,
    label_mode: str,
    target_side: str,
    up_threshold: float,
    down_threshold: float,
    horizon_bars: int,
) -> Dict[str, int]:
    rows = resample_rows(rows_1m, bar_minutes)
    label_map = build_label_map(
        argparse.Namespace(
            horizon_bars=horizon_bars,
            min_history_bars=20,
            label_mode=label_mode,
            target_side=target_side,
            up_threshold=up_threshold,
            down_threshold=down_threshold,
            atr_up_mult=2.5,
            atr_down_mult=1.2,
            atr_floor_pct=0.003,
        ),
        rows,
    )
    image_root = out_dir / "images"
    meta_rows: List[Dict[str, str | int]] = []
    generated = 0
    for end_idx in range(max(0, window_bars - 1), len(rows), max(1, stride)):
        if limit > 0 and generated >= limit:
            break
        end_date = rows[end_idx]["date"]
        label_row = label_map.get(end_date)
        if label_row is None:
            continue
        window = rows[end_idx - window_bars + 1 : end_idx + 1]
        if len(window) < window_bars or not same_day_window(window):
            continue
        label = int(label_row["label"])
        day = end_date[:10].replace("-", "")
        t = end_date[11:19].replace(":", "")
        filename = f"{symbol}_{day}_{t}_w{window_bars}_b{bar_minutes}.png"
        rel_path = Path("images") / str(label) / filename
        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        render_chart_png(window, out_path, 640, 640)
        meta_rows.append(
            {
                "symbol": symbol,
                "date": end_date,
                "label": label,
                "fwd_close_ret": label_row["fwd_close_ret"],
                "image_path": str(rel_path.as_posix()),
                "window_bars": window_bars,
                "bar_minutes": bar_minutes,
            }
        )
        generated += 1

    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["symbol", "date", "label", "fwd_close_ret", "image_path", "window_bars", "bar_minutes"],
        )
        writer.writeheader()
        writer.writerows(meta_rows)
    counts = Counter(int(r["label"]) for r in meta_rows)
    return {"images": generated, "label0": counts.get(0, 0), "label1": counts.get(1, 0)}


def combine_metadata(per_symbol_root: Path, combined_dir: Path) -> Dict[str, int]:
    combined_dir.mkdir(parents=True, exist_ok=True)
    merged_rows: List[Dict[str, str]] = []
    counts = Counter()
    for symbol_dir in sorted(p for p in per_symbol_root.iterdir() if p.is_dir()):
        meta_path = symbol_dir / "metadata.csv"
        if not meta_path.exists():
            continue
        with meta_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["image_path"] = str((symbol_dir / row["image_path"]).resolve().as_posix())
                merged_rows.append(row)
                counts[int(row["label"])] += 1
    with (combined_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["symbol", "date", "label", "fwd_close_ret", "image_path", "window_bars", "bar_minutes"],
        )
        writer.writeheader()
        writer.writerows(merged_rows)
    return {"rows": len(merged_rows), "label0": counts.get(0, 0), "label1": counts.get(1, 0)}


def load_existing_symbol_stats(symbol_out: Path) -> Dict[str, int] | None:
    meta_path = symbol_out / "metadata.csv"
    if not meta_path.exists():
        return None
    counts = Counter()
    rows = 0
    with meta_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                counts[int(row["label"])] += 1
                rows += 1
            except Exception:
                continue
    return {"images": rows, "label0": counts.get(0, 0), "label1": counts.get(1, 0)}


def main() -> None:
    load_dotenv(str(ROOT / ".env"))
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET required")

    quote_base_url = str(args.quote_base_url or args.base_url).strip()
    token = get_access_token(args.app_key, args.app_secret, base_url=quote_base_url)
    picked: List[Tuple[str, str]] = []
    if str(args.use_selected_summary or "").strip():
        summary_path = Path(str(args.use_selected_summary)).expanduser()
        if not summary_path.is_absolute():
            summary_path = (ROOT / summary_path).resolve()
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        for obj in payload.get("selected_symbols", []) or []:
            s = str(obj.get("symbol", "")).strip()
            n = str(obj.get("name", "")).strip()
            if not s:
                continue
            picked.append((s, n))
        if not picked:
            raise RuntimeError(f"no selected_symbols in summary: {summary_path}")
    else:
        universe = fetch_candidate_universe(
            quote_base_url,
            token,
            args.app_key,
            args.app_secret,
            max_universe=max(1, int(args.symbols * 2)),
        )
        for symbol, name in universe:
            if args.skip_etf and should_skip_name(name):
                continue
            picked.append((symbol, name))
            if len(picked) >= int(args.symbols):
                break
        if not picked:
            raise RuntimeError("no candidate symbols selected")

    symbol_offset = max(0, int(args.symbol_offset))
    symbol_limit = max(0, int(args.symbol_limit))
    if symbol_offset > 0:
        picked = picked[symbol_offset:]
    if symbol_limit > 0:
        picked = picked[:symbol_limit]
    if not picked:
        raise RuntimeError("no symbols remain after applying symbol offset/limit")

    fetch_bdays = int(args.fetch_bdays) if int(args.fetch_bdays) > 0 else int(args.fetch_days)
    ymd_list = recent_business_days_list(days=fetch_bdays, exclude_recent_bdays=int(args.exclude_recent_bdays))
    start_ymd, end_ymd = ymd_list[0], ymd_list[-1]
    run_name = args.model_name.strip() or f"live50_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root = (ROOT / args.out_root / run_name).resolve()
    raw_root = out_root / "raw_1m"
    per_symbol_root = out_root / "per_symbol"
    combined_dir = out_root / "combined"
    reuse_raw_root = Path(args.reuse_raw_root).resolve() if str(args.reuse_raw_root).strip() else None
    model_out = ROOT / "data" / "chart_models" / f"{run_name}.pkl"
    report_out = ROOT / "data" / "chart_models" / f"{run_name}.report.json"

    summary = {
        "run_name": run_name,
        "selected_symbols": [{"symbol": s, "name": n} for s, n in picked],
        "symbol_offset": symbol_offset,
        "symbol_limit": symbol_limit,
        "fetch_range": {"start": start_ymd, "end": end_ymd},
        "fetch_bdays": fetch_bdays,
        "exclude_recent_bdays": int(args.exclude_recent_bdays),
        "per_symbol": {},
    }

    for idx, (symbol, name) in enumerate(picked, start=1):
        raw_csv = raw_root / f"{symbol}_1m.csv"
        source_csv = (reuse_raw_root / f"{symbol}_1m.csv") if reuse_raw_root else raw_csv
        existing_csv = raw_csv if raw_csv.exists() else source_csv
        if args.fetch_only and csv_unique_days(existing_csv) >= max(1, fetch_bdays - 1):
            summary["per_symbol"][symbol] = {"name": name, "rows_1m": -1, "images": 0, "label0": 0, "label1": 0, "skipped_existing": True}
            print(f"[{idx}/{len(picked)}] {symbol} {name} skip_existing")
            continue
        existing_rows = load_existing_csv_rows(existing_csv) if existing_csv.exists() else []
        existing_days = csv_day_set(existing_rows)
        all_rows: List[Dict[str, str]] = list(existing_rows)
        day_errors = 0
        consecutive_day_errors = 0
        remaining_days = [] if args.offline_raw else [ymd for ymd in ymd_list if ymd not in existing_days]
        print(f"[{idx}/{len(picked)}] {symbol} {name} start remaining_days={len(remaining_days)} existing_days={len(existing_days)}")
        if remaining_days:
            for ymd in remaining_days:
                last_exc: Exception | None = None
                day_rows: List[Dict[str, str]] = []
                for attempt in range(max(1, int(args.retry_count))):
                    try:
                        day_rows = fetch_one_day_1m(
                            app_key=args.app_key,
                            app_secret=args.app_secret,
                            token=token,
                            symbol=symbol,
                            yyyymmdd=ymd,
                            max_bars_per_day=int(args.max_bars_per_day),
                            pause_ms=int(args.pause_ms),
                            base_url=quote_base_url,
                        )
                        if not day_rows and fetch_bdays > 1:
                            day_rows = fetch_one_day_1m_10230(
                                app_key=args.app_key,
                                app_secret=args.app_secret,
                                token=token,
                                symbol=symbol,
                                yyyymmdd=ymd,
                                pause_ms=int(args.pause_ms),
                                max_rows=int(args.max_bars_per_day),
                                base_url=quote_base_url,
                            )
                        break
                    except Exception as exc:
                        last_exc = exc
                        if _is_token_expired_error(exc):
                            token = get_access_token(args.app_key, args.app_secret, base_url=quote_base_url)
                            time.sleep(1.2)
                            continue
                        if _is_rate_limit_error(exc):
                            time.sleep(2.5 * (attempt + 1))
                            continue
                        if _is_transient_network_error(exc):
                            time.sleep(3.0 * (attempt + 1))
                            continue
                        time.sleep(0.8)
                        break
                if day_rows:
                    all_rows.extend(day_rows)
                    consecutive_day_errors = 0
                    if args.fetch_only:
                        dedup_partial = {r["date"]: r for r in all_rows}
                        rows_partial = [dedup_partial[k] for k in sorted(dedup_partial.keys())]
                        write_csv(raw_csv, rows_partial)
                        print(f"[day] {symbol} {ymd} rows={len(day_rows)} total_rows={len(rows_partial)}")
                elif last_exc is not None:
                    day_errors += 1
                    consecutive_day_errors += 1
                    print(f"[warn] {symbol} {ymd} {type(last_exc).__name__}: {last_exc}")
                    if day_errors >= max(1, int(args.max_day_errors_per_symbol)):
                        print(f"[skip] {symbol} too_many_day_errors={day_errors}")
                        break
                    if consecutive_day_errors >= max(1, int(args.max_consecutive_day_errors)):
                        print(f"[skip] {symbol} consecutive_day_errors={consecutive_day_errors}")
                        break
        dedup = {r["date"]: r for r in all_rows}
        rows_1m = [dedup[k] for k in sorted(dedup.keys())]
        if not rows_1m:
            summary["per_symbol"][symbol] = {"name": name, "rows_1m": 0, "images": 0, "label0": 0, "label1": 0}
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        write_csv(raw_csv, rows_1m)
        if args.fetch_only:
            summary["per_symbol"][symbol] = {"name": name, "rows_1m": len(rows_1m), "images": 0, "label0": 0, "label1": 0}
            print(f"[{idx}/{len(picked)}] {symbol} {name} rows={len(rows_1m)} fetched_only")
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        symbol_out = per_symbol_root / symbol
        if args.resume:
            existing_stats = load_existing_symbol_stats(symbol_out)
            if existing_stats is not None:
                existing_stats["rows_1m"] = len(rows_1m)
                existing_stats["name"] = name
                summary["per_symbol"][symbol] = existing_stats
                print(f"[{idx}/{len(picked)}] {symbol} {name} reuse_existing images={existing_stats['images']} pos={existing_stats['label1']}")
                out_root.mkdir(parents=True, exist_ok=True)
                (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
        stats = build_symbol_dataset(
            rows_1m,
            symbol,
            symbol_out,
            bar_minutes=int(args.bar_minutes),
            window_bars=int(args.window_bars),
            stride=int(args.stride),
            limit=int(args.limit_per_symbol),
            label_mode=str(args.label_mode),
            target_side=str(args.target_side),
            up_threshold=float(args.up_threshold),
            down_threshold=float(args.down_threshold),
            horizon_bars=int(args.horizon_bars),
        )
        stats["rows_1m"] = len(rows_1m)
        stats["name"] = name
        summary["per_symbol"][symbol] = stats
        print(f"[{idx}/{len(picked)}] {symbol} {name} rows={len(rows_1m)} images={stats['images']} pos={stats['label1']}")
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    combined_stats = {"rows": 0, "label0": 0, "label1": 0}
    if not args.fetch_only:
        combined_stats = combine_metadata(per_symbol_root, combined_dir)
    summary["combined"] = combined_stats
    summary["model_out"] = str(model_out)
    summary["report_out"] = str(report_out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.fetch_only:
        print(json.dumps({"run_name": run_name, "combined": combined_stats, "fetch_only": True, "out_root": str(out_root)}, ensure_ascii=False))
        return

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "train_chart_image_classifier.py",
            "--dataset-dir",
            str(combined_dir),
            "--model-out",
            str(model_out),
            "--report-out",
            str(report_out),
            "--target-side",
            str(args.target_side),
        ]
        if str(args.preferred_model or "").strip():
            sys.argv.extend(["--preferred-model", str(args.preferred_model)])
        if float(args.preferred_threshold) > 0:
            sys.argv.extend(["--preferred-threshold", str(float(args.preferred_threshold))])
        train_classifier_main()
    finally:
        sys.argv = old_argv

    env_path = ROOT / ".env"
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        replaced = False
        out_lines = []
        for line in lines:
            if line.startswith("CHART_CLASSIFIER_MODEL="):
                out_lines.append(f"CHART_CLASSIFIER_MODEL={model_out.as_posix()}")
                replaced = True
            else:
                out_lines.append(line)
        if not replaced:
            out_lines.append(f"CHART_CLASSIFIER_MODEL={model_out.as_posix()}")
        env_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(json.dumps({"run_name": run_name, "combined": combined_stats, "model_out": str(model_out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
