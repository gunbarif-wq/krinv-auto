from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import requests
from fetch_kis_daily import DEFAULT_BASE_URL, get_access_token
from fetch_kis_minute import iter_business_days


DEFAULT_SYMBOLS = ["225190"]


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
    p = argparse.ArgumentParser(
        description="Fetch 1m data and prepare train/val/test backtest datasets in one run"
    )
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma-separated symbols")
    p.add_argument("--business-days", type=int, default=30, help="number of business days to fetch")
    p.add_argument("--include-today", action="store_true", default=True, help="include today's data")
    p.add_argument("--train-ratio", type=float, default=0.7, help="train split ratio by time")
    p.add_argument("--val-ratio", type=float, default=0.15, help="validation split ratio by time")
    p.add_argument("--out-dir", default="data/backtest_sets_225190_1y", help="output root directory")
    p.add_argument("--max-bars-per-day", type=int, default=450)
    p.add_argument("--pause-ms", type=int, default=500)
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", DEFAULT_BASE_URL))
    return p.parse_args()


def save_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(rows)


def split_rows(
    rows: List[Dict[str, str]], train_ratio: float, val_ratio: float
) -> tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    if not rows:
        return [], [], []

    n = len(rows)
    train_r = max(0.05, min(0.95, train_ratio))
    val_r = max(0.0, min(0.90, val_ratio))
    if train_r + val_r >= 0.98:
        val_r = max(0.0, 0.98 - train_r)

    if n == 1:
        return rows, [], []

    train_n = max(1, int(round(n * train_r)))
    val_n = int(round(n * val_r))
    test_n = n - train_n - val_n

    # Keep at least one row for test when possible.
    if test_n < 1:
        deficit = 1 - test_n
        take = min(val_n, deficit)
        val_n -= take
        deficit -= take
        if deficit > 0:
            train_n = max(1, train_n - deficit)
        test_n = n - train_n - val_n

    # Keep indices in range in edge cases.
    if train_n + val_n > n - 1:
        overflow = train_n + val_n - (n - 1)
        take = min(val_n, overflow)
        val_n -= take
        overflow -= take
        if overflow > 0:
            train_n = max(1, train_n - overflow)

    cut1 = max(1, min(n, train_n))
    cut2 = max(cut1, min(n, cut1 + max(0, val_n)))
    return rows[:cut1], rows[cut1:cut2], rows[cut2:]


def fetch_one_day_1m_10230(
    app_key: str,
    app_secret: str,
    token: str,
    base_url: str,
    symbol: str,
    yyyymmdd: str,
    pause_ms: int = 500,
    max_rows: int = 480,
) -> List[Dict[str, str]]:
    # 10230 is real-domain only.
    if "openapivts" in base_url:
        raise RuntimeError("10230 API is not supported on mock(VTS) domain. Use https://openapi.koreainvestment.com:9443")

    url = f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice"
    out: Dict[str, Dict[str, str]] = {}
    req_count = 0
    cursor_time = "153000"
    last_cursor = ""

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
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_HOUR_1": cursor_time,
            "FID_INPUT_DATE_1": yyyymmdd,
            "FID_PW_DATA_INCU_YN": "Y",
            "FID_FAKE_TICK_INCU_YN": "",
        }
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("rt_cd") != "0":
            raise RuntimeError(f"rt_cd={data.get('rt_cd')} msg1={data.get('msg1')}")

        rows = data.get("output2", []) or []
        if not rows:
            break
        min_time = None
        for x in rows:
            d = str(x.get("stck_bsop_date", "")).strip()
            t = str(x.get("stck_cntg_hour", "")).strip()
            if d != yyyymmdd or len(t) != 6:
                continue
            if min_time is None or t < min_time:
                min_time = t
            dt = f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
            out[dt] = {
                "date": dt,
                "open": str(x.get("stck_oprc", "0")),
                "high": str(x.get("stck_hgpr", "0")),
                "low": str(x.get("stck_lwpr", "0")),
                "close": str(x.get("stck_prpr", "0")),
                "volume": str(x.get("cntg_vol", "0")),
            }

        req_count += 1
        if len(out) >= max_rows:
            break
        # Move cursor backward by oldest bar time from this response.
        if not min_time:
            break
        # Prevent infinite loop when API keeps returning same tail chunk.
        if min_time == last_cursor or min_time >= cursor_time:
            break
        last_cursor = cursor_time
        cursor_time = min_time
        if cursor_time <= "090000":
            break
        if req_count > 20:
            break
        time.sleep(max(0, pause_ms) / 1000.0)

    return [out[k] for k in sorted(out.keys())]


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET required")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise RuntimeError("No symbols provided")

    today = datetime.now().date()
    end_d = today if args.include_today else (today - timedelta(days=1))
    # Build a wide calendar window first, then take the most recent N business days.
    cal_lookback = max(30, args.business_days * 4)
    start_d = end_d - timedelta(days=cal_lookback)
    start = start_d.strftime("%Y%m%d")
    end = end_d.strftime("%Y%m%d")
    days_all = iter_business_days(start, end)
    days = days_all[-max(1, args.business_days) :]
    if not days:
        raise RuntimeError("No business days in requested range")

    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
    root = Path(args.out_dir)
    train_dir = root / "train_1m"
    val_dir = root / "val_1m"
    test_dir = root / "test_1m"
    merged_dir = root / "full_1m"

    print(
        f"period={days[0].strftime('%Y%m%d')}~{days[-1].strftime('%Y%m%d')} "
        f"business_days={len(days)} symbols={len(symbols)} include_today={args.include_today}"
    )
    print(f"output={root}")

    for sym in symbols:
        merged: Dict[str, Dict[str, str]] = {}
        for d in days:
            ymd = d.strftime("%Y%m%d")
            try:
                rows = fetch_one_day_1m_10230(
                    app_key=args.app_key,
                    app_secret=args.app_secret,
                    token=token,
                    symbol=sym,
                    yyyymmdd=ymd,
                    max_rows=args.max_bars_per_day,
                    pause_ms=args.pause_ms,
                    base_url=args.base_url,
                )
                for r in rows:
                    merged[r["date"]] = r
                time.sleep(max(0, args.pause_ms) / 1000.0)
            except Exception as e:
                print(f"[WARN] {sym} {ymd} skipped: {e}")

        rows_all = [merged[k] for k in sorted(merged.keys())]
        if len(rows_all) < 200:
            print(f"[WARN] {sym} low rows={len(rows_all)} (may be insufficient)")
        train_rows, val_rows, test_rows = split_rows(rows_all, args.train_ratio, args.val_ratio)

        save_csv(merged_dir / f"{sym}_1m_full.csv", rows_all)
        save_csv(train_dir / f"{sym}_1m_train.csv", train_rows)
        save_csv(val_dir / f"{sym}_1m_val.csv", val_rows)
        save_csv(test_dir / f"{sym}_1m_test.csv", test_rows)
        print(
            f"{sym}: full={len(rows_all)} train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
            f"train_end={train_rows[-1]['date'] if train_rows else '-'} "
            f"val_end={val_rows[-1]['date'] if val_rows else '-'}"
        )

    print("done")


if __name__ == "__main__":
    main()
