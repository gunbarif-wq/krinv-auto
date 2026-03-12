from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import date, datetime, timedelta
from typing import Dict, List

import requests

from fetch_kis_daily import DEFAULT_BASE_URL, get_access_token


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KIS 분봉(1/3분) 시세를 CSV로 저장")
    p.add_argument("--symbol", required=True, help="종목코드 (예: 012450)")
    p.add_argument("--start", required=True, help="시작일 YYYYMMDD")
    p.add_argument("--end", required=True, help="종료일 YYYYMMDD")
    p.add_argument("--out", required=True, help="출력 CSV 경로")
    p.add_argument("--interval", type=int, choices=[1, 3, 5], default=1, help="분봉 간격")
    p.add_argument("--max-bars-per-day", type=int, default=450, help="일자별 최대 수집 개수")
    p.add_argument("--pause-ms", type=int, default=380, help="요청 간 대기(ms)")
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", DEFAULT_BASE_URL))
    return p.parse_args()


def iter_business_days(start_yyyymmdd: str, end_yyyymmdd: str) -> List[date]:
    s = datetime.strptime(start_yyyymmdd, "%Y%m%d").date()
    e = datetime.strptime(end_yyyymmdd, "%Y%m%d").date()
    out: List[date] = []
    d = s
    while d <= e:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def fetch_one_day_1m(
    app_key: str,
    app_secret: str,
    token: str,
    symbol: str,
    yyyymmdd: str,
    max_bars_per_day: int,
    pause_ms: int,
    base_url: str,
) -> List[Dict[str, str]]:
    url = f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
    headers = {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST03010200",
    }

    rows_out: List[Dict[str, str]] = []
    cursor_time = "153000"
    seen = set()

    while len(rows_out) < max_bars_per_day:
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": yyyymmdd,
            "FID_INPUT_HOUR_1": cursor_time,
            "FID_PW_DATA_INCU_YN": "N",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        if resp.status_code >= 400:
            body = resp.text[:300]
            raise RuntimeError(f"status={resp.status_code}, body={body}")
        data = resp.json()
        rows = data.get("output2", [])
        if not rows:
            break

        appended = 0
        for r in rows:
            d = r.get("stck_bsop_date", "")
            t = r.get("stck_cntg_hour", "")
            if d != yyyymmdd or len(t) != 6:
                continue
            dt = datetime.strptime(d + t, "%Y%m%d%H%M%S")
            k = dt.strftime("%Y-%m-%d %H:%M:%S")
            if k in seen:
                continue
            seen.add(k)
            rows_out.append(
                {
                    "date": k,
                    "open": r.get("stck_oprc", "0"),
                    "high": r.get("stck_hgpr", "0"),
                    "low": r.get("stck_lwpr", "0"),
                    "close": r.get("stck_prpr", "0"),
                    "volume": r.get("cntg_vol", "0"),
                }
            )
            appended += 1
            if len(rows_out) >= max_bars_per_day:
                break

        if appended == 0:
            break
        last = rows[-1].get("stck_cntg_hour", "")
        if not last or last == cursor_time:
            break
        cursor_time = last
        time.sleep(max(0, pause_ms) / 1000.0)

    return sorted(rows_out, key=lambda x: x["date"])


def resample_3m(rows_1m: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return resample_nm(rows_1m, 3)


def resample_nm(rows_1m: List[Dict[str, str]], n: int) -> List[Dict[str, str]]:
    buckets: Dict[datetime, List[Dict[str, str]]] = {}
    for r in rows_1m:
        dt = datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S")
        floor_minute = (dt.minute // n) * n
        key = dt.replace(minute=floor_minute, second=0, microsecond=0)
        buckets.setdefault(key, []).append(r)

    out: List[Dict[str, str]] = []
    for key in sorted(buckets.keys()):
        grp = buckets[key]
        open_p = float(grp[0]["open"])
        high_p = max(float(x["high"]) for x in grp)
        low_p = min(float(x["low"]) for x in grp)
        close_p = float(grp[-1]["close"])
        vol = sum(float(x.get("volume", 0)) for x in grp)
        out.append(
            {
                "date": key.strftime("%Y-%m-%d %H:%M:%S"),
                "open": f"{open_p:.2f}",
                "high": f"{high_p:.2f}",
                "low": f"{low_p:.2f}",
                "close": f"{close_p:.2f}",
                "volume": f"{vol:.0f}",
            }
        )
    return out


def save_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("저장할 분봉 데이터가 없습니다.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("앱키/시크릿이 없습니다. KIS_APP_KEY, KIS_APP_SECRET 설정 필요")

    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
    dates = iter_business_days(args.start, args.end)
    all_1m: List[Dict[str, str]] = []

    for d in dates:
        yyyymmdd = d.strftime("%Y%m%d")
        try:
            day_rows = fetch_one_day_1m(
                app_key=args.app_key,
                app_secret=args.app_secret,
                token=token,
                symbol=args.symbol,
                yyyymmdd=yyyymmdd,
                max_bars_per_day=args.max_bars_per_day,
                pause_ms=args.pause_ms,
                base_url=args.base_url,
            )
            if day_rows:
                all_1m.extend(day_rows)
            time.sleep(max(0, args.pause_ms) / 1000.0)
        except Exception as e:
            print(f"[WARN] {yyyymmdd} skipped: {e}")

    # de-dup
    uniq: Dict[str, Dict[str, str]] = {}
    for r in all_1m:
        uniq[r["date"]] = r
    rows_1m = [uniq[k] for k in sorted(uniq.keys())]

    if args.interval == 3:
        rows = resample_nm(rows_1m, 3)
    elif args.interval == 5:
        rows = resample_nm(rows_1m, 5)
    else:
        rows = rows_1m

    save_csv(args.out, rows)
    print(f"saved: {args.out} ({len(rows)} rows, interval={args.interval}m)")


if __name__ == "__main__":
    main()
