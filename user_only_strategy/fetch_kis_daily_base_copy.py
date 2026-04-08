from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

import requests


DEFAULT_BASE_URL = "https://openapi.koreainvestment.com:9443"
TOKEN_CACHE_PATH = ".kis_token_cache.json"


def _load_cached_token(app_key: str, base_url: str) -> str | None:
    if not os.path.exists(TOKEN_CACHE_PATH):
        return None
    try:
        with open(TOKEN_CACHE_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        item = obj.get(f"{base_url}|{app_key}", {})
        token = item.get("access_token", "")
        expires_at = item.get("expires_at", "")
        if not token or not expires_at:
            return None
        if datetime.utcnow() + timedelta(seconds=30) >= datetime.fromisoformat(expires_at):
            return None
        return token
    except Exception:
        return None


def _save_cached_token(app_key: str, base_url: str, token: str, expires_in: int) -> None:
    obj: Dict[str, Dict[str, str]] = {}
    if os.path.exists(TOKEN_CACHE_PATH):
        try:
            with open(TOKEN_CACHE_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            obj = {}
    expires_at = (datetime.utcnow() + timedelta(seconds=max(60, int(expires_in) - 30))).isoformat()
    obj[f"{base_url}|{app_key}"] = {"access_token": token, "expires_at": expires_at}
    with open(TOKEN_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_access_token(app_key: str, app_secret: str, base_url: str = DEFAULT_BASE_URL) -> str:
    cached = _load_cached_token(app_key, base_url)
    if cached:
        return cached

    url = f"{base_url}/oauth2/tokenP"
    payload = {
        "grant_type": "client_credentials",
        "appkey": app_key,
        "appsecret": app_secret,
    }
    resp = requests.post(url, json=payload, timeout=15)
    if resp.status_code >= 400:
        body = resp.text[:400]
        raise RuntimeError(
            "토큰 발급 실패. "
            f"status={resp.status_code}, base_url={base_url}, body={body}\n"
            "실전키면 openapi(9443), 모의키면 openapivts(29443) 사용 여부를 확인하세요."
        )
    data = resp.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"token 발급 실패: {data}")
    _save_cached_token(app_key, base_url, token, int(data.get("expires_in", 3600)))
    return token


def fetch_daily_prices(
    app_key: str,
    app_secret: str,
    symbol: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    access_token: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
) -> List[Dict[str, str]]:
    token = access_token or get_access_token(app_key, app_secret, base_url=base_url)
    data = {}
    rows = []
    try:
        data = _call_chart_price(
            token=token,
            app_key=app_key,
            app_secret=app_secret,
            symbol=symbol,
            start_yyyymmdd=start_yyyymmdd,
            end_yyyymmdd=end_yyyymmdd,
            base_url=base_url,
        )
        rows = data.get("output2", [])
    except requests.HTTPError:
        rows = []

    if not rows:
        try:
            data = _call_daily_price(
                token=token,
                app_key=app_key,
                app_secret=app_secret,
                symbol=symbol,
                start_yyyymmdd=start_yyyymmdd,
                end_yyyymmdd=end_yyyymmdd,
                base_url=base_url,
            )
            rows = data.get("output2", [])
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            body = e.response.text[:300] if e.response is not None else ""
            raise RuntimeError(
                f"시세 조회 HTTP 오류(symbol={symbol}, status={status}). body={body}"
            ) from e

    if not rows:
        msg = data.get("msg1", "응답에 output2 데이터가 없습니다.")
        raise RuntimeError(
            "저장할 데이터가 없습니다. "
            f"(rt_cd={data.get('rt_cd')}, msg={msg})\n"
            "앱키 권한(모의/실전), 종목코드, 날짜 범위를 확인하세요."
        )

    result: List[Dict[str, str]] = []
    for r in rows:
        result.append(
            {
                "date": _fmt_date(r.get("stck_bsop_date", "")),
                "open": r.get("stck_oprc", "0"),
                "high": r.get("stck_hgpr", "0"),
                "low": r.get("stck_lwpr", "0"),
                "close": r.get("stck_clpr", "0"),
                "volume": r.get("acml_vol", "0"),
            }
        )
    result.sort(key=lambda x: x["date"])
    return result


def _common_headers(token: str, app_key: str, app_secret: str, tr_id: str) -> Dict[str, str]:
    return {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": tr_id,
    }


def _call_chart_price(
    token: str,
    app_key: str,
    app_secret: str,
    symbol: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    base_url: str,
) -> Dict:
    url = f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    headers = _common_headers(token, app_key, app_secret, "FHKST03010100")
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": symbol,
        "FID_INPUT_DATE_1": start_yyyymmdd,
        "FID_INPUT_DATE_2": end_yyyymmdd,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "0",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _call_daily_price(
    token: str,
    app_key: str,
    app_secret: str,
    symbol: str,
    start_yyyymmdd: str,
    end_yyyymmdd: str,
    base_url: str,
) -> Dict:
    url = f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    headers = _common_headers(token, app_key, app_secret, "FHKST01010400")
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": symbol,
        "FID_INPUT_DATE_1": start_yyyymmdd,
        "FID_INPUT_DATE_2": end_yyyymmdd,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "0",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def save_csv(rows: List[Dict[str, str]], out_path: str) -> None:
    if not rows:
        raise RuntimeError("저장할 데이터가 없습니다.")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["date", "open", "high", "low", "close", "volume"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _fmt_date(yyyymmdd: str) -> str:
    return datetime.strptime(yyyymmdd, "%Y%m%d").strftime("%Y-%m-%d")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KIS 일봉 시세를 CSV로 저장")
    p.add_argument("--symbol", required=True, help="종목코드 (예: 005930)")
    p.add_argument("--start", required=True, help="시작일 YYYYMMDD")
    p.add_argument("--end", required=True, help="종료일 YYYYMMDD")
    p.add_argument("--out", required=True, help="출력 CSV 경로")
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", DEFAULT_BASE_URL))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError(
            "앱키/시크릿이 없습니다. --app-key --app-secret 또는 "
            "환경변수 KIS_APP_KEY, KIS_APP_SECRET 를 설정하세요."
        )

    rows = fetch_daily_prices(
        app_key=args.app_key,
        app_secret=args.app_secret,
        symbol=args.symbol,
        start_yyyymmdd=args.start,
        end_yyyymmdd=args.end,
        base_url=args.base_url,
    )
    save_csv(rows, args.out)
    print(f"saved: {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
