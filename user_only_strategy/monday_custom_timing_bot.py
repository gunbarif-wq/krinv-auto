from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import requests
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from fetch_kis_daily import fetch_daily_prices, get_access_token  # noqa: E402


KST = ZoneInfo("Asia/Seoul")
VTS_BASE_URL = "https://openapivts.koreainvestment.com:29443"
PROD_BASE_URL = "https://openapi.koreainvestment.com:9443"
DEFAULT_TRADING_OPEN_HHMM = 800
DEFAULT_TRADING_CLOSE_HHMM = 2000
DEFAULT_SEARCH_START_HHMM = 700
DEFAULT_MINUTE_MARKET_CODE = "UN"
SYMBOL_ALIAS_MAP = {
    "하이닉스": "000660",
    "sk하이닉스": "000660",
    "에스케이하이닉스": "000660",
    "삼전": "005930",
    "삼성전자": "005930",
    "넥스원": "079550",
    "lig넥스원": "079550",
    "엘아이지넥스원": "079550",
    "lk삼양": "225190",
    "엘케이삼양": "225190",
}


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


def request_json_with_retry(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    params: Dict[str, str],
    timeout: int = 15,
    retries: int = 3,
    sleep_base: float = 0.7,
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
    if last_err:
        raise last_err
    raise RuntimeError("request failed")


def is_demo_base_url(base_url: str) -> bool:
    return "openapivts" in base_url.lower()


def order_tr_id(base_url: str, side: str) -> str:
    demo = is_demo_base_url(base_url)
    if side.lower() == "buy":
        return "VTTC0012U" if demo else "TTTC0012U"
    return "VTTC0011U" if demo else "TTTC0011U"


def order_inquiry_tr_id(base_url: str) -> str:
    demo = is_demo_base_url(base_url)
    return "VTTC8001R" if demo else "TTTC8001R"


def get_hashkey(base_url: str, app_key: str, app_secret: str, body: Dict) -> str:
    url = f"{base_url}/uapi/hashkey"
    headers = {"appKey": app_key, "appSecret": app_secret, "content-type": "application/json"}
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=15)
    r.raise_for_status()
    return r.json().get("HASH", "")


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
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=15)
    if r.status_code >= 400:
        return {"rt_cd": "1", "msg1": f"http_{r.status_code} {r.text[:220]}"}
    return r.json()


def _to_int(value: str | int | float | None) -> int:
    try:
        if value is None:
            return 0
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return 0


def _to_float(value: str | int | float | None) -> float:
    try:
        if value is None:
            return 0.0
        return float(str(value).replace(",", "").strip())
    except Exception:
        return 0.0


def inquire_daily_order_row(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    cano: str,
    acnt_prdt_cd: str,
    odno: str,
) -> Dict:
    url = f"{base_url}/uapi/domestic-stock/v1/trading/inquire-daily-ccld"
    today = datetime.now(KST).strftime("%Y%m%d")
    headers = {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": order_inquiry_tr_id(base_url),
        "custtype": "P",
    }
    params = {
        "CANO": cano,
        "ACNT_PRDT_CD": acnt_prdt_cd,
        "INQR_STRT_DT": today,
        "INQR_END_DT": today,
        "SLL_BUY_DVSN_CD": "00",
        "INQR_DVSN": "00",
        "PDNO": "",
        "CCLD_DVSN": "00",
        "ORD_GNO_BRNO": "",
        "ODNO": "",
        "INQR_DVSN_3": "00",
        "INQR_DVSN_1": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }
    data = request_json_with_retry("get", url, headers=headers, params=params, timeout=15, retries=2)
    rows = data.get("output1", [])
    if not isinstance(rows, list):
        return {}
    target = str(odno).strip().lstrip("0")
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_odno = str(row.get("odno", "")).strip().lstrip("0")
        if target and row_odno == target:
            return row
    return {}


def wait_order_fill_status(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    cano: str,
    acnt_prdt_cd: str,
    odno: str,
    *,
    retries: int = 3,
    sleep_sec: float = 1.0,
) -> Tuple[str, int, int]:
    last_status = ("pending", 0, 0)
    for i in range(max(1, retries)):
        row = inquire_daily_order_row(
            base_url=base_url,
            token=token,
            app_key=app_key,
            app_secret=app_secret,
            cano=cano,
            acnt_prdt_cd=acnt_prdt_cd,
            odno=odno,
        )
        if row:
            ord_qty = _to_int(row.get("ord_qty"))
            filled = _to_int(row.get("tot_ccld_qty"))
            if ord_qty > 0 and filled >= ord_qty:
                return ("filled", filled, ord_qty)
            if filled > 0:
                return ("partial", filled, ord_qty)
            last_status = ("pending", filled, ord_qty)
        if i + 1 < retries:
            time.sleep(max(0.2, float(sleep_sec)))
    return last_status


def fetch_ranking_rows(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    api_url: str,
    tr_id: str,
    params: Dict[str, str],
) -> List[Dict]:
    url = f"{base_url}{api_url}"
    headers = {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": tr_id,
        "custtype": "P",
    }
    data = request_json_with_retry("get", url, headers=headers, params=params, timeout=15, retries=3)
    if str(data.get("rt_cd", "0")) not in {"0", ""}:
        return []
    for key in ("output", "output1", "output2"):
        rows = data.get(key)
        if isinstance(rows, list):
            return [x for x in rows if isinstance(x, dict)]
    return []


def extract_symbol_name(row: Dict) -> Tuple[str, str]:
    symbol_keys = ("mksc_shrn_iscd", "stck_shrn_iscd", "pdno", "isu_cd", "shrn_iscd")
    name_keys = ("hts_kor_isnm", "prdt_name", "isu_nm", "stck_shrn_iscd_name")
    symbol = ""
    for k in symbol_keys:
        v = str(row.get(k, "")).strip()
        if v and v.isdigit() and 4 <= len(v) <= 6:
            symbol = v.zfill(6)
            break
    name = ""
    for k in name_keys:
        v = str(row.get(k, "")).strip()
        if v:
            name = html.unescape(v)
            break
    return symbol, name


def fetch_candidate_universe(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    max_universe: int,
    extra_symbols: List[str] | None = None,
) -> List[Tuple[str, str]]:
    score: Dict[str, float] = defaultdict(float)
    names: Dict[str, str] = {}

    endpoints = [
        (
            "/uapi/domestic-stock/v1/quotations/volume-rank",
            "FHPST01710000",
            {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "20171",
                "FID_INPUT_ISCD": "0000",
                "FID_DIV_CLS_CODE": "0",
                "FID_BLNG_CLS_CODE": "3",
                "FID_TRGT_CLS_CODE": "111111111",
                "FID_TRGT_EXLS_CLS_CODE": "0000000000",
                "FID_INPUT_PRICE_1": "0",
                "FID_INPUT_PRICE_2": "0",
                "FID_VOL_CNT": "0",
                "FID_INPUT_DATE_1": "0",
            },
            1.0,
        ),
        (
            "/uapi/domestic-stock/v1/ranking/fluctuation",
            "FHPST01700000",
            {
                "fid_cond_mrkt_div_code": "J",
                "fid_cond_scr_div_code": "20170",
                "fid_input_iscd": "0000",
                "fid_rank_sort_cls_code": "0",
                "fid_input_cnt_1": "0",
                "fid_prc_cls_code": "0",
                "fid_input_price_1": "",
                "fid_input_price_2": "",
                "fid_vol_cnt": "",
                "fid_trgt_cls_code": "0",
                "fid_trgt_exls_cls_code": "0",
                "fid_div_cls_code": "0",
                "fid_rsfl_rate1": "",
                "fid_rsfl_rate2": "",
            },
            1.1,
        ),
        (
            "/uapi/domestic-stock/v1/ranking/near-new-highlow",
            "FHPST01870000",
            {
                "fid_aply_rang_vol": "0",
                "fid_cond_mrkt_div_code": "J",
                "fid_cond_scr_div_code": "20187",
                "fid_div_cls_code": "0",
                "fid_input_cnt_1": "0",
                "fid_input_cnt_2": "100",
                "fid_prc_cls_code": "0",
                "fid_input_iscd": "0000",
                "fid_trgt_cls_code": "0",
                "fid_trgt_exls_cls_code": "0",
                "fid_aply_rang_prc_1": "",
                "fid_aply_rang_prc_2": "",
            },
            0.9,
        ),
        (
            "/uapi/domestic-stock/v1/quotations/foreign-institution-total",
            "FHPTJ04400000",
            {
                "FID_COND_MRKT_DIV_CODE": "V",
                "FID_COND_SCR_DIV_CODE": "16449",
                "FID_INPUT_ISCD": "0000",
                "FID_DIV_CLS_CODE": "0",
                "FID_RANK_SORT_CLS_CODE": "0",
                "FID_ETC_CLS_CODE": "0",
            },
            1.2,
        ),
    ]

    for api_url, tr_id, params, weight in endpoints:
        rows = fetch_ranking_rows(base_url, token, app_key, app_secret, api_url, tr_id, params)
        for i, row in enumerate(rows[:100]):
            symbol, name = extract_symbol_name(row)
            if not symbol:
                continue
            names[symbol] = name or names.get(symbol, "")
            score[symbol] += max(0.0, 100.0 - float(i)) * weight

    if extra_symbols:
        for raw in extra_symbols:
            symbol = str(raw).strip().zfill(6)
            if not (symbol.isdigit() and len(symbol) == 6):
                continue
            names[symbol] = names.get(symbol, "") or symbol
            score[symbol] += 1000.0

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)[: max(1, max_universe)]
    return [(sym, names.get(sym, "")) for sym, _ in ranked]


def format_candidate_preview(universe: List[Tuple[str, str]], limit: int = 9999) -> str:
    out: List[str] = []
    for symbol, name in universe[: max(1, int(limit))]:
        out.append(display_name(name, symbol))
    return ", ".join(out)


def send_candidate_list_messages(notifier: "Notifier", universe: List[Tuple[str, str]], chunk_size: int = 25) -> None:
    names = [display_name(name, symbol) for symbol, name in universe]
    if not names:
        return
    total_parts = (len(names) + max(1, int(chunk_size)) - 1) // max(1, int(chunk_size))
    for i in range(total_parts):
        start = i * chunk_size
        end = min(len(names), start + chunk_size)
        part = ", ".join(names[start:end])
        notifier.send(f"후보리스트 {i + 1}/{total_parts} | {part}")


def parse_symbol_csv(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text or "").split(","):
        s = raw.strip()
        if not s:
            continue
        if s.isdigit() and 4 <= len(s) <= 6:
            out.append(s.zfill(6))
    return out


def read_symbols_file(path_text: str) -> List[str]:
    path = Path(path_text).expanduser()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.extend(parse_symbol_csv(s))
    return rows


def fallback_candidate_from_universe(universe: List[Tuple[str, str]]) -> Candidate | None:
    if not universe:
        return None
    symbol, name = universe[0]
    return Candidate(
        symbol=symbol,
        name=name,
        close=0.0,
        ma3=0.0,
        ma5=0.0,
        ma10=0.0,
        ma20=0.0,
        ma60=0.0,
    )


def fetch_minute_ohlcv(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    symbol: str,
    count_hint: int = 180,
    market_code: str = DEFAULT_MINUTE_MARKET_CODE,
) -> List[Dict[str, float]]:
    url = f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
    headers = {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST03010200",
    }
    now = datetime.now(KST)
    ymd = now.strftime("%Y%m%d")
    cursor_time = now.strftime("%H%M%S")
    out_map: Dict[str, Dict[str, float]] = {}

    # KIS returns roughly 30 one-minute rows per request. The NXT window can
    # span about 12 hours, so the old fixed 14-page cap cut off morning data
    # when the cursor was near 20:00.
    max_pages = max(14, min(30, (max(30, int(count_hint)) + 29) // 30 + 2))
    for _ in range(max_pages):
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": (market_code or DEFAULT_MINUTE_MARKET_CODE).strip().upper(),
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": ymd,
            "FID_INPUT_HOUR_1": cursor_time,
            "FID_PW_DATA_INCU_YN": "N",
        }
        try:
            data = request_json_with_retry("get", url, headers=headers, params=params, timeout=15, retries=3)
        except Exception as e:
            print(f"[WARN] minute_ohlcv_skip symbol={symbol} time={cursor_time} err={e}")
            break
        if str(data.get("rt_cd", "0")) not in {"0", ""}:
            print(f"[WARN] minute_ohlcv_rtcd symbol={symbol} time={cursor_time} rt_cd={data.get('rt_cd')} msg={data.get('msg1', '')}")
            break
        rows = data.get("output2", [])
        if not isinstance(rows, list) or not rows:
            break
        min_time = None
        for r in rows:
            d = str(r.get("stck_bsop_date", ""))
            t = str(r.get("stck_cntg_hour", ""))
            if d != ymd or len(t) != 6:
                continue
            key = f"{d}{t}"
            if min_time is None or t < min_time:
                min_time = t
            try:
                out_map[key] = {
                    "date": float(key),
                    "open": float(r.get("stck_oprc", "0")),
                    "high": float(r.get("stck_hgpr", "0")),
                    "low": float(r.get("stck_lwpr", "0")),
                    "close": float(r.get("stck_prpr", "0")),
                    "volume": float(r.get("cntg_vol", "0")),
                }
            except Exception:
                continue
        if not min_time or min_time >= cursor_time:
            break
        cursor_time = min_time
        if len(out_map) >= count_hint * 2:
            break
        time.sleep(0.2)

    rows = [out_map[k] for k in sorted(out_map.keys())]
    return rows[-max(30, count_hint) :]


def sma(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if window <= 0 or n < window:
        return out
    for i in range(window - 1, n):
        chunk = arr[i - window + 1 : i + 1]
        if np.all(np.isfinite(chunk)):
            out[i] = float(np.mean(chunk))
    return out


def ema(arr: np.ndarray, window: int) -> np.ndarray:
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if window <= 0 or n < window:
        return out
    alpha = 2.0 / (float(window) + 1.0)
    out[window - 1] = float(np.mean(arr[:window]))
    for i in range(window, n):
        out[i] = (float(arr[i]) * alpha) + (out[i - 1] * (1.0 - alpha))
    return out


def bollinger(close: np.ndarray, window: int = 20, mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(close, window)
    n = close.shape[0]
    std = np.full(n, np.nan, dtype=float)
    if n >= window:
        for i in range(window - 1, n):
            std[i] = float(np.std(close[i - window + 1 : i + 1]))
    upper = mid + mult * std
    lower = mid - mult * std
    return mid, upper, lower


def rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return out
    diff = np.diff(close, prepend=close[0])
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    avg_gain = sma(gain, period)
    avg_loss = sma(loss, period)
    for i in range(n):
        if not np.isfinite(avg_gain[i]) or not np.isfinite(avg_loss[i]):
            continue
        if avg_loss[i] <= 1e-12:
            out[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def slow_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    n = close.shape[0]
    raw_k = np.full(n, np.nan, dtype=float)
    for i in range(period - 1, n):
        hi = float(np.max(high[i - period + 1 : i + 1]))
        lo = float(np.min(low[i - period + 1 : i + 1]))
        den = hi - lo
        raw_k[i] = 50.0 if den <= 1e-12 else (close[i] - lo) / den * 100.0
    slow_k = sma(raw_k, smooth_k)
    slow_d = sma(slow_k, smooth_d)
    return slow_k, slow_d


def dmi_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    plus_dm = np.zeros(n, dtype=float)
    minus_dm = np.zeros(n, dtype=float)
    tr = np.zeros(n, dtype=float)

    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    atr = sma(tr, period)
    plus = sma(plus_dm, period)
    minus = sma(minus_dm, period)
    plus_di = np.full(n, np.nan, dtype=float)
    minus_di = np.full(n, np.nan, dtype=float)
    dx = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if not np.isfinite(atr[i]) or atr[i] <= 1e-12:
            continue
        p = 100.0 * plus[i] / atr[i]
        m = 100.0 * minus[i] / atr[i]
        plus_di[i] = p
        minus_di[i] = m
        den = p + m
        if den > 1e-12:
            dx[i] = 100.0 * abs(p - m) / den
    adx = sma(dx, period)
    return plus_di, minus_di, adx


def impulse_macd(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ma_len: int = 34,
    signal_len: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    md = np.full(n, np.nan, dtype=float)
    hi_ma = ema(high, ma_len)
    lo_ma = ema(low, ma_len)
    mid_ma = ema((high + low + close) / 3.0, ma_len)

    for i in range(n):
        if not (np.isfinite(hi_ma[i]) and np.isfinite(lo_ma[i]) and np.isfinite(mid_ma[i])):
            continue
        if mid_ma[i] > hi_ma[i]:
            md[i] = mid_ma[i] - hi_ma[i]
        elif mid_ma[i] < lo_ma[i]:
            md[i] = mid_ma[i] - lo_ma[i]
        else:
            md[i] = 0.0

    signal = sma(md, signal_len)
    hist = md - signal
    return md, signal, hist


def crossed_up(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return False
    if not np.isfinite(a[-1]) or not np.isfinite(a[-2]) or not np.isfinite(b[-1]) or not np.isfinite(b[-2]):
        return False
    return a[-2] <= b[-2] and a[-1] > b[-1]


def crossed_down(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape[0] < 2 or b.shape[0] < 2:
        return False
    if not np.isfinite(a[-1]) or not np.isfinite(a[-2]) or not np.isfinite(b[-1]) or not np.isfinite(b[-2]):
        return False
    return a[-2] >= b[-2] and a[-1] < b[-1]


@dataclass
class Candidate:
    symbol: str
    name: str
    close: float
    ma3: float
    ma5: float
    ma10: float
    ma20: float
    ma60: float
    leader_score: float = 0.0
    leader_reason: str = ""
    theme_id: int = 0
    theme_name: str = ""


@dataclass
class PreviousDayStats:
    asof: str
    prev_close: float
    prev_high: float
    prev_low: float
    prev_volume: float
    prev_turnover_bil: float
    prev_ret_pct: float
    avg_volume_5: float
    volume_ratio_5: float


@dataclass
class ThemeGroup:
    theme_id: int
    leader_symbol: str
    leader_name: str
    members: List[Candidate]


def _parse_bar_datetime(v: float) -> datetime:
    s = str(int(v)).zfill(14)
    return datetime.strptime(s, "%Y%m%d%H%M%S")


def _bar_hhmm(row: Dict[str, float]) -> int:
    s = str(int(row["date"])).zfill(14)
    return int(s[8:12])


def resample_bars(rows: List[Dict[str, float]], bar_minutes: int) -> List[Dict[str, float]]:
    if bar_minutes <= 1:
        return rows
    buckets: Dict[datetime, List[Dict[str, float]]] = {}
    for r in rows:
        dt = _parse_bar_datetime(float(r["date"]))
        floor_min = (dt.minute // bar_minutes) * bar_minutes
        key = dt.replace(minute=floor_min, second=0, microsecond=0)
        buckets.setdefault(key, []).append(r)
    out: List[Dict[str, float]] = []
    for key in sorted(buckets.keys()):
        g = buckets[key]
        out.append(
            {
                "date": float(key.strftime("%Y%m%d%H%M%S")),
                "open": float(g[0]["open"]),
                "high": float(max(x["high"] for x in g)),
                "low": float(min(x["low"] for x in g)),
                "close": float(g[-1]["close"]),
                "volume": float(sum(x["volume"] for x in g)),
            }
        )
    return out


def raw_count_hint_for_resampled_bars(target_bars: int, bar_minutes: int) -> int:
    return max(180, int(target_bars) * max(1, int(bar_minutes)) + 30)


def previous_day_stats_from_daily_rows(rows: List[Dict[str, str]], today: datetime) -> PreviousDayStats | None:
    today_date = today.date()
    past: List[Tuple[datetime, Dict[str, str]]] = []
    for row in rows:
        try:
            d = datetime.strptime(str(row.get("date", "")), "%Y-%m-%d")
        except Exception:
            continue
        if d.date() < today_date:
            past.append((d, row))
    if not past:
        return None
    past.sort(key=lambda x: x[0])
    prev_date, prev = past[-1]
    prev2 = past[-2][1] if len(past) >= 2 else None
    prev_close = _to_float(prev.get("close"))
    prev_high = _to_float(prev.get("high"))
    prev_low = _to_float(prev.get("low"))
    prev_volume = _to_float(prev.get("volume"))
    prev2_close = _to_float(prev2.get("close")) if prev2 else 0.0
    prev_ret_pct = ((prev_close / prev2_close) - 1.0) * 100.0 if prev_close > 0 and prev2_close > 0 else 0.0
    recent_volumes = [_to_float(r.get("volume")) for _, r in past[-5:]]
    recent_volumes = [v for v in recent_volumes if v > 0]
    avg_volume_5 = float(np.mean(recent_volumes)) if recent_volumes else 0.0
    volume_ratio_5 = prev_volume / avg_volume_5 if avg_volume_5 > 0 else 0.0
    prev_turnover_bil = (prev_close * prev_volume) / 1_000_000_000.0 if prev_close > 0 and prev_volume > 0 else 0.0
    return PreviousDayStats(
        asof=prev_date.strftime("%Y-%m-%d"),
        prev_close=prev_close,
        prev_high=prev_high,
        prev_low=prev_low,
        prev_volume=prev_volume,
        prev_turnover_bil=prev_turnover_bil,
        prev_ret_pct=prev_ret_pct,
        avg_volume_5=avg_volume_5,
        volume_ratio_5=volume_ratio_5,
    )


def fetch_previous_day_stats(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    universe: List[Tuple[str, str]],
    lookback_days: int,
) -> Dict[str, PreviousDayStats]:
    today = datetime.now(KST)
    start_yyyymmdd = (today - timedelta(days=max(10, int(lookback_days)))).strftime("%Y%m%d")
    end_yyyymmdd = today.strftime("%Y%m%d")
    out: Dict[str, PreviousDayStats] = {}
    for symbol, _ in universe:
        try:
            rows = fetch_daily_prices(
                app_key=app_key,
                app_secret=app_secret,
                symbol=symbol,
                start_yyyymmdd=start_yyyymmdd,
                end_yyyymmdd=end_yyyymmdd,
                access_token=token,
                base_url=base_url,
            )
        except Exception as e:
            print(f"[WARN] prev_day_skip symbol={symbol} err={e}")
            continue
        stats = previous_day_stats_from_daily_rows(rows, today)
        if stats is not None:
            out[symbol] = stats
        time.sleep(0.08)
    return out


def previous_day_leader_bonus(prev: PreviousDayStats | None, current_price: float) -> Tuple[float, str]:
    if prev is None or prev.prev_close <= 0 or current_price <= 0:
        return 0.0, "전일없음"
    today_gap_pct = ((current_price / prev.prev_close) - 1.0) * 100.0
    prev_high_break = prev.prev_high > 0 and current_price >= prev.prev_high
    bonus = 0.0
    if prev.prev_ret_pct > 0:
        bonus += min(22.0, prev.prev_ret_pct * 1.6)
    if prev.prev_turnover_bil > 0:
        bonus += min(24.0, np.log1p(prev.prev_turnover_bil) * 6.0)
    if prev.volume_ratio_5 > 1.0:
        bonus += min(18.0, (prev.volume_ratio_5 - 1.0) * 9.0)
    if today_gap_pct > 0:
        bonus += min(32.0, today_gap_pct * 2.2)
    if prev_high_break:
        bonus += 18.0
    if today_gap_pct < -3.0:
        bonus -= min(18.0, abs(today_gap_pct) * 3.0)
    reason = (
        f"전일({prev.asof}) 등락={prev.prev_ret_pct:.2f}% "
        f"거래대금={prev.prev_turnover_bil:.1f}억 vol5={prev.volume_ratio_5:.2f} "
        f"전일대비={today_gap_pct:.2f}% 전고돌파={prev_high_break} 보너스={bonus:.1f}"
    )
    return float(bonus), reason


def _candidate_from_bars(
    symbol: str,
    name: str,
    bars: List[Dict[str, float]],
    leader_score: float,
    leader_reason: str,
) -> Candidate | None:
    if not bars:
        return None
    close = np.asarray([r["close"] for r in bars], dtype=float)
    ma3 = sma(close, 3)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    last_close = float(close[-1])
    return Candidate(
        symbol=symbol,
        name=name,
        close=last_close,
        ma3=float(ma3[-1]) if np.isfinite(ma3[-1]) else last_close,
        ma5=float(ma5[-1]) if np.isfinite(ma5[-1]) else last_close,
        ma10=float(ma10[-1]) if np.isfinite(ma10[-1]) else last_close,
        ma20=float(ma20[-1]) if np.isfinite(ma20[-1]) else last_close,
        ma60=float(ma60[-1]) if np.isfinite(ma60[-1]) else last_close,
        leader_score=leader_score,
        leader_reason=leader_reason,
    )


def _movement_signature(bars: List[Dict[str, float]], size: int = 24) -> np.ndarray:
    close = np.asarray([r["close"] for r in bars], dtype=float)
    if close.shape[0] < 6:
        return np.zeros(0, dtype=float)
    take = min(int(size), close.shape[0])
    window = close[-take:]
    base = float(window[0]) if float(window[0]) > 1e-9 else 1.0
    normalized = (window / base) - 1.0
    normalized = normalized - float(np.mean(normalized))
    norm = float(np.linalg.norm(normalized))
    if norm <= 1e-9:
        return np.zeros_like(normalized)
    return normalized / norm


def _signature_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    size = min(a.shape[0], b.shape[0])
    if size <= 1:
        return 0.0
    return float(np.dot(a[-size:], b[-size:]))


def _theme_name(theme_id: int, leader_name: str) -> str:
    base_name = leader_name.strip() if leader_name else f"theme{theme_id}"
    return f"테마{theme_id}:{base_name}"


def select_theme_leaders(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    universe: List[Tuple[str, str]],
    bar_minutes: int,
    minute_market_code: str,
    prev_day_stats: Dict[str, PreviousDayStats] | None = None,
    theme_count: int = 2,
    progress_cb: Callable[[int, int, int, str], None] | None = None,
) -> Tuple[List[Candidate], List[ThemeGroup]]:
    analyzed: List[Tuple[Candidate, np.ndarray]] = []
    total = len(universe)
    for idx, (symbol, name) in enumerate(universe, start=1):
        rows = fetch_minute_ohlcv(
            base_url=base_url,
            token=token,
            app_key=app_key,
            app_secret=app_secret,
            symbol=symbol,
            count_hint=raw_count_hint_for_resampled_bars(80, bar_minutes),
            market_code=minute_market_code,
        )
        bars = resample_bars(rows, bar_minutes=bar_minutes)
        if len(bars) < 6:
            continue
        prev_stats = prev_day_stats.get(symbol) if prev_day_stats else None
        leader_score, leader_reason = leader_score_from_minute_bars(bars, prev_stats)
        candidate = _candidate_from_bars(symbol, name, bars, leader_score, leader_reason)
        if candidate is None:
            continue
        signature = _movement_signature(bars)
        analyzed.append((candidate, signature))
        if progress_cb:
            progress_cb(idx, total, len(analyzed), display_name(name, symbol))

    if not analyzed:
        return [], []

    analyzed.sort(key=lambda item: item[0].leader_score, reverse=True)
    max_themes = max(1, int(theme_count))
    anchors: List[Tuple[Candidate, np.ndarray]] = [analyzed[0]]
    for candidate, signature in analyzed[1:]:
        if len(anchors) >= max_themes:
            break
        similarity = max(_signature_similarity(signature, anchor_signature) for _, anchor_signature in anchors)
        if similarity < 0.92:
            anchors.append((candidate, signature))
    for candidate, signature in analyzed[1:]:
        if len(anchors) >= max_themes:
            break
        if any(candidate.symbol == anchor.symbol for anchor, _ in anchors):
            continue
        anchors.append((candidate, signature))

    groups: Dict[int, List[Candidate]] = {i + 1: [anchor] for i, (anchor, _) in enumerate(anchors)}
    for candidate, signature in analyzed:
        if any(candidate.symbol == anchor.symbol for anchor, _ in anchors):
            continue
        best_theme_id = 1
        best_similarity = -9e9
        for idx, (_, anchor_signature) in enumerate(anchors, start=1):
            similarity = _signature_similarity(signature, anchor_signature)
            if similarity > best_similarity:
                best_similarity = similarity
                best_theme_id = idx
        groups.setdefault(best_theme_id, []).append(candidate)

    theme_groups: List[ThemeGroup] = []
    leaders: List[Candidate] = []
    for theme_id in sorted(groups.keys()):
        members = sorted(groups[theme_id], key=lambda item: item.leader_score, reverse=True)
        leader = members[0]
        theme_name = _theme_name(theme_id, leader.name if leader.name else leader.symbol)
        leader.theme_id = theme_id
        leader.theme_name = theme_name
        for member in members:
            member.theme_id = theme_id
            member.theme_name = theme_name
        theme_groups.append(
            ThemeGroup(
                theme_id=theme_id,
                leader_symbol=leader.symbol,
                leader_name=leader.name if leader.name else leader.symbol,
                members=members,
            )
        )
        leaders.append(leader)

    leaders.sort(key=lambda item: item.leader_score, reverse=True)
    return leaders[:max_themes], theme_groups[:max_themes]


def format_theme_group(group: ThemeGroup, limit: int = 5) -> str:
    preview = ", ".join([display_name(c.name, c.symbol) for c in group.members[: max(1, int(limit))]])
    return f"{group.theme_id} | 대장 {display_name(group.leader_name, group.leader_symbol)} | 구성 {preview}"


def extract_limit_up_price(prev: PreviousDayStats | None) -> float:
    if prev is None or prev.prev_close <= 0:
        return 0.0
    return float(prev.prev_close) * 1.30


def fetch_single_previous_day_stat(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    symbol: str,
) -> PreviousDayStats | None:
    stats = fetch_previous_day_stats(
        base_url=base_url,
        token=token,
        app_key=app_key,
        app_secret=app_secret,
        universe=[(symbol, symbol)],
        lookback_days=45,
    )
    return stats.get(symbol)


def fetch_telegram_updates(
    token: str,
    offset: int,
) -> Tuple[List[Dict], int]:
    token = str(token or "").strip()
    if not token:
        return [], offset
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        r = requests.get(url, params={"offset": max(0, int(offset)), "timeout": 0}, timeout=8)
        r.raise_for_status()
        payload = r.json()
        if not payload.get("ok"):
            return [], offset
        rows = payload.get("result", [])
        if not isinstance(rows, list):
            return [], offset
        next_offset = offset
        for row in rows:
            try:
                next_offset = max(next_offset, int(row.get("update_id", 0)) + 1)
            except Exception:
                continue
        return rows, next_offset
    except Exception:
        return [], offset


def resolve_watch_symbol(query: str, known_name_map: Dict[str, str]) -> Tuple[str, str]:
    raw = str(query or "").strip()
    if not raw:
        return "", ""
    compact = re.sub(r"[\s\-_/]", "", raw)
    compact_l = compact.lower()
    if compact.upper().startswith("A") and compact[1:].isdigit():
        compact = compact[1:]
    if compact.isdigit() and 4 <= len(compact) <= 6:
        symbol = compact.zfill(6)
        return symbol, known_name_map.get(symbol, symbol)
    alias_symbol = SYMBOL_ALIAS_MAP.get(compact_l) or SYMBOL_ALIAS_MAP.get(compact)
    if alias_symbol:
        return alias_symbol, known_name_map.get(alias_symbol, alias_symbol)
    normalized = compact
    for symbol, name in known_name_map.items():
        if str(name).replace(" ", "").lower() == normalized.lower():
            return symbol, name
    partial: List[Tuple[str, str]] = []
    for symbol, name in known_name_map.items():
        nm = str(name).replace(" ", "").lower()
        if normalized and normalized.lower() in nm:
            partial.append((symbol, name))
    if len(partial) == 1:
        return partial[0]
    return "", ""


def parse_telegram_watch_command(text: str, known_name_map: Dict[str, str]) -> Tuple[str, List[Tuple[str, str]]]:
    raw = str(text or "").strip()
    if not raw:
        return "ignore", []
    payload = raw

    resolved: List[Tuple[str, str]] = []
    parts = [item.strip() for item in re.split(r"[,\n]+", payload) if item.strip()]
    for part in parts:
        symbol, name = resolve_watch_symbol(part, known_name_map)
        if symbol:
            label = name if (name and not str(name).isdigit()) else part
            resolved.append((symbol, label))
    if resolved:
        return "watch", resolved
    return "ignore", []


def early_momentum_buy_signal_from_minute_bars(
    rows: List[Dict[str, float]],
    *,
    early_end_hhmm: int,
    min_gain_pct: float,
    volume_mult: float,
) -> Tuple[bool, str, float]:
    if len(rows) < 12:
        return False, "warmup", -1.0
    hhmm = _bar_hhmm(rows[-1])
    if hhmm > int(early_end_hhmm):
        return False, "after_early", -1.0

    high = np.asarray([r["high"] for r in rows], dtype=float)
    close = np.asarray([r["close"] for r in rows], dtype=float)
    volume = np.asarray([r["volume"] for r in rows], dtype=float)
    ma3 = sma(close, 3)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    if not all(np.isfinite(v[-1]) for v in (ma3, ma5, ma10)):
        return False, "ma_warmup", -1.0

    open0 = float(rows[0]["open"])
    gain = (float(close[-1]) / open0 - 1.0) if open0 > 1e-9 else 0.0
    trend_stack = ma3[-1] >= ma5[-1] >= ma10[-1] * 0.997 and close[-1] >= ma5[-1]

    v = np.maximum(volume, 0.0)
    vol_sum = float(np.sum(v))
    typical = (np.asarray([r["high"] for r in rows], dtype=float) + np.asarray([r["low"] for r in rows], dtype=float) + close) / 3.0
    vwap = float(np.sum(typical * v) / vol_sum) if vol_sum > 1e-9 else float(np.mean(close))
    vwap_ok = float(close[-1]) >= vwap * 0.998

    lookback = min(8, len(rows) - 1)
    recent_high = float(np.max(high[-lookback - 1 : -1]))
    breakout_ok = float(close[-1]) >= recent_high * 0.998

    vol_window = min(20, len(rows) - 1)
    vol_base = float(np.mean(volume[-vol_window - 1 : -1])) if vol_window >= 3 else float(np.mean(volume[:-1]))
    vol_ratio = (float(volume[-1]) / vol_base) if vol_base > 1e-9 else 0.0
    vol_cluster = float(np.sum(volume[-3:])) / max(1e-9, float(np.mean(volume[:-3])) * 3.0) if len(rows) > 15 else vol_ratio
    volume_ok = vol_ratio >= max(1.0, float(volume_mult)) or vol_cluster >= max(1.2, float(volume_mult) * 0.85)

    ok = gain >= float(min_gain_pct) and trend_stack and vwap_ok and (breakout_ok or volume_ok)
    reason = (
        f"early gain={gain*100:.2f}% trend={trend_stack} vwap={vwap_ok} "
        f"breakout={breakout_ok} vol={volume_ok} vol_ratio={vol_ratio:.2f}"
    )
    if not ok:
        return False, reason, -1.0
    score = 120.0 + (gain * 1000.0) + (5.0 * vol_ratio) + (20.0 if breakout_ok else 0.0)
    return True, reason, score


def leader_score_from_minute_bars(rows: List[Dict[str, float]], prev: PreviousDayStats | None = None) -> Tuple[float, str]:
    if len(rows) < 3:
        return 0.0, "leader_warmup"
    high = np.asarray([r["high"] for r in rows], dtype=float)
    low = np.asarray([r["low"] for r in rows], dtype=float)
    close = np.asarray([r["close"] for r in rows], dtype=float)
    volume = np.maximum(np.asarray([r["volume"] for r in rows], dtype=float), 0.0)

    open0 = float(rows[0]["open"])
    cur = float(close[-1])
    day_high = float(np.max(high))
    ret_pct = ((cur / open0) - 1.0) * 100.0 if open0 > 1e-9 else 0.0
    high_ret_pct = ((day_high / open0) - 1.0) * 100.0 if open0 > 1e-9 else 0.0
    high_keep_pct = (cur / day_high) * 100.0 if day_high > 1e-9 else 0.0

    typical = (high + low + close) / 3.0
    turnover_bil = float(np.sum(typical * volume) / 1_000_000_000.0)
    ma3 = sma(close, 3)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    trend_stack = bool(
        np.isfinite(ma3[-1])
        and np.isfinite(ma5[-1])
        and np.isfinite(ma10[-1])
        and ma3[-1] >= ma5[-1] >= ma10[-1] * 0.997
        and cur >= ma5[-1]
    )

    if len(rows) > 20:
        vol_base = float(np.mean(volume[:-3]))
        vol_cluster = float(np.sum(volume[-3:])) / max(1e-9, vol_base * 3.0)
    else:
        vol_cluster = 1.0
    score = (
        (ret_pct * 8.0)
        + (high_ret_pct * 3.0)
        + (min(80.0, np.log1p(max(0.0, turnover_bil)) * 18.0))
        + (min(30.0, vol_cluster * 6.0))
        + (25.0 if high_keep_pct >= 97.0 else 0.0)
        + (15.0 if trend_stack else 0.0)
    )
    prev_bonus, prev_reason = previous_day_leader_bonus(prev, cur)
    score += prev_bonus
    reason = (
        f"등락={ret_pct:.2f}% 고점={high_ret_pct:.2f}% 고점유지={high_keep_pct:.1f}% "
        f"거래대금={turnover_bil:.1f}억 vol_cluster={vol_cluster:.2f} trend={trend_stack} | {prev_reason}"
    )
    return float(score), reason


def minute_filter(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    universe: List[Tuple[str, str]],
    ma_converge_pct: float,
    ma60_no_break_days: int,
    ma20_support_days: int,
    bar_minutes: int,
    minute_market_code: str,
    early_momentum_end_hhmm: int,
    early_momentum_min_gain_pct: float,
    early_momentum_volume_mult: float,
    leader_only: bool,
    leader_max_symbols: int,
    prev_day_stats: Dict[str, PreviousDayStats] | None = None,
    progress_cb: Callable[[int, int, int, str], None] | None = None,
) -> Tuple[List[Candidate], Candidate | None]:
    selected: List[Candidate] = []
    nearest: Candidate | None = None
    nearest_score = float("inf")
    total = len(universe)
    for idx, (symbol, name) in enumerate(universe, start=1):
        rows = fetch_minute_ohlcv(
            base_url=base_url,
            token=token,
            app_key=app_key,
            app_secret=app_secret,
            symbol=symbol,
            count_hint=raw_count_hint_for_resampled_bars(90, bar_minutes),
            market_code=minute_market_code,
        )
        bars = resample_bars(rows, bar_minutes=bar_minutes)
        prev_stats = prev_day_stats.get(symbol) if prev_day_stats else None
        leader_score, leader_reason = leader_score_from_minute_bars(bars, prev_stats)
        early_ok, _, _ = early_momentum_buy_signal_from_minute_bars(
            bars,
            early_end_hhmm=early_momentum_end_hhmm,
            min_gain_pct=early_momentum_min_gain_pct,
            volume_mult=early_momentum_volume_mult,
        )
        if len(bars) < 70:
            if early_ok and bars:
                close = np.asarray([r["close"] for r in bars], dtype=float)
                selected.append(
                    Candidate(
                        symbol=symbol,
                        name=name,
                        close=float(close[-1]),
                        ma3=float(close[-1]),
                        ma5=float(close[-1]),
                        ma10=float(close[-1]),
                        ma20=float(close[-1]),
                        ma60=float(close[-1]),
                        leader_score=leader_score,
                        leader_reason=leader_reason,
                    )
                )
                if progress_cb:
                    progress_cb(idx, total, len(selected), display_name(name, symbol))
            continue
        close = np.asarray([r["close"] for r in bars], dtype=float)
        low = np.asarray([r["low"] for r in bars], dtype=float)
        ma3 = sma(close, 3)
        ma5 = sma(close, 5)
        ma10 = sma(close, 10)
        ma20 = sma(close, 20)
        ma60 = sma(close, 60)
        if not all(np.isfinite(v[-1]) for v in (ma3, ma5, ma10, ma20, ma60)):
            continue

        ma_values = [ma3[-1], ma5[-1], ma10[-1]]
        convergence = (max(ma_values) - min(ma_values)) / max(1e-12, close[-1])
        converge_gap = max(0.0, convergence - float(ma_converge_pct))
        ma60_gap = max(0.0, (ma60[-1] - close[-1]) / max(1e-12, close[-1]))
        pass_converge = convergence <= ma_converge_pct
        pass_ma60 = close[-1] >= ma60[-1]

        keep_60 = True
        break_60_cnt = 0
        for i in range(1, max(1, ma60_no_break_days) + 1):
            if i >= close.shape[0]:
                break
            if not np.isfinite(ma60[-i]) or low[-i] < ma60[-i]:
                keep_60 = False
                break_60_cnt += 1

        keep_20 = True
        break_20_cnt = 0
        for i in range(1, max(1, ma20_support_days) + 1):
            if i >= close.shape[0]:
                break
            if not np.isfinite(ma20[-i]) or close[-i] < ma20[-i]:
                keep_20 = False
                break_20_cnt += 1

        candidate = Candidate(
            symbol=symbol,
            name=name,
            close=float(close[-1]),
            ma3=float(ma3[-1]),
            ma5=float(ma5[-1]),
            ma10=float(ma10[-1]),
            ma20=float(ma20[-1]),
            ma60=float(ma60[-1]),
            leader_score=leader_score,
            leader_reason=leader_reason,
        )

        # Lower score means "closer" to passing all daily/minute filters.
        near_score = (converge_gap * 1000.0) + (ma60_gap * 1000.0) + (float(break_60_cnt) * 10.0) + (float(break_20_cnt) * 5.0)
        if near_score < nearest_score:
            nearest = candidate
            nearest_score = near_score

        bullish_stack = ma3[-1] >= ma5[-1] >= ma10[-1] and close[-1] >= ma20[-1] and close[-1] >= ma60[-1]
        aggressive_trend = bullish_stack and keep_60 and close[-1] >= ma20[-1]

        if not (((pass_converge and pass_ma60 and keep_60 and keep_20) or aggressive_trend) or early_ok):
            continue

        selected.append(candidate)
        if progress_cb:
            progress_cb(idx, total, len(selected), display_name(name, symbol))
    if selected:
        selected.sort(key=lambda c: c.leader_score, reverse=True)
        if leader_only:
            selected = selected[: max(1, int(leader_max_symbols))]
    return selected, nearest


class Notifier:
    def __init__(self, log_path: Path, telegram_token: str, telegram_chat_id: str, message_prefix: str = ""):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.telegram_token = telegram_token.strip()
        self.telegram_chat_id = telegram_chat_id.strip()
        self.message_prefix = message_prefix.strip()

    def send(self, text: str) -> None:
        ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
        body = f"[{self.message_prefix}] {text}" if self.message_prefix else text
        line = f"[{ts}] {body}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.telegram_token and self.telegram_chat_id:
            sent = False
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                r = requests.post(url, json={"chat_id": self.telegram_chat_id, "text": line}, timeout=8)
                sent = bool(getattr(r, "ok", False))
            except Exception:
                sent = False
            if not sent:
                try:
                    url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                    subprocess.run(
                        [
                            "curl.exe" if os.name == "nt" else "curl",
                            "-sS",
                            "-X",
                            "POST",
                            url,
                            "-d",
                            f"chat_id={self.telegram_chat_id}",
                            "--data-urlencode",
                            f"text={line}",
                        ],
                        check=False,
                        timeout=8,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Custom timing bot (user rule only)")
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", PROD_BASE_URL))
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--cano", default=os.getenv("KIS_CANO", ""))
    p.add_argument("--acnt-prdt-cd", default=os.getenv("KIS_ACNT_PRDT_CD", "01"))
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--max-universe", type=int, default=10)
    p.add_argument("--scan-interval-sec", type=int, default=60)
    p.add_argument("--max-cycles", type=int, default=200)
    p.add_argument("--bar-minutes", type=int, choices=[1, 3, 5], default=3)
    p.add_argument("--minute-market-code", default=os.getenv("KIS_MINUTE_MARKET_CODE", DEFAULT_MINUTE_MARKET_CODE), help="minute chart market code: UN includes NXT pre/after hours")
    p.add_argument("--refresh-start-hhmm", type=int, default=DEFAULT_SEARCH_START_HHMM)
    p.add_argument("--refresh-end-hhmm", type=int, default=DEFAULT_TRADING_CLOSE_HHMM)
    p.add_argument("--refresh-interval-min", type=int, default=120)
    p.add_argument("--empty-refresh-interval-min", type=int, default=120, help="refresh interval when no symbol passed filters")
    p.add_argument("--early-refresh-end-hhmm", type=int, default=1030, help="use faster refresh until this time for early theme leaders")
    p.add_argument("--early-refresh-interval-min", type=int, default=3, help="refresh interval during early momentum window")
    p.add_argument("--premarket-refresh-interval-min", type=int, default=30, help="theme/candidate refresh interval before market-open trading starts")
    p.add_argument("--market-open-hhmm", type=int, default=DEFAULT_TRADING_OPEN_HHMM)
    p.add_argument("--market-close-hhmm", type=int, default=DEFAULT_TRADING_CLOSE_HHMM)
    p.add_argument("--watch-report-interval-min", type=int, default=10, help="tracking report interval while watching selected symbols")
    p.add_argument("--max-watch-candidates", type=int, default=4, help="maximum symbols to monitor at once")
    p.add_argument("--initial-cash", type=float, default=10_000_000, help="account seed cash for position sizing")
    p.add_argument("--position-size-pct", type=float, default=0.50, help="max position size per symbol (0~1)")
    p.add_argument("--fallback-position-size-pct", type=float, default=0.10, help="position size for nearest fallback symbol (0~1)")
    p.add_argument("--max-buys-per-scan", type=int, default=2, help="max buy orders per scan cycle")
    p.add_argument("--max-positions", type=int, default=2, help="max concurrent holding symbols")
    p.add_argument("--theme-count", type=int, default=2, help="number of theme groups to keep")
    p.add_argument("--disable-leader-only", action="store_true", default=False, help="allow non-leader selected symbols too")
    p.add_argument("--leader-max-symbols", type=int, default=1, help="number of theme leaders to keep when leader-only is enabled")
    p.add_argument("--disable-prev-day-score", action="store_true", default=False, help="do not add previous-day strength to leader scoring")
    p.add_argument("--prev-day-lookback-days", type=int, default=45, help="calendar days to fetch for previous-day strength")
    p.add_argument("--prev-day-max-symbols", type=int, default=30, help="max symbols per refresh for previous-day daily data fetch")
    p.add_argument("--order-krw", type=float, default=0.0, help="optional fixed order amount; 0 uses initial-cash * position-size-pct")
    p.add_argument("--ma-converge-pct", type=float, default=0.025)
    p.add_argument("--ma60-no-break-days", type=int, default=5, help="minute bars count for no-break check")
    p.add_argument("--ma20-support-days", type=int, default=3, help="minute bars count for MA20 support check")
    p.add_argument("--adx-min", type=float, default=16.0)
    p.add_argument("--disable-breakout-entry", action="store_true", default=False, help="disable breakout fallback entry")
    p.add_argument("--breakout-lookback-bars", type=int, default=8, help="bars for recent high breakout check")
    p.add_argument("--breakout-buffer-pct", type=float, default=0.0, help="breakout buffer over recent high")
    p.add_argument("--breakout-volume-window", type=int, default=20, help="bars for average volume baseline")
    p.add_argument("--breakout-volume-mult", type=float, default=1.4, help="required volume spike multiplier")
    p.add_argument("--breakout-adx-min", type=float, default=12.0, help="minimum adx for breakout entry")
    p.add_argument("--early-momentum-end-hhmm", type=int, default=1030, help="allow early theme-leader momentum entries until this time")
    p.add_argument("--early-momentum-min-gain-pct", type=float, default=0.015, help="minimum gain from first bar for early momentum entry")
    p.add_argument("--early-momentum-volume-mult", type=float, default=1.6, help="volume spike multiplier for early momentum entry")
    p.add_argument("--sell-stop-loss-pct", type=float, default=0.015, help="hard stop loss from entry (e.g. 0.015=1.5%)")
    p.add_argument("--sell-trailing-stop-pct", type=float, default=0.02, help="trailing stop from peak (e.g. 0.02=2%)")
    p.add_argument("--sell-trailing-activate-pct", type=float, default=0.012, help="arm trailing stop after this gain from entry")
    p.add_argument("--sell-max-hold-min", type=int, default=180, help="time-based exit after holding minutes, 0 disables")
    p.add_argument("--log-file", default="logs/user_only_strategy_signals.txt")
    p.add_argument("--telegram-bot-token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""))
    p.add_argument("--telegram-chat-id", default=os.getenv("TELEGRAM_CHAT_ID", ""))
    p.add_argument("--telegram-command-poll-sec", type=int, default=15, help="telegram command polling interval")
    p.add_argument("--extra-symbols", default=os.getenv("NEXT_MARKET_SYMBOLS", ""), help="comma-separated extra symbols")
    p.add_argument("--extra-symbols-file", default=os.getenv("NEXT_MARKET_SYMBOLS_FILE", ""), help="extra symbols file path")
    return p.parse_args()


def watch_preview(candidates: List["Candidate"], max_items: int = 12) -> str:
    if not candidates:
        return "-"
    names = []
    for c in candidates[:max_items]:
        names.append(display_name(c.name, c.symbol))
    return ", ".join(names)


def display_name(name: str, symbol: str = "") -> str:
    n = str(name or "").strip()
    if n and not n.isdigit():
        return n
    return "종목"


def in_korean_trading_session(now: datetime, market_open_hhmm: int, market_close_hhmm: int) -> bool:
    hhmm = now.hour * 100 + now.minute
    # NXT extended window for this strategy: open is inclusive, close is exclusive.
    return now.weekday() < 5 and int(market_open_hhmm) <= hhmm < int(market_close_hhmm)


def in_korean_regular_session(now: datetime, market_open_hhmm: int, market_close_hhmm: int) -> bool:
    return in_korean_trading_session(now, market_open_hhmm, market_close_hhmm)


def in_refresh_window(now: datetime, start_hhmm: int, end_hhmm: int) -> bool:
    hhmm = now.hour * 100 + now.minute
    return now.weekday() < 5 and int(start_hhmm) <= hhmm < int(end_hhmm)


def is_network_block_error(exc: Exception) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return (
        "winerror 10013" in s
        or "failed to establish a new connection" in s
        or "max retries exceeded" in s
        or "connection refused" in s
        or "connection reset" in s
        or "timed out" in s
    )


def buy_signal_from_minute_bars(rows: List[Dict[str, float]], adx_min: float) -> Tuple[bool, str, float]:
    if len(rows) < 70:
        return False, "warmup", -1.0
    high = np.asarray([r["high"] for r in rows], dtype=float)
    low = np.asarray([r["low"] for r in rows], dtype=float)
    close = np.asarray([r["close"] for r in rows], dtype=float)
    st_k, st_d = slow_stochastic(high, low, close, period=14, smooth_k=3, smooth_d=3)
    plus_di, minus_di, adx = dmi_adx(high, low, close, period=14)
    rsi14 = rsi(close, period=14)
    mid, _, lower = bollinger(close, window=20, mult=2.0)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    imp, imp_signal, imp_hist = impulse_macd(high, low, close)

    stoch_cross = crossed_up(st_k, st_d)
    stoch_up = (
        stoch_cross
        or (
            np.isfinite(st_k[-1])
            and np.isfinite(st_k[-2])
            and np.isfinite(st_d[-1])
            and st_k[-1] > st_d[-1]
            and st_k[-1] >= st_k[-2]
        )
    )
    dmi_cross = crossed_up(plus_di, minus_di)
    dmi_up = (
        dmi_cross
        or (
            np.isfinite(plus_di[-1])
            and np.isfinite(plus_di[-2])
            and np.isfinite(minus_di[-1])
            and plus_di[-1] > minus_di[-1]
            and plus_di[-1] >= plus_di[-2]
        )
    )
    adx_ok = (
        np.isfinite(adx[-1])
        and np.isfinite(adx[-2])
        and (adx[-1] >= adx_min or (adx[-1] > adx[-2] and dmi_up))
    )
    trend_ok = (
        np.isfinite(ma20[-1])
        and np.isfinite(ma60[-1])
        and close[-1] >= ma20[-1]
        and ma20[-1] >= ma60[-1] * 0.99
    )
    rsi_boll_ok = False
    if np.isfinite(rsi14[-1]) and np.isfinite(rsi14[-2]) and np.isfinite(mid[-1]) and np.isfinite(lower[-1]):
        rsi_boll_ok = (
            (rsi14[-2] < 40 and rsi14[-1] > rsi14[-2])
            or (close[-2] < lower[-2] and close[-1] > lower[-1])
            or (rsi14[-1] > 50 and close[-1] > mid[-1])
            or (rsi14[-1] >= 45 and rsi14[-1] > rsi14[-2] and close[-1] >= mid[-1])
        )

    impulse_ok = (
        np.isfinite(imp[-1])
        and np.isfinite(imp_signal[-1])
        and np.isfinite(imp_hist[-1])
        and np.isfinite(imp_hist[-2])
        and imp[-1] > imp_signal[-1]
        and imp_hist[-1] > 0
        and imp_hist[-1] >= imp_hist[-2]
    )

    classic_trigger = stoch_up and rsi_boll_ok
    impulse_trigger = impulse_ok and (stoch_up or rsi_boll_ok)
    ok = dmi_up and adx_ok and trend_ok and (classic_trigger or impulse_trigger)
    reason = f"stoch_up={stoch_up} dmi_up={dmi_up} adx={adx_ok} trend={trend_ok} rsi_boll={rsi_boll_ok} impulse={impulse_ok}"
    if not ok:
        return ok, reason, -1.0
    adx_v = float(adx[-1]) if np.isfinite(adx[-1]) else 0.0
    dmi_gap = float(plus_di[-1] - minus_di[-1]) if np.isfinite(plus_di[-1]) and np.isfinite(minus_di[-1]) else 0.0
    stoch_gap = float(st_k[-1] - st_d[-1]) if np.isfinite(st_k[-1]) and np.isfinite(st_d[-1]) else 0.0
    impulse_gap = float(imp[-1] - imp_signal[-1]) if np.isfinite(imp[-1]) and np.isfinite(imp_signal[-1]) else 0.0
    score = adx_v + (0.5 * dmi_gap) + (0.2 * stoch_gap) + (0.2 * impulse_gap) + (8.0 if impulse_ok else 0.0)
    return ok, reason, score


def breakout_buy_signal_from_minute_bars(
    rows: List[Dict[str, float]],
    lookback_bars: int,
    breakout_buffer_pct: float,
    volume_window: int,
    volume_mult: float,
    adx_min: float,
) -> Tuple[bool, str, float]:
    min_need = max(40, lookback_bars + 5, volume_window + 5)
    if len(rows) < min_need:
        return False, "warmup", -1.0
    high = np.asarray([r["high"] for r in rows], dtype=float)
    close = np.asarray([r["close"] for r in rows], dtype=float)
    volume = np.asarray([r["volume"] for r in rows], dtype=float)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    plus_di, minus_di, adx = dmi_adx(high, np.asarray([r["low"] for r in rows], dtype=float), close, period=14)
    imp, imp_signal, imp_hist = impulse_macd(high, np.asarray([r["low"] for r in rows], dtype=float), close)

    if lookback_bars < 2:
        lookback_bars = 2
    recent_high = float(np.max(high[-lookback_bars - 1 : -1]))
    breakout_px = recent_high * (1.0 + max(0.0, breakout_buffer_pct))
    breakout_ok = np.isfinite(close[-1]) and close[-1] > breakout_px

    vol_base = float(np.mean(volume[-volume_window - 1 : -1])) if volume_window >= 2 else float(np.mean(volume[:-1]))
    vol_ratio = (float(volume[-1]) / vol_base) if vol_base > 1e-9 else 0.0
    volume_ok = vol_ratio >= max(1.0, volume_mult)

    trend_ok = (
        np.isfinite(ma20[-1])
        and np.isfinite(ma60[-1])
        and close[-1] > ma20[-1]
        and ma20[-1] >= ma60[-1] * 0.995
    )
    adx_ok = np.isfinite(adx[-1]) and adx[-1] >= adx_min
    dmi_ok = (
        np.isfinite(plus_di[-1])
        and np.isfinite(plus_di[-2])
        and np.isfinite(minus_di[-1])
        and plus_di[-1] > minus_di[-1]
        and plus_di[-1] >= plus_di[-2] * 0.98
    )
    impulse_ok = (
        np.isfinite(imp[-1])
        and np.isfinite(imp_signal[-1])
        and np.isfinite(imp_hist[-1])
        and np.isfinite(imp_hist[-2])
        and imp[-1] > 0
        and imp[-1] > imp_signal[-1]
        and imp_hist[-1] >= imp_hist[-2]
    )

    ok = breakout_ok and volume_ok and trend_ok and dmi_ok and (adx_ok or impulse_ok)
    reason = (
        f"breakout={breakout_ok} vol={volume_ok} trend={trend_ok} "
        f"adx={adx_ok} dmi={dmi_ok} impulse={impulse_ok} vol_ratio={vol_ratio:.2f}"
    )
    if not ok:
        return False, reason, -1.0

    adx_v = float(adx[-1]) if np.isfinite(adx[-1]) else 0.0
    breakout_strength = ((close[-1] / breakout_px) - 1.0) * 100.0 if breakout_px > 1e-9 else 0.0
    impulse_gap = float(imp[-1] - imp_signal[-1]) if np.isfinite(imp[-1]) and np.isfinite(imp_signal[-1]) else 0.0
    score = 80.0 + (12.0 * breakout_strength) + (4.0 * vol_ratio) + (0.5 * adx_v) + (0.05 * impulse_gap)
    return True, reason, score


def sell_signal_from_minute_bars(rows: List[Dict[str, float]]) -> Tuple[bool, str]:
    if len(rows) < 70:
        return False, "warmup"
    high = np.asarray([r["high"] for r in rows], dtype=float)
    low = np.asarray([r["low"] for r in rows], dtype=float)
    close = np.asarray([r["close"] for r in rows], dtype=float)
    st_k, st_d = slow_stochastic(high, low, close, period=14, smooth_k=3, smooth_d=3)
    plus_di, minus_di, adx = dmi_adx(high, low, close, period=14)
    rsi14 = rsi(close, period=14)
    mid, _, _ = bollinger(close, window=20, mult=2.0)
    ma20 = sma(close, 20)
    imp, imp_signal, imp_hist = impulse_macd(high, low, close)

    stoch_dead = (
        crossed_down(st_k, st_d)
        or (
            np.isfinite(st_k[-1])
            and np.isfinite(st_k[-2])
            and np.isfinite(st_d[-1])
            and st_k[-1] < st_d[-1]
            and st_k[-1] < st_k[-2]
            and st_k[-2] >= 65
        )
    )
    dmi_dead = (
        crossed_down(plus_di, minus_di)
        or (
            np.isfinite(plus_di[-1])
            and np.isfinite(minus_di[-1])
            and minus_di[-1] > plus_di[-1]
        )
    )
    impulse_dead = (
        crossed_down(imp, imp_signal)
        or (
            np.isfinite(imp_hist[-1])
            and np.isfinite(imp_hist[-2])
            and imp_hist[-1] < 0
            and imp_hist[-1] < imp_hist[-2]
        )
    )
    rsi_mid_break = (
        np.isfinite(rsi14[-1]) and np.isfinite(mid[-1]) and close[-1] < mid[-1] and rsi14[-1] < 50
    )
    ma20_break = np.isfinite(ma20[-1]) and close[-1] < ma20[-1]
    ok = (impulse_dead and (stoch_dead or dmi_dead)) or (ma20_break and (rsi_mid_break or impulse_dead))
    reason = f"stoch_dead={stoch_dead} dmi_dead={dmi_dead} impulse_dead={impulse_dead} rsi_mid_break={rsi_mid_break} ma20_break={ma20_break}"
    return ok, reason


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET required")
    if not args.dry_run and (not args.cano or not args.acnt_prdt_cd):
        raise RuntimeError("KIS_CANO and KIS_ACNT_PRDT_CD required for live order")

    notifier = Notifier(
        Path(args.log_file),
        args.telegram_bot_token,
        args.telegram_chat_id,
        message_prefix=f"{int(args.bar_minutes)}분봉",
    )
    extra_symbols = parse_symbol_csv(args.extra_symbols)
    if args.extra_symbols_file:
        extra_symbols.extend(read_symbols_file(args.extra_symbols_file))
    if extra_symbols:
        extra_symbols = sorted(set(extra_symbols))
        notifier.send(f"추가후보 {len(extra_symbols)}개 반영")
    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
    watch_candidates: List[Candidate] = []
    strict_filtered_count = 0
    blocked_unbuyable_symbols: set[str] = set()
    last_refresh: datetime | None = None
    last_watch_report: datetime | None = None
    active_session_day = None
    close_notified_day = None
    positions: Dict[str, int] = {}
    entry_price: Dict[str, float] = {}
    peak_price: Dict[str, float] = {}
    entry_time: Dict[str, datetime] = {}
    known_name_map: Dict[str, str] = {}
    manual_watch_symbols: set[str] = set()
    bought_symbols_today: set[str] = set()
    limit_up_hold_day: Dict[str, datetime.date] = {}
    theme_selection_day = None
    daily_trade_finished_day = None
    telegram_update_offset = 0
    last_telegram_poll_at: datetime | None = None
    last_slots_full_notice_at: datetime | None = None
    signal_first_seen_at: Dict[str, datetime] = {}
    net_err_streak = 0
    last_net_alert_at: datetime | None = None

    def notify_net_error(exc: Exception) -> int:
        nonlocal net_err_streak, last_net_alert_at
        net_err_streak += 1
        wait_sec = min(300, max(10, net_err_streak * 10))
        now_local = datetime.now(KST)
        if last_net_alert_at is None or (now_local - last_net_alert_at).total_seconds() >= 30:
            short = str(exc).replace("\n", " ")[:180]
            notifier.send(f"네트워크오류 감지: {type(exc).__name__} | {short}")
            notifier.send(f"자동재시도 대기 {wait_sec}초")
            last_net_alert_at = now_local
        return wait_sec

    def apply_blocked_universe(universe: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        if not blocked_unbuyable_symbols:
            return universe
        return [(s, n) for s, n in universe if s not in blocked_unbuyable_symbols]

    def is_daily_trade_finished(day_value: datetime.date) -> bool:
        return daily_trade_finished_day == day_value

    def finish_trading_day(reason: str) -> None:
        nonlocal strict_filtered_count, last_refresh, last_watch_report, daily_trade_finished_day, theme_selection_day
        watch_candidates.clear()
        strict_filtered_count = 0
        last_refresh = datetime.now(KST)
        last_watch_report = None
        theme_selection_day = datetime.now(KST).date()
        daily_trade_finished_day = datetime.now(KST).date()
        notifier.send(f"당일마감 | {reason}")

    def set_position_entry(symbol: str, qty: int, price: float) -> None:
        positions[symbol] = max(0, int(qty))
        entry_price[symbol] = float(price)
        peak_price[symbol] = float(price)
        entry_time[symbol] = datetime.now(KST)
        bought_symbols_today.add(symbol)
        signal_first_seen_at.pop(symbol, None)

    def clear_position_state(symbol: str) -> None:
        positions[symbol] = 0
        signal_first_seen_at.pop(symbol, None)
        entry_price.pop(symbol, None)
        peak_price.pop(symbol, None)
        entry_time.pop(symbol, None)
        limit_up_hold_day.pop(symbol, None)

    def finish_if_all_closed(carryover_exit: bool = False) -> None:
        if sum(1 for q in positions.values() if q > 0) == 0 and (
            len(bought_symbols_today) >= max(1, int(args.theme_count)) or carryover_exit
        ):
            finish_trading_day("보유 종목 청산 완료")

    def poll_telegram_commands(now_local: datetime) -> None:
        nonlocal telegram_update_offset, last_telegram_poll_at, last_refresh
        if not args.telegram_bot_token or not args.telegram_chat_id:
            return
        if last_telegram_poll_at is not None:
            elapsed_sec = (now_local - last_telegram_poll_at).total_seconds()
            if elapsed_sec < max(5, int(args.telegram_command_poll_sec)):
                return
        updates, next_offset = fetch_telegram_updates(args.telegram_bot_token, telegram_update_offset)
        telegram_update_offset = next_offset
        last_telegram_poll_at = now_local
        for row in updates:
            message = row.get("message") or row.get("edited_message") or {}
            if not isinstance(message, dict):
                continue
            chat = message.get("chat") or {}
            if str(chat.get("id", "")).strip() != str(args.telegram_chat_id).strip():
                continue
            text = str(message.get("text", "")).strip()
            if not text:
                continue
            action, payload = parse_telegram_watch_command(text, known_name_map)
            if action == "ignore":
                continue
            if action == "watch":
                added_names: List[str] = []
                for symbol, name in payload:
                    manual_watch_symbols.add(symbol)
                    known_name_map[symbol] = name if name else symbol
                    added_names.append(display_name(name, symbol))
                    if all(candidate.symbol != symbol for candidate in watch_candidates):
                        if len(watch_candidates) >= max(1, int(args.max_watch_candidates)):
                            notifier.send(f"수동감시 추가대기 | 감시최대 {int(args.max_watch_candidates)}개 도달")
                            continue
                        # Keep manual symbols in watch list even off-session, so user can verify immediately.
                        candidate = Candidate(
                            symbol=symbol,
                            name=name if name else symbol,
                            close=0.0,
                            ma3=0.0,
                            ma5=0.0,
                            ma10=0.0,
                            ma20=0.0,
                            ma60=0.0,
                        )
                        candidate.theme_id = 99
                        candidate.theme_name = "수동감시"
                        watch_candidates.append(candidate)
                    if (
                        in_korean_trading_session(now_local, args.market_open_hhmm, args.market_close_hhmm)
                        and not is_daily_trade_finished(now_local.date())
                    ):
                        try:
                            prev = fetch_single_previous_day_stat(args.base_url, token, args.app_key, args.app_secret, symbol)
                            rows = fetch_minute_ohlcv(
                                base_url=args.base_url,
                                token=token,
                                app_key=args.app_key,
                                app_secret=args.app_secret,
                                symbol=symbol,
                                count_hint=raw_count_hint_for_resampled_bars(80, args.bar_minutes),
                                market_code=args.minute_market_code,
                            )
                            bars = resample_bars(rows, bar_minutes=args.bar_minutes)
                            leader_score, leader_reason = leader_score_from_minute_bars(bars, prev) if bars else (0.0, "manual")
                            enriched = _candidate_from_bars(symbol, known_name_map.get(symbol, symbol), bars, leader_score, leader_reason)
                            if enriched is not None:
                                enriched.theme_id = 99
                                enriched.theme_name = "수동감시"
                                for idx, existing in enumerate(watch_candidates):
                                    if existing.symbol == symbol:
                                        watch_candidates[idx] = enriched
                                        break
                        except Exception as exc:
                            notifier.send(f"수동감시 즉시추가 실패 | {display_name(name, symbol)} | {type(exc).__name__}")
                if added_names:
                    current_watch = watch_preview(watch_candidates) if watch_candidates else "-"
                    notifier.send(f"현재 모니터링종목 | {current_watch}")
                    if not watch_candidates and not is_daily_trade_finished(now_local.date()):
                        last_refresh = None

    boot_now = datetime.now(KST)
    boot_hhmm = boot_now.hour * 100 + boot_now.minute
    first_run_at_window_open = boot_hhmm < int(args.refresh_start_hhmm)
    suppress_initial_refresh_once = not first_run_at_window_open

    for cycle in range(max(1, args.max_cycles)):
        now = datetime.now(KST)
        hhmm = now.hour * 100 + now.minute
        if active_session_day != now.date():
            active_session_day = now.date()
            close_notified_day = None
            theme_selection_day = None if not any(q > 0 for q in positions.values()) else theme_selection_day
            daily_trade_finished_day = None if daily_trade_finished_day != now.date() else daily_trade_finished_day
            bought_symbols_today.clear()
            blocked_unbuyable_symbols.clear()
            if not any(q > 0 for q in positions.values()):
                watch_candidates.clear()
                strict_filtered_count = 0
        poll_telegram_commands(now)
        if in_refresh_window(now, args.refresh_start_hhmm, args.refresh_end_hhmm):
            need_refresh = last_refresh is None
            if last_refresh is not None:
                elapsed = (now - last_refresh).total_seconds() / 60.0
                if hhmm < int(args.market_open_hhmm):
                    refresh_interval_min = int(args.premarket_refresh_interval_min)
                elif theme_selection_day != now.date():
                    refresh_interval_min = int(args.early_refresh_interval_min)
                else:
                    refresh_interval_min = int(args.market_close_hhmm)
                if elapsed >= max(1, refresh_interval_min):
                    need_refresh = True
            if last_refresh is None and suppress_initial_refresh_once:
                # Apply startup-suppress only once.
                last_refresh = now
                need_refresh = False
                suppress_initial_refresh_once = False
            if need_refresh:
                tracking_active = bool(watch_candidates and strict_filtered_count > 0) or any(q > 0 for q in positions.values()) or is_daily_trade_finished(now.date()) or theme_selection_day == now.date()
                if tracking_active:
                    last_refresh = now
                    need_refresh = False
                    notifier.send("재선정대기: 현재 감시 흐름 유지")
                else:
                    notifier.send("후보군선정 시작" if hhmm < int(args.market_open_hhmm) else "종목선정 시작")
            if need_refresh:
                # Fetch a wider pool first, then fill top N after exclusions.
                fetch_pool_size = max(int(args.max_universe), int(args.max_universe) * 8)
                try:
                    raw_universe = fetch_candidate_universe(
                        base_url=args.base_url,
                        token=token,
                        app_key=args.app_key,
                        app_secret=args.app_secret,
                        max_universe=fetch_pool_size,
                        extra_symbols=sorted(set(extra_symbols + list(manual_watch_symbols))),
                    )
                except Exception as e:
                    if is_network_block_error(e):
                        wait_sec = notify_net_error(e)
                        time.sleep(wait_sec)
                        continue
                    raise
                filtered_pool = apply_blocked_universe(raw_universe)
                universe = filtered_pool[: max(1, int(args.max_universe))]
                removed_cnt = max(0, len(raw_universe) - len(filtered_pool))
                trimmed_cnt = max(0, len(filtered_pool) - len(universe))
                if removed_cnt > 0:
                    notifier.send(f"제외종목 적용 {removed_cnt}개")
                if trimmed_cnt > 0:
                    notifier.send(f"후보풀 축약 {trimmed_cnt}개")
                notifier.send(f"후보 {len(universe)}개")
                for symbol, name in universe:
                    if name:
                        known_name_map[symbol] = name
                if universe:
                    send_candidate_list_messages(notifier, universe, chunk_size=25)
                prev_stats_map: Dict[str, PreviousDayStats] = {}
                if universe and not args.disable_prev_day_score:
                    prev_universe = universe[: max(1, int(args.prev_day_max_symbols))]
                    prev_stats_map = fetch_previous_day_stats(
                        base_url=args.base_url,
                        token=token,
                        app_key=args.app_key,
                        app_secret=args.app_secret,
                        universe=prev_universe,
                        lookback_days=int(args.prev_day_lookback_days),
                    )
                    if prev_stats_map:
                        notifier.send(f"전일데이터 {len(prev_stats_map)}개 반영")
                if not in_korean_trading_session(now, args.market_open_hhmm, args.market_close_hhmm):
                    notifier.send("대장주판정대기: 08:00 이후 분봉 확인")
                    last_refresh = datetime.now(KST)
                    first_run_at_window_open = False
                    suppress_initial_refresh_once = False
                    if cycle + 1 < args.max_cycles:
                        time.sleep(max(1, int(args.scan_interval_sec)))
                    continue
                try:
                    leaders, theme_groups = select_theme_leaders(
                        base_url=args.base_url,
                        token=token,
                        app_key=args.app_key,
                        app_secret=args.app_secret,
                        universe=universe,
                        bar_minutes=args.bar_minutes,
                        minute_market_code=args.minute_market_code,
                        prev_day_stats=prev_stats_map,
                        theme_count=int(args.theme_count),
                        progress_cb=lambda done, total, selected_cnt, cur: notifier.send(f"테마진행 {done}/{total} | 분석 {selected_cnt} | {cur}"),
                    )
                except Exception as e:
                    if is_network_block_error(e):
                        wait_sec = notify_net_error(e)
                        time.sleep(wait_sec)
                        continue
                    raise
                strict_filtered_count = len(leaders)
                net_err_streak = 0
                watch_candidates = []
                if not leaders:
                    notifier.send("테마대장 선정 없음")
                else:
                    watch_candidates = leaders[: max(1, int(args.theme_count))]
                    theme_selection_day = now.date()
                    notifier.send(f"테마선정 {len(theme_groups)}개")
                    for group in theme_groups:
                        notifier.send(f"테마그룹 | {format_theme_group(group)}")
                    notifier.send(f"대장감시 {len(watch_candidates)}개 | {watch_preview(watch_candidates)}")
                if watch_candidates:
                    notifier.send(f"추적 {len(watch_candidates)}개 | {watch_preview(watch_candidates)}")
                last_refresh = datetime.now(KST)
                first_run_at_window_open = False
                suppress_initial_refresh_once = False

        if (
            now.weekday() < 5
            and active_session_day == now.date()
            and close_notified_day != now.date()
            and hhmm >= int(args.market_close_hhmm)
        ):
            notifier.send("운영종료: 오늘 운용 종료")
            close_notified_day = now.date()

        if (
            in_korean_trading_session(now, args.market_open_hhmm, args.market_close_hhmm)
            and watch_candidates
            and (
                last_watch_report is None
                or (now - last_watch_report).total_seconds() >= max(60, int(args.watch_report_interval_min) * 60)
            )
        ):
            notifier.send(f"실시간 모니터링 | {watch_preview(watch_candidates)}")
            last_watch_report = now

        if not in_korean_trading_session(now, args.market_open_hhmm, args.market_close_hhmm) or not watch_candidates:
            if cycle + 1 < args.max_cycles:
                time.sleep(max(1, int(args.scan_interval_sec)))
            continue
        buy_candidates: List[Tuple[datetime, float, Candidate, float, str]] = []
        signaled_this_cycle: set[str] = set()
        for c in watch_candidates:
            if is_daily_trade_finished(now.date()):
                break
            if positions.get(c.symbol, 0) <= 0 and c.symbol in blocked_unbuyable_symbols:
                continue
            try:
                raw_rows = fetch_minute_ohlcv(
                    base_url=args.base_url,
                    token=token,
                    app_key=args.app_key,
                    app_secret=args.app_secret,
                    symbol=c.symbol,
                    count_hint=raw_count_hint_for_resampled_bars(80, args.bar_minutes),
                    market_code=args.minute_market_code,
                )
            except Exception as e:
                if is_network_block_error(e):
                    wait_sec = notify_net_error(e)
                    time.sleep(wait_sec)
                    break
                raise
            rows = resample_bars(raw_rows, bar_minutes=args.bar_minutes)
            if not rows:
                continue
            net_err_streak = 0
            close = rows[-1]["close"]
            has_pos = positions.get(c.symbol, 0) > 0
            if not has_pos:
                early_ok, early_reason, early_score = early_momentum_buy_signal_from_minute_bars(
                    rows,
                    early_end_hhmm=args.early_momentum_end_hhmm,
                    min_gain_pct=args.early_momentum_min_gain_pct,
                    volume_mult=args.early_momentum_volume_mult,
                )
                buy_ok, buy_reason, buy_score = buy_signal_from_minute_bars(rows, adx_min=float(args.adx_min))
                brk_ok, brk_reason, brk_score = (False, "", -1.0)
                if not args.disable_breakout_entry:
                    brk_ok, brk_reason, brk_score = breakout_buy_signal_from_minute_bars(
                        rows,
                        lookback_bars=int(args.breakout_lookback_bars),
                        breakout_buffer_pct=float(args.breakout_buffer_pct),
                        volume_window=int(args.breakout_volume_window),
                        volume_mult=float(args.breakout_volume_mult),
                        adx_min=float(args.breakout_adx_min),
                    )
                if not early_ok and not buy_ok and not brk_ok:
                    signal_first_seen_at.pop(c.symbol, None)
                    continue
                chosen_score = buy_score
                chosen_tag = f"기본:{buy_reason}"
                if early_ok and early_score >= max(buy_score, brk_score):
                    chosen_score = early_score
                    chosen_tag = f"조기:{early_reason}"
                elif brk_ok and brk_score >= buy_score:
                    chosen_score = brk_score
                    chosen_tag = f"돌파:{brk_reason}"
                signal_at = signal_first_seen_at.setdefault(c.symbol, now)
                signaled_this_cycle.add(c.symbol)
                buy_candidates.append((signal_at, chosen_score, c, float(close), chosen_tag))
            else:
                prev_stat = fetch_single_previous_day_stat(
                    args.base_url,
                    token,
                    args.app_key,
                    args.app_secret,
                    c.symbol,
                )
                limit_up_price = extract_limit_up_price(prev_stat)
                intraday_high = max(float(r["high"]) for r in rows)
                if limit_up_price > 0 and intraday_high >= limit_up_price * 0.999:
                    if limit_up_hold_day.get(c.symbol) != now.date():
                        limit_up_hold_day[c.symbol] = now.date()
                        notifier.send(f"상한가보유전환 | {display_name(c.name, c.symbol)} | 기준 {limit_up_price:.0f}원")
                hold_due_limit_today = limit_up_hold_day.get(c.symbol) == now.date()
                if hold_due_limit_today:
                    continue
                sell_ok, sell_reason = sell_signal_from_minute_bars(rows)
                ep = float(entry_price.get(c.symbol, close))
                pp = max(float(peak_price.get(c.symbol, ep)), float(close))
                peak_price[c.symbol] = pp
                stop_loss_ok = float(close) <= ep * (1.0 - max(0.0, float(args.sell_stop_loss_pct)))
                trail_armed = pp >= ep * (1.0 + max(0.0, float(args.sell_trailing_activate_pct)))
                trailing_ok = trail_armed and (float(close) <= pp * (1.0 - max(0.0, float(args.sell_trailing_stop_pct))))
                hold_min = -1
                hold_time_ok = False
                et = entry_time.get(c.symbol)
                if et is not None:
                    hold_min = int((now - et).total_seconds() // 60)
                    hold_time_ok = int(args.sell_max_hold_min) > 0 and hold_min >= int(args.sell_max_hold_min)
                final_sell_ok = bool(sell_ok or stop_loss_ok or trailing_ok or hold_time_ok)
                reason_parts: List[str] = []
                if sell_ok:
                    reason_parts.append(f"지표:{sell_reason}")
                if stop_loss_ok:
                    reason_parts.append(f"손절(entry={ep:.0f},close={float(close):.0f})")
                if trailing_ok:
                    reason_parts.append(f"트레일링(entry={ep:.0f},peak={pp:.0f},close={float(close):.0f})")
                if hold_time_ok:
                    reason_parts.append(f"시간청산(hold={hold_min}m)")
                final_sell_reason = " | ".join(reason_parts) if reason_parts else "no-sell"
                if not final_sell_ok:
                    continue
                qty = positions.get(c.symbol, 0)
                if qty <= 0:
                    continue
                carryover_exit = c.symbol in limit_up_hold_day and limit_up_hold_day.get(c.symbol) != now.date()
                if args.dry_run:
                    notifier.send(f"매도 {display_name(c.name, c.symbol)} {qty}주 {close:.0f}원 (DRY) | {final_sell_reason}")
                    clear_position_state(c.symbol)
                    finish_if_all_closed(carryover_exit)
                else:
                    try:
                        res = place_order(
                            base_url=args.base_url,
                            token=token,
                            app_key=args.app_key,
                            app_secret=args.app_secret,
                            cano=args.cano,
                            acnt_prdt_cd=args.acnt_prdt_cd,
                            symbol=c.symbol,
                            qty=qty,
                            side="sell",
                        )
                    except Exception as e:
                        if is_network_block_error(e):
                            wait_sec = notify_net_error(e)
                            time.sleep(wait_sec)
                            continue
                        raise
                    ok = str(res.get("rt_cd", "")) == "0"
                    if ok:
                        odno = str(res.get("output", {}).get("ODNO", "")).strip()
                        notifier.send(f"매도주문접수 {display_name(c.name, c.symbol)} {qty}주 {close:.0f}원 | 주문번호:{odno} | {final_sell_reason}")
                        try:
                            status, filled_qty, ord_qty = wait_order_fill_status(
                                base_url=args.base_url,
                                token=token,
                                app_key=args.app_key,
                                app_secret=args.app_secret,
                                cano=args.cano,
                                acnt_prdt_cd=args.acnt_prdt_cd,
                                odno=odno,
                            )
                        except Exception as e:
                            if is_network_block_error(e):
                                wait_sec = notify_net_error(e)
                                time.sleep(wait_sec)
                                continue
                            raise
                        if status == "filled":
                            notifier.send(f"매도체결 {display_name(c.name, c.symbol)} {filled_qty}주 | 주문번호:{odno}")
                            clear_position_state(c.symbol)
                            finish_if_all_closed(carryover_exit)
                        elif status == "partial":
                            notifier.send(f"매도부분체결 {display_name(c.name, c.symbol)} {filled_qty}/{max(ord_qty, qty)}주 | 주문번호:{odno}")
                            remain = max(0, qty - max(0, filled_qty))
                            positions[c.symbol] = remain
                            if remain <= 0:
                                clear_position_state(c.symbol)
                                finish_if_all_closed(carryover_exit)
                        else:
                            notifier.send(f"매도미체결 {display_name(c.name, c.symbol)} {filled_qty}/{max(ord_qty, qty)}주 | 주문번호:{odno}")
                    else:
                        notifier.send(f"매도실패 {display_name(c.name, c.symbol)} {qty}주 {close:.0f}원")
        for c in watch_candidates:
            if c.symbol not in signaled_this_cycle and positions.get(c.symbol, 0) <= 0:
                signal_first_seen_at.pop(c.symbol, None)
        if is_daily_trade_finished(now.date()):
            if cycle + 1 < args.max_cycles:
                time.sleep(max(1, int(args.scan_interval_sec)))
            continue
        if buy_candidates:
            holding_count = sum(1 for q in positions.values() if q > 0)
            slots_left = max(0, int(args.max_positions) - holding_count)
            buys_left = min(max(0, int(args.max_buys_per_scan)), slots_left)
            if buys_left <= 0:
                if last_slots_full_notice_at is None or (now - last_slots_full_notice_at).total_seconds() >= 300:
                    notifier.send(f"매수대기 | 보유 {holding_count}/{int(args.max_positions)}종목으로 감시만 진행")
                    last_slots_full_notice_at = now
                if cycle + 1 < args.max_cycles:
                    time.sleep(max(1, int(args.scan_interval_sec)))
                continue
            if buys_left > 0:
                # Priority: symbols whose buy signal appeared earlier are traded first.
                ranked = sorted(buy_candidates, key=lambda x: (x[0], -x[1]))
                for signal_at, score, c, close, buy_tag in ranked[:buys_left]:
                    if float(args.order_krw) > 0:
                        budget = float(args.order_krw)
                    else:
                        budget = max(0.0, float(args.initial_cash) * min(1.0, max(0.0, float(args.position_size_pct))))
                    qty = int(max(0.0, budget) // max(1.0, close))
                    if qty <= 0:
                        notifier.send(f"매수스킵 {display_name(c.name, c.symbol)} 수량0")
                        continue
                    if args.dry_run:
                        notifier.send(f"매수 {display_name(c.name, c.symbol)} {qty}주 {close:.0f}원 (DRY) | {buy_tag} | 신호시각:{signal_at.strftime('%H:%M:%S')}")
                        set_position_entry(c.symbol, qty, float(close))
                    else:
                        try:
                            res = place_order(
                                base_url=args.base_url,
                                token=token,
                                app_key=args.app_key,
                                app_secret=args.app_secret,
                                cano=args.cano,
                                acnt_prdt_cd=args.acnt_prdt_cd,
                                symbol=c.symbol,
                                qty=qty,
                                side="buy",
                            )
                        except Exception as e:
                            if is_network_block_error(e):
                                wait_sec = notify_net_error(e)
                                time.sleep(wait_sec)
                                continue
                            raise
                        ok = str(res.get("rt_cd", "")) == "0"
                        if ok:
                            odno = str(res.get("output", {}).get("ODNO", "")).strip()
                            notifier.send(f"매수주문접수 {display_name(c.name, c.symbol)} {qty}주 {close:.0f}원 | 주문번호:{odno} | {buy_tag} | 신호시각:{signal_at.strftime('%H:%M:%S')}")
                            try:
                                status, filled_qty, ord_qty = wait_order_fill_status(
                                    base_url=args.base_url,
                                    token=token,
                                    app_key=args.app_key,
                                    app_secret=args.app_secret,
                                    cano=args.cano,
                                    acnt_prdt_cd=args.acnt_prdt_cd,
                                    odno=odno,
                                )
                            except Exception as e:
                                if is_network_block_error(e):
                                    wait_sec = notify_net_error(e)
                                    time.sleep(wait_sec)
                                    continue
                                raise
                            if status == "filled":
                                notifier.send(f"매수체결 {display_name(c.name, c.symbol)} {filled_qty}주 | 주문번호:{odno} | {buy_tag}")
                                set_position_entry(c.symbol, filled_qty, float(close))
                            elif status == "partial":
                                notifier.send(f"매수부분체결 {display_name(c.name, c.symbol)} {filled_qty}/{max(ord_qty, qty)}주 | 주문번호:{odno} | {buy_tag}")
                                if filled_qty > 0:
                                    set_position_entry(c.symbol, filled_qty, float(close))
                            else:
                                notifier.send(f"매수미체결 {display_name(c.name, c.symbol)} {filled_qty}/{max(ord_qty, qty)}주 | 주문번호:{odno} | {buy_tag}")
                                blocked_unbuyable_symbols.add(c.symbol)
                                notifier.send(f"제외등록 {display_name(c.name, c.symbol)} | 매수미체결")
                        else:
                            notifier.send(f"매수실패 {display_name(c.name, c.symbol)} {qty}주 {close:.0f}원 | {buy_tag}")
        if cycle + 1 < args.max_cycles:
            time.sleep(max(1, int(args.scan_interval_sec)))


if __name__ == "__main__":
    main()


