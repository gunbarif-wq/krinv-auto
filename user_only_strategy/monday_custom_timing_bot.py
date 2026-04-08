from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from fetch_kis_daily import get_access_token  # noqa: E402


KST = ZoneInfo("Asia/Seoul")
VTS_BASE_URL = "https://openapivts.koreainvestment.com:29443"
PROD_BASE_URL = "https://openapi.koreainvestment.com:9443"


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

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)[: max(1, max_universe)]
    return [(sym, names.get(sym, "")) for sym, _ in ranked]


def fetch_minute_ohlcv(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    symbol: str,
    count_hint: int = 180,
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

    for _ in range(14):
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": ymd,
            "FID_INPUT_HOUR_1": cursor_time,
            "FID_PW_DATA_INCU_YN": "N",
        }
        data = request_json_with_retry("get", url, headers=headers, params=params, timeout=15, retries=3)
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
    pref = np.concatenate(([0.0], np.cumsum(arr, dtype=float)))
    out[window - 1 :] = (pref[window:] - pref[:-window]) / float(window)
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


def _parse_bar_datetime(v: float) -> datetime:
    s = str(int(v)).zfill(14)
    return datetime.strptime(s, "%Y%m%d%H%M%S")


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
) -> List[Candidate]:
    selected: List[Candidate] = []
    for symbol, name in universe:
        rows = fetch_minute_ohlcv(
            base_url=base_url,
            token=token,
            app_key=app_key,
            app_secret=app_secret,
            symbol=symbol,
            count_hint=raw_count_hint_for_resampled_bars(90, bar_minutes),
        )
        bars = resample_bars(rows, bar_minutes=bar_minutes)
        if len(bars) < 70:
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
        if convergence > ma_converge_pct:
            continue

        if close[-1] < ma60[-1]:
            continue

        keep_60 = True
        for i in range(1, max(1, ma60_no_break_days) + 1):
            if i >= close.shape[0]:
                break
            if not np.isfinite(ma60[-i]) or low[-i] < ma60[-i]:
                keep_60 = False
                break
        if not keep_60:
            continue

        keep_20 = True
        for i in range(1, max(1, ma20_support_days) + 1):
            if i >= close.shape[0]:
                break
            if not np.isfinite(ma20[-i]) or close[-i] < ma20[-i]:
                keep_20 = False
                break
        if not keep_20:
            continue

        selected.append(
            Candidate(
                symbol=symbol,
                name=name,
                close=float(close[-1]),
                ma3=float(ma3[-1]),
                ma5=float(ma5[-1]),
                ma10=float(ma10[-1]),
                ma20=float(ma20[-1]),
                ma60=float(ma60[-1]),
            )
        )
    return selected


class Notifier:
    def __init__(self, log_path: Path, telegram_token: str, telegram_chat_id: str):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.telegram_token = telegram_token.strip()
        self.telegram_chat_id = telegram_chat_id.strip()

    def send(self, text: str) -> None:
        ts = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {text}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.telegram_token and self.telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                requests.post(url, json={"chat_id": self.telegram_chat_id, "text": line}, timeout=8)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monday-only custom timing bot (user rule only)")
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", PROD_BASE_URL))
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--cano", default=os.getenv("KIS_CANO", ""))
    p.add_argument("--acnt-prdt-cd", default=os.getenv("KIS_ACNT_PRDT_CD", "01"))
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--max-universe", type=int, default=30)
    p.add_argument("--scan-interval-sec", type=int, default=60)
    p.add_argument("--max-cycles", type=int, default=200)
    p.add_argument("--bar-minutes", type=int, choices=[1, 3, 5], default=5)
    p.add_argument("--refresh-start-hhmm", type=int, default=800)
    p.add_argument("--refresh-end-hhmm", type=int, default=2000)
    p.add_argument("--refresh-interval-min", type=int, default=60)
    p.add_argument("--order-krw", type=float, default=300_000)
    p.add_argument("--ma-converge-pct", type=float, default=0.015)
    p.add_argument("--ma60-no-break-days", type=int, default=5, help="minute bars count for no-break check")
    p.add_argument("--ma20-support-days", type=int, default=3, help="minute bars count for MA20 support check")
    p.add_argument("--adx-min", type=float, default=20.0)
    p.add_argument("--log-file", default="logs/user_only_strategy_signals.txt")
    p.add_argument("--telegram-bot-token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""))
    p.add_argument("--telegram-chat-id", default=os.getenv("TELEGRAM_CHAT_ID", ""))
    return p.parse_args()


def in_korean_regular_session(now: datetime) -> bool:
    hhmm = now.hour * 100 + now.minute
    return now.weekday() < 5 and 900 <= hhmm <= 1530


def in_refresh_window(now: datetime, start_hhmm: int, end_hhmm: int) -> bool:
    hhmm = now.hour * 100 + now.minute
    return now.weekday() < 5 and int(start_hhmm) <= hhmm <= int(end_hhmm)


def buy_signal_from_minute_bars(rows: List[Dict[str, float]], adx_min: float) -> Tuple[bool, str]:
    if len(rows) < 70:
        return False, "warmup"
    high = np.asarray([r["high"] for r in rows], dtype=float)
    low = np.asarray([r["low"] for r in rows], dtype=float)
    close = np.asarray([r["close"] for r in rows], dtype=float)
    st_k, st_d = slow_stochastic(high, low, close, period=14, smooth_k=3, smooth_d=3)
    plus_di, minus_di, adx = dmi_adx(high, low, close, period=14)
    rsi14 = rsi(close, period=14)
    mid, _, lower = bollinger(close, window=20, mult=2.0)

    stoch_cross = crossed_up(st_k, st_d)
    dmi_cross = crossed_up(plus_di, minus_di)
    adx_ok = np.isfinite(adx[-1]) and np.isfinite(adx[-2]) and adx[-1] >= adx_min and adx[-1] > adx[-2]
    rsi_boll_ok = False
    if np.isfinite(rsi14[-1]) and np.isfinite(rsi14[-2]) and np.isfinite(mid[-1]) and np.isfinite(lower[-1]):
        rsi_boll_ok = (
            (rsi14[-2] < 40 and rsi14[-1] > rsi14[-2])
            or (close[-2] < lower[-2] and close[-1] > lower[-1])
            or (rsi14[-1] > 50 and close[-1] > mid[-1])
        )

    ok = stoch_cross and dmi_cross and adx_ok and rsi_boll_ok
    reason = f"stoch={stoch_cross} dmi={dmi_cross} adx={adx_ok} rsi_boll={rsi_boll_ok}"
    return ok, reason


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

    stoch_dead = crossed_down(st_k, st_d) and np.isfinite(st_k[-2]) and st_k[-2] >= 75
    dmi_dead = crossed_down(plus_di, minus_di) and np.isfinite(adx[-1]) and np.isfinite(adx[-2]) and adx[-1] >= adx[-2]
    rsi_mid_break = (
        np.isfinite(rsi14[-1]) and np.isfinite(mid[-1]) and close[-1] < mid[-1] and rsi14[-1] < 50
    )
    ma20_break = np.isfinite(ma20[-1]) and close[-1] < ma20[-1]
    ok = stoch_dead or dmi_dead or rsi_mid_break or ma20_break
    reason = f"stoch_dead={stoch_dead} dmi_dead={dmi_dead} rsi_mid_break={rsi_mid_break} ma20_break={ma20_break}"
    return ok, reason


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET required")
    if not args.dry_run and (not args.cano or not args.acnt_prdt_cd):
        raise RuntimeError("KIS_CANO and KIS_ACNT_PRDT_CD required for live order")

    notifier = Notifier(Path(args.log_file), args.telegram_bot_token, args.telegram_chat_id)
    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
    filtered: List[Candidate] = []
    last_refresh: datetime | None = None
    positions: Dict[str, int] = {}
    for cycle in range(max(1, args.max_cycles)):
        now = datetime.now(KST)
        if in_refresh_window(now, args.refresh_start_hhmm, args.refresh_end_hhmm):
            need_refresh = last_refresh is None
            if last_refresh is not None:
                elapsed = (now - last_refresh).total_seconds() / 60.0
                if elapsed >= max(1, int(args.refresh_interval_min)):
                    need_refresh = True
            if need_refresh:
                universe = fetch_candidate_universe(
                    base_url=args.base_url,
                    token=token,
                    app_key=args.app_key,
                    app_secret=args.app_secret,
                    max_universe=args.max_universe,
                )
                notifier.send(f"universe_candidates={len(universe)}")
                filtered = minute_filter(
                    base_url=args.base_url,
                    token=token,
                    app_key=args.app_key,
                    app_secret=args.app_secret,
                    universe=universe,
                    ma_converge_pct=args.ma_converge_pct,
                    ma60_no_break_days=args.ma60_no_break_days,
                    ma20_support_days=args.ma20_support_days,
                    bar_minutes=args.bar_minutes,
                )
                if not filtered:
                    notifier.send("no symbol passed minute filters")
                else:
                    preview = ", ".join([f"{c.symbol}({c.name})" for c in filtered[:10]])
                    notifier.send(f"minute_filter_pass={len(filtered)} symbols={preview}")
                last_refresh = now

        if not in_korean_regular_session(now) or not filtered:
            if cycle + 1 < args.max_cycles:
                time.sleep(max(1, int(args.scan_interval_sec)))
            continue
        for c in filtered:
            raw_rows = fetch_minute_ohlcv(
                base_url=args.base_url,
                token=token,
                app_key=args.app_key,
                app_secret=args.app_secret,
                symbol=c.symbol,
                count_hint=raw_count_hint_for_resampled_bars(80, args.bar_minutes),
            )
            rows = resample_bars(raw_rows, bar_minutes=args.bar_minutes)
            if len(rows) < 70:
                continue
            close = rows[-1]["close"]
            has_pos = positions.get(c.symbol, 0) > 0
            if not has_pos:
                buy_ok, buy_reason = buy_signal_from_minute_bars(rows, adx_min=float(args.adx_min))
                if not buy_ok:
                    continue
                qty = int(max(0.0, args.order_krw) // max(1.0, close))
                if qty <= 0:
                    notifier.send(f"[BUY_SKIP] {c.symbol} qty=0 close={close:.0f}")
                    continue
                if args.dry_run:
                    notifier.send(f"[BUY] {c.symbol} {c.name} qty={qty} close={close:.0f} reason={buy_reason} dry_run=1")
                    positions[c.symbol] = qty
                else:
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
                    ok = str(res.get("rt_cd", "")) == "0"
                    notifier.send(
                        f"[BUY{'_OK' if ok else '_FAIL'}] {c.symbol} {c.name} qty={qty} close={close:.0f} "
                        f"reason={buy_reason} msg={res.get('msg1', '')}"
                    )
                    if ok:
                        positions[c.symbol] = qty
            else:
                sell_ok, sell_reason = sell_signal_from_minute_bars(rows)
                if not sell_ok:
                    continue
                qty = positions.get(c.symbol, 0)
                if qty <= 0:
                    continue
                if args.dry_run:
                    notifier.send(f"[SELL] {c.symbol} {c.name} qty={qty} close={close:.0f} reason={sell_reason} dry_run=1")
                    positions[c.symbol] = 0
                else:
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
                    ok = str(res.get("rt_cd", "")) == "0"
                    notifier.send(
                        f"[SELL{'_OK' if ok else '_FAIL'}] {c.symbol} {c.name} qty={qty} close={close:.0f} "
                        f"reason={sell_reason} msg={res.get('msg1', '')}"
                    )
                    if ok:
                        positions[c.symbol] = 0
        if cycle + 1 < args.max_cycles:
            time.sleep(max(1, int(args.scan_interval_sec)))


if __name__ == "__main__":
    main()
