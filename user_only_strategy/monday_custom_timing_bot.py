from __future__ import annotations

import argparse
import html
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from fetch_kis_daily import fetch_daily_prices, get_access_token  # noqa: E402
from user_only_strategy.krx_symbol_names import (  # noqa: E402
    DEFAULT_SYMBOL_NAME_FILE,
    load_symbol_name_map,
    refresh_symbol_name_map_from_krx,
    save_symbol_name_map,
)


KST = ZoneInfo("Asia/Seoul")
VTS_BASE_URL = "https://openapivts.koreainvestment.com:29443"
PROD_BASE_URL = "https://openapi.koreainvestment.com:9443"
DEFAULT_TRADING_OPEN_HHMM = 850
DEFAULT_TRADING_CLOSE_HHMM = 1530
DEFAULT_SEARCH_START_HHMM = 801
DEFAULT_MINUTE_MARKET_CODE = "UN"
DEFAULT_WATCH_STATE_FILE = str(ROOT / "user_only_strategy" / "watch_state.json")
DEFAULT_BAR_CACHE_FILE = str(ROOT / "logs" / "bar_cache.json")
CHART_WINDOW_BARS = 60
SYMBOL_ALIAS_MAP = {
    "하이닉스": "000660",
    "sk하이닉스": "000660",
    "에스케이하이닉스": "000660",
    "삼전": "005930",
    "삼성전자": "005930",
    "삼성sdi": "006400",
    "삼성에스디아이": "006400",
    "넥스원": "079550",
    "lig넥스원": "079550",
    "엘아이지넥스원": "079550",
    "lg전자": "066570",
    "엘지전자": "066570",
    "우리기술": "032820",
    "에코프로": "086520",
    "대우건설": "047040",
    "lk삼양": "225190",
    "엘케이삼양": "225190",
}


class _HttpThrottle:
    def __init__(self, min_interval_sec: float) -> None:
        self.min_interval_sec = float(min_interval_sec)
        self._last_at = 0.0

    def wait(self) -> None:
        now = time.time()
        delta = now - self._last_at
        if delta < self.min_interval_sec:
            time.sleep(self.min_interval_sec - delta)
        self._last_at = time.time()


HTTP_THROTTLE = _HttpThrottle(float(os.getenv("KIS_HTTP_MIN_INTERVAL_SEC", "0.35")))


class KisHttpError(RuntimeError):
    def __init__(self, payload: Dict | None = None, message: str = "") -> None:
        self.payload = payload or {}
        super().__init__(message or json.dumps(self.payload, ensure_ascii=False))


def is_token_expired_error(res: Dict | None) -> bool:
    if not isinstance(res, dict):
        return False
    msg_cd = str(res.get("msg_cd", "")).strip()
    msg1 = str(res.get("msg1", "")).strip()
    return "EGW00123" in msg_cd or "기간이 만료된 token" in msg1


def is_token_issue_rate_limited_error(exc: Exception) -> bool:
    text = str(exc)
    return "EGW00133" in text or "1분당 1회" in text or "잠시 후 다시 시도" in text


def _extract_error_payload(response: requests.Response) -> Dict:
    try:
        data = response.json()
        if isinstance(data, dict):
            return {
                "rt_cd": str(data.get("rt_cd", "1")).strip() or "1",
                "msg_cd": str(data.get("msg_cd", "")).strip(),
                "msg1": str(data.get("msg1", "")).strip() or f"http_{response.status_code}",
                "output": data.get("output") if isinstance(data.get("output"), dict) else {},
            }
    except Exception:
        pass
    return {"rt_cd": "1", "msg1": f"http_{response.status_code} {response.text[:220]}"}


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
    request_headers = dict(headers)
    refreshed_once = False
    for i in range(max(1, retries)):
        try:
            HTTP_THROTTLE.wait()
            if method.lower() == "get":
                r = requests.get(url, headers=request_headers, params=params, timeout=timeout)
            else:
                r = requests.post(url, headers=request_headers, data=json.dumps(params), timeout=timeout)
            if r.status_code >= 400:
                payload = _extract_error_payload(r)
                if is_token_expired_error(payload) and not refreshed_once:
                    refreshed_once = True
                    request_headers["authorization"] = (
                        f"Bearer {get_access_token(request_headers.get('appKey', ''), request_headers.get('appSecret', ''), base_url=_base_url_from_url(url), force_refresh=True)}"
                    )
                    continue
                raise KisHttpError(payload)
            return r.json()
        except KisHttpError as e:
            last_err = e
            if i + 1 < retries:
                time.sleep(sleep_base * (i + 1))
        except Exception as e:
            last_err = e
            if i + 1 < retries:
                time.sleep(sleep_base * (i + 1))
    if last_err:
        if isinstance(last_err, KisHttpError):
            raise last_err
        raise last_err
    raise RuntimeError("request failed")


def is_demo_base_url(base_url: str) -> bool:
    return "openapivts" in base_url.lower()


def _base_url_from_url(url: str) -> str:
    m = re.match(r"^(https?://[^/]+)", str(url).strip())
    return m.group(1) if m else str(url).strip()


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
    HTTP_THROTTLE.wait()
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
    auth_token = token
    for attempt in range(2):
        hashkey = get_hashkey(base_url, app_key, app_secret, body)
        headers = {
            "authorization": f"Bearer {auth_token}",
            "appKey": app_key,
            "appSecret": app_secret,
            "tr_id": tr_id,
            "custtype": "P",
            "hashkey": hashkey,
            "content-type": "application/json",
        }
        HTTP_THROTTLE.wait()
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=15)
        if r.status_code < 400:
            return r.json()
        payload = _extract_error_payload(r)
        if is_token_expired_error(payload) and attempt == 0:
            auth_token = get_access_token(app_key, app_secret, base_url=base_url, force_refresh=True)
            continue
        payload["msg1"] = f"http_{r.status_code} {json.dumps(payload, ensure_ascii=False)}"
        return payload
    return {"rt_cd": "1", "msg1": "unexpected_order_retry_failure"}


def format_kis_error(res: Dict) -> str:
    rt_cd = str(res.get("rt_cd", "")).strip()
    msg_cd = str(res.get("msg_cd", "")).strip()
    msg1 = str(res.get("msg1", "")).strip()
    out = res.get("output") or res.get("output1") or {}
    extra = ""
    if isinstance(out, dict) and out:
        keys = ["ODNO", "ORD_TMD", "KRX_FWDG_ORD_ORGNO"]
        parts = []
        for k in keys:
            v = out.get(k)
            if v:
                parts.append(f"{k}={v}")
        if parts:
            extra = " | " + " ".join(parts)[:120]
    base = f"rt_cd={rt_cd} msg_cd={msg_cd} msg1={msg1}".strip()
    return (base + extra).strip()


def is_rate_limited_error(res: Dict) -> bool:
    msg_cd = str(res.get("msg_cd", "")).strip()
    msg1 = str(res.get("msg1", "")).strip()
    return "EGW00201" in msg_cd or "EGW00201" in msg1 or "초당 거래건수" in msg1


def is_insufficient_order_cash_error(res: Dict) -> bool:
    msg_cd = str(res.get("msg_cd", "")).strip()
    msg1 = str(res.get("msg1", "")).strip()
    return "40250000" in msg_cd or "주문가능금액이 부족" in msg1


def inquire_orderable_qty(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    cano: str,
    acnt_prdt_cd: str,
    *,
    symbol: str,
    price: float,
    qty_hint: int,
) -> int:
    """
    KIS '주문가능조회' returns the true max orderable qty for a given symbol/price.
    In mock/VTS, this can be lower than what cash inquiry suggests.
    """
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
        "PDNO": str(symbol).zfill(6),
        "ORD_DVSN": "01",  # market
        "ORD_QTY": str(max(1, int(qty_hint))),
        "ORD_UNPR": str(int(max(0.0, float(price)))),
    }
    data = request_json_with_retry("get", url, headers=headers, params=params, timeout=15, retries=3)
    if str(data.get("rt_cd", "")).strip() != "0":
        raise KisHttpError(data)
    candidates: List[Dict] = []
    out = data.get("output", {})
    if isinstance(out, dict):
        candidates.append(out)
    elif isinstance(out, list):
        candidates.extend([x for x in out if isinstance(x, dict)])
    for extra_key in ("output1", "output2"):
        extra = data.get(extra_key, {})
        if isinstance(extra, dict):
            candidates.append(extra)
        elif isinstance(extra, list):
            candidates.extend([x for x in extra if isinstance(x, dict)])

    for obj in candidates:
        for key in (
            "ord_psbl_qty",
            "max_ord_qty",
            "max_buy_qty",
            "psbl_qty",
            "buy_psbl_qty",
        ):
            v = obj.get(key)
            if v is None:
                continue
            s = str(v).replace(",", "").strip()
            if not s:
                continue
            if s.isdigit():
                return max(0, int(s))
            try:
                return max(0, int(float(s)))
            except Exception:
                continue
    # If schema changes, safest fallback is to not clamp.
    return max(0, int(qty_hint))


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
    sleep_sec: float = 1.2,
) -> Tuple[str, int, int]:
    last_status = ("pending", 0, 0)
    for i in range(max(1, retries)):
        time.sleep(max(0.2, float(sleep_sec)))
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
    return last_status


def confirm_holding_qty(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    cano: str,
    acnt_prdt_cd: str,
    symbol: str,
) -> int:
    try:
        holdings = fetch_account_holdings(
            base_url=base_url,
            token=token,
            app_key=app_key,
            app_secret=app_secret,
            cano=cano,
            acnt_prdt_cd=acnt_prdt_cd,
        )
    except Exception:
        return -1
    return int((holdings.get(symbol) or {}).get("qty", 0))


def reconcile_sell_fill_status(
    status: str,
    filled_qty: int,
    ord_qty: int,
    expected_qty: int,
    remaining_qty: int,
) -> Tuple[str, int, int]:
    expected = max(0, int(expected_qty))
    filled = max(0, int(filled_qty))
    ordered = max(0, int(ord_qty))
    remaining = int(remaining_qty)
    if remaining < 0:
        return status, filled, ordered
    sold_by_balance = max(0, expected - remaining)
    filled = max(filled, sold_by_balance)
    ordered = max(ordered, expected)
    if remaining <= 0 and expected > 0:
        return ("filled", max(filled, expected), ordered)
    if sold_by_balance > 0:
        return ("partial", filled, ordered)
    return (status, filled, ordered)


def balance_inquiry_tr_id(base_url: str) -> str:
    return "VTTC8434R" if is_demo_base_url(base_url) else "TTTC8434R"


def fetch_account_holdings(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    cano: str,
    acnt_prdt_cd: str,
) -> Dict[str, Dict[str, float | int | str]]:
    url = f"{base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": balance_inquiry_tr_id(base_url),
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
    data = request_json_with_retry("get", url, headers=headers, params=params, timeout=15, retries=2)
    rows = data.get("output1", [])
    if not isinstance(rows, list):
        return {}
    holdings: Dict[str, Dict[str, float | int | str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("pdno", "")).strip().zfill(6)
        qty = _to_int(row.get("hldg_qty"))
        if not (symbol.isdigit() and len(symbol) == 6 and qty > 0):
            continue
        holdings[symbol] = {
            "name": str(row.get("prdt_name", "")).strip(),
            "qty": qty,
            "avg_price": _to_float(row.get("pchs_avg_pric")),
        }
    return holdings


def fetch_account_budget_info(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    cano: str,
    acnt_prdt_cd: str,
) -> Tuple[float, float]:
    url = f"{base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": balance_inquiry_tr_id(base_url),
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
    data = request_json_with_retry("get", url, headers=headers, params=params, timeout=15, retries=2)
    holdings_rows = data.get("output1", [])
    summary_rows = data.get("output2", [])
    available_cash = 0.0
    total_equity = 0.0
    if isinstance(summary_rows, list) and summary_rows:
        row = summary_rows[0] if isinstance(summary_rows[0], dict) else {}
        ord_psbl_cash = _to_float(row.get("ord_psbl_cash"))
        dnca_tot_amt = _to_float(row.get("dnca_tot_amt"))
        prvs_rcdl_excc_amt = _to_float(row.get("prvs_rcdl_excc_amt"))
        cash_only_candidates = [x for x in (dnca_tot_amt, prvs_rcdl_excc_amt) if x > 0]
        if cash_only_candidates:
            available_cash = min(cash_only_candidates)
        else:
            available_cash = 0.0
        total_equity = max(
            _to_float(row.get("tot_evlu_amt")),
            _to_float(row.get("scts_evlu_amt")) + available_cash,
            _to_float(row.get("tot_evlu_pfls_amt")) + _to_float(row.get("tot_pchs_amt")) + available_cash,
        )
    if total_equity <= 0 and isinstance(holdings_rows, list):
        holdings_value = 0.0
        for row in holdings_rows:
            if not isinstance(row, dict):
                continue
            qty = _to_int(row.get("hldg_qty"))
            if qty <= 0:
                continue
            holdings_value += max(_to_float(row.get("evlu_amt")), _to_float(row.get("pchs_amt")))
        total_equity = holdings_value + available_cash
    return max(0.0, available_cash), max(0.0, total_equity)


def compute_order_budget_krw(
    args: argparse.Namespace,
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    cano: str,
    acnt_prdt_cd: str,
) -> float:
    pct = min(1.0, max(0.0, float(args.position_size_pct)))
    order_cap = max(0.0, float(getattr(args, "max_order_krw", 0.0)))
    requested_budget = max(0.0, float(args.order_krw)) if float(args.order_krw) > 0 else -1.0
    fallback_budget = requested_budget if requested_budget >= 0 else max(0.0, float(args.initial_cash) * pct)
    try:
        available_cash, total_equity = fetch_account_budget_info(
            base_url=base_url,
            token=token,
            app_key=app_key,
            app_secret=app_secret,
            cano=cano,
            acnt_prdt_cd=acnt_prdt_cd,
        )
    except Exception:
        return min(fallback_budget, order_cap) if order_cap > 0 else fallback_budget
    if requested_budget >= 0:
        budget = max(0.0, min(available_cash, requested_budget))
        return min(budget, order_cap) if order_cap > 0 else budget
    if total_equity <= 0:
        budget = min(available_cash, fallback_budget) if available_cash > 0 else fallback_budget
        return min(budget, order_cap) if order_cap > 0 else budget
    budget = max(0.0, min(available_cash, total_equity * pct))
    return min(budget, order_cap) if order_cap > 0 else budget


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


def send_candidate_list_messages(
    notifier: "Notifier",
    universe: List[Tuple[str, str]],
    chunk_size: int = 25,
    *,
    enabled: bool = False,
) -> None:
    if not enabled:
        return
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


def load_watch_state(path_text: str) -> Dict[str, object]:
    path = Path(path_text).expanduser()
    default: Dict[str, object] = {
        "manual_watch_symbols": [],
        "entry_price": {},
        "peak_price": {},
        "entry_time": {},
        "profit_take_stage": {},
        "trading_paused": False,
        "limit_up_hold_day": {},
        "watch_candidates": [],
        "bought_symbols_today": [],
        "traded_symbols_today": [],
        "strict_filtered_count": 0,
        "theme_selection_day": "",
        "daily_trade_finished_day": "",
    }
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    if not isinstance(payload, dict):
        return default
    merged = dict(default)
    for key, default_value in default.items():
        value = payload.get(key)
        if isinstance(default_value, list) and isinstance(value, list):
            merged[key] = value
        elif isinstance(default_value, dict) and isinstance(value, dict):
            merged[key] = value
    return merged


def candidate_to_state_row(candidate: "Candidate") -> Dict[str, object]:
    return {
        "symbol": str(candidate.symbol).zfill(6),
        "name": str(candidate.name or ""),
        "close": float(candidate.close),
        "ma3": float(candidate.ma3),
        "ma5": float(candidate.ma5),
        "ma10": float(candidate.ma10),
        "ma20": float(candidate.ma20),
        "ma60": float(candidate.ma60),
        "leader_score": float(candidate.leader_score),
        "leader_reason": str(candidate.leader_reason or ""),
        "theme_id": int(candidate.theme_id),
        "theme_name": str(candidate.theme_name or ""),
    }


def candidate_from_state_row(row: Dict[str, object]) -> "Candidate | None":
    symbol = str(row.get("symbol", "")).strip()
    if not symbol.isdigit():
        return None
    return Candidate(
        symbol=symbol.zfill(6),
        name=str(row.get("name", "") or symbol),
        close=_to_float(row.get("close")),
        ma3=_to_float(row.get("ma3")),
        ma5=_to_float(row.get("ma5")),
        ma10=_to_float(row.get("ma10")),
        ma20=_to_float(row.get("ma20")),
        ma60=_to_float(row.get("ma60")),
        leader_score=_to_float(row.get("leader_score")),
        leader_reason=str(row.get("leader_reason", "") or ""),
        theme_id=_to_int(row.get("theme_id")),
        theme_name=str(row.get("theme_name", "") or ""),
    )


def save_watch_state(
    path_text: str,
    *,
    manual_watch_symbols: set[str],
    entry_price: Dict[str, float],
    peak_price: Dict[str, float],
    entry_time: Dict[str, datetime],
    profit_take_stage: Dict[str, int],
    trading_paused: bool,
    limit_up_hold_day: Dict[str, datetime.date],
    watch_candidates: List["Candidate"],
    bought_symbols_today: set[str],
    traded_symbols_today: set[str],
    strict_filtered_count: int,
    theme_selection_day: datetime.date | None,
    daily_trade_finished_day: datetime.date | None,
) -> None:
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
        "manual_watch_symbols": sorted({str(x).zfill(6) for x in manual_watch_symbols if str(x).isdigit()}),
        "entry_price": {str(k).zfill(6): float(v) for k, v in entry_price.items() if str(k).isdigit()},
        "peak_price": {str(k).zfill(6): float(v) for k, v in peak_price.items() if str(k).isdigit()},
        "entry_time": {str(k).zfill(6): v.isoformat() for k, v in entry_time.items() if isinstance(v, datetime)},
        "profit_take_stage": {str(k).zfill(6): int(v) for k, v in profit_take_stage.items() if str(k).isdigit()},
        "trading_paused": bool(trading_paused),
        "limit_up_hold_day": {str(k).zfill(6): v.isoformat() for k, v in limit_up_hold_day.items() if hasattr(v, "isoformat")},
        "watch_candidates": [candidate_to_state_row(candidate) for candidate in watch_candidates if str(candidate.symbol).isdigit()],
        "bought_symbols_today": sorted({str(x).zfill(6) for x in bought_symbols_today if str(x).isdigit()}),
        "traded_symbols_today": sorted({str(x).zfill(6) for x in traded_symbols_today if str(x).isdigit()}),
        "strict_filtered_count": max(0, int(strict_filtered_count)),
        "theme_selection_day": theme_selection_day.isoformat() if hasattr(theme_selection_day, "isoformat") else "",
        "daily_trade_finished_day": daily_trade_finished_day.isoformat() if hasattr(daily_trade_finished_day, "isoformat") else "",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _previous_business_day(ref_day: date) -> date:
    d = ref_day - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def fetch_minute_ohlcv(
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    symbol: str,
    count_hint: int = 180,
    market_code: str = DEFAULT_MINUTE_MARKET_CODE,
    ymd: str | None = None,
    cursor_time: str | None = None,
) -> List[Dict[str, float]]:
    url = f"{base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
    headers = {
        "authorization": f"Bearer {token}",
        "appKey": app_key,
        "appSecret": app_secret,
        "tr_id": "FHKST03010200",
    }
    now = datetime.now(KST)
    ymd = str(ymd or now.strftime("%Y%m%d"))
    cursor_time = str(cursor_time or now.strftime("%H%M%S"))
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


def ensure_chart_window_rows(
    rows: List[Dict[str, float]],
    *,
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    symbol: str,
    market_code: str,
    bar_minutes: int,
    min_rows: int = CHART_WINDOW_BARS,
    bar_cache: Dict[str, List[Dict[str, float]]] | None = None,
) -> List[Dict[str, float]]:
    if len(rows) >= min_rows:
        return rows

    def _merge_rows(extra_rows: List[Dict[str, float]], base_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
        merged_map: Dict[str, Dict[str, float]] = {}
        for row in extra_rows + base_rows:
            merged_map[str(int(row["date"]))] = row
        return [merged_map[k] for k in sorted(merged_map.keys())]
    # First, try to pad using locally cached bars (previous run / previous day).
    try:
        if bar_cache is not None:
            cached = list(bar_cache.get(symbol, []) or [])
            if cached:
                merged = _merge_rows(cached, rows)
                if len(merged) >= min_rows:
                    return merged[-min_rows:]
    except Exception:
        pass
    try:
        today = datetime.now(KST).date()
        prev_day = _previous_business_day(today)
        fetch_hint = max(180, raw_count_hint_for_resampled_bars(min_rows, bar_minutes))
        prev_rows_raw = fetch_minute_ohlcv(
            base_url=base_url,
            token=token,
            app_key=app_key,
            app_secret=app_secret,
            symbol=symbol,
            count_hint=fetch_hint,
            market_code=market_code,
            ymd=prev_day.strftime("%Y%m%d"),
            cursor_time="153000",
        )
        prev_rows = resample_bars(prev_rows_raw, bar_minutes=bar_minutes) if prev_rows_raw else []
        merged = _merge_rows(prev_rows, rows) if prev_rows else rows
        if len(merged) < min_rows:
            prev2_day = _previous_business_day(prev_day)
            prev2_rows_raw = fetch_minute_ohlcv(
                base_url=base_url,
                token=token,
                app_key=app_key,
                app_secret=app_secret,
                symbol=symbol,
                count_hint=fetch_hint,
                market_code=market_code,
                ymd=prev2_day.strftime("%Y%m%d"),
                cursor_time="153000",
            )
            prev2_rows = resample_bars(prev2_rows_raw, bar_minutes=bar_minutes) if prev2_rows_raw else []
            if prev2_rows:
                merged = _merge_rows(prev2_rows, merged)
        if len(merged) >= min_rows:
            return merged[-min_rows:]
        if bar_cache is not None:
            cached = list(bar_cache.get(symbol, []) or [])
            if cached:
                merged = _merge_rows(cached, merged)
                if len(merged) >= min_rows:
                    return merged[-min_rows:]
        return merged[-max(min_rows, len(merged)) :]
    except Exception:
        return rows


def _load_bar_cache(path: Path) -> Dict[str, List[Dict[str, float]]]:
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        out: Dict[str, List[Dict[str, float]]] = {}
        raw = payload.get("symbols", {})
        if not isinstance(raw, dict):
            return {}
        for sym, rows in raw.items():
            if not isinstance(sym, str) or not isinstance(rows, list):
                continue
            cleaned: List[Dict[str, float]] = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                try:
                    cleaned.append(
                        {
                            "date": float(r.get("date", 0.0)),
                            "open": float(r.get("open", 0.0)),
                            "high": float(r.get("high", 0.0)),
                            "low": float(r.get("low", 0.0)),
                            "close": float(r.get("close", 0.0)),
                            "volume": float(r.get("volume", 0.0)),
                        }
                    )
                except Exception:
                    continue
            if cleaned:
                out[sym.strip().zfill(6)] = cleaned
        return out
    except Exception:
        return {}


def _save_bar_cache(path: Path, cache: Dict[str, List[Dict[str, float]]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"), "symbols": cache}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        return


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


def load_chart_classifier_payload(model_path_text: str) -> Dict[str, object] | None:
    model_path = Path(str(model_path_text).strip()) if model_path_text else None
    if not model_path:
        return None
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    if not model_path.exists():
        return None
    try:
        with model_path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return None
        if "model" not in payload and "portable_model" not in payload:
            return None
        payload["model_path"] = str(model_path)
        return payload
    except Exception:
        return None


def inspect_chart_classifier_payload(model_path_text: str) -> Tuple[Dict[str, object] | None, str]:
    model_path = Path(str(model_path_text).strip()) if model_path_text else None
    if not model_path:
        return None, "모델경로없음"
    if not model_path.is_absolute():
        model_path = (ROOT / model_path).resolve()
    if not model_path.exists():
        return None, f"모델파일없음 | {model_path}"
    try:
        with model_path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return None, f"모델형식오류 | {model_path}"
        if "model" not in payload and "portable_model" not in payload:
            return None, f"모델키없음 | {model_path}"
        payload["model_path"] = str(model_path)
        return payload, f"로드완료 | {model_path}"
    except Exception as exc:
        return None, f"로드실패 {type(exc).__name__} | {model_path}"


def score_portable_chart_model(portable_model: Dict[str, object], x: np.ndarray) -> float:
    kind = str(portable_model.get("kind", "")).strip()
    if kind == "constant":
        label = int(portable_model.get("label", 0))
        return 1.0 if label == 1 else 0.0
    if kind == "logistic_raw":
        coef = np.asarray(portable_model.get("coef", []), dtype=np.float32)
        intercept = np.asarray(portable_model.get("intercept", []), dtype=np.float32)
        if coef.size == 0:
            return 0.0
        logits = x @ coef.T
        if intercept.size:
            logits = logits + intercept.reshape(1, -1)
        score = float(logits.reshape(-1)[0])
        return float(1.0 / (1.0 + np.exp(-score)))
    if kind == "pca_logistic":
        mean = np.asarray(portable_model.get("mean", []), dtype=np.float32)
        components = np.asarray(portable_model.get("components", []), dtype=np.float32)
        coef = np.asarray(portable_model.get("coef", []), dtype=np.float32)
        intercept = np.asarray(portable_model.get("intercept", []), dtype=np.float32)
        if mean.size == 0 or components.size == 0 or coef.size == 0:
            return 0.0
        x_centered = x - mean.reshape(1, -1)
        x_pca = x_centered @ components.T
        logits = x_pca @ coef.T
        if intercept.size:
            logits = logits + intercept.reshape(1, -1)
        score = float(logits.reshape(-1)[0])
        return float(1.0 / (1.0 + np.exp(-score)))
    return 0.0


def resolve_chart_model_path(primary_text: str, fallback_relpath: str) -> str:
    text = str(primary_text or "").strip()
    if text:
        return text
    fallback = (ROOT / fallback_relpath).resolve()
    if fallback.exists():
        return str(fallback)
    return ""


def resolve_payload_preferred_threshold(
    payload: Dict[str, object] | None,
    cli_value: float,
    default_value: float,
) -> float:
    resolved = float(cli_value)
    if abs(resolved - float(default_value)) > 1e-9:
        return resolved
    if not payload:
        return resolved
    try:
        preferred = float(payload.get("preferred_threshold", 0.0) or 0.0)
    except Exception:
        preferred = 0.0
    if preferred > 0:
        return preferred
    return resolved


def score_chart_classifier_bonus(
    rows: List[Dict[str, float]],
    payload: Dict[str, object] | None,
    *,
    symbol: str,
    cache_dir: Path,
    threshold: float,
    bonus_scale: float,
    bar_minutes: int,
) -> Tuple[float, float, str]:
    if not payload:
        return 0.0, 0.0, "차트점수=off"
    if len(rows) < CHART_WINDOW_BARS:
        return 0.0, 0.0, f"차트점수=warmup({len(rows)}/{CHART_WINDOW_BARS})"
    try:
        from user_only_strategy.build_chart_image_dataset import render_chart_png
    except Exception as e:
        return 0.0, 0.0, f"차트점수=import_fail({type(e).__name__})"

    model = payload.get("model")
    portable_model = payload.get("portable_model")
    image_size = int(payload.get("image_size", 96))
    window = rows[-CHART_WINDOW_BARS:]
    chart_rows = [
        {
            "date": _parse_bar_datetime(float(r["date"])).strftime("%Y-%m-%d %H:%M:%S"),
            "open": f"{float(r['open']):.6f}",
            "high": f"{float(r['high']):.6f}",
            "low": f"{float(r['low']):.6f}",
            "close": f"{float(r['close']):.6f}",
            "volume": f"{float(r.get('volume', 0.0)):.0f}",
        }
        for r in window
    ]
    cache_dir.mkdir(parents=True, exist_ok=True)
    image_path = cache_dir / f"{symbol}_{bar_minutes}m_latest.png"
    try:
        render_chart_png(chart_rows, image_path, 640, 640)
        img_arr = plt.imread(str(image_path))
        if img_arr.ndim == 3:
            if img_arr.shape[2] >= 3:
                rgb = img_arr[..., :3].astype(np.float32)
                img_arr = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
            else:
                img_arr = img_arr[..., 0].astype(np.float32)
        else:
            img_arr = img_arr.astype(np.float32)
        if img_arr.max() > 1.5:
            img_arr = img_arr / 255.0
        src_h, src_w = img_arr.shape[:2]
        y_idx = np.linspace(0, max(0, src_h - 1), image_size).astype(int)
        x_idx = np.linspace(0, max(0, src_w - 1), image_size).astype(int)
        resized = img_arr[np.ix_(y_idx, x_idx)]
        x = resized.reshape(1, -1).astype(np.float32)
        prob = 0.0
        if isinstance(portable_model, dict):
            prob = score_portable_chart_model(portable_model, x)
        elif hasattr(model, "predict_proba"):
            proba = model.predict_proba(x)[0]
            classes = getattr(model, "classes_", [])
            if len(classes) >= 2:
                try:
                    pos_idx = list(classes).index(1)
                    prob = float(proba[pos_idx])
                except Exception:
                    prob = float(proba[-1])
            elif len(proba) > 0:
                prob = float(proba[-1])
        else:
            pred = model.predict(x)[0]
            prob = 1.0 if int(pred) == 1 else 0.0
        thr = min(0.95, max(0.05, float(threshold)))
        scale = max(0.0, float(bonus_scale))
        bonus = max(0.0, (prob - thr) / max(1e-6, 1.0 - thr)) * scale
        return prob, bonus, f"차트점수={prob:.2f} 보너스={bonus:.1f}"
    except Exception as e:
        return 0.0, 0.0, f"차트점수=fail({type(e).__name__})"


def resolve_intraday_chart_threshold(
    hhmm: int,
    *,
    base_threshold: float,
    morning_threshold: float,
    afternoon_threshold: float,
    morning_end_hhmm: int,
    afternoon_start_hhmm: int,
) -> float:
    if hhmm <= int(morning_end_hhmm):
        return float(morning_threshold)
    if hhmm >= int(afternoon_start_hhmm):
        return float(afternoon_threshold)
    return float(base_threshold)


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
    stop_cb: Callable[[], bool] | None = None,
) -> Tuple[List[Candidate], List[ThemeGroup]]:
    analyzed: List[Tuple[Candidate, np.ndarray]] = []
    total = len(universe)
    for idx, (symbol, name) in enumerate(universe, start=1):
        if stop_cb and stop_cb():
            return [], []
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
    monitor_key = re.sub(r"\s+", "", raw)
    if monitor_key == "매매중지":
        return "pause_trading", []
    if monitor_key == "매매재개":
        return "resume_trading", []
    if monitor_key == "종목선정중지":
        return "cancel_select", []
    if monitor_key == "종목선정":
        return "select", []
    if monitor_key == "보유":
        return "holdings", []
    if "모니터" in monitor_key:
        return "status", []
    parts = [item.strip() for item in re.split(r"[,\n]+", raw) if item.strip()]
    stop_parts: List[str] = []
    buy_parts: List[str] = []
    sell_parts: List[str] = []
    watch_parts: List[str] = []
    for part in parts:
        compact = re.sub(r"\s+", "", part)
        if compact.endswith("중지"):
            target = re.sub(r"\s*중지\s*$", "", part).strip()
            if target:
                stop_parts.append(target)
        elif compact.endswith("매수"):
            target = re.sub(r"\s*매수\s*$", "", part).strip()
            if target:
                buy_parts.append(target)
        elif compact.endswith("매도"):
            target = re.sub(r"\s*매도\s*$", "", part).strip()
            if target:
                sell_parts.append(target)
        else:
            watch_parts.append(part)

    if stop_parts and not watch_parts:
        resolved_stop: List[Tuple[str, str]] = []
        for part in stop_parts:
            symbol, name = resolve_watch_symbol(part, known_name_map)
            if symbol:
                label = name if (name and not str(name).isdigit()) else part
                resolved_stop.append((symbol, label))
        if resolved_stop:
            return "unwatch", resolved_stop

    if buy_parts and not watch_parts and not stop_parts and not sell_parts:
        resolved_buy: List[Tuple[str, str]] = []
        for part in buy_parts:
            symbol, name = resolve_watch_symbol(part, known_name_map)
            if symbol:
                label = name if (name and not str(name).isdigit()) else part
                resolved_buy.append((symbol, label))
        if resolved_buy:
            return "buy", resolved_buy

    if sell_parts and not watch_parts and not stop_parts and not buy_parts:
        resolved_sell: List[Tuple[str, str]] = []
        for part in sell_parts:
            symbol, name = resolve_watch_symbol(part, known_name_map)
            if symbol:
                label = name if (name and not str(name).isdigit()) else part
                resolved_sell.append((symbol, label))
        if resolved_sell:
            return "sell", resolved_sell

    resolved_watch: List[Tuple[str, str]] = []
    for part in watch_parts:
        symbol, name = resolve_watch_symbol(part, known_name_map)
        if symbol:
            label = name if (name and not str(name).isdigit()) else part
            resolved_watch.append((symbol, label))
    if resolved_watch:
        return "watch", resolved_watch
    return "ignore", []


def format_holdings_report(
    *,
    base_url: str,
    token: str,
    app_key: str,
    app_secret: str,
    holdings: Dict[str, Dict[str, float | int | str]],
    minute_market_code: str,
    bar_minutes: int,
) -> str:
    if not holdings:
        return "현재 보유종목 | -"
    lines: List[str] = []
    for symbol, row in sorted(holdings.items(), key=lambda item: str(item[1].get("name", item[0]))):
        qty = int(row.get("qty", 0) or 0)
        if qty <= 0:
            continue
        name = display_name(str(row.get("name", "")).strip(), symbol)
        avg_price = float(row.get("avg_price", 0.0) or 0.0)
        current_price = avg_price
        try:
            rows = fetch_minute_ohlcv(
                base_url=base_url,
                token=token,
                app_key=app_key,
                app_secret=app_secret,
                symbol=symbol,
                count_hint=raw_count_hint_for_resampled_bars(2, max(1, int(bar_minutes))),
                market_code=minute_market_code,
            )
            bars = resample_bars(rows, bar_minutes=max(1, int(bar_minutes)))
            if bars:
                current_price = float(bars[-1]["close"])
            elif rows:
                current_price = float(rows[-1]["close"])
        except Exception:
            current_price = avg_price
        pnl = (current_price - avg_price) * qty if avg_price > 0 else 0.0
        pnl_text = f"{pnl:+,.0f}원"
        pnl_pct = ((current_price / avg_price) - 1.0) * 100.0 if avg_price > 0 else 0.0
        lines.append(
            f"{name} | 수익률 {pnl_pct:+.2f}% | 평가손익 {pnl_text} | 보유 {qty}주 | 매입단가 {avg_price:,.0f} | 현재가 {current_price:,.0f}"
        )
    return "\n".join(lines) if lines else "현재 보유종목 | -"


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


def _volume_ratio_from_rows(volume: np.ndarray, window: int = 20) -> Tuple[float, float]:
    if volume.shape[0] < 3:
        return 0.0, 0.0
    base_window = min(max(3, int(window)), volume.shape[0] - 1)
    vol_base = float(np.mean(volume[-base_window - 1 : -1])) if base_window >= 3 else float(np.mean(volume[:-1]))
    vol_ratio = (float(volume[-1]) / vol_base) if vol_base > 1e-9 else 0.0
    vol_cluster = float(np.sum(volume[-3:])) / max(1e-9, float(np.mean(volume[:-3])) * 3.0) if volume.shape[0] > 15 else vol_ratio
    return vol_ratio, vol_cluster


def _vwap_from_rows(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> float:
    v = np.maximum(volume, 0.0)
    vol_sum = float(np.sum(v))
    typical = (high + low + close) / 3.0
    return float(np.sum(typical * v) / vol_sum) if vol_sum > 1e-9 else float(np.mean(close))


def morning_phase_buy_signal_from_minute_bars(
    rows: List[Dict[str, float]],
    *,
    early_end_hhmm: int,
    min_gain_pct: float,
    volume_mult: float,
) -> Tuple[bool, str, float]:
    if len(rows) < 20:
        return False, "warmup", -1.0
    hhmm = _bar_hhmm(rows[-1])
    if hhmm > int(early_end_hhmm):
        return False, "after_morning", -1.0

    high = np.asarray([r["high"] for r in rows], dtype=float)
    low = np.asarray([r["low"] for r in rows], dtype=float)
    close = np.asarray([r["close"] for r in rows], dtype=float)
    volume = np.asarray([r["volume"] for r in rows], dtype=float)
    ma3 = sma(close, 3)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    plus_di, minus_di, adx = dmi_adx(high, low, close, period=14)
    imp, imp_signal, imp_hist = impulse_macd(high, low, close)
    if not all(np.isfinite(v[-1]) for v in (ma3, ma5, ma10, ma20)):
        return False, "ma_warmup", -1.0

    open0 = float(rows[0]["open"])
    gain = (float(close[-1]) / open0 - 1.0) if open0 > 1e-9 else 0.0
    day_high = float(np.max(high))
    high_keep = (float(close[-1]) / day_high) if day_high > 1e-9 else 0.0
    vwap = _vwap_from_rows(high, low, close, volume)
    vol_ratio, vol_cluster = _volume_ratio_from_rows(volume, window=20)
    recent_high = float(np.max(high[-6:-1])) if len(rows) >= 7 else float(np.max(high[:-1]))

    trend_ok = ma3[-1] >= ma5[-1] >= ma10[-1] * 0.998 and close[-1] >= ma5[-1] and close[-1] >= ma20[-1] * 0.995
    breakout_ok = float(close[-1]) >= recent_high * 0.999
    vwap_ok = float(close[-1]) >= vwap * 0.998
    volume_ok = vol_ratio >= max(1.15, float(volume_mult) * 0.85) or vol_cluster >= max(1.25, float(volume_mult) * 0.80)
    dmi_ok = (
        np.isfinite(plus_di[-1])
        and np.isfinite(minus_di[-1])
        and np.isfinite(adx[-1])
        and plus_di[-1] > minus_di[-1]
        and adx[-1] >= 12.0
    )
    impulse_ok = (
        np.isfinite(imp[-1])
        and np.isfinite(imp_signal[-1])
        and np.isfinite(imp_hist[-1])
        and imp[-1] >= imp_signal[-1]
        and imp_hist[-1] >= 0
    )

    ok = gain >= float(min_gain_pct) * 0.70 and trend_ok and breakout_ok and vwap_ok and volume_ok and (dmi_ok or impulse_ok) and high_keep >= 0.985
    reason = (
        f"오전 gain={gain*100:.2f}% breakout={breakout_ok} vwap={vwap_ok} "
        f"vol={volume_ok} dmi={dmi_ok} impulse={impulse_ok} 고점유지={high_keep*100:.1f}%"
    )
    if not ok:
        return False, reason, -1.0
    score = 150.0 + (gain * 1200.0) + (6.0 * vol_ratio) + (25.0 if breakout_ok else 0.0) + (15.0 if high_keep >= 0.99 else 0.0)
    return True, reason, score


def afternoon_phase_buy_signal_from_minute_bars(
    rows: List[Dict[str, float]],
    *,
    start_hhmm: int = 1300,
    end_hhmm: int = 1420,
) -> Tuple[bool, str, float]:
    if len(rows) < 30:
        return False, "warmup", -1.0
    hhmm = _bar_hhmm(rows[-1])
    if hhmm < int(start_hhmm) or hhmm >= int(end_hhmm):
        return False, "outside_afternoon", -1.0

    high = np.asarray([r["high"] for r in rows], dtype=float)
    low = np.asarray([r["low"] for r in rows], dtype=float)
    close = np.asarray([r["close"] for r in rows], dtype=float)
    volume = np.asarray([r["volume"] for r in rows], dtype=float)
    ma5 = sma(close, 5)
    ma10 = sma(close, 10)
    ma20 = sma(close, 20)
    ma60 = sma(close, 60)
    plus_di, minus_di, adx = dmi_adx(high, low, close, period=14)
    imp, imp_signal, imp_hist = impulse_macd(high, low, close)
    if not all(np.isfinite(v[-1]) for v in (ma5, ma10, ma20, ma60)):
        return False, "ma_warmup", -1.0

    vwap = _vwap_from_rows(high, low, close, volume)
    vol_ratio, vol_cluster = _volume_ratio_from_rows(volume, window=18)
    recent_pullback = float(np.min(low[-6:-1])) if len(rows) >= 7 else float(np.min(low[:-1]))
    recent_high = float(np.max(high[-10:-1])) if len(rows) >= 11 else float(np.max(high[:-1]))
    day_high = float(np.max(high))
    near_high = float(close[-1]) >= day_high * 0.985

    pullback_ok = recent_pullback >= ma20[-2] * 0.985 if np.isfinite(ma20[-2]) else False
    reclaim_ok = float(close[-1]) >= max(ma5[-1], ma10[-1], vwap * 0.998)
    breakout_ok = float(close[-1]) >= recent_high * 0.998
    trend_ok = ma5[-1] >= ma10[-1] >= ma20[-1] * 0.998 and ma20[-1] >= ma60[-1] * 0.99
    dmi_ok = np.isfinite(plus_di[-1]) and np.isfinite(minus_di[-1]) and plus_di[-1] >= minus_di[-1]
    impulse_ok = (
        np.isfinite(imp[-1])
        and np.isfinite(imp_signal[-1])
        and np.isfinite(imp_hist[-1])
        and np.isfinite(imp_hist[-2])
        and imp[-1] >= imp_signal[-1]
        and imp_hist[-1] >= imp_hist[-2]
    )
    volume_ok = vol_ratio >= 1.15 or vol_cluster >= 1.20
    adx_ok = np.isfinite(adx[-1]) and adx[-1] >= 12.0

    ok = pullback_ok and reclaim_ok and breakout_ok and trend_ok and volume_ok and (dmi_ok or impulse_ok or adx_ok) and near_high
    reason = (
        f"오후 pullback={pullback_ok} reclaim={reclaim_ok} breakout={breakout_ok} "
        f"trend={trend_ok} vol={volume_ok} near_high={near_high}"
    )
    if not ok:
        return False, reason, -1.0
    score = 135.0 + (20.0 if breakout_ok else 0.0) + (5.0 * vol_ratio) + (10.0 if near_high else 0.0)
    return True, reason, score


def timed_buy_signal_from_minute_bars(
    rows: List[Dict[str, float]],
    *,
    early_end_hhmm: int,
    min_gain_pct: float,
    volume_mult: float,
) -> Tuple[bool, str, float]:
    hhmm = _bar_hhmm(rows[-1]) if rows else 0
    morning_ok, morning_reason, morning_score = morning_phase_buy_signal_from_minute_bars(
        rows,
        early_end_hhmm=early_end_hhmm,
        min_gain_pct=min_gain_pct,
        volume_mult=volume_mult,
    )
    if morning_ok:
        return True, morning_reason, morning_score
    afternoon_ok, afternoon_reason, afternoon_score = afternoon_phase_buy_signal_from_minute_bars(rows)
    if afternoon_ok:
        return True, afternoon_reason, afternoon_score
    return False, f"시간대불일치({hhmm})", -1.0


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
        ts = datetime.now(KST).strftime("%H:%M:%S")
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
    p.add_argument("--sync-holdings", action=argparse.BooleanOptionalAction, default=True, help="sync account holdings to monitor manual buys")
    p.add_argument("--holdings-sync-interval-sec", type=int, default=30, help="how often to sync account holdings")
    p.add_argument("--theme-count", type=int, default=2, help="number of theme groups to keep")
    p.add_argument("--notify-theme-progress", action=argparse.BooleanOptionalAction, default=False, help="send verbose theme-selection progress messages")
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
    p.add_argument("--max-order-krw", type=float, default=float(os.getenv("MAX_ORDER_KRW", "4999999")), help="per-symbol order cap under cash-only sizing")
    p.add_argument("--sell-max-hold-min", type=int, default=180, help="time-based exit after holding minutes, 0 disables")
    p.add_argument("--log-file", default="logs/user_only_strategy_signals.txt")
    p.add_argument("--telegram-bot-token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""))
    p.add_argument("--telegram-chat-id", default=os.getenv("TELEGRAM_CHAT_ID", ""))
    p.add_argument("--telegram-command-poll-sec", type=int, default=3, help="telegram command polling interval")
    p.add_argument("--extra-symbols", default=os.getenv("NEXT_MARKET_SYMBOLS", ""), help="comma-separated extra symbols")
    p.add_argument("--extra-symbols-file", default=os.getenv("NEXT_MARKET_SYMBOLS_FILE", ""), help="extra symbols file path")
    p.add_argument("--symbol-name-file", default=os.getenv("KRX_SYMBOL_NAME_FILE", DEFAULT_SYMBOL_NAME_FILE))
    p.add_argument("--watch-state-file", default=os.getenv("WATCH_STATE_FILE", DEFAULT_WATCH_STATE_FILE))
    p.add_argument("--bar-cache-file", default=os.getenv("BAR_CACHE_FILE", DEFAULT_BAR_CACHE_FILE), help="local cache to pad early-session bars")
    p.add_argument("--chart-classifier-model", default=os.getenv("CHART_CLASSIFIER_MODEL", ""), help="optional buy image classifier pickle path")
    p.add_argument("--chart-classifier-threshold", type=float, default=float(os.getenv("CHART_CLASSIFIER_THRESHOLD", "0.50")))
    p.add_argument("--chart-classifier-threshold-morning", type=float, default=float(os.getenv("CHART_CLASSIFIER_THRESHOLD_MORNING", os.getenv("CHART_CLASSIFIER_THRESHOLD", "0.50"))))
    p.add_argument("--chart-classifier-threshold-afternoon", type=float, default=float(os.getenv("CHART_CLASSIFIER_THRESHOLD_AFTERNOON", str(min(0.95, float(os.getenv('CHART_CLASSIFIER_THRESHOLD', '0.50')) + 0.04)))))
    p.add_argument("--chart-classifier-bonus-scale", type=float, default=float(os.getenv("CHART_CLASSIFIER_BONUS_SCALE", "45.0")))
    p.add_argument("--sell-chart-classifier-model", default=os.getenv("SELL_CHART_CLASSIFIER_MODEL", ""), help="optional sell image classifier pickle path")
    p.add_argument("--sell-chart-classifier-threshold", type=float, default=float(os.getenv("SELL_CHART_CLASSIFIER_THRESHOLD", "0.55")))
    p.add_argument("--sell-chart-classifier-threshold-morning", type=float, default=float(os.getenv("SELL_CHART_CLASSIFIER_THRESHOLD_MORNING", os.getenv("SELL_CHART_CLASSIFIER_THRESHOLD", "0.55"))))
    p.add_argument("--sell-chart-classifier-threshold-afternoon", type=float, default=float(os.getenv("SELL_CHART_CLASSIFIER_THRESHOLD_AFTERNOON", str(max(0.05, float(os.getenv('SELL_CHART_CLASSIFIER_THRESHOLD', '0.55')) - 0.03)))))
    p.add_argument("--sell-chart-weight", type=float, default=float(os.getenv("SELL_CHART_WEIGHT", "0.70")))
    p.add_argument("--auto-selection-enabled", action=argparse.BooleanOptionalAction, default=True, help="auto start theme selection at --auto-selection-hhmm")
    p.add_argument("--auto-selection-hhmm", type=int, default=int(os.getenv("AUTO_SELECTION_HHMM", "801")))
    p.add_argument("--auto-selection-retry-min", type=int, default=int(os.getenv("AUTO_SELECTION_RETRY_MIN", "10")))
    p.add_argument("--chart-threshold-morning-end-hhmm", type=int, default=int(os.getenv("CHART_THRESHOLD_MORNING_END_HHMM", "1030")))
    p.add_argument("--chart-threshold-afternoon-start-hhmm", type=int, default=int(os.getenv("CHART_THRESHOLD_AFTERNOON_START_HHMM", "1300")))
    return p.parse_args()


def watch_preview(candidates: List["Candidate"], max_items: int = 12) -> str:
    if not candidates:
        return "-"
    names = []
    for c in candidates[:max_items]:
        names.append(display_name(c.name, c.symbol))
    return ", ".join(names)


def strip_chart_bonus(reason: str) -> str:
    text = str(reason or "").strip()
    if not text:
        return text
    text = re.sub(r"\s*보너스=[^|,)]+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def append_chart_reason(name: str, chart_reason: str, *, held: bool = False) -> str:
    reason = strip_chart_bonus(chart_reason)
    if not reason:
        return name
    label = "매도차트" if held else "매수차트"
    if reason.startswith("차트점수="):
        reason = f"{label}={reason.split('=', 1)[1]}"
    else:
        reason = f"{label}={reason}"
    return f"{name}({reason})"


def format_order_amount(price: float, qty: int) -> str:
    amount = max(0.0, float(price)) * max(0, int(qty))
    return f"{amount:,.0f}원"


def format_monitoring_groups(
    held_items: List[str],
    watch_items: List[str],
) -> str:
    held_text = ", ".join(held_items) if held_items else "-"
    watch_text = ", ".join(watch_items) if watch_items else "-"
    return f"보유 | {held_text} | 미보유 | {watch_text}"


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


def in_call_auction_window(now: datetime) -> bool:
    hhmm = now.hour * 100 + now.minute
    return now.weekday() < 5 and ((830 <= hhmm < 900) or (1520 <= hhmm < 1530))


def is_hard_market_closed(now: datetime) -> bool:
    # Paper/demo safeguard: after 15:30 KST do not send any orders.
    hhmm = now.hour * 100 + now.minute
    return now.weekday() < 5 and hhmm >= 1530


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


def near_buy_signal_from_minute_bars(rows: List[Dict[str, float]], adx_min: float) -> Tuple[bool, str, float]:
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

    stoch_near = (
        np.isfinite(st_k[-1])
        and np.isfinite(st_k[-2])
        and np.isfinite(st_d[-1])
        and st_k[-1] >= st_d[-1] - 3.0
        and st_k[-1] >= st_k[-2]
    )
    dmi_near = (
        np.isfinite(plus_di[-1])
        and np.isfinite(minus_di[-1])
        and plus_di[-1] >= minus_di[-1] * 0.92
    )
    adx_near = (
        np.isfinite(adx[-1])
        and np.isfinite(adx[-2])
        and adx[-1] >= max(12.0, float(adx_min) * 0.80)
        and adx[-1] >= adx[-2]
    )
    trend_near = (
        np.isfinite(ma20[-1])
        and np.isfinite(ma60[-1])
        and close[-1] >= ma20[-1] * 0.995
        and ma20[-1] >= ma60[-1] * 0.985
    )
    rsi_boll_near = (
        np.isfinite(rsi14[-1])
        and np.isfinite(mid[-1])
        and np.isfinite(lower[-1])
        and (
            (rsi14[-1] >= 43 and close[-1] >= lower[-1])
            or (close[-1] >= mid[-1] * 0.995)
        )
    )
    impulse_near = (
        np.isfinite(imp[-1])
        and np.isfinite(imp_signal[-1])
        and np.isfinite(imp_hist[-1])
        and np.isfinite(imp_hist[-2])
        and imp[-1] >= imp_signal[-1] - max(1e-6, abs(imp_signal[-1]) * 0.15)
        and imp_hist[-1] >= imp_hist[-2]
    )

    conditions = {
        "스토근접": stoch_near,
        "DMI근접": dmi_near,
        "ADX상승": adx_near,
        "추세유지": trend_near,
        "RSI볼린저": rsi_boll_near,
        "임펄스예열": impulse_near,
    }
    passed = [label for label, ok in conditions.items() if ok]
    ok = len(passed) >= 4 and stoch_near and dmi_near and trend_near
    if not ok:
        return False, " ".join(passed) if passed else "near-miss", -1.0
    score = float(len(passed)) * 10.0
    if np.isfinite(adx[-1]):
        score += float(adx[-1])
    if np.isfinite(st_k[-1]) and np.isfinite(st_d[-1]):
        score += max(0.0, float(st_k[-1] - st_d[-1]))
    return True, ",".join(passed), score


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


def sell_signal_from_minute_bars(rows: List[Dict[str, float]]) -> Tuple[bool, str, float]:
    if len(rows) < 70:
        return False, "warmup", 0.0
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
    ma20_rising = (
        np.isfinite(ma20[-1])
        and np.isfinite(ma20[-2])
        and ma20[-1] >= ma20[-2]
    )
    dmi_bull_alive = (
        np.isfinite(plus_di[-1])
        and np.isfinite(minus_di[-1])
        and plus_di[-1] > minus_di[-1] * 1.05
    )
    impulse_alive = (
        np.isfinite(imp[-1])
        and np.isfinite(imp_signal[-1])
        and np.isfinite(imp_hist[-1])
        and np.isfinite(imp_hist[-2])
        and imp[-1] >= imp_signal[-1]
        and imp_hist[-1] >= imp_hist[-2] * 0.85
    )
    strong_trend_hold = (
        np.isfinite(close[-1])
        and np.isfinite(ma20[-1])
        and close[-1] >= ma20[-1]
        and ma20_rising
        and dmi_bull_alive
        and impulse_alive
    )
    raw_sell = (impulse_dead and (stoch_dead or dmi_dead)) or (ma20_break and (rsi_mid_break or impulse_dead))
    ok = raw_sell and not strong_trend_hold
    score = 0.0
    score += 0.18 if stoch_dead else 0.0
    score += 0.22 if dmi_dead else 0.0
    score += 0.18 if impulse_dead else 0.0
    score += 0.18 if rsi_mid_break else 0.0
    score += 0.24 if ma20_break else 0.0
    if strong_trend_hold:
        score *= 0.35
    score = max(0.0, min(1.0, score))
    reason = (
        f"stoch_dead={stoch_dead} dmi_dead={dmi_dead} impulse_dead={impulse_dead} "
        f"rsi_mid_break={rsi_mid_break} ma20_break={ma20_break} strong_trend_hold={strong_trend_hold}"
    )
    return ok, reason, score


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
        message_prefix="",
    )
    notifier.send("봇기동/재기동")
    notifier.send(f"설정확인 | chart_window={CHART_WINDOW_BARS} | file={__file__}")
    buy_chart_model_path = resolve_chart_model_path(
        args.chart_classifier_model,
        "data/chart_models/live50_30d_final.pkl",
    )
    sell_chart_model_path = resolve_chart_model_path(
        args.sell_chart_classifier_model,
        "data/chart_models/live50_30d_sell_final.pkl",
    )
    chart_classifier_payload, buy_chart_model_status = inspect_chart_classifier_payload(buy_chart_model_path)
    sell_chart_classifier_payload, sell_chart_model_status = inspect_chart_classifier_payload(sell_chart_model_path)
    buy_chart_threshold_base = resolve_payload_preferred_threshold(
        chart_classifier_payload,
        float(args.chart_classifier_threshold),
        0.50,
    )
    buy_chart_threshold_morning = resolve_payload_preferred_threshold(
        chart_classifier_payload,
        float(args.chart_classifier_threshold_morning),
        0.50,
    )
    buy_chart_threshold_afternoon = resolve_payload_preferred_threshold(
        chart_classifier_payload,
        float(args.chart_classifier_threshold_afternoon),
        0.54,
    )
    sell_chart_threshold_base = resolve_payload_preferred_threshold(
        sell_chart_classifier_payload,
        float(args.sell_chart_classifier_threshold),
        0.55,
    )
    sell_chart_threshold_morning = resolve_payload_preferred_threshold(
        sell_chart_classifier_payload,
        float(args.sell_chart_classifier_threshold_morning),
        0.55,
    )
    sell_chart_threshold_afternoon = resolve_payload_preferred_threshold(
        sell_chart_classifier_payload,
        float(args.sell_chart_classifier_threshold_afternoon),
        0.52,
    )
    chart_classifier_cache_dir = ROOT / "logs" / "chart_classifier_cache_buy"
    sell_chart_classifier_cache_dir = ROOT / "logs" / "chart_classifier_cache_sell"
    if chart_classifier_payload:
        notifier.send(f"매수차트분류기 {buy_chart_model_status}")
    else:
        notifier.send(f"매수차트분류기 비활성 | {buy_chart_model_status}")
    if sell_chart_classifier_payload:
        notifier.send(f"매도차트분류기 {sell_chart_model_status}")
    else:
        notifier.send(f"매도차트분류기 비활성 | {sell_chart_model_status}")
    extra_symbols = parse_symbol_csv(args.extra_symbols)
    if args.extra_symbols_file:
        extra_symbols.extend(read_symbols_file(args.extra_symbols_file))
    if extra_symbols:
        extra_symbols = sorted(set(extra_symbols))
        notifier.send(f"추가후보 {len(extra_symbols)}개 반영")
    while True:
        try:
            token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
            break
        except Exception as exc:
            if not is_token_issue_rate_limited_error(exc):
                raise
            notifier.send("토큰발급제한 | 65초 후 자동재시도")
            sleep_with_telegram_poll(65)
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
    profit_take_stage: Dict[str, int] = {}
    trading_paused = False
    known_name_map: Dict[str, str] = {}
    manual_watch_symbols: set[str] = set()
    bought_symbols_today: set[str] = set()
    traded_symbols_today: set[str] = set()
    limit_up_hold_day: Dict[str, datetime.date] = {}
    theme_selection_day = None
    daily_trade_finished_day = None
    telegram_update_offset = 0
    last_telegram_poll_at: datetime | None = None
    manual_selection_requested = False
    next_auto_selection_at: datetime | None = None
    selection_cancel_requested = False
    last_trading_paused_notice_at: datetime | None = None
    last_slots_full_notice_at: datetime | None = None
    signal_first_seen_at: Dict[str, datetime] = {}
    last_near_buy_notice_at: Dict[str, datetime] = {}
    latest_chart_reason_by_symbol: Dict[str, str] = {}
    latest_sell_chart_reason_by_symbol: Dict[str, str] = {}
    net_err_streak = 0
    last_net_alert_at: datetime | None = None
    order_cooldown_until: Dict[str, datetime] = {}
    last_holdings_sync_at: datetime | None = None
    sell_auction_wait_notified: set[str] = set()
    last_auction_skip_notice_at: datetime | None = None
    last_close_skip_notice_at: datetime | None = None

    refreshed_symbol_names = refresh_symbol_name_map_from_krx(args.symbol_name_file)
    loaded_symbol_names = load_symbol_name_map(args.symbol_name_file)
    symbol_name_store: Dict[str, str] = dict(loaded_symbol_names)
    known_name_map.update(symbol_name_store)
    if refreshed_symbol_names > 0:
        notifier.send(f"종목명파일 갱신 {refreshed_symbol_names}개")
    elif symbol_name_store:
        notifier.send(f"종목명파일 로드 {len(symbol_name_store)}개")

    today_local = datetime.now(KST).date()
    if theme_selection_day != today_local:
        theme_selection_day = None
        strict_filtered_count = 0
        watch_candidates[:] = [candidate for candidate in watch_candidates if candidate.theme_name == "수동감시"]
    if daily_trade_finished_day != today_local:
        daily_trade_finished_day = None
    if entry_time:
        bought_symbols_today = {symbol for symbol, when in entry_time.items() if when.date() == today_local}
    else:
        bought_symbols_today.clear()

    def sync_holdings_from_account(now: datetime) -> None:
        nonlocal last_holdings_sync_at
        if args.dry_run or not bool(args.sync_holdings):
            return
        if not (args.cano and args.acnt_prdt_cd):
            return
        interval = max(5, int(args.holdings_sync_interval_sec))
        if last_holdings_sync_at is not None and (now - last_holdings_sync_at).total_seconds() < interval:
            return
        last_holdings_sync_at = now
        try:
            holdings = fetch_account_holdings(
                base_url=args.base_url,
                token=token,
                app_key=args.app_key,
                app_secret=args.app_secret,
                cano=args.cano,
                acnt_prdt_cd=args.acnt_prdt_cd,
            )
        except Exception:
            return
        stale_symbols = [
            symbol for symbol, qty in list(positions.items())
            if int(qty) > 0 and symbol not in holdings
        ]
        for symbol in stale_symbols:
            clear_position_state(symbol)
            sell_auction_wait_notified.discard(symbol)
        added_any = False
        for symbol, row in holdings.items():
            qty = _to_int(row.get("qty"))
            if qty <= 0:
                continue
            prev_qty = int(positions.get(symbol, 0))
            positions[symbol] = qty
            name = str(row.get("name", "")).strip()
            if name:
                known_name_map[symbol] = name
                symbol_name_store[symbol] = name
            avg_price = _to_float(row.get("avg_price"))
            if avg_price > 0 and entry_price.get(symbol, 0.0) <= 0:
                entry_price[symbol] = avg_price
            if avg_price > 0 and peak_price.get(symbol, 0.0) <= 0:
                peak_price[symbol] = avg_price
            if symbol not in entry_time:
                entry_time[symbol] = now
            if prev_qty <= 0:
                added_any = True
                restored = Candidate(
                    symbol=symbol,
                    name=known_name_map.get(symbol, name or symbol),
                    close=avg_price,
                    ma3=0.0,
                    ma5=0.0,
                    ma10=0.0,
                    ma20=0.0,
                    ma60=0.0,
                )
                restored.theme_id = 98
                restored.theme_name = "수동보유"
                if all(candidate.symbol != symbol for candidate in watch_candidates):
                    watch_candidates.append(restored)
                notifier.send(f"계좌보유 감지 | {display_name(restored.name, symbol)} {qty}주")
        if added_any or stale_symbols:
            persist_symbol_name_map()
            persist_runtime_state()
            notifier.send(f"현재 모니터링종목 | {monitoring_preview()}")

    def persist_symbol_name_map() -> None:
        merged = load_symbol_name_map(args.symbol_name_file)
        merged.update(symbol_name_store)
        merged.update({k: v for k, v in known_name_map.items() if str(k).isdigit() and str(v).strip() and not str(v).isdigit()})
        if merged:
            save_symbol_name_map(merged, args.symbol_name_file)
            symbol_name_store.clear()
            symbol_name_store.update(merged)
            known_name_map.update(merged)

    saved_state = load_watch_state(args.watch_state_file)
    bar_cache_path = Path(str(args.bar_cache_file)).expanduser()
    if not bar_cache_path.is_absolute():
        bar_cache_path = (ROOT / bar_cache_path).resolve()
    bar_cache: Dict[str, List[Dict[str, float]]] = _load_bar_cache(bar_cache_path)
    last_bar_cache_save_at: datetime | None = None
    manual_watch_symbols.update(
        {
            str(x).strip().zfill(6)
            for x in saved_state.get("manual_watch_symbols", [])
            if str(x).strip().isdigit() and 4 <= len(str(x).strip()) <= 6
        }
    )
    for symbol, price in dict(saved_state.get("entry_price", {})).items():
        if str(symbol).isdigit():
            entry_price[str(symbol).zfill(6)] = _to_float(price)
    for symbol, price in dict(saved_state.get("peak_price", {})).items():
        if str(symbol).isdigit():
            peak_price[str(symbol).zfill(6)] = _to_float(price)
    for symbol, value in dict(saved_state.get("entry_time", {})).items():
        if not str(symbol).isdigit():
            continue
        try:
            entry_time[str(symbol).zfill(6)] = datetime.fromisoformat(str(value))
        except Exception:
            continue
    for symbol, value in dict(saved_state.get("profit_take_stage", {})).items():
        if str(symbol).isdigit():
            profit_take_stage[str(symbol).zfill(6)] = max(0, _to_int(value))
    if isinstance(saved_state.get("trading_paused"), bool):
        trading_paused = bool(saved_state.get("trading_paused"))
    for symbol, value in dict(saved_state.get("limit_up_hold_day", {})).items():
        if not str(symbol).isdigit():
            continue
        try:
            limit_up_hold_day[str(symbol).zfill(6)] = datetime.fromisoformat(f"{value}T00:00:00").date()
        except Exception:
            continue
    for symbol in list(saved_state.get("bought_symbols_today", [])):
        if str(symbol).strip().isdigit():
            bought_symbols_today.add(str(symbol).strip().zfill(6))
    for symbol in list(saved_state.get("traded_symbols_today", [])):
        if str(symbol).strip().isdigit():
            traded_symbols_today.add(str(symbol).strip().zfill(6))
    strict_filtered_count = max(0, _to_int(saved_state.get("strict_filtered_count")))
    try:
        raw_theme_selection_day = str(saved_state.get("theme_selection_day", "")).strip()
        if raw_theme_selection_day:
            theme_selection_day = datetime.fromisoformat(f"{raw_theme_selection_day}T00:00:00").date()
    except Exception:
        theme_selection_day = None
    try:
        raw_daily_trade_finished_day = str(saved_state.get("daily_trade_finished_day", "")).strip()
        if raw_daily_trade_finished_day:
            daily_trade_finished_day = datetime.fromisoformat(f"{raw_daily_trade_finished_day}T00:00:00").date()
    except Exception:
        daily_trade_finished_day = None
    for row in list(saved_state.get("watch_candidates", [])):
        if not isinstance(row, dict):
            continue
        restored_candidate = candidate_from_state_row(row)
        if restored_candidate is None:
            continue
        watch_candidates.append(restored_candidate)
        if restored_candidate.name:
            known_name_map[restored_candidate.symbol] = restored_candidate.name

    def persist_runtime_state() -> None:
        save_watch_state(
            args.watch_state_file,
            manual_watch_symbols=manual_watch_symbols,
            entry_price=entry_price,
            peak_price=peak_price,
            entry_time=entry_time,
            profit_take_stage=profit_take_stage,
            trading_paused=trading_paused,
            limit_up_hold_day=limit_up_hold_day,
            watch_candidates=watch_candidates,
            bought_symbols_today=bought_symbols_today,
            traded_symbols_today=traded_symbols_today,
            strict_filtered_count=strict_filtered_count,
            theme_selection_day=theme_selection_day,
            daily_trade_finished_day=daily_trade_finished_day,
        )

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

    def send_trade_result_message(symbol: str, name: str, qty: int, price: float) -> None:
        ep = float(entry_price.get(symbol, 0.0) or 0.0)
        filled_qty = max(0, int(qty))
        if ep <= 0 or filled_qty <= 0:
            return
        pnl = (float(price) - ep) * filled_qty
        pnl_pct = ((float(price) / ep) - 1.0) * 100.0 if ep > 0 else 0.0
        notifier.send(
            f"매매손익 {display_name(name, symbol)} | 수익률 {pnl_pct:+.2f}% | 손익 {pnl:+,.0f}원 | 체결금액 {format_order_amount(price, filled_qty)}"
        )

    def prime_chart_cache(candidates: List[Candidate]) -> None:
        nonlocal last_bar_cache_save_at
        if not candidates:
            return
        updated = False
        for candidate in candidates:
            try:
                today_rows_raw = fetch_minute_ohlcv(
                    base_url=args.base_url,
                    token=token,
                    app_key=args.app_key,
                    app_secret=args.app_secret,
                    symbol=candidate.symbol,
                    count_hint=raw_count_hint_for_resampled_bars(CHART_WINDOW_BARS, args.bar_minutes),
                    market_code=args.minute_market_code,
                )
                today_rows = resample_bars(today_rows_raw, bar_minutes=args.bar_minutes)
                merged_rows = ensure_chart_window_rows(
                    today_rows,
                    base_url=args.base_url,
                    token=token,
                    app_key=args.app_key,
                    app_secret=args.app_secret,
                    symbol=candidate.symbol,
                    market_code=args.minute_market_code,
                    bar_minutes=int(args.bar_minutes),
                    min_rows=CHART_WINDOW_BARS,
                    bar_cache=bar_cache,
                )
                if len(merged_rows) >= CHART_WINDOW_BARS:
                    bar_cache[candidate.symbol] = merged_rows[-CHART_WINDOW_BARS:]
                    updated = True
            except Exception:
                continue
        if updated:
            _save_bar_cache(bar_cache_path, bar_cache)
            last_bar_cache_save_at = datetime.now(KST)

    def apply_blocked_universe(universe: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        if not blocked_unbuyable_symbols:
            return universe
        return [(s, n) for s, n in universe if s not in blocked_unbuyable_symbols]

    def is_daily_trade_finished(day_value: datetime.date) -> bool:
        return daily_trade_finished_day == day_value

    def finish_trading_day(reason: str) -> None:
        nonlocal strict_filtered_count, last_refresh, last_watch_report, daily_trade_finished_day, theme_selection_day
        watch_candidates[:] = []
        strict_filtered_count = 0
        last_refresh = datetime.now(KST)
        last_watch_report = None
        theme_selection_day = datetime.now(KST).date()
        daily_trade_finished_day = datetime.now(KST).date()
        notifier.send(f"당일마감 | {reason}")

    def schedule_reselection_if_needed(now_local: datetime) -> None:
        nonlocal manual_selection_requested, next_auto_selection_at
        holding_count = sum(1 for q in positions.values() if q > 0)
        if is_daily_trade_finished(now_local.date()):
            return
        if now_local.weekday() >= 5:
            return
        if not in_refresh_window(now_local, args.refresh_start_hhmm, args.refresh_end_hhmm):
            return
        slots_left = max(0, int(args.max_positions) - holding_count)
        if slots_left <= 0:
            return
        active_watch_symbols = {c.symbol for c in watch_candidates} | set(manual_watch_symbols)
        nonholding_watch_count = sum(1 for symbol in active_watch_symbols if positions.get(symbol, 0) <= 0)
        required_watch_count = max(1, slots_left)
        if nonholding_watch_count >= required_watch_count:
            return
        if manual_selection_requested and next_auto_selection_at is not None and next_auto_selection_at <= now_local:
            return
        manual_selection_requested = True
        next_auto_selection_at = now_local
        notifier.send(
            f"재선정예약 | 보유 {holding_count}/{int(args.max_positions)}종목 | 모니터링 {nonholding_watch_count}/{required_watch_count}"
        )

    def set_position_entry(symbol: str, qty: int, price: float) -> None:
        positions[symbol] = max(0, int(qty))
        entry_price[symbol] = float(price)
        peak_price[symbol] = float(price)
        entry_time[symbol] = datetime.now(KST)
        profit_take_stage[symbol] = 0
        bought_symbols_today.add(symbol)
        traded_symbols_today.add(symbol)
        signal_first_seen_at.pop(symbol, None)
        persist_runtime_state()

    def clear_position_state(symbol: str) -> None:
        positions[symbol] = 0
        traded_symbols_today.add(symbol)
        signal_first_seen_at.pop(symbol, None)
        entry_price.pop(symbol, None)
        peak_price.pop(symbol, None)
        entry_time.pop(symbol, None)
        profit_take_stage.pop(symbol, None)
        limit_up_hold_day.pop(symbol, None)
        latest_sell_chart_reason_by_symbol.pop(symbol, None)
        watch_candidates[:] = [candidate for candidate in watch_candidates if candidate.symbol != symbol]
        manual_watch_symbols.discard(symbol)
        persist_runtime_state()

    def finish_if_all_closed(carryover_exit: bool = False) -> None:
        if sum(1 for q in positions.values() if q > 0) == 0 and (
            len(bought_symbols_today) >= max(1, int(args.theme_count)) or carryover_exit
        ):
            finish_trading_day("보유 종목 청산 완료")

    def refresh_manual_candidate(symbol: str, name: str) -> Candidate | None:
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
            enriched = _candidate_from_bars(symbol, known_name_map.get(symbol, name), bars, leader_score, leader_reason)
            if enriched is not None:
                enriched.theme_id = 99
                enriched.theme_name = "수동감시"
            return enriched
        except Exception as exc:
            notifier.send(f"수동모니터링 즉시추가 실패 | {display_name(name, symbol)} | {type(exc).__name__}")
            return None

    def monitoring_preview() -> str:
        held_names: List[str] = []
        watch_names: List[str] = []
        for candidate in watch_candidates:
            nm = display_name(candidate.name, candidate.symbol)
            held = positions.get(candidate.symbol, 0) > 0
            chart_reason = (
                latest_sell_chart_reason_by_symbol.get(candidate.symbol, "")
                if held
                else latest_chart_reason_by_symbol.get(candidate.symbol, "")
            )
            nm = append_chart_reason(nm, chart_reason, held=held)
            target = held_names if held else watch_names
            if nm not in target:
                target.append(nm)
        for symbol in sorted(manual_watch_symbols):
            nm = display_name(known_name_map.get(symbol, symbol), symbol)
            held = positions.get(symbol, 0) > 0
            chart_reason = (
                latest_sell_chart_reason_by_symbol.get(symbol, "")
                if held
                else latest_chart_reason_by_symbol.get(symbol, "")
            )
            nm = append_chart_reason(nm, chart_reason, held=held)
            target = held_names if held else watch_names
            if nm not in target:
                target.append(nm)
        return format_monitoring_groups(held_names, watch_names)

    def refresh_chart_reasons_from_cache() -> None:
        # Off-session or right after restart, we may not have fresh per-symbol chart reasons.
        # Use the persisted bar_cache window to compute and attach the last known score.
        symbols: set[str] = {c.symbol for c in watch_candidates} | set(manual_watch_symbols)
        for symbol in sorted(symbols):
            rows = bar_cache.get(symbol) or []
            if not isinstance(rows, list) or len(rows) < CHART_WINDOW_BARS:
                continue
            held = positions.get(symbol, 0) > 0
            try:
                if held:
                    _p, _b, reason = score_chart_classifier_bonus(
                        rows,
                        sell_chart_classifier_payload,
                        symbol=symbol,
                        cache_dir=sell_chart_classifier_cache_dir,
                        threshold=sell_chart_threshold_base,
                        bonus_scale=10.0,
                        bar_minutes=int(args.bar_minutes),
                    )
                    if reason:
                        latest_sell_chart_reason_by_symbol[symbol] = reason
                else:
                    _p, _b, reason = score_chart_classifier_bonus(
                        rows,
                        chart_classifier_payload,
                        symbol=symbol,
                        cache_dir=chart_classifier_cache_dir,
                        threshold=buy_chart_threshold_base,
                        bonus_scale=float(args.chart_classifier_bonus_scale),
                        bar_minutes=int(args.bar_minutes),
                    )
                    if reason:
                        latest_chart_reason_by_symbol[symbol] = reason
            except Exception:
                continue

    def watch_preview_with_chart_scores(candidates: List[Candidate], max_items: int = 12) -> str:
        if not candidates:
            return "-"
        names: List[str] = []
        for c in candidates[:max_items]:
            nm = display_name(c.name, c.symbol)
            held = positions.get(c.symbol, 0) > 0
            chart_reason = (
                latest_sell_chart_reason_by_symbol.get(c.symbol, "")
                if held
                else latest_chart_reason_by_symbol.get(c.symbol, "")
            )
            names.append(append_chart_reason(nm, chart_reason, held=held))
        return ", ".join(names)

    def rebuild_manual_watch_candidates() -> List[Candidate]:
        rebuilt: List[Candidate] = []
        existing_by_symbol = {c.symbol: c for c in watch_candidates}
        for symbol in sorted(manual_watch_symbols):
            if len(rebuilt) >= max(1, int(args.max_watch_candidates)):
                break
            if symbol in existing_by_symbol:
                rebuilt.append(existing_by_symbol[symbol])
                continue
            nm = known_name_map.get(symbol, symbol)
            c = Candidate(
                symbol=symbol,
                name=nm,
                close=0.0,
                ma3=0.0,
                ma5=0.0,
                ma10=0.0,
                ma20=0.0,
                ma60=0.0,
            )
            c.theme_id = 99
            c.theme_name = "수동감시"
            rebuilt.append(c)
        return rebuilt

    def merge_manual_watch_candidates() -> None:
        for candidate in rebuild_manual_watch_candidates():
            if all(existing.symbol != candidate.symbol for existing in watch_candidates):
                watch_candidates.append(candidate)

    def drop_nonholding_after_close() -> None:
        keep_symbols = {s for s, q in positions.items() if q > 0}
        watch_candidates[:] = [c for c in watch_candidates if c.symbol in keep_symbols or c.symbol in manual_watch_symbols]
        persist_runtime_state()

    def maybe_notify_trading_paused(now_local: datetime) -> None:
        # Intentionally no-op: user requested only the explicit pause/resume messages.
        # Keep the function so call sites remain simple.
        nonlocal last_trading_paused_notice_at
        last_trading_paused_notice_at = now_local

    def maybe_notify_auction_skip(now_local: datetime) -> None:
        nonlocal last_auction_skip_notice_at
        if not in_call_auction_window(now_local):
            return
        if last_auction_skip_notice_at is not None and (now_local - last_auction_skip_notice_at).total_seconds() < 300:
            return
        notifier.send("동시호가구간 매매중지 | 주문 시도 안함")
        last_auction_skip_notice_at = now_local

    def maybe_notify_close_skip(now_local: datetime) -> None:
        nonlocal last_close_skip_notice_at
        if not is_hard_market_closed(now_local):
            return
        if last_close_skip_notice_at is not None and (now_local - last_close_skip_notice_at).total_seconds() < 300:
            return
        notifier.send("장마감 이후 매매중지 | 주문 시도 안함")
        last_close_skip_notice_at = now_local

    def poll_telegram_commands(now_local: datetime) -> None:
        nonlocal telegram_update_offset, last_telegram_poll_at, last_refresh, manual_selection_requested, selection_cancel_requested, next_auto_selection_at, trading_paused, last_trading_paused_notice_at
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
                refreshed_now = refresh_symbol_name_map_from_krx(args.symbol_name_file)
                if refreshed_now > 0:
                    symbol_name_store.update(load_symbol_name_map(args.symbol_name_file))
                    known_name_map.update(symbol_name_store)
                    action, payload = parse_telegram_watch_command(text, known_name_map)
            if action == "ignore":
                notifier.send(f"미인식입력 | {text}")
                continue
            if action == "select":
                manual_selection_requested = True
                selection_cancel_requested = False
                last_refresh = None
                notifier.send("종목선정 요청접수")
                continue
            if action == "pause_trading":
                trading_paused = True
                last_trading_paused_notice_at = None
                persist_runtime_state()
                notifier.send("매매중지 설정: 매수/매도 주문 중지")
                continue
            if action == "resume_trading":
                trading_paused = False
                persist_runtime_state()
                notifier.send("매매재개: 매수/매도 주문 재개")
                continue
            if action == "cancel_select":
                selection_cancel_requested = True
                manual_selection_requested = False
                next_auto_selection_at = None
                notifier.send("종목선정중지 요청접수")
                continue
            if action == "status":
                refresh_chart_reasons_from_cache()
                current_watch = monitoring_preview()
                notifier.send(f"현재 모니터링종목 | {current_watch}")
                continue
            if action == "holdings":
                sync_holdings_from_account(datetime.now(KST))
                try:
                    holdings = fetch_account_holdings(
                        base_url=args.base_url,
                        token=token,
                        app_key=args.app_key,
                        app_secret=args.app_secret,
                        cano=args.cano,
                        acnt_prdt_cd=args.acnt_prdt_cd,
                    )
                except Exception as exc:
                    notifier.send(f"보유조회실패 | {type(exc).__name__}")
                    continue
                notifier.send(
                    format_holdings_report(
                        base_url=args.base_url,
                        token=token,
                        app_key=args.app_key,
                        app_secret=args.app_secret,
                        holdings=holdings,
                        minute_market_code=args.minute_market_code,
                        bar_minutes=int(args.bar_minutes),
                    )
                )
                continue
            if action == "buy":
                for symbol, name in payload[:1]:
                    sync_holdings_from_account(datetime.now(KST))
                    if is_hard_market_closed(now_local):
                        maybe_notify_close_skip(now_local)
                        continue
                    if in_call_auction_window(now_local):
                        maybe_notify_auction_skip(now_local)
                        continue
                    holding_count = sum(1 for q in positions.values() if q > 0)
                    if positions.get(symbol, 0) <= 0 and holding_count >= int(args.max_positions):
                        notifier.send(f"수동매수거부 | 보유 {holding_count}/{int(args.max_positions)}종목 | {display_name(name, symbol)}")
                        continue
                    candidate = refresh_manual_candidate(symbol, name)
                    close = float(candidate.close) if candidate is not None else 0.0
                    if close <= 0:
                        notifier.send(f"수동매수실패 | 현재가 확인불가 | {display_name(name, symbol)}")
                        continue
                    if trading_paused:
                        maybe_notify_trading_paused(now_local)
                        continue
                    budget = compute_order_budget_krw(
                        args=args,
                        base_url=args.base_url,
                        token=token,
                        app_key=args.app_key,
                        app_secret=args.app_secret,
                        cano=args.cano,
                        acnt_prdt_cd=args.acnt_prdt_cd,
                    )
                    qty = int(max(0.0, budget) // max(1.0, close))
                    if qty <= 0:
                        notifier.send(f"수동매수실패 | 수량0 | {display_name(name, symbol)}")
                        continue
                    try:
                        max_qty = inquire_orderable_qty(
                            args.base_url,
                            token,
                            args.app_key,
                            args.app_secret,
                            args.cano,
                            args.acnt_prdt_cd,
                            symbol=symbol,
                            price=float(close),
                            qty_hint=qty,
                        )
                        if max_qty > 0:
                            qty = min(qty, max_qty)
                    except Exception:
                        # If psbl-order query fails, fall back to cash-only qty.
                        pass
                    attempted_qty = qty
                    while attempted_qty > 0:
                        try:
                            res = place_order(
                                base_url=args.base_url,
                                token=token,
                                app_key=args.app_key,
                                app_secret=args.app_secret,
                                cano=args.cano,
                                acnt_prdt_cd=args.acnt_prdt_cd,
                                symbol=symbol,
                                qty=attempted_qty,
                                side="buy",
                            )
                        except Exception as e:
                            if is_network_block_error(e):
                                wait_sec = notify_net_error(e)
                                time.sleep(wait_sec)
                                notifier.send(f"수동매수실패 | 네트워크 | {display_name(name, symbol)}")
                                attempted_qty = 0
                                break
                            raise
                        if str(res.get("rt_cd", "")) == "0" or not is_insufficient_order_cash_error(res):
                            qty = attempted_qty
                            break
                        attempted_qty -= 1
                    if attempted_qty <= 0:
                        notifier.send(f"수동매수실패 | 수량0 | {display_name(name, symbol)}")
                        continue
                    ok = str(res.get("rt_cd", "")) == "0"
                    in_auction = in_call_auction_window(now_local)
                    if ok:
                        odno = str(res.get("output", {}).get("ODNO", "")).strip()
                        notifier.send(
                            f"수동매수주문 {display_name(name, symbol)} {qty}주 {close:.0f}원 | 주문금액 {format_order_amount(close, qty)}"
                        )
                        sleep_with_telegram_poll(5)
                        status, filled_qty, ord_qty = wait_order_fill_status(
                            base_url=args.base_url,
                            token=token,
                            app_key=args.app_key,
                            app_secret=args.app_secret,
                            cano=args.cano,
                            acnt_prdt_cd=args.acnt_prdt_cd,
                            odno=odno,
                        )
                        if status in {"filled", "partial"} and filled_qty > 0:
                            set_position_entry(symbol, filled_qty, float(close))
                            manual_watch_symbols.add(symbol)
                            known_name_map[symbol] = name if name else symbol
                            if all(existing.symbol != symbol for existing in watch_candidates):
                                new_candidate = candidate or Candidate(symbol=symbol, name=name, close=close, ma3=0.0, ma5=0.0, ma10=0.0, ma20=0.0, ma60=0.0)
                                new_candidate.theme_id = 99
                                new_candidate.theme_name = "수동감시"
                                watch_candidates.append(new_candidate)
                            persist_runtime_state()
                            notifier.send(
                                f"수동매수체결 {display_name(name, symbol)} {filled_qty}/{max(ord_qty, qty)}주 | 체결금액 {format_order_amount(close, filled_qty)}"
                            )
                        else:
                            if in_auction:
                                notifier.send(f"수동매수동시호가대기 {display_name(name, symbol)} {filled_qty}/{max(ord_qty, qty)}주")
                            else:
                                notifier.send(f"수동매수미체결 {display_name(name, symbol)} {filled_qty}/{max(ord_qty, qty)}주")
                    else:
                        if in_auction:
                            notifier.send(f"수동매수동시호가대기 {display_name(name, symbol)} {qty}주 {close:.0f}원 | {format_kis_error(res)}")
                        else:
                            notifier.send(f"수동매수실패 {display_name(name, symbol)} {qty}주 {close:.0f}원 | {format_kis_error(res)}")
                continue
            if action == "sell":
                for symbol, name in payload[:1]:
                    sync_holdings_from_account(datetime.now(KST))
                    if is_hard_market_closed(now_local):
                        maybe_notify_close_skip(now_local)
                        continue
                    if in_call_auction_window(now_local):
                        maybe_notify_auction_skip(now_local)
                        continue
                    if trading_paused:
                        maybe_notify_trading_paused(now_local)
                        continue
                    qty = int(positions.get(symbol, 0))
                    if qty <= 0:
                        holdings = fetch_account_holdings(
                            base_url=args.base_url,
                            token=token,
                            app_key=args.app_key,
                            app_secret=args.app_secret,
                            cano=args.cano,
                            acnt_prdt_cd=args.acnt_prdt_cd,
                        )
                        qty = int((holdings.get(symbol) or {}).get("qty", 0))
                    if qty <= 0:
                        notifier.send(f"수동매도실패 | 보유수량없음 | {display_name(name, symbol)}")
                        continue
                    candidate = refresh_manual_candidate(symbol, name)
                    close = float(candidate.close) if candidate is not None else 0.0
                    try:
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
                    except Exception as e:
                        if is_network_block_error(e):
                            wait_sec = notify_net_error(e)
                            time.sleep(wait_sec)
                            notifier.send(f"수동매도실패 | 네트워크 | {display_name(name, symbol)}")
                            continue
                        raise
                    ok = str(res.get("rt_cd", "")) == "0"
                    in_auction = in_call_auction_window(now_local)
                    if ok:
                        odno = str(res.get("output", {}).get("ODNO", "")).strip()
                        notifier.send(
                            f"수동매도주문 {display_name(name, symbol)} {qty}주 {close:.0f}원 | 주문금액 {format_order_amount(close, qty)}"
                        )
                        sleep_with_telegram_poll(5)
                        status, filled_qty, ord_qty = wait_order_fill_status(
                            base_url=args.base_url,
                            token=token,
                            app_key=args.app_key,
                            app_secret=args.app_secret,
                            cano=args.cano,
                            acnt_prdt_cd=args.acnt_prdt_cd,
                            odno=odno,
                        )
                        remaining_qty = confirm_holding_qty(
                            base_url=args.base_url,
                            token=token,
                            app_key=args.app_key,
                            app_secret=args.app_secret,
                            cano=args.cano,
                            acnt_prdt_cd=args.acnt_prdt_cd,
                            symbol=symbol,
                        )
                        if remaining_qty == 0:
                            status = "filled"
                            filled_qty = max(filled_qty, qty)
                            ord_qty = max(ord_qty, qty)
                        if status in {"filled", "partial"} and filled_qty > 0:
                            remain = remaining_qty if remaining_qty >= 0 else max(0, qty - filled_qty)
                            if remain <= 0:
                                manual_watch_symbols.discard(symbol)
                                watch_candidates[:] = [c for c in watch_candidates if c.symbol != symbol]
                                clear_position_state(symbol)
                            else:
                                positions[symbol] = remain
                                persist_runtime_state()
                            notifier.send(
                                f"수동매도체결 {display_name(name, symbol)} {filled_qty}/{max(ord_qty, qty)}주 | 체결금액 {format_order_amount(close, filled_qty)}"
                            )
                            send_trade_result_message(symbol, name, filled_qty, close)
                            schedule_reselection_if_needed(now_local)
                        else:
                            if in_auction:
                                notifier.send(f"수동매도동시호가대기 {display_name(name, symbol)} {filled_qty}/{max(ord_qty, qty)}주")
                            else:
                                notifier.send(f"수동매도미체결 {display_name(name, symbol)} {filled_qty}/{max(ord_qty, qty)}주")
                    else:
                        if in_auction:
                            notifier.send(f"수동매도동시호가대기 {display_name(name, symbol)} {qty}주 | {format_kis_error(res)}")
                        else:
                            notifier.send(f"수동매도실패 {display_name(name, symbol)} {qty}주 | {format_kis_error(res)}")
                continue
            if action == "unwatch":
                for symbol, name in payload:
                    manual_watch_symbols.discard(symbol)
                    if positions.get(symbol, 0) > 0:
                        continue
                    watch_candidates[:] = [c for c in watch_candidates if c.symbol != symbol]
                    signal_first_seen_at.pop(symbol, None)
                persist_runtime_state()
                current_watch = monitoring_preview()
                notifier.send(f"현재 모니터링종목 | {current_watch}")
                schedule_reselection_if_needed(now_local)
                if not watch_candidates and not is_daily_trade_finished(now_local.date()):
                    last_refresh = None
                continue
            if action == "watch":
                added_names: List[str] = []
                for symbol, name in payload:
                    known_name_map[symbol] = name if name else symbol
                    if all(candidate.symbol != symbol for candidate in watch_candidates):
                        if len(watch_candidates) >= max(1, int(args.max_watch_candidates)):
                            notifier.send(f"모니터링가득참 | 최대 {int(args.max_watch_candidates)}개")
                            continue
                    manual_watch_symbols.add(symbol)
                    added_names.append(display_name(name, symbol))
                    if all(candidate.symbol != symbol for candidate in watch_candidates):
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
                        enriched = refresh_manual_candidate(symbol, name)
                        if enriched is not None:
                            for idx, existing in enumerate(watch_candidates):
                                if existing.symbol == symbol:
                                    watch_candidates[idx] = enriched
                                    break
                if added_names:
                    persist_symbol_name_map()
                    persist_runtime_state()
                    current_watch = monitoring_preview()
                    notifier.send(f"현재 모니터링종목 | {current_watch}")
                    if not watch_candidates and not is_daily_trade_finished(now_local.date()):
                        last_refresh = None

    def sleep_with_telegram_poll(total_sec: float) -> None:
        remain = max(0.0, float(total_sec))
        while remain > 0:
            chunk = min(1.0, remain)
            time.sleep(chunk)
            remain -= chunk
            try:
                poll_telegram_commands(datetime.now(KST))
            except Exception:
                pass

    if not args.dry_run and args.cano and args.acnt_prdt_cd:
        try:
            restored_holdings = fetch_account_holdings(
                base_url=args.base_url,
                token=token,
                app_key=args.app_key,
                app_secret=args.app_secret,
                cano=args.cano,
                acnt_prdt_cd=args.acnt_prdt_cd,
            )
        except Exception as exc:
            restored_holdings = {}
            notifier.send(f"보유종목복원 실패 | {type(exc).__name__}")
        if restored_holdings:
            for symbol, row in restored_holdings.items():
                qty = _to_int(row.get("qty"))
                if qty <= 0:
                    continue
                positions[symbol] = qty
                name = str(row.get("name", "")).strip()
                if name:
                    known_name_map[symbol] = name
                avg_price = _to_float(row.get("avg_price"))
                if avg_price > 0 and entry_price.get(symbol, 0.0) <= 0:
                    entry_price[symbol] = avg_price
                if avg_price > 0 and peak_price.get(symbol, 0.0) <= 0:
                    peak_price[symbol] = avg_price
                if symbol not in entry_time:
                    entry_time[symbol] = datetime.now(KST)
                if all(candidate.symbol != symbol for candidate in watch_candidates):
                    restored = Candidate(
                        symbol=symbol,
                        name=known_name_map.get(symbol, name or symbol),
                        close=avg_price,
                        ma3=0.0,
                        ma5=0.0,
                        ma10=0.0,
                        ma20=0.0,
                        ma60=0.0,
                    )
                    restored.theme_id = 98
                    restored.theme_name = "보유복원"
                    watch_candidates.append(restored)
            restored_names = [display_name(known_name_map.get(symbol, ""), symbol) for symbol in sorted(restored_holdings)]
            notifier.send(f"보유종목 복원 | {', '.join(restored_names)}")
            persist_symbol_name_map()
            persist_runtime_state()

    if manual_watch_symbols:
        merge_manual_watch_candidates()
        prime_chart_cache(watch_candidates)
        notifier.send(f"수동모니터링 복원 | {monitoring_preview()}")
        persist_runtime_state()

    boot_now = datetime.now(KST)
    boot_hhmm = boot_now.hour * 100 + boot_now.minute
    first_run_at_window_open = boot_hhmm < int(args.refresh_start_hhmm)
    suppress_initial_refresh_once = (
        (not first_run_at_window_open)
        and bool(watch_candidates or any(q > 0 for q in positions.values()) or theme_selection_day == boot_now.date())
    )

    for cycle in range(max(1, args.max_cycles)):
        now = datetime.now(KST)
        hhmm = now.hour * 100 + now.minute
        if active_session_day != now.date():
            first_cycle_boot = active_session_day is None
            active_session_day = now.date()
            close_notified_day = None
            theme_selection_day = None if not any(q > 0 for q in positions.values()) else theme_selection_day
            daily_trade_finished_day = None if daily_trade_finished_day != now.date() else daily_trade_finished_day
            bought_symbols_today.clear()
            traded_symbols_today.clear()
            blocked_unbuyable_symbols.clear()
            if not any(q > 0 for q in positions.values()):
                if first_cycle_boot:
                    merge_manual_watch_candidates()
                else:
                    watch_candidates[:] = rebuild_manual_watch_candidates()
                    strict_filtered_count = 0
        poll_telegram_commands(now)
        sync_holdings_from_account(now)
        schedule_reselection_if_needed(now)

        # Auto theme-selection schedule:
        # - Start at 08:01 (or configured) once per day.
        # - If no leader is found, retry every N minutes until success.
        if args.auto_selection_enabled and now.weekday() < 5:
            auto_hhmm = int(args.auto_selection_hhmm)
            retry_min = max(1, int(args.auto_selection_retry_min))
            has_active_monitoring = bool(
                any(q > 0 for q in positions.values())
                or any(positions.get(c.symbol, 0) <= 0 for c in watch_candidates)
                or any(positions.get(symbol, 0) <= 0 for symbol in manual_watch_symbols)
            )
            if theme_selection_day != now.date() and hhmm >= auto_hhmm and not has_active_monitoring:
                if next_auto_selection_at is None:
                    next_auto_selection_at = now
                if now >= next_auto_selection_at:
                    manual_selection_requested = True

        if manual_selection_requested and not in_refresh_window(now, args.refresh_start_hhmm, args.refresh_end_hhmm):
            if hhmm < int(args.refresh_start_hhmm):
                notifier.send(f"종목선정대기 | {int(args.refresh_start_hhmm):04d} 이후 실행")
            else:
                notifier.send(f"종목선정불가 | {int(args.refresh_end_hhmm):04d} 이후 종료")
                manual_selection_requested = False
        if in_refresh_window(now, args.refresh_start_hhmm, args.refresh_end_hhmm):
            holding_count = sum(1 for q in positions.values() if q > 0)
            slots_left = max(0, int(args.max_positions) - holding_count)
            need_refresh = bool(manual_selection_requested)
            if last_refresh is None and suppress_initial_refresh_once:
                # Apply startup-suppress only once.
                last_refresh = now
                need_refresh = False
                suppress_initial_refresh_once = False
            if slots_left <= 0:
                # When slots are full, do not waste API calls on reselection; keep monitoring for exits.
                if last_slots_full_notice_at is None or (now - last_slots_full_notice_at).total_seconds() >= 600:
                    notifier.send(
                        f"재선정생략 | 보유 {holding_count}/{int(args.max_positions)}종목 | 모니터링 최대 {int(args.max_watch_candidates)}종목 유지"
                    )
                    last_slots_full_notice_at = now
                last_refresh = now
                need_refresh = False
                manual_selection_requested = False
            if need_refresh:
                active_watch_symbols = {c.symbol for c in watch_candidates}
                nonholding_watch_count = sum(
                    1 for symbol in active_watch_symbols if positions.get(symbol, 0) <= 0
                )
                tracking_active = (
                    is_daily_trade_finished(now.date())
                    or (
                        theme_selection_day == now.date()
                        and nonholding_watch_count >= max(1, slots_left)
                    )
                    or (
                        bool(watch_candidates and strict_filtered_count > 0)
                        and nonholding_watch_count >= max(1, slots_left)
                    )
                )
                if tracking_active and not manual_selection_requested:
                    last_refresh = now
                    need_refresh = False
                    manual_selection_requested = False
                    notifier.send("재선정대기: 현재 모니터링 흐름 유지")
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
                if removed_cnt > 0 and args.notify_theme_progress:
                    notifier.send(f"제외종목 적용 {removed_cnt}개")
                if trimmed_cnt > 0 and args.notify_theme_progress:
                    notifier.send(f"후보풀 축약 {trimmed_cnt}개")
                if args.notify_theme_progress:
                    notifier.send(f"후보 {len(universe)}개")
                for symbol, name in universe:
                    if name:
                        known_name_map[symbol] = name
                if universe:
                    persist_symbol_name_map()
                if universe:
                    send_candidate_list_messages(
                        notifier,
                        universe,
                        chunk_size=25,
                        enabled=bool(args.notify_theme_progress),
                    )
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
                    if prev_stats_map and args.notify_theme_progress:
                        notifier.send(f"전일데이터 {len(prev_stats_map)}개 반영")
                if not in_korean_trading_session(now, args.market_open_hhmm, args.market_close_hhmm):
                    notifier.send("대장주판정대기: 08:00 이후 분봉 확인")
                    last_refresh = datetime.now(KST)
                    manual_selection_requested = False
                    first_run_at_window_open = False
                    suppress_initial_refresh_once = False
                    if cycle + 1 < args.max_cycles:
                        sleep_with_telegram_poll(max(1, int(args.scan_interval_sec)))
                    continue
                try:
                    progress_cb = None
                    if bool(args.notify_theme_progress):
                        progress_cb = lambda done, total, selected_cnt, cur: notifier.send(
                            f"테마진행 {done}/{total} | 분석 {selected_cnt} | {cur}"
                        )
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
                        progress_cb=progress_cb,
                        stop_cb=lambda: selection_cancel_requested,
                    )
                except Exception as e:
                    if is_network_block_error(e):
                        wait_sec = notify_net_error(e)
                        time.sleep(wait_sec)
                        continue
                    raise
                if selection_cancel_requested:
                    notifier.send("종목선정중지 완료")
                    selection_cancel_requested = False
                    manual_selection_requested = False
                    next_auto_selection_at = None
                    last_refresh = now
                    if cycle + 1 < args.max_cycles:
                        sleep_with_telegram_poll(max(1, int(args.scan_interval_sec)))
                    continue
                strict_filtered_count = len(leaders)
                net_err_streak = 0
                merged_watch: List[Candidate] = rebuild_manual_watch_candidates()
                if not leaders:
                    notifier.send("테마대장 선정 없음")
                    # Schedule next auto retry (if enabled).
                    if bool(args.auto_selection_enabled):
                        retry_min = max(1, int(args.auto_selection_retry_min))
                        next_auto_selection_at = now + timedelta(minutes=retry_min)
                else:
                    for leader in leaders:
                        if len(merged_watch) >= max(1, int(args.max_watch_candidates)):
                            break
                        if any(x.symbol == leader.symbol for x in merged_watch):
                            continue
                        merged_watch.append(leader)
                    theme_selection_day = now.date()
                    notifier.send(f"테마선정 {len(theme_groups)}개")
                    for group in theme_groups:
                        notifier.send(f"테마그룹 | {format_theme_group(group)}")
                    prime_chart_cache(merged_watch)
                    notifier.send(f"대장모니터링 {len(merged_watch)}개 | {watch_preview_with_chart_scores(merged_watch)}")
                    next_auto_selection_at = None
                watch_candidates = merged_watch
                persist_runtime_state()
                if watch_candidates:
                    notifier.send(f"모니터링등록 {len(watch_candidates)}개 | {watch_preview_with_chart_scores(watch_candidates)}")
                last_refresh = datetime.now(KST)
                manual_selection_requested = False
                first_run_at_window_open = False
                suppress_initial_refresh_once = False

        if (
            now.weekday() < 5
            and active_session_day == now.date()
            and close_notified_day != now.date()
            and hhmm >= int(args.market_close_hhmm)
        ):
            drop_nonholding_after_close()
            notifier.send("운영종료: 오늘 운용 종료")
            close_notified_day = now.date()

        if not in_korean_trading_session(now, args.market_open_hhmm, args.market_close_hhmm) or not watch_candidates:
            if cycle + 1 < args.max_cycles:
                sleep_with_telegram_poll(max(1, int(args.scan_interval_sec)))
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
            rows = ensure_chart_window_rows(
                rows,
                base_url=args.base_url,
                token=token,
                app_key=args.app_key,
                app_secret=args.app_secret,
                symbol=c.symbol,
                market_code=args.minute_market_code,
                bar_minutes=int(args.bar_minutes),
                min_rows=CHART_WINDOW_BARS,
                bar_cache=bar_cache,
            )
            # Keep a rolling cache of the latest window to reduce warmup at session open.
            try:
                if len(rows) >= CHART_WINDOW_BARS:
                    bar_cache[c.symbol] = rows[-CHART_WINDOW_BARS:]
                    if (
                        last_bar_cache_save_at is None
                        or (now - last_bar_cache_save_at).total_seconds() >= 60
                    ):
                        _save_bar_cache(bar_cache_path, bar_cache)
                        last_bar_cache_save_at = now
            except Exception:
                pass
            net_err_streak = 0
            close = rows[-1]["close"]
            has_pos = positions.get(c.symbol, 0) > 0
            if not has_pos:
                if c.symbol in traded_symbols_today:
                    signal_first_seen_at.pop(c.symbol, None)
                    continue
                chart_buy_threshold = resolve_intraday_chart_threshold(
                    now.hour * 100 + now.minute,
                    base_threshold=buy_chart_threshold_base,
                    morning_threshold=buy_chart_threshold_morning,
                    afternoon_threshold=buy_chart_threshold_afternoon,
                    morning_end_hhmm=int(args.chart_threshold_morning_end_hhmm),
                    afternoon_start_hhmm=int(args.chart_threshold_afternoon_start_hhmm),
                )
                chart_prob, chart_bonus, chart_reason = score_chart_classifier_bonus(
                    rows,
                    chart_classifier_payload,
                    symbol=c.symbol,
                    cache_dir=chart_classifier_cache_dir,
                    threshold=chart_buy_threshold,
                    bonus_scale=float(args.chart_classifier_bonus_scale),
                    bar_minutes=int(args.bar_minutes),
                )
                latest_chart_reason_by_symbol[c.symbol] = chart_reason
                chart_buy_ok = bool(
                    chart_classifier_payload
                    and "차트점수=" in chart_reason
                    and "off" not in chart_reason
                    and "warmup" not in chart_reason
                    and "fail(" not in chart_reason
                    and "import_fail" not in chart_reason
                    and chart_prob >= chart_buy_threshold
                )
                if not chart_buy_ok:
                    signal_first_seen_at.pop(c.symbol, None)
                    continue
                chosen_score = chart_prob * 100.0 + max(0.0, chart_bonus)
                chosen_tag = chart_reason
                signal_at = signal_first_seen_at.setdefault(c.symbol, now)
                signaled_this_cycle.add(c.symbol)
                buy_candidates.append((signal_at, chosen_score, c, float(close), chosen_tag))
            else:
                sell_ok, sell_reason, sell_signal_score = sell_signal_from_minute_bars(rows)
                ep = float(entry_price.get(c.symbol, close))
                pp = max(float(peak_price.get(c.symbol, ep)), float(close))
                peak_price[c.symbol] = pp
                profit_pct = ((float(close) / ep) - 1.0) if ep > 0 else 0.0
                current_stage = int(profit_take_stage.get(c.symbol, 0) or 0)
                if current_stage < 1 and profit_pct >= 0.05:
                    current_stage = 1
                    profit_take_stage[c.symbol] = 1
                    persist_runtime_state()
                stop_loss_ok = float(close) <= ep * (1.0 - max(0.0, float(args.sell_stop_loss_pct)))
                trail_activate_pct = max(0.0, float(args.sell_trailing_activate_pct))
                trail_stop_pct = max(0.0, float(args.sell_trailing_stop_pct))
                if profit_pct >= 0.04:
                    trail_stop_pct = max(trail_stop_pct, 0.03)
                elif profit_pct >= 0.025:
                    trail_stop_pct = max(trail_stop_pct, 0.025)
                trail_armed = pp >= ep * (1.0 + trail_activate_pct)
                trailing_ok = trail_armed and (float(close) <= pp * (1.0 - trail_stop_pct))
                profit_floor_pct = 0.03
                profit_floor_price = ep * (1.0 + profit_floor_pct)
                profit_floor_armed = current_stage >= 1
                profit_floor_ok = profit_floor_armed and float(close) <= profit_floor_price
                hold_min = -1
                hold_time_ok = False
                et = entry_time.get(c.symbol)
                if et is not None:
                    hold_min = int((now - et).total_seconds() // 60)
                    hold_time_ok = int(args.sell_max_hold_min) > 0 and hold_min >= int(args.sell_max_hold_min)
                if profit_pct >= 0.015 and not stop_loss_ok and not trailing_ok:
                    hold_time_ok = False
                chart_sell_threshold = resolve_intraday_chart_threshold(
                    now.hour * 100 + now.minute,
                    base_threshold=sell_chart_threshold_base,
                    morning_threshold=sell_chart_threshold_morning,
                    afternoon_threshold=sell_chart_threshold_afternoon,
                    morning_end_hhmm=int(args.chart_threshold_morning_end_hhmm),
                    afternoon_start_hhmm=int(args.chart_threshold_afternoon_start_hhmm),
                )
                chart_sell_prob, _chart_sell_bonus, chart_sell_reason = score_chart_classifier_bonus(
                    rows,
                    sell_chart_classifier_payload,
                    symbol=c.symbol,
                    cache_dir=sell_chart_classifier_cache_dir,
                    threshold=chart_sell_threshold,
                    bonus_scale=10.0,
                    bar_minutes=int(args.bar_minutes),
                )
                latest_sell_chart_reason_by_symbol[c.symbol] = chart_sell_reason
                chart_sell_ready = bool(
                    sell_chart_classifier_payload
                    and "차트점수=" in chart_sell_reason
                    and "off" not in chart_sell_reason
                    and "warmup" not in chart_sell_reason
                    and "fail(" not in chart_sell_reason
                    and "import_fail" not in chart_sell_reason
                )
                sell_chart_weight = max(0.0, min(1.0, float(args.sell_chart_weight)))
                indicator_sell_weight = max(0.0, 1.0 - sell_chart_weight)
                chart_sell_weak_threshold = min(0.95, chart_sell_threshold)
                chart_sell_strong_threshold = min(0.95, chart_sell_threshold + 0.18)
                chart_sell_strong = bool(chart_sell_ready and chart_sell_prob >= chart_sell_strong_threshold)
                chart_sell_weak = bool(chart_sell_ready and chart_sell_prob >= chart_sell_weak_threshold)
                chart_sell_neutral = bool(
                    chart_sell_ready
                    and not chart_sell_weak
                    and chart_sell_prob >= max(0.05, chart_sell_threshold - 0.08)
                )
                combined_sell_score = (chart_sell_prob * sell_chart_weight) + (sell_signal_score * indicator_sell_weight)
                weighted_sell_ok = bool(chart_sell_weak and combined_sell_score >= chart_sell_threshold)
                chart_indicator_sell = bool(chart_sell_weak and sell_ok and weighted_sell_ok)
                hold_time_sell = bool(hold_time_ok and chart_sell_weak)
                final_sell_ok = bool(
                    stop_loss_ok
                    or profit_floor_ok
                    or chart_sell_strong
                    or trailing_ok
                    or chart_indicator_sell
                    or hold_time_sell
                )
                reason_parts: List[str] = [
                    f"차트:{strip_chart_bonus(chart_sell_reason)}",
                    f"매도결합={combined_sell_score:.2f}(차트{sell_chart_weight:.2f}+지표{indicator_sell_weight:.2f})",
                ]
                if chart_sell_strong:
                    reason_parts.append(f"차트붕괴강신호({chart_sell_prob:.2f}>={chart_sell_strong_threshold:.2f})")
                elif chart_indicator_sell:
                    reason_parts.append(
                        f"차트약화+지표약화({chart_sell_prob:.2f}>={chart_sell_weak_threshold:.2f},지표={sell_signal_score:.2f})"
                    )
                elif chart_sell_neutral:
                    reason_parts.append("차트중립=유지")
                elif sell_ok:
                    reason_parts.append("지표약화단독=관찰")
                if sell_ok and (chart_sell_strong or chart_indicator_sell):
                    reason_parts.append(f"지표:{sell_reason}")
                if stop_loss_ok:
                    reason_parts.append(f"손절(entry={ep:.0f},close={float(close):.0f})")
                if profit_floor_ok:
                    reason_parts.append(
                        f"수익보존(entry={ep:.0f},peak={pp:.0f},floor={profit_floor_price:.0f},close={float(close):.0f})"
                    )
                if trailing_ok:
                    reason_parts.append(f"트레일링(entry={ep:.0f},peak={pp:.0f},close={float(close):.0f},trail={trail_stop_pct*100:.1f}%)")
                if hold_time_sell:
                    reason_parts.append(f"시간청산(hold={hold_min}m)")
                final_sell_reason = " | ".join(reason_parts) if reason_parts else "no-sell"
                if not final_sell_ok:
                    continue
                sync_holdings_from_account(now)
                qty = positions.get(c.symbol, 0)
                if qty <= 0:
                    qty = confirm_holding_qty(
                        base_url=args.base_url,
                        token=token,
                        app_key=args.app_key,
                        app_secret=args.app_secret,
                        cano=args.cano,
                        acnt_prdt_cd=args.acnt_prdt_cd,
                        symbol=c.symbol,
                    )
                    positions[c.symbol] = max(0, int(qty))
                if qty <= 0:
                    clear_position_state(c.symbol)
                    continue
                sell_qty = qty
                if current_stage < 1 and profit_pct >= 0.05 and qty >= 2:
                    sell_qty = max(1, qty // 2)
                elif profit_floor_ok:
                    sell_qty = qty
                carryover_exit = False
                if args.dry_run:
                    notifier.send(f"매도 {display_name(c.name, c.symbol)} {sell_qty}주 {close:.0f}원 (DRY) | {final_sell_reason}")
                    remain = max(0, qty - sell_qty)
                    if remain <= 0:
                        clear_position_state(c.symbol)
                        finish_if_all_closed(carryover_exit)
                    else:
                        positions[c.symbol] = remain
                        profit_take_stage[c.symbol] = max(1, current_stage)
                        persist_runtime_state()
                else:
                    if is_hard_market_closed(now):
                        maybe_notify_close_skip(now)
                        continue
                    if in_call_auction_window(now):
                        maybe_notify_auction_skip(now)
                        continue
                    if trading_paused:
                        maybe_notify_trading_paused(now)
                        continue
                    try:
                        res = place_order(
                            base_url=args.base_url,
                            token=token,
                            app_key=args.app_key,
                            app_secret=args.app_secret,
                            cano=args.cano,
                            acnt_prdt_cd=args.acnt_prdt_cd,
                            symbol=c.symbol,
                            qty=sell_qty,
                            side="sell",
                        )
                    except Exception as e:
                        if is_network_block_error(e):
                            wait_sec = notify_net_error(e)
                            time.sleep(wait_sec)
                            continue
                        raise
                    ok = str(res.get("rt_cd", "")) == "0"
                    in_auction = in_call_auction_window(now)
                    if ok:
                        odno = str(res.get("output", {}).get("ODNO", "")).strip()
                        notifier.send(
                            f"매도주문접수 {display_name(c.name, c.symbol)} {sell_qty}주 {close:.0f}원 | 주문금액 {format_order_amount(close, sell_qty)} | {final_sell_reason}"
                        )
                        sleep_with_telegram_poll(5)
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
                        remaining_qty = confirm_holding_qty(
                            base_url=args.base_url,
                            token=token,
                            app_key=args.app_key,
                            app_secret=args.app_secret,
                            cano=args.cano,
                            acnt_prdt_cd=args.acnt_prdt_cd,
                            symbol=c.symbol,
                        )
                        status, filled_qty, ord_qty = reconcile_sell_fill_status(
                            status=status,
                            filled_qty=filled_qty,
                            ord_qty=ord_qty,
                            expected_qty=sell_qty,
                            remaining_qty=remaining_qty,
                        )
                        if status == "filled":
                            sell_auction_wait_notified.discard(c.symbol)
                            notifier.send(
                                f"매도체결 {display_name(c.name, c.symbol)} {filled_qty}주 | 체결금액 {format_order_amount(close, filled_qty)}"
                            )
                            send_trade_result_message(c.symbol, c.name, filled_qty, close)
                            remain = remaining_qty if remaining_qty >= 0 else max(0, qty - filled_qty)
                            if remain <= 0:
                                clear_position_state(c.symbol)
                                schedule_reselection_if_needed(now)
                                finish_if_all_closed(carryover_exit)
                            else:
                                positions[c.symbol] = remain
                                profit_take_stage[c.symbol] = max(1, current_stage)
                                persist_runtime_state()
                        elif status == "partial":
                            sell_auction_wait_notified.discard(c.symbol)
                            notifier.send(
                                f"매도부분체결 {display_name(c.name, c.symbol)} {filled_qty}/{max(ord_qty, qty)}주 | 체결금액 {format_order_amount(close, filled_qty)}"
                            )
                            send_trade_result_message(c.symbol, c.name, filled_qty, close)
                            remain = remaining_qty if remaining_qty >= 0 else max(0, qty - max(0, filled_qty))
                            positions[c.symbol] = remain
                            profit_take_stage[c.symbol] = max(1, current_stage)
                            persist_runtime_state()
                            if remain <= 0:
                                clear_position_state(c.symbol)
                                schedule_reselection_if_needed(now)
                                finish_if_all_closed(carryover_exit)
                        else:
                            if in_auction:
                                if c.symbol not in sell_auction_wait_notified:
                                    notifier.send(f"매도동시호가대기 {display_name(c.name, c.symbol)} {filled_qty}/{max(ord_qty, qty)}주")
                                    sell_auction_wait_notified.add(c.symbol)
                            else:
                                sell_auction_wait_notified.discard(c.symbol)
                                notifier.send(f"매도미체결 {display_name(c.name, c.symbol)} {filled_qty}/{max(ord_qty, qty)}주")
                    else:
                        remaining_qty = confirm_holding_qty(
                            base_url=args.base_url,
                            token=token,
                            app_key=args.app_key,
                            app_secret=args.app_secret,
                            cano=args.cano,
                            acnt_prdt_cd=args.acnt_prdt_cd,
                            symbol=c.symbol,
                        )
                        status, filled_qty, ord_qty = reconcile_sell_fill_status(
                            status="pending",
                            filled_qty=0,
                            ord_qty=sell_qty,
                            expected_qty=sell_qty,
                            remaining_qty=remaining_qty,
                        )
                        if status == "filled":
                            sell_auction_wait_notified.discard(c.symbol)
                            notifier.send(
                                f"매도체결 {display_name(c.name, c.symbol)} {filled_qty}주 | 체결금액 {format_order_amount(close, filled_qty)}"
                            )
                            send_trade_result_message(c.symbol, c.name, filled_qty, close)
                            clear_position_state(c.symbol)
                            schedule_reselection_if_needed(now)
                            finish_if_all_closed(carryover_exit)
                        elif status == "partial":
                            sell_auction_wait_notified.discard(c.symbol)
                            remain = max(0, remaining_qty)
                            positions[c.symbol] = remain
                            profit_take_stage[c.symbol] = max(1, current_stage)
                            persist_runtime_state()
                            notifier.send(
                                f"매도부분체결 {display_name(c.name, c.symbol)} {filled_qty}/{max(ord_qty, qty)}주 | 체결금액 {format_order_amount(close, filled_qty)}"
                            )
                            send_trade_result_message(c.symbol, c.name, filled_qty, close)
                        elif in_auction:
                            if c.symbol not in sell_auction_wait_notified:
                                notifier.send(f"매도동시호가대기 {display_name(c.name, c.symbol)} {sell_qty}주 {close:.0f}원 | {format_kis_error(res)}")
                                sell_auction_wait_notified.add(c.symbol)
                        else:
                            sell_auction_wait_notified.discard(c.symbol)
                            notifier.send(f"매도실패 {display_name(c.name, c.symbol)} {sell_qty}주 {close:.0f}원")
        for c in watch_candidates:
            if c.symbol not in signaled_this_cycle and positions.get(c.symbol, 0) <= 0:
                signal_first_seen_at.pop(c.symbol, None)
        if is_daily_trade_finished(now.date()):
            if cycle + 1 < args.max_cycles:
                sleep_with_telegram_poll(max(1, int(args.scan_interval_sec)))
            continue
        if buy_candidates:
            holding_count = sum(1 for q in positions.values() if q > 0)
            slots_left = max(0, int(args.max_positions) - holding_count)
            buys_left = min(max(0, int(args.max_buys_per_scan)), slots_left)
            if buys_left <= 0:
                if last_slots_full_notice_at is None or (now - last_slots_full_notice_at).total_seconds() >= 300:
                    notifier.send(f"매수대기 | 보유 {holding_count}/{int(args.max_positions)}종목으로 모니터링만 진행")
                    last_slots_full_notice_at = now
                if cycle + 1 < args.max_cycles:
                    sleep_with_telegram_poll(max(1, int(args.scan_interval_sec)))
                continue
            if buys_left > 0:
                # Priority: symbols whose buy signal appeared earlier are traded first.
                ranked = sorted(buy_candidates, key=lambda x: (x[0], -x[1]))
                for signal_at, score, c, close, buy_tag in ranked[:buys_left]:
                    sync_holdings_from_account(now)
                    if is_hard_market_closed(now):
                        maybe_notify_close_skip(now)
                        continue
                    if in_call_auction_window(now):
                        maybe_notify_auction_skip(now)
                        continue
                    if trading_paused:
                        maybe_notify_trading_paused(now)
                        continue
                    holding_count = sum(1 for q in positions.values() if q > 0)
                    if positions.get(c.symbol, 0) <= 0 and holding_count >= int(args.max_positions):
                        continue
                    cooldown_until = order_cooldown_until.get(c.symbol)
                    if cooldown_until and now < cooldown_until:
                        continue
                    budget = compute_order_budget_krw(
                        args=args,
                        base_url=args.base_url,
                        token=token,
                        app_key=args.app_key,
                        app_secret=args.app_secret,
                        cano=args.cano,
                        acnt_prdt_cd=args.acnt_prdt_cd,
                    )
                    qty = int(max(0.0, budget) // max(1.0, close))
                    if qty <= 0:
                        notifier.send(f"매수스킵 {display_name(c.name, c.symbol)} 수량0")
                        continue
                    display_buy_tag = strip_chart_bonus(buy_tag)
                    if args.dry_run:
                        notifier.send(f"매수 {display_name(c.name, c.symbol)} {qty}주 {close:.0f}원 (DRY) | {display_buy_tag} | 신호시각:{signal_at.strftime('%H:%M:%S')}")
                        set_position_entry(c.symbol, qty, float(close))
                    else:
                        try:
                            max_qty = inquire_orderable_qty(
                                args.base_url,
                                token,
                                args.app_key,
                                args.app_secret,
                                args.cano,
                                args.acnt_prdt_cd,
                                symbol=c.symbol,
                                price=float(close),
                                qty_hint=qty,
                            )
                            if max_qty > 0:
                                qty = min(qty, max_qty)
                        except Exception:
                            pass
                        attempted_qty = qty
                        while attempted_qty > 0:
                            try:
                                res = place_order(
                                    base_url=args.base_url,
                                    token=token,
                                    app_key=args.app_key,
                                    app_secret=args.app_secret,
                                    cano=args.cano,
                                    acnt_prdt_cd=args.acnt_prdt_cd,
                                    symbol=c.symbol,
                                    qty=attempted_qty,
                                    side="buy",
                                )
                            except Exception as e:
                                if is_network_block_error(e):
                                    wait_sec = notify_net_error(e)
                                    time.sleep(wait_sec)
                                    attempted_qty = 0
                                    break
                                raise
                            if str(res.get("rt_cd", "")) == "0" or not is_insufficient_order_cash_error(res):
                                qty = attempted_qty
                                break
                            attempted_qty -= 1
                        if attempted_qty <= 0:
                            notifier.send(f"매수실패 {display_name(c.name, c.symbol)} 0주 {close:.0f}원 | {display_buy_tag} | 수량0")
                            continue
                        ok = str(res.get("rt_cd", "")) == "0"
                        in_auction = in_call_auction_window(now)
                        if ok:
                            odno = str(res.get("output", {}).get("ODNO", "")).strip()
                            notifier.send(
                                f"매수주문접수 {display_name(c.name, c.symbol)} {qty}주 {close:.0f}원 | 주문금액 {format_order_amount(close, qty)} | {display_buy_tag} | 신호시각:{signal_at.strftime('%H:%M:%S')}"
                            )
                            sleep_with_telegram_poll(5)
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
                            remaining_qty = confirm_holding_qty(
                                base_url=args.base_url,
                                token=token,
                                app_key=args.app_key,
                                app_secret=args.app_secret,
                                cano=args.cano,
                                acnt_prdt_cd=args.acnt_prdt_cd,
                                symbol=c.symbol,
                            )
                            if remaining_qty > 0:
                                status = "filled" if remaining_qty >= qty else "partial"
                                filled_qty = max(filled_qty, remaining_qty)
                                ord_qty = max(ord_qty, qty)
                            if status == "filled":
                                notifier.send(
                                    f"매수체결 {display_name(c.name, c.symbol)} {filled_qty}주 | 체결금액 {format_order_amount(close, filled_qty)} | {display_buy_tag}"
                                )
                                set_position_entry(c.symbol, filled_qty, float(close))
                            elif status == "partial":
                                notifier.send(
                                    f"매수부분체결 {display_name(c.name, c.symbol)} {filled_qty}/{max(ord_qty, qty)}주 | 체결금액 {format_order_amount(close, filled_qty)} | {display_buy_tag}"
                                )
                                if filled_qty > 0:
                                    set_position_entry(c.symbol, filled_qty, float(close))
                            else:
                                if in_auction:
                                    notifier.send(f"매수동시호가대기 {display_name(c.name, c.symbol)} {filled_qty}/{max(ord_qty, qty)}주 | {display_buy_tag}")
                                else:
                                    notifier.send(f"매수미체결 {display_name(c.name, c.symbol)} {filled_qty}/{max(ord_qty, qty)}주 | {display_buy_tag}")
                                order_cooldown_until[c.symbol] = now + timedelta(seconds=30)
                                notifier.send(f"재시도대기 {display_name(c.name, c.symbol)} | 30초 후 재확인")
                        else:
                            if is_rate_limited_error(res):
                                order_cooldown_until[c.symbol] = now + timedelta(seconds=15)
                                notifier.send(
                                    f"매수지연 {display_name(c.name, c.symbol)} | 호출제한 15초대기 | {display_buy_tag} | {format_kis_error(res)}"
                                )
                            else:
                                if in_auction:
                                    notifier.send(
                                        f"매수동시호가대기 {display_name(c.name, c.symbol)} {qty}주 {close:.0f}원 | {display_buy_tag} | {format_kis_error(res)}"
                                    )
                                else:
                                    notifier.send(
                                        f"매수실패 {display_name(c.name, c.symbol)} {qty}주 {close:.0f}원 | {display_buy_tag} | {format_kis_error(res)}"
                                    )
        if cycle + 1 < args.max_cycles:
            sleep_with_telegram_poll(max(1, int(args.scan_interval_sec)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            load_dotenv()
            args = parse_args()
            Notifier(
                Path(args.log_file),
                args.telegram_bot_token,
                args.telegram_chat_id,
                message_prefix="",
            ).send(f"봇중단 | {type(exc).__name__} | {exc}")
        except Exception:
            pass
        raise


