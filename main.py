from __future__ import annotations

import argparse
import json
import os
import time
import math
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List

import requests

from fetch_kis_daily import get_access_token
import strategy_runtime as rt


VTS_BASE_URL = "https://openapivts.koreainvestment.com:29443"
DEFAULT_SYMBOLS = [
    # Defense (5)
    "012450",  # Hanwha Aerospace
    "079550",  # LIG Nex1
    "047810",  # KAI
    "272210",  # Hanwha Systems
    "103140",  # Poongsan
    # Space (5)
    "099320",  # SATREC INITIATIVE
    "211270",  # AP Satellite
    "274090",  # Kenko Aerospace
    "214270",  # Genohco
    "271940",  # ILJIN Hysolus (aerospace supply-chain proxy)
]
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
    if args.strategy_mode == "ma_cross_level":
        return max(max(args.ma_a, args.ma_b) + 1, args.cross_level_window)
    return max(args.long, args.mom_window + 1, args.stoch_window + args.stoch_smooth)


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
    p = argparse.ArgumentParser(description="KIS realtime paper trader (multi-factor)")
    p.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="comma-separated symbols")
    p.add_argument("--interval-sec", type=int, default=0, help="main loop interval in seconds")
    p.add_argument("--bar-minutes", type=int, choices=[1, 3, 5], default=1, help="signal timeframe")
    p.add_argument("--startup-lookback-minutes", type=int, default=25, help="extra warmup minutes for startup/immediate trading")
    p.add_argument("--after-close-action", choices=["wait", "exit"], default="wait", help="behavior after market close")
    p.add_argument(
        "--strategy-mode",
        choices=["multi_factor", "ma_cross_level"],
        default="ma_cross_level",
        help="signal strategy mode",
    )

    # Signal parameters
    p.add_argument("--short", type=int, default=5, help="short SMA window")
    p.add_argument("--long", type=int, default=20, help="long SMA window")
    p.add_argument("--mom-window", type=int, default=12, help="momentum window")
    p.add_argument("--stoch-window", type=int, default=14, help="stochastic K window")
    p.add_argument("--stoch-smooth", type=int, default=3, help="stochastic D smoothing")
    p.add_argument("--entry-threshold", type=float, default=0.40, help="buy score threshold")
    p.add_argument("--exit-threshold", type=float, default=-0.28, help="sell score threshold")
    p.add_argument("--ma-a", type=int, default=6, help="A moving-average window")
    p.add_argument("--ma-b", type=int, default=26, help="B moving-average window")
    p.add_argument("--cross-level-window", type=int, default=120, help="lookback bars for [0,1] normalization")
    p.add_argument("--cross-buy-level", type=float, default=0.93, help="buy threshold for ma_cross_level")
    p.add_argument("--cross-sell-level", type=float, default=0.55, help="sell threshold for ma_cross_level")

    # Risk and execution
    p.add_argument("--cash-buffer-pct", type=float, default=0.12, help="keep this cash ratio unused")
    p.add_argument("--stop-loss-pct", type=float, default=0.008, help="stop-loss ratio")
    p.add_argument("--take-profit-pct", type=float, default=0.020, help="take-profit ratio")
    p.add_argument("--cooldown-bars", type=int, default=50, help="bars to wait after exit")
    p.add_argument("--entry-confirm-bars", type=int, default=4, help="consecutive buy signals required")
    p.add_argument("--exit-confirm-bars", type=int, default=4, help="consecutive sell signals required")
    p.add_argument("--min-hold-bars", type=int, default=3, help="minimum bars to hold before normal exit")
    p.add_argument("--fee-rate", type=float, default=0.0005, help="fee rate for sizing/pnl")
    p.add_argument("--paper-cash", type=float, default=10_000_000, help="fallback paper cash when balance inquiry fails")
    p.add_argument("--max-invested-pct", type=float, default=0.30, help="max total invested fraction of portfolio equity")
    p.add_argument("--max-positions", type=int, default=2, help="max number of concurrent positions")
    p.add_argument("--min-order-krw", type=float, default=250_000, help="skip buy when target order value is below this amount")
    p.add_argument("--retry", type=int, default=3, help="quote retry count")
    p.add_argument("--throttle-ms", type=int, default=800, help="delay between symbols in milliseconds")
    p.add_argument(
        "--no-buy-before-close-min",
        type=int,
        default=25,
        help="block new buys during last N minutes before market close",
    )
    p.add_argument("--no-buy-morning-start-hhmm", type=int, default=900, help="morning no-buy start (HHMM)")
    p.add_argument("--no-buy-morning-end-hhmm", type=int, default=1000, help="morning no-buy end (HHMM, exclusive)")
    p.add_argument("--dry-run", action="store_true", help="print only, no order requests")
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


def resample_ohlc(rows: List[Dict], bar_minutes: int) -> List[Dict[str, float]]:
    if bar_minutes == 1:
        out = []
        for r in rows:
            o = r.get("stck_oprc")
            h = r.get("stck_hgpr")
            l = r.get("stck_lwpr")
            c = r.get("stck_prpr")
            if not (o and h and l and c):
                continue
            out.append({"open": float(o), "high": float(h), "low": float(l), "close": float(c)})
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
        if key not in buckets:
            buckets[key] = {"open": float(o), "high": float(h), "low": float(l), "close": float(c)}
        else:
            b = buckets[key]
            b["high"] = max(b["high"], float(h))
            b["low"] = min(b["low"], float(l))
            b["close"] = float(c)
    return [buckets[k] for k in sorted(buckets.keys())]


def _stoch_k(highs: List[float], lows: List[float], closes: List[float], window: int) -> float | None:
    if len(closes) < window:
        return None
    hh = max(highs[-window:])
    ll = min(lows[-window:])
    if hh == ll:
        return 50.0
    return (closes[-1] - ll) / (hh - ll) * 100.0


def multi_factor_signal(
    ohlc: List[Dict[str, float]],
    short: int,
    long: int,
    mom_window: int,
    stoch_window: int,
    stoch_smooth: int,
    entry_threshold: float,
    exit_threshold: float,
) -> tuple[int, Dict[str, float]]:
    closes = [x["close"] for x in ohlc]
    highs = [x["high"] for x in ohlc]
    lows = [x["low"] for x in ohlc]
    need = max(long, mom_window + 1, stoch_window + stoch_smooth)
    if len(closes) < need:
        return 0, {"score": 0.0}

    short_sma = sum(closes[-short:]) / short
    long_sma = sum(closes[-long:]) / long
    trend = 1.0 if short_sma > long_sma else -1.0

    mom = (closes[-1] / closes[-1 - mom_window]) - 1.0
    mom_score = max(-1.0, min(1.0, mom / 0.01))

    k_vals: List[float] = []
    for i in range(stoch_smooth):
        end = len(closes) - i
        k = _stoch_k(highs[:end], lows[:end], closes[:end], stoch_window)
        if k is not None:
            k_vals.append(k)
    if not k_vals:
        return 0, {"score": 0.0}
    k_now = k_vals[0]
    d_now = sum(k_vals) / len(k_vals)
    stoch_bias = (50.0 - k_now) / 50.0
    cross = 0.3 if k_now > d_now else -0.3

    score = 0.45 * trend + 0.35 * mom_score + 0.15 * stoch_bias + 0.05 * cross
    if score >= entry_threshold:
        signal = 1
    elif score <= exit_threshold:
        signal = -1
    else:
        signal = 0
    return signal, {
        "score": score,
        "mom": mom,
        "k": k_now,
        "d": d_now,
        "short_sma": short_sma,
        "long_sma": long_sma,
    }


def ma_cross_level_signal(
    ohlc: List[Dict[str, float]],
    ma_a: int,
    ma_b: int,
    level_window: int,
    buy_level: float,
    sell_level: float,
) -> tuple[int, Dict[str, float]]:
    if ma_a <= 0 or ma_b <= 0:
        return 0, {"score": 0.0}
    short = min(ma_a, ma_b)
    long = max(ma_a, ma_b)
    need = max(long + 1, level_window)

    closes = [x["close"] for x in ohlc]
    if len(closes) < need:
        return 0, {"score": 0.0}

    short_now = sum(closes[-short:]) / short
    long_now = sum(closes[-long:]) / long
    short_prev = sum(closes[-1 - short : -1]) / short
    long_prev = sum(closes[-1 - long : -1]) / long
    spread_prev = short_prev - long_prev
    spread_now = short_now - long_now

    recent = closes[-level_window:]
    lo = min(recent)
    hi = max(recent)
    if hi == lo:
        level = 0.5
    else:
        level = (closes[-1] - lo) / (hi - lo)
    level = max(0.0, min(1.0, level))

    cross_up = spread_prev <= 0 and spread_now > 0
    cross_down = spread_prev >= 0 and spread_now < 0
    if cross_up and level >= buy_level:
        signal = 1
    elif cross_down and level <= sell_level:
        signal = -1
    else:
        signal = 0
    spread_pct = 0.0 if long_now == 0 else (spread_now / long_now) * 100.0
    return signal, {"score": level, "ma_short": short_now, "ma_long": long_now, "spread_pct": spread_pct}


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
    tr_id = "VTTC0802U" if side == "buy" else "VTTC0801U"
    body = {
        "CANO": cano,
        "ACNT_PRDT_CD": acnt_prdt_cd,
        "PDNO": symbol,
        "ORD_DVSN": "01",
        "ORD_QTY": str(qty),
        "ORD_UNPR": "0",
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
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            # Retry only for transient classes: 5xx or network/timeout.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status < 500:
                raise
            if i < 3:
                time.sleep(0.6 * (i + 1))
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
    # VTS cash balance inquiry. Field names can vary by account type, so parse defensively.
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
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("rt_cd") != "0":
        raise RuntimeError(f"balance inquiry failed: {data.get('msg1', '')}")
    out2 = data.get("output2", {})
    candidates: List[Dict] = []
    if isinstance(out2, dict):
        candidates = [out2]
    elif isinstance(out2, list):
        candidates = [x for x in out2 if isinstance(x, dict)]

    # Some responses keep cash fields in output1 as well.
    out1 = data.get("output1", {})
    if isinstance(out1, dict):
        candidates.append(out1)
    elif isinstance(out1, list):
        candidates.extend([x for x in out1 if isinstance(x, dict)])

    # Only use cash-like fields. Do not use total valuation fields.
    for obj in candidates:
        for key in ("ord_psbl_cash", "dnca_tot_amt", "prvs_rcdl_excc_amt"):
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
    r = requests.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
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
    entry_streak: Dict[str, int] = {s: 0 for s in symbols}
    exit_streak: Dict[str, int] = {s: 0 for s in symbols}
    realized_pnl_krw: Dict[str, float] = {s: 0.0 for s in symbols}
    last_price: Dict[str, float] = {s: 0.0 for s in symbols}
    last_processed_bar: Dict[str, str] = {s: "" for s in symbols}
    paper_cash = args.paper_cash
    last_known_cash: float | None = None
    predicted_cash: float | None = None
    cash_fail_streak = 0
    base_log_path = Path(args.log_file)
    base_log_path.parent.mkdir(parents=True, exist_ok=True)
    signal_need = rt.required_bars_for_signal(args)
    warmup_bars = max(1, math.ceil(args.startup_lookback_minutes / max(1, args.bar_minutes)))
    count_hint = max(signal_need + warmup_bars, signal_need + 5, 40)

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
        f"model={args.strategy_mode}(short={args.short},long={args.long},mom={args.mom_window},"
        f"stoch={args.stoch_window}/{args.stoch_smooth},ma_a={args.ma_a},ma_b={args.ma_b}) "
        f"count_hint={count_hint}",
        save=True,
    )
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
            emit(f"initial_orderable_cash=ERROR ({str(e)[:200]})", save=True)
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
                emit(f"initial_positions=UNKNOWN ({str(pos_e)[:120]})", save=True)

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

                ohlc = resample_ohlc(bars, args.bar_minutes)
                if not ohlc:
                    emit(f"[{ts}] {symbol} no data")
                    continue
                if args.strategy_mode == "ma_cross_level":
                    signal, m = rt.ma_cross_level_signal(
                        ohlc=ohlc,
                        ma_a=args.ma_a,
                        ma_b=args.ma_b,
                        level_window=args.cross_level_window,
                        buy_level=args.cross_buy_level,
                        sell_level=args.cross_sell_level,
                    )
                else:
                    signal, m = rt.multi_factor_signal(
                        ohlc=ohlc,
                        short=args.short,
                        long=args.long,
                        mom_window=args.mom_window,
                        stoch_window=args.stoch_window,
                        stoch_smooth=args.stoch_smooth,
                        entry_threshold=args.entry_threshold,
                        exit_threshold=args.exit_threshold,
                    )
                buy_reason = "multi_factor_signal" if args.strategy_mode == "multi_factor" else "ma_cross_crossup_level"
                sell_reason = "signal"
                cross_info = ""
                if args.strategy_mode == "ma_cross_level":
                    cross_info = f" gap={m.get('spread_pct', 0.0):+.2f}%"
                last_px = ohlc[-1]["close"]
                last_price[symbol] = last_px
                bar_ts = ""
                if bars:
                    d = bars[-1].get("stck_bsop_date", "")
                    t = bars[-1].get("stck_cntg_hour", "")
                    if len(d) == 8 and len(t) == 6:
                        bar_ts = f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
                        cycle_bar_hms = f"{t[:2]}:{t[2:4]}:{t[4:6]}"
                if bar_ts and last_processed_bar[symbol] == bar_ts:
                    time.sleep(max(0, args.throttle_ms) / 1000.0)
                    continue
                if bar_ts:
                    last_processed_bar[symbol] = bar_ts
                ignore_close_window_signal = False
                if bars:
                    t = str(bars[-1].get("stck_cntg_hour", ""))
                    if len(t) == 6:
                        hhmm = int(t[:4])
                        if 1520 <= hhmm <= 1529:
                            ignore_close_window_signal = True
                if ignore_close_window_signal:
                    # Ignore noisy close-auction minutes (15:20~15:29) for strategy signals.
                    signal = 0

                st = rt.SymbolState(
                    has_position=has_position[symbol],
                    held_qty=held_qty[symbol],
                    entry_price=entry_price[symbol],
                    entry_total_cost=entry_total_cost[symbol],
                    held_bars=held_bars[symbol],
                    cooldown_left=cooldown_left[symbol],
                    entry_streak=entry_streak[symbol],
                    exit_streak=exit_streak[symbol],
                )
                transition = rt.evaluate_state_transition(
                    state=st,
                    signal=signal,
                    last_px=last_px,
                    strategy_mode=args.strategy_mode,
                    stop_loss_pct=args.stop_loss_pct,
                    take_profit_pct=args.take_profit_pct,
                    entry_confirm_bars=args.entry_confirm_bars,
                    exit_confirm_bars=args.exit_confirm_bars,
                    min_hold_bars=args.min_hold_bars,
                    hard_liquidation_window=hard_liquidation_window,
                    ease_sell_window=ease_sell_window,
                    ease_ratio=ease_ratio,
                )
                signal = int(transition["signal"])
                sell_reason = str(transition["sell_reason"])
                has_position[symbol] = st.has_position
                held_qty[symbol] = st.held_qty
                entry_price[symbol] = st.entry_price
                entry_total_cost[symbol] = st.entry_total_cost
                held_bars[symbol] = st.held_bars
                cooldown_left[symbol] = st.cooldown_left
                entry_streak[symbol] = st.entry_streak
                exit_streak[symbol] = st.exit_streak

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
                    entry_wait = transition.get("entry_wait")
                    if entry_wait:
                        cur, req = entry_wait
                        cycle_summary.append(
                            f"{symbol} score={m.get('score', 0):.2f}{cross_info} "
                            f"cd={cooldown_left[symbol]} e={cur}/{req}"
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
                            f"score={m.get('score', 0):.2f} mom={m.get('mom', 0)*100:.2f}% "
                            f"k={m.get('k', 0):.1f} d={m.get('d', 0):.1f} "
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
                        entry_streak[symbol] = 0
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
                    hold_wait = transition.get("hold_wait")
                    if hold_wait:
                        cur, req = hold_wait
                        cycle_summary.append(
                            f"{symbol} score={m.get('score', 0):.2f}{cross_info} "
                            f"cd={cooldown_left[symbol]} h={cur}/{req}"
                        )
                        time.sleep(max(0, args.throttle_ms) / 1000.0)
                        continue
                    exit_wait = transition.get("exit_wait")
                    if exit_wait:
                        cur, req = exit_wait
                        cycle_summary.append(
                            f"{symbol} score={m.get('score', 0):.2f}{cross_info} "
                            f"cd={cooldown_left[symbol]} x={cur}/{req}"
                        )
                        time.sleep(max(0, args.throttle_ms) / 1000.0)
                        continue

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
                    exit_streak[symbol] = 0
                    cooldown_left[symbol] = args.cooldown_bars
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
                        exit_streak[symbol] = 0
                        cooldown_left[symbol] = args.cooldown_bars
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
