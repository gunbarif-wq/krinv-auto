from __future__ import annotations

import argparse
import json
import os
import time
import math
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Any

import requests
import numpy as np

from fetch_kis_daily import get_access_token
from build_ml_dataset import build_rows


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
    return max(80, int(args.ml_feature_warmup_bars))


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
    p = argparse.ArgumentParser(description="KIS realtime paper trader (ML-only)")
    p.add_argument("--symbols", default="047810", help="comma-separated symbols")
    p.add_argument("--interval-sec", type=int, default=0, help="main loop interval in seconds")
    p.add_argument("--bar-minutes", type=int, choices=[1, 3, 5], default=1, help="signal timeframe")
    p.add_argument("--startup-lookback-minutes", type=int, default=25, help="extra warmup minutes for startup/immediate trading")
    p.add_argument("--after-close-action", choices=["wait", "exit"], default="wait", help="behavior after market close")
    p.add_argument(
        "--strategy-mode",
        choices=["ml_alpha"],
        default="ml_alpha",
        help="signal strategy mode",
    )

    # ML signal parameters
    p.add_argument("--ml-model-path", default="data/ml/047810/047810_model.pkl", help="trained ML model bundle path")
    p.add_argument("--ml-threshold", type=float, default=0.96, help="entry threshold for ML alpha/prob score")
    p.add_argument("--ml-signal-mode", choices=["alpha", "prob"], default="alpha", help="ml signal score mode")
    p.add_argument("--ml-alpha-ret-scale", type=float, default=0.004, help="sigmoid scale for expected return in alpha mode")
    p.add_argument("--ml-alpha-rank-window", type=int, default=180, help="rolling rank window for alpha mode")
    p.add_argument("--ml-hold-bars", type=int, default=16, help="timeout exit bars for ML mode")
    p.add_argument("--ml-feature-warmup-bars", type=int, default=120, help="minimum bars for ML feature extraction")
    p.add_argument("--ml-entry-gap-bars", type=int, default=2, help="minimum bars between entries in ML mode")
    p.add_argument("--trailing-stop-pct", type=float, default=0.004, help="trailing stop ratio")

    # Risk and execution
    p.add_argument("--cash-buffer-pct", type=float, default=0.10, help="keep this cash ratio unused")
    p.add_argument("--stop-loss-pct", type=float, default=0.006, help="stop-loss ratio")
    p.add_argument("--take-profit-pct", type=float, default=0.020, help="take-profit ratio")
    p.add_argument("--fee-rate", type=float, default=0.0005, help="fee rate for sizing/pnl")
    p.add_argument("--paper-cash", type=float, default=10_000_000, help="fallback paper cash when balance inquiry fails")
    p.add_argument("--max-invested-pct", type=float, default=0.30, help="max total invested fraction of portfolio equity")
    p.add_argument("--max-positions", type=int, default=4, help="max number of concurrent positions")
    p.add_argument("--min-order-krw", type=float, default=150_000, help="skip buy when target order value is below this amount")
    p.add_argument("--retry", type=int, default=3, help="quote retry count")
    p.add_argument("--throttle-ms", type=int, default=800, help="delay between symbols in milliseconds")
    p.add_argument(
        "--no-buy-before-close-min",
        type=int,
        default=35,
        help="block new buys during last N minutes before market close",
    )
    p.add_argument("--no-buy-morning-start-hhmm", type=int, default=900, help="morning no-buy start (HHMM)")
    p.add_argument("--no-buy-morning-end-hhmm", type=int, default=930, help="morning no-buy end (HHMM, exclusive)")
    p.add_argument("--dry-run", action="store_true", default=True, help="print only, no order requests")
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
            d = r.get("stck_bsop_date", "")
            t = r.get("stck_cntg_hour", "")
            o = r.get("stck_oprc")
            h = r.get("stck_hgpr")
            l = r.get("stck_lwpr")
            c = r.get("stck_prpr")
            v = r.get("cntg_vol") or r.get("acml_vol") or "0"
            if not (o and h and l and c) or len(d) != 8 or len(t) != 6:
                continue
            out.append(
                {
                    "date": f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}",
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v),
                }
            )
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
        v = float(r.get("cntg_vol") or r.get("acml_vol") or 0.0)
        if key not in buckets:
            buckets[key] = {
                "date": f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{floored:02d}:00",
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": v,
            }
        else:
            b = buckets[key]
            b["high"] = max(b["high"], float(h))
            b["low"] = min(b["low"], float(l))
            b["close"] = float(c)
            b["volume"] += v
    return [buckets[k] for k in sorted(buckets.keys())]


def _sigmoid(x: float) -> float:
    x = max(-40.0, min(40.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _rolling_rank_01(values: List[float], x: float) -> float:
    if not values:
        return 0.5
    n = len(values)
    le = 0
    for v in values:
        if v <= x:
            le += 1
    return max(0.0, min(1.0, le / n))


def ml_signal_from_ohlc(
    ohlc: List[Dict[str, float]],
    feature_columns: List[str],
    clf_model: Any,
    reg_model: Any,
    signal_mode: str,
    alpha_ret_scale: float,
    alpha_rank_window: int,
    alpha_raw_hist: List[float],
) -> tuple[float | None, Dict[str, float]]:
    if len(ohlc) < 70:
        return None, {"score": 0.0}
    data = {
        "date": [str(x.get("date", "")) for x in ohlc],
        "open": np.asarray([float(x["open"]) for x in ohlc], dtype=float),
        "high": np.asarray([float(x["high"]) for x in ohlc], dtype=float),
        "low": np.asarray([float(x["low"]) for x in ohlc], dtype=float),
        "close": np.asarray([float(x["close"]) for x in ohlc], dtype=float),
        "volume": np.asarray([float(x.get("volume", 0.0)) for x in ohlc], dtype=float),
    }
    rows = build_rows(
        data=data,
        horizon=1,
        label_mode="fixed",
        up_threshold=0.01,
        down_threshold=0.01,
        atr_up_mult=2.0,
        atr_down_mult=1.2,
        atr_floor_pct=0.003,
    )
    if not rows:
        return None, {"score": 0.0}
    last = rows[-1]
    try:
        x = np.asarray([[float(last[c]) for c in feature_columns]], dtype=float)
    except Exception:
        return None, {"score": 0.0}

    prob = float(clf_model.predict_proba(x)[0, 1])  # type: ignore[attr-defined]
    if signal_mode == "prob":
        return prob, {"score": prob, "prob": prob, "alpha_raw": prob, "ret_pred": 0.0}

    ret_pred = float(reg_model.predict(x)[0]) if reg_model is not None else 0.0
    if reg_model is None:
        alpha_raw = prob
    else:
        ret_score = _sigmoid(ret_pred / max(1e-6, float(alpha_ret_scale)))
        alpha_raw = prob * ret_score
    alpha_raw_hist.append(alpha_raw)
    w = max(10, int(alpha_rank_window))
    if len(alpha_raw_hist) > w * 3:
        del alpha_raw_hist[: len(alpha_raw_hist) - (w * 3)]
    recent = alpha_raw_hist[-w:]
    score = _rolling_rank_01(recent, alpha_raw)
    return score, {"score": score, "prob": prob, "alpha_raw": alpha_raw, "ret_pred": ret_pred}


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
    entry_bar_index: Dict[str, int] = {s: -10**9 for s in symbols}
    last_entry_bar_index: Dict[str, int] = {s: -10**9 for s in symbols}
    bar_index: Dict[str, int] = {s: 0 for s in symbols}
    peak_price: Dict[str, float] = {s: 0.0 for s in symbols}
    alpha_raw_hist: Dict[str, List[float]] = {s: [] for s in symbols}
    realized_pnl_krw: Dict[str, float] = {s: 0.0 for s in symbols}
    last_price: Dict[str, float] = {s: 0.0 for s in symbols}
    last_processed_bar: Dict[str, str] = {s: "" for s in symbols}
    paper_cash = args.paper_cash
    last_known_cash: float | None = None
    predicted_cash: float | None = None
    cash_fail_streak = 0
    base_log_path = Path(args.log_file)
    base_log_path.parent.mkdir(parents=True, exist_ok=True)
    signal_need = required_bars_for_signal(args)
    warmup_bars = max(1, math.ceil(args.startup_lookback_minutes / max(1, args.bar_minutes)))
    count_hint = max(signal_need + warmup_bars, signal_need + 5, 120)

    model_path = Path(args.ml_model_path)
    if not model_path.exists():
        raise RuntimeError(f"ML model not found: {model_path}")
    with model_path.open("rb") as f:
        ml_bundle = pickle.load(f)
    ml_model = ml_bundle["model"]
    ml_feature_columns: List[str] = list(ml_bundle["feature_columns"])
    ml_reg_model = ml_bundle.get("reg_model")

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
        f"model={args.strategy_mode}(signal={args.ml_signal_mode},thr={args.ml_threshold:.2f},"
        f"hold={args.ml_hold_bars},tp={args.take_profit_pct:.4f},sl={args.stop_loss_pct:.4f},"
        f"trail={args.trailing_stop_pct:.4f}) "
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
                buy_reason = "ml_score"
                sell_reason = "signal"
                cross_info = ""
                last_px = ohlc[-1]["close"]
                last_price[symbol] = last_px
                bar_ts = str(ohlc[-1].get("date", ""))
                if len(bar_ts) >= 19:
                    cycle_bar_hms = bar_ts[11:19]
                if bar_ts and last_processed_bar[symbol] == bar_ts:
                    time.sleep(max(0, args.throttle_ms) / 1000.0)
                    continue
                if bar_ts:
                    last_processed_bar[symbol] = bar_ts
                bar_index[symbol] += 1
                if cooldown_left[symbol] > 0:
                    cooldown_left[symbol] -= 1

                score, m = ml_signal_from_ohlc(
                    ohlc=ohlc,
                    feature_columns=ml_feature_columns,
                    clf_model=ml_model,
                    reg_model=ml_reg_model,
                    signal_mode=args.ml_signal_mode,
                    alpha_ret_scale=args.ml_alpha_ret_scale,
                    alpha_rank_window=args.ml_alpha_rank_window,
                    alpha_raw_hist=alpha_raw_hist[symbol],
                )
                if score is None or len(ohlc) < max(70, int(args.ml_feature_warmup_bars)):
                    cycle_summary.append(f"{symbol} score=NA warmup")
                    time.sleep(max(0, args.throttle_ms) / 1000.0)
                    continue

                if has_position[symbol]:
                    held_bars[symbol] = max(0, bar_index[symbol] - entry_bar_index[symbol])
                    peak_price[symbol] = max(peak_price[symbol], last_px)
                else:
                    held_bars[symbol] = 0

                signal = 0
                if has_position[symbol]:
                    gross_ret = (last_px / entry_price[symbol] - 1.0) if entry_price[symbol] > 0 else 0.0
                    dd_from_peak = (
                        (last_px / peak_price[symbol] - 1.0) if peak_price[symbol] > 0 else 0.0
                    )
                    if hard_liquidation_window:
                        signal = -1
                        sell_reason = "hard_close"
                    elif args.stop_loss_pct > 0 and gross_ret <= -args.stop_loss_pct:
                        signal = -1
                        sell_reason = "stop_loss"
                    elif args.take_profit_pct > 0 and gross_ret >= args.take_profit_pct:
                        signal = -1
                        sell_reason = "take_profit"
                    elif args.trailing_stop_pct > 0 and dd_from_peak <= -args.trailing_stop_pct:
                        signal = -1
                        sell_reason = "trailing_stop"
                    elif held_bars[symbol] >= max(1, int(args.ml_hold_bars)):
                        signal = -1
                        sell_reason = "timeout"
                else:
                    allow_entry_gap = (bar_index[symbol] - last_entry_bar_index[symbol]) >= max(1, int(args.ml_entry_gap_bars))
                    if cooldown_left[symbol] <= 0 and allow_entry_gap and score >= float(args.ml_threshold):
                        signal = 1

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
                            f"score={m.get('score', 0):.2f} prob={m.get('prob', 0):.3f} "
                            f"alpha={m.get('alpha_raw', 0):.4f} "
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
                        entry_bar_index[symbol] = bar_index[symbol]
                        last_entry_bar_index[symbol] = bar_index[symbol]
                        peak_price[symbol] = last_px
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
                    entry_bar_index[symbol] = -10**9
                    peak_price[symbol] = 0.0
                    cooldown_left[symbol] = 0
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
                        entry_bar_index[symbol] = -10**9
                        peak_price[symbol] = 0.0
                        cooldown_left[symbol] = 0
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
