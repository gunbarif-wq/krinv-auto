from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List

import requests

from fetch_kis_daily import get_access_token


VTS_BASE_URL = "https://openapivts.koreainvestment.com:29443"
DEFENSE_SYMBOLS = ["012450", "079550", "047810", "272210", "064350"]
KST = ZoneInfo("Asia/Seoul")


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
    p.add_argument("--symbols", default=",".join(DEFENSE_SYMBOLS), help="comma-separated symbols")
    p.add_argument("--interval-sec", type=int, default=30, help="main loop interval in seconds")
    p.add_argument("--bar-minutes", type=int, choices=[1, 3, 5], default=1, help="signal timeframe")

    # Signal parameters
    p.add_argument("--short", type=int, default=5, help="short SMA window")
    p.add_argument("--long", type=int, default=20, help="long SMA window")
    p.add_argument("--mom-window", type=int, default=12, help="momentum window")
    p.add_argument("--stoch-window", type=int, default=14, help="stochastic K window")
    p.add_argument("--stoch-smooth", type=int, default=3, help="stochastic D smoothing")
    p.add_argument("--entry-threshold", type=float, default=0.40, help="buy score threshold")
    p.add_argument("--exit-threshold", type=float, default=-0.28, help="sell score threshold")

    # Risk and execution
    p.add_argument("--cash-buffer-pct", type=float, default=0.15, help="keep this cash ratio unused")
    p.add_argument("--stop-loss-pct", type=float, default=0.012, help="stop-loss ratio")
    p.add_argument("--take-profit-pct", type=float, default=0.020, help="take-profit ratio")
    p.add_argument("--cooldown-bars", type=int, default=8, help="bars to wait after exit")
    p.add_argument("--entry-confirm-bars", type=int, default=4, help="consecutive buy signals required")
    p.add_argument("--exit-confirm-bars", type=int, default=4, help="consecutive sell signals required")
    p.add_argument("--min-hold-bars", type=int, default=8, help="minimum bars to hold before normal exit")
    p.add_argument("--fee-rate", type=float, default=0.0005, help="fee rate for sizing/pnl")
    p.add_argument("--paper-cash", type=float, default=10_000_000, help="fallback paper cash when balance inquiry fails")
    p.add_argument("--position-size-pct", type=float, default=0.08, help="max position size as fraction of portfolio equity")
    p.add_argument("--min-order-krw", type=float, default=200_000, help="skip buy when target order value is below this amount")
    p.add_argument("--retry", type=int, default=3, help="quote retry count")
    p.add_argument("--throttle-ms", type=int, default=800, help="delay between symbols in milliseconds")
    p.add_argument("--dry-run", action="store_true", help="print only, no order requests")
    p.add_argument("--log-file", default="data/realtime_events.txt", help="important event log path")
    p.add_argument("--log-rotate-minutes", type=int, default=10, help="create a new log file every N minutes")

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
    r = requests.post(url, headers=headers, data=json.dumps(body), timeout=15)
    r.raise_for_status()
    return r.json()


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

    for obj in candidates:
        for key in ("ord_psbl_cash", "dnca_tot_amt", "prvs_rcdl_excc_amt", "tot_evlu_amt"):
            v = obj.get(key)
            if v is not None and str(v).strip() != "":
                try:
                    return float(str(v).replace(",", ""))
                except ValueError:
                    pass
    raise RuntimeError("cannot parse orderable cash from balance response")


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
    paper_cash = args.paper_cash
    last_known_cash: float | None = None
    predicted_cash: float | None = None
    cash_fail_streak = 0
    base_log_path = Path(args.log_file)
    base_log_path.parent.mkdir(parents=True, exist_ok=True)

    def current_log_path(now: datetime) -> Path:
        rotate = max(1, args.log_rotate_minutes)
        bucket_minute = (now.minute // rotate) * rotate
        bucket = now.replace(minute=bucket_minute, second=0, microsecond=0)
        return base_log_path.with_name(
            f"{base_log_path.stem}_{bucket.strftime('%Y%m%d_%H%M')}{base_log_path.suffix or '.txt'}"
        )

    def emit(message: str, *, save: bool = False) -> None:
        print(message)
        if save:
            path = current_log_path(datetime.now())
            with path.open("a", encoding="utf-8") as f:
                f.write(message + "\n")

    emit("start realtime paper trader", save=True)
    emit(
        f"symbols={','.join(symbols)} dry_run={args.dry_run} bar={args.bar_minutes}m "
        f"model=multi_factor(short={args.short},long={args.long},mom={args.mom_window},"
        f"stoch={args.stoch_window}/{args.stoch_smooth})",
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
            raise RuntimeError("initial cash inquiry failed in live mode")

    while True:
        try:
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
                        count_hint=max(args.long + args.stoch_window + 12, 80),
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
                signal, m = multi_factor_signal(
                    ohlc=ohlc,
                    short=args.short,
                    long=args.long,
                    mom_window=args.mom_window,
                    stoch_window=args.stoch_window,
                    stoch_smooth=args.stoch_smooth,
                    entry_threshold=args.entry_threshold,
                    exit_threshold=args.exit_threshold,
                )
                last_px = ohlc[-1]["close"]
                last_price[symbol] = last_px
                bar_ts = ""
                if bars:
                    d = bars[-1].get("stck_bsop_date", "")
                    t = bars[-1].get("stck_cntg_hour", "")
                    if len(d) == 8 and len(t) == 6:
                        bar_ts = f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
                        cycle_bar_hms = f"{t[:2]}:{t[2:4]}:{t[4:6]}"

                if cooldown_left[symbol] > 0:
                    cooldown_left[symbol] -= 1
                    if signal == 1:
                        signal = 0

                if has_position[symbol] and entry_price[symbol] > 0:
                    held_bars[symbol] += 1
                    pnl_pct = (last_px / entry_price[symbol]) - 1.0
                    if pnl_pct <= -args.stop_loss_pct or pnl_pct >= args.take_profit_pct:
                        signal = -1

                if signal == 1:
                    entry_streak[symbol] += 1
                    exit_streak[symbol] = 0
                elif signal == -1:
                    exit_streak[symbol] += 1
                    entry_streak[symbol] = 0
                else:
                    entry_streak[symbol] = 0
                    exit_streak[symbol] = 0

                if signal == 1 and not has_position[symbol]:
                    if entry_streak[symbol] < args.entry_confirm_bars:
                        cycle_summary.append(
                            f"{symbol} px={last_px:.0f} score={m.get('score', 0):.2f} "
                            f"cd={cooldown_left[symbol]} e={entry_streak[symbol]}/{args.entry_confirm_bars}"
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
                    mtm_positions = sum(
                        held_qty[s] * (last_price[s] if last_price[s] > 0 else entry_price[s])
                        for s in symbols
                        if has_position[s] and held_qty[s] > 0
                    )
                    portfolio_equity_est = orderable_cash + mtm_positions
                    capped_by_equity = max(0.0, portfolio_equity_est * args.position_size_pct)
                    capped_by_cash = max(0.0, orderable_cash * (1.0 - args.cash_buffer_pct))
                    order_budget = min(capped_by_equity, capped_by_cash)
                    if order_budget < args.min_order_krw:
                        cycle_summary.append(
                            f"{symbol} px={last_px:.0f} score={m.get('score', 0):.2f} cd={cooldown_left[symbol]}"
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
                            f"{symbol} px={last_px:.0f} score={m.get('score', 0):.2f} cd={cooldown_left[symbol]} qty=0"
                        )
                        time.sleep(max(0, args.throttle_ms) / 1000.0)
                        continue
                    if args.dry_run:
                        emit(
                            f"[{ts}] >>> BUY <<< {symbol} bar={bar_ts or '-'} px={last_px:.0f} qty={qty} "
                            f"score={m.get('score', 0):.2f} mom={m.get('mom', 0)*100:.2f}% "
                            f"k={m.get('k', 0):.1f} d={m.get('d', 0):.1f} "
                            f"cash={orderable_cash:.0f} budget={order_budget:.0f} fee={buy_fee:,.0f} (dry-run)",
                            save=True,
                        )
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
                        emit(
                            f"[{ts}] >>> BUY <<< {symbol} bar={bar_ts or '-'} order -> "
                            f"{res.get('msg1', '')} / rt_cd={res.get('rt_cd')}",
                            save=True,
                        )
                    if qty > 0:
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
                    if held_bars[symbol] < args.min_hold_bars:
                        cycle_summary.append(
                            f"{symbol} px={last_px:.0f} score={m.get('score', 0):.2f} "
                            f"cd={cooldown_left[symbol]} h={held_bars[symbol]}/{args.min_hold_bars}"
                        )
                        time.sleep(max(0, args.throttle_ms) / 1000.0)
                        continue
                    if exit_streak[symbol] < args.exit_confirm_bars:
                        cycle_summary.append(
                            f"{symbol} px={last_px:.0f} score={m.get('score', 0):.2f} "
                            f"cd={cooldown_left[symbol]} x={exit_streak[symbol]}/{args.exit_confirm_bars}"
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
                            f"[{ts}] <<< SELL >>> {symbol} bar={bar_ts or '-'} px={last_px:.0f} qty={qty} "
                            f"pnl={pnl_pct:.2f}% ({trade_pnl_krw:,.0f} KRW) "
                            f"score={m.get('score', 0):.2f} fee={sell_fee:,.0f} (dry-run)",
                            save=True,
                        )
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
                        emit(
                            f"[{ts}] <<< SELL >>> {symbol} bar={bar_ts or '-'} order -> "
                            f"{res.get('msg1', '')} / rt_cd={res.get('rt_cd')}",
                            save=True,
                        )
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
                    cycle_summary.append(f"{symbol} px={last_px:.0f} score={m.get('score', 0):.2f} cd={cooldown_left[symbol]}")

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
