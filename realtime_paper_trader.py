from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import requests

from fetch_kis_daily import get_access_token


VTS_BASE_URL = "https://openapivts.koreainvestment.com:29443"
DEFENSE_SYMBOLS = ["012450", "079550", "047810", "272210", "064350"]


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
    p.add_argument("--entry-threshold", type=float, default=0.25, help="buy score threshold")
    p.add_argument("--exit-threshold", type=float, default=-0.10, help="sell score threshold")

    # Risk and execution
    p.add_argument("--cash-buffer-pct", type=float, default=0.15, help="keep this cash ratio unused")
    p.add_argument("--stop-loss-pct", type=float, default=0.012, help="stop-loss ratio")
    p.add_argument("--take-profit-pct", type=float, default=0.020, help="take-profit ratio")
    p.add_argument("--cooldown-bars", type=int, default=3, help="bars to wait after exit")
    p.add_argument("--entry-confirm-bars", type=int, default=2, help="consecutive buy signals required")
    p.add_argument("--exit-confirm-bars", type=int, default=2, help="consecutive sell signals required")
    p.add_argument("--min-hold-bars", type=int, default=3, help="minimum bars to hold before normal exit")
    p.add_argument("--fee-rate", type=float, default=0.0005, help="fee rate for sizing/pnl")
    p.add_argument("--paper-cash", type=float, default=10_000_000, help="fallback paper cash when balance inquiry fails")
    p.add_argument("--retry", type=int, default=3, help="quote retry count")
    p.add_argument("--throttle-ms", type=int, default=800, help="delay between symbols in milliseconds")
    p.add_argument("--dry-run", action="store_true", help="print only, no order requests")

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
    ymd = datetime.now().strftime("%Y%m%d")
    cursor_time = "153000"
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

    print("start realtime paper trader")
    print(
        f"symbols={','.join(symbols)} dry_run={args.dry_run} bar={args.bar_minutes}m "
        f"model=multi_factor(short={args.short},long={args.long},mom={args.mom_window},"
        f"stoch={args.stoch_window}/{args.stoch_smooth})"
    )
    if args.dry_run:
        print(f"paper_cash={paper_cash:,.0f} KRW")
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
            print(f"initial_orderable_cash={initial_cash:,.0f} KRW")
        except Exception as e:
            print(f"initial_orderable_cash=ERROR ({str(e)[:200]})")
            raise RuntimeError("initial cash inquiry failed in live mode")

    while True:
        try:
            for symbol in symbols:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
                    print(f"[{ts}] {symbol} fetch error: {str(e)[:180]}")
                    time.sleep(max(0, args.throttle_ms) / 1000.0)
                    continue

                ohlc = resample_ohlc(bars, args.bar_minutes)
                if not ohlc:
                    print(f"[{ts}] {symbol} no data")
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
                        print(
                            f"[{ts}] {symbol} hold px={last_px:.0f} pos={has_position[symbol]} "
                            f"score={m.get('score', 0):.2f} entry_wait={entry_streak[symbol]}/{args.entry_confirm_bars}"
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
                                print(f"[{ts}] {symbol} cash inquiry error: {str(e)[:180]} (no fallback)")
                                time.sleep(max(0, args.throttle_ms) / 1000.0)
                                continue
                            orderable_cash = fallback_cash
                            print(
                                f"[{ts}] {symbol} cash inquiry error -> using predicted cash "
                                f"{orderable_cash:,.0f} KRW (fail_streak={cash_fail_streak})"
                            )
                    usable_cash = max(0.0, orderable_cash * (1.0 - args.cash_buffer_pct))
                    qty = int(usable_cash // (last_px * (1.0 + args.fee_rate)))
                    buy_cost = qty * last_px
                    buy_fee = buy_cost * args.fee_rate
                    total_buy = buy_cost + buy_fee
                    if args.dry_run and total_buy > paper_cash:
                        qty = int(paper_cash // (last_px * (1.0 + args.fee_rate)))
                        buy_cost = qty * last_px
                        buy_fee = buy_cost * args.fee_rate
                        total_buy = buy_cost + buy_fee
                    if args.dry_run:
                        print(
                            f"[{ts}] {symbol} BUY px={last_px:.0f} qty={qty} "
                            f"score={m.get('score', 0):.2f} mom={m.get('mom', 0)*100:.2f}% "
                            f"k={m.get('k', 0):.1f} d={m.get('d', 0):.1f} "
                            f"cash={orderable_cash:.0f} usable={usable_cash:.0f} fee={buy_fee:,.0f} (dry-run)"
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
                        print(f"[{ts}] {symbol} BUY order -> {res.get('msg1', '')} / rt_cd={res.get('rt_cd')}")
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
                        print(
                            f"[{ts}] portfolio realized={total_realized:,.0f} KRW "
                            f"unrealized={total_unrealized:,.0f} KRW"
                        )
                elif signal == -1 and has_position[symbol]:
                    if held_bars[symbol] < args.min_hold_bars:
                        print(
                            f"[{ts}] {symbol} hold px={last_px:.0f} pos={has_position[symbol]} "
                            f"score={m.get('score', 0):.2f} hold_lock={held_bars[symbol]}/{args.min_hold_bars}"
                        )
                        time.sleep(max(0, args.throttle_ms) / 1000.0)
                        continue
                    if exit_streak[symbol] < args.exit_confirm_bars:
                        print(
                            f"[{ts}] {symbol} hold px={last_px:.0f} pos={has_position[symbol]} "
                            f"score={m.get('score', 0):.2f} exit_wait={exit_streak[symbol]}/{args.exit_confirm_bars}"
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
                        print(
                            f"[{ts}] {symbol} SELL px={last_px:.0f} qty={qty} "
                            f"pnl={pnl_pct:.2f}% ({trade_pnl_krw:,.0f} KRW) "
                            f"score={m.get('score', 0):.2f} fee={sell_fee:,.0f} (dry-run)"
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
                        print(f"[{ts}] {symbol} SELL order -> {res.get('msg1', '')} / rt_cd={res.get('rt_cd')}")
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
                    print(
                        f"[{ts}] portfolio realized={total_realized:,.0f} KRW "
                        f"unrealized={total_unrealized:,.0f} KRW"
                    )
                else:
                    print(
                        f"[{ts}] {symbol} hold px={last_px:.0f} pos={has_position[symbol]} "
                        f"score={m.get('score', 0):.2f} cd={cooldown_left[symbol]}"
                    )

                time.sleep(max(0, args.throttle_ms) / 1000.0)
        except KeyboardInterrupt:
            print("stopped by user")
            break
        except Exception as e:
            print(f"[ERR] {e}")
            if "Unauthorized" in str(e) or "access token" in str(e).lower():
                token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)
            time.sleep(max(2, args.interval_sec))
            continue

        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
