from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from fetch_kis_daily import get_access_token
from main import VTS_BASE_URL, place_order


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
    p = argparse.ArgumentParser(description="KIS round-trip test: buy 1 share then sell 1 share")
    p.add_argument("--symbol", default="005930", help="stock code")
    p.add_argument("--qty", type=int, default=1, help="test quantity")
    p.add_argument("--wait-sec", type=int, default=2, help="seconds between buy and sell")
    p.add_argument("--execute", action="store_true", help="actually place orders")
    p.add_argument("--base-url", default=os.getenv("KIS_BASE_URL", VTS_BASE_URL))
    p.add_argument("--app-key", default=os.getenv("KIS_APP_KEY", ""))
    p.add_argument("--app-secret", default=os.getenv("KIS_APP_SECRET", ""))
    p.add_argument("--cano", default=os.getenv("KIS_CANO", ""))
    p.add_argument("--acnt-prdt-cd", default=os.getenv("KIS_ACNT_PRDT_CD", "01"))
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.app_key or not args.app_secret:
        raise RuntimeError("KIS_APP_KEY / KIS_APP_SECRET required")
    if not args.cano or not args.acnt_prdt_cd:
        raise RuntimeError("KIS_CANO / KIS_ACNT_PRDT_CD required")
    if args.qty <= 0:
        raise RuntimeError("qty must be > 0")

    print(
        f"roundtrip_test symbol={args.symbol} qty={args.qty} wait_sec={args.wait_sec} "
        f"execute={args.execute} base_url={args.base_url}"
    )
    if not args.execute:
        print("dry-run only. add --execute to send real orders.")
        return

    token = get_access_token(args.app_key, args.app_secret, base_url=args.base_url)

    buy_res = place_order(
        base_url=args.base_url,
        token=token,
        app_key=args.app_key,
        app_secret=args.app_secret,
        cano=args.cano,
        acnt_prdt_cd=args.acnt_prdt_cd,
        symbol=args.symbol,
        qty=args.qty,
        side="buy",
    )
    print(f"BUY  -> rt_cd={buy_res.get('rt_cd')} msg1={buy_res.get('msg1', '')}")

    time.sleep(max(0, args.wait_sec))

    sell_res = place_order(
        base_url=args.base_url,
        token=token,
        app_key=args.app_key,
        app_secret=args.app_secret,
        cano=args.cano,
        acnt_prdt_cd=args.acnt_prdt_cd,
        symbol=args.symbol,
        qty=args.qty,
        side="sell",
    )
    print(f"SELL -> rt_cd={sell_res.get('rt_cd')} msg1={sell_res.get('msg1', '')}")


if __name__ == "__main__":
    main()
