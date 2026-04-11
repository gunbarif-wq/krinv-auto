from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict
from zoneinfo import ZoneInfo


KST = ZoneInfo("Asia/Seoul")
DEFAULT_SYMBOL_NAME_FILE = str(Path(__file__).resolve().with_name("krx_symbol_names.json"))


def _normalize_symbol_name_map(data: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for raw_symbol, raw_name in data.items():
        symbol = str(raw_symbol).strip().zfill(6)
        name = str(raw_name or "").strip()
        if symbol.isdigit() and len(symbol) == 6 and name:
            normalized[symbol] = name
    return dict(sorted(normalized.items()))


def load_symbol_name_map(path_text: str = DEFAULT_SYMBOL_NAME_FILE) -> Dict[str, str]:
    path = Path(path_text).expanduser()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("symbols"), dict):
        return _normalize_symbol_name_map(payload["symbols"])
    if isinstance(payload, dict):
        return _normalize_symbol_name_map(payload)
    return {}


def save_symbol_name_map(data: Dict[str, str], path_text: str = DEFAULT_SYMBOL_NAME_FILE) -> int:
    cleaned = _normalize_symbol_name_map(data)
    path = Path(path_text).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(cleaned),
        "symbols": cleaned,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(cleaned)


def refresh_symbol_name_map_from_krx(path_text: str = DEFAULT_SYMBOL_NAME_FILE) -> int:
    try:
        from pykrx import stock  # type: ignore
    except Exception:
        return 0

    names: Dict[str, str] = {}
    for market in ("KOSPI", "KOSDAQ", "KONEX"):
        try:
            tickers = stock.get_market_ticker_list(market=market)
        except Exception:
            continue
        for ticker in tickers:
            symbol = str(ticker).strip().zfill(6)
            if not (symbol.isdigit() and len(symbol) == 6):
                continue
            try:
                name = str(stock.get_market_ticker_name(symbol) or "").strip()
            except Exception:
                name = ""
            if name:
                names[symbol] = name
    if not names:
        return 0
    return save_symbol_name_map(names, path_text)
