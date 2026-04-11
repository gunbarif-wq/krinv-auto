from __future__ import annotations

import json
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

import requests


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


def _header_key(text: str) -> str:
    return "".join(ch for ch in str(text).strip() if not ch.isspace())


class _HtmlTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: List[List[List[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_table: List[List[str]] = []
        self._current_row: List[str] = []
        self._cell_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: Sequence[tuple[str, Optional[str]]]) -> None:
        lower = tag.lower()
        if lower == "table":
            self._in_table = True
            self._current_table = []
        elif lower == "tr" and self._in_table:
            self._in_row = True
            self._current_row = []
        elif lower in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        lower = tag.lower()
        if lower in {"td", "th"} and self._in_cell:
            self._current_row.append("".join(self._cell_parts).strip())
            self._cell_parts = []
            self._in_cell = False
        elif lower == "tr" and self._in_row:
            if any(cell.strip() for cell in self._current_row):
                self._current_table.append(self._current_row[:])
            self._current_row = []
            self._in_row = False
        elif lower == "table" and self._in_table:
            if self._current_table:
                self.tables.append(self._current_table[:])
            self._current_table = []
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)


def _decode_kind_html(raw: bytes) -> str:
    for encoding in ("euc-kr", "cp949", "utf-8"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\ud68c\uc0ac\uba85" in text or "\uc885\ubaa9\ucf54\ub4dc" in text:
            return text
    return raw.decode("utf-8", errors="ignore")


def _extract_names_from_table(rows: Sequence[Sequence[str]]) -> Dict[str, str]:
    if not rows:
        return {}
    header = [_header_key(cell) for cell in rows[0]]
    try:
        name_idx = header.index("\ud68c\uc0ac\uba85")
        symbol_idx = header.index("\uc885\ubaa9\ucf54\ub4dc")
    except ValueError:
        return {}
    names: Dict[str, str] = {}
    for row in rows[1:]:
        if max(name_idx, symbol_idx) >= len(row):
            continue
        symbol = str(row[symbol_idx]).strip().split(".")[0].zfill(6)
        name = str(row[name_idx]).strip()
        if symbol.isdigit() and len(symbol) == 6 and name:
            names[symbol] = name
    return names


def _load_from_kind_download() -> Dict[str, str]:
    resp = requests.get(
        "https://kind.krx.co.kr/corpgeneral/corpList.do",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://kind.krx.co.kr/corpgeneral/corpList.do",
        },
        params={"method": "download", "searchType": "13"},
        timeout=20,
    )
    resp.raise_for_status()
    parser = _HtmlTableParser()
    parser.feed(_decode_kind_html(resp.content))
    for table in parser.tables:
        names = _extract_names_from_table(table)
        if names:
            return names
    return {}


def refresh_symbol_name_map_from_krx(path_text: str = DEFAULT_SYMBOL_NAME_FILE, verbose: bool = False) -> int:
    names: Dict[str, str] = {}
    try:
        from pykrx import stock  # type: ignore

        for market in ("KOSPI", "KOSDAQ", "KONEX"):
            try:
                tickers = stock.get_market_ticker_list(market=market)
            except Exception as exc:
                if verbose:
                    print(f"pykrx_market_error[{market}]={exc}")
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
        if verbose:
            print(f"pykrx_count={len(names)}")
    except Exception as exc:
        if verbose:
            print(f"pykrx_error={exc}")
        names = {}

    if not names:
        try:
            names = _load_from_kind_download()
            if verbose:
                print(f"kind_count={len(names)}")
        except Exception as exc:
            if verbose:
                print(f"kind_error={exc}")
            names = {}

    if not names:
        if verbose:
            print("krx_symbol_refresh_failed=1")
        return 0
    return save_symbol_name_map(names, path_text)
