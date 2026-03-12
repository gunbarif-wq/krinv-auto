from __future__ import annotations

import csv
from datetime import datetime
from typing import List

from .models import PriceBar


def load_ohlcv_from_csv(path: str) -> List[PriceBar]:
    bars: List[PriceBar] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(
                PriceBar(
                    date=_parse_date_like(row["date"]),
                    open=float(row.get("open", row["close"])),
                    high=float(row.get("high", row["close"])),
                    low=float(row.get("low", row["close"])),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                )
            )
    return bars


def _parse_date_like(value: str) -> datetime:
    v = value.strip()
    # Supports both daily("YYYY-MM-DD") and minute("YYYY-MM-DD HH:MM:SS") formats.
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
