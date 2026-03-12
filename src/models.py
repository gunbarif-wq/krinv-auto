from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List


@dataclass
class PriceBar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacktestReport:
    initial_cash: float
    final_equity: float
    total_return_pct: float
    trades: int
    trade_logs: List[str]
