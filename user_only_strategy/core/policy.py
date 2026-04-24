from __future__ import annotations

from datetime import datetime


def hhmm(now: datetime) -> int:
    return now.hour * 100 + now.minute


def in_call_auction_window(now: datetime) -> bool:
    t = hhmm(now)
    return now.weekday() < 5 and ((830 <= t < 900) or (1520 <= t < 1530))


def is_hard_market_closed(now: datetime) -> bool:
    # After 15:30 KST: no orders and no notifications (paper/demo safeguard).
    t = hhmm(now)
    return now.weekday() < 5 and t >= 1530


def can_trade(now: datetime, market_open_hhmm: int, market_close_hhmm: int) -> bool:
    t = hhmm(now)
    return now.weekday() < 5 and int(market_open_hhmm) <= t < int(market_close_hhmm)


def can_notify(now: datetime, *, alerts_muted: bool) -> bool:
    return (not alerts_muted) and (not is_hard_market_closed(now))

