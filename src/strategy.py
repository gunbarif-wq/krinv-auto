from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Optional

from .models import PriceBar


class Strategy(ABC):
    @abstractmethod
    def on_bar(self, bar: PriceBar) -> int:
        """Return signal: 1 (buy), -1 (sell), 0 (hold)."""


class SmaCrossStrategy(Strategy):
    def __init__(self, short_window: int = 5, long_window: int = 20):
        if short_window <= 0 or long_window <= 0:
            raise ValueError("window size must be positive")
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        self.short_window = short_window
        self.long_window = long_window
        self._prices: Deque[float] = deque(maxlen=long_window)
        self._prev_state: Optional[int] = None

    def on_bar(self, bar: PriceBar) -> int:
        self._prices.append(bar.close)
        if len(self._prices) < self.long_window:
            return 0

        prices = list(self._prices)
        short_sma = sum(prices[-self.short_window :]) / self.short_window
        long_sma = sum(prices) / self.long_window
        state = 1 if short_sma > long_sma else -1

        if self._prev_state is None:
            self._prev_state = state
            return 1 if state == 1 else 0
        if state == self._prev_state:
            return 0

        self._prev_state = state
        return 1 if state == 1 else -1


class MomentumStrategy(Strategy):
    def __init__(self, lookback: int = 20, buy_threshold: float = 0.03, sell_threshold: float = -0.02):
        if lookback <= 0:
            raise ValueError("lookback must be positive")
        self.lookback = lookback
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self._prices: Deque[float] = deque(maxlen=lookback + 1)
        self._position = 0

    def on_bar(self, bar: PriceBar) -> int:
        self._prices.append(bar.close)
        if len(self._prices) < self.lookback + 1:
            return 0

        old = self._prices[0]
        new = self._prices[-1]
        ret = (new / old) - 1.0

        if ret >= self.buy_threshold and self._position == 0:
            self._position = 1
            return 1
        if ret <= self.sell_threshold and self._position == 1:
            self._position = 0
            return -1
        return 0


class MeanReversionStrategy(Strategy):
    def __init__(self, window: int = 20, z_buy: float = 1.5, z_sell: float = 0.2):
        if window <= 1:
            raise ValueError("window must be > 1")
        self.window = window
        self.z_buy = z_buy
        self.z_sell = z_sell
        self._prices: Deque[float] = deque(maxlen=window)
        self._position = 0

    def on_bar(self, bar: PriceBar) -> int:
        self._prices.append(bar.close)
        if len(self._prices) < self.window:
            return 0

        prices = list(self._prices)
        mean = sum(prices) / self.window
        var = sum((p - mean) ** 2 for p in prices) / self.window
        std = var ** 0.5
        if std == 0:
            return 0

        z = (bar.close - mean) / std
        if z <= -self.z_buy and self._position == 0:
            self._position = 1
            return 1
        if z >= self.z_sell and self._position == 1:
            self._position = 0
            return -1
        return 0
