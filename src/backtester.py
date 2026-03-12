from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .models import BacktestReport, PriceBar
from .strategy import Strategy


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000
    fee_rate: float = 0.0005


class Backtester:
    def __init__(self, config: BacktestConfig, strategy: Strategy):
        self.config = config
        self.strategy = strategy

    def run(self, bars: List[PriceBar]) -> BacktestReport:
        cash = self.config.initial_cash
        shares = 0
        trades = 0
        trade_logs: List[str] = []

        for bar in bars:
            signal = self.strategy.on_bar(bar)
            price = bar.close

            if signal == 1 and shares == 0:
                max_shares = int(cash // (price * (1 + self.config.fee_rate)))
                if max_shares > 0:
                    cost = max_shares * price
                    fee = cost * self.config.fee_rate
                    cash -= cost + fee
                    shares = max_shares
                    trades += 1
                    trade_logs.append(
                        f"{bar.date.date()} BUY  qty={max_shares} price={price:.2f} fee={fee:.2f}"
                    )
            elif signal == -1 and shares > 0:
                proceeds = shares * price
                fee = proceeds * self.config.fee_rate
                sold_shares = shares
                cash += proceeds - fee
                shares = 0
                trades += 1
                trade_logs.append(
                    f"{bar.date.date()} SELL qty={sold_shares} price={price:.2f} fee={fee:.2f}"
                )

        if bars:
            final_equity = cash + shares * bars[-1].close
        else:
            final_equity = cash
        total_return_pct = ((final_equity / self.config.initial_cash) - 1) * 100
        return BacktestReport(
            initial_cash=self.config.initial_cash,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
            trades=trades,
            trade_logs=trade_logs,
        )
