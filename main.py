from __future__ import annotations

import argparse

from src.backtester import Backtester, BacktestConfig
from src.data_loader import load_ohlcv_from_csv
from src.strategy import MeanReversionStrategy, MomentumStrategy, SmaCrossStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="기본 주식 투자 프로그램 틀")
    parser.add_argument("--data", required=True, help="CSV 파일 경로")
    parser.add_argument("--cash", type=float, default=1_000_000, help="초기 자본")
    parser.add_argument("--short", type=int, default=5, help="단기 SMA 기간")
    parser.add_argument("--long", type=int, default=20, help="장기 SMA 기간")
    parser.add_argument("--fee", type=float, default=0.0005, help="거래 수수료 비율")
    parser.add_argument(
        "--strategy",
        choices=["sma", "momentum", "mean_reversion"],
        default="sma",
        help="전략 선택",
    )
    parser.add_argument("--lookback", type=int, default=20, help="모멘텀/평균회귀 기본 기간")
    parser.add_argument("--mom-buy", type=float, default=0.03, help="모멘텀 매수 임계값(수익률)")
    parser.add_argument("--mom-sell", type=float, default=-0.02, help="모멘텀 매도 임계값(수익률)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prices = load_ohlcv_from_csv(args.data)

    if args.strategy == "sma":
        strategy = SmaCrossStrategy(short_window=args.short, long_window=args.long)
    elif args.strategy == "momentum":
        strategy = MomentumStrategy(
            lookback=args.lookback,
            buy_threshold=args.mom_buy,
            sell_threshold=args.mom_sell,
        )
    else:
        strategy = MeanReversionStrategy(window=args.lookback)
    config = BacktestConfig(initial_cash=args.cash, fee_rate=args.fee)
    engine = Backtester(config=config, strategy=strategy)

    report = engine.run(prices)
    print("=== Backtest Result ===")
    print(f"Initial Cash : {report.initial_cash:,.2f}")
    print(f"Final Equity : {report.final_equity:,.2f}")
    print(f"Total Return : {report.total_return_pct:.2f}%")
    print(f"Trades       : {report.trades}")
    print("Trade Logs   :")
    if report.trade_logs:
        for log in report.trade_logs:
            print(f"- {log}")
    else:
        print("- No trades")


if __name__ == "__main__":
    main()
