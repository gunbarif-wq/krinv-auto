# Instagram-derived rule notes

This strategy file is isolated from existing `main.py` logic.

## Source reels
- https://www.instagram.com/reel/DWD3Xubku1W/
- https://www.instagram.com/reel/DWTiIT8khY_/
- https://www.instagram.com/reel/DWqo5hLlDKJ/

## Parsed metadata summary
- `DWD3Xubku1W`: caption theme is ADX-focused entry timing.
- `DWTiIT8khY_`: caption theme is RSI + Bollinger Band combination.
- `DWqo5hLlDKJ`: caption theme is "20 moving average support is not just touching, but holding".

## Rules encoded in `monday_custom_timing_bot.py`
- Monday candidate universe from:
  - volume rank
  - fluctuation rank
  - near-new-highlow rank
  - foreign/institution total
- Daily filter:
  - MA3/MA5/MA10 convergence
  - no MA60 break
  - MA20 support hold
- Buy timing:
  - Slow stochastic golden cross
  - DMI golden cross (+DI over -DI)
  - ADX rising and above threshold
  - RSI/Bollinger confirmation
- Sell timing:
  - stochastic dead cross in high zone, or
  - DMI dead cross, or
  - RSI + Bollinger weakness, or
  - MA20 break
