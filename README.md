# krinv-auto

## 1) Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Configure .env
Create `.env` in project root:
```env
KIS_APP_KEY=YOUR_APP_KEY
KIS_APP_SECRET=YOUR_APP_SECRET
KIS_CANO=YOUR_ACCOUNT_NO
KIS_ACNT_PRDT_CD=01
KIS_BASE_URL=https://openapivts.koreainvestment.com:29443
```

## 3) Dry-run
```bash
python main.py --dry-run --log-file data/realtime_events.txt --log-rotate-minutes 10
```

## 4) Dry-run with anti-churn options
```bash
python main.py --dry-run \
  --entry-confirm-bars 3 --exit-confirm-bars 3 --min-hold-bars 5 \
  --entry-threshold 0.35 --exit-threshold -0.20 --cooldown-bars 5 \
  --position-size-pct 0.15 --min-order-krw 200000 \
  --log-file data/realtime_events.txt --log-rotate-minutes 10
```

## 5) Live trading
```bash
python main.py --log-file data/realtime_events.txt --log-rotate-minutes 10
```

## 6) Fetch last week 1m bars
```bash
python fetch_last_week_1m.py --days 7 --out-dir data/intraday_1w
```

## 7) Replay backtest from CSV
```bash
python backtest_realtime_from_csv.py --data-dir data/intraday_1w
```
