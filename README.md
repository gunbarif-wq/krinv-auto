# krinv-auto

## 1) 설치
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

## 2) .env 설정
프로젝트 루트에 `.env` 파일 생성:
```env
KIS_APP_KEY=YOUR_APP_KEY
KIS_APP_SECRET=YOUR_APP_SECRET
KIS_CANO=YOUR_ACCOUNT_NO
KIS_ACNT_PRDT_CD=01
KIS_BASE_URL=https://openapivts.koreainvestment.com:29443
```

## 3) 드라이런 실행
```bash
python realtime_paper_trader.py --dry-run --log-file data/realtime_events.txt --log-rotate-minutes 10
```

## 4) 과매매 완화 옵션 예시
```bash
python realtime_paper_trader.py --dry-run \
  --entry-confirm-bars 3 --exit-confirm-bars 3 --min-hold-bars 5 \
  --entry-threshold 0.35 --exit-threshold -0.20 --cooldown-bars 5 \
  --log-file data/realtime_events.txt --log-rotate-minutes 10
```

## 5) 실거래 실행
```bash
python realtime_paper_trader.py --log-file data/realtime_events.txt --log-rotate-minutes 10
```
