# 기본 주식 투자 프로그램 틀 (Python)

간단한 구조로 만든 주식 투자 프로그램 스켈레톤입니다.

## 기능
- CSV 가격 데이터 로드
- SMA(단순 이동평균) 크로스 전략 예시
- 간단 백테스트(매수/매도, 수수료 반영)
- 결과 요약(최종 자산, 수익률, 거래 횟수)

## 실행 방법
```bash
python main.py --data data/sample_prices.csv --cash 1000000 --short 5 --long 20 --fee 0.0005
```

## KIS 실데이터로 연습
1) 의존성 설치
```bash
pip install -r requirements.txt
```

2) 앱키 설정 (조회 전용 연습 권장)
```bash
# PowerShell
$env:KIS_APP_KEY="발급받은앱키"
$env:KIS_APP_SECRET="발급받은시크릿"
```

3) 일봉 데이터 CSV 저장
```bash
python fetch_kis_daily.py --symbol 005930 --start 20250101 --end 20260311 --out data/005930.csv
```

4) 백테스트 실행
```bash
python main.py --data data/005930.csv --cash 1000000 --short 5 --long 20 --fee 0.0005
```

## 방산 테마 모의투자 (실데이터)
기본 바스켓:
- 012450 (한화에어로스페이스)
- 079550 (LIG넥스원)
- 047810 (한국항공우주)
- 272210 (한화시스템)
- 064350 (현대로템)

실행:
```bash
python theme_mock_invest.py --start 20240101 --end 20260311 --cash 10000000 --ma-window 20 --rebalance-every 20
```

옵션:
- `--symbols`: 콤마 구분 종목코드 직접 지정
- `--curve-out`: 자산곡선 CSV 저장 경로

## 1분봉 모의투자
1) 1분봉 CSV 수집
```bash
python fetch_kis_minute.py --symbol 012450 --max-bars 300 --out data/012450_1m.csv
```

2) 기존 백테스터로 실행
```bash
python main.py --data data/012450_1m.csv --strategy sma --short 5 --long 20 --cash 1000000 --fee 0.0005
```

## CSV 형식
헤더 예시:
```csv
date,open,high,low,close,volume
2026-01-02,100,103,99,102,120000
```

`close` 컬럼은 필수입니다.

## 구조
- `main.py`: 실행 진입점
- `src/models.py`: 데이터 모델
- `src/data_loader.py`: CSV 로더
- `src/strategy.py`: 전략 인터페이스 + SMA 전략
- `src/backtester.py`: 백테스트 로직
- `data/sample_prices.csv`: 샘플 데이터

## ?? 3?? ????
```bash
python fetch_kis_minute.py --symbol 012450 --start 20260201 --end 20260311 --interval 3 --out data/012450_3m_1month.csv
python main.py --data data/012450_3m_1month.csv --strategy momentum --lookback 10 --mom-buy 0.003 --mom-sell -0.003 --cash 2000000 --fee 0.0005
```

## Docker? ?? ??
`.env` ??? ?? ?? ??:
```bash
docker compose up -d --build
docker compose logs -f
```
