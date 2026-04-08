# krinv-auto-mod

## 1) 오라클 클라우드(OCI) 준비
처음 설치하는 기준으로 아래 순서대로 진행하세요.

1. OCI 로그인
2. 좌측 상단 메뉴(햄버거) 클릭
3. `Compute` -> `Instances` -> `Create instance`
4. 인스턴스 설정
   - Name: 예) `krinv-auto-mod`
   - Image: `Oracle Linux`(권장) 또는 `Ubuntu`
   - Shape: 무료 티어면 `VM.Standard.E2.1.Micro` 또는 사용 가능한 무료 Shape
5. 네트워크 설정
   - 기본값 사용 가능 (`Create new virtual cloud network`)
   - Public IP 할당 체크 유지
6. `Create` 클릭
7. 상태가 `Running`이 될 때까지 대기
8. 인스턴스 상세 화면에서 Public IP 확인

권장 확인 사항:
- 보안 규칙에서 `22` 포트(SSH) 허용
- 공식 가이드: https://docs.oracle.com/en-us/iaas/Content/Compute/Tasks/launchinginstance.htm

## 2) 설치
OCI Code Editor 기준:

1. 우측 상단 `Code Editor` 열기
2. 터미널 열기 (Ctrl + ~)
3. GitHub SSH 키 생성/등록
4. SSH로 프로젝트 클론 및 설치

### 2-1) GitHub SSH 키 생성/등록
1. 키 생성
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
2. 공개키 확인
```bash
cat ~/.ssh/id_ed25519.pub
```
3. 출력된 키를 복사해서 GitHub에 등록
   - GitHub 우측 상단 프로필 -> `Settings`
   - 좌측 `SSH and GPG keys`
   - `New SSH key` -> Key 내용 붙여넣기 -> 저장

### 2-2) SSH로 클론
```bash
git clone git@github.com:osutaiko/krinv-auto-mod.git
cd krinv-auto-mod
pip install -r requirements.txt
```

## 3) .env 설정
프로젝트 루트(`krinv-auto-mod`)에 `.env` 파일을 만들고 아래 값을 입력하세요.

```env
KIS_APP_KEY=한투에서 받은 앱키
KIS_APP_SECRET=한투에서 받은 시크릿키
KIS_CANO=계좌번호 8자리
KIS_ACNT_PRDT_CD=01
KIS_BASE_URL=https://openapivts.koreainvestment.com:29443
```

## 4) 실행
실거래(모의투자 API 주문 전송):

```bash
python main.py
```

드라이런(주문 미전송):

```bash
python main.py --dry-run
```

## 5) 커스터마이징 인자
필요하면 아래 인자로 전략/출력/운영을 조정할 수 있습니다.

- `--symbols`: 대상 종목 코드 목록(콤마 구분)
- `--bar-minutes`: 봉 주기 (`1`, `3`, `5`)
- `--interval-sec`: 장중 루프 주기(초)
- `--entry-threshold`: 매수 점수 임계값
- `--entry-confirm-bars`: 연속 매수 신호 확정 횟수
- `--min-hold-bars`: 최소 보유 봉 수
- `--exit-threshold`: score-drop 청산 임계값
- `--cooldown-bars`: 청산 후 재진입 대기 봉 수
- `--trailing-stop-pct`: 추세 훼손 시 청산하는 트레일링 스탑 비율
- `--vwap-exit-min-hold-bars`: VWAP 이탈 청산 허용 최소 보유 봉 수
- `--vwap-exit-max-profit-pct`: VWAP 이탈 청산을 허용할 최대 수익률
- `--position-size-pct`: 종목당 최대 비중
- `--cash-buffer-pct`: 현금 버퍼 비중
- `--min-order-krw`: 최소 주문 금액
- `--fee-rate`: 수수료 비율
- `--after-close-action`: 장외 동작 (`wait` 또는 `exit`)
- `--log-file`: 중요 이벤트 로그 파일 경로
- `--log-rotate-minutes`: 로그 파일 로테이션 주기(분)

## 6) 워크포워드 튜닝
과거 전체 구간을 여러 윈도우로 나눠서 `train -> 바로 다음 test`를 반복 평가합니다.

예시:

```bash
python tune_backtest_walkforward.py \
  --source-data-dir data/backtest_sets/full_1m \
  --strategy-mode ma_cross_level \
  --wf-train-days 30 \
  --wf-test-days 10 \
  --wf-step-days 10 \
  --n-trials 20
```

## 7) ML 신호 파이프라인
### 7-1) 피처/라벨 데이터셋 생성
```bash
python build_ml_dataset.py --data-root data/backtest_sets_047810_5y --symbol 047810 --out-dir data/ml
```

ATR 동적 라벨 예시:
```bash
python build_ml_dataset.py --data-root data/backtest_sets_047810_5y --symbol 047810 --label-mode atr --atr-up-mult 2.0 --atr-down-mult 1.2 --atr-floor-pct 0.003 --out-dir data/ml
```

### 7-2) 학습 + 임계값 선택(Validation 기준)
```bash
python train_ml_signal.py --dataset-dir data/ml/047810 --symbol 047810
```

### 7-3) 테스트 구간 백테스트
```bash
python backtest_ml_signal.py --dataset-csv data/ml/047810/047810_test_ml.csv --model-path data/ml/047810/047810_model.pkl
```

### 7-4) 정책 파라미터 자동 탐색(수동 튜닝 없이)
```bash
python optimize_ml_policy.py --model-path data/ml/047810/047810_model.pkl --val-csv data/ml/047810/047810_val_ml.csv --test-csv data/ml/047810/047810_test_ml.csv --symbol 047810 --max-evals 1500
```

### 7-5) 심화 학습(워크포워드 + 다중 모델 자동선택)
`ml_walkforward.py`는 fold마다 `train->val`에서 모델(로지스틱/부스팅/트리 앙상블)을 quick-score로 선별한 뒤, 상위 모델만 정책 탐색해서 `test(OOS)`를 평가합니다.

```bash
python ml_walkforward.py --data-root data/backtest_sets_047810_5y --symbol 047810 --model-kind auto --model-top-k 2 --max-model-candidates 12 --wf-train-days 60 --wf-val-days 15 --wf-test-days 10 --wf-step-days 5 --max-policy-evals 1200 --threshold-grid 0.60,0.65,0.70,0.75,0.80,0.85 --hold-grid 10,15,20,30,40 --skip-open-grid 0,10,20 --skip-close-grid 0,10,20 --loss-streak-grid 0,2,3 --cooldown-grid 0,30,60
```

ATR 동적 라벨 + 워크포워드 예시:
```bash
python ml_walkforward.py --data-root data/backtest_sets_047810_5y --symbol 047810 --model-kind gboost --label-mode atr --atr-up-mult 2.0 --atr-down-mult 1.2 --atr-floor-pct 0.003 --wf-train-days 60 --wf-val-days 15 --wf-test-days 10 --wf-step-days 5 --max-policy-evals 800 --mdd-penalty 0.30 --min-trades 20 --max-trades 60 --threshold-grid 0.65,0.70 --hold-grid 15,20 --skip-open-grid 15,20 --skip-close-grid 15,20 --loss-streak-grid 2,3 --cooldown-grid 30,60,90
```

수익 최대화형(익절/손절/트레일링 포함) 예시:
```bash
python ml_walkforward.py --data-root data/backtest_sets_047810_5y --symbol 047810 --model-kind gboost --label-mode fixed --wf-train-days 60 --wf-val-days 15 --wf-test-days 10 --wf-step-days 5 --max-policy-evals 800 --objective-mode profit_max --mdd-penalty 0.35 --max-mdd-allowed 10 --min-profit-factor 1.0 --min-trades 10 --max-trades 45 --threshold-grid 0.72,0.75 --hold-grid 20 --skip-open-grid 20 --skip-close-grid 20 --loss-streak-grid 3 --cooldown-grid 60,90 --take-profit-grid 0.01,0.015 --stop-loss-grid 0.006,0.01 --trailing-stop-grid 0.0,0.006
```

겹침 포지션 포함(빈도 증가) 예시:
```bash
python ml_walkforward.py --data-root data/backtest_sets_047810_5y --symbol 047810 --model-kind gboost --label-mode fixed --wf-train-days 60 --wf-val-days 15 --wf-test-days 10 --wf-step-days 5 --max-policy-evals 1200 --objective-mode profit_max --mdd-penalty 0.30 --max-mdd-allowed 12 --min-profit-factor 0.95 --min-trades 80 --max-trades 180 --threshold-grid 0.52,0.55,0.58,0.60,0.62 --hold-grid 4,6,8 --skip-open-grid 0,10 --skip-close-grid 0,10 --loss-streak-grid 0,1 --cooldown-grid 0,10 --take-profit-grid 0.006,0.008,0.010 --stop-loss-grid 0.006,0.008,0.010 --trailing-stop-grid 0.0,0.004 --max-positions-grid 2,3 --position-size-grid 0.25,0.35 --min-entry-gap-grid 1,2
```

매수/매도 시각화:
```bash
python visualize_ml_trades.py --dataset-csv data/ml/047810/047810_test_ml.csv --model-path data/ml/047810/047810_model.pkl --hold-bars 20 --take-profit-pct 0.01 --output-png data/ml/047810/047810_trade_plot.png
```

