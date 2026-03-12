# krinv-auto

## 1) 오라클 클라우드(OCI) 준비
처음 설치하는 기준으로 아래 순서대로 진행하세요.

1. OCI 로그인
2. 좌측 상단 메뉴(햄버거) 클릭
3. `Compute` -> `Instances` -> `Create instance`
4. 인스턴스 설정
   - Name: 예) `krinv-auto`
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
2. 터미널 열기 (`Ctrl + \``)
3. 아래 명령 실행

```bash
git clone https://github.com/osutaiko/krinv-auto
cd krinv-auto
pip install -r requirements.txt
```

## 3) .env 설정
프로젝트 루트(`krinv-auto`)에 `.env` 파일을 만들고 아래 값을 입력하세요.

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
- `--exit-threshold`: 매도 점수 임계값
- `--entry-confirm-bars`: 연속 매수 신호 확정 횟수
- `--exit-confirm-bars`: 연속 매도 신호 확정 횟수
- `--min-hold-bars`: 최소 보유 봉 수
- `--cooldown-bars`: 청산 후 재진입 대기 봉 수
- `--stop-loss-pct`: 손절 비율
- `--take-profit-pct`: 익절 비율
- `--position-size-pct`: 종목당 최대 비중
- `--cash-buffer-pct`: 현금 버퍼 비중
- `--min-order-krw`: 최소 주문 금액
- `--fee-rate`: 수수료 비율
- `--after-close-action`: 장외 동작 (`wait` 또는 `exit`)
- `--log-file`: 중요 이벤트 로그 파일 경로
- `--log-rotate-minutes`: 로그 파일 로테이션 주기(분)
