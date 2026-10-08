# stock-dashboard

한국 주식 스윙(며칠 보유) 판단용 대시보드의 데이터 수집부입니다.
평일 오전 9시와 오후 2시(한국시간)에 GitHub Actions가 자동으로 실행됩니다.

**이 저장소에는 주문 기능이 없습니다.** 계좌 연동이나 자동 매매 코드를 넣지 않습니다.
신호와 예측은 참고용 통계 추정치이고, 최종 판단과 손실 책임은 본인에게 있습니다.

## 진행 단계

| 단계 | 내용 | 상태 |
|---|---|---|
| STEP 1 | 저장소 구조, 종목 목록, 일봉 수집, 자동 실행 설정 | 완료 |
| STEP 2 | 지표 계산 (이동평균, RSI, ATR, 거래량 등) | 완료 |
| STEP 3 | 예측 모델과 워크포워드 검증 | 완료 |
| STEP 4 | 매수·매도 신호와 대시보드 페이지 생성 | 완료 (docs/index.html) |
| STEP 5 | GitHub Pages 공개, 휴대폰 홈 화면 추가 | 예정 |

각 단계는 확인을 받은 뒤 다음으로 넘어갑니다.

## 폴더 구조

```
data/universe.csv        종목 277개 (코드, 이름, 섹터, 가격 조회 심볼)
data/prices/{코드}.csv    종목별 일봉 (date, open, high, low, close, volume)
data/status.json         마지막 실행 결과 (성공·실패 종목, 최신 봉 날짜)
src/universe.py          종목 목록 읽기
src/storage.py           일봉 저장과 병합
src/fetch_prices.py      일봉 가져오기 (yfinance)
src/indicators.py        지표 계산, 전 종목 최근 지표 -> data/latest.json
src/model.py, predict.py  예측 모델, 워크포워드 검증 -> data/predictions.json
src/signals.py, dashboard.py  신호 규칙, 화면 생성 -> docs/index.html
data/latest.json         종목별 최근 지표 (이동평균, RSI, ATR, 거래량 비율 등)
tests/test_pipeline.py   인터넷 없이 도는 시험
.github/workflows/update.yml   평일 09:00, 14:00 KST 자동 실행
```

## 내 컴퓨터에서 실행하기 (선택)

```
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.fetch_prices --limit 10
```

## 알려진 한계

- 가격은 Yahoo Finance(yfinance)에서 가져옵니다. 약 20분 지연될 수 있고, 일부 종목은 조회되지 않을 수 있습니다. 조회 실패 종목은 `data/status.json`의 `failed`에 기록됩니다.
- 14시 실행분의 당일 봉은 장중 값입니다. 다음 실행에서 확정 값으로 덮어씁니다.
- 수정주가 기준이라 배당·분할이 생기면 과거 값이 바뀝니다. 최근 7일만 다시 받으므로, 이후 단계에서 주 1회 전체 재수집을 추가할 예정입니다.
- 한국 공휴일 판정은 아직 없습니다. 휴장일에는 새 봉이 생기지 않고, `latest_bar_date`로 확인할 수 있습니다.
- GitHub Actions의 예약 실행은 정각보다 몇 분에서 수십 분 늦을 수 있습니다.
