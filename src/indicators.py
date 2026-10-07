"""일봉으로 기술 지표를 계산하는 모듈.

compute(df)       : 일봉 한 종목 -> 날짜별 지표 표 (STEP 3 예측 모델도 이 표를 씁니다)
latest_row(df)    : 가장 최근 날짜의 지표만 뽑기
build_latest()    : 전 종목의 최근 지표를 data/latest.json 으로 저장

주의: 모든 지표는 '그 날짜까지의 값'만 사용합니다(미래 값 사용 없음).
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.fetch_prices import KST
from src.storage import PRICES_DIR, ROOT, load_prices
from src.universe import load_universe

LATEST_PATH = ROOT / "data" / "latest.json"
MIN_BARS = 61          # 60일 이동평균을 계산하려면 최소 이만큼 필요
INDICATOR_COLUMNS = [
    "ma5", "ma20", "ma60", "gap20", "gap60", "trend_up",
    "rsi14", "atr14", "atr_pct", "vol_ratio",
    "ret1", "ret5", "ret20", "from_high20", "from_high60",
    "pullback5", "macd_hist",
]


def _rsi(close, n=14):
    """Wilder 방식 RSI."""
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # 하락이 전혀 없으면(avg_loss == 0) 100
    rsi = rsi.where(~((avg_loss == 0) & avg_gain.notna()), 100.0)
    return rsi


def compute(df):
    """일봉(date, open, high, low, close, volume) -> 지표 표."""
    d = df.sort_values("date").reset_index(drop=True)
    close, high, low, vol = d["close"], d["high"], d["low"], d["volume"]
    out = pd.DataFrame({"date": d["date"]})

    out["ma5"] = close.rolling(5).mean()
    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["gap20"] = close / out["ma20"] - 1          # 20일선 대비 위치
    out["gap60"] = close / out["ma60"] - 1
    out["trend_up"] = (out["ma20"] > out["ma60"]).astype(float).where(out["ma60"].notna())

    out["rsi14"] = _rsi(close)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr14"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["atr_pct"] = out["atr14"] / close           # 하루 평균 변동폭(%)

    # 거래량 비율: 오늘 거래량 / 직전 20일 평균 (오늘 값은 평균에 넣지 않음)
    avg_vol = vol.shift(1).rolling(20).mean()
    out["vol_ratio"] = (vol / avg_vol).where(avg_vol > 0)

    out["ret1"] = close.pct_change(1)
    out["ret5"] = close.pct_change(5)
    out["ret20"] = close.pct_change(20)

    out["from_high20"] = close / high.rolling(20).max() - 1
    out["from_high60"] = close / high.rolling(60).max() - 1
    out["pullback5"] = close / close.rolling(5).max() - 1   # 최근 5일 고점 대비 되돌림

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close

    out["close"] = close
    return out[["date", "close"] + INDICATOR_COLUMNS]


def latest_row(df):
    """가장 최근 날짜의 지표 한 줄(dict). 데이터가 모자라면 None."""
    if df is None or len(df) < MIN_BARS:
        return None
    row = compute(df).iloc[-1]
    result = {"date": row["date"], "close": _num(row["close"])}
    for col in INDICATOR_COLUMNS:
        result[col] = _num(row[col])
    return result


def _num(x):
    if x is None or pd.isna(x):
        return None
    return round(float(x), 6)


def build_latest(stocks=None, prices_dir=PRICES_DIR, out_path=LATEST_PATH, now=None):
    """전 종목의 최근 지표를 모아 JSON 으로 저장합니다."""
    stocks = stocks if stocks is not None else load_universe()
    now = now or datetime.now(KST)
    rows, skipped = [], []
    for s in stocks:
        df = load_prices(s.code, base=prices_dir)
        r = latest_row(df)
        if r is None:
            skipped.append(s.code)
            continue
        r.update(code=s.code, name=s.name, market=s.market,
                 sector_key=s.sector_key, sector=s.sector)
        rows.append(r)
    payload = {
        "built_at_kst": now.strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(rows),
        "skipped_short_history": skipped,
        "stocks": rows,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def main(argv=None):
    argparse.ArgumentParser(description="전 종목 최근 지표 계산").parse_args(argv)
    p = build_latest()
    print(f"지표 계산 완료: {p['count']}종목, 이력 부족 {len(p['skipped_short_history'])}종목")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
