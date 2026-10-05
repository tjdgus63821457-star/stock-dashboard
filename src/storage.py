"""종목별 일봉 CSV를 읽고, 새 데이터와 합쳐서 저장하는 모듈.

파일 위치: data/prices/{종목코드}.csv
열: date, open, high, low, close, volume
"""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRICES_DIR = ROOT / "data" / "prices"
COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def load_prices(code, base=PRICES_DIR):
    """저장된 일봉을 읽습니다. 파일이 없으면 None을 돌려줍니다."""
    path = Path(base) / f"{code}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"date": str})
    return df[COLUMNS]


def save_prices(code, df, base=PRICES_DIR):
    path = Path(base)
    path.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_csv(path / f"{code}.csv", index=False)


def merge_prices(old, new):
    """기존 데이터에 새 데이터를 합칩니다.

    같은 날짜가 겹치면 새 데이터가 이깁니다. 장중에 저장한 당일 봉을
    다음 실행의 확정 값으로 덮어쓰기 위한 규칙입니다.
    """
    if old is None or old.empty:
        merged = new.copy()
    else:
        merged = pd.concat([old[COLUMNS], new[COLUMNS]], ignore_index=True)
    merged = merged.dropna(subset=["close"])
    merged = merged.drop_duplicates(subset="date", keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged[COLUMNS]


def start_date_for(old, today, lookback_days=400, overlap_days=7):
    """이번에 가져올 시작 날짜를 정합니다.

    처음이면 lookback_days일 전부터, 이미 있으면 마지막 날짜에서
    overlap_days일 전부터 다시 가져와 최근 값을 바로잡습니다.
    """
    if old is None or old.empty:
        return today - timedelta(days=lookback_days)
    last = date.fromisoformat(str(old["date"].iloc[-1]))
    return last - timedelta(days=overlap_days)
