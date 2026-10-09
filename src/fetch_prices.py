"""한국 종목 일봉을 받아 data/prices/ 에 저장합니다.

실행: python -m src.fetch_prices            (전체 종목)
      python -m src.fetch_prices --limit 10 (앞의 10종목만, 시험용)

결과 요약은 data/status.json 에 기록됩니다.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from .storage import (
    PRICES_DIR,
    ROOT,
    load_prices,
    merge_prices,
    save_prices,
    start_date_for,
)
from .universe import load_universe

KST = ZoneInfo("Asia/Seoul")
STATUS_PATH = ROOT / "data" / "status.json"
BATCH_SIZE = 40
MIN_OK_RATIO = 0.8
DATA_SOURCE = "yfinance"
FIELDS = ["open", "high", "low", "close", "volume"]


def _clean(sub):
    """종목 하나의 원본 표를 date/open/high/low/close/volume 형태로 정리합니다."""
    df = sub.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if not all(c in df.columns for c in FIELDS):
        return None
    df = df[FIELDS].dropna(subset=["close"])
    if df.empty:
        return None
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.reset_index(drop=True)
    df.insert(0, "date", idx.strftime("%Y-%m-%d"))
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float).round(2)
    df["volume"] = df["volume"].fillna(0).astype("int64")
    return df


def split_download(raw, symbols):
    """yfinance가 돌려준 표를 심볼별 일봉으로 나눕니다.

    표 구조가 (심볼, 항목)인지 (항목, 심볼)인지는 버전에 따라 달라서
    어느 쪽이든 읽을 수 있게 만들었습니다.
    """
    out = {}
    if raw is None or getattr(raw, "empty", True):
        return out
    cols = raw.columns
    if isinstance(cols, pd.MultiIndex):
        level = 0 if set(cols.get_level_values(0)) & set(symbols) else 1
        present = set(cols.get_level_values(level))
        for sym in symbols:
            if sym not in present:
                continue
            df = _clean(raw.xs(sym, axis=1, level=level))
            if df is not None:
                out[sym] = df
    elif len(symbols) == 1:
        df = _clean(raw)
        if df is not None:
            out[symbols[0]] = df
    return out


def fetch_batch(symbols, start, end):
    """심볼 여러 개의 일봉을 한 번에 받습니다. (end 날짜는 포함되지 않음)"""
    import yfinance as yf  # 시험할 때는 설치되어 있지 않아도 되도록 여기서 불러옵니다.

    raw = yf.download(
        symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    return split_download(raw, symbols)


def _fetch_with_retry(fetcher, symbols, start, end, tries=2, wait=5.0):
    for attempt in range(tries):
        try:
            return fetcher(symbols, start, end)
        except Exception as exc:  # 네트워크 오류 등은 한 번 더 시도합니다.
            print(f"  가져오기 실패({attempt + 1}/{tries}): {exc}", file=sys.stderr)
            if attempt + 1 < tries:
                time.sleep(wait)
    return {}


NEW_STOCK_YEARS = 5      # 처음 받는 종목은 5년치를 한 번에 받습니다
FULL_REFRESH_WEEKDAY = 0  # 월요일 오전 첫 실행은 전체 재수집(배당·분할로 바뀐 수정주가 반영)


def should_full_refresh(now):
    """월요일 10시 이전(첫 실행)이면 전체 재수집을 합니다.

    수정주가는 배당·분할이 생기면 과거 값이 전부 바뀌므로, 최근 7일만 덧붙이면
    이어붙인 지점에서 가격이 어긋납니다. 주 1회 전체를 다시 받아 이를 막습니다.
    """
    return now.weekday() == FULL_REFRESH_WEEKDAY and now.hour < 10


def run(
    stocks,
    fetcher=fetch_batch,
    now=None,
    base=PRICES_DIR,
    status_path=STATUS_PATH,
    batch_size=BATCH_SIZE,
    pause=1.0,
    backfill_years=0,
):
    """종목 목록의 일봉을 받아 저장하고, 실행 결과 요약(dict)을 돌려줍니다."""
    now = now or datetime.now(KST)
    today = now.date()
    olds = {s.code: load_prices(s.code, base) for s in stocks}
    if backfill_years > 0:
        # 전체 재수집: 기존 파일과 상관없이 N년 전부터 다시 받아 수정주가를 맞춥니다.
        first = today - timedelta(days=int(backfill_years * 366))
        starts = {s.code: first for s in stocks}
    else:
        new_first = today - timedelta(days=int(NEW_STOCK_YEARS * 366))
        starts = {
            s.code: (new_first if olds[s.code] is None else start_date_for(olds[s.code], today))
            for s in stocks
        }

    ok, failed, last_dates = [], [], {}
    for i in range(0, len(stocks), batch_size):
        batch = stocks[i : i + batch_size]
        symbols = [s.symbol for s in batch]
        start = min(starts[s.code] for s in batch)
        end = today + timedelta(days=1)
        print(f"[{i + 1}-{i + len(batch)} / {len(stocks)}] 가져오는 중...")
        got = _fetch_with_retry(fetcher, symbols, start, end)
        for s in batch:
            df = got.get(s.symbol)
            if df is None or df.empty:
                failed.append(s.code)
                continue
            merged = merge_prices(olds[s.code], df)
            save_prices(s.code, merged, base)
            ok.append(s.code)
            last_dates[s.code] = str(merged["date"].iloc[-1])
        if pause and i + batch_size < len(stocks):
            time.sleep(pause)

    latest = max(last_dates.values()) if last_dates else None
    stale = sorted(c for c, d in last_dates.items() if latest and d < latest)
    status = {
        "run_at_kst": now.strftime("%Y-%m-%d %H:%M:%S"),
        "today_kst": today.isoformat(),
        "data_source": DATA_SOURCE,
        "total": len(stocks),
        "ok": len(ok),
        "failed": sorted(failed),
        "latest_bar_date": latest,
        "stale": stale,
    }
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description="한국 종목 일봉 수집")
    parser.add_argument("--limit", type=int, default=0, help="앞에서부터 N종목만 (0이면 전체)")
    parser.add_argument("--backfill-years", type=float, default=0,
                        help="N년 전부터 전체를 다시 받기 (0이면 최근분만 갱신)")
    args = parser.parse_args(argv)

    stocks = load_universe()
    if args.limit > 0:
        stocks = stocks[: args.limit]

    years = args.backfill_years
    if years == 0 and should_full_refresh(datetime.now(KST)):
        years = NEW_STOCK_YEARS
        print("월요일 첫 실행: 전체 재수집으로 수정주가를 다시 맞춥니다.")
    status = run(stocks, backfill_years=years)
    print(
        f"완료: {status['ok']}/{status['total']}종목 저장, "
        f"실패 {len(status['failed'])}종목, 최신 봉 날짜 {status['latest_bar_date']}"
    )
    if status["failed"]:
        print("실패 종목코드:", ", ".join(status["failed"]))
    if status["stale"]:
        print("최신 날짜보다 늦은(거래정지·지연 의심) 종목코드:", ", ".join(status["stale"]))

    ratio = status["ok"] / status["total"] if status["total"] else 0
    if ratio < MIN_OK_RATIO:
        print(f"성공 비율 {ratio:.0%}가 기준 {MIN_OK_RATIO:.0%} 미만이라 실패로 처리합니다.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
