"""가짜 데이터로 STEP 1 코드가 맞게 동작하는지 확인합니다.

실행: python -m unittest discover -s tests -v
(인터넷도, yfinance 설치도 필요 없습니다.)
"""
import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src import fetch_prices as fp
from src.storage import (
    COLUMNS,
    load_prices,
    merge_prices,
    save_prices,
    start_date_for,
)
from src.universe import Stock, load_universe


def make_df(dates, close=100.0):
    return pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        }
    )


class UniverseTest(unittest.TestCase):
    def test_universe_is_complete(self):
        stocks = load_universe()
        self.assertEqual(len(stocks), 280)
        self.assertEqual(len({s.code for s in stocks}), 280)
        self.assertEqual(len({s.sector_key for s in stocks}), 11)
        for s in stocks:
            self.assertEqual(len(s.code), 6)
            self.assertTrue(s.symbol.startswith(s.code))
            self.assertIn(s.symbol[-3:], (".KS", ".KQ"))
            self.assertIn(s.market, ("KOSPI", "KOSDAQ"))
            self.assertTrue(s.name and s.sector)

    def test_previously_missing_large_caps_are_present(self):
        by_code = {s.code: s for s in load_universe()}
        for code, name in (("000660", "SK하이닉스"), ("017670", "SK텔레콤"), ("030200", "KT")):
            self.assertEqual(by_code[code].name, name)
            self.assertEqual(by_code[code].symbol, code + ".KS")

    def test_samsung_is_kospi(self):
        by_code = {s.code: s for s in load_universe()}
        self.assertEqual(by_code["005930"].name, "삼성전자")
        self.assertEqual(by_code["005930"].symbol, "005930.KS")
        self.assertEqual(by_code["196170"].market, "KOSDAQ")


class StorageTest(unittest.TestCase):
    def test_merge_overwrites_same_day_and_sorts(self):
        old = make_df(["2026-10-01", "2026-10-02"], close=100.0)
        new = make_df(["2026-10-02", "2026-10-05"], close=105.0)
        merged = merge_prices(old, new)
        self.assertEqual(list(merged["date"]), ["2026-10-01", "2026-10-02", "2026-10-05"])
        self.assertEqual(list(merged["close"]), [100.0, 105.0, 105.0])
        self.assertEqual(list(merged.columns), COLUMNS)

    def test_merge_from_nothing(self):
        merged = merge_prices(None, make_df(["2026-10-02", "2026-10-01"]))
        self.assertEqual(list(merged["date"]), ["2026-10-01", "2026-10-02"])

    def test_merge_drops_missing_close(self):
        new = make_df(["2026-10-01", "2026-10-02"])
        new.loc[1, "close"] = float("nan")
        self.assertEqual(len(merge_prices(None, new)), 1)

    def test_start_date(self):
        today = date(2026, 10, 5)
        self.assertEqual(start_date_for(None, today), date(2025, 8, 31))
        old = make_df(["2026-10-01", "2026-10-02"])
        self.assertEqual(start_date_for(old, today), date(2026, 9, 25))

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = make_df(["2026-10-01", "2026-10-02"])
            save_prices("005930", df, base=tmp)
            back = load_prices("005930", base=tmp)
            self.assertEqual(list(back["date"]), ["2026-10-01", "2026-10-02"])
            self.assertIsNone(load_prices("000000", base=tmp))


def fake_download(symbols, dates, layout="symbol_first"):
    """yfinance가 돌려주는 표와 같은 모양의 가짜 표를 만듭니다."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    frames = {}
    for n, sym in enumerate(symbols):
        base = 100.0 + n
        for field, value in [
            ("Open", base),
            ("High", base + 2),
            ("Low", base - 2),
            ("Close", base + 1),
            ("Volume", 5000 + n),
        ]:
            key = (sym, field) if layout == "symbol_first" else (field, sym)
            frames[key] = [value] * len(idx)
    df = pd.DataFrame(frames, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class ParseTest(unittest.TestCase):
    def test_both_column_layouts(self):
        symbols = ["005930.KS", "000660.KS"]
        for layout in ("symbol_first", "field_first"):
            raw = fake_download(symbols, ["2026-10-01", "2026-10-02"], layout)
            got = fp.split_download(raw, symbols)
            self.assertEqual(set(got), set(symbols), layout)
            df = got["005930.KS"]
            self.assertEqual(list(df.columns), COLUMNS)
            self.assertEqual(list(df["date"]), ["2026-10-01", "2026-10-02"])
            self.assertEqual(df["close"].iloc[0], 101.0)
            self.assertEqual(str(df["volume"].dtype), "int64")

    def test_missing_symbol_is_skipped(self):
        raw = fake_download(["005930.KS"], ["2026-10-01"])
        got = fp.split_download(raw, ["005930.KS", "999999.KS"])
        self.assertEqual(set(got), {"005930.KS"})

    def test_empty_download(self):
        self.assertEqual(fp.split_download(pd.DataFrame(), ["005930.KS"]), {})
        self.assertEqual(fp.split_download(None, ["005930.KS"]), {})


def make_stocks(n):
    return [
        Stock(f"{i:06d}", f"종목{i}", "KOSPI", "tech", "IT·반도체", "Semiconductors", f"{i:06d}.KS")
        for i in range(1, n + 1)
    ]


class RefreshRuleTest(unittest.TestCase):
    def test_full_refresh_only_monday_morning(self):
        kst = fp.KST
        self.assertTrue(fp.should_full_refresh(datetime(2026, 10, 5, 9, 5, tzinfo=kst)))     # 월 오전
        self.assertFalse(fp.should_full_refresh(datetime(2026, 10, 5, 14, 5, tzinfo=kst)))   # 월 오후
        self.assertFalse(fp.should_full_refresh(datetime(2026, 10, 6, 9, 5, tzinfo=kst)))    # 화 오전

    def test_new_stock_starts_five_years_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen = {}

            def fetcher(symbols, start, end):
                seen["start"] = start
                return {}

            now = datetime(2026, 10, 6, 14, 0, tzinfo=fp.KST)
            fp.run(make_stocks(1), fetcher=fetcher, now=now, base=Path(tmp) / "p",
                   status_path=Path(tmp) / "s.json", batch_size=3, pause=0)
            self.assertLessEqual((now.date() - seen["start"]).days, 5 * 366)
            self.assertGreater((now.date() - seen["start"]).days, 4 * 366)


class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name) / "prices"
        self.status = Path(self.tmp.name) / "status.json"
        self.now = datetime(2026, 10, 6, 14, 0, tzinfo=fp.KST)

    def tearDown(self):
        self.tmp.cleanup()

    def run_with(self, stocks, fetcher):
        return fp.run(
            stocks, fetcher=fetcher, now=self.now, base=self.base,
            status_path=self.status, batch_size=3, pause=0,
        )

    def test_saves_and_reports_failures_and_stale(self):
        stocks = make_stocks(7)

        def fetcher(symbols, start, end):
            out = {}
            for sym in symbols:
                if sym == "000002.KS":      # 가져오지 못한 종목
                    continue
                dates = ["2026-10-01", "2026-10-02"] if sym == "000005.KS" else ["2026-10-02", "2026-10-06"]
                out[sym] = make_df(dates)
            return out

        status = self.run_with(stocks, fetcher)
        self.assertEqual(status["total"], 7)
        self.assertEqual(status["ok"], 6)
        self.assertEqual(status["failed"], ["000002"])
        self.assertEqual(status["latest_bar_date"], "2026-10-06")
        self.assertEqual(status["stale"], ["000005"])
        self.assertTrue((self.base / "000001.csv").exists())
        self.assertFalse((self.base / "000002.csv").exists())
        saved = json.loads(self.status.read_text(encoding="utf-8"))
        self.assertEqual(saved["run_at_kst"], "2026-10-06 14:00:00")

    def test_second_run_merges_without_duplicates(self):
        stocks = make_stocks(2)
        self.run_with(stocks, lambda s, a, b: {x: make_df(["2026-10-01", "2026-10-02"]) for x in s})
        self.run_with(stocks, lambda s, a, b: {x: make_df(["2026-10-02", "2026-10-06"], close=110.0) for x in s})
        df = load_prices("000001", base=self.base)
        self.assertEqual(list(df["date"]), ["2026-10-01", "2026-10-02", "2026-10-06"])
        self.assertEqual(list(df["close"]), [100.0, 110.0, 110.0])

    def test_backfill_ignores_existing_start(self):
        stocks = make_stocks(1)
        self.run_with(stocks, lambda s, a, b: {x: make_df(["2026-10-02"]) for x in s})
        seen = []

        def fetcher(symbols, start, end):
            seen.append(start)
            return {x: make_df(["2026-10-02", "2026-10-06"]) for x in symbols}

        fp.run(stocks, fetcher=fetcher, now=self.now, base=self.base,
               status_path=self.status, batch_size=3, pause=0, backfill_years=5)
        self.assertEqual(seen[0], date(2026, 10, 6) - __import__("datetime").timedelta(days=5 * 366))

    def test_retry_then_give_up(self):
        calls = []

        def broken(symbols, start, end):
            calls.append(1)
            raise RuntimeError("network down")

        result = fp._fetch_with_retry(broken, ["000001.KS"], date(2026, 1, 1), date(2026, 1, 2), tries=2, wait=0)
        self.assertEqual(result, {})
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
