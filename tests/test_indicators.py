"""지표 계산이 맞는지 손으로 계산한 값과 비교합니다."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import indicators as ind
from src.storage import save_prices
from src.universe import Stock


def bars(closes, volume=1000, spread=1.0):
    n = len(closes)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2026-01-01", periods=n)]
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [c + spread for c in closes],
        "low": [c - spread for c in closes],
        "close": closes,
        "volume": volume,
    })


class IndicatorTest(unittest.TestCase):
    def test_moving_average_and_gap(self):
        closes = [float(x) for x in range(1, 81)]          # 1..80
        out = ind.compute(bars(closes))
        last = out.iloc[-1]
        self.assertAlmostEqual(last["ma5"], sum(range(76, 81)) / 5)
        self.assertAlmostEqual(last["ma20"], sum(range(61, 81)) / 20)
        self.assertAlmostEqual(last["ma60"], sum(range(21, 81)) / 60)
        self.assertAlmostEqual(last["gap20"], 80 / 70.5 - 1)
        self.assertEqual(last["trend_up"], 1.0)

    def test_rsi_extremes(self):
        up = ind.compute(bars([float(x) for x in range(1, 81)])).iloc[-1]
        down = ind.compute(bars([float(x) for x in range(80, 0, -1)])).iloc[-1]
        self.assertEqual(up["rsi14"], 100.0)
        self.assertAlmostEqual(down["rsi14"], 0.0, places=6)

    def test_rsi_matches_hand_calculation(self):
        closes = [100 + ((i * 7) % 11) - 5 + i * 0.1 for i in range(40)]
        out = ind.compute(bars(closes))
        # Wilder 방식을 반복문으로 직접 계산
        gains = [max(closes[i] - closes[i - 1], 0) for i in range(1, 40)]
        losses = [max(closes[i - 1] - closes[i], 0) for i in range(1, 40)]
        ag = sum(gains[:14]) / 14
        al = sum(losses[:14]) / 14
        # ewm(adjust=False)는 첫 값부터 재귀 -> 같은 식으로 처음부터 계산
        ag, al = gains[0], losses[0]
        for g, l in zip(gains[1:], losses[1:]):
            ag = ag / 14 * 13 + g / 14
            al = al / 14 * 13 + l / 14
        expected = 100 - 100 / (1 + ag / al)
        self.assertAlmostEqual(out.iloc[-1]["rsi14"], expected, places=6)

    def test_atr_constant_range(self):
        out = ind.compute(bars([100.0] * 80, spread=2.0)).iloc[-1]
        self.assertAlmostEqual(out["atr14"], 4.0)          # high-low = 4
        self.assertAlmostEqual(out["atr_pct"], 0.04)

    def test_volume_ratio_excludes_today(self):
        df = bars([100.0] * 80, volume=1000)
        df.loc[79, "volume"] = 3000
        self.assertAlmostEqual(ind.compute(df).iloc[-1]["vol_ratio"], 3.0)

    def test_zero_volume_gives_none(self):
        out = ind.compute(bars([100.0] * 80, volume=0)).iloc[-1]
        self.assertTrue(pd.isna(out["vol_ratio"]))

    def test_no_future_leak(self):
        closes = [100 + (i % 9) for i in range(100)]
        full = ind.compute(bars(closes))
        cut = ind.compute(bars(closes[:70]))
        pd.testing.assert_frame_equal(
            full.iloc[:70].reset_index(drop=True), cut.reset_index(drop=True)
        )

    def test_pullback_and_highs(self):
        closes = [100.0] * 70 + [110.0, 108.0, 107.0, 106.0, 105.0]
        out = ind.compute(bars(closes, spread=0.0)).iloc[-1]
        self.assertAlmostEqual(out["pullback5"], 105 / 110 - 1)
        self.assertAlmostEqual(out["from_high20"], 105 / 110 - 1)

    def test_short_history_returns_none(self):
        self.assertIsNone(ind.latest_row(bars([100.0] * 30)))
        self.assertIsNone(ind.latest_row(None))


class BuildLatestTest(unittest.TestCase):
    def test_build_writes_json_and_skips_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            prices = Path(tmp) / "prices"
            save_prices("000001", bars([100.0 + i for i in range(80)]), base=prices)
            save_prices("000002", bars([100.0] * 10), base=prices)
            stocks = [
                Stock("000001", "가", "KOSPI", "tech", "IT", "x", "000001.KS"),
                Stock("000002", "나", "KOSPI", "tech", "IT", "x", "000002.KS"),
                Stock("000003", "다", "KOSPI", "tech", "IT", "x", "000003.KS"),
            ]
            out = Path(tmp) / "latest.json"
            payload = ind.build_latest(stocks, prices, out)
            self.assertEqual(payload["count"], 1)
            self.assertEqual(payload["skipped_short_history"], ["000002", "000003"])
            saved = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(saved["stocks"][0]["code"], "000001")
            self.assertEqual(saved["stocks"][0]["name"], "가")


if __name__ == "__main__":
    unittest.main()
