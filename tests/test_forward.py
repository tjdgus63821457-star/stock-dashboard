import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import forward as F
from src.storage import save_prices


def bars(closes, start="2026-10-01"):
    d = pd.bdate_range(start, periods=len(closes)).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": d, "open": closes, "high": [c * 1.01 for c in closes],
                         "low": [c * 0.99 for c in closes], "close": closes, "volume": 1000})


class ForwardTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = self.tmp / "log.csv"
        self.pred = {"asof_date": "2026-10-01", "model_version": "abc", "stocks": [
            {"code": "000001", "score_pct": 1.0, "p_up": .3, "p_down": .1, "p_stop": .2, "range_lo": 90, "range_hi": 110, "stop": 95},
            {"code": "000002", "score_pct": .5, "p_up": .2, "p_down": .1, "p_stop": .1, "range_lo": 90, "range_hi": 110, "stop": 95}]}
        self.latest = {"stocks": [{"code": "000001", "close": 100.0}, {"code": "000002", "close": 100.0}]}

    def test_only_finalized_days_are_logged_once(self):
        self.assertEqual(F.append_log(self.pred, self.latest, "2026-10-01", self.log), 0)   # 당일 봉은 미확정
        self.assertEqual(F.append_log(self.pred, self.latest, "2026-10-02", self.log), 2)
        self.assertEqual(F.append_log(self.pred, self.latest, "2026-10-05", self.log), 0)   # 중복 방지

    def test_score_waits_for_five_bars_then_scores(self):
        F.append_log(self.pred, self.latest, "2026-10-02", self.log)
        log = pd.read_csv(self.log, dtype={"asof_date": str, "code": str})
        save_prices("000001", bars([100, 101, 102, 103, 104, 108]), self.tmp)
        save_prices("000002", bars([100, 99, 98, 97, 96, 94]), self.tmp)
        self.assertEqual(F.score(log, self.tmp)["scored_days"], 1)
        save_prices("000001", bars([100, 101, 102, 103, 104]), self.tmp)
        save_prices("000002", bars([100, 99, 98, 97, 96]), self.tmp)
        self.assertEqual(F.score(log, self.tmp)["scored_days"], 0)   # 5거래일 미경과
        save_prices("000001", bars([100, 101, 102, 103, 104, 108]), self.tmp)
        save_prices("000002", bars([100, 99, 98, 97, 96, 94]), self.tmp)
        s = F.score(log, self.tmp, top_n=1)
        self.assertGreater(s["top_vs_universe_5d"], 0)   # 1위(000001)가 올라 평균보다 높음
        self.assertEqual(s["stop_actual"], 0.5)          # 000002만 95 아래로 내려감


if __name__ == "__main__":
    unittest.main()
