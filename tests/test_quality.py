import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import quality as Q
from src.storage import save_prices
from src.universe import Stock


def bars(n=320, start="2025-01-01", close=100.0, vol=1000):
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, periods=n)]
    return pd.DataFrame({"date": dates, "open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": vol})


def stock(code):
    return Stock(code, "x" + code, "KOSPI", "tech", "IT", "x", code + ".KS")


class PriceCheckTest(unittest.TestCase):
    def run_checks(self, frames):
        with tempfile.TemporaryDirectory() as tmp:
            for code, df in frames.items():
                if df is not None:
                    save_prices(code, df, base=tmp)
            return Q.check_prices([stock(c) for c in frames], tmp)

    def test_clean_data_has_no_issues(self):
        issues, summary = self.run_checks({"000001": bars(), "000002": bars()})
        self.assertEqual(issues, {})
        self.assertEqual(summary["with_issues"], 0)

    def test_detects_each_problem(self):
        short = bars(100)
        stale = bars(300)                                    # 다른 종목보다 일찍 끝남
        full = bars(320)
        jump = bars(320)
        jump.loc[310, "close"] = 200.0
        jump.loc[310, "high"] = 201.0
        zero = bars(320)
        zero.loc[zero.index[-10:], "volume"] = 0
        badpx = bars(320)
        badpx.loc[100, "low"] = 0.0
        hl = bars(320)
        hl.loc[50, "high"] = 50.0
        issues, _ = self.run_checks({"000001": full, "000002": short, "000003": stale, "000004": jump,
                                     "000005": zero, "000006": badpx, "000007": hl, "000008": None})
        self.assertIn("이력 부족", issues["000002"][0])
        self.assertTrue(any("갱신 지연" in m for m in issues["000003"]))
        self.assertTrue(any("30% 초과" in m for m in issues["000004"]))
        self.assertTrue(any("거래량 0" in m for m in issues["000005"]))
        self.assertTrue(any("0 이하" in m for m in issues["000006"]))
        self.assertTrue(any("고가·저가" in m for m in issues["000007"]))
        self.assertEqual(issues["000008"], ["가격 파일 없음"])
        self.assertNotIn("000001", issues)


def write_outputs(root, mutate=None):
    (Path(root) / "data").mkdir(parents=True, exist_ok=True)
    status = {"latest_bar_date": "2026-10-08", "ok": 2, "total": 2}
    latest = {"stocks": [{"code": "000001", "close": 100.0}, {"code": "000002", "close": 50.0}]}
    pred = {"asof_date": "2026-10-08",
            "validation": {"rank": {"range90_coverage": 0.9}},
            "stocks": [
                {"code": "000001", "score_pct": 0.5, "score": 0.1, "p_up": 0.4, "p_down": 0.3, "p_stop": 0.1,
                 "range_lo": 90.0, "range_hi": 112.0, "stop": 92.0},
                {"code": "000002", "score_pct": 1.0, "score": 0.2, "p_up": 0.5, "p_down": 0.2, "p_stop": 0.1,
                 "range_lo": 45.0, "range_hi": 56.0, "stop": 46.0}]}
    if mutate:
        mutate(status, latest, pred)
    for name, obj in (("status", status), ("latest", latest), ("predictions", pred)):
        (Path(root) / "data" / f"{name}.json").write_text(json.dumps(obj), encoding="utf-8")


class OutputCheckTest(unittest.TestCase):
    def check(self, mutate=None):
        with tempfile.TemporaryDirectory() as tmp:
            write_outputs(tmp, mutate)
            return Q.check_outputs(tmp, [stock("000001"), stock("000002")])

    def test_consistent_outputs_pass(self):
        errors, warnings = self.check()
        self.assertEqual((errors, warnings), ([], []))

    def test_date_mismatch(self):
        errors, _ = self.check(lambda s, l, p: s.update(latest_bar_date="2026-10-07"))
        self.assertTrue(any("기준일" in e for e in errors))

    def test_probability_out_of_range(self):
        errors, _ = self.check(lambda s, l, p: p["stocks"][0].update(p_stop=1.4))
        self.assertTrue(any("p_stop" in e for e in errors))

    def test_range_must_contain_price_and_stop_below(self):
        errors, _ = self.check(lambda s, l, p: p["stocks"][0].update(range_lo=101.0))
        self.assertTrue(any("범위" in e for e in errors))
        errors, _ = self.check(lambda s, l, p: p["stocks"][1].update(stop=51.0))
        self.assertTrue(any("손절선" in e for e in errors))

    def test_unknown_or_duplicate_codes_and_missing_stocks(self):
        errors, _ = self.check(lambda s, l, p: p["stocks"].append(dict(p["stocks"][0])))
        self.assertTrue(any("중복" in e for e in errors))
        errors, _ = self.check(lambda s, l, p: p["stocks"][0].update(code="999999"))
        self.assertTrue(any("목록에 없는" in e for e in errors))
        errors, _ = self.check(lambda s, l, p: p.update(stocks=p["stocks"][:1]))
        self.assertTrue(any("너무 적습니다" in e for e in errors))

    def test_nan_in_json_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_outputs(tmp)
            f = Path(tmp) / "data" / "predictions.json"
            f.write_text(f.read_text(encoding="utf-8").replace('"score": 0.1', '"score": NaN'), encoding="utf-8")
            errors, _ = Q.check_outputs(tmp, [stock("000001"), stock("000002")])
            self.assertTrue(errors and "NaN" in errors[0])

    def test_coverage_warning_and_error(self):
        _, warnings = self.check(lambda s, l, p: p["validation"]["rank"].update(range90_coverage=0.82))
        self.assertTrue(any("90%" in w for w in warnings))
        errors, _ = self.check(lambda s, l, p: p["validation"]["rank"].update(range90_coverage=0.2))
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
