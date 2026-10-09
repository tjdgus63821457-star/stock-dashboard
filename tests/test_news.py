import json
import tempfile
import unittest
from pathlib import Path

from src.news import level, load_news


def entry(cur=0.0, base=0.0, mom=0.0, z1=0.0, zq=0.0, docs=100, events=()):
    return {"sentiment": {"current": cur, "baseline": base, "momentum": mom, "z1mo": z1, "z1qt": zq},
            "doc_count": docs, "events": list(events)}


class NewsLevelTest(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(level(entry(0.01, 0.08, -0.07, -0.67, -0.66)), "ok")

    def test_warn_on_unusual_zscore(self):
        self.assertEqual(level(entry(-0.58, -0.29, -0.29, -2.18, -2.46)), "warn")
        self.assertEqual(level(entry(z1=-1.0, zq=-2.0)), "warn")

    def test_caution_on_momentum_or_current(self):
        self.assertEqual(level(entry(-0.18, -0.08, -0.10, -0.79, -0.56)), "caution")
        self.assertEqual(level(entry(cur=-0.25, mom=0.0)), "caution")

    def test_events_raise_level_but_never_lower(self):
        self.assertEqual(level(entry(events=[{"sev": "caution"}])), "caution")
        self.assertEqual(level(entry(events=[{"sev": "warn"}])), "warn")
        self.assertEqual(level(entry(z1=-3, events=[{"sev": "caution"}])), "warn")

    def test_thin_coverage_is_none(self):
        self.assertEqual(level(entry(docs=0, z1=-5)), "none")
        self.assertEqual(level(entry(docs=4)), "none")

    def test_missing_zscores_ok(self):
        e = entry()
        e["sentiment"]["z1mo"] = None
        e["sentiment"]["z1qt"] = None
        self.assertEqual(level(e), "ok")

    def test_load_news_adds_level_and_handles_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_news(Path(tmp) / "none.json")["stocks"], {})
            p = Path(tmp) / "n.json"
            p.write_text(json.dumps({"updated_kst": "x", "stocks": {"000001": entry(z1=-3)}}), encoding="utf-8")
            self.assertEqual(load_news(p)["stocks"]["000001"]["level"], "warn")

    def test_shipped_news_file_levels(self):
        n = load_news()["stocks"]
        self.assertEqual(n["010950"]["level"], "warn")      # S-Oil: z점수 -2 이하
        self.assertEqual(n["082920"]["level"], "none")      # 비츠로셀: 기사 없음
        self.assertEqual(n["402340"]["level"], "caution")


if __name__ == "__main__":
    unittest.main()
