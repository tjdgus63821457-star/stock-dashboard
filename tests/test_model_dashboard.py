"""모델 보조 함수, 규칙, 대시보드 생성을 가짜 데이터로 확인합니다."""
import json
import unittest

import numpy as np
import pandas as pd

from src import dashboard as dash
from src import model as M
from src import signals as S


def oos_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "date": np.repeat([f"2026-01-{d:02d}" for d in range(1, 21)], n // 20),
        "fold": np.repeat([0, 1], n // 2),
        "fwd5": rng.normal(0.01, 0.05, n),
        "s_pct": rng.random(n),
        "gap20": rng.normal(0, 0.05, n), "rsi14": rng.uniform(20, 90, n),
        "vol_ratio": rng.uniform(0.3, 2, n), "ret1": rng.normal(0, 0.02, n),
        "ret5": rng.normal(0, 0.05, n), "pullback5": -rng.random(n) * 0.05,
        "atr_pct": np.full(n, 0.03),
    })
    df["up"] = (df["fwd5"] >= 0.03).astype(float)
    df["down"] = (df["fwd5"] <= -0.03).astype(float)
    return df


class RuleTest(unittest.TestCase):
    def test_masks_follow_definitions(self):
        d = pd.DataFrame({
            "s_pct": [0.95, 0.85, 0.1], "gap20": [0.02, -0.01, -0.02], "rsi14": [60, 80, 40],
            "vol_ratio": [1.2, 1.1, 0.5], "ret1": [0.01, -0.02, -0.01], "ret5": [0, 0, 0],
            "pullback5": [-0.03, 0, 0], "atr_pct": [0.03] * 3,
        })
        self.assertEqual(list(S.rule_buy_watch(d)), [True, False, False])
        self.assertEqual(list(S.rule_break20(d)), [False, True, False])
        self.assertEqual(list(S.rule_sell_watch(d)), [False, True, True])
        self.assertEqual(list(S.rule_overheat(d)), [False, True, False])
        self.assertEqual(list(S.rule_rebound(d)), [True, False, False])

    def test_evaluate_rule_compares_with_same_day_average(self):
        d = oos_frame()
        res = S.evaluate_rule(d, d["s_pct"] >= 0.5)
        self.assertGreater(res["n"], 30)
        self.assertEqual(res["folds"], 2)
        self.assertAlmostEqual(res["base_up3"], d["up"].mean())
        self.assertEqual(S.evaluate_rule(d, d["s_pct"] > 2), {"n": 0})


class ModelHelpersTest(unittest.TestCase):
    def test_rank_columns_and_excess(self):
        p = pd.DataFrame({
            "date": ["a", "a", "a", "b", "b", "b"], "code": list("xyzxyz"),
            "fwd5": [0.01, 0.02, 0.03, 0.0, 0.0, 0.3],
        })
        for c in M.INDICATOR_COLUMNS:
            p[c] = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]
        p["atr_pct"] = 0.02          # 지표 열을 채운 뒤에 변동폭을 고정
        out = M.add_rank_columns(p)
        self.assertEqual(list(out.loc[:2, "r_rsi14"].round(3)), [0.333, 0.667, 1.0])
        self.assertAlmostEqual(out.loc[1, "ex"], 0.0)            # 그날 중앙값과 같음
        self.assertAlmostEqual(out.loc[5, "exn"], 5.0)            # 변동성 대비 값은 +-5로 제한


class DashboardTest(unittest.TestCase):
    def test_render_embeds_json_and_escapes_script_end(self):
        data = {"stocks": [{"name": "</script><b>"}], "validation": {}}
        html = dash.render(data)
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertNotIn("</script><b>", html)
        blob = html.split('type="application/json">')[1].split("</script>")[0]
        self.assertEqual(json.loads(blob)["stocks"][0]["name"], "</script><b>")

    def test_fragment_has_no_document_tags(self):
        html = dash.render({"stocks": [], "validation": {}}, fragment=True)
        self.assertNotIn("<!doctype", html.lower())
        self.assertIn("<title>스윙 시황판</title>", html)


if __name__ == "__main__":
    unittest.main()
