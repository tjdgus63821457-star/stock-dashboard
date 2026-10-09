"""점수 근거(기여 분해)가 정확한지 확인합니다."""
import unittest

import numpy as np
import pandas as pd

from src import model as M
from src.explain import INFO, stock_reasons


def fake_frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(M.RANK_FEATURES))), columns=M.RANK_FEATURES)
    for f in M.RANK_FEATURES:
        df[f[2:]] = df[f] * 0.1
    df["exn"] = df[M.RANK_FEATURES[0]] * 0.5 - df[M.RANK_FEATURES[1]] * 0.3 + rng.normal(scale=0.1, size=n)
    return df


class ExplainTest(unittest.TestCase):
    def test_contributions_sum_to_prediction(self):
        df = fake_frame()
        model = M.make_rank_model().fit(df[M.RANK_FEATURES], df["exn"])
        contrib, intercept = M.contributions(model, df)
        np.testing.assert_allclose(contrib.sum(axis=1) + intercept, model.predict(df[M.RANK_FEATURES]), atol=1e-9)

    def test_reasons_pick_largest_and_signs(self):
        df = fake_frame()
        model = M.make_rank_model().fit(df[M.RANK_FEATURES], df["exn"])
        reasons, _ = stock_reasons({"rank": model}, df)
        contrib, _ = M.contributions(model, df)
        for n, idx in enumerate(df.index):
            why = reasons[idx]["why"]
            self.assertLessEqual(len(why), 5)
            pos = [w for w in why if w[1] > 0]
            if pos:
                best = M.RANK_FEATURES[int(np.argmax(contrib[n]))][2:]
                self.assertEqual(pos[0][0], best)
            self.assertTrue(all(w[1] != 0 for w in why))

    def test_price_level_features_excluded_and_all_have_labels(self):
        for k in ("ma5", "ma20", "ma60", "atr14"):
            self.assertNotIn("r_" + k, M.RANK_FEATURES)
            self.assertNotIn(k, M.FEATURES)
        for f in M.RANK_FEATURES:
            self.assertIn(f[2:], INFO)


if __name__ == "__main__":
    unittest.main()
