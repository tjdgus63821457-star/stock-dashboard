"""모델 검증 + 최신 예측을 계산해 data/predictions.json 으로 저장합니다.

저장 내용
- validation : 워크포워드 검증 결과(과거에 실제로 얼마나 맞았는지)
- stocks     : 종목별 최신 점수, +3%/-3% 확률, 5일 뒤 80% 가격 범위, 손절 참고선, 규칙 해당 여부
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import model as M
from src import signals as S
from src.explain import factor_table, stock_reasons
from src.fetch_prices import KST
from src.storage import ROOT

PREDICTIONS_PATH = ROOT / "data" / "predictions.json"
STOP_ATR = 2.0         # 손절 참고선 = 종가 - 2 x ATR


def _calibration_table(oos, label):
    t = M.summarize(oos, label)
    return {"auc": t["auc"], "base": t["base"], "calibration": t["calibration"]}


def validate(panel):
    """워크포워드 검증. (순위 모델, 상승/하락 확률 모델, 규칙별 성과)"""
    rank_oos = M.walk_forward_rank(panel)
    result = {"rank": M.rank_summary(rank_oos)}
    result["rules"] = {}
    for group, rules in (("buy", S.BUY_RULES), ("sell", S.SELL_RULES)):
        for key, (name, fn) in rules.items():
            ev = S.evaluate_rule(rank_oos, fn(rank_oos))
            ev["name"] = name
            ev["group"] = group
            result["rules"][key] = ev
    up = M.walk_forward(panel, "up", "logit")
    down = M.walk_forward(panel, "down", "logit")
    result["prob_up"] = _calibration_table(up, "up")
    result["prob_down"] = _calibration_table(down, "down")
    return result


def predict_latest(panel, models):
    """가장 최근 날짜의 종목별 예측."""
    last = panel["date"].max()
    cur = panel[(panel["date"] == last) & (~panel["bad"])].dropna(subset=M.FEATURES + M.RANK_FEATURES).copy()
    cur["score"] = models["rank"].predict(cur[M.RANK_FEATURES])
    cur["s_pct"] = cur["score"].rank(pct=True)
    cur["p_up"] = models["up"].predict_proba(cur[M.FEATURES])[:, 1]
    cur["p_down"] = models["down"].predict_proba(cur[M.FEATURES])[:, 1]
    cur["lo"] = cur["close"] * (1 + models["q10"] * cur["atr_pct"])
    cur["hi"] = cur["close"] * (1 + models["q90"] * cur["atr_pct"])
    cur["stop"] = cur["close"] - STOP_ATR * cur["close"] * cur["atr_pct"]
    flags = {}
    for rules in (S.BUY_RULES, S.SELL_RULES):
        for key, (_, fn) in rules.items():
            flags[key] = fn(cur)
    reasons, intercept = stock_reasons(models, cur)
    rows = []
    for i, r in cur.iterrows():
        rows.append({
            "code": r["code"], "date": r["date"],
            "score_pct": round(float(r["s_pct"]), 4),
            "p_up": round(float(r["p_up"]), 4), "p_down": round(float(r["p_down"]), 4),
            "range_lo": round(float(r["lo"]), 2), "range_hi": round(float(r["hi"]), 2),
            "stop": round(float(r["stop"]), 2),
            "flags": [k for k, v in flags.items() if bool(v.loc[i])],
            "score": reasons[i]["score"], "why": reasons[i]["why"],
        })
    return last, rows, intercept


def build(now=None, out_path=PREDICTIONS_PATH, validation_path=None):
    now = now or datetime.now(KST)
    panel = M.add_rank_columns(M.build_panel())
    models = M.fit_final(panel)
    asof, rows, intercept = predict_latest(panel, models)
    factors, span = factor_table(panel, models)
    validation = validate(panel)
    payload = {
        "built_at_kst": now.strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": asof,
        "q10": models["q10"], "q90": models["q90"], "stop_atr": STOP_ATR,
        "horizon_days": M.HORIZON, "target": M.TARGET,
        "validation": validation,
        "explain": {"intercept": round(intercept, 4), "factors": factors, "factor_span": span},
        "stocks": rows,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def main(argv=None):
    argparse.ArgumentParser(description="모델 검증과 최신 예측").parse_args(argv)
    p = build()
    r = p["validation"]["rank"]
    print(f"예측 완료: 기준일 {p['asof_date']}, {len(p['stocks'])}종목, "
          f"평가 {r['days']}일 / 80% 범위 적중 {r['range80_coverage']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
