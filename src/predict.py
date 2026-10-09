"""모델 검증 + 최신 예측을 계산해 data/predictions.json 으로 저장합니다.

저장 내용
- validation : 워크포워드 검증 결과(과거에 실제로 얼마나 맞았는지)
- stocks     : 종목별 최신 점수, +3%/-3% 확률, 5일 뒤 90% 가격 범위, 5일 안 손절선 터치 확률, 손절 참고선, 규칙 해당 여부
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
STOP_ATR = M.STOP_ATR  # 손절 참고선 = 종가 - 2 x ATR


def model_version():
    """모델 정의 파일의 해시. 기록된 예측이 어떤 모델에서 나왔는지 구분하는 데 씁니다."""
    import hashlib
    h = hashlib.sha1()
    for n in ("model.py", "explain.py", "signals.py", "indicators.py"):
        h.update((Path(__file__).with_name(n)).read_bytes())
    return h.hexdigest()[:8]


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
    oos = {k: M.walk_forward(panel, k, "logit") for k in ("up", "down", "stop")}
    for k in oos:
        result["prob_" + k] = _calibration_table(oos[k], k)
        result["prob_" + k]["isotonic"] = M.calibration_check(oos[k], k)
    return result, oos


def predict_latest(panel, models, calibrators=None):
    """가장 최근 날짜의 종목별 예측."""
    last = panel["date"].max()
    cur = panel[(panel["date"] == last) & (~panel["bad"])].dropna(subset=M.FEATURES + M.RANK_FEATURES).copy()
    cur["score"] = models["rank"].predict(cur[M.RANK_FEATURES])
    cur["s_pct"] = cur["score"].rank(pct=True)
    cur["p_up"] = models["up"].predict_proba(cur[M.FEATURES])[:, 1]
    cur["p_down"] = models["down"].predict_proba(cur[M.FEATURES])[:, 1]
    cur["p_stop"] = models["stop"].predict_proba(cur[M.FEATURES])[:, 1]
    for k, cal in (calibrators or {}).items():     # 보정이 검증에서 도움이 된 확률만 보정
        cur["p_" + k] = cal.predict(cur["p_" + k].values)
    cur["lo"] = cur["close"] * (1 + models["q05"] * cur["atr_pct"])      # 90% 범위
    cur["hi"] = cur["close"] * (1 + models["q95"] * cur["atr_pct"])
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
            "p_stop": round(float(r["p_stop"]), 4),
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
    validation, oos = validate(panel)
    calibrators = {k: M.fit_calibrator(oos[k], k) for k in oos if validation["prob_" + k]["isotonic"]["use"]}
    asof, rows, intercept = predict_latest(panel, models, calibrators)
    factors, span = factor_table(panel, models)
    for k in oos:
        validation["prob_" + k]["calibrated"] = k in calibrators
    payload = {
        "built_at_kst": now.strftime("%Y-%m-%d %H:%M:%S"),
        "asof_date": asof,
        "q05": models["q05"], "q95": models["q95"], "q10": models["q10"], "q90": models["q90"], "stop_atr": STOP_ATR,
        "horizon_days": M.HORIZON, "target": M.TARGET,
        "model_version": model_version(),
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
          f"평가 {r['days']}일 / 90% 범위 적중 {r['range90_coverage']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
