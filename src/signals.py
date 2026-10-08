"""매수 관심 / 매도 경계 규칙. 검증(과거 평가 구간)과 실제 신호 생성에 같은 함수를 씁니다.

입력 표에 필요한 열: s_pct(점수 백분위), close, ma20, gap20, rsi14, vol_ratio, ret1, ret5,
pullback5, from_high20, atr_pct
"""
import numpy as np

# 매수 쪽 규칙 -------------------------------------------------------------
def rule_score_top(d):
    """순위 점수 상위 10%."""
    return d["s_pct"] >= 0.90


def rule_rebound(d):
    """눌림 후 반등: 최근 5일 고점에서 2%↑ 눌렸다가 오늘 반등, 20일선 위, 거래량 평균 이상."""
    return (d["pullback5"] <= -0.02) & (d["ret1"] > 0) & (d["gap20"] > 0) & (d["vol_ratio"] >= 1.0)


def rule_buy_watch(d):
    """매수 관심 = 점수 상위 20% + 20일선 위 + 과열 아님(RSI 75 이하)."""
    return (d["s_pct"] >= 0.80) & (d["gap20"] > 0) & (d["rsi14"] <= 75)


# 매도 쪽 규칙 -------------------------------------------------------------
def rule_break20(d):
    """20일선 이탈: 오늘 20일선 아래, 하락, 거래량 평균 이상."""
    return (d["gap20"] < 0) & (d["ret1"] < 0) & (d["vol_ratio"] >= 1.0)


def rule_overheat(d):
    """과열: RSI 80 이상."""
    return d["rsi14"] >= 80


def rule_sharp_drop(d):
    """급락: 5일 수익률이 일일변동폭의 -3배 이하."""
    return d["ret5"] <= -3 * d["atr_pct"]


def rule_sell_watch(d):
    """매도 경계 = 20일선 이탈 또는 (점수 하위 20% 이면서 20일선 아래)."""
    return rule_break20(d) | ((d["s_pct"] <= 0.20) & (d["gap20"] < 0))


BUY_RULES = {
    "score_top": ("점수 상위 10%", rule_score_top),
    "rebound": ("눌림 후 반등", rule_rebound),
    "buy_watch": ("매수 관심", rule_buy_watch),
}
SELL_RULES = {
    "break20": ("20일선 이탈(거래량 동반)", rule_break20),
    "overheat": ("과열(RSI 80↑)", rule_overheat),
    "sharp_drop": ("5일 급락", rule_sharp_drop),
    "sell_watch": ("매도 경계", rule_sell_watch),
}


def evaluate_rule(oos, mask, cost=0.0025):
    """규칙에 걸린 종목-날짜의 5일 뒤 수익률을 '같은 날 전체 평균'과 비교합니다."""
    d = oos.assign(um=oos.groupby("date")["fwd5"].transform("mean"))
    sel = d[mask]
    if len(sel) < 30:
        return {"n": int(len(sel))}
    diff = sel["fwd5"] - sel["um"]
    per_fold = diff.groupby(sel["fold"]).mean()
    return {
        "n": int(len(sel)),
        "mean_5d": float(sel["fwd5"].mean()),
        "universe_mean_5d": float(d["fwd5"].mean()),
        "vs_universe_5d": float(diff.mean()),
        "hit_up3": float(sel["up"].mean()),
        "hit_down3": float(sel["down"].mean()),
        "base_up3": float(d["up"].mean()),
        "base_down3": float(d["down"].mean()),
        "folds_beat": int((per_fold > 0).sum()),
        "folds": int(len(per_fold)),
    }
