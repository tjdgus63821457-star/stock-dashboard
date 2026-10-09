"""5거래일 안에 +3% 오를 확률(상승)과 -3% 내릴 확률(하락)을 추정하고,
워크포워드 방식으로 실제 적중률을 측정하는 모듈.

워크포워드: 과거 데이터로만 학습하고, 그 이후 구간(학습에 쓰지 않은 구간)에서만 평가합니다.
정답 라벨이 앞으로 5일 값을 쓰므로, 학습 구간 끝과 평가 구간 사이에 5일을 비웁니다(purge).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.indicators import INDICATOR_COLUMNS, compute
from src.storage import PRICES_DIR, load_prices
from src.universe import load_universe

HORIZON = 5            # 앞으로 며칠 안에
TARGET = 0.03          # +3% (상승 라벨), -3% (하락 라벨)
MIN_TRAIN_DAYS = 500   # 첫 평가 전 최소 학습 일수
TEST_DAYS = 60         # 평가 구간 길이(약 3개월)
MARKET_FEATURES = ["mkt_ret5", "mkt_ret20", "mkt_breadth20"]
# 주가 '수준'(원 단위)을 그대로 담은 지표는 종목끼리 비교할 수 없고 종목 식별자 역할만 하므로 모델에서 뺍니다.
PRICE_LEVEL = {"ma5", "ma20", "ma60", "atr14"}
MODEL_INDICATORS = [c for c in INDICATOR_COLUMNS if c not in PRICE_LEVEL]
FEATURES = MODEL_INDICATORS + MARKET_FEATURES
STOP_ATR = 2.0         # 손절 참고선 = 종가 - 2 x ATR
MAX_JUMP = 0.31        # 하루 31% 초과 변동(분할·신규상장 의심) 전후 구간은 학습에서 제외


def build_panel(stocks=None, prices_dir=PRICES_DIR):
    """전 종목 지표 + 시장 전체 지표 + 정답 라벨을 한 표로 합칩니다."""
    stocks = stocks if stocks is not None else load_universe()
    frames = []
    for s in stocks:
        df = load_prices(s.code, base=prices_dir)
        if df is None or len(df) < 70:
            continue
        ind = compute(df)
        srt = df.sort_values("date").reset_index(drop=True)
        close, low = srt["close"], srt["low"]
        # 앞으로 1~5일 종가의 최고/최저 (오늘 값 제외)
        fwd = pd.concat([close.shift(-k) for k in range(1, HORIZON + 1)], axis=1)
        ind["fwd_max"] = fwd.max(axis=1, skipna=False) / close - 1
        ind["fwd_min"] = fwd.min(axis=1, skipna=False) / close - 1
        ind["fwd5"] = close.shift(-HORIZON) / close - 1          # 5일 뒤 종가 수익률
        # 앞으로 1~5일 중 장중 최저가 (손절선 터치 여부 판정용)
        lows = pd.concat([low.shift(-k) for k in range(1, HORIZON + 1)], axis=1)
        ind["fwd_minlow"] = lows.min(axis=1, skipna=False) / close - 1
        jump = close.pct_change().abs() > MAX_JUMP
        near = jump.rolling(2 * HORIZON + 1, center=True, min_periods=1).max().astype(bool)
        # 지표 창(60일) 안에 점프가 있으면 지표도 오염 -> 점프 후 60일은 제외
        polluted = jump.rolling(61, min_periods=1).max().astype(bool) | near
        ind["bad"] = polluted.values
        ind["code"] = s.code
        frames.append(ind)
    panel = pd.concat(frames, ignore_index=True)

    # 시장 전체(전 종목 중앙값) 상태: 그날 시장 분위기
    g = panel.groupby("date")
    mkt = pd.DataFrame({
        "mkt_ret5": g["ret5"].median(),
        "mkt_ret20": g["ret20"].median(),
        "mkt_breadth20": g["gap20"].apply(lambda x: (x > 0).mean()),
    })
    panel = panel.join(mkt, on="date")
    panel["up"] = (panel["fwd_max"] >= TARGET).astype(float).where(panel["fwd_max"].notna())
    panel["down"] = (panel["fwd_min"] <= -TARGET).astype(float).where(panel["fwd_min"].notna())
    ok = panel["fwd_minlow"].notna() & panel["atr_pct"].notna()
    panel["stop"] = (panel["fwd_minlow"] <= -STOP_ATR * panel["atr_pct"]).astype(float).where(ok)
    return panel.sort_values(["date", "code"]).reset_index(drop=True)


def make_model(kind):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500))
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=400,
            l2_regularization=1.0, random_state=0,
        )
    raise ValueError(kind)


def _clean(df, label):
    d = df[~df["bad"]].dropna(subset=FEATURES + [label])
    return d


def walk_forward(panel, label="up", kind="logit"):
    """평가 구간별로 '그 이전 데이터만' 학습해 예측한 결과를 모아 돌려줍니다."""
    dates = np.array(sorted(panel["date"].unique()))
    out = []
    start = MIN_TRAIN_DAYS
    fold = 0
    while start < len(dates) - HORIZON:
        test_dates = dates[start:start + TEST_DAYS]
        train_end = dates[start - HORIZON - 1]           # purge: 5일 비움
        train = _clean(panel[panel["date"] <= train_end], label)
        test = _clean(panel[panel["date"].isin(test_dates)], label)
        if len(test) and train[label].nunique() == 2:
            m = make_model(kind).fit(train[FEATURES], train[label])
            t = test[["date", "code", label, "fwd_max", "fwd_min", "fwd5", "atr_pct"]].copy()
            t["p"] = m.predict_proba(test[FEATURES])[:, 1]
            t["fold"] = fold
            out.append(t)
        fold += 1
        start += TEST_DAYS
    return pd.concat(out, ignore_index=True)


def summarize(oos, label="up", top_n=10):
    """평가 결과 요약: AUC, 기본 확률, 상위 종목 적중률, 보정표."""
    base = float(oos[label].mean())
    auc = float(roc_auc_score(oos[label], oos["p"]))
    # 매일 확률 상위 top_n 종목의 실제 적중률
    ranked = oos.sort_values(["date", "p"], ascending=[True, False])
    top = ranked.groupby("date").head(top_n)
    folds = []
    for f, d in oos.groupby("fold"):
        tf = ranked[ranked["fold"] == f].groupby("date").head(top_n)
        folds.append({"fold": int(f), "start": d["date"].min(), "end": d["date"].max(),
                      "base": float(d[label].mean()), "top": float(tf[label].mean()),
                      "auc": float(roc_auc_score(d[label], d["p"])) if d[label].nunique() == 2 else None})
    bins = [0, .1, .2, .3, .4, .5, .6, 1.0001]
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = oos[(oos["p"] >= lo) & (oos["p"] < hi)]
        cal.append({"lo": lo, "hi": min(hi, 1.0), "n": int(len(s)),
                    "pred": float(s["p"].mean()) if len(s) else None,
                    "actual": float(s[label].mean()) if len(s) else None})
    return {"label": label, "n": int(len(oos)), "base": base, "auc": auc,
            "top_n": top_n, "top_hit": float(top[label].mean()), "folds": folds, "calibration": cal}


# ---------------------------------------------------------------------------
# 순위 점수 모델: "같은 날 다른 종목보다, 변동성 대비 얼마나 더 오를 것인가"
# ---------------------------------------------------------------------------
from sklearn.linear_model import Ridge  # noqa: E402

RANK_FEATURES = ["r_" + c for c in MODEL_INDICATORS]
RIDGE_ALPHA = 1000.0
KEEP_COLS = ["close", "gap20", "gap60", "rsi14", "vol_ratio", "ret1", "ret5", "ret20",
             "pullback5", "from_high20", "atr_pct", "ma20"]


def add_rank_columns(panel):
    """날짜별 종목 간 순위(0~1) 특징과, 시장 대비 초과수익 라벨(ex, exn)을 붙입니다."""
    g = panel.groupby("date")
    ranks = g[INDICATOR_COLUMNS].rank(pct=True)
    for c in INDICATOR_COLUMNS:
        panel["r_" + c] = ranks[c]
    panel["ex"] = panel["fwd5"] - g["fwd5"].transform("median")      # 시장(중앙값) 대비 초과수익
    panel["exn"] = (panel["ex"] / panel["atr_pct"]).clip(-5, 5)      # 변동성으로 나눈 값
    return panel


def make_rank_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def _rank_clean(df):
    return df[~df["bad"]].dropna(subset=RANK_FEATURES + ["exn", "ex", "fwd5"])


def walk_forward_rank(panel):
    """순위 모델의 워크포워드 결과. 날짜별 점수 백분위(s_pct)와 규칙 검증용 지표를 함께 돌려줍니다."""
    dates = np.array(sorted(panel["date"].unique()))
    out, start, fold = [], MIN_TRAIN_DAYS, 0
    while start < len(dates) - HORIZON:
        test_dates = dates[start:start + TEST_DAYS]
        train_end = dates[start - HORIZON - 1]
        train = _rank_clean(panel[panel["date"] <= train_end])
        test = _rank_clean(panel[panel["date"].isin(test_dates)])
        if len(test):
            m = make_rank_model().fit(train[RANK_FEATURES], train["exn"])
            t = test[["date", "code", "fwd5", "ex", "fwd_max", "fwd_min"] + KEEP_COLS].copy()
            t["s"] = m.predict(test[RANK_FEATURES])
            # 훈련 구간에서 본 '5일 수익률 / 일일변동폭' 분포 -> 평가 구간의 80% 범위 적중 확인용
            ratio = (train["fwd5"] / train["atr_pct"])
            t["q10"], t["q90"] = float(ratio.quantile(0.10)), float(ratio.quantile(0.90))
            t["q05"], t["q95"] = float(ratio.quantile(0.05)), float(ratio.quantile(0.95))
            t["fold"] = fold
            out.append(t)
        fold += 1
        start += TEST_DAYS
    oos = pd.concat(out, ignore_index=True)
    oos["s_pct"] = oos.groupby("date")["s"].rank(pct=True)
    oos["up"] = (oos["fwd_max"] >= TARGET).astype(float)
    oos["down"] = (oos["fwd_min"] <= -TARGET).astype(float)
    return oos


def _tstat(x):
    """구간별 평균이 0과 다른지 보는 t값과 양측 p값. 구간이 3개 미만이면 None."""
    import numpy as _np
    from scipy import stats as _st
    x = _np.asarray(x, dtype=float)
    x = x[~_np.isnan(x)]
    if len(x) < 3 or x.std(ddof=1) == 0:
        return None
    t = float(x.mean() / (x.std(ddof=1) / _np.sqrt(len(x))))
    return {"t": round(t, 2), "p": round(float(2 * _st.t.sf(abs(t), len(x) - 1)), 4), "n": int(len(x))}


def rank_summary(oos, top_n=10, cost=0.0025):
    """순위 모델 요약. 모든 비교는 '같은 날 전체 종목 평균'을 기준으로 합니다."""
    oos = oos.copy()
    oos["um"] = oos.groupby("date")["fwd5"].transform("mean")        # 그날 전체 평균 5일 수익률
    ic = oos.groupby("date").apply(lambda x: x["s"].corr(x["ex"], method="spearman"), include_groups=False)
    top = oos.sort_values(["date", "s"], ascending=[True, False]).groupby("date").head(top_n)
    diff = top.groupby("date")["fwd5"].mean() - top.groupby("date")["um"].first()
    fold_diff = top.assign(d=top["fwd5"] - top["um"]).groupby("fold")["d"].mean()
    fold_ic = oos.groupby("fold").apply(lambda x: x["s"].corr(x["ex"], method="spearman"), include_groups=False)
    ratio = oos["fwd5"] / oos["atr_pct"]
    cover = ((ratio >= oos["q10"]) & (ratio <= oos["q90"])).mean()
    inside90 = (ratio >= oos["q05"]) & (ratio <= oos["q95"])
    cover90 = inside90.mean()
    by_fold90 = inside90.groupby(oos["fold"]).mean()
    # 유의성: 겹치는 5일 수익률 때문에 일 단위 t값은 부풀려지므로, 60일 구간(fold) 평균으로 t검정합니다.
    ic_t = _tstat(fold_ic.values)
    diff_t = _tstat(fold_diff.values)
    return {
        "ic_t": ic_t, "top_diff_t": diff_t,
        "n": int(len(oos)), "days": int(oos["date"].nunique()),
        "start": oos["date"].min(), "end": oos["date"].max(),
        "ic_mean": float(ic.mean()), "ic_pos_days": float((ic > 0).mean()),
        "fold_ic": [round(float(v), 4) for v in fold_ic],
        "top_n": top_n,
        "top_mean_5d": float(top["fwd5"].mean()),
        "universe_mean_5d": float(oos["fwd5"].mean()),
        "top_vs_universe_5d": float(diff.mean()),
        "top_vs_universe_after_cost": float(diff.mean() - cost),
        "folds_top_beats_universe": int((fold_diff > 0).sum()),
        "folds": int(len(fold_diff)),
        "range80_coverage": float(cover),
        "range90_coverage": float(cover90),
        "range90_by_fold": [round(float(v), 4) for v in by_fold90],
    }


def fit_final(panel):
    """전체 라벨 데이터로 최종 모델을 학습합니다 (실서비스용)."""
    r = _rank_clean(panel)
    rank_model = make_rank_model().fit(r[RANK_FEATURES], r["exn"])
    up = _clean(panel, "up")
    down = _clean(panel, "down")
    up_model = make_model("logit").fit(up[FEATURES], up["up"])
    down_model = make_model("logit").fit(down[FEATURES], down["down"])
    ratio = (r["fwd5"] / r["atr_pct"])
    return {"rank": rank_model, "up": up_model, "down": down_model,
            "q10": float(ratio.quantile(0.10)), "q90": float(ratio.quantile(0.90)),
            "q05": float(ratio.quantile(0.05)), "q95": float(ratio.quantile(0.95)),
            "stop": make_model("logit").fit(_clean(panel, "stop")[FEATURES], _clean(panel, "stop")["stop"])}


def contributions(rank_model, X):
    """종목별 점수에 각 특징이 얼마나 기여했는지. 점수 = 절편 + 기여의 합 (선형 모델이라 정확히 분해됩니다)."""
    scaler = rank_model.named_steps["standardscaler"]
    ridge = rank_model.named_steps["ridge"]
    z = scaler.transform(X[RANK_FEATURES])
    return z * ridge.coef_, float(ridge.intercept_)


# ---- 확률 보정 (isotonic) -------------------------------------------------
def brier(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def calibration_check(oos, label, min_train_folds=3):
    """보정이 실제로 도움이 되는지 확인합니다.

    각 구간(fold)은 '그 이전 구간들의 표본 외 예측'으로만 보정기를 학습해 평가하므로
    미래 정보가 섞이지 않습니다. 개선이 없으면 보정을 쓰지 않습니다.
    """
    from sklearn.isotonic import IsotonicRegression
    d = oos.dropna(subset=["p", label])
    folds = sorted(d["fold"].unique())
    raw, cal, ys = [], [], []
    for f in folds[min_train_folds:]:
        tr, te = d[d["fold"] < f], d[d["fold"] == f]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(tr["p"], tr[label])
        raw.append(te["p"].values); cal.append(iso.predict(te["p"].values)); ys.append(te[label].values)
    if not ys:
        return {"use": False}
    raw, cal, ys = np.concatenate(raw), np.concatenate(cal), np.concatenate(ys)
    b0, b1 = brier(raw, ys), brier(cal, ys)
    return {"use": bool(b1 < b0), "brier_raw": round(b0, 5), "brier_cal": round(b1, 5),
            "folds_checked": len(folds) - min_train_folds}


def fit_calibrator(oos, label):
    from sklearn.isotonic import IsotonicRegression
    d = oos.dropna(subset=["p", label])
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(d["p"], d[label])
