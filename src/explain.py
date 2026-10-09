"""점수의 근거를 설명하는 모듈.

1) 종목별 근거: 점수는 선형 모델이라 '특징별 기여'의 합으로 정확히 나뉩니다.
2) 특징별 증거: 각 특징 하나만으로 본 과거 상관(IC)과, 구간(약 60거래일)별 방향 일관성.
"""
import numpy as np

from src import model as M

# 키: (이름, 값 표시 형식, 한 줄 의미)
INFO = {
    "gap20": ("20일선 대비 위치", "pct", "현재가가 20일 평균선보다 얼마나 위(+)·아래(−)에 있는지"),
    "gap60": ("60일선 대비 위치", "pct", "현재가가 60일 평균선보다 얼마나 위·아래에 있는지"),
    "trend_up": ("중기 추세", "flag", "20일선이 60일선 위에 있으면 상승 추세"),
    "rsi14": ("RSI(14일)", "num", "최근 14일의 상승·하락 힘. 높을수록 과열, 낮을수록 침체"),
    "atr_pct": ("하루 변동폭", "pct", "최근 14일 평균 하루 움직임(고가-저가 기준)이 종가의 몇 %인지"),
    "vol_ratio": ("거래량 비율", "x", "오늘 거래량 ÷ 직전 20일 평균 거래량"),
    "ret1": ("1일 수익률", "pct", "어제 종가 대비 오늘"),
    "ret5": ("5일 수익률", "pct", "5거래일 전 종가 대비"),
    "ret20": ("20일 수익률", "pct", "20거래일 전 종가 대비"),
    "from_high20": ("20일 고점 대비", "pct", "최근 20일 최고가에서 얼마나 내려와 있는지"),
    "from_high60": ("60일 고점 대비", "pct", "최근 60일 최고가에서 얼마나 내려와 있는지"),
    "pullback5": ("5일 고점 대비 되돌림", "pct", "최근 5일 종가 최고점에서 얼마나 내려왔는지"),
    "mom12_1": ("12-1개월 모멘텀", "pct", "1개월 전 가격 대비 12개월 전부터의 수익률(최근 1개월 제외). 모멘텀 연구의 표준 지표"),
    "ret60": ("60일 수익률", "pct", "60거래일 전 종가 대비"),
    "ret120": ("120일 수익률", "pct", "120거래일 전 종가 대비"),
    "high52": ("52주 고점 대비", "pct", "최근 1년 최고가에서 얼마나 내려와 있는지(0에 가까울수록 신고가 근처)"),
    "above_low52": ("52주 저점 위 거리", "pct", "최근 1년 최저가보다 얼마나 올라와 있는지"),
    "break55": ("55일 신고가 돌파", "pct", "직전 55일 최고가 대비. 0 이상이면 돌파(터틀 규칙)"),
    "gap200": ("200일선 대비 위치", "pct", "현재가가 200일 평균선보다 얼마나 위·아래에 있는지"),
    "slope200": ("200일선 기울기", "pct", "200일선이 한 달 전보다 얼마나 올랐는지(상승 추세 여부)"),
    "vol60": ("60일 변동성", "pct", "최근 60일 일간 수익률의 표준편차. 낮을수록 안정적(저변동 효과)"),
    "minervini": ("미너비니 추세 조건", "cnt", "장기 상승 추세 조건 7개 중 충족한 개수(이동평균 정렬, 52주 범위 등)"),
    "macd_hist": ("MACD 히스토그램", "pct", "단기·장기 추세 평균의 차이가 최근 평균에서 벌어진 정도(종가 대비 %)"),
}
BLOCK_DAYS = 60
TOP_POS, TOP_NEG = 3, 2


def stock_reasons(models, cur):
    """cur(그날 종목별 지표 표)에서 종목마다 점수에 가장 크게 더한/뺀 특징을 고릅니다.

    반환: {행 인덱스: {"score": 점수, "why": [[키, 기여, 실제 값, 오늘 종목 중 순위(0~1)], ...]}}
    """
    contrib, intercept = M.contributions(models["rank"], cur)
    keys = [f[2:] for f in M.RANK_FEATURES]
    out = {}
    for n, idx in enumerate(cur.index):
        c = contrib[n]
        order = np.argsort(c)
        pos = [i for i in order[::-1][:TOP_POS] if c[i] > 0]
        neg = [i for i in order[:TOP_NEG] if c[i] < 0]
        why = []
        for i in pos + neg:
            k = keys[i]
            why.append([k, round(float(c[i]), 4), _num(cur.at[idx, k]), _num(cur.at[idx, "r_" + k])])
        out[idx] = {"score": round(intercept + float(c.sum()), 4), "why": why}
    return out, intercept


def _num(x):
    return None if x is None or x != x else round(float(x), 4)


def factor_table(panel, models):
    """특징 하나씩, 그 특징의 '종목 간 순위'와 '이후 5일 시장 대비 초과수익 순위'의 일별 상관(IC)."""
    d = panel[~panel["bad"]].dropna(subset=["ex"] + M.RANK_FEATURES)
    date = d["date"]
    exr = d.groupby("date")["ex"].rank(pct=True)
    exm = exr - exr.groupby(date).transform("mean")
    ex_ss = (exm ** 2).groupby(date).sum()
    ridge = models["rank"].named_steps["ridge"]
    coef = dict(zip(M.RANK_FEATURES, ridge.coef_))
    dates = np.array(sorted(d["date"].unique()))
    blocks = np.array_split(dates, max(1, len(dates) // BLOCK_DAYS))
    rows = []
    for k in [f[2:] for f in M.RANK_FEATURES]:
        a = d["r_" + k]
        am = a - a.groupby(date).transform("mean")
        num = (am * exm).groupby(date).sum()
        den = np.sqrt((am ** 2).groupby(date).sum() * ex_ss)
        ic = (num / den).replace([np.inf, -np.inf], np.nan).dropna()
        by_block = [float(ic.reindex(b).mean()) for b in blocks]
        name, fmt, meaning = INFO[k]
        rows.append({
            "key": k, "label": name, "fmt": fmt, "meaning": meaning,
            "ic_mean": round(float(ic.mean()), 4),
            "pos_blocks": int(sum(v > 0 for v in by_block)), "blocks": len(by_block),
            "coef": round(float(coef["r_" + k]), 4),
        })
    rows.sort(key=lambda r: -abs(r["coef"]))
    return rows, {"start": str(dates[0]), "end": str(dates[-1]), "days": int(len(dates))}
