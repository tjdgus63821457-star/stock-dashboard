"""데이터 품질 검사와 산출물 일치 검사.

1) check_prices : 종목별 일봉에 이상이 없는지 (값 오류, 이력 부족, 갱신 지연, 미조정 분할 의심 등)
2) check_outputs: 지표·예측·상태 파일이 서로 맞는지, 숫자가 말이 되는 범위인지
오류(error)가 하나라도 있으면 종료 코드 1로 끝나서, 자동 실행이 대시보드를 게시하지 않고 멈춥니다.
"""
import argparse
import json
import math
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.fetch_prices import KST
from src.storage import PRICES_DIR, ROOT, load_prices
from src.universe import load_universe

QUALITY_PATH = ROOT / "data" / "quality.json"
MIN_BARS = 300            # 252일 지표와 모델 학습에 필요한 최소 이력
STALE_DAYS = 6            # 전체 최신 봉 날짜보다 이만큼(일) 뒤처지면 갱신 지연
JUMP = 0.30               # 하루 30% 초과 변동은 분할·병합이 조정되지 않았을 가능성
OHLC_TOL = 0.01           # 고가·저가가 시가·종가와 어긋나도 허용하는 비율(수정주가 반올림 오차)
MIN_PRED_RATIO = 0.80     # 예측 파일에 종목이 이 비율 이상 있어야 함


def check_prices(stocks=None, prices_dir=PRICES_DIR):
    """종목별 이상 항목 목록을 돌려줍니다. 반환: (issues{code:[문구]}, 요약 dict)."""
    stocks = stocks if stocks is not None else load_universe()
    data = {}
    for s in stocks:
        df = load_prices(s.code, base=prices_dir)
        if df is not None and len(df):
            data[s.code] = df.sort_values("date").reset_index(drop=True)
    latest = max((d["date"].iloc[-1] for d in data.values()), default=None)
    # 시장 달력: 최근 60일 동안 대다수 종목에 있는 날짜
    counts = {}
    for d in data.values():
        for x in d["date"].tail(60):
            counts[x] = counts.get(x, 0) + 1
    cal = sorted(x for x, c in counts.items() if c >= 0.5 * len(data))
    issues = {}
    for s in stocks:
        out = []
        d = data.get(s.code)
        if d is None:
            issues[s.code] = ["가격 파일 없음"]
            continue
        if len(d) < MIN_BARS:
            out.append(f"이력 부족({len(d)}일)")
        if latest and (date.fromisoformat(latest) - date.fromisoformat(d["date"].iloc[-1])).days > STALE_DAYS:
            out.append(f"갱신 지연(마지막 {d['date'].iloc[-1]})")
        px = d[["open", "high", "low", "close"]]
        if (px <= 0).any().any() or px.isna().any().any():
            out.append("0 이하 또는 빈 가격")
        hi_bad = (d["high"] < d[["open", "close"]].max(axis=1) * (1 - OHLC_TOL)).sum()
        lo_bad = (d["low"] > d[["open", "close"]].min(axis=1) * (1 + OHLC_TOL)).sum()
        if hi_bad + lo_bad:
            out.append(f"고가·저가 불일치 {int(hi_bad + lo_bad)}일")
        if d["date"].duplicated().any():
            out.append("중복 날짜")
        jumps = d["close"].pct_change().abs() > JUMP
        recent = d.loc[jumps & (d["date"] >= (cal[0] if cal else "0000")), "date"]
        if len(recent):
            out.append("최근 60일 내 하루 30% 초과 변동: " + ", ".join(recent.tolist()[:3]))
        if (d["volume"].tail(20) == 0).sum() > 5:
            out.append("최근 20일 중 거래량 0이 5일 초과(거래정지 의심)")
        if cal:
            have = set(d["date"])
            missing = [x for x in cal if x >= d["date"].iloc[0] and x not in have]
            if len(missing) > 3:
                out.append(f"최근 60일 중 빠진 거래일 {len(missing)}일")
        if out:
            issues[s.code] = out
    summary = {
        "checked": len(stocks),
        "with_issues": len(issues),
        "latest_bar_date": latest,
        "calendar_days_60": len(cal),
    }
    return issues, summary


def _load_strict(path):
    """NaN, Infinity 가 들어 있으면 오류로 보는 JSON 읽기."""
    def bad(token):
        raise ValueError(f"JSON에 {token} 값이 있습니다")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


def check_outputs(root=ROOT, universe=None):
    """산출 파일 사이의 일치와 값의 범위를 검사합니다. 반환: (errors, warnings)."""
    root = Path(root)
    errors, warnings = [], []
    try:
        status = _load_strict(root / "data" / "status.json")
        latest = _load_strict(root / "data" / "latest.json")
        pred = _load_strict(root / "data" / "predictions.json")
    except Exception as e:  # 파일 없음, JSON 오류, NaN
        return [f"산출 파일을 읽지 못했습니다: {e}"], warnings
    universe = universe if universe is not None else load_universe()
    codes = {s.code for s in universe}
    lat = {r["code"]: r for r in latest["stocks"]}
    stocks = pred["stocks"]

    if pred["asof_date"] != status["latest_bar_date"]:
        errors.append(f"예측 기준일({pred['asof_date']})과 최신 봉 날짜({status['latest_bar_date']})가 다릅니다")
    if len(stocks) < MIN_PRED_RATIO * len(codes):
        errors.append(f"예측 종목이 너무 적습니다({len(stocks)}/{len(codes)})")
    seen = set()
    for r in stocks:
        c = r["code"]
        if c in seen:
            errors.append(f"{c} 예측이 중복됩니다")
        seen.add(c)
        if c not in codes:
            errors.append(f"{c}는 종목 목록에 없는 코드입니다")
            continue
        if c not in lat:
            errors.append(f"{c}의 지표가 latest.json에 없습니다")
            continue
        close = lat[c]["close"]
        for k in ("p_up", "p_down", "p_stop"):
            if k in r and not (0.0 <= r[k] <= 1.0):
                errors.append(f"{c} {k}={r[k]} 가 0~1 범위를 벗어났습니다")
        if not (0.0 < r["score_pct"] <= 1.0):
            errors.append(f"{c} score_pct={r['score_pct']} 범위 오류")
        if not (r["range_lo"] < close < r["range_hi"]):
            errors.append(f"{c} 5일 범위({r['range_lo']}~{r['range_hi']})가 현재가({close})를 포함하지 않습니다")
        if not (0 < r["stop"] < close):
            errors.append(f"{c} 손절선({r['stop']})이 현재가({close}) 아래가 아닙니다")
        if not math.isfinite(r["score"]):
            errors.append(f"{c} 점수가 유한한 수가 아닙니다")
    rk = pred["validation"]["rank"]
    cov = rk.get("range90_coverage")
    if cov is None or not (0.5 <= cov <= 1.0):
        errors.append(f"90% 범위 적중률({cov})이 이상합니다")
    elif abs(cov - 0.90) > 0.05:
        warnings.append(f"90% 범위 적중률이 {cov:.1%}로 목표(90%)에서 5%p 넘게 벗어났습니다")
    if status["ok"] / max(1, status["total"]) < 0.9:
        warnings.append(f"가격 수집 성공 비율이 {status['ok']}/{status['total']}로 낮습니다")
    return errors, warnings


def build(now=None, out_path=QUALITY_PATH):
    now = now or datetime.now(KST)
    issues, summary = check_prices()
    errors, warnings = check_outputs()
    payload = {
        "checked_at_kst": now.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary, "errors": errors, "warnings": warnings, "issues": issues,
    }
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return payload


def main(argv=None):
    argparse.ArgumentParser(description="데이터 품질과 산출물 일치 검사").parse_args(argv)
    p = build()
    s = p["summary"]
    print(f"품질 검사: {s['checked']}종목 중 이상 {s['with_issues']}종목, 오류 {len(p['errors'])}건, 경고 {len(p['warnings'])}건")
    for code, msgs in list(p["issues"].items())[:20]:
        print(f"  - {code}: {'; '.join(msgs)}")
    for w in p["warnings"]:
        print("경고:", w)
    for e in p["errors"]:
        print("오류:", e)
    return 1 if p["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
