"""전진 검증: 실제로 예측을 먼저 기록해 두고, 5거래일 뒤 결과와 맞춰 봅니다.

과거 데이터로 한 검증은 설계 과정에서 같은 기간을 여러 번 봤기 때문에 낙관적일 수 있습니다.
여기서는 '확정된 일봉'으로 만든 예측만 추가 전용 파일(data/forward_log.csv)에 쌓고,
5거래일이 지난 뒤 채점하므로 사후에 고칠 수 없는 가장 정직한 검증이 됩니다.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.fetch_prices import KST
from src.storage import PRICES_DIR, ROOT, load_prices

LOG_PATH = ROOT / "data" / "forward_log.csv"
SUMMARY_PATH = ROOT / "data" / "forward_summary.json"
COLS = ["asof_date", "code", "close", "score_pct", "p_up", "p_down", "p_stop",
        "range_lo", "range_hi", "stop", "model_version"]
HORIZON = 5
MIN_DAYS_FOR_VERDICT = 20


def append_log(pred, latest, today, log_path=LOG_PATH):
    """확정된 날짜(오늘보다 이전)의 예측만 추가합니다. 같은 (날짜, 종목)은 다시 쓰지 않습니다."""
    if pred["asof_date"] >= today:
        return 0
    close = {s["code"]: s["close"] for s in latest["stocks"]}
    old = pd.read_csv(log_path, dtype={"asof_date": str, "code": str}) if Path(log_path).exists() else pd.DataFrame(columns=COLS)
    seen = set(zip(old["asof_date"], old["code"]))
    rows = []
    for r in pred["stocks"]:
        if (pred["asof_date"], r["code"]) in seen or r["code"] not in close:
            continue
        rows.append({"asof_date": pred["asof_date"], "code": r["code"], "close": close[r["code"]],
                     "score_pct": r["score_pct"], "p_up": r["p_up"], "p_down": r["p_down"],
                     "p_stop": r["p_stop"], "range_lo": r["range_lo"], "range_hi": r["range_hi"],
                     "stop": r["stop"], "model_version": pred.get("model_version", "")})
    if rows:
        out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)[COLS]
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(log_path, index=False)
    return len(rows)


def score(log, prices_dir=PRICES_DIR, top_n=10):
    """기록된 예측 중 5거래일이 지난 것만 채점합니다."""
    cache, recs = {}, []
    for r in log.itertuples():
        if r.code not in cache:
            cache[r.code] = load_prices(r.code, prices_dir)
        df = cache[r.code]
        if df is None:
            continue
        idx = df.index[df["date"] == r.asof_date]
        if len(idx) == 0 or idx[0] + HORIZON > len(df) - 1:
            continue
        i = idx[0]
        nxt = df.iloc[i + 1:i + 1 + HORIZON]
        c5 = float(nxt["close"].iloc[-1])
        recs.append({"asof_date": r.asof_date, "code": r.code, "fwd5": c5 / r.close - 1,
                     "stop_hit": float(nxt["low"].min() <= r.stop),
                     "in_range": float(r.range_lo <= c5 <= r.range_hi),
                     "up": float(float(nxt["close"].max()) / r.close - 1 >= 0.03),
                     "score_pct": r.score_pct, "p_up": r.p_up, "p_stop": r.p_stop})
    d = pd.DataFrame(recs)
    if d.empty:
        return {"scored_days": 0, "min_days_for_verdict": MIN_DAYS_FOR_VERDICT, "logged_days": int(log["asof_date"].nunique()) if len(log) else 0}
    d["um"] = d.groupby("asof_date")["fwd5"].transform("mean")
    top = d.sort_values(["asof_date", "score_pct"], ascending=[True, False]).groupby("asof_date").head(top_n)
    per_day = (top["fwd5"] - top["um"]).groupby(top["asof_date"]).mean()
    ic = d.groupby("asof_date").apply(lambda x: x["score_pct"].corr(x["fwd5"], method="spearman"), include_groups=False)
    days = int(d["asof_date"].nunique())
    return {
        "scored_days": days, "logged_days": int(log["asof_date"].nunique()),
        "first": str(d["asof_date"].min()), "last": str(d["asof_date"].max()),
        "min_days_for_verdict": MIN_DAYS_FOR_VERDICT,
        "top_vs_universe_5d": float(per_day.mean()),
        "ic_mean": float(ic.mean()),
        "range_hit": float(d["in_range"].mean()),
        "stop_pred_mean": float(d["p_stop"].mean()), "stop_actual": float(d["stop_hit"].mean()),
        "up3_pred_mean": float(d["p_up"].mean()), "up3_actual": float(d["up"].mean()),
        "n": int(len(d)),
    }


def main(argv=None):
    today = datetime.now(KST).date().isoformat()
    pred = json.loads((ROOT / "data" / "predictions.json").read_text(encoding="utf-8"))
    latest = json.loads((ROOT / "data" / "latest.json").read_text(encoding="utf-8"))
    n = append_log(pred, latest, today)
    log = pd.read_csv(LOG_PATH, dtype={"asof_date": str, "code": str}) if LOG_PATH.exists() else pd.DataFrame(columns=COLS)
    summ = score(log) if len(log) else {"scored_days": 0, "logged_days": 0, "min_days_for_verdict": MIN_DAYS_FOR_VERDICT}
    summ["updated_kst"] = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    SUMMARY_PATH.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"전진 기록: 새로 {n}건 추가, 기록 {summ['logged_days']}일, 채점 완료 {summ['scored_days']}일")
    return 0


if __name__ == "__main__":
    sys.exit(main())
