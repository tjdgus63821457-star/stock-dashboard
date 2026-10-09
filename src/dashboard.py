"""대시보드 페이지(docs/index.html)를 만듭니다. 데이터는 페이지 안에 함께 넣어 한 파일로 열립니다."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from src.fetch_prices import KST
from src.news import load_news
from src.storage import PRICES_DIR, ROOT, load_prices
from src.universe import load_universe

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
DOCS_INDEX = ROOT / "docs" / "index.html"
SPARK_DAYS = 60

HEAD = (
    '<!doctype html>\n<html lang="ko"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
    '<meta name="color-scheme" content="light dark">'
    '<meta name="apple-mobile-web-app-capable" content="yes">'
    '<meta name="apple-mobile-web-app-title" content="시황판">'
    "<style>body{margin:0}</style></head><body>\n"
)
TAIL = "\n</body></html>\n"


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def collect(root=ROOT, prices_dir=PRICES_DIR):
    """페이지에 넣을 데이터를 한곳에 모읍니다."""
    root = Path(root)
    latest = {s["code"]: s for s in _load_json(root / "data" / "latest.json")["stocks"]}
    pred = _load_json(root / "data" / "predictions.json")
    status = _load_json(root / "data" / "status.json")
    uni = {s.code: s for s in load_universe()}
    news = load_news(root / "data" / "news.json")
    rows = []
    for p in pred["stocks"]:
        ind = latest.get(p["code"])
        s = uni.get(p["code"])
        if ind is None or s is None:
            continue
        df = load_prices(p["code"], base=prices_dir)
        spark = [round(float(x), 2) for x in df["close"].tail(SPARK_DAYS)] if df is not None else []
        rows.append({
            "code": s.code, "name": s.name, "market": s.market, "sector": s.sector,
            "close": ind["close"], "ret1": ind["ret1"], "ret5": ind["ret5"], "ret20": ind["ret20"],
            "gap20": ind["gap20"], "rsi14": ind["rsi14"], "vol_ratio": ind["vol_ratio"],
            "score_pct": p["score_pct"], "p_up": p["p_up"], "p_down": p["p_down"], "p_stop": p.get("p_stop"),
            "range_lo": p["range_lo"], "range_hi": p["range_hi"], "stop": p["stop"],
            "flags": p["flags"], "spark": spark,
            "news": news["stocks"].get(p["code"]), "score": p.get("score"), "why": p.get("why", []), "atr_pct": ind.get("atr_pct"),
        })
    def opt(name, default):
        f = root / "data" / name
        return _load_json(f) if f.exists() else default
    quality = opt("quality.json", {})
    fwd = opt("forward_summary.json", {})
    return {
        "asof_date": pred["asof_date"],
        "model_version": pred.get("model_version"),
        "quality": {"summary": quality.get("summary"), "errors": quality.get("errors", []),
                    "warnings": quality.get("warnings", []),
                    "checked_at_kst": quality.get("checked_at_kst")},
        "forward": fwd,
        "built_at_kst": pred["built_at_kst"],
        "today_kst": status["today_kst"],
        "latest_bar_date": status["latest_bar_date"],
        "failed": status.get("failed", []),
        "validation": pred["validation"],
        "explain": pred.get("explain", {"factors": []}),
        "news_meta": {"updated_kst": news.get("updated_kst"), "source": news.get("source"), "count": len(news["stocks"])},
        "stocks": rows,
    }


def render(data, fragment=False):
    html = TEMPLATE.read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace("__DATA__", blob)
    return html if fragment else HEAD + html + TAIL


def main(argv=None):
    ap = argparse.ArgumentParser(description="대시보드 HTML 생성")
    ap.add_argument("--out", default=str(DOCS_INDEX))
    ap.add_argument("--fragment", action="store_true", help="문서 틀 없이 본문만 (Artifact 게시용)")
    args = ap.parse_args(argv)
    data = collect()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data, fragment=args.fragment), encoding="utf-8")
    print(f"대시보드 생성: {out} ({out.stat().st_size / 1024:.0f}KB, {len(data['stocks'])}종목)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
