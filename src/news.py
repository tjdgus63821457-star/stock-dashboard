"""뉴스 흐름(센티먼트) 표시 등급을 정하는 모듈.

뉴스는 과거 검증 자료가 없어 점수에 넣지 않습니다. 대신 아래 고정 규칙으로 등급을 매겨
종목 옆에 '경고/주의/양호/기사 적음' 표시로만 보여 줍니다.

  경고(warn)   : 심리 z점수(1개월 또는 1분기)가 -2 이하이거나, 경고급 사건이 있음
  주의(caution): 심리 모멘텀이 -0.10 이하이거나 현재 심리가 -0.20 이하이거나, 주의급 사건이 있음
  기사 적음(none): 최근 90일 기사가 5건 미만 (수치를 믿을 수 없음)
  양호(ok)     : 위에 해당하지 않음
"""
import json
from pathlib import Path

from src.storage import ROOT

NEWS_PATH = ROOT / "data" / "news.json"
MIN_DOCS = 5
Z_WARN = -2.0
MOMENTUM_CAUTION = -0.10
CURRENT_CAUTION = -0.20
ORDER = {"ok": 0, "none": 0, "caution": 1, "warn": 2}


def level(entry):
    """뉴스 항목 하나의 등급 문자열."""
    if entry.get("doc_count", 0) < MIN_DOCS:
        return "none"
    s = entry.get("sentiment") or {}
    lv = "ok"
    zs = [z for z in (s.get("z1mo"), s.get("z1qt")) if z is not None]
    if zs and min(zs) <= Z_WARN:
        lv = "warn"
    elif (s.get("momentum", 0) <= MOMENTUM_CAUTION + 1e-9) or (s.get("current", 0) <= CURRENT_CAUTION + 1e-9):
        lv = "caution"
    for ev in entry.get("events", []):
        if ORDER.get(ev.get("sev"), 0) > ORDER[lv]:
            lv = ev["sev"]
    return lv


def load_news(path=NEWS_PATH):
    """data/news.json 을 읽어 종목별 등급을 붙여 돌려줍니다. 파일이 없으면 빈 값."""
    path = Path(path)
    if not path.exists():
        return {"updated_kst": None, "source": None, "stocks": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    for entry in raw.get("stocks", {}).values():
        entry["level"] = level(entry)
    return raw
