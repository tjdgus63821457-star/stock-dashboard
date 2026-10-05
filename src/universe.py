"""종목 목록(data/universe.csv)을 읽는 모듈.

universe.csv 한 줄이 종목 하나입니다.
code        6자리 종목코드 (예: 005930)
name        종목명
market      KOSPI 또는 KOSDAQ
sector_key  섹터 영문 키 (tech, fin, ind ...)
sector      섹터 한글 이름
industry_en 데이터 제공처의 세부 업종 (영문)
symbol      가격 조회에 쓰는 심볼 (예: 005930.KS, 코스닥은 .KQ)
"""
import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_CSV = ROOT / "data" / "universe.csv"


@dataclass(frozen=True)
class Stock:
    code: str
    name: str
    market: str
    sector_key: str
    sector: str
    industry_en: str
    symbol: str


def load_universe(path=UNIVERSE_CSV):
    """universe.csv를 읽어 Stock 목록으로 돌려줍니다."""
    stocks = []
    seen = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = row["code"].strip()
            if code in seen:
                raise ValueError(f"종목코드가 중복되었습니다: {code}")
            seen.add(code)
            stocks.append(
                Stock(
                    code=code,
                    name=row["name"].strip(),
                    market=row["market"].strip(),
                    sector_key=row["sector_key"].strip(),
                    sector=row["sector"].strip(),
                    industry_en=row["industry_en"].strip(),
                    symbol=row["symbol"].strip(),
                )
            )
    return stocks
