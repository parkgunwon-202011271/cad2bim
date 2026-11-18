"""층 읽기. 한 파일에 여러 층이 있을 수 있어 평면 제목을 파일명보다 먼저 본다."""
from __future__ import annotations

import re
from typing import Any

_FLOOR = re.compile(r"(지하|지상|B)?\s*(\d{1,2})\s*(?:층|F)", re.I)
_TITLE_WORD = re.compile(r"평면|(?<![a-z])plan(?![a-z])", re.I)
_TITLE_MAX_CHARS = 30
_TITLE_MARGIN = 0.15


def normalize_name(text: str) -> str:
    """공백과 기호를 뺀다. 표와 평면의 표기가 조금 달라도 맞추기 위해서다."""
    return re.sub(r"[\s\-_.·,()\[\]]", "", text)


_ENGLISH_FLOOR = re.compile(r"(?<![a-z])(ground|first|second|third|fourth|fifth)\s+floor(?![a-z])", re.I)
_ENGLISH_ORDER = {"ground": 1, "first": 2, "second": 3, "third": 4, "fourth": 5, "fifth": 6}


def floor_key(text: str) -> str | None:
    """'지상 3층', '3F'는 F3, '지하1층', 'B1'은 B1. 이름 대조 없이 층으로 잇기 위해서다.

    영문 'first floor'는 'ground floor'가 1층인 표기를 따라 F2로 본다.
    """
    english = _ENGLISH_FLOOR.search(text)
    if english is not None:
        return f"F{_ENGLISH_ORDER[english.group(1).lower()]}"
    match = _FLOOR.search(text)
    if match is None:
        return None
    return ("B" if (match.group(1) or "").upper() in ("지하", "B") else "F") + str(int(match.group(2)))


_LEVEL_NOTE = re.compile(r"(지하|지상|B)?\s*(\d{1,2})\s*층\s*바닥\s*(?:레벨)?\s*(?:EL|FL)\s*\+?\s*(-?\d+(?:\.\d+)?)", re.I)


def read_level_notes(texts: list[Any]) -> dict[str, float]:
    """'3층 바닥 EL+26.50' 같은 주기에서 층 바닥 높이(mm)를 읽는다. 없으면 빈 값이다."""
    levels: dict[str, float] = {}
    for p in texts:
        if p.kind != "text":
            continue
        for match in _LEVEL_NOTE.finditer(p.geometry["text"]):
            key = ("B" if (match.group(1) or "").upper() in ("지하", "B") else "F") + str(int(match.group(2)))
            levels.setdefault(key, float(match.group(3)) * 1000.0)
    return levels


def plan_title_text(texts: list[Any], region: Any) -> str | None:
    """평면 제목 글자. region은 원본 좌표의 평면 범위이고, 없으면 파일 전체를 본다.

    제목은 평면 바로 바깥에 놓이기도 해서 범위를 조금 넓혀 찾고, 후보가 여럿이면 가장 큰 글자를 쓴다.
    """
    # '도면 번호 / 도면 이름 / 축척'처럼 나눠 쓴 제목은 조각별로 본다.
    titles = []
    for p in texts:
        if p.kind != "text":
            continue
        for token in p.geometry["text"].split("/"):
            if _TITLE_WORD.search(token) and len(normalize_name(token)) <= _TITLE_MAX_CHARS and floor_key(token):
                titles.append((p, token))
                break
    if region:
        margin = max(region[2] - region[0], region[3] - region[1]) * _TITLE_MARGIN
        titles = [
            (p, token) for p, token in titles
            if region[0] - margin <= p.geometry["position"][0] <= region[2] + margin
            and region[1] - margin <= p.geometry["position"][1] <= region[3] + margin
        ]
    if not titles:
        return None
    return max(titles, key=lambda item: item[0].geometry["height"])[1]


def plan_title_floor(texts: list[Any], region: Any) -> str | None:
    title = plan_title_text(texts, region)
    return floor_key(title) if title else None
