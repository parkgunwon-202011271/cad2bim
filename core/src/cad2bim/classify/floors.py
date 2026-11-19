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


_ROOF_TITLE = re.compile(r"(지붕|옥상)\s*층?\s*평면|roof\s*(floor\s*)?plan", re.I)


def roof_key(text: str) -> str | None:
    """'지붕층 평면도', 'ROOF PLAN'은 지붕(R). 층 번호가 붙은 제목은 floor_key가 먼저다."""
    return "R" if _ROOF_TITLE.search(text) else None


_PARTIAL_TITLE = re.compile(r"상부|중층|메자닌|mezzanine|다락", re.I)
PARTIAL = "M"  # 부분층 표시. 4층 상부 평면은 F4M


def partial_key(label: str, floor: str | None) -> str | None:
    """'지상4층 상부 평면도'는 F4M. 4층 바닥과 지붕 사이에 일부만 있는 층이다."""
    if floor and floor[0] in "BF" and floor[1:].isdigit() and _PARTIAL_TITLE.search(label):
        return floor + PARTIAL
    return floor


def plan_level(texts: list[Any], region: Any = None) -> float | None:
    """평면 안의 레벨 글자(FL:135.10 등) 중 가장 많이 나온 값(mm). 부분층의 바닥 높이로 쓴다."""
    counts: dict[float, int] = {}
    for p in texts:
        if p.kind != "text":
            continue
        if region:
            x, y = p.geometry["position"]
            if not (region[0] <= x <= region[2] and region[1] <= y <= region[3]):
                continue
        mm = _mark_mm(p.geometry["text"])
        if mm is not None:
            counts[round(mm)] = counts.get(round(mm), 0) + 1
    return float(max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]) if counts else None


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


# 단면도·입면도의 층 레벨 표시. 층 이름 글자와 'FH:109.75' 같은 레벨 글자가 나란히 붙어 있다.
_MARK_FLOOR = re.compile(r"^(?:(지하|B)\s*(\d{1,2})\s*(?:층|F)?|(?:지상\s*)?(\d{1,2})\s*(?:층|F)|(지붕층?|옥상층?|ROOF|R\.?F))$", re.I)
_MARK_LEVEL = re.compile(r"^(FH|FL)\s*[:：=]?\s*([+-]?\d+(?:[.,]\d+)?)$", re.I)
_MARK_REACH = 6.0  # 층 이름과 레벨 글자 사이 거리 한도. 글자 높이의 배수
ROOF = "R"


def _mark_floor(text: str) -> str | None:
    match = _MARK_FLOOR.match(re.sub(r"\s+", "", text))
    if match is None:
        return None
    if match.group(4):
        return ROOF
    if match.group(1):
        return f"B{int(match.group(2))}"
    return f"F{int(match.group(3))}"


def _mark_mm(text: str) -> float | None:
    match = _MARK_LEVEL.match(text.strip())
    if match is None:
        return None
    raw = match.group(2)
    if "," in raw:  # 6,280 처럼 천 단위 쉼표면 mm
        return float(raw.replace(",", ""))
    value = float(raw)
    return value * 1000.0 if "." in raw and abs(value) < 1000 else value  # 109.75는 m


def read_level_marks(texts: list[Any]) -> dict[str, list[float]]:
    """층 이름마다 가장 가까운 레벨 글자를 짝짓는다. 같은 층이 여러 번 나오면 값을 모두 모은다."""
    floors, levels = [], []
    for p in texts:
        if p.kind != "text":
            continue
        text = p.geometry["text"]
        key = _mark_floor(text)
        if key is not None:
            floors.append((key, p.geometry["position"], p.geometry["height"]))
            continue
        mm = _mark_mm(text)
        if mm is not None:
            levels.append((mm, p.geometry["position"], p.geometry["height"]))
    found: dict[str, list[float]] = {}
    for key, (x, y), h in floors:
        best = None
        for mm, (lx, ly), lh in levels:
            d = ((lx - x) ** 2 + (ly - y) ** 2) ** 0.5
            if d <= _MARK_REACH * max(h, lh) and (best is None or d < best[0]):
                best = (d, mm)
        if best is not None:
            found.setdefault(key, []).append(best[1])
    return found


def agreed_levels(marks: dict[str, list[float]]) -> dict[str, float]:
    """층마다 가장 많이 나온 값. 두 번 이상 나왔거나 값이 하나뿐일 때만 믿는다."""
    agreed = {}
    for key, values in marks.items():
        counts: dict[float, int] = {}
        for v in values:
            counts[round(v)] = counts.get(round(v), 0) + 1
        value, n = max(counts.items(), key=lambda kv: (kv[1], -abs(kv[0])))
        if n >= 2 or len(counts) == 1:
            agreed[key] = float(value)
    return agreed


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
