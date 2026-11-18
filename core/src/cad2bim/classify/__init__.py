"""도면 종류 분류. 파일명을 쓰지 않고 도면 내용만 본다 (상세설계 2장).

근거 둘을 따로 구한다.
- 글자: 도면 안의 도면명. 표제란처럼 가장자리에 있는 글자에 무게를 둔다.
- 도형: 등고선, 문 호, 그리드 기호, 층 높이 표기 등. 언어와 회사 양식에 덜 영향받는다.

축척을 몰라도 동작한다. 단위 판정에 실패한 도면도 분류해야 하기 때문이다.
임계값은 초기값이며 파일명 기준 채점으로 조정한다.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..preprocess.primitive import Primitive, faceted_door_arc

KINDS = ("plan", "vertical", "site", "table", "detail")

_EDGE_BAND = 0.2
_EDGE_BONUS = 1.0
_SIZE_WEIGHT = 2.0
_COUNT_WEIGHT = 0.2
_COUNT_CAP = 1.0
_CURVE_MIN_VERTICES = 10
_CURVE_OFF_AXIS_DEG = 5.0
_SITE_CURVE_SHARE = 0.3
_TABLE_TEXT_SHARE = 0.25
_TRUSTED_SHAPES = ("table", "site")
_MIN_TEXTS_FOR_TABLE = 30
_LEVEL_MARKS = 3
_DOOR_ARCS = 3
_GRID_TAG_MAX_CHARS = 3
_GRID_MIN_TAGS = 4
_GRID_MIN_SPREAD = 0.3
_QUARTER_ARC_TOL_DEG = 10.0


@dataclass(frozen=True)
class SheetGuess:
    kind: str  # 최종 판단
    by_text: str  # 글자 근거만의 판단
    by_shape: str  # 도형 근거만의 판단
    level_marks: int = 0
    door_arcs: int = 0
    grid_axes: int = 0
    curve_share: float = 0.0
    text_share: float = 0.0


class SheetDictionary:
    def __init__(self, data: dict[str, Any]):
        self.kinds = [(k["kind"], re.compile(k["pattern"], re.I)) for k in data["kinds"]]
        self.level = re.compile(data["level_mark"], re.I)
        self.ignored = re.compile(data["title_ignore"], re.I)
        suffix = data["title_suffix"]
        self.titles = [
            (k["kind"], re.compile(rf"(?:{k['pattern']})\s*(?:{suffix})?\s*$", re.I)) for k in data["title_kinds"]
        ]

    def kind_of(self, text: str) -> str | None:
        return next((kind for kind, pattern in self.kinds if pattern.search(text)), None)


def _title_kind(dictionary: SheetDictionary, text: str) -> str | None:
    if dictionary.ignored.search(text):
        return None
    return next((kind for kind, pattern in dictionary.titles if pattern.search(text)), None)


SheetDictionary.title_kind = _title_kind


def label_from_name(name: str, dictionary: SheetDictionary) -> str | None:
    """파일명으로 정한 채점용 정답. 분류기에는 넘기지 않는다."""
    return dictionary.kind_of(name)


def _extent(prims: list[Primitive]) -> tuple[float, float, float, float] | None:
    boxes = [b for b in (p.bbox() for p in prims) if b]
    if not boxes:
        return None
    array = np.array(boxes, dtype=float)
    return (array[:, 0].min(), array[:, 1].min(), array[:, 2].max(), array[:, 3].max())


def _by_text(texts: list[Primitive], extent: Any, dictionary: SheetDictionary) -> str:
    """도면명 형태의 글자만 본다. 글자가 크거나 가장자리에 있을수록 무게를 둔다.

    제목은 도면 가운데 아래에 크게 쓰기도 하므로 위치만으로 판단하지 않는다.
    """
    best: dict[str, float] = {}
    counts: dict[str, int] = {}
    x0, y0, x1, y1 = extent
    width, height = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    tallest = max((p.geometry["height"] for p in texts), default=0.0) or 1.0
    for p in texts:
        kind = dictionary.title_kind(p.geometry["text"])
        if kind is None:
            continue
        x, y = p.geometry["position"]
        edge = (
            x < x0 + width * _EDGE_BAND or x > x1 - width * _EDGE_BAND
            or y < y0 + height * _EDGE_BAND or y > y1 - height * _EDGE_BAND
        )
        weight = 1.0 + _SIZE_WEIGHT * p.geometry["height"] / tallest + (_EDGE_BONUS if edge else 0.0)
        best[kind] = max(best.get(kind, 0.0), weight)
        counts[kind] = counts.get(kind, 0) + 1
    scores = {k: best[k] + min(counts[k] * _COUNT_WEIGHT, _COUNT_CAP) for k in best}
    if not scores:
        return "unknown"
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return "unknown"
    return ranked[0][0]


def curve_share(prims: list[Primitive]) -> float:
    """꼭짓점 많은 폴리선 중 건물 주축과 어긋난 선의 길이 비율. 등고선은 높고 벽 폴리선은 0이다."""
    angles: list[float] = []
    lengths: list[float] = []
    in_long: list[bool] = []
    for p in prims:
        if p.kind == "line":
            pairs, is_long = [(p.geometry["p1"], p.geometry["p2"])], False
        elif p.kind == "polyline":
            points = list(p.geometry["points"])
            pairs, is_long = list(zip(points, points[1:])), len(points) >= _CURVE_MIN_VERTICES
        else:
            continue
        for a, b in pairs:
            length = math.dist(a, b)
            if length > 0:
                angles.append(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 90)
                lengths.append(length)
                in_long.append(is_long)
    if not angles:
        return 0.0
    angle = np.array(angles)
    weight = np.array(lengths)
    hist, _ = np.histogram(angle, bins=90, range=(0, 90), weights=weight)
    main = float(np.argmax(hist)) + 0.5
    off = np.abs(angle - main)
    off = np.minimum(off, 90 - off)
    return float(weight[(off > _CURVE_OFF_AXIS_DEG) & np.array(in_long)].sum() / weight.sum())


def _quarter_arcs(prims: list[Primitive]) -> int:
    """약 90도 호. 문 여닫이 표시이며 입면과 단면에는 드물다. 크기와 무관하게 센다."""
    count = 0
    for original in prims:
        p = faceted_door_arc(original) or original
        if p.kind != "arc":
            continue
        (cx, cy), start, end = p.geometry["center"], p.geometry["start"], p.geometry["end"]
        sweep = math.degrees(
            (math.atan2(end[1] - cy, end[0] - cx) - math.atan2(start[1] - cy, start[0] - cx)) % (2 * math.pi)
        )
        count += abs(sweep - 90) <= _QUARTER_ARC_TOL_DEG
    return count


def _grid_axes(prims: list[Primitive], extent: Any) -> int:
    """글자가 든 원(그리드 기호)이 가로·세로로 퍼진 축의 수. 평면은 2, 입면과 단면은 보통 1이다."""
    texts = [p for p in prims if p.kind == "text" and 0 < len(p.geometry["text"]) <= _GRID_TAG_MAX_CHARS]
    circles = [p for p in prims if p.kind == "circle"]
    if not texts or not circles:
        return 0
    positions = np.array([p.geometry["position"] for p in texts], dtype=float)
    centers = []
    for c in circles:
        center = np.array(c.geometry["center"], dtype=float)
        if (np.linalg.norm(positions - center, axis=1) <= c.geometry["radius"]).any():
            centers.append(center)
    if len(centers) < _GRID_MIN_TAGS:
        return 0
    array = np.array(centers)
    spread = array.max(axis=0) - array.min(axis=0)
    size = (max(extent[2] - extent[0], 1e-9), max(extent[3] - extent[1], 1e-9))
    return int(spread[0] >= size[0] * _GRID_MIN_SPREAD) + int(spread[1] >= size[1] * _GRID_MIN_SPREAD)


def _paper_text(paper: Sequence[Primitive], dictionary: SheetDictionary) -> str:
    extent = _extent(list(paper))
    texts = [p for p in paper if p.kind == "text"]
    return _by_text(texts, extent, dictionary) if extent and texts else "unknown"


def classify_sheet(
    prims: list[Primitive], dictionary: SheetDictionary, paper: Sequence[Primitive] = ()
) -> SheetGuess:
    """표제란은 레이아웃에 있는 경우가 많아 그쪽 글자를 먼저 본다."""
    paper_text = _paper_text(paper, dictionary)
    extent = _extent(prims)
    if extent is None:
        return SheetGuess(paper_text, paper_text, "unknown")
    texts = [p for p in prims if p.kind == "text"]
    by_text = paper_text if paper_text != "unknown" else _by_text(texts, extent, dictionary)

    levels = sum(bool(dictionary.level.search(p.geometry["text"])) for p in texts)
    arcs = _quarter_arcs(prims)
    axes = _grid_axes(prims, extent)
    curves = curve_share(prims)
    lines = sum(p.kind in ("line", "polyline") for p in prims)
    text_share = len(texts) / (lines + len(texts)) if lines + len(texts) else 0.0

    if curves >= _SITE_CURVE_SHARE:
        by_shape = "site"
    elif len(texts) >= _MIN_TEXTS_FOR_TABLE and text_share >= _TABLE_TEXT_SHARE:
        by_shape = "table"
    elif levels >= _LEVEL_MARKS and arcs < _DOOR_ARCS:
        by_shape = "vertical"
    elif arcs >= _DOOR_ARCS or axes == 2:
        by_shape = "plan"
    else:
        by_shape = "unknown"

    # 실제 도면에서 평면 판단은 대부분의 도면에 나와 근거가 되지 못했다. 표와 등고선만 믿는다.
    kind = by_text if by_text != "unknown" else (by_shape if by_shape in _TRUSTED_SHAPES else "unknown")
    # 평면 옆에 일람표를 함께 그린 도면이 많다. 글자 비율이 낮고 문 호나 격자가 있으면 표가 아니다.
    if by_text == "table" and by_shape == "plan":
        kind = "plan"
    return SheetGuess(kind, by_text, by_shape, levels, arcs, axes, round(curves, 3), round(text_share, 3))


__all__ = ["KINDS", "SheetDictionary", "SheetGuess", "classify_sheet", "curve_share", "label_from_name"]
