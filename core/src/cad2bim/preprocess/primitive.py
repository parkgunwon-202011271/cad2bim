"""전처리 산출물 Primitive. 이후 단계는 ezdxf를 몰라도 되도록 평면 기하만 담는다."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

BBox = tuple[float, float, float, float]

# geometry 키별 변환 방식. 새 키를 추가하면 여기에도 넣는다.
_POINT_KEYS = frozenset({"p1", "p2", "center", "start", "end", "position"})
_POINTS_KEYS = frozenset({"points"})
_RINGS_KEYS = frozenset({"rings"})
_LENGTH_KEYS = frozenset({"radius", "semi_major", "semi_minor", "height", "measured"})


@dataclass(frozen=True)
class Primitive:
    """kind별 geometry:

    line      {p1, p2}
    polyline  {points, closed}
    solid     {points}  채워진 다각형 (SOLID, TRACE)
    arc       {center, radius, start, end}  start→end는 반시계 방향
    circle    {center, radius}
    ellipse   {center, semi_major, semi_minor, points}  points는 평탄화한 호
    text      {position, text, height, rotation, centered}
    leader    {position, text, height, rotation, centered}  지시선 글자. position은 화살표 끝(가리키는 대상)
    hatch     {rings, pattern, solid}
    dimension {p1, p2, position, measured, value}  value는 렌더링된 표기값(숫자), 없으면 None
    block_ref {position, name, rotation, xscale, yscale}  블록 삽입점. 문·창 블록 인식용
    """

    id: str
    kind: str
    geometry: dict[str, Any]
    layer: str
    source_handles: tuple[str, ...]
    block_path: tuple[str, ...] = ()
    linetype: str = "Continuous"
    color: int = 256
    extra: dict[str, Any] = field(default_factory=dict)

    def transformed(self, scale: float, dx: float, dy: float) -> "Primitive":
        return self._mapped(lambda p: (p[0] * scale + dx, p[1] * scale + dy), scale)

    def rotated(self, angle: float) -> "Primitive":
        """원점 기준 회전. 호의 방향은 유지된다."""
        c, s = math.cos(angle), math.sin(angle)
        return self._mapped(lambda p: (p[0] * c - p[1] * s, p[0] * s + p[1] * c), 1.0)

    def _mapped(self, pt: Any, scale: float) -> "Primitive":
        geometry: dict[str, Any] = {}
        for key, value in self.geometry.items():
            if value is None:
                geometry[key] = None
            elif key in _POINT_KEYS:
                geometry[key] = pt(value)
            elif key in _POINTS_KEYS:
                geometry[key] = [pt(p) for p in value]
            elif key in _RINGS_KEYS:
                geometry[key] = [[pt(p) for p in ring] for ring in value]
            elif key in _LENGTH_KEYS:
                geometry[key] = value * scale
            else:
                geometry[key] = value
        return replace(self, geometry=geometry)

    def points(self) -> list[tuple[float, float]]:
        """범위 계산에 쓰는 대표 점들."""
        g = self.geometry
        match self.kind:
            case "line":
                return [g["p1"], g["p2"]]
            case "polyline" | "ellipse" | "solid":
                return list(g["points"])
            case "arc" | "circle":
                (cx, cy), r = g["center"], g["radius"]
                return [(cx - r, cy - r), (cx + r, cy + r)]
            case "text" | "leader":
                return [g["position"]]
            case "hatch":
                return [p for ring in g["rings"] for p in ring]
            case "dimension":
                return [g["p1"], g["p2"], g["position"]]
            case _:
                return []

    def bbox(self) -> BBox | None:
        # 범위를 여러 단계에서 반복해 묻는다. 큰 도면에서는 다시 계산하는 비용이 크다.
        if "_bbox" in self.__dict__:
            return self.__dict__["_bbox"]
        pts = self.points()
        box = None
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            box = (min(xs), min(ys), max(xs), max(ys))
        self.__dict__["_bbox"] = box
        return box

    def extent(self) -> BBox | None:
        """영역 판정용 범위. 블록 삽입점은 점으로 본다. 꼭짓점이 없는 요소는 None이다."""
        box = self.bbox()
        if box is None and self.kind == "block_ref":
            x, y = self.geometry["position"]
            return (x, y, x, y)
        return box


def union_bbox(prims: list[Primitive]) -> BBox | None:
    boxes = [b for b in (p.bbox() for p in prims) if b]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


_ARC_MIN_POINTS = 5
_ARC_RADIUS_TOL = 0.015
_DOOR_ARC_MIN_RADIUS = 500.0
_DOOR_ARC_MAX_RADIUS = 2000.0
_QUARTER_MIN_DEG = 75.0
_QUARTER_MAX_DEG = 105.0


def faceted_door_arc(p: Primitive) -> Primitive | None:
    """짧은 선 여러 개로 그린 문 여닫이 호를 호로 돌려준다. 호로 저장하지 않는 프로그램이 많다.

    문 크기의 약 90도 호일 때만 바꾼다. 곡선 벽을 호로 바꾸면 벽 인식이 달라지기 때문이다.
    """
    if p.kind != "polyline" or p.geometry.get("closed"):
        return None
    points = p.geometry["points"]
    if len(points) < _ARC_MIN_POINTS:
        return None
    (x1, y1), (x2, y2), (x3, y3) = points[0], points[len(points) // 2], points[-1]
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6:
        return None
    s1, s2, s3 = x1 * x1 + y1 * y1, x2 * x2 + y2 * y2, x3 * x3 + y3 * y3
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    radius = math.dist((cx, cy), points[0])
    if not _DOOR_ARC_MIN_RADIUS <= radius <= _DOOR_ARC_MAX_RADIUS:
        return None
    if any(abs(math.dist((cx, cy), q) - radius) > radius * _ARC_RADIUS_TOL for q in points):
        return None
    angles = [math.atan2(q[1] - cy, q[0] - cx) for q in points]
    steps = [(b - a + math.pi) % (2 * math.pi) - math.pi for a, b in zip(angles, angles[1:])]
    if not (all(v > 0 for v in steps) or all(v < 0 for v in steps)):
        return None
    sweep = abs(sum(steps))
    if not _QUARTER_MIN_DEG <= math.degrees(sweep) <= _QUARTER_MAX_DEG:
        return None
    start, end = (points[0], points[-1]) if steps[0] > 0 else (points[-1], points[0])
    return replace(p, kind="arc", geometry={"center": (cx, cy), "radius": radius, "start": tuple(start), "end": tuple(end)})

