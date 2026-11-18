"""표준 정답 형식(*.truth.json) 읽기. 실제 정답 모델도 이 형식으로 변환한다 (M6)."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, Point as ShapelyPoint
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

Point = tuple[float, float]

MATERIAL_TO_KIND = {"RC": "concrete", "BLOCK": "masonry"}

# 합성 생성기는 벽 솔리드를 양 끝으로 100mm 늘려 만든다.
_SYNTHETIC_WALL_EXTEND_MM = 100.0
_MIN_REGION_AREA_MM2 = 1e5


@dataclass(frozen=True)
class TruthWall:
    id: str
    p1: Point
    p2: Point
    thickness: float
    kind: str


@dataclass(frozen=True)
class TruthColumn:
    center: Point
    size: tuple[float, float]


@dataclass(frozen=True)
class TruthOpening:
    id: str
    kind: str
    wall: str
    center: Point
    width: float
    hinge: Point | None = None
    swing_dir: Point | None = None


@dataclass(frozen=True)
class TruthRoom:
    name: str
    label_point: Point
    region_id: int
    clear_area_m2: float
    polygon: BaseGeometry | None = None


@dataclass
class Truth:
    walls: list[TruthWall] = field(default_factory=list)
    columns: list[TruthColumn] = field(default_factory=list)
    openings: list[TruthOpening] = field(default_factory=list)
    rooms: list[TruthRoom] = field(default_factory=list)
    layer_classes: dict[str, list[str]] = field(default_factory=dict)


def _pt(v: list[float] | None) -> Point | None:
    return (float(v[0]), float(v[1])) if v is not None else None


def _size(v: float | list[float]) -> tuple[float, float]:
    if isinstance(v, (int, float)):
        return (float(v), float(v))
    return (float(v[0]), float(v[1]))


def _wall_solid(wall: TruthWall) -> BaseGeometry:
    (x1, y1), (x2, y2) = wall.p1, wall.p2
    length = math.dist(wall.p1, wall.p2)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    extend = _SYNTHETIC_WALL_EXTEND_MM
    line = LineString([(x1 - ux * extend, y1 - uy * extend), (x2 + ux * extend, y2 + uy * extend)])
    return line.buffer(wall.thickness / 2, cap_style="flat")


def derive_room_polygons(truth: Truth) -> Truth:
    """폴리곤이 없는 정답의 실 영역을 벽과 기둥에서 복원한다.

    합성 생성기와 같은 규칙이다. 복원 결과는 정답 면적과 대조해 검증한다 (test_evaluation).
    실제 정답 모델은 Space 형상을 그대로 쓰므로 이 경로를 타지 않는다.
    """
    if not truth.rooms or all(r.polygon is not None for r in truth.rooms):
        return truth
    xs = [p[0] for w in truth.walls for p in (w.p1, w.p2)]
    ys = [p[1] for w in truth.walls for p in (w.p1, w.p2)]
    solids = [_wall_solid(w) for w in truth.walls]
    solids += [
        box(c.center[0] - c.size[0] / 2, c.center[1] - c.size[1] / 2,
            c.center[0] + c.size[0] / 2, c.center[1] + c.size[1] / 2)
        for c in truth.columns
    ]
    free = box(min(xs), min(ys), max(xs), max(ys)).difference(unary_union(solids))
    regions: list[Polygon] = [g for g in getattr(free, "geoms", [free]) if g.area > _MIN_REGION_AREA_MM2]

    rooms = []
    for room in truth.rooms:
        label = ShapelyPoint(room.label_point)
        rooms.append(replace(room, polygon=next((r for r in regions if r.contains(label)), None)))
    return replace(truth, rooms=rooms)


def _room(data: dict[str, Any]) -> TruthRoom:
    polygon = Polygon(data["polygon"]) if data.get("polygon") else None
    return TruthRoom(
        data["name"], _pt(data["label_point"]), int(data["region_id"]),
        float(data["clear_area_m2"]), polygon,
    )


def load_truth(path: Path) -> Truth:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    truth = Truth(
        walls=[
            TruthWall(w["id"], _pt(w["p1"]), _pt(w["p2"]), float(w["thickness"]),
                      MATERIAL_TO_KIND.get(w["material"], "unknown"))
            for w in data["walls"]
        ],
        columns=[TruthColumn(_pt(c["center"]), _size(c["size"])) for c in data["columns"]],
        openings=[
            TruthOpening(o["id"], o["kind"], o["wall"], _pt(o["center"]), float(o["width"]),
                         _pt(o.get("hinge")), _pt(o.get("swing_dir")))
            for o in data["openings"]
        ],
        rooms=[_room(r) for r in data["rooms"]],
        layer_classes=data.get("layer_classes", {}),
    )
    return derive_room_polygons(truth)
