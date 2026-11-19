"""도면의 가구·위생기구를 앱에서 보여 줄 형태로 내보낸다.

- 선(segments): 그린 선 그대로 층 바닥에 둔다. 평면 기호처럼 보인다.
- 입체(solids): 닫힌 도형마다 크기와 레이어로 종류를 가늠해 높이를 준다. 평면에는 높이가 없고 이 도면들은
  블록 이름도 없어 추정이다(의자 0.45m, 책상 0.75m, 수납장 1.8m, 변기 0.4m, 세면대 0.85m, 칸막이 2.0m).
좌표는 building.dae와 같다(mm, Z 위). 모델 파일(DAE·IFC·SKP)에는 넣지 않는다.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from shapely.geometry import Point, Polygon

from ..preprocess import load_primitives
from ..preprocess.primitive import Primitive
from .collada import common_base

_ROLE = "FURNITURE"
_ARC_STEP_DEG = 15.0  # 호를 이 각도마다 꺾은선으로 바꾼다
_MARGIN_MM = 500.0  # 건물 범위 밖으로 이만큼까지 받는다(외부 벤치 등)
# 위생기구 레이어 이름. 같은 FURNITURE 역할이라도 높이와 모양이 다르다.
_FIXTURE = ("fixt", "plumb", "위생", "toilet", "변기")
_MIN_AREA_MM2 = 0.01e6  # 이보다 작은 도형은 손잡이·기호다
_MAX_AREA_MM2 = 12e6  # 이보다 큰 도형은 가구가 아니라 구역 표시다
_NESTED_SHARE = 0.9  # 같은 레이어의 더 큰 도형에 이만큼 들어가면 그 가구의 세부 선이다
_TABLE_TOP = (720.0, 750.0)
_LEG_MM = 50.0


def _segments(p: Primitive) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    g = p.geometry
    if p.kind == "line":
        return [(g["p1"], g["p2"])]
    if p.kind in ("polyline", "solid", "ellipse"):
        points = list(g["points"])
        if p.kind == "solid" or g.get("closed"):
            points.append(points[0])
        return list(zip(points, points[1:]))
    if p.kind in ("arc", "circle"):
        cx, cy = g["center"]
        r = g["radius"]
        if p.kind == "circle":
            a0, sweep = 0.0, 2 * math.pi
        else:
            a0 = math.atan2(g["start"][1] - cy, g["start"][0] - cx)
            a1 = math.atan2(g["end"][1] - cy, g["end"][0] - cx)
            sweep = (a1 - a0) % (2 * math.pi) or 2 * math.pi
        n = max(2, math.ceil(math.degrees(sweep) / _ARC_STEP_DEG))
        points = [(cx + r * math.cos(a0 + sweep * i / n), cy + r * math.sin(a0 + sweep * i / n)) for i in range(n + 1)]
        return list(zip(points, points[1:]))
    return []


def _outline(p: Primitive) -> Polygon | None:
    """닫힌 도형의 바닥 모양. 열린 선과 글자는 None."""
    g = p.geometry
    if p.kind == "circle":
        return Point(g["center"]).buffer(g["radius"], quad_segs=6)
    if p.kind in ("polyline", "ellipse", "solid") and (p.kind != "polyline" or g.get("closed")):
        points = list(g["points"])
        if len(points) < 3:
            return None
        polygon = Polygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        return polygon if isinstance(polygon, Polygon) and not polygon.is_empty else None
    return None


def _solids(polygon: Polygon, fixture: bool) -> list[tuple[str, Polygon, float, float]]:
    """한 가구의 (종류, 바닥 모양, 아래 높이, 위 높이) 목록. 높이는 층 바닥 기준 mm."""
    box = polygon.minimum_rotated_rectangle
    ring = list(box.exterior.coords)
    sides = sorted((math.dist(ring[0], ring[1]), math.dist(ring[1], ring[2])))
    width, length = sides
    area = polygon.area
    if fixture:
        if width < 120.0:
            return [("partition", polygon, 100.0, 2000.0)]  # 화장실 칸막이. 바닥에서 조금 띄운다
        if length <= 750.0 and area <= 0.3e6:
            return [("fixture", polygon, 0.0, 400.0)]  # 변기·소변기
        if width <= 700.0 and length >= 1000.0:
            return [("counter", polygon, 0.0, 850.0)]  # 세면대 상판
        return [("fixture", polygon, 0.0, 800.0)]
    if width < 100.0:
        return [("partition", polygon, 0.0, 1200.0)]  # 사무실 칸막이
    if length <= 750.0:
        return [("chair", polygon, 0.0, 450.0)]
    if width <= 500.0 and length >= 900.0:
        return [("shelf", polygon, 0.0, 1800.0)]
    # 탁자: 상판과 다리. 둥근 탁자는 가운데 기둥 하나.
    top = [("table", polygon, *_TABLE_TOP)]
    # 사각형을 다 채우지 못하거나 꼭짓점이 많으면 둥근 탁자다.
    if area < box.area * 0.85 or len(polygon.exterior.coords) > 8:
        c = polygon.centroid
        return top + [("table", Point(c.x, c.y).buffer(_LEG_MM, quad_segs=2), 0.0, _TABLE_TOP[0])]
    inner = polygon.buffer(-_LEG_MM * 2, join_style="mitre")
    corners = list(inner.exterior.coords)[:-1] if isinstance(inner, Polygon) and not inner.is_empty else []
    legs = [("table", Point(x, y).buffer(_LEG_MM / 2, cap_style="square"), 0.0, _TABLE_TOP[0]) for x, y in corners[:4]]
    return top + legs


def _label_rule(
    polygon: Polygon, labels: list[tuple[Point, str]], rules: list[tuple[re.Pattern[str], dict[str, Any]]], reach: float,
) -> dict[str, Any] | None:
    """도형 안, 없으면 reach 안에서 가장 가까운 이름표가 맞는 규칙. 실 이름처럼 규칙에 없는 글자는 건너뛴다."""
    best = None
    for point, text in labels:
        distance = 0.0 if polygon.contains(point) else polygon.distance(point)
        if distance > reach:
            continue
        rule = next((r for pattern, r in rules if pattern.search(text)), None)
        if rule is not None and (best is None or distance < best[0]):
            best = (distance, rule)
    return best[1] if best else None


def _extent(ir: dict[str, Any]) -> tuple[float, float, float, float] | None:
    points = [p for w in ir["walls"] for p in (w["start"], w["end"])]
    if not points:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs) - _MARGIN_MM, min(ys) - _MARGIN_MM, max(xs) + _MARGIN_MM, max(ys) + _MARGIN_MM


def floor_furniture(
    storey: Any, drawing: Path, roles: dict[str, str], base: tuple[float, float], cfg: dict[str, Any], cache_dir: Path | None,
) -> list[list[float]]:
    """한 층의 가구 선분 [[x1, y1, x2, y2], ...]. building.dae 좌표(mm)다."""
    return floor_items(storey, drawing, roles, base, cfg, cache_dir)[0]


def floor_items(
    storey: Any, drawing: Path, roles: dict[str, str], base: tuple[float, float], cfg: dict[str, Any], cache_dir: Path | None,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    """한 층의 (선분, 입체). 입체는 {k: 종류, p: [[x, y], ...], z0, z1}이고 높이는 층 바닥 기준이다."""
    ir = storey.part.ir
    box = _extent(ir)
    if box is None:
        return [], []
    scale = float(ir["source"].get("unit_scale") or 1.0)
    ox, oy = ir["origin_offset"]
    dx = ox + storey.shift[0] - base[0]
    dy = oy + storey.shift[1] - base[1]
    raw, _ = load_primitives(drawing, cfg, cache_dir)
    export = cfg.get("export", cfg)
    rules = [(re.compile(r["pattern"], re.IGNORECASE), r) for r in export.get("furniture_labels", [])]
    reach = export.get("furniture_label_reach_mm", 0.0)
    shafts = list(getattr(storey, "elevators", []))  # 승강로 안의 X 표시는 가구가 아니다
    segments = []
    shapes: list[tuple[Polygon, str]] = []
    labels: list[tuple[Point, str]] = []
    inside = lambda pts: all(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in pts)  # noqa: E731
    in_shaft = lambda pts: any(s.buffer(50.0).contains(Point(x, y)) for s in shafts for x, y in pts)  # noqa: E731
    for p in raw:
        if p.kind == "text" and rules:
            local = p.transformed(scale, -ox, -oy)
            labels.append((Point(local.geometry["position"]), local.geometry["text"]))
        if roles.get(p.layer) != _ROLE:
            continue
        local = p.transformed(scale, -ox, -oy)
        for a, b in _segments(local):
            # 다른 평면(같은 도면의 다른 층)이나 범례의 가구는 뺀다.
            if a == b or not inside((a, b)) or in_shaft((a, b)):
                continue
            segments.append([round(a[0] + dx, 1), round(a[1] + dy, 1), round(b[0] + dx, 1), round(b[1] + dy, 1)])
        outline = _outline(local)
        if outline is not None and _MIN_AREA_MM2 <= outline.area <= _MAX_AREA_MM2 and inside(outline.exterior.coords):
            if not any(s.buffer(50.0).contains(outline) for s in shafts):
                shapes.append((outline, p.layer))

    solids = []
    shapes.sort(key=lambda item: -item[0].area)
    kept: list[tuple[Polygon, str]] = []
    for polygon, layer in shapes:
        # 큰 도형 안에 그린 서랍·등받이 선은 같은 가구의 세부다.
        if any(other_layer == layer and polygon.intersection(other).area >= polygon.area * _NESTED_SHARE for other, other_layer in kept):
            continue
        kept.append((polygon, layer))
        rule = _label_rule(polygon, labels, rules, reach)
        if rule is not None:
            if rule["kind"] == "skip":
                continue
            parts = [(rule["kind"], polygon.simplify(5.0), float(rule["z0"]), float(rule["z1"]))]
        else:
            parts = _solids(polygon.simplify(5.0), any(word in layer.lower() for word in _FIXTURE))
        for kind, shape, z0, z1 in parts:
            if shape.is_empty or not isinstance(shape, Polygon):
                continue
            points = [[round(x + dx, 1), round(y + dy, 1)] for x, y in list(shape.exterior.coords)[:-1]]
            solids.append({"k": kind, "p": points, "z0": z0, "z1": z1})
    return segments, solids


def write_furniture(storeys: list[Any], drawings: dict[str, Path], out_dir: Path, cfg: dict[str, Any]) -> int:
    """drawings는 평면 조각 이름(D03.2) → DXF 경로. 선분 수를 돌려준다."""
    base = common_base(storeys)
    floors = {}
    total = 0
    for s in storeys:
        drawing = drawings.get(s.part.part_id)
        layer_map = out_dir / s.part.part_id / "layer_map.json"
        if drawing is None or not layer_map.is_file():
            continue
        roles = {name: info["role"] for name, info in json.loads(layer_map.read_text(encoding="utf-8"))["layers"].items()}
        segments, solids = floor_items(s, drawing, roles, base, cfg, out_dir / "_prims")
        floors[s.key] = {"z": round(s.elevation, 1), "segments": segments, "solids": solids}
        total += len(segments)
    from .elevator import elevator_data

    elevators = elevator_data(storeys, base, cfg.get("export", cfg))
    (out_dir / "furniture.json").write_text(
        json.dumps({"unit": "mm", "up": "Z", "floors": floors, "elevators": elevators}, ensure_ascii=False), encoding="utf-8"
    )
    return total
