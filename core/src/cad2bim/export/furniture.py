"""도면의 가구·위생기구를 앱에서 보여 줄 형태로 내보낸다.

- 선(segments): 그린 선 그대로 층 바닥에 둔다. 평면 기호처럼 보인다. 입체로 만든 가구의 선은 뺀다.
- 입체(solids): 가구 하나하나를 찾아(furniture_items) 종류별 모양으로 세운다(furniture_models).
  의자는 다리·좌판·등받이, 탁자는 상판·다리, 조리대는 받침·몸통·상판처럼 부재를 나눠 만든다.
  바닥 모양은 도면 그대로이고 높이는 추정이다(평면에는 높이가 없다).
- 목록(items): 찾은 가구의 종류와 근거(이름표·기호·크기). 화면의 요약에 쓴다.
좌표는 building.dae와 같다(mm, Z 위). 모델 파일(DAE·IFC·SKP)에는 넣지 않는다.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import shapely
from shapely.geometry import Point, Polygon

from ..preprocess import load_primitives
from ..preprocess.primitive import Primitive
from .collada import common_base
from .furniture_items import attach_parts, classify, face_tables, find_items
from .furniture_models import build

_ROLE = "FURNITURE"
_ARC_STEP_DEG = 15.0  # 호를 이 각도마다 꺾은선으로 바꾼다
_MARGIN_MM = 500.0  # 건물 범위 밖으로 이만큼까지 받는다(외부 벤치 등)
_MIN_AREA_MM2 = 0.01e6  # 이보다 작은 도형은 손잡이·기호다
_MAX_AREA_MM2 = 12e6  # 이보다 큰 도형은 가구가 아니라 구역 표시다


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


def _chain(p: Primitive) -> tuple[list[tuple[float, float]], bool] | None:
    """요소 하나를 이어진 점 목록으로. (점 목록, 직선 한 토막인가). 닫힌 도형은 첫 점으로 돌아온다."""
    segments = _segments(p)
    if not segments:
        return None
    points = [segments[0][0]] + [b for _, b in segments]
    return points, p.kind in ("line", "polyline") and len(points) == 2


def _ring(p: Primitive) -> tuple[float, float, float, float] | None:
    """원·호의 (중심 x, 중심 y, 반지름, 도는 각도)."""
    if p.kind not in ("circle", "arc"):
        return None
    g = p.geometry
    cx, cy = g["center"]
    if p.kind == "circle":
        return cx, cy, g["radius"], 360.0
    a0 = math.atan2(g["start"][1] - cy, g["start"][0] - cx)
    a1 = math.atan2(g["end"][1] - cy, g["end"][0] - cx)
    return cx, cy, g["radius"], math.degrees((a1 - a0) % (2 * math.pi) or 2 * math.pi)


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
    """한 층의 (선분, 입체). 입체는 {k: 종류, m: 재질, p: [[x, y], ...], h: 구멍들, z0, z1}이고 높이는 층 바닥 기준이다."""
    segments, solids, _ = floor_objects(storey, drawing, roles, base, cfg, cache_dir)
    return segments, solids


def floor_objects(
    storey: Any, drawing: Path, roles: dict[str, str], base: tuple[float, float], cfg: dict[str, Any], cache_dir: Path | None,
) -> tuple[list[list[float]], list[dict[str, Any]], list[dict[str, Any]]]:
    """한 층의 (선분, 입체, 가구 목록). 가구 목록은 {kind, how, at: [x, y]}다."""
    ir = storey.part.ir
    box = _extent(ir)
    if box is None:
        return [], [], []
    scale = float(ir["source"].get("unit_scale") or 1.0)
    ox, oy = ir["origin_offset"]
    dx = ox + storey.shift[0] - base[0]
    dy = oy + storey.shift[1] - base[1]
    raw, _ = load_primitives(drawing, cfg, cache_dir)
    export = cfg.get("export", cfg)
    rules = [(re.compile(r["pattern"], re.IGNORECASE), r) for r in export.get("furniture_labels", [])]
    reach = export.get("furniture_label_reach_mm", 0.0)
    shafts = list(getattr(storey, "elevators", []))  # 승강로 안의 X 표시는 가구가 아니다
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
    shapes: list[tuple[Polygon, str]] = []
    chains: list[tuple[list[tuple[float, float]], str, bool]] = []
    labels: list[tuple[Point, str, bool]] = []
    rings: list[tuple[float, float, float, float, str]] = []
    inside = lambda pts: all(box[0] <= x <= box[2] and box[1] <= y <= box[3] for x, y in pts)  # noqa: E731
    in_shaft = lambda pts: any(s.buffer(50.0).contains(Point(x, y)) for s in shafts for x, y in pts)  # noqa: E731
    for p in raw:
        if p.kind in ("text", "leader") and rules:
            local = p.transformed(scale, -ox, -oy)
            labels.append((Point(local.geometry["position"]), local.geometry["text"], p.kind == "leader"))
        if roles.get(p.layer) != _ROLE:
            continue
        local = p.transformed(scale, -ox, -oy)
        for a, b in _segments(local):
            # 다른 평면(같은 도면의 다른 층)이나 범례의 가구는 뺀다.
            if a == b or not inside((a, b)) or in_shaft((a, b)):
                continue
            lines.append((a, b))
        chain = _chain(local)
        if chain is not None and inside(chain[0]) and not in_shaft(chain[0]):
            chains.append((chain[0], p.layer, chain[1]))
            ring = _ring(local)
            if ring is not None:
                rings.append((*ring, p.layer))
        outline = _outline(local)
        if outline is not None and _MIN_AREA_MM2 <= outline.area <= _MAX_AREA_MM2 and inside(outline.exterior.coords):
            if not any(s.buffer(50.0).contains(outline) for s in shafts):
                shapes.append((outline, p.layer))

    items = attach_parts(find_items(shapes, chains, rings))
    items = classify(items, labels, rules, reach)
    face_tables(items)
    solids = []
    listing = []
    made: list[Polygon] = []
    ring = lambda polygon: [[round(x + dx, 1), round(y + dy, 1)] for x, y in list(polygon.exterior.coords)[:-1]]  # noqa: E731
    for item in items:
        parts = build(item)
        if not parts:
            continue
        for part in parts:
            shape = part["polygon"].simplify(5.0)
            if shape.is_empty or not isinstance(shape, Polygon):
                continue
            solid = {"k": part["k"], "m": part["m"], "p": ring(shape), "z0": part["z0"], "z1": part["z1"]}
            if part["holes"]:
                solid["h"] = [ring(hole.simplify(5.0)) for hole in part["holes"]]
            solids.append(solid)
        made.append(item.polygon if item.tank is None else item.polygon.union(item.tank))
        c = item.polygon.centroid
        listing.append({"kind": item.kind, "how": item.how, "at": [round(c.x + dx, 1), round(c.y + dy, 1)]})
    # 입체로 만든 가구의 평면 선은 뺀다. 입체 아래에 깔려 보이지 않거나 다리 사이로 비쳐 지저분하다.
    cover = shapely.STRtree([m.buffer(15.0) for m in made]) if made else None
    segments = []
    for a, b in lines:
        if cover is not None and cover.query(shapely.LineString([a, b]), predicate="within").size:
            continue
        segments.append([round(a[0] + dx, 1), round(a[1] + dy, 1), round(b[0] + dx, 1), round(b[1] + dy, 1)])
    return segments, solids, listing


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
        segments, solids, items = floor_objects(s, drawing, roles, base, cfg, out_dir / "_prims")
        floors[s.key] = {"z": round(s.elevation, 1), "segments": segments, "solids": solids, "items": items}
        total += len(segments)
    from .elevator import elevator_data

    elevators = elevator_data(storeys, base, cfg.get("export", cfg))
    (out_dir / "furniture.json").write_text(
        json.dumps({"unit": "mm", "up": "Z", "floors": floors, "elevators": elevators}, ensure_ascii=False), encoding="utf-8"
    )
    return total
