"""비정형 기둥과 짧은 벽 조각. 벽이나 기둥으로 안 잡힌 닫힌 도형을 그린 모양 그대로 살린다.

모서리의 오각형 기둥이나 300mm 미만 벽 조각이 빠지면 벽 사이가 비어 찌그러져 보인다.
벽과 기둥 레이어만 본다. 다른 레이어의 닫힌 도형은 가구나 설비일 수 있다.
"""
from __future__ import annotations

from typing import Any

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from ..layers import LayerMap
from ..preprocess.primitive import Primitive
from .columns import ColumnDraft
from .walls import WallDraft


def find_piers(
    prims: list[Primitive], layers: LayerMap, walls: list[WallDraft], columns: list[ColumnDraft], cfg: dict[str, Any]
) -> list[ColumnDraft]:
    solids = [
        LineString([w.start, w.end]).buffer(w.thickness / 2 + cfg["pier_cover_margin_mm"], cap_style="square", join_style="mitre")
        for w in walls if w.length > 0
    ] + [c.polygon.buffer(cfg["pier_cover_margin_mm"]) for c in columns]
    covered = unary_union(solids) if solids else None
    result: list[ColumnDraft] = []
    taken: list[Polygon] = []
    # 큰 것부터 받는다. 겹쳐 그린 외곽선과 안쪽 선은 한 번만 만든다.
    candidates = []
    for p in prims:
        if p.kind != "polyline" or not p.geometry["closed"] or layers.role(p.layer) not in ("WALL", "COLUMN"):
            continue
        if len(p.geometry["points"]) < 3:
            continue
        polygon = Polygon(p.geometry["points"])
        if not polygon.is_valid or polygon.is_empty:
            continue
        # 기둥 레이어의 사각형은 기둥 크기 판단을 이미 거쳤다. 여기서 되살리지 않는다.
        if layers.role(p.layer) == "COLUMN" and polygon.area >= polygon.minimum_rotated_rectangle.area * 0.98:
            continue
        x0, y0, x1, y1 = polygon.bounds
        if max(x1 - x0, y1 - y0) > cfg["pier_max_size_mm"] or polygon.area < cfg["pier_min_area_mm2"]:
            continue
        candidates.append((polygon, p))
    for polygon, p in sorted(candidates, key=lambda item: -item[0].area):
        if covered is not None and polygon.intersection(covered).area >= polygon.area * cfg["pier_max_covered_share"]:
            continue
        if any(polygon.intersection(other).area >= polygon.area * 0.5 for other in taken):
            continue
        taken.append(polygon)
        x0, y0, x1, y1 = polygon.bounds
        centroid = polygon.centroid
        result.append(ColumnDraft(
            "polygon", (centroid.x, centroid.y), (x1 - x0, y1 - y0), 0.0, polygon, {p.id}, list(p.source_handles), p.layer, False,
        ))
    return result
