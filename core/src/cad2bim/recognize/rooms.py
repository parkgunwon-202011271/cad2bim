"""실 인식 (상세설계 5.6). 벽과 기둥을 뺀 닫힌 영역이 실이고, 안의 글자로 이름을 붙인다.

벽은 개구부 위치에서도 이어져 있으므로 문이 있어도 영역이 닫힌다.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from ..layers import LayerMap
from ..preprocess.primitive import Primitive
from .columns import ColumnDraft
from .walls import WallDraft, small_instances

_EXCLUDED_ROLES = ("DIM", "GRID", "OTHER", "FURNITURE", "STAIR", "DOOR", "WINDOW", "HATCH")
_AREA_NOTE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:㎡|m2|m²)", re.I)
_SHORT_TAG = re.compile(r"^[A-Za-z0-9.\-_]{1,3}$")
_HAS_LETTER = re.compile(r"[A-Za-z가-힣]")


@dataclass
class RoomDraft:
    names: list[str]
    point: tuple[float, float]
    polygon: list[tuple[float, float]]
    area_m2: float
    wall_indices: list[int]
    handles: list[str] = field(default_factory=list)
    area_note_ok: bool | None = None  # 도면에 적힌 면적과 맞는가. 적혀 있지 않으면 None
    has_name_role: bool = False


def _is_room_name(text: str) -> bool:
    """숫자, 치수, 면적 표기, 그리드 기호를 뺀 글자."""
    text = text.strip()
    if len(text) < 2 or not _HAS_LETTER.search(text) or _AREA_NOTE.search(text):
        return False
    return not _SHORT_TAG.match(text)


def _solid(wall: WallDraft) -> Any:
    # 끝을 두께 반만큼 늘려 코너와 접합이 빈틈없이 닫히게 한다.
    return LineString([wall.start, wall.end]).buffer(wall.thickness / 2, cap_style="square", join_style="mitre")


def recognize_rooms(
    prims: list[Primitive], layers: LayerMap, walls: list[WallDraft], columns: list[ColumnDraft],
    cfg: dict[str, Any], part_block_max_mm: float,
) -> tuple[list[RoomDraft], list[tuple[list[str], tuple[float, float, float, float]]]]:
    """(실 목록, 벽이 닫히지 않아 실로 만들지 못한 이름 있는 영역)."""
    if not walls:
        return [], []
    solid_walls = [w for w in walls if w.length > 0]
    solids = [_solid(w) for w in solid_walls]
    axes = [p for w in walls for p in (w.start, w.end)]
    xs, ys = [float(p[0]) for p in axes], [float(p[1]) for p in axes]
    frame = box(min(xs), min(ys), max(xs), max(ys))
    free = frame.difference(unary_union(solids + [c.polygon for c in columns]))
    regions = [g for g in getattr(free, "geoms", [free]) if isinstance(g, Polygon) and g.area >= cfg["min_area_mm2"]]
    if not regions:
        return [], []

    small = small_instances(prims, part_block_max_mm)
    texts = [
        p for p in prims
        if p.kind == "text" and layers.role(p.layer) not in _EXCLUDED_ROLES and p.extra.get("instance", 0) not in small
    ]
    region_tree = shapely.STRtree(regions)
    in_region: dict[int, list[Primitive]] = {}
    for p in texts:
        hits = region_tree.query(Point(p.geometry["position"]), predicate="within")
        for index in hits.tolist():
            in_region.setdefault(index, []).append(p)

    solid_tree = shapely.STRtree(solids)
    frame_line = frame.exterior
    rooms = []
    open_regions = []
    for index, region in enumerate(regions):
        members = in_region.get(index, [])
        # 바깥 경계에 닿았으면 벽으로 닫히지 않은 영역이다. 글자가 있어도 실을 지어내지 않는다.
        if region.exterior.distance(frame_line) <= cfg["outside_tol_mm"]:
            if any(_is_room_name(p.geometry["text"]) for p in members):
                open_regions.append((sorted({h for p in members for h in p.source_handles}), region.bounds))
            continue
        names = list(dict.fromkeys(p.geometry["text"].strip() for p in members if _is_room_name(p.geometry["text"])))
        if not names and not members and region.area < cfg["unnamed_min_area_mm2"]:
            continue
        area = region.area / 1e6
        touching = solid_tree.query(region, predicate="dwithin", distance=cfg["boundary_tol_mm"]).tolist()
        notes = [float(m.group(1)) for p in members if (m := _AREA_NOTE.search(p.geometry["text"]))]
        note_ok = None
        if notes:
            # 도면의 면적 표기는 벽 안쪽 면적일 수도, 벽 중심선 면적일 수도 있다. 둘 다 맞는 표기다.
            half = statistics.median(solid_walls[i].thickness for i in touching) / 2 if touching else 0.0
            centerline = region.buffer(half, join_style="mitre").area / 1e6
            note_ok = any(
                abs(value / n - 1) <= cfg["area_note_tolerance"] for n in notes if n > 0 for value in (area, centerline)
            )
        named = [p for p in members if _is_room_name(p.geometry["text"])]
        point = named[0].geometry["position"] if named else (region.representative_point().x, region.representative_point().y)
        rooms.append(RoomDraft(
            names or ["미지정"], (float(point[0]), float(point[1])),
            [(float(x), float(y)) for x, y in region.exterior.coords], round(area, 3), sorted(touching),
            sorted({h for p in members for h in p.source_handles}), note_ok,
            any(layers.role(p.layer) == "ROOM_TEXT" for p in members),
        ))
    return rooms, open_regions
