"""슬래브, 지붕, 계단 형상. IFC, DAE, DXF가 같은 값을 쓰도록 한곳에서 만든다."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from shapely.affinity import translate
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union


def wall_top(height: float, cfg: dict[str, Any]) -> float:
    """벽과 기둥은 윗층 슬래브 아래까지다. 층고 전체로 올리면 슬래브와 겹친다."""
    return height - cfg["slab_thickness_mm"]


def glazing_band(height: float, cfg: dict[str, Any]) -> tuple[float, float]:
    """창호 선만 그린 구간의 유리 높이. 평면에 없는 값이라 일반 창과 같은 값을 쓴다."""
    sill = cfg["window_sill_mm"]
    return sill, min(sill + cfg["window_height_mm"], wall_top(height, cfg))


def door_top(width: float, cfg: dict[str, Any]) -> float:
    return cfg["wide_door_height_mm"] if width >= cfg["wide_door_min_width_mm"] else cfg["door_height_mm"]


@dataclass
class StairBox:
    center: tuple[float, float]
    angle: float
    length: float
    width: float
    z0: float
    z1: float


@dataclass
class FloorPlan:
    slabs: list[Polygon] = field(default_factory=list)
    roofs: list[Polygon] = field(default_factory=list)
    stairs: list[tuple[str, list[StairBox], list[tuple[Polygon, float, float]]]] = field(default_factory=list)


def _polygons(shape: Any, min_area_mm2: float) -> list[Polygon]:
    parts = [g for g in getattr(shape, "geoms", [shape]) if isinstance(g, Polygon) and not g.is_empty]
    return [p for p in parts if p.area >= min_area_mm2 and p.is_valid]


def floor_outlines(ir: dict[str, Any], close_mm: float, min_area_m2: float, trim_mm: float = 1000.0) -> list[Polygon]:
    """벽, 기둥, 실을 틈 메워 합친 바닥 외곽.

    튀어나온 벽 하나가 슬래브의 가는 돌기가 되지 않게 깎는다. 깎이면서 끊긴 좁은 복도는 두 덩어리를
    잇는 부분이므로 되살린다. 한쪽만 닿은 부분은 돌기다.
    """
    shapes = [
        LineString([w["start"], w["end"]]).buffer(w["thickness"] / 2, cap_style="square", join_style="mitre")
        for w in ir["walls"] if w["start"] != w["end"]
    ]
    for c in ir["columns"]:
        if c.get("polygon"):
            shapes.append(Polygon(c["polygon"]))
            continue
        half_w, half_d = c["size"][0] / 2, c["size"][1] / 2
        shapes.append(box(c["center"][0] - half_w, c["center"][1] - half_d, c["center"][0] + half_w, c["center"][1] + half_d))
    for room in ir.get("rooms", []):
        polygon = Polygon(room["polygon"])
        if polygon.is_valid and polygon.area > 0:
            shapes.append(polygon)
    if not shapes:
        return []
    closed = unary_union(shapes).buffer(close_mm, join_style="mitre").buffer(-close_mm, join_style="mitre")
    closed = unary_union([Polygon(p.exterior) for p in getattr(closed, "geoms", [closed])])
    opened = closed.buffer(-trim_mm, join_style="mitre").buffer(trim_mm, join_style="mitre")
    bodies = [g for g in getattr(opened, "geoms", [opened]) if not g.is_empty]
    if not bodies:
        bodies = list(getattr(closed, "geoms", [closed]))
    rest = closed.difference(unary_union(bodies))
    necks = [
        piece for piece in getattr(rest, "geoms", [rest])
        if not piece.is_empty and sum(piece.distance(body) <= 1.0 for body in bodies) >= 2
    ]
    # 되살린 부분과 몸통 사이의 틈을 막는다.
    merged = unary_union([*bodies, *necks]).buffer(1.0, join_style="mitre").buffer(-1.0, join_style="mitre")
    outlines = []
    for part in getattr(merged, "geoms", [merged]):
        polygon = Polygon(part.exterior).simplify(20.0)
        if polygon.area >= min_area_m2 * 1e6 and polygon.geom_type == "Polygon" and polygon.is_valid:
            outlines.append(polygon)
    return outlines


def _stair_boxes(stair: dict[str, Any], height: float, elevation: float, cfg: dict[str, Any]) -> tuple[list[StairBox], list[tuple[Polygon, float, float]]]:
    """오르는 방향을 평면에서 알 수 없어 단 묶음을 번갈아 오르는 것으로 가정한다."""
    flights = sorted(stair["flights"], key=lambda f: (f["start"][1], f["start"][0]))
    total = sum(f["treads"] for f in flights) + len(flights)
    rise = height / total
    boxes: list[StairBox] = []
    landings: list[tuple[Polygon, float, float]] = []
    base = elevation
    footprints = []
    for number, flight in enumerate(flights):
        ax, ay = flight["start"]
        bx, by = flight["end"]
        if number % 2:
            ax, ay, bx, by = bx, by, ax, ay
        length = math.hypot(bx - ax, by - ay)
        treads = flight["treads"]
        run = length / max(treads - 1, 1)
        ux, uy = ((bx - ax) / length, (by - ay) / length) if length else (1.0, 0.0)
        angle = math.atan2(uy, ux)
        for k in range(treads):
            top = base + rise * (k + 1)
            boxes.append(StairBox(
                (ax + ux * run * k, ay + uy * run * k), angle, run, flight["width"],
                base, top,  # 단이 떠 보이지 않게 그 단 묶음의 바닥부터 채운다
            ))
        base += rise * (treads + 1)
        half = flight["width"] / 2
        nx, ny = -uy * half, ux * half
        footprints.append(Polygon([
            (ax - ux * run / 2 + nx, ay - uy * run / 2 + ny), (bx + ux * run / 2 + nx, by + uy * run / 2 + ny),
            (bx + ux * run / 2 - nx, by + uy * run / 2 - ny), (ax - ux * run / 2 - nx, ay - uy * run / 2 - ny),
        ]))
        if number < len(flights) - 1:
            whole = box(*stair["bounds"])
            rest = whole.difference(unary_union(footprints + [
                Polygon([
                    tuple(f["start"]), tuple(f["end"]),
                    (f["end"][0] + 1.0, f["end"][1] + 1.0), (f["start"][0] + 1.0, f["start"][1] + 1.0),
                ]).buffer(f["width"] / 2, join_style="mitre") for f in flights
            ]))
            pieces = _polygons(rest, 0.25e6)
            if pieces:
                landings.append((max(pieces, key=lambda p: p.area), base - cfg["landing_thickness_mm"], base))
    return boxes, landings


def plan_floors(storeys: list[Any], cfg: dict[str, Any]) -> None:
    """층마다 슬래브, 노출된 지붕, 계단을 정해 storey.floor에 넣는다. 좌표는 그 층의 IR 좌표다."""
    from .quality import building_only

    ordered = sorted(storeys, key=lambda s: s.elevation)
    outlines = {}
    bodies = {}
    for s in ordered:
        bodies[id(s)] = building_only(s.part.ir)  # 건물 밖 벽이 슬래브 돌기가 되지 않게 한다
        outlines[id(s)] = floor_outlines(bodies[id(s)], cfg["slab_close_mm"], cfg["slab_min_area_m2"], cfg["slab_trim_mm"])

    def world(s: Any) -> tuple[float, float]:
        ox, oy = s.part.ir["origin_offset"]
        return ox + s.shift[0], oy + s.shift[1]

    for index, s in enumerate(ordered):
        plan = FloorPlan()
        ir = s.part.ir
        holes = [box(*v["bounds"]) for v in ir.get("voids", [])]
        if index > 0:  # 맨 아래층은 아래로 내려갈 곳이 없다
            holes += [box(*stair["bounds"]) for stair in bodies[id(s)].get("stairs", [])]
        hole = unary_union(holes) if holes else None
        for outline in outlines[id(s)]:
            shape = outline.difference(hole) if hole is not None else outline
            plan.slabs.extend(_polygons(shape, cfg["slab_min_area_m2"] * 1e6))

        # 지붕은 위층에 덮이지 않은 부분이다. 위층 위치를 모르면 덮이지 않은 것으로 본다.
        above = ordered[index + 1] if index + 1 < len(ordered) else None
        offset = world(s)
        cover = None
        if above is not None and s.align != "unaligned" and above.align != "unaligned":
            ax, ay = world(above)
            shifted = [translate(p, ax - offset[0], ay - offset[1]) for p in outlines[id(above)]]
            cover = unary_union(shifted) if shifted else None
        for outline in outlines[id(s)]:
            shape = outline.difference(cover) if cover is not None else outline
            site = getattr(s, "site", None)
            if site is not None:  # 천창 자리는 지붕을 뚫는다
                for skylight in site.roof_holes:
                    shape = shape.difference(translate(skylight, -offset[0], -offset[1]))
            plan.roofs.extend(_polygons(shape, cfg["slab_min_area_m2"] * 1e6))

        if above is not None:  # 맨 위층 계단은 올라갈 곳이 없다
            for stair in ir.get("stairs", []):
                boxes, landings = _stair_boxes(stair, s.height, s.elevation, cfg)
                plan.stairs.append((stair["id"], boxes, landings))
        s.floor = plan
