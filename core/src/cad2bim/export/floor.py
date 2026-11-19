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


def wall_height(wall: dict[str, Any], height: float, cfg: dict[str, Any]) -> float:
    """벽 하나의 높이. top_mm가 있으면 그 높이에서 멈춘다(부분층 아래 벽: 그 위는 부분층 벽이 잇는다)."""
    top = wall_top(height, cfg)
    return min(top, float(wall["top_mm"])) if wall.get("top_mm") else top


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
    rails: list[dict[str, Any]] = field(default_factory=list)  # 바닥 위에 선 부분만 남긴 난간 {id, start, end}


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
    try:
        merged = unary_union(shapes)
    except Exception:  # GEOS 위상 오류(아주 작은 좌표 어긋남). 1mm 격자로 맞춰 다시 합친다
        import shapely

        merged = shapely.union_all([shapely.set_precision(s.buffer(0), 1.0) for s in shapes])
    closed = merged.buffer(close_mm, join_style="mitre").buffer(-close_mm, join_style="mitre")
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


def _point_segment(p: Any, a: Any, b: Any) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    length2 = dx * dx + dy * dy
    k = 0.0 if not length2 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return math.hypot(p[0] - ax - dx * k, p[1] - ay - dy * k)


def _u_turn(stair: dict[str, Any], flights: list[dict[str, Any]]) -> tuple[tuple[float, float], float, float, float, float] | None:
    """나란한 단 묶음이 계단 범위 끝에서 끝까지 차 있으면 (방향, 축 위 시작·끝, 폭 방향 아래·위). 아니면 None.

    이런 도면은 계단참을 계단 범위 밖에 그린다. 범위 안에서 남는 곳을 찾으면 계단참이 빠진다.
    """
    (ax, ay), (bx, by) = flights[0]["start"], flights[0]["end"]
    length = math.hypot(bx - ax, by - ay)
    if len(flights) < 2 or not length:
        return None
    u = ((bx - ax) / length, (by - ay) / length)
    n = (-u[1], u[0])
    along, across = [], []
    for f in flights:
        (fx, fy), (gx, gy) = f["start"], f["end"]
        flen = math.hypot(gx - fx, gy - fy)
        if not flen or abs(((gx - fx) * u[0] + (gy - fy) * u[1]) / flen) < 0.99:
            return None  # 꺾인 계단
        along += [fx * u[0] + fy * u[1], gx * u[0] + gy * u[1]]
        half = f["width"] / 2
        middle = fx * n[0] + fy * n[1]
        across += [middle - half, middle + half]
    x0, y0, x1, y1 = stair["bounds"]
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    span = [c[0] * u[0] + c[1] * u[1] for c in corners]
    if max(along) - min(along) < 0.9 * (max(span) - min(span)):
        return None  # 범위 안에 계단참 자리가 있다
    return u, min(along), max(along), min(across), max(across)


def _strip(u: tuple[float, float], s0: float, s1: float, t0: float, t1: float) -> Polygon:
    """축 u 방향 [s0, s1], 폭 방향 [t0, t1]인 사각형."""
    n = (-u[1], u[0])
    return Polygon([
        (u[0] * s + n[0] * t, u[1] * s + n[1] * t) for s, t in ((s0, t0), (s1, t0), (s1, t1), (s0, t1))
    ])


def _clear_depth(walls: Any, u: tuple[float, float], end: float, sign: float, t0: float, t1: float, reach: float) -> float | None:
    """계단 끝에서 바깥으로 벽까지 거리. reach 안에 벽이 없으면 None."""
    if walls is None:
        return None
    lo, hi = (end, end + reach) if sign > 0 else (end - reach, end)
    hit = walls.intersection(_strip(u, lo, hi, t0 + 50.0, t1 - 50.0))
    if hit.is_empty:
        return None
    coords = [c for g in getattr(hit, "geoms", [hit]) for c in (g.exterior.coords if hasattr(g, "exterior") else g.coords)]
    return min(abs(x * u[0] + y * u[1] - end) for x, y in coords)


def _stair_boxes(
    stair: dict[str, Any], height: float, elevation: float, cfg: dict[str, Any],
    walls: Any = None, floor_above: Any = None,
) -> tuple[list[StairBox], list[tuple[Polygon, float, float]]]:
    """오르는 방향은 평면에서 알 수 없어 단 묶음을 번갈아 오르는 것으로 가정한다.

    두 단 묶음이 나란히 계단 범위를 다 채우면(ㄷ자 계단) 계단참은 범위 밖, 꺾이는 쪽 끝에 둔다.
    오르기 시작하는 끝은 윗층 바닥이 이어지는 쪽이다. 반대쪽 끝에 닿으면 허공에 내린다.
    walls와 floor_above는 이 층 좌표의 벽 입체와 윗층 바닥(없으면 None)이다.
    """
    # 방향을 맞추며 고치므로 복사한다. IR은 다른 출력도 읽는다.
    flights = [dict(f) for f in sorted(stair["flights"], key=lambda f: (f["start"][1], f["start"][0]))]
    up = stair.get("up_at")
    if up and flights:
        # 도면의 UP 글자에 가장 가까운 단 묶음이 처음 오르는 묶음이고, 글자 쪽 끝에서 오르기 시작한다.
        first = min(range(len(flights)), key=lambda i: _point_segment(up, flights[i]["start"], flights[i]["end"]))
        flights.insert(0, flights.pop(first))
        if math.dist(up, flights[0]["end"]) < math.dist(up, flights[0]["start"]):
            flights[0]["start"], flights[0]["end"] = flights[0]["end"], flights[0]["start"]
    total = sum(f["treads"] for f in flights) + len(flights)
    # 평면은 한 바퀴만 그린다. 층고가 높아 단 높이가 한도를 넘으면 같은 계단을 위로 되풀이한다(물리 제약).
    max_riser = cfg.get("stair_max_riser_mm")
    cycles = max(1, math.ceil(height / (total * max_riser))) if max_riser and height / total > max_riser else 1
    rise = height / (total * cycles)
    waist = cfg["landing_thickness_mm"]
    boxes: list[StairBox] = []
    landings: list[tuple[Polygon, float, float]] = []
    turn = _u_turn(stair, flights)
    landing_strip = None
    if turn is not None:
        u, s0, s1, t0, t1 = turn
        # 첫 단 묶음은 도면대로 s0 → s1로 오른다. 윗층 바닥이 s1 쪽에만 있으면 뒤집는다.
        if floor_above is not None and not up:  # 도면에 UP 표시가 없을 때만 윗층 바닥으로 추측한다
            probe = cfg["stair_access_probe_mm"]

            def landed(strip: Polygon) -> bool:
                # 보이드 가장자리의 가는 바닥 조각은 내려설 곳이 아니다. 절반 넘게 바닥이어야 한다.
                return floor_above.intersection(strip).area >= strip.area * cfg["stair_access_min_share"]

            at_start = landed(_strip(u, s0 - probe, s0 - 1.0, t0, t1))
            at_end = landed(_strip(u, s1 + 1.0, s1 + probe, t0, t1))
            if at_end and not at_start:
                u, s0, s1, t0, t1 = (-u[0], -u[1]), -s1, -s0, -t1, -t0
        depth = _clear_depth(walls, u, s1, 1.0, t0, t1, cfg["stair_landing_max_mm"])
        depth = min(max(depth if depth is not None else flights[0]["width"], cfg["stair_landing_min_mm"]), cfg["stair_landing_max_mm"])
        landing_strip = _strip(u, s1, s1 + depth, t0, t1)
        start_strip = _strip(u, s0 - depth, s0, t0, t1)  # 되풀이할 때 바퀴 사이 계단참
        for flight in flights:  # 모든 단 묶음을 같은 축 방향(s0 → s1)으로 맞춘 뒤 번갈아 뒤집는다
            (fx, fy), (gx, gy) = flight["start"], flight["end"]
            if (gx - fx) * u[0] + (gy - fy) * u[1] < 0:
                flight["start"], flight["end"] = flight["end"], flight["start"]
    base = elevation
    footprints = []
    for cycle in range(cycles):
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
                    # 한 단 높이에 계단 두께만큼 아래로. 이어 붙이면 경사진 계단판이 된다.
                    # 단 묶음 바닥까지 채우면 두 번째 묶음이 계단참 위에 벽처럼 선다.
                    max(base, top - rise - waist), top,
                ))
            base += rise * (treads + 1)
            if landing_strip is not None:
                if number < len(flights) - 1:
                    landings.append((landing_strip, base - waist, base))
                elif cycle < cycles - 1:
                    landings.append((start_strip, base - waist, base))
                continue
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
            # 바닥은 아래층 계단이 올라오는 자리에 뚫는다. 이 층 도면의 계단 자리에 뚫으면, 층마다 계단
            # 위치가 다른 도면에서 아래층 방으로 떨어지는 구멍과 천장에 막힌 계단이 생긴다.
            below = ordered[index - 1]
            if below.align != "unaligned" and s.align != "unaligned":
                bx, by = world(below)
                sx, sy = world(s)
                holes += [
                    translate(box(*stair["bounds"]), bx - sx, by - sy) for stair in bodies[id(below)].get("stairs", [])
                ]
            else:  # 층 위치를 모르면 예전처럼 이 층의 계단 자리를 쓴다
                holes += [box(*stair["bounds"]) for stair in bodies[id(s)].get("stairs", [])]
            # 아래층에서 이어지는 승강로는 바닥을 뚫는다. 맨 아래 칸의 바닥(피트)은 남는다.
            from .elevator import continues_below

            holes += [shaft for shaft in getattr(s, "elevators", []) if continues_below(s, below, shaft)]
        hole = unary_union(holes) if holes else None
        for outline in outlines[id(s)]:
            shape = outline.difference(hole) if hole is not None else outline
            plan.slabs.extend(_polygons(shape, cfg["slab_min_area_m2"] * 1e6))

        # 지붕은 위층에 덮이지 않은 부분이다. 위층 위치를 모르면 덮이지 않은 것으로 본다.
        above = ordered[index + 1] if index + 1 < len(ordered) else None
        offset = world(s)
        cover = None
        if s.key == "R":  # 지붕층은 하늘로 열려 있다. 바닥판이 곧 지붕이고 그 위는 덮지 않는다
            cover = unary_union(outlines[id(s)]) if outlines[id(s)] else None
        elif above is not None and s.align != "unaligned" and above.align != "unaligned":
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
            walls = unary_union([
                LineString([w["start"], w["end"]]).buffer(w["thickness"] / 2, cap_style="flat")
                for w in ir["walls"] if w["start"] != w["end"]
            ]) if ir["walls"] else None
            floor_above = None
            if s.align != "unaligned" and above.align != "unaligned":
                ax, ay = world(above)
                above_voids = [box(*v["bounds"]) for v in above.part.ir.get("voids", [])]
                shapes = [translate(p, ax - offset[0], ay - offset[1]) for p in outlines[id(above)]]
                if shapes:
                    floor_above = unary_union(shapes)
                    if above_voids:
                        floor_above = floor_above.difference(
                            unary_union([translate(v, ax - offset[0], ay - offset[1]) for v in above_voids])
                        )
            for stair in ir.get("stairs", []):
                boxes, landings = _stair_boxes(stair, s.height, s.elevation, cfg, walls, floor_above)
                plan.stairs.append((stair["id"], boxes, landings))

        # 난간은 도면에 그린 그대로 둔다. 층마다 도면이 맞지 않아 난간이 아래층 계단 구멍을 가로질러도
        # 지우지 않는다(도면을 그대로 옮기는 것이 목표다. 맞지 않는 곳은 고치지 않고 알린다).
        plan.rails = [dict(r) for r in ir.get("rails", [])]
        s.floor = plan
