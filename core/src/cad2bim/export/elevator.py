"""엘리베이터. 승강로 표시(닫힌 사각형)와 그 안의 'EV'·'엘리베이터' 글자로 찾는다.

평면의 엘리베이터는 보통 벽으로 둘러싼 칸 안에 X를 친 사각형과 글자로 그린다. 문은 슬라이딩이라 호가 없다.
- 위아래 층에 같은 자리의 엘리베이터가 있으면 그 사이 바닥을 뚫어 승강로로 잇는다(맨 아래층 바닥은 피트로 남긴다).
- 앱에는 승강로 모양, 서는 층, 층마다 문 위치를 넘겨 타고 오르내리게 한다.
층마다 위치가 다르면(도면이 서로 맞지 않으면) 잇지 않고 그 층에만 둔다.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from shapely.affinity import translate
from shapely.geometry import LineString, Point, Polygon

from ..preprocess import load_primitives
from ..preprocess.primitive import Primitive

_NOT_SHAFT_ROLES = ("WALL", "COLUMN", "DOOR", "WINDOW")


def _label_pattern(cfg: dict[str, Any]) -> re.Pattern[str]:
    return re.compile(cfg["elevator_label_pattern"], re.IGNORECASE)


def find_elevators(prims: list[Primitive], roles: dict[str, str], cfg: dict[str, Any]) -> list[Polygon]:
    """IR 좌표 요소에서 승강로 바닥 모양들. 같은 글자에 걸린 도형이 여럿이면 가장 작은 것(카 표시)을 쓴다."""
    pattern = _label_pattern(cfg)
    labels = [Point(p.geometry["position"]) for p in prims if p.kind == "text" and pattern.search(p.geometry["text"].strip())]
    if not labels:
        return []
    candidates = []
    for p in prims:
        if p.kind != "polyline" or not p.geometry["closed"] or roles.get(p.layer) in _NOT_SHAFT_ROLES:
            continue
        points = list(p.geometry["points"])
        if len(points) < 4:
            continue
        polygon = Polygon(points)
        if not polygon.is_valid or polygon.area <= 0:
            continue
        ring = list(polygon.minimum_rotated_rectangle.exterior.coords)
        sides = sorted((math.dist(ring[0], ring[1]), math.dist(ring[1], ring[2])))
        if sides[0] < cfg["elevator_min_side_mm"] or sides[1] > cfg["elevator_max_side_mm"]:
            continue
        if polygon.area < polygon.minimum_rotated_rectangle.area * 0.9:
            continue  # 승강로는 사각형이다
        candidates.append(polygon)
    found: list[Polygon] = []
    for label in labels:
        holding = [c for c in candidates if c.buffer(cfg["elevator_label_reach_mm"]).contains(label)]
        if not holding:
            continue
        best = min(holding, key=lambda c: c.area)
        if not any(best.intersection(f).area >= 0.5 * min(best.area, f.area) for f in found):
            found.append(best)
    return found


def attach_elevators(storeys: list[Any], drawings: dict[str, Path], out_dir: Path, cfg: dict[str, Any]) -> int:
    """층마다 storey.elevators(IR 좌표 다각형 목록)를 채운다. 찾은 수를 돌려준다."""
    import json

    total = 0
    for s in storeys:
        s.elevators = []
        drawing = drawings.get(s.part.part_id)
        layer_map = out_dir / s.part.part_id / "layer_map.json"
        if drawing is None or not layer_map.is_file():
            continue
        roles = {name: info["role"] for name, info in json.loads(layer_map.read_text(encoding="utf-8"))["layers"].items()}
        ir = s.part.ir
        scale = float(ir["source"].get("unit_scale") or 1.0)
        ox, oy = ir["origin_offset"]
        raw, _ = load_primitives(drawing, cfg, out_dir / "_prims")
        local = [p.transformed(scale, -ox, -oy) for p in raw]
        walls = [(w["start"], w["end"]) for w in ir["walls"]]
        if not walls:
            continue
        xs = [c[0] for w in walls for c in w]
        ys = [c[1] for w in walls for c in w]
        inside = Polygon([(min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys))])
        # 같은 도면에 다른 평면이 있으면 그 평면의 엘리베이터는 뺀다.
        s.elevators = [e for e in find_elevators(local, roles, cfg["export"]) if inside.contains(e.centroid)]
        total += len(s.elevators)
    return total


def _world(s: Any) -> tuple[float, float]:
    ox, oy = s.part.ir["origin_offset"]
    return ox + s.shift[0], oy + s.shift[1]


def continues_below(s: Any, below: Any, shaft: Polygon) -> bool:
    """이 층의 승강로(이 층 IR 좌표)가 아래층 승강로와 같은 자리인가."""
    if below is None or s.align == "unaligned" or below.align == "unaligned":
        return False
    (sx, sy), (bx, by) = _world(s), _world(below)
    here = translate(shaft, sx, sy)
    return any(
        translate(other, bx, by).intersection(here).area >= 0.5 * min(here.area, other.area)
        for other in getattr(below, "elevators", [])
    )


def _door(ir: dict[str, Any], shaft: Polygon, cfg: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """승강로 벽에 난 개구부(문). 호가 없어 종류를 모르는 틈으로 남은 경우가 많다. IR 좌표 (x1, y1, x2, y2)."""
    walls = {w["id"]: w for w in ir["walls"]}
    best = None
    for o in ir["openings"]:
        w = walls.get(o["host_wall"])
        if w is None:
            continue
        (x0, y0), (x1, y1) = w["start"], w["end"]
        length = math.hypot(x1 - x0, y1 - y0)
        if not length:
            continue
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        a = (x0 + ux * (o["position"] - o["width"] / 2), y0 + uy * (o["position"] - o["width"] / 2))
        b = (x0 + ux * (o["position"] + o["width"] / 2), y0 + uy * (o["position"] + o["width"] / 2))
        d = shaft.distance(LineString([a, b]))
        if d <= cfg["elevator_door_reach_mm"] and (best is None or d < best[0]):
            best = (d, (*a, *b))
    return best[1] if best else None


def elevator_data(storeys: list[Any], base: tuple[float, float], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """앱에 넘길 승강로 목록. 좌표는 building.dae와 같다(mm). 아래층부터 이어진 층을 한 승강로로 묶는다."""
    ordered = sorted(storeys, key=lambda s: s.elevation)
    shafts: list[dict[str, Any]] = []
    open_shafts: dict[int, dict[str, Any]] = {}  # 아래층 승강로 id → 묶음
    for index, s in enumerate(ordered):
        below = ordered[index - 1] if index > 0 else None
        ox, oy = s.part.ir["origin_offset"]
        dx, dy = ox + s.shift[0] - base[0], oy + s.shift[1] - base[1]
        next_open: dict[int, dict[str, Any]] = {}
        for shaft in getattr(s, "elevators", []):
            group = None
            if below is not None and continues_below(s, below, shaft):
                (sx, sy), (bx, by) = _world(s), _world(below)
                here = translate(shaft, sx, sy)
                for key, g in open_shafts.items():
                    other = g["_last"]
                    if translate(other, bx, by).intersection(here).area >= 0.5 * min(here.area, other.area):
                        group = g
                        break
            if group is None:
                group = {
                    "p": [[round(x + dx, 1), round(y + dy, 1)] for x, y in list(shaft.exterior.coords)[:-1]],
                    "floors": [], "levels": {}, "doors": {},
                }
                shafts.append(group)
            group["floors"].append(s.key)
            group["levels"][s.key] = round(s.elevation, 1)
            door = _door(s.part.ir, shaft, cfg)
            if door is not None:
                group["doors"][s.key] = [round(door[0] + dx, 1), round(door[1] + dy, 1), round(door[2] + dx, 1), round(door[3] + dy, 1)]
            group["_last"] = shaft
            next_open[id(shaft)] = group
        open_shafts = next_open
    for g in shafts:
        g.pop("_last", None)
    return shafts
