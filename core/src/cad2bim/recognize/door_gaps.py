"""문 자리 메우기. 벽선이 문 앞에서 끊기고 문틀만 그린 자리를 벽으로 잇는다.

문 호가 벽에 걸리지 않아 버려지는 흔한 경우다. 벽 끝 바로 옆에서 같은 벽 축에 문짝(호의 닫힌 쪽)이
놓이고, 반대쪽 문짝 끝이 문틀·벽·기둥에 닿으면 그 사이를 같은 두께의 벽으로 채운다. 그러면 개구부
인식이 그 벽에서 문을 찾는다. 메우지 않으면 문도 벽도 없는, 바닥부터 천장까지 뚫린 틈이 남는다.
도면에 벽선이 없는 자리를 만드는 추정이므로 항상 검토 대상(wall_from_door)이다.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from ..layers import LayerMap
from ..preprocess.primitive import Primitive
from .columns import ColumnDraft
from .door_walls import _overlaps_existing, _unhosted_arcs
from .openings import _wall_tree
from .walls import WallDraft


def _leaves(arc: Primitive, walls: list[WallDraft], tree: Any, cfg: dict[str, Any], reach: float) -> list[tuple[int, str, float, float]]:
    """호의 닫힌 문짝이 놓일 수 있는 (벽 번호, 벽의 어느 끝 바깥인가, 벽 축 위 구간)."""
    center = np.array(arc.geometry["center"], dtype=float)
    radius = float(arc.geometry["radius"])
    legs = [np.array(arc.geometry[k], dtype=float) - center for k in ("start", "end")]
    tolerance = cfg["door_layer_tolerance_mm"]
    span = 2 * radius + reach + tolerance + max(w.thickness for w in walls)
    found = []
    for index in tree.query(shapely.box(center[0] - span, center[1] - span, center[0] + span, center[1] + span)).tolist():
        wall = walls[index]
        u = wall.direction
        n = np.array([-u[1], u[0]])
        if abs(float((center - wall.start) @ n)) > wall.thickness / 2 + tolerance:
            continue
        for closed in legs:
            if abs(float(closed @ u)) < radius * cfg["parallel_cos"]:
                continue  # 닫힌 문짝은 벽 방향이다
            s = float((center - wall.start) @ u)
            lo, hi = sorted((s, s + float(closed @ u)))
            if hi <= 0.0:
                found.append((index, "start", lo, hi))
            elif lo >= wall.length:
                found.append((index, "end", lo, hi))
    return found


def walls_across_door_gaps(
    prims: list[Primitive], layers: LayerMap, walls: list[WallDraft], columns: list[ColumnDraft], cfg: dict[str, Any],
) -> list[WallDraft]:
    walls_cfg, openings_cfg = cfg["walls"], cfg["openings"]
    if not walls:
        return []
    arcs = _unhosted_arcs(prims, layers, walls, openings_cfg)
    if not arcs:
        return []
    tree = _wall_tree(walls)
    reach = walls_cfg["door_gap_reach_mm"]
    join = openings_cfg["host_tolerance_mm"]

    # 같은 벽 끝 바깥의 문짝을 모아 맞닿는 것끼리 잇는다. 양개 문은 두 문짝이 한 구간이 된다.
    by_end: dict[tuple[int, str], list[tuple[float, float, int]]] = {}
    for number, arc in enumerate(arcs):
        for index, side, lo, hi in _leaves(arc, walls, tree, openings_cfg, reach):
            by_end.setdefault((index, side), []).append((lo, hi, number))
    candidates = []
    for (index, side), leaves in by_end.items():
        leaves.sort()
        runs: list[list[Any]] = []
        for lo, hi, number in leaves:
            if runs and lo <= runs[-1][1] + join:
                runs[-1][1] = max(runs[-1][1], hi)
                runs[-1][2].add(number)
            else:
                runs.append([lo, hi, {number}])
        length = walls[index].length
        for lo, hi, numbers in runs:
            gap = lo - length if side == "end" else -hi
            if gap <= reach:
                candidates.append((gap, index, side, lo, hi, numbers))

    solids = [LineString([w.start, w.end]).buffer(w.thickness / 2, cap_style="square") for w in walls]
    shapes = solids + [c.polygon for c in columns]
    shape_tree = shapely.STRtree(shapes)
    used: set[int] = set()
    new_walls: list[WallDraft] = []
    for gap, index, side, lo, hi, numbers in sorted(candidates, key=lambda c: c[0]):
        if numbers & used:
            continue
        wall = walls[index]
        u = wall.direction
        near, far = (wall.length, hi) if side == "end" else (0.0, lo)
        start, end = wall.start + u * near, wall.start + u * far
        # 반대쪽 문짝 끝이 문틀·벽·기둥에 닿아야 한다. 허공으로 열린 문을 벽으로 막지 않는다.
        tip = Point(end)
        distances = [
            shapes[j].distance(tip)
            for j in shape_tree.query(tip.buffer(walls_cfg["door_gap_anchor_mm"])).tolist()
            if j != index
        ]
        distances = [d for d in distances if d <= walls_cfg["door_gap_anchor_mm"]]
        if not distances:
            continue
        # 닿을 듯 떨어진 끝은 마저 늘려 붙인다. 남겨 두면 그만큼 틈이 보인다.
        outward = u if side == "end" else -u
        end = end + outward * min(distances)
        if side == "start":
            start, end = end, start
        draft = WallDraft(
            np.array(start, dtype=float), np.array(end, dtype=float), wall.thickness, "unknown",
            {h for k in numbers for h in arcs[k].source_handles}, {arcs[k].layer for k in numbers}, 0.0,
            from_door=True,
        )
        if draft.length < walls_cfg["min_wall_length_mm"] or _overlaps_existing(draft, walls + new_walls):
            continue
        used |= numbers
        new_walls.append(draft)
    return new_walls
