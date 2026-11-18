"""문 호를 근거로 놓친 벽 찾기.

문 호가 있는데 벽이 없으면 그 자리의 벽을 놓쳤을 가능성이 크다. 실제 평면에서 문 호의 40%가
그랬다. 도면 전체의 미분류 선을 벽으로 올리면 오인이 늘어나므로, 호 옆에서 문이 닫히는 방향과
나란한 평행선 쌍과 그 축으로 이어진 조각만 벽으로 올린다. 추정이므로 항상 검토 대상이다.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import shapely

from ..layers import LayerInfo, LayerMap
from ..preprocess.primitive import Primitive
from .columns import ColumnDraft
from .hatches import HatchIndex
from .openings import _OPENING_ROLES, _door_from_arc, _has_door_shape, _wall_tree
from .walls import (
    Piece,
    WallDraft,
    _annotate_hatch,
    _axis_groups,
    _merge_axis,
    collect_segments,
    find_pieces,
    group_by_direction,
)

_PARALLEL_COS = 0.999


def _unhosted_arcs(prims: list[Primitive], layers: LayerMap, walls: list[WallDraft], cfg: dict[str, Any]) -> list[Primitive]:
    tree = _wall_tree(walls)
    margin = max((w.thickness for w in walls), default=0.0) / 2 + cfg["door_layer_tolerance_mm"]
    found = []
    for p in prims:
        if p.kind != "arc" or layers.role(p.layer) not in _OPENING_ROLES or not _has_door_shape(p, cfg):
            continue
        if not walls or _door_from_arc(p, walls, tree, cfg, margin, cfg["door_layer_tolerance_mm"]) is None:
            found.append(p)
    return found


def _candidate_pieces(
    prims: list[Primitive], layers: LayerMap, columns: list[ColumnDraft], hatches: HatchIndex, walls_cfg: dict[str, Any]
) -> list[Piece]:
    """미분류 레이어의 벽 두께 평행선 쌍."""
    view = LayerMap({
        name: info if info.role == "UNKNOWN" else LayerInfo("OTHER", info.source)
        for name, info in layers.layers.items()
    })
    segments = collect_segments(prims, view, columns, walls_cfg)
    tolerance = math.radians(walls_cfg["direction_tolerance_deg"])
    pieces: list[Piece] = []
    for group in group_by_direction(segments, tolerance):
        pieces.extend(find_pieces(group, walls_cfg, True))
    pieces = [p for p in pieces if p.thickness <= walls_cfg["door_evidence_max_thickness_mm"]]
    _annotate_hatch(pieces, hatches, walls_cfg)
    return pieces


def _piece_supports_arc(piece: Piece, arc: Primitive, cfg: dict[str, Any], reach: float) -> bool:
    """호의 닫힌 쪽 끝이 조각의 축 방향이고, 호 중심이 조각 옆에 있는가."""
    center = np.array(arc.geometry["center"], dtype=float)
    radius = float(arc.geometry["radius"])
    u = np.array(piece.u, dtype=float)
    n = np.array([-u[1], u[0]])
    start = np.array(arc.geometry["start"], dtype=float) - center
    end = np.array(arc.geometry["end"], dtype=float) - center
    tol = radius * cfg["axis_tolerance"]
    aligned = any(
        abs(float(closed @ n)) <= tol and abs(float(opened @ u)) <= tol
        for closed, opened in ((start, end), (end, start))
    )
    if not aligned:
        return False
    if abs(float(center @ n) - piece.offset) > piece.thickness / 2 + cfg["door_layer_tolerance_mm"]:
        return False
    along = float(center @ u)
    gap = max(piece.s0 - along, along - piece.s1, 0.0)
    return gap <= radius + reach


def _overlaps_existing(run: WallDraft, walls: list[WallDraft]) -> bool:
    for wall in walls:
        if abs(float(wall.direction @ run.direction)) < _PARALLEL_COS:
            continue
        n = np.array([-wall.direction[1], wall.direction[0]])
        if abs(float((run.start - wall.start) @ n)) > (wall.thickness + run.thickness) / 2:
            continue
        lo = sorted((float((run.start - wall.start) @ wall.direction), float((run.end - wall.start) @ wall.direction)))
        if lo[1] > 0 and lo[0] < wall.length:
            return True
    return False


def walls_from_door_evidence(
    prims: list[Primitive], layers: LayerMap, walls: list[WallDraft], columns: list[ColumnDraft],
    hatches: HatchIndex, cfg: dict[str, Any],
) -> tuple[list[WallDraft], list[Piece]]:
    """(새 벽, 그 벽의 조각). 벽 레이어를 하나도 못 찾은 도면은 이미 전체가 후보라 다루지 않는다."""
    walls_cfg, openings_cfg = cfg["walls"], cfg["openings"]
    if not layers.has_walls:
        return [], []
    arcs = _unhosted_arcs(prims, layers, walls, openings_cfg)
    if not arcs:
        return [], []
    pieces = _candidate_pieces(prims, layers, columns, hatches, walls_cfg)
    if not pieces:
        return [], []

    reach = walls_cfg["door_evidence_reach_mm"]
    tree = shapely.STRtree(shapely.linestrings([[p.point(p.s0), p.point(p.s1)] for p in pieces]))
    supported: set[int] = set()
    for arc in arcs:
        x, y = arc.geometry["center"]
        span = float(arc.geometry["radius"]) + reach + openings_cfg["door_layer_tolerance_mm"] + walls_cfg["door_evidence_max_thickness_mm"]
        for index in tree.query(shapely.box(x - span, y - span, x + span, y + span)).tolist():
            if _piece_supports_arc(pieces[index], arc, openings_cfg, reach):
                supported.add(index)
    if not supported:
        return [], []

    chosen_ids = {id(pieces[i]) for i in supported}
    new_walls: list[WallDraft] = []
    used: list[Piece] = []
    for axis in _axis_groups([(p, True) for p in pieces], walls_cfg):
        if not any(id(p) in chosen_ids for p, _ in axis):
            continue
        ordered = sorted(axis, key=lambda item: item[0].s0)
        # 근거 조각과 이어진 구간만 쓴다. 같은 축의 먼 조각까지 벽으로 만들지 않는다.
        runs: list[list[tuple[Piece, bool]]] = [[ordered[0]]]
        stop = ordered[0][0].s1
        for item in ordered[1:]:
            if item[0].s0 - stop <= walls_cfg["max_bridge_mm"]:
                runs[-1].append(item)
                stop = max(stop, item[0].s1)
            else:
                runs.append([item])
                stop = item[0].s1
        for run in runs:
            if not any(id(p) in chosen_ids for p, _ in run):
                continue
            for wall in _merge_axis(run, layers, walls_cfg):
                if wall.length >= walls_cfg["min_wall_length_mm"] and not _overlaps_existing(wall, walls + new_walls):
                    wall.from_door = True
                    new_walls.append(wall)
                    used.extend(p for p, _ in run)
    return new_walls, used
