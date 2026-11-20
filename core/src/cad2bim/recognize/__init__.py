"""요소 인식. 기둥, 벽, 개구부, 실 순서로 찾는다."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..confidence import score_column, score_opening, score_room, score_wall
from ..ir import Column, Confidence, Flight, Opening, Rail, Room, Stair, Unsupported, Void, Wall
from ..layers import LayerMap
from ..preprocess.primitive import Primitive, faceted_door_arc
from ..timing import stage
from .columns import ColumnDraft, recognize_columns
from .hatches import HatchIndex
from .openings import OpeningDraft, recognize_openings
from .rooms import RoomDraft, recognize_rooms
from .candidate_walls import candidate_walls
from .door_gaps import walls_across_door_gaps
from .door_walls import walls_from_door_evidence
from .glazed import glazed_walls
from .piers import find_piers
from .stairs import recognize_stairs
from .voids import recognize_voids
from .walls import (
    WallResult,
    _ColumnIndex,
    _LineIndex,
    clean_junctions,
    bridge_across_columns,
    door_panel_walls,
    close_collinear_gaps,
    slit_fillers,
    extend_to_anchors,
    recognize_walls,
)


@dataclass
class Recognition:
    walls: list[Wall]
    columns: list[Column]
    unsupported: list[Unsupported]
    raw_walls: list[Wall]  # 잇기 전 조각. 단계별 평가용
    warnings: list[str]
    openings: list[Opening] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    door_stats: Counter = field(default_factory=Counter)
    stairs: list[Stair] = field(default_factory=list)
    voids: list[Void] = field(default_factory=list)
    rails: list[Rail] = field(default_factory=list)


def _rounded(point: Any) -> tuple[float, float]:
    """0.001mm 단위로 맞춘다. 계산 오차가 IR과 C# 쪽으로 넘어가지 않게 한다."""
    return (round(float(point[0]), 3) + 0.0, round(float(point[1]), 3) + 0.0)


def _to_wall(draft: Any, index: int, layers: LayerMap, has_hatches: bool, cfg: dict) -> Wall:
    return Wall(
        id=f"W-{index:04d}",
        start=_rounded(draft.start),
        end=_rounded(draft.end),
        thickness=draft.thickness,
        kind=draft.kind,
        confidence=score_wall(draft, layers, has_hatches, cfg),
        source_handles=sorted(draft.handles),
    )


def _to_column(draft: ColumnDraft, index: int, layers: LayerMap, cfg: dict) -> Column:
    return Column(
        id=f"C-{index:04d}",
        shape=draft.shape,
        center=draft.center,
        size=draft.size,
        rotation=draft.rotation,
        polygon=[_rounded(p) for p in list(draft.polygon.exterior.coords)[:-1]] if draft.shape == "polygon" else None,
        wall_piece=draft.shape == "polygon" and layers.role(draft.layer) == "WALL",
        confidence=score_column(draft, layers, cfg),
        source_handles=draft.handles,
    )


_OPENING_PREFIX = {"door": "D", "window": "WN", "unknown": "U"}


def _to_opening(draft: OpeningDraft, index: int, walls: list[Wall], cfg: dict) -> Opening:
    return Opening(
        id=f"{_OPENING_PREFIX[draft.type]}-{index:04d}",
        type=draft.type,
        operation=draft.operation,
        host_wall=walls[draft.wall].id,
        position=round(max(draft.position, 0.0), 3),
        width=round(draft.width, 3),
        hinge=draft.hinge,
        swing_dir=draft.swing_dir,
        confidence=score_opening(draft, cfg),
        source_handles=draft.handles,
    )


def _to_room(draft: RoomDraft, index: int, walls: list[Wall], cfg: dict) -> Room:
    return Room(
        id=f"R-{index:04d}",
        names=draft.names,
        point=draft.point,
        polygon=draft.polygon,
        clear_area_m2=draft.area_m2,
        boundary_wall_ids=[walls[i].id for i in draft.wall_indices],
        confidence=score_room(draft, cfg),
        source_handles=draft.handles,
    )


def _raw_walls(result: WallResult) -> list[Wall]:
    placeholder = Confidence(level="review", score=0.0)
    walls = []
    for i, p in enumerate(result.pieces, start=1):
        walls.append(Wall(
            id=f"P-{i:04d}", start=p.point(p.s0), end=p.point(p.s1), thickness=p.thickness,
            kind="unknown", confidence=placeholder,
        ))
    return walls


def _pair_rails(rails: list[tuple[Any, Any, Primitive]], cfg: dict[str, Any]) -> list[tuple[Any, Any, Primitive]]:
    """난간 하나를 양쪽 면 두 선으로 그린 경우 가운데 선 하나로 합친다.

    나란하고 가까운(pair_max_offset_mm 안) 두 선이 짧은 쪽 길이의 pair_min_overlap 이상 겹치면 같은 난간이다.
    따로 두면 난간 두 개가 겹쳐 서고 모서리에서 엇갈린다.
    """
    segments = [(np.array(a, dtype=float), np.array(b, dtype=float), p) for a, b, p in rails]
    used: set[int] = set()
    merged = []
    for i, (a, b, p) in enumerate(segments):
        if i in used:
            continue
        length = float(np.linalg.norm(b - a))
        u = (b - a) / length
        n = np.array([-u[1], u[0]])
        best = None
        for j in range(i + 1, len(segments)):
            if j in used:
                continue
            c, d, _ = segments[j]
            other = float(np.linalg.norm(d - c))
            if abs(float((d - c) @ u)) / other < cfg["pair_parallel_cos"]:
                continue
            offset = float((c - a) @ n)
            if not 1.0 <= abs(offset) <= cfg["pair_max_offset_mm"] or abs(float((d - a) @ n) - offset) > cfg["pair_offset_tol_mm"]:
                continue
            lo, hi = sorted((float((c - a) @ u), float((d - a) @ u)))
            if min(hi, length) - max(lo, 0.0) < cfg["pair_min_overlap"] * min(length, other):
                continue
            if best is None or abs(offset) < abs(best[1]):
                best = (j, offset, lo, hi)
        if best is None:
            merged.append((tuple(a), tuple(b), p))
            continue
        j, offset, lo, hi = best
        used.add(j)
        start = a + u * min(0.0, lo) + n * offset / 2
        end = a + u * max(length, hi) + n * offset / 2
        merged.append((tuple(start), tuple(end), p))
    return merged


_ARC_CENTER_TOL_MM = 5.0


def facet_wall_arcs(prims: list[Primitive], layers: LayerMap, step_deg: float) -> list[Primitive]:
    """벽 레이어의 호를 현(짧은 직선)으로 바꾼다. step_deg가 0이면 그대로 둔다(곡선 벽은 '미지원'으로 남는다).

    벽의 안쪽 호와 바깥쪽 호를 같은 각도에서 끊어야 현끼리 나란해져 벽 두 줄로 짝지어진다.
    그래서 호마다 등분하지 않고, 0도에서 step_deg씩 나눈 같은 각도 눈금에서 끊는다.
    같은 중심의 벽 호 사이에 낀 창호 레이어의 호(곡면 유리)도 함께 바꾼다. 문이 열리는 호는 중심이 달라 남는다.
    """
    if step_deg <= 0:
        return prims
    wall_arcs = [p for p in prims if p.kind == "arc" and not p.block_path and layers.role(p.layer) == "WALL"]
    if not wall_arcs:
        return prims

    def between_wall_arcs(p: Primitive) -> bool:
        cx, cy = p.geometry["center"]
        radii = [
            w.geometry["radius"] for w in wall_arcs
            if math.dist((cx, cy), w.geometry["center"]) <= _ARC_CENTER_TOL_MM
        ]
        return len(radii) >= 2 and min(radii) - 1.0 <= p.geometry["radius"] <= max(radii) + 1.0

    step = math.radians(step_deg)
    result: list[Primitive] = []
    for p in prims:
        curved = p.kind == "arc" and not p.block_path and (
            layers.role(p.layer) == "WALL" or (layers.role(p.layer) == "WINDOW" and between_wall_arcs(p))
        )
        if not curved:
            result.append(p)
            continue
        (cx, cy), radius = p.geometry["center"], p.geometry["radius"]
        start = math.atan2(p.geometry["start"][1] - cy, p.geometry["start"][0] - cx)
        sweep = (math.atan2(p.geometry["end"][1] - cy, p.geometry["end"][0] - cx) - start) % (2 * math.pi) or 2 * math.pi
        cuts = [0.0]
        tick = (math.floor(start / step) + 1) * step - start  # 시작점 다음의 첫 눈금까지
        while tick < sweep - 1e-9:
            if tick > 1e-9:
                cuts.append(tick)
            tick += step
        cuts.append(sweep)
        points = [(cx + radius * math.cos(start + t), cy + radius * math.sin(start + t)) for t in cuts]
        for k, (a, b) in enumerate(zip(points, points[1:])):
            result.append(replace(
                p, id=f"{p.id}~{k}", kind="line", geometry={"p1": a, "p2": b}, extra={**p.extra, "from_arc": True},
            ))
    return result


def recognize(
    prims: list[Primitive], layers: LayerMap, cfg: dict[str, Any], patterns: dict, walls_only: bool = False
) -> Recognition:
    """walls_only는 벽만 다시 찾는 자동 검증용이다."""
    # 벽 레이어의 곡선은 그대로 둔다. 곡선 벽이지 문이 아니다.
    prims = [
        (faceted_door_arc(p) or p) if layers.role(p.layer) != "WALL" else p
        for p in prims
    ]
    # 곡선 벽은 짧은 직선 벽 여러 개로 바꿔 직선 벽과 같은 방법으로 찾는다.
    wall_arcs = [p for p in prims if p.kind == "arc" and not p.block_path and layers.role(p.layer) == "WALL"]
    prims = facet_wall_arcs(prims, layers, cfg["walls"].get("arc_step_deg", 0.0))
    hatches = HatchIndex(prims, patterns)
    with stage("columns"):
        drafts = recognize_columns(prims, layers, hatches, cfg["columns"])
    with stage("walls"):
        result = recognize_walls(prims, layers, drafts, hatches, cfg["walls"])
        # 짝이 없어 벽이 되지 못한 호(선 한 줄로 그린 곡선 등)는 지금처럼 '미지원'으로 알린다.
        built = {h for w in result.walls for h in w.handles}
        result.curved = [p for p in wall_arcs if not built.intersection(p.source_handles)] + [
            p for p in result.curved if p not in wall_arcs
        ]

    if not walls_only:
        with stage("piers"):
            drafts = drafts + find_piers(prims, layers, result.walls, drafts, cfg["walls"])
        with stage("door_walls"):
            new_walls, new_pieces = walls_from_door_evidence(prims, layers, result.walls, drafts, hatches, cfg)
        with stage("glazed_walls"):
            curtain = glazed_walls(prims, layers, result.walls + new_walls, drafts, cfg["walls"])
        # 유리벽까지 찾은 뒤에 한다. 문 옆 벽이 유리벽인 경우가 많다.
        with stage("door_gaps"):
            new_walls = new_walls + walls_across_door_gaps(prims, layers, result.walls + new_walls + curtain, drafts, cfg)
        extra: list = []
        if cfg["walls"]["candidate_enabled"]:
            with stage("candidate_walls"):
                extra = candidate_walls(prims, layers, result.walls + new_walls + curtain, drafts, cfg["walls"])
        if new_walls or curtain or extra:
            result.walls.extend(new_walls)
            result.walls.extend(curtain)
            result.walls.extend(extra)
            result.pieces.extend(new_pieces)
            column_index = _ColumnIndex(drafts)
            extend_to_anchors(result.walls, column_index, _LineIndex(prims, layers), cfg["walls"])
            clean_junctions(result.walls, column_index, cfg["walls"])
            close_collinear_gaps(result.walls, cfg["walls"])
    if drafts:
        bridge_across_columns(result.walls, _ColumnIndex(drafts), cfg["walls"])
    panels = door_panel_walls(prims, layers, result.walls, cfg["walls"])
    result.walls.extend(panels)

    has_hatches = len(hatches) > 0
    ordered = sorted(result.walls, key=lambda w: tuple(np.round(w.start)))
    walls = [_to_wall(d, i, layers, has_hatches, cfg["confidence"]) for i, d in enumerate(ordered, start=1)]
    columns = [_to_column(d, i, layers, cfg["confidence"]) for i, d in enumerate(drafts, start=1)]
    unsupported = [
        Unsupported(kind="curved_wall", handles=list(p.source_handles), bbox=p.bbox())
        for p in result.curved
    ]
    warnings = [] if layers.has_walls else ["layer_unresolved"]
    recognition = Recognition(walls, columns, unsupported, _raw_walls(result), warnings)
    if walls_only:
        return recognition
    with stage("openings"):
        opening_drafts = recognize_openings(
            prims, layers, ordered, result.pieces, drafts, cfg["openings"], recognition.door_stats
        )
        # 유리 구간 전체가 창이다. 문이 아닌 개구부를 그 위에 또 뚫으면 구멍만 남는다.
        opening_drafts = [d for d in opening_drafts if not (ordered[d.wall].glazed and d.type != "door")]
        # 문 레이어 선으로만 그린 큰 문은 벽 전체가 문이다(미닫이로 둔다).
        index_of = {id(w): k for k, w in enumerate(ordered)}
        opening_drafts += [
            OpeningDraft("door", index_of[id(w)], w.length / 2, w.length, "sliding", handles=sorted(w.handles), role_known=True)
            for w in panels
        ]
    with stage("rooms"):
        room_drafts, open_regions = recognize_rooms(
            prims, layers, ordered, drafts, cfg["rooms"], cfg["walls"]["part_block_max_mm"]
        )
    recognition.openings = [_to_opening(d, i, walls, cfg["confidence"]) for i, d in enumerate(opening_drafts, start=1)]
    # 두 겹 벽 사이 틈 메우기는 개구부를 찾은 뒤에 한다. 메운 벽에는 그려진 선이 없어 틈으로 잘못 읽힌다.
    holes: dict[int, list[tuple[float, float]]] = {}
    for d in opening_drafts:
        holes.setdefault(d.wall, []).append((d.position - d.width / 2, d.position + d.width / 2))
    fillers = slit_fillers(ordered, cfg["walls"], holes)
    walls.extend(_to_wall(d, len(walls) + k, layers, has_hatches, cfg["confidence"]) for k, d in enumerate(fillers, start=1))
    recognition.rooms = [_to_room(d, i, walls, cfg["confidence"]) for i, d in enumerate(room_drafts, start=1)]
    # 평면에서 오르는 방향과 층 사이 연결은 확실하지 않아 근거와 상관없이 검토 대상이다.
    with stage("stairs"):
        recognition.stairs = [
            Stair(
                id=f"ST-{i:04d}", bounds=draft.bounds,
                flights=[Flight(start=f.start, end=f.end, width=f.width, treads=f.treads) for f in draft.flights],
                confidence=Confidence(level="review", score=0.5, reasons=["stair_direction_assumed"]),
                source_handles=sorted({h for f in draft.flights for h in f.handles}),
            )
            for i, draft in enumerate(recognize_stairs(prims, layers, cfg["stairs"]), start=1)
        ]
        rails = []
        for p in prims:
            if layers.role(p.layer) != "RAIL":
                continue
            if p.kind == "line":
                pairs = [(p.geometry["p1"], p.geometry["p2"])]
            elif p.kind == "polyline":
                points = list(p.geometry["points"]) + ([p.geometry["points"][0]] if p.geometry["closed"] else [])
                pairs = list(zip(points, points[1:]))
            else:
                continue
            rails.extend((a, b, p) for a, b in pairs if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 >= 300.0)
        rails = _pair_rails(rails, cfg["rails"])
        recognition.rails = [
            Rail(
                id=f"RL-{i:04d}", start=_rounded(a), end=_rounded(b),
                confidence=Confidence(level="high", score=0.8), source_handles=list(p.source_handles),
            )
            for i, (a, b, p) in enumerate(rails, start=1)
        ]
        recognition.voids = [
            Void(
                id=f"V-{i:04d}", bounds=draft.bounds,
                confidence=Confidence(level="review", score=0.7, reasons=["void_from_cross"]),
                source_handles=sorted(draft.handles),
            )
            for i, draft in enumerate(recognize_voids(prims, layers, cfg["voids"]), start=1)
        ]
    recognition.unsupported += [
        Unsupported(kind="open_room", handles=handles, bbox=bounds) for handles, bounds in open_regions
    ]
    return recognition
