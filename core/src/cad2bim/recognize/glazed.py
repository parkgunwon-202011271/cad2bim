"""유리벽. 받쳐 줄 벽 없이 창호 선만으로 그려진 구간을 벽으로 만든다.

개구부 틈으로 보기에는 길어서 벽이 끊기고 실이 닫히지 않는다. 양 끝이 벽이나 기둥에 닿아야 유리벽이다.
혼자 떠 있는 창호 선은 천창일 수 있어 만들지 않는다. 결과는 항상 검토 대상이다.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from ..layers import LayerInfo, LayerMap
from ..preprocess.primitive import Primitive
from .columns import ColumnDraft
from .walls import (
    Piece,
    WallDraft,
    collect_segments,
    find_pieces,
    group_by_direction,
)


def _snap_ends(a: np.ndarray, b: np.ndarray, anchors: Any, reach: float) -> tuple[np.ndarray, np.ndarray]:
    from .candidate_walls import snap_ends

    return snap_ends(a, b, anchors, reach)


def _solid(wall: WallDraft, margin: float = 0.0) -> Any:
    return LineString([wall.start, wall.end]).buffer(wall.thickness / 2 + margin, cap_style="square", join_style="mitre")


def _full_width_drafts(pieces: list[Any], cfg: dict[str, Any]) -> list[WallDraft]:
    """같은 창의 평행선 쌍을 벽 하나로 합친다. 창호는 흔히 틀과 유리를 선 네 줄로 그린다.

    평면의 창호 선은 대개 창대와 인방 사이의 띠창이다. 그래서 벽 폭은 전체 폭으로 두고, 유리는 내보낼 때 얇게 넣는다.
    """
    limit = cfg["glazed_max_thickness_mm"]
    clusters: list[list[Any]] = []
    for piece in sorted(pieces, key=lambda p: (p.u, p.s0)):
        for cluster in clusters:
            first = cluster[0]
            if first.u != piece.u:
                continue
            lo = min(p.offset - p.thickness / 2 for p in cluster)
            hi = max(p.offset + p.thickness / 2 for p in cluster)
            if max(hi, piece.offset + piece.thickness / 2) - min(lo, piece.offset - piece.thickness / 2) > limit:
                continue
            start, stop = min(p.s0 for p in cluster), max(p.s1 for p in cluster)
            if min(stop, piece.s1) - max(start, piece.s0) >= 0.5 * min(stop - start, piece.s1 - piece.s0):
                cluster.append(piece)
                break
        else:
            clusters.append([piece])
    drafts = []
    for cluster in clusters:
        lo = min(p.offset - p.thickness / 2 for p in cluster)
        hi = max(p.offset + p.thickness / 2 for p in cluster)
        s0, s1 = min(p.s0 for p in cluster), max(p.s1 for p in cluster)
        frame = Piece(cluster[0].u, s0, s1, (lo + hi) / 2, hi - lo, set(), set())
        drafts.append(WallDraft(
            np.array(frame.point(s0)), np.array(frame.point(s1)), hi - lo, "curtain",
            set().union(*(p.handles for p in cluster)), set().union(*(p.layers for p in cluster)), 0.0, glazed=True,
        ))
    return drafts


def _connected_parts(parts: list[Any], anchors: Any, reach: float) -> list[Any]:
    """양 끝이 벽, 기둥, 또는 살아남은 다른 유리에 닿는 구간만 남긴다.

    모서리에서는 두 유리가 서로의 끝에 기대어 어느 쪽도 혼자서는 벽에 양 끝이 닿지 않는다.
    그래서 끝이 비는 구간을 차례로 지우고, 벽이나 기둥에 이어지지 않은 묶음은 천창처럼 떠 있는 것으로 보고 뺀다.
    """
    alive = list(range(len(parts)))
    lines = [LineString([a, b]) for _, a, b in parts]

    def supported(point: np.ndarray, own: int) -> bool:
        spot = Point(point)
        if anchors.distance(spot) <= reach:
            return True
        return any(other != own and lines[other].distance(spot) <= reach for other in alive)

    changed = True
    while changed:
        changed = False
        for index in list(alive):
            _, a, b = parts[index]
            if not supported(a, index) or not supported(b, index):
                alive.remove(index)
                changed = True

    groups: list[set[int]] = []
    for index in alive:
        linked = [g for g in groups if any(lines[index].distance(lines[o]) <= reach for o in g)]
        merged = {index}.union(*linked) if linked else {index}
        groups = [g for g in groups if g not in linked] + [merged]
    keep = []
    for group in groups:
        if any(
            anchors.distance(Point(point)) <= reach
            for index in group for point in (parts[index][1], parts[index][2])
        ):
            keep.extend(sorted(group))
    return [parts[index] for index in sorted(keep)]


def glazed_walls(
    prims: list[Primitive], layers: LayerMap, walls: list[WallDraft], columns: list[ColumnDraft], cfg: dict[str, Any]
) -> list[WallDraft]:
    """기존 벽이 덮지 못한 창호 선 쌍을 벽으로 돌려준다."""
    view = LayerMap({
        name: LayerInfo("WALL", info.source) if info.role == "WINDOW" else LayerInfo("OTHER", info.source)
        for name, info in layers.layers.items()
    })
    if not view.has_walls:
        return []
    pair_cfg = {**cfg, "min_thickness_mm": cfg["glazed_min_thickness_mm"], "max_thickness_mm": cfg["glazed_max_thickness_mm"]}
    segments = collect_segments(prims, view, [], pair_cfg)
    pieces = []
    for group in group_by_direction(segments, math.radians(cfg["direction_tolerance_deg"])):
        pieces.extend(find_pieces(group, pair_cfg, False))
    if not pieces:
        return []
    drafts = _full_width_drafts(pieces, cfg)

    anchors = [_solid(w) for w in walls if w.length > 0] + [c.polygon for c in columns]
    if not anchors:
        return []
    anchor_union = unary_union(anchors)
    covered = unary_union([_solid(w, cfg["glazed_cover_margin_mm"]) for w in walls if w.length > 0]) if walls else None
    reach = cfg["glazed_anchor_reach_mm"]

    parts = []
    # 긴 것부터 받는다. 겹치는 후보는 한 번만 센다.
    taken: list[Any] = []
    for draft in sorted(drafts, key=lambda d: -d.length):
        line = LineString([draft.start, draft.end])
        rest = line.difference(covered) if covered is not None else line
        for part in getattr(rest, "geoms", [rest]):
            if part.is_empty or part.length < cfg["glazed_min_length_mm"]:
                continue
            if any(shape.contains(part.interpolate(0.5, normalized=True)) for shape in taken):
                continue
            a, b = np.array(part.coords[0]), np.array(part.coords[-1])
            parts.append((draft, a, b))
            taken.append(LineString([a, b]).buffer(draft.thickness / 2 + cfg["glazed_duplicate_margin_mm"]))

    kept = _connected_parts(parts, anchor_union, reach)
    result = []
    for index, (draft, a, b) in enumerate(kept):
        others = [LineString([p, q]).buffer(d.thickness / 2) for k, (d, p, q) in enumerate(kept) if k != index]
        a, b = _snap_ends(a, b, unary_union([anchor_union, *others]), reach)
        result.append(WallDraft(
            a, b, max(draft.thickness, cfg["glazed_min_output_mm"]), "curtain", set(draft.handles), set(draft.layers),
            0.0, glazed=True,
        ))
    return result
