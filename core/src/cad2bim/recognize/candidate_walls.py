"""미분류 선에서 찾는 벽 후보. 고정 기준 대신 그 도면의 근거로 고르고, 닫힘에 기여할 때만 남긴다.

1. 평행선 쌍을 넓게 모은다. 레이어 전체가 벽처럼 보이지 않아도 선 단위로 본다.
2. 그 도면의 확실한 벽과 기둥에 양 끝이 닿아야 한다.
3. 같은 길이로 일정하게 반복되면 랙이나 주차선 같은 패턴으로 보고 뺀다.
4. 이미 닫힌 실 안을 가르는 후보는 뺀다.
5. 전부 넣어 본 뒤, 닫힌 영역의 경계를 이루는 후보만 남긴다. 한 개로는 안 닫혀도 여럿이 함께 닫는 경우를 살리기 위해서다.

결과는 항상 검토 대상이다.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ..layers import LayerInfo, LayerMap
from ..preprocess.primitive import Primitive
from .columns import ColumnDraft
from .walls import (
    WallDraft,
    _axis_groups,
    _merge_axis,
    _snap_thickness,
    collect_segments,
    find_pieces,
    group_by_direction,
    thickness_peaks,
)


def _solid(wall: WallDraft, margin: float = 0.0) -> Any:
    return LineString([wall.start, wall.end]).buffer(wall.thickness / 2 + margin, cap_style="square", join_style="mitre")


def _repeated(drafts: list[WallDraft], cfg: dict[str, Any]) -> set[int]:
    """같은 방향과 길이로 일정한 간격을 두고 반복되는 후보."""
    groups: dict[tuple[int, int], list[int]] = {}
    for index, draft in enumerate(drafts):
        angle = math.degrees(math.atan2(draft.direction[1], draft.direction[0])) % 180.0
        key = (round(angle / 2.0), round(draft.length / cfg["candidate_repeat_length_bin_mm"]))
        groups.setdefault(key, []).append(index)
    found: set[int] = set()
    for members in groups.values():
        if len(members) < cfg["candidate_repeat_min"]:
            continue
        d = drafts[members[0]].direction
        normal = np.array([-d[1], d[0]])
        ordered = sorted(members, key=lambda i: float((drafts[i].start + drafts[i].end) @ normal))
        offsets = [float((drafts[i].start + drafts[i].end) @ normal) / 2 for i in ordered]
        gaps = np.diff(offsets)
        run = [0]
        for k, gap in enumerate(gaps, start=1):
            if len(run) >= 2 and abs(gap - gaps[k - 2]) > cfg["candidate_repeat_gap_tol_mm"]:
                if len(run) >= cfg["candidate_repeat_min"]:
                    found.update(ordered[i] for i in run)
                run = [k - 1]
            run.append(k)
        if len(run) >= cfg["candidate_repeat_min"]:
            found.update(ordered[i] for i in run)
    return found


def snap_ends(a: np.ndarray, b: np.ndarray, anchors: Any, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """끝을 축 방향으로 늘려 벽이나 기둥에 붙인다. 작은 틈이 남으면 실이 닫히지 않는다."""
    length = float(np.linalg.norm(b - a))
    if length <= 0:
        return a, b
    direction = (b - a) / length

    def extended(point: np.ndarray, outward: np.ndarray) -> np.ndarray:
        if anchors.contains(Point(point)):
            return point
        hit = LineString([point, point + outward * reach]).intersection(anchors)
        if hit.is_empty:
            return point
        return point + outward * float(Point(point).distance(hit))

    return extended(a, -direction), extended(b, direction)


def _closed_regions(solids: list[Any], gap: float) -> list[Polygon]:
    """작은 틈은 문으로 보고 메운 뒤 닫힌 영역을 돌려준다."""
    joined = unary_union(solids).buffer(gap / 2, join_style="mitre").buffer(-gap / 2, join_style="mitre")
    regions = []
    for part in getattr(joined, "geoms", [joined]):
        if isinstance(part, Polygon):
            regions.extend(Polygon(ring) for ring in part.interiors)
    return regions


def candidate_walls(
    prims: list[Primitive], layers: LayerMap, walls: list[WallDraft], columns: list[ColumnDraft], cfg: dict[str, Any]
) -> list[WallDraft]:
    if not layers.has_walls or not walls:
        return []
    view = LayerMap({
        name: info if info.role == "UNKNOWN" else LayerInfo("OTHER", info.source)
        for name, info in layers.layers.items()
    })
    pair_cfg = {**cfg, "min_thickness_mm": cfg["candidate_min_thickness_mm"], "max_thickness_mm": cfg["candidate_max_thickness_mm"]}
    segments = collect_segments(prims, view, [], pair_cfg)
    pieces = []
    for group in group_by_direction(segments, math.radians(cfg["direction_tolerance_deg"])):
        pieces.extend(find_pieces(group, pair_cfg, True))
    if not pieces:
        return []
    snapped = _snap_thickness(pieces, thickness_peaks(pieces, cfg), cfg)
    drafts: list[WallDraft] = []
    for axis in _axis_groups(snapped, cfg) if snapped else []:
        drafts.extend(_merge_axis(axis, view, cfg))
    drafts = [d for d in drafts if d.length >= cfg["candidate_min_length_mm"]]
    if not drafts:
        return []

    repeated = _repeated(drafts, cfg)
    drafts = [d for i, d in enumerate(drafts) if i not in repeated]

    # 이 도면에서 벽으로 확실한 것만 기준으로 삼는다.
    trusted = [w for w in walls if w.length > 0 and not w.glazed and not w.from_door and not w.single_line]
    trusted = [w for w in trusted if not any(layers.layers[n].source == "inferred" for n in w.layers if n in layers.layers)] or [w for w in walls if w.length > 0]
    solids = [_solid(w) for w in walls if w.length > 0]
    columns_shapes = [c.polygon for c in columns]
    existing = solids + columns_shapes
    covered = unary_union([_solid(w, cfg["glazed_cover_margin_mm"]) for w in walls if w.length > 0])
    closed = unary_union(_closed_regions(existing, cfg["candidate_door_gap_mm"])) if existing else None
    reach = cfg["candidate_anchor_reach_mm"]

    # 후보끼리 이어져 닫는 경우가 있어 기준을 넓혀 가며 반복한다.
    anchor_shapes = [_solid(w) for w in trusted] + columns_shapes
    remaining: list[tuple[WallDraft, Any, np.ndarray, np.ndarray]] = []
    for draft in sorted(drafts, key=lambda d: -d.length):
        line = LineString([draft.start, draft.end])
        rest = line.difference(covered)
        for part in getattr(rest, "geoms", [rest]):
            if part.is_empty or part.length < cfg["candidate_min_length_mm"]:
                continue
            middle = part.interpolate(0.5, normalized=True)
            if closed is not None and closed.contains(middle):
                continue  # 이미 닫힌 실을 가르는 선은 벽을 찾는 근거가 되지 못한다
            remaining.append((draft, part, np.array(part.coords[0]), np.array(part.coords[-1])))

    accepted: list[tuple[WallDraft, np.ndarray, np.ndarray]] = []
    taken: list[Any] = []
    for _ in range(cfg["candidate_passes"]):
        if not remaining:
            break
        anchors = unary_union(anchor_shapes)
        kept = []
        for draft, part, a, b in remaining:
            if anchors.distance(Point(a)) <= reach and anchors.distance(Point(b)) <= reach:
                if any(shape.contains(part.interpolate(0.5, normalized=True)) for shape in taken):
                    continue
                a, b = snap_ends(a, b, anchors, reach)
                accepted.append((draft, a, b))
                shape = LineString([a, b]).buffer(max(draft.thickness, cfg["glazed_min_output_mm"]) / 2, cap_style="square", join_style="mitre")
                anchor_shapes.append(shape)
                taken.append(shape.buffer(cfg["glazed_duplicate_margin_mm"]))
            else:
                kept.append((draft, part, a, b))
        if len(kept) == len(remaining):
            break
        remaining = kept
    if not accepted:
        return []

    # 전체 최적화: 닫힌 영역의 경계에 닿는 후보만 남긴다.
    shapes = [
        LineString([a, b]).buffer(max(d.thickness, cfg["glazed_min_output_mm"]) / 2, cap_style="square", join_style="mitre")
        for d, a, b in accepted
    ]
    regions = _closed_regions(existing + shapes, cfg["candidate_door_gap_mm"])
    new_regions = [r for r in regions if closed is None or not closed.contains(r.representative_point()) and r.area >= cfg["candidate_min_region_mm2"]]
    if not new_regions:
        return []
    tree = shapely.STRtree(new_regions)
    result = []
    for (draft, a, b), shape in zip(accepted, shapes):
        if len(tree.query(shape, predicate="dwithin", distance=cfg["candidate_door_gap_mm"])):
            result.append(WallDraft(
                a, b, max(draft.thickness, cfg["glazed_min_output_mm"]), "unknown", set(draft.handles), set(draft.layers),
                0.0, candidate=True,
            ))
    return result
