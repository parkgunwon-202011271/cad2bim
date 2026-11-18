"""문·창·미확인 틈 인식 (상세설계 5.5).

- 문: 약 90도 호의 중심이 벽 면에 있고, 호의 한쪽 끝이 벽 방향이면 여닫이 문이다.
- 창: 벽선이 끊긴 틈 안에 벽과 나란한 선이 여러 개 있으면 창이다.
- 틈: 위 어느 것도 아니면 미확인으로 남긴다. 놓친 문을 벽으로 숨기는 쪽이 더 나쁘다.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from ..layers import LayerMap
from ..preprocess.primitive import Primitive
from .columns import ColumnDraft
from .walls import Piece, WallDraft

# 한국 도면의 창호 레이어에는 창과 문이 함께 있다. 호가 있으면 문이다.
_OPENING_ROLES = ("DOOR", "WINDOW")
_DOOR_ROLES = (*_OPENING_ROLES, "UNKNOWN")
_WINDOW_ROLES = ("WINDOW", "UNKNOWN", "DOOR")


@dataclass
class OpeningDraft:
    type: str
    wall: int  # walls 목록의 번호
    position: float
    width: float
    operation: str | None = None
    hinge: tuple[float, float] | None = None
    swing_dir: tuple[float, float] | None = None
    handles: list[str] = field(default_factory=list)
    role_known: bool = False
    reasons: list[str] = field(default_factory=list)


def _wall_tree(walls: list[WallDraft]) -> Any:
    return shapely.STRtree(shapely.linestrings([[w.start, w.end] for w in walls]) if walls else [])


def _has_door_shape(p: Primitive, cfg: dict[str, Any]) -> bool:
    center = np.array(p.geometry["center"], dtype=float)
    radius = float(p.geometry["radius"])
    start = np.array(p.geometry["start"], dtype=float) - center
    end = np.array(p.geometry["end"], dtype=float) - center
    sweep = math.degrees((math.atan2(end[1], end[0]) - math.atan2(start[1], start[0])) % (2 * math.pi))
    return abs(sweep - 90) <= cfg["quarter_tol_deg"] and cfg["min_radius_mm"] <= radius <= cfg["max_radius_mm"]


def _door_from_arc(
    p: Primitive, walls: list[WallDraft], tree: Any, cfg: dict[str, Any], margin: float, tolerance: float
) -> OpeningDraft | None:
    center = np.array(p.geometry["center"], dtype=float)
    radius = float(p.geometry["radius"])
    start = np.array(p.geometry["start"], dtype=float) - center
    end = np.array(p.geometry["end"], dtype=float) - center
    candidates = []
    near = shapely.box(center[0] - margin, center[1] - margin, center[0] + margin, center[1] + margin)
    for index in tree.query(near).tolist():
        wall = walls[index]
        u = wall.direction
        n = np.array([-u[1], u[0]])
        offset = abs(float((center - wall.start) @ n))
        if offset > wall.thickness / 2 + tolerance:
            continue
        # 닫힌 쪽 호 끝은 벽 방향, 열린 쪽은 벽에 수직이다.
        for closed, opened in ((start, end), (end, start)):
            if abs(float(closed @ n)) <= radius * cfg["axis_tolerance"] and abs(float(opened @ u)) <= radius * cfg["axis_tolerance"]:
                position = float((center + closed / 2 - wall.start) @ u)
                if -cfg["host_tolerance_mm"] <= position <= wall.length + cfg["host_tolerance_mm"]:
                    side = 1.0 if float(opened @ n) > 0 else -1.0
                    candidates.append((offset, index, position, tuple(float(v) for v in n * side), bool(closed @ u > 0)))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    offset, index, position, swing, _ = candidates[0]
    reasons = []
    if len({c[1] for c in candidates}) > 1 and candidates[1][0] - offset <= cfg["host_tolerance_mm"]:
        reasons.append("host_wall_ambiguous")
    return OpeningDraft(
        "door", index, max(position, 0.0), radius, "swing",
        (float(center[0]), float(center[1])), (round(swing[0], 6) + 0.0, round(swing[1], 6) + 0.0),
        list(p.source_handles), reasons=reasons,
    )


def _merge_double_doors(doors: list[OpeningDraft], cfg: dict[str, Any]) -> list[OpeningDraft]:
    """같은 벽에서 맞닿아 같은 쪽으로 열리는 두 문은 양개 문이다."""
    by_wall: dict[int, list[OpeningDraft]] = {}
    for door in doors:
        by_wall.setdefault(door.wall, []).append(door)
    result = []
    for members in by_wall.values():
        members.sort(key=lambda d: d.position)
        index = 0
        while index < len(members):
            door = members[index]
            following = members[index + 1] if index + 1 < len(members) else None
            if (
                following is not None
                and door.swing_dir == following.swing_dir
                and abs(following.position - door.position - (door.width + following.width) / 2) <= cfg["host_tolerance_mm"]
            ):
                result.append(OpeningDraft(
                    "door", door.wall, (door.position - door.width / 2 + following.position + following.width / 2) / 2,
                    door.width + following.width, "double", None, door.swing_dir,
                    door.handles + following.handles, door.role_known or following.role_known, door.reasons + following.reasons,
                ))
                index += 2
                continue
            result.append(door)
            index += 1
    return result


def _subtract(spans: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    out = []
    for a, b in spans:
        if hi <= a or lo >= b:
            out.append((a, b))
            continue
        if lo > a:
            out.append((a, lo))
        if hi < b:
            out.append((hi, b))
    return out


def _wall_gaps(
    index: int, walls: list[WallDraft], pieces: list[Piece], piece_tree: Any, wall_tree: Any,
    columns: list[ColumnDraft], column_tree: Any, cfg: dict[str, Any],
) -> list[tuple[float, float]]:
    """벽 축에서 벽선이 없는 구간. 기둥과 교차하는 벽이 차지한 부분은 틈이 아니다."""
    wall = walls[index]
    u = wall.direction
    n = np.array([-u[1], u[0]])
    corridor = shapely.buffer(shapely.linestrings([wall.start, wall.end]), wall.thickness / 2 + 1.0)
    spans = [(0.0, wall.length)]
    for j in piece_tree.query(corridor).tolist():
        piece = pieces[j]
        if abs(piece.u[0] * u[0] + piece.u[1] * u[1]) < cfg["parallel_cos"]:
            continue
        a, b = np.array(piece.point(piece.s0)), np.array(piece.point(piece.s1))
        if abs(float((a - wall.start) @ n)) > cfg["axis_offset_mm"] or abs(piece.thickness - wall.thickness) > cfg["axis_offset_mm"]:
            continue
        lo, hi = sorted((float((a - wall.start) @ u), float((b - wall.start) @ u)))
        spans = _subtract(spans, lo, hi)
    for j in column_tree.query(corridor).tolist():
        coords = np.array(columns[j].polygon.exterior.coords)
        along = (coords - wall.start) @ u
        spans = _subtract(spans, float(along.min()), float(along.max()))
    for j in wall_tree.query(corridor).tolist():
        other = walls[j]
        if j == index or abs(float(other.direction @ u)) > 0.5:
            continue
        along = sorted((float((other.start - wall.start) @ u), float((other.end - wall.start) @ u)))
        half = other.thickness / 2
        spans = _subtract(spans, along[0] - half, along[1] + half)
    return [(a, b) for a, b in spans if b - a >= cfg["min_gap_mm"]]


def _leaf_drawn_along_wall(
    door: OpeningDraft, arc: Primitive, wall: WallDraft, leaf_lines: list[tuple[np.ndarray, np.ndarray]], cfg: dict[str, Any],
) -> bool:
    """닫힌 자리로 고른 호의 다리 위에 문짝 선이 그려져 있는가."""
    center = np.array(arc.geometry["center"], dtype=float)
    u = wall.direction
    along = float((center - wall.start) @ u)
    closed_end = center + u * 2.0 * (door.position - along)
    tol = cfg["leaf_line_tolerance_mm"]
    for a, b in leaf_lines:
        for p, q in ((a, b), (b, a)):
            if np.linalg.norm(p - center) <= tol and np.linalg.norm(q - closed_end) <= tol:
                return True
    return False


def _drawn_through(
    door: OpeningDraft, walls: list[WallDraft], pieces: list[Piece], piece_tree: Any, wall_tree: Any,
    columns: list[ColumnDraft], column_tree: Any, cfg: dict[str, Any],
) -> bool:
    """문짝 자리에 벽선이 끊기지 않고 그려져 있는가. 벽선 쌍으로 찾은 벽만 판단한다."""
    wall = walls[door.wall]
    if wall.single_line or wall.from_door or wall.glazed or wall.candidate or not pieces:
        return False  # 벽선 쌍이 없는 벽은 끊김을 알 수 없다
    gaps = _wall_gaps(door.wall, walls, pieces, piece_tree, wall_tree, columns, column_tree, cfg)
    lo, hi = door.position - door.width / 2, door.position + door.width / 2
    open_share = sum(max(0.0, min(hi, b) - max(lo, a)) for a, b in gaps) / max(door.width, 1e-9)
    return open_share < cfg["door_min_gap_share"]


def _window_lines(prims: list[Primitive], layers: LayerMap) -> tuple[list[tuple[np.ndarray, np.ndarray, str, tuple[str, ...]]], Any]:
    lines = []
    for p in prims:
        if p.kind == "line" and layers.role(p.layer) in _WINDOW_ROLES:
            lines.append((np.array(p.geometry["p1"], dtype=float), np.array(p.geometry["p2"], dtype=float), p.layer, p.source_handles))
    tree = shapely.STRtree(shapely.linestrings([[a, b] for a, b, _, _ in lines]) if lines else [])
    return lines, tree


def _classify_gap(
    wall: WallDraft, gap: tuple[float, float], lines: list[Any], tree: Any, layers: LayerMap, cfg: dict[str, Any],
) -> tuple[str, list[str], bool]:
    """틈 안에 벽과 나란하고 틈을 대부분 덮는 선이 있으면 창이다."""
    u = wall.direction
    n = np.array([-u[1], u[0]])
    a = wall.start + u * gap[0]
    b = wall.start + u * gap[1]
    half = wall.thickness / 2 + 1.0
    area = shapely.buffer(shapely.linestrings([a, b]), half, cap_style="flat")
    length = gap[1] - gap[0]
    found, handles, known = 0, [], False
    for j in tree.query(area).tolist():
        p1, p2, layer, source = lines[j]
        if abs(float((p2 - p1) @ n)) > 1.0:
            continue
        if abs(float((p1 - wall.start) @ n)) > half:
            continue
        lo, hi = sorted((float((p1 - wall.start) @ u), float((p2 - wall.start) @ u)))
        if min(hi, gap[1]) - max(lo, gap[0]) >= length * cfg["line_min_coverage"]:
            found += 1
            handles += list(source)
            known = known or layers.role(layer) == "WINDOW"
    if found >= cfg["window_min_lines"]:
        return "window", sorted(set(handles)), known
    return "unknown", sorted(set(handles)), False


def _distance_bin(center: tuple[float, float], walls: list[WallDraft], tree: Any) -> str:
    """벽이 없어 탈락한 호가 벽에서 얼마나 먼가. 멀면 그 자리의 벽을 놓쳤다는 뜻이다."""
    point = Point(center)
    index = int(tree.nearest(point))
    distance = LineString([walls[index].start, walls[index].end]).distance(point)
    for limit, name in ((500, "lt500"), (2000, "lt2000"), (10000, "lt10000")):
        if distance < limit:
            return name
    return "far"


def recognize_openings(
    prims: list[Primitive], layers: LayerMap, walls: list[WallDraft], pieces: list[Piece],
    columns: list[ColumnDraft], cfg: dict[str, Any], stats: Counter | None = None,
) -> list[OpeningDraft]:
    """stats에는 호가 어느 단계에서 탈락했는지 센다. 문 호는 벽 놓침을 재는 객관적 근거다."""
    if stats is None:
        stats = Counter()
    if not walls:
        return []
    tree = _wall_tree(walls)
    margin = max(w.thickness for w in walls) / 2 + cfg["door_layer_tolerance_mm"]
    piece_tree = shapely.STRtree(
        shapely.linestrings([[p.point(p.s0), p.point(p.s1)] for p in pieces]) if pieces else []
    )
    column_tree = shapely.STRtree([c.polygon for c in columns])
    # 문 레이어의 곧은 선(문짝). 경첩에서 시작하는 선이 문짝의 열린 자리다.
    leaf_lines = [
        (np.array(p.geometry["p1"], dtype=float), np.array(p.geometry["p2"], dtype=float))
        for p in prims if p.kind == "line" and layers.role(p.layer) in _OPENING_ROLES
    ]

    doors = []
    for p in prims:
        if p.kind != "arc":
            continue
        if not _has_door_shape(p, cfg):
            continue
        role = layers.role(p.layer)
        group = "door_layer" if role in _OPENING_ROLES else "unknown_layer" if role == "UNKNOWN" else "other_layer"
        stats[f"{group}_arcs"] += 1
        if role not in _DOOR_ROLES:
            stats[f"role_{role}_arcs"] += 1
            continue
        # 문 레이어의 호는 문틀 때문에 벽 면에서 조금 떨어져도 문으로 믿는다.
        tolerance = cfg["door_layer_tolerance_mm"] if role in _OPENING_ROLES else cfg["host_tolerance_mm"]
        door = _door_from_arc(p, walls, tree, cfg, margin, tolerance)
        if door is None:
            if role in _OPENING_ROLES:
                stats[f"door_layer_nohost_{_distance_bin(p.geometry['center'], walls, tree)}"] += 1
            continue
        if _leaf_drawn_along_wall(door, p, walls[door.wall], leaf_lines, cfg):
            # 문짝 선은 열린 자리에 그린다. 그 선이 벽을 따라 놓였으면 닫힌 자리로 고른 쪽이 사실은 열린 자리다.
            stats[f"{group}_leaf_along_wall"] += 1
            continue
        if _drawn_through(door, walls, pieces, piece_tree, tree, columns, column_tree, cfg):
            # 벽선이 문짝 자리를 끊김 없이 지나가면 그 벽의 문이 아니다. 화장실 칸막이 문처럼
            # 칸막이와 바깥벽이 만나는 곳에 경첩이 있으면, 열린 문짝이 벽을 따라 놓여 벽에 걸린 것처럼 보인다.
            stats[f"{group}_solid_wall"] += 1
            continue
        stats[f"{group}_hosted"] += 1
        door.role_known = role in _OPENING_ROLES
        doors.append(door)
    doors = _merge_double_doors(doors, cfg)
    lines, line_tree = _window_lines(prims, layers)
    others = []
    for index, wall in enumerate(walls):
        # 선 한 줄로 그린 벽에는 벽선 쌍이 없어 틈을 판단할 수 없다. 문 호로 찾은 벽은 일부만 찾은
        # 조각이라 끊긴 곳을 틈으로 세면 미확인 틈만 늘어난다.
        if wall.single_line or wall.from_door:
            continue
        taken = [d for d in doors if d.wall == index]
        for gap in _wall_gaps(index, walls, pieces, piece_tree, tree, columns, column_tree, cfg):
            center = (gap[0] + gap[1]) / 2
            if any(abs(d.position - center) <= (d.width + gap[1] - gap[0]) / 2 for d in taken):
                continue
            kind, handles, known = _classify_gap(wall, gap, lines, line_tree, layers, cfg)
            others.append(OpeningDraft(
                kind, index, center, gap[1] - gap[0], "fixed" if kind == "window" else None,
                handles=handles, role_known=known,
                reasons=["unknown_opening"] if kind == "unknown" else [],
            ))
    return doors + others
