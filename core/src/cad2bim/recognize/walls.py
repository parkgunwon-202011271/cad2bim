"""벽 인식 (상세설계 5.3).

1. 벽 후보 선을 방향별로 묶고 평행선 쌍에서 벽 조각을 만든다.
2. 두께 봉우리에 맞추고, 같은 축의 조각을 이어 벽을 만든다. 개구부·기둥 틈은 잇는다.
3. 짝 없는 단선 벽을 더한다.
4. 개구부 때문에 끊긴 벽 끝을 기둥이나 직교 벽까지 늘린다.
5. 축 교차점으로 끝을 정리한다.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from ..layers import LayerInfo, LayerMap
from ..preprocess.primitive import Primitive
from .columns import ColumnDraft
from .hatches import HatchIndex

_EPS = 1e-6
_MIN_CROSS = math.sin(math.radians(30))  # 이보다 작은 각으로 만나는 벽은 접합으로 보지 않는다
_COLLINEAR_COS = math.cos(math.radians(1))  # 이보다 나란하면 같은 방향의 벽이다


@dataclass
class Segment:
    a: tuple[float, float]
    b: tuple[float, float]
    handles: tuple[str, ...]
    layer: str
    length: float = 0.0
    used: bool = False

    def __post_init__(self) -> None:
        self.length = math.dist(self.a, self.b)


@dataclass
class Piece:
    """평행선 쌍 하나가 만든 벽 조각. u 방향 구간 [s0, s1], 법선 위치 offset."""
    u: tuple[float, float]
    s0: float
    s1: float
    offset: float
    thickness: float
    handles: set[str]
    layers: set[str]
    hatch: list[str | None] = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.s1 - self.s0

    def point(self, s: float) -> tuple[float, float]:
        ux, uy = self.u
        return (s * ux - self.offset * uy, s * uy + self.offset * ux)


@dataclass
class WallDraft:
    start: np.ndarray
    end: np.ndarray
    thickness: float
    kind: str
    handles: set[str]
    layers: set[str]
    hatch_share: float
    on_peak: bool = True
    single_line: bool = False
    from_door: bool = False  # 문 호를 근거로 미분류 선에서 찾은 벽
    glazed: bool = False  # 창호 선으로만 그려진 유리벽
    candidate: bool = False  # 미분류 선에서 닫힘을 근거로 찾은 벽
    joined: list[bool] = field(default_factory=lambda: [False, False])

    @property
    def length(self) -> float:
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def direction(self) -> np.ndarray:
        return (self.end - self.start) / max(self.length, _EPS)


def _direction_key(seg: Segment) -> float:
    return math.atan2(seg.b[1] - seg.a[1], seg.b[0] - seg.a[0]) % math.pi


def small_instances(prims: list[Primitive], max_size: float) -> set[int]:
    """크기가 작은 블록 삽입의 번호. 가구·설비·창호 같은 부품이다.

    도면 전체를 외부참조로 결합하면 벽도 블록 안에 있으므로 블록이라고 다 뺄 수 없다.
    """
    bounds: dict[int, list[float]] = {}
    for p in prims:
        instance = p.extra.get("instance", 0)
        box = p.bbox()
        if not instance or box is None:
            continue
        current = bounds.get(instance)
        if current is None:
            bounds[instance] = list(box)
        else:
            current[0] = min(current[0], box[0])
            current[1] = min(current[1], box[1])
            current[2] = max(current[2], box[2])
            current[3] = max(current[3], box[3])
    return {i for i, b in bounds.items() if max(b[2] - b[0], b[3] - b[1]) < max_size}


def _longest(p: Primitive) -> float:
    pts = [p.geometry["p1"], p.geometry["p2"]] if p.kind == "line" else list(p.geometry["points"])
    return max((math.dist(a, b) for a, b in zip(pts, pts[1:])), default=0.0)


def collect_segments(
    prims: list[Primitive], layers: LayerMap, columns: list[ColumnDraft], cfg: dict[str, Any]
) -> list[Segment]:
    """벽 레이어가 있으면 그 선만, 없으면 기하로 볼 수 있는 모든 선을 후보로 한다."""
    consumed = {i for c in columns for i in c.prim_ids}
    wanted = "WALL" if layers.has_walls else "UNKNOWN"
    small = small_instances(prims, cfg["part_block_max_mm"]) if wanted == "UNKNOWN" else set()
    segments = []
    # 기둥 레이어에 함께 그린 콘크리트 벽(코어벽 등)도 본다. 기둥이 쓰지 않은 긴 선만이다.
    from_columns = cfg.get("column_layer_wall_min_mm") if wanted == "WALL" else None
    for p in prims:
        if p.id in consumed:
            continue
        role = layers.role(p.layer)
        long_column_line = (
            from_columns is not None and role == "COLUMN" and p.kind in ("line", "polyline") and _longest(p) >= from_columns
        )
        if role != wanted and not long_column_line:
            continue
        if wanted == "UNKNOWN" and layers.shape_walls and p.layer not in layers.shape_walls:
            continue
        if wanted == "UNKNOWN" and (
            p.extra.get("instance", 0) in small or p.linetype.upper() not in ("CONTINUOUS", "BYLAYER")
        ):
            continue
        if p.kind == "line":
            pts, closed = [p.geometry["p1"], p.geometry["p2"]], False
        elif p.kind == "polyline":
            pts, closed = list(p.geometry["points"]), p.geometry["closed"]
            if closed:
                pts.append(pts[0])
        else:
            continue
        for a, b in zip(pts, pts[1:]):
            if math.dist(a, b) >= cfg["min_segment_mm"]:
                segments.append(Segment(a, b, p.source_handles, p.layer))
    return segments


def group_by_direction(segments: list[Segment], tolerance: float) -> list[list[Segment]]:
    ordered = sorted(segments, key=_direction_key)
    groups: list[list[Segment]] = []
    for seg in ordered:
        if groups and _direction_key(seg) - _direction_key(groups[-1][-1]) <= tolerance:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    if len(groups) > 1 and _direction_key(groups[0][0]) + math.pi - _direction_key(groups[-1][-1]) <= tolerance:
        groups[0].extend(groups.pop())
    return groups


def _frame(group: list[Segment]) -> tuple[float, float]:
    longest = max(group, key=lambda s: s.length)
    angle = _direction_key(longest)
    return (math.cos(angle), math.sin(angle))


def _project(seg: Segment, u: tuple[float, float]) -> tuple[float, float, float]:
    ux, uy = u
    s0 = seg.a[0] * ux + seg.a[1] * uy
    s1 = seg.b[0] * ux + seg.b[1] * uy
    mx, my = (seg.a[0] + seg.b[0]) / 2, (seg.a[1] + seg.b[1]) / 2
    return min(s0, s1), max(s0, s1), -uy * mx + ux * my


def _subtract(interval: tuple[float, float], blockers: list[tuple[float, float]]) -> list[tuple[float, float]]:
    remaining = [interval]
    for b0, b1 in blockers:
        next_round = []
        for r0, r1 in remaining:
            if b1 <= r0 or b0 >= r1:
                next_round.append((r0, r1))
                continue
            if b0 > r0:
                next_round.append((r0, b0))
            if b1 < r1:
                next_round.append((b1, r1))
        remaining = next_round
    return remaining


def _chain_close(items: Any, key: Any, tolerance: float) -> list[list[int]]:
    """key 값이 tolerance 이내로 이어지는 묶음."""
    groups: list[list[int]] = []
    previous = 0.0
    for item in sorted(items, key=key):
        value = key(item)
        if groups and value - previous <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])
        previous = value
    return groups


def _drop_repeated_treads(rows: list[tuple[float, float, float, Segment]], cfg: dict[str, Any]) -> list:
    """같은 길이의 짧은 평행선이 일정 간격으로 3개 이상이면 계단으로 보고 뺀다."""
    drop: set[int] = set()
    tolerance = cfg["stair_extent_tolerance_mm"]
    # 고정 구간으로 나누면 값이 경계에 걸릴 때 계산 오차로 같은 계단이 갈라진다. 거리로 묶는다.
    extents = [
        same_end
        for same_start in _chain_close(range(len(rows)), lambda i: rows[i][0], tolerance)
        for same_end in _chain_close(same_start, lambda i: rows[i][1], tolerance)
    ]
    for members in extents:
        if len(members) < cfg["stair_min_repeats"]:
            continue
        offsets = sorted(rows[i][2] for i in members)
        gaps = np.diff(offsets)
        if gaps.max() <= cfg["stair_max_spacing_mm"] and gaps.std() <= cfg["stair_spacing_tolerance_mm"]:
            drop.update(members)
    return [r for i, r in enumerate(rows) if i not in drop]


def find_pieces(group: list[Segment], cfg: dict[str, Any], geometric: bool) -> list[Piece]:
    u = _frame(group)
    rows = [(*_project(s, u), s) for s in group]
    if geometric:
        rows = _drop_repeated_treads(rows, cfg)
    rows.sort(key=lambda r: r[2])
    if not rows:
        return []
    starts = np.array([r[0] for r in rows])
    stops = np.array([r[1] for r in rows])
    offsets = np.array([r[2] for r in rows])
    min_overlap = cfg["min_overlap_mm"]

    pieces: list[Piece] = []
    for i, (s0, s1, off0, seg0) in enumerate(rows):
        # 경계값은 오차 범위까지 포함한다. 좌표를 옮겼다고 결과가 달라지면 안 된다.
        lo = max(int(np.searchsorted(offsets, off0 + cfg["min_thickness_mm"] - _EPS, side="left")), i + 1)
        hi = int(np.searchsorted(offsets, off0 + cfg["max_thickness_mm"] + _EPS, side="right"))
        if lo >= hi:
            continue
        gaps = offsets[lo:hi] - off0
        overlaps = np.minimum(stops[lo:hi], s1) - np.maximum(starts[lo:hi], s0)
        for k in np.flatnonzero(overlaps >= np.maximum(min_overlap, gaps) - _EPS).tolist():
            j = lo + k
            t0, t1, off1, seg1 = rows[j]
            gap = off1 - off0
            start, stop = max(s0, t0), min(s1, t1)
            # 사이에 다른 평행선이 있는 구간은 쌍이 아니다. 겹치는 선만 보면 된다.
            between = (
                (offsets[i + 1:j] > off0 + _EPS) & (offsets[i + 1:j] < off1 - _EPS)
                & (starts[i + 1:j] < stop) & (stops[i + 1:j] > start)
            )
            blockers = [(rows[i + 1 + b][0], rows[i + 1 + b][1]) for b in np.flatnonzero(between).tolist()]
            for x, y in _subtract((start, stop), blockers):
                if y - x < min_overlap - _EPS:
                    continue
                seg0.used = seg1.used = True
                pieces.append(Piece(
                    u, x, y, (off0 + off1) / 2, gap,
                    set(seg0.handles) | set(seg1.handles), {seg0.layer, seg1.layer},
                ))
    return pieces


def infer_wall_layers(prims: list[Primitive], layers: LayerMap, cfg: dict[str, Any]) -> set[str]:
    """선 대부분이 벽 두께 간격의 평행선 쌍을 이루는 미분류 레이어.

    실제 물류 평면에서 사전에 없는 벽 레이어가 많았다. 랙처럼 쌍을 이루는 다른 요소도 걸릴 수
    있으므로 이렇게 찾은 벽은 항상 검토 대상으로 표시한다. 벽 레이어를 하나도 못 찾은 도면은
    기하 모드가 이미 전체를 보므로 여기서 다루지 않는다.
    """
    if not layers.has_walls:
        return set()
    view = LayerMap({
        name: info if info.role == "UNKNOWN" else LayerInfo("OTHER", info.source)
        for name, info in layers.layers.items()
    })
    segments = collect_segments(prims, view, [], cfg)
    total: Counter = Counter()
    for seg in segments:
        total[seg.layer] += seg.length
    paired: Counter = Counter()
    thickness: dict[str, Counter] = defaultdict(Counter)
    tolerance = math.radians(cfg["direction_tolerance_deg"])
    for group in group_by_direction(segments, tolerance):
        for piece in find_pieces(group, cfg, True):
            for layer in piece.layers:
                # 쌍 하나는 양쪽 선을 덮는다. 한 레이어에 두 선이 다 있으면 길이의 두 배다.
                paired[layer] += piece.length * (2 if len(piece.layers) == 1 else 1)
                thickness[layer][round(piece.thickness / 10) * 10] += piece.length

    found = set()
    for layer, length in total.items():
        if length < cfg["infer_min_length_mm"] or not thickness[layer]:
            continue
        mode, mode_length = thickness[layer].most_common(1)[0]
        # 벽 레이어는 두께가 몇 가지로 모인다. 간격이 제각각이면 벽이 아닌 요소가 우연히 짝을 이룬 것이다.
        concentrated = mode_length / sum(thickness[layer].values()) >= cfg["infer_min_mode_share"]
        if paired[layer] / length >= cfg["infer_min_paired"] and mode <= cfg["infer_max_thickness_mm"] and concentrated:
            found.add(layer)
    return found


def infer_walls_by_shape(prims: list[Primitive], layers: LayerMap, cfg: dict[str, Any]) -> set[str]:
    """이름으로 벽을 못 찾은 도면에서 벽 후보 레이어를 고른다. 비어 있으면 모든 선을 후보로 둔다.

    벽 레이어는 긴 평행선 쌍이 선 길이의 상당 부분을 차지한다. 가구와 계단도 쌍을 이루지만 대부분 짧다.
    창호도 함께 걸릴 수 있으나 건물 외곽 위에 있어 위치는 틀리지 않는다.
    """
    if layers.has_walls:
        return set()
    view = LayerMap({
        name: info if info.role == "UNKNOWN" else LayerInfo("OTHER", info.source)
        for name, info in layers.layers.items()
    })
    segments = collect_segments(prims, view, [], cfg)
    total: Counter = Counter()
    for seg in segments:
        total[seg.layer] += seg.length
    long_pairs: Counter = Counter()
    tolerance = math.radians(cfg["direction_tolerance_deg"])
    for group in group_by_direction(segments, tolerance):
        for piece in find_pieces(group, cfg, True):
            if piece.length >= cfg["infer_shape_long_mm"] and piece.thickness <= cfg["infer_max_thickness_mm"]:
                for layer in piece.layers:
                    long_pairs[layer] += piece.length
    everything = sum(total.values())
    chosen = {
        layer for layer, length in total.items()
        if length >= everything * cfg["infer_shape_min_share"] and long_pairs[layer] / length >= cfg["infer_shape_min_long_share"]
    }
    # 기준값 근처의 도면에서 벽 레이어가 통째로 빠지지 않게 긴 쌍이 가장 많은 쪽은 항상 넣는다.
    # 창호가 벽보다 약간 많은 도면이 있어 1등에 가까운 레이어도 함께 넣는다.
    if long_pairs:
        most = long_pairs.most_common(1)[0][1]
        chosen |= {layer for layer, value in long_pairs.items() if value >= most * cfg["infer_shape_near_best"]}
    return chosen


def thickness_peaks(pieces: list[Piece], cfg: dict[str, Any]) -> list[float]:
    """길이로 가중한 두께 히스토그램의 봉우리."""
    weights: Counter = Counter()
    bin_mm = cfg["thickness_bin_mm"]
    for p in pieces:
        weights[round(p.thickness / bin_mm) * bin_mm] += p.length
    total = sum(weights.values())
    return sorted(v for v, w in weights.items() if total and w / total >= cfg["peak_min_share"])


def _snap_thickness(pieces: list[Piece], peaks: list[float], cfg: dict[str, Any]) -> list[tuple[Piece, bool]]:
    result = []
    for p in pieces:
        nearest = min(peaks, key=lambda t: abs(t - p.thickness), default=None)
        on_peak = nearest is not None and abs(nearest - p.thickness) <= cfg["peak_snap_mm"]
        if on_peak:
            p.thickness = nearest
        elif p.length < cfg["outlier_min_length_mm"]:
            continue  # 끝선과 옆 벽면이 우연히 만든 짧은 쌍
        result.append((p, on_peak))
    return result


def _annotate_hatch(pieces: list[Piece], hatches: HatchIndex, cfg: dict[str, Any]) -> None:
    n = cfg["hatch_samples"]
    for p in pieces:
        pts = [p.point(p.s0 + p.length * (i + 0.5) / n) for i in range(n)]
        p.hatch = hatches.kinds_at(pts)


def _hatch_share(p: Piece) -> float:
    return sum(k is not None for k in p.hatch) / len(p.hatch) if p.hatch else 0.0


def _kind_of(pieces: list[Piece], layers: LayerMap) -> str:
    votes: Counter = Counter()
    for p in pieces:
        for kind in p.hatch:
            if kind not in (None, "unknown"):
                votes[kind] += p.length / len(p.hatch)
    if votes:
        return votes.most_common(1)[0][0]
    for p in pieces:
        for layer in sorted(p.layers):
            material = layers.material(layer)
            if material:
                return material
    return "unknown"


def _merge_axis(axis: list[tuple[Piece, bool]], layers: LayerMap, cfg: dict[str, Any]) -> list[WallDraft]:
    """같은 축, 같은 두께의 조각을 잇는다. 벽이 문마다 끊기면 호스팅이 안 되기 때문이다."""
    ordered = sorted(axis, key=lambda x: x[0].s0)
    runs: list[list[tuple[Piece, bool]]] = [[ordered[0]]]
    end = ordered[0][0].s1
    for item in ordered[1:]:
        if item[0].s0 - end <= cfg["max_bridge_mm"]:
            runs[-1].append(item)
        else:
            runs.append([item])
        end = max(end, item[0].s1) if len(runs[-1]) > 1 else item[0].s1

    drafts = []
    for run in runs:
        members = [p for p, _ in run]
        first = members[0]
        s0 = min(p.s0 for p in members)
        s1 = max(p.s1 for p in members)
        offset = sum(p.offset * p.length for p in members) / sum(p.length for p in members)
        frame = Piece(first.u, s0, s1, offset, first.thickness, set(), set())
        total = sum(p.length for p in members)
        drafts.append(WallDraft(
            np.array(frame.point(s0)), np.array(frame.point(s1)), first.thickness,
            _kind_of(members, layers),
            set().union(*(p.handles for p in members)),
            set().union(*(p.layers for p in members)),
            sum(_hatch_share(p) * p.length for p in members) / total,
            on_peak=all(ok for _, ok in run),
        ))
    return drafts


def _axis_groups(snapped: list[tuple[Piece, bool]], cfg: dict[str, Any]) -> list[list[tuple[Piece, bool]]]:
    # 방향이 다르면 오프셋이 같아도 다른 축이다.
    by_axis: dict[tuple, list[tuple[Piece, bool]]] = defaultdict(list)
    for item in snapped:
        by_axis[(item[0].u, round(item[0].thickness))].append(item)
    groups = []
    for items in by_axis.values():
        items.sort(key=lambda x: x[0].offset)
        current = [items[0]]
        for item in items[1:]:
            if item[0].offset - current[-1][0].offset <= cfg["axis_tolerance_mm"]:
                current.append(item)
            else:
                groups.append(current)
                current = [item]
        groups.append(current)
    return groups


def _wall_tree(walls: list[WallDraft]) -> shapely.STRtree:
    return shapely.STRtree(shapely.linestrings([[w.start, w.end] for w in walls]) if walls else [])


def _near_pairs(walls: list[WallDraft], margin: float) -> list[tuple[int, int]]:
    """범위가 margin 이내인 벽 쌍 (i < j). 전체 쌍을 비교하지 않기 위한 후보다."""
    if not walls:
        return []
    corners = np.array([[*w.start, *w.end] for w in walls])
    lo = np.minimum(corners[:, :2], corners[:, 2:]) - margin
    hi = np.maximum(corners[:, :2], corners[:, 2:]) + margin
    left, right = _wall_tree(walls).query(shapely.box(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1]))
    keep = left < right
    return sorted(zip(left[keep].tolist(), right[keep].tolist()))


def _drop_nested(walls: list[WallDraft], cfg: dict[str, Any]) -> list[WallDraft]:
    """더 두꺼운 벽 안에 들어가는 얇은 벽은 창틀 같은 내부선이다."""
    def contains(other: WallDraft, w: WallDraft) -> bool:
        if other.thickness < w.thickness:
            return False
        if other.thickness == w.thickness and other.length <= w.length:
            return False
        direction = other.direction
        if abs(float(direction @ w.direction)) < 0.999:
            return False
        normal = np.array([-direction[1], direction[0]])
        for point in (w.start, w.end):
            rel = point - other.start
            along = float(rel @ direction)
            across = abs(float(rel @ normal))
            if not (-_EPS <= along <= other.length + _EPS and across <= other.thickness / 2 + _EPS):
                return False
        return True

    nested: set[int] = set()
    max_half = max((w.thickness for w in walls), default=0.0) / 2
    for i, j in _near_pairs(walls, max_half + _EPS):
        if contains(walls[j], walls[i]):
            nested.add(i)
        if contains(walls[i], walls[j]):
            nested.add(j)
    return [w for i, w in enumerate(walls) if i not in nested]


def _distance_to_wall(point: np.ndarray, wall: WallDraft) -> float:
    rel = point - wall.start
    t = min(max(float(rel @ wall.direction), 0.0), wall.length)
    return float(np.linalg.norm(point - (wall.start + wall.direction * t)))


def add_single_line_walls(
    walls: list[WallDraft], segments: list[Segment], cfg: dict[str, Any], layers: LayerMap
) -> None:
    """짝이 없는 선이 서로 다른 두 벽에 양 끝을 대고 있으면 벽으로 만든다. 실이 닫혀야 한다."""
    tolerance = cfg["junction_tolerance_mm"]
    existing = len(walls)
    tree = _wall_tree(walls)
    reach = max((w.thickness for w in walls), default=0.0) / 2 + tolerance

    def touching(point: np.ndarray) -> set[int]:
        near = tree.query(shapely.box(*(point - reach), *(point + reach))).tolist()
        near += range(existing, len(walls))  # 이번에 새로 만든 벽은 인덱스에 없다
        return {
            i for i in near
            if _distance_to_wall(point, walls[i]) <= walls[i].thickness / 2 + tolerance
        }

    for seg in segments:
        if seg.used or seg.length < cfg["single_line_min_mm"]:
            continue
        a, b = np.array(seg.a), np.array(seg.b)
        at_a, at_b = touching(a), touching(b)
        if at_a and at_b and (at_a - at_b) and (at_b - at_a):
            seg.used = True
            walls.append(WallDraft(
                a, b, cfg["single_line_thickness_mm"], layers.material(seg.layer) or "unknown",
                set(seg.handles), {seg.layer}, 0.0, single_line=True,
            ))


def _intersection(a: WallDraft, b: WallDraft) -> np.ndarray | None:
    da, db = a.direction, b.direction
    cross = da[0] * db[1] - da[1] * db[0]
    if abs(cross) < _MIN_CROSS:
        return None
    rel = b.start - a.start
    t = (rel[0] * db[1] - rel[1] * db[0]) / cross
    return a.start + da * t


class _ColumnIndex:
    """기둥 공간 인덱스. 코너마다 도면 전체의 기둥을 훑지 않는다."""

    def __init__(self, columns: list[ColumnDraft]):
        self.columns = columns
        self.polygons = shapely.STRtree([c.polygon for c in columns])
        self.halves = [max(c.size) / 2 * math.sqrt(2) for c in columns]
        self.max_half = max(self.halves, default=0.0)
        self.reach_boxes = shapely.STRtree([
            shapely.box(c.center[0] - h, c.center[1] - h, c.center[0] + h, c.center[1] + h)
            for c, h in zip(columns, self.halves)
        ])

    def near(self, point: np.ndarray, tolerance: float) -> bool:
        return self.polygons.query(Point(point), predicate="dwithin", distance=tolerance).size > 0

    def crossings(self, ray: LineString) -> list[Any]:
        found = self.polygons.query(ray, predicate="intersects")
        return [ray.intersection(self.columns[int(i)].polygon) for i in sorted(found.tolist())]

    def reach(self, point: np.ndarray) -> float:
        """교차점이 기둥 안에 있으면 기둥 면까지의 거리만큼 더 닿을 수 있다."""
        for i in sorted(self.reach_boxes.query(Point(point)).tolist()):
            if np.linalg.norm(point - np.array(self.columns[i].center)) <= self.halves[i]:
                return self.halves[i]
        return 0.0


def _end_near(wall: WallDraft, point: np.ndarray, reach: float) -> int | None:
    distances = [float(np.linalg.norm(wall.start - point)), float(np.linalg.norm(wall.end - point))]
    index = int(distances[1] < distances[0])
    return index if distances[index] <= reach else None


def _inside(wall: WallDraft, point: np.ndarray, margin: float) -> bool:
    # 축 좌표에 부동소수 오차가 있어 정확히 끝점인 교차도 살짝 밖으로 나올 수 있다.
    along = float((point - wall.start) @ wall.direction)
    return -margin - _EPS <= along <= wall.length + margin + _EPS


class _LineIndex:
    """벽 틈을 채울 수 있는 선. 벽 레이어가 없으면 분류를 몰라 모든 선을 본다."""

    def __init__(self, prims: list[Primitive], layers: LayerMap):
        wanted = ("DOOR", "WINDOW") if layers.has_walls else ("UNKNOWN",)
        self.lines = [
            (np.array(p.geometry["p1"]), np.array(p.geometry["p2"]))
            for p in prims if p.kind == "line" and layers.role(p.layer) in wanted
        ]
        self.tree = shapely.STRtree(
            shapely.linestrings([[a, b] for a, b in self.lines]) if self.lines else []
        )

    def around(self, a: np.ndarray, b: np.ndarray, margin: float) -> list[tuple[np.ndarray, np.ndarray]]:
        lo, hi = np.minimum(a, b) - margin, np.maximum(a, b) + margin
        return [self.lines[int(i)] for i in self.tree.query(shapely.box(*lo, *hi))]


def _gap_filled(
    origin: np.ndarray, direction: np.ndarray, length: float, half_width: float,
    lines: _LineIndex, min_share: float,
) -> bool:
    """벽 축을 따라 틈의 min_share 이상이 문·창 선으로 덮이면 개구부로 본다."""
    normal = np.array([-direction[1], direction[0]])
    spans = []
    for a, b in lines.around(origin, origin + direction * length, half_width):
        if abs(float((a - origin) @ normal)) > half_width or abs(float((b - origin) @ normal)) > half_width:
            continue
        s = sorted((float((a - origin) @ direction), float((b - origin) @ direction)))
        lo, hi = max(s[0], 0.0), min(s[1], length)
        if hi > lo:
            spans.append((lo, hi))
    covered, reach = 0.0, 0.0
    for lo, hi in sorted(spans):
        covered += max(hi - max(lo, reach), 0.0)
        reach = max(reach, hi)
    return covered >= length * min_share


def _extend_once(
    walls: list[WallDraft], columns: _ColumnIndex, lines: _LineIndex, cfg: dict[str, Any]
) -> None:
    limit = cfg["max_bridge_mm"]
    tolerance = cfg["junction_tolerance_mm"]
    tree = _wall_tree(walls)
    margin = max((w.thickness for w in walls), default=0.0) / 2 + tolerance
    moves: list[tuple[WallDraft, int, np.ndarray]] = []
    for wall in walls:
        for end in (0, 1):
            point = wall.end if end else wall.start
            if columns.near(point, tolerance):
                continue
            outward = wall.direction if end else -wall.direction
            tip = point + outward * limit
            ray = LineString([point, tip])
            origin = Point(point)
            hits = [(origin.distance(c), True) for c in columns.crossings(ray)]  # (거리, 기둥인가)
            # 같은 축의 다른 벽을 넘어 늘리지 않는다. 넘으면 두 벽이 겹친다.
            blocked = limit
            normal = np.array([-wall.direction[1], wall.direction[0]])

            lo, hi = np.minimum(point, tip) - margin, np.maximum(point, tip) + margin
            for i in tree.query(shapely.box(*lo, *hi)).tolist():
                other = walls[i]
                if other is wall:
                    continue
                if abs(float(other.direction @ wall.direction)) >= _COLLINEAR_COS:
                    if abs(float((other.start - point) @ normal)) <= max(wall.thickness, other.thickness) / 2:
                        a, b = float((other.start - point) @ outward), float((other.end - point) @ outward)
                        if max(a, b) > tolerance:
                            blocked = min(blocked, max(min(a, b), 0.0))
                    continue
                x = _intersection(wall, other)
                # 상대 벽도 아직 코너까지 닿지 않았을 수 있다.
                if x is None or not _inside(other, x, other.thickness / 2 + tolerance):
                    continue
                along = float((x - point) @ outward)
                if 0 < along <= limit:
                    hits.append((along, False))
            hits = [h for h in hits if h[0] <= blocked + tolerance]
            if not hits:
                continue
            distance, is_column = min(hits)
            if distance <= wall.thickness / 2 + tolerance:
                continue
            if is_column or _gap_filled(
                point, outward, distance, wall.thickness / 2 + tolerance, lines,
                cfg["opening_fill_min_share"],
            ):
                moves.append((wall, end, point + outward * distance))
    for wall, end, target in moves:
        if end:
            wall.end = target
        else:
            wall.start = target


def extend_to_anchors(
    walls: list[WallDraft], columns: _ColumnIndex, lines: _LineIndex, cfg: dict[str, Any]
) -> None:
    """끝에서 축을 따라 가장 먼저 만나는 기둥이나 직교 벽까지 늘린다.

    개구부가 기둥이나 코너에 붙으면 그 사이 벽이 너무 짧아 쌍이 생기지 않는다.
    기둥은 바로 잇고, 직교 벽은 틈에 문·창이 그려져 있을 때만 잇는다.
    양쪽 벽이 모두 짧은 경우 한쪽이 먼저 늘어나야 다른 쪽이 닿으므로 두 번 돈다.
    """
    for _ in range(2):
        _extend_once(walls, columns, lines, cfg)


def door_panel_walls(prims: list[Primitive], layers: LayerMap, walls: list[WallDraft], cfg: dict[str, Any]) -> list[WallDraft]:
    """벽 없이 문 레이어의 나란한 두 선으로만 그린 큰 문(하역장 오버헤드 도어 등)을 문짝 벽으로 만든다.

    벽선이 없으니 벽 틈도 없어 문으로 읽히지 않고 면 전체가 뚫린다. 기존 벽의 범위 안(벽의 틈)에 있는 것은
    그 벽의 문이므로 만들지 않는다. 만든 벽은 전체가 문이다(from_door).
    """
    min_len = cfg.get("door_panel_min_mm")
    if min_len is None:
        return []
    lo_gap, hi_gap = cfg["door_panel_min_gap_mm"], cfg["door_panel_max_gap_mm"]
    lines = []
    for p in prims:
        if p.kind != "line" or layers.role(p.layer) != "DOOR":
            continue
        a, b = np.array(p.geometry["p1"], float), np.array(p.geometry["p2"], float)
        if np.linalg.norm(b - a) >= min_len:
            lines.append((a, b, p))
    tree = _wall_tree(walls)
    found: list[WallDraft] = []
    used: set[int] = set()
    for i, (a, b, p) in enumerate(lines):
        if i in used:
            continue
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        for j in range(i + 1, len(lines)):
            c, d, q = lines[j]
            v = (d - c) / np.linalg.norm(d - c)
            if j in used or abs(float(u @ v)) < _COLLINEAR_COS:
                continue
            gap = abs(float((c - a) @ n))
            if not lo_gap <= gap <= hi_gap:
                continue
            t0, t1 = sorted((float((c - a) @ u), float((d - a) @ u)))
            s0, s1 = max(0.0, t0), min(float(np.linalg.norm(b - a)), t1)
            if s1 - s0 < min_len:
                continue
            side = 1.0 if float((c - a) @ n) > 0 else -1.0
            start = a + u * s0 + n * side * gap / 2
            end = a + u * s1 + n * side * gap / 2
            middle = LineString([(start + end) / 2 - u, (start + end) / 2 + u])
            inside_wall = any(
                abs(float((walls[k].start - start) @ n)) <= walls[k].thickness / 2 + gap
                and abs(float(walls[k].direction @ u)) >= _COLLINEAR_COS
                for k in tree.query(middle.buffer(max((w.thickness for w in walls), default=0.0))).tolist()
            )
            if inside_wall:
                continue
            found.append(WallDraft(start, end, gap, "unknown", set(p.source_handles) | set(q.source_handles), {p.layer}, 0.0, from_door=True))
            used.update((i, j))
            break
    return found


def bridge_across_columns(walls: list[WallDraft], columns: _ColumnIndex, cfg: dict[str, Any]) -> int:
    """같은 축의 두 벽 사이를 기둥이 막고 있으면 잇는다. 이은 수를 돌려준다.

    외벽 패널을 기둥 바로 바깥에 그리고 기둥 자리에서 끊으면, 벽 끝과 기둥 사이로 밖이 보인다.
    그 사이에 기둥이 서 있으니 문이나 창일 수 없다. 기둥이 벽 축에서 offset 안에 있을 때만 잇는다.
    """
    offset = cfg.get("column_bridge_offset_mm")
    if offset is None or not columns.columns:
        return 0
    slack = cfg["column_bridge_slack_mm"]
    reach = 2 * columns.max_half + slack
    tree = _wall_tree(walls)
    joined = 0
    for i, wall in enumerate(walls):
        normal = np.array([-wall.direction[1], wall.direction[0]])
        for end in (0, 1):
            point = wall.end if end else wall.start
            outward = wall.direction if end else -wall.direction
            best = None
            for j in tree.query(shapely.box(*(point - reach), *(point + reach))).tolist():
                other = walls[j]
                if j == i or abs(float(other.direction @ wall.direction)) < _COLLINEAR_COS:
                    continue
                if abs(float((other.start - point) @ normal)) > max(wall.thickness, other.thickness) / 2:
                    continue
                near_end = min((other.start, other.end), key=lambda q: float((q - point) @ outward))
                gap = float((near_end - point) @ outward)
                far = max(float((other.start - point) @ outward), float((other.end - point) @ outward))
                if not 1.0 < gap <= reach or far <= gap:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, near_end)
            if best is None:
                continue
            gap, target = best
            band = shapely.buffer(LineString([point, target]), offset, cap_style="flat")
            between = [columns.columns[k] for k in columns.polygons.query(band, predicate="intersects").tolist()]
            if not any(0.0 <= float((np.array(c.center) - point) @ outward) <= gap for c in between):
                continue
            if end:
                wall.end = target.copy()
            else:
                wall.start = target.copy()
            wall.joined[end] = True
            joined += 1
    return joined


def slit_fillers(walls: list[WallDraft], cfg: dict[str, Any], holes: dict[int, list[tuple[float, float]]] | None = None) -> list[WallDraft]:
    """나란히 겹쳐 그린 두 벽(외벽 패널 두 겹 등) 사이 좁은 틈을 메우는 얇은 벽.

    도면 축척에서는 붙어 있는 두 겹인데, 3D에서는 그 틈으로 밖이 보인다. 겹친 길이만큼만 메운다.
    holes는 벽 번호별 개구부 구간(벽 시작에서의 거리)이다. 어느 한쪽 벽에 문·창이 있으면 그 자리는 메우지 않는다.
    """
    max_gap = cfg.get("slit_max_mm")
    if max_gap is None:
        return []
    min_overlap = cfg["slit_min_overlap_mm"]
    tree = _wall_tree(walls)
    fillers = []
    for i, a in enumerate(walls):
        normal = np.array([-a.direction[1], a.direction[0]])
        reach = a.thickness / 2 + max_gap + max((w.thickness for w in walls), default=0.0) / 2
        lo = np.minimum(a.start, a.end) - reach
        hi = np.maximum(a.start, a.end) + reach
        for j in tree.query(shapely.box(*lo, *hi)).tolist():
            if j <= i:
                continue
            b = walls[j]
            if abs(float(b.direction @ a.direction)) < _COLLINEAR_COS:
                continue
            offset = float((b.start - a.start) @ normal)
            gap = abs(offset) - (a.thickness + b.thickness) / 2
            if not 0.0 < gap <= max_gap:
                continue
            u = a.direction
            s0, s1 = 0.0, a.length
            t0, t1 = sorted((float((b.start - a.start) @ u), float((b.end - a.start) @ u)))
            o0, o1 = max(s0, t0), min(s1, t1)
            if o1 - o0 < min_overlap:
                continue
            side = 1.0 if offset > 0 else -1.0
            middle = side * (a.thickness / 2 + gap / 2)
            cuts = list((holes or {}).get(i, []))
            for lo_b, hi_b in (holes or {}).get(j, []):  # 다른 벽의 개구부를 이 벽 축으로 옮긴다
                p0 = float((b.start + b.direction * lo_b - a.start) @ u)
                p1 = float((b.start + b.direction * hi_b - a.start) @ u)
                cuts.append((min(p0, p1), max(p0, p1)))
            pieces = [(o0, o1)]
            for c0, c1 in cuts:
                pieces = [q for s0, s1 in pieces for q in ((s0, min(s1, c0)), (max(s0, c1), s1)) if q[1] - q[0] > 1.0]
            for s0, s1 in pieces:
                if s1 - s0 < 200.0:
                    continue
                start = a.start + u * s0 + normal * middle
                fillers.append(WallDraft(
                    start, start + u * (s1 - s0), gap + 2.0, a.kind, set(), set(a.layers) | set(b.layers), 0.0, candidate=True,
                ))
    return fillers


def close_collinear_gaps(walls: list[WallDraft], cfg: dict[str, Any]) -> None:
    """같은 축에서 끝과 끝이 벽 두께 절반 남짓 떨어진 두 벽을 맞붙인다.

    유리벽은 이웃 벽의 외곽(끝에서 두께 절반만큼 더 나간 사각 끝)에 끝을 맞추므로, 내보낸 벽 사이에
    두께 절반(보통 150mm)짜리 틈이 남아 밖이 보인다. 개구부 최소 폭보다 훨씬 좁아 개구부일 수 없다.
    추정해 만든 벽(유리·문 메움·후보) 쪽을 옮기고, 둘 다 같은 종류면 가운데서 만난다.
    """
    tolerance = cfg["junction_tolerance_mm"]
    tree = _wall_tree(walls)
    inferred = lambda w: w.glazed or w.from_door or w.candidate  # noqa: E731
    for i, wall in enumerate(walls):
        normal = np.array([-wall.direction[1], wall.direction[0]])
        for end in (0, 1):
            point = wall.end if end else wall.start
            outward = wall.direction if end else -wall.direction
            reach = wall.thickness / 2 + tolerance
            for j in tree.query(shapely.box(*(point - reach), *(point + reach))).tolist():
                if j <= i:
                    continue
                other = walls[j]
                if abs(float(other.direction @ wall.direction)) < _COLLINEAR_COS:
                    continue
                if abs(float((other.start - point) @ normal)) > max(wall.thickness, other.thickness) / 2 * 0.5:
                    continue  # 축이 어긋난 나란한 벽
                for other_end in (0, 1):
                    target = other.end if other_end else other.start
                    gap = float((target - point) @ outward)
                    # 바깥쪽으로 떨어진 끝만. 겹친 것은 다른 단계에서 다룬다.
                    if not 1.0 < gap <= max(wall.thickness, other.thickness) / 2 + tolerance:
                        continue
                    if inferred(wall) and not inferred(other):
                        meet = target.copy()
                    elif inferred(other) and not inferred(wall):
                        meet = point.copy()
                    else:
                        meet = (point + target) / 2
                    if end:
                        wall.end = meet.copy()
                    else:
                        wall.start = meet.copy()
                    if other_end:
                        other.end = meet.copy()
                    else:
                        other.start = meet.copy()
                    wall.joined[end] = other.joined[other_end] = True
                    point = meet


def clean_junctions(walls: list[WallDraft], columns: _ColumnIndex, cfg: dict[str, Any]) -> None:
    """벽 축의 교차점으로 끝을 옮긴다. L자는 양쪽, T자는 끝나는 벽만 움직인다."""
    moves: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    joined: set[tuple[int, int]] = set()
    tolerance = cfg["junction_tolerance_mm"]
    max_half = max((w.thickness for w in walls), default=0.0) / 2
    # 교차점은 두 벽 모두에서 닿는 거리 안에 있어야 하므로 벽 사이는 그 두 배 이내다.
    for i, j in _near_pairs(walls, 2 * (max_half + tolerance + columns.max_half)):
        a, b = walls[i], walls[j]
        x = _intersection(a, b)
        if x is None:
            continue
        reach = max(a.thickness, b.thickness) / 2 + tolerance + columns.reach(x)
        end_a, end_b = _end_near(a, x, reach), _end_near(b, x, reach)
        meets = (
            (end_a is not None or _inside(a, x, 0.0))
            and (end_b is not None or _inside(b, x, 0.0))
        )
        if not meets or (end_a is None and end_b is None):
            continue
        for index, wall, end in ((i, a, end_a), (j, b, end_b)):
            if end is None:
                continue
            current = wall.end if end else wall.start
            distance = float(np.linalg.norm(current - x))
            key = (index, end)
            joined.add(key)
            if key not in moves or distance < moves[key][0]:
                moves[key] = (distance, x)

    for (index, end), (_, point) in moves.items():
        if end:
            walls[index].end = point
        else:
            walls[index].start = point
    for index, end in joined:
        walls[index].joined[end] = True
    _mark_column_joins(walls, columns, tolerance)


def _mark_column_joins(walls: list[WallDraft], columns: _ColumnIndex, tolerance: float) -> None:
    for wall in walls:
        for end, point in enumerate((wall.start, wall.end)):
            if not wall.joined[end] and columns.near(point, tolerance):
                wall.joined[end] = True


@dataclass
class WallResult:
    walls: list[WallDraft]
    pieces: list[Piece]
    peaks: list[float]
    curved: list[Primitive]


def recognize_walls(
    prims: list[Primitive],
    layers: LayerMap,
    columns: list[ColumnDraft],
    hatches: HatchIndex,
    cfg: dict[str, Any],
) -> WallResult:
    geometric = not layers.has_walls
    segments = collect_segments(prims, layers, columns, cfg)
    pieces: list[Piece] = []
    tolerance = math.radians(cfg["direction_tolerance_deg"])
    for group in group_by_direction(segments, tolerance):
        pieces.extend(find_pieces(group, cfg, geometric))

    _annotate_hatch(pieces, hatches, cfg)
    if geometric and pieces:
        # 해치가 그려진 도면이면 해치 밖의 쌍은 벽이 아닐 가능성이 크다.
        total = sum(p.length for p in pieces)
        covered = sum(_hatch_share(p) * p.length for p in pieces)
        if covered / total >= cfg["geometric_hatch_min_share"]:
            pieces = [p for p in pieces if _hatch_share(p) >= cfg["geometric_piece_min_share"]]

    peaks = thickness_peaks(pieces, cfg)
    snapped = _snap_thickness(pieces, peaks, cfg)

    walls: list[WallDraft] = []
    if snapped:
        for axis in _axis_groups(snapped, cfg):
            walls.extend(_merge_axis(axis, layers, cfg))
    walls = [w for w in walls if w.length >= cfg["min_wall_length_mm"]]
    walls = _drop_nested(walls, cfg)

    if not geometric:
        add_single_line_walls(walls, segments, cfg, layers)
    column_index = _ColumnIndex(columns)
    extend_to_anchors(walls, column_index, _LineIndex(prims, layers), cfg)
    clean_junctions(walls, column_index, cfg)

    curved = [
        p for p in prims
        if p.kind in ("arc", "ellipse") and not p.block_path and layers.role(p.layer) == "WALL"
    ]
    return WallResult(walls, [p for p, _ in snapped], peaks, curved)
