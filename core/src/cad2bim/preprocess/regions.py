"""도곽·표제란을 빼고 평면 영역을 고른다. 평면이 둘 이상이면 자동 선택하지 않는다.

닫힌 사각형만으로 도곽을 판정하면 외벽(B1처럼 닫힌 폴리선)을 지울 수 있다.
그래서 표제란이 안쪽 모서리에 붙어 있거나, 치수까지 감싸고 거의 모든 요소를
담는 경우에만 도곽으로 본다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..errors import RegionAmbiguousError
from .primitive import BBox, Primitive, union_bbox

_EDGE_TOL = 1.0
_CLUSTER_BATCH = 20000
_INNER_FRAME_MIN_AREA = 0.5
_TITLE_KINDS = frozenset({"line", "polyline", "text"})


def _rect_of(p: Primitive) -> BBox | None:
    if p.kind != "polyline" or not p.geometry["closed"]:
        return None
    pts = list(p.geometry["points"])
    if len(pts) == 5 and pts[0] == pts[4]:
        pts = pts[:4]
    if len(pts) != 4:
        return None
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        if abs(x1 - x2) > _EDGE_TOL and abs(y1 - y2) > _EDGE_TOL:
            return None
    return p.bbox()


def _inside(inner: BBox, outer: BBox) -> bool:
    return (
        inner[0] >= outer[0] - _EDGE_TOL
        and inner[1] >= outer[1] - _EDGE_TOL
        and inner[2] <= outer[2] + _EDGE_TOL
        and inner[3] <= outer[3] + _EDGE_TOL
    )


def _area(b: BBox) -> float:
    return (b[2] - b[0]) * (b[3] - b[1])


def _in_corner(inner: BBox, outer: BBox) -> bool:
    x_edge = abs(inner[0] - outer[0]) <= _EDGE_TOL or abs(inner[2] - outer[2]) <= _EDGE_TOL
    y_edge = abs(inner[1] - outer[1]) <= _EDGE_TOL or abs(inner[3] - outer[3]) <= _EDGE_TOL
    return x_edge and y_edge


class _Drawn:
    """그려진 요소의 범위 배열. 큰 도면에서 포함 검사를 빠르게 한다."""

    def __init__(self, prims: list[Primitive]):
        self.items = [(p, b) for p in prims if (b := p.bbox()) is not None]
        self.boxes = np.array([b for _, b in self.items], dtype=float).reshape(-1, 4)

    def inside(self, outer: BBox) -> np.ndarray:
        """outer 안에 든 요소의 번호."""
        b = self.boxes
        mask = (
            (b[:, 0] >= outer[0] - _EDGE_TOL) & (b[:, 1] >= outer[1] - _EDGE_TOL)
            & (b[:, 2] <= outer[2] + _EDGE_TOL) & (b[:, 3] <= outer[3] + _EDGE_TOL)
        )
        return np.flatnonzero(mask)


def _find_title_box(
    rect: Primitive,
    outer: BBox,
    rects: list[tuple[Primitive, BBox]],
    drawn: _Drawn,
    max_area_ratio: float,
) -> tuple[Primitive, BBox] | None:
    for candidate, b in rects:
        if candidate is rect or not _inside(b, outer) or not _in_corner(b, outer):
            continue
        if _area(b) > _area(outer) * max_area_ratio:
            continue
        content = [
            p for p in (drawn.items[int(i)][0] for i in drawn.inside(b))
            if p is not candidate and p is not rect
        ]
        if content and any(p.kind == "text" for p in content) and all(
            p.kind in _TITLE_KINDS for p in content
        ):
            return candidate, b
    return None


def _inner_frame_lines(rects: list[tuple[Primitive, BBox]], frame: BBox) -> set[int]:
    """도곽 안쪽에 겹으로 그린 큰 사각형도 도곽선이다."""
    return {
        id(p) for p, b in rects
        if _inside(b, frame) and _area(b) >= _area(frame) * _INNER_FRAME_MIN_AREA
    }


def _edges(p: Primitive) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if p.linetype.upper() not in ("CONTINUOUS", "BYLAYER"):
        return []
    if p.kind == "line":
        return [(p.geometry["p1"], p.geometry["p2"])]
    if p.kind == "polyline":
        points = list(p.geometry["points"])
        if p.geometry["closed"] and points:
            points.append(points[0])
        return list(zip(points, points[1:]))
    return []


def _drawing_area(inside: list[Primitive], frame: BBox, cfg: dict[str, Any]) -> BBox:
    """도곽을 가로지르는 긴 선으로 안쪽 그림 영역을 찾는다. 표제란 띠는 그 바깥에 남는다."""
    x0, y0, x1, y1 = frame
    width, height = x1 - x0, y1 - y0
    span, band = cfg["frame_span_share"], cfg["frame_band_share"]
    left, right, bottom, top = x0, x1, y0, y1
    for p in inside:
        for a, b in _edges(p):
            if abs(a[0] - b[0]) <= _EDGE_TOL and abs(a[1] - b[1]) >= height * span:
                if a[0] < x0 + width * band:
                    left = max(left, a[0])
                elif a[0] > x1 - width * band:
                    right = min(right, a[0])
            elif abs(a[1] - b[1]) <= _EDGE_TOL and abs(a[0] - b[0]) >= width * span:
                if a[1] < y0 + height * band:
                    bottom = max(bottom, a[1])
                elif a[1] > y1 - height * band:
                    top = min(top, a[1])
    return (left, bottom, right, top)


def _sheet_furniture(rect: Primitive, frame: BBox, drawn: _Drawn, cfg: dict[str, Any]) -> set[int]:
    """그림 영역 밖의 요소와, 도곽 블록에 함께 든 요소. 평면까지 지우지 않게 비율로 막는다."""
    indices = drawn.inside(frame)
    inside = [drawn.items[int(i)] for i in indices]
    if not inside:
        return set()
    area = _drawing_area([p for p, _ in inside], frame, cfg)
    found: set[int] = set()
    for p, b in inside:
        outside_area = (
            b[2] <= area[0] + _EDGE_TOL or b[0] >= area[2] - _EDGE_TOL
            or b[3] <= area[1] + _EDGE_TOL or b[1] >= area[3] - _EDGE_TOL
        )
        same_block = bool(rect.block_path) and p.source_handles == rect.source_handles
        if outside_area or same_block:
            found.add(id(p))
    if len(found) > len(inside) * cfg["frame_clip_max_share"]:
        return set()
    return found


def _sheet_row(
    outers: list[tuple[Primitive, BBox]], drawn: _Drawn, is_dimension: np.ndarray, cfg: dict[str, Any]
) -> list[tuple[Primitive, BBox]]:
    """한 파일에 나란히 놓인 같은 크기의 도곽들.

    표제란이 사각형이 아니면 도곽 한 장으로는 알아볼 수 없고, 한 장이 전체 내용을 담지도 못한다.
    같은 크기이고 겹치지 않으며 저마다 치수가 있고, 합쳐서 도면 내용 대부분을 담을 때만 도곽으로 본다.
    """
    tolerance = cfg["sheet_size_tolerance"]
    groups: list[list[tuple[Primitive, BBox]]] = []
    for item in sorted(outers, key=lambda o: _area(o[1])):
        b = item[1]
        for group in groups:
            g = group[0][1]
            if abs((b[2] - b[0]) / (g[2] - g[0]) - 1) <= tolerance and abs((b[3] - b[1]) / (g[3] - g[1]) - 1) <= tolerance:
                group.append(item)
                break
        else:
            groups.append([item])

    result: list[tuple[Primitive, BBox]] = []
    for group in groups:
        if len(group) < 2:
            continue
        boxes = [b for _, b in group]
        overlap = any(
            min(a[2], c[2]) - max(a[0], c[0]) > _EDGE_TOL and min(a[3], c[3]) - max(a[1], c[1]) > _EDGE_TOL
            for i, a in enumerate(boxes) for c in boxes[i + 1:]
        )
        if overlap:
            continue
        members = [drawn.inside(b) for b in boxes]
        if not all(is_dimension[m].any() for m in members):
            continue
        covered = np.unique(np.concatenate(members)).size
        if covered >= cfg["frame_min_content_share"] * len(drawn.items):
            result.extend(group)
    return result


def strip_frames(
    prims: list[Primitive], cfg: dict[str, Any]
) -> tuple[list[Primitive], list[Primitive], list[BBox]]:
    """(남은 요소, 제외한 요소, 도곽 범위)를 돌려준다."""
    drawn = _Drawn(prims)
    rects = [(p, r) for p, _ in drawn.items if (r := _rect_of(p)) is not None and _area(r) > 0]
    # 도곽은 크다. 기둥·가구 같은 작은 사각형은 바깥 후보에서 뺀다.
    min_side = cfg["min_frame_side_mm"]
    outers = [(p, r) for p, r in rects if min(r[2] - r[0], r[3] - r[1]) >= min_side]
    is_dimension = np.array([p.kind == "dimension" for p, _ in drawn.items], dtype=bool)

    found: list[tuple[Primitive, BBox, tuple[Primitive, BBox] | None]] = []
    for rect, r in outers:
        title = _find_title_box(rect, r, rects, drawn, cfg["title_box_max_area_ratio"])
        if title is not None:
            found.append((rect, r, title))
            continue
        inside = drawn.inside(r)
        others = inside.size - 1  # 자기 자신은 뺀다
        if others >= cfg["frame_min_content_share"] * (len(drawn.items) - 1) and is_dimension[inside].any():
            found.append((rect, r, None))

    found.extend(
        (rect, r, None) for rect, r in _sheet_row(outers, drawn, is_dimension, cfg)
        if not any(rect is f[0] for f in found)
    )

    # 도곽 안의 사각형(외벽 등)은 도곽이 아니다.
    frames = [
        f for f in found
        if not any(o is not f and _inside(f[1], o[1]) and _area(o[1]) > _area(f[1]) for o in found)
    ]

    removed: set[int] = set()
    for rect, r, title in frames:
        removed.add(id(rect))
        removed.update(_inner_frame_lines(rects, r))
        if title is not None:
            title_rect, tb = title
            removed.add(id(title_rect))
            removed.update(id(drawn.items[int(i)][0]) for i in drawn.inside(tb))
        removed.update(_sheet_furniture(rect, r, drawn, cfg))

    kept = [p for p in prims if id(p) not in removed]
    excluded = [p for p in prims if id(p) in removed]
    return kept, excluded, [f[1] for f in frames]


def _cluster(drawn: list[Primitive], gap: float, frames: list[BBox]) -> list[list[Primitive]]:
    """가까운 요소끼리 묶는다. 다른 도곽에 든 요소는 가까워도 합치지 않는다."""
    boxes = np.array([p.bbox() for p in drawn], dtype=float)
    home = np.full(len(drawn), -1)
    for index in range(len(frames) - 1, -1, -1):
        f = frames[index]
        home[
            (boxes[:, 0] >= f[0] - _EDGE_TOL) & (boxes[:, 1] >= f[1] - _EDGE_TOL)
            & (boxes[:, 2] <= f[2] + _EDGE_TOL) & (boxes[:, 3] <= f[3] + _EDGE_TOL)
        ] = index

    tree = shapely.STRtree(shapely.box(boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]))
    queries = shapely.box(boxes[:, 0] - gap, boxes[:, 1] - gap, boxes[:, 2] + gap, boxes[:, 3] + gap)
    labels = np.arange(len(drawn))
    # 긴 그리드선은 후보 쌍이 많다. 나눠서 합쳐 메모리가 쌍 수만큼 늘지 않게 한다.
    for begin in range(0, len(drawn), _CLUSTER_BATCH):
        left, right = tree.query(queries[begin:begin + _CLUSTER_BATCH])
        left = left + begin
        keep = (home[left] == home[right]) & (labels[left] != labels[right])
        if not keep.any():
            continue
        graph = coo_matrix(
            (np.ones(int(keep.sum()), dtype=bool), (labels[left[keep]], labels[right[keep]])),
            shape=(len(drawn), len(drawn)),
        )
        _, components = connected_components(graph, directed=False)
        labels = components[labels]

    groups: dict[int, list[Primitive]] = {}
    for label, p in zip(labels.tolist(), drawn):
        groups.setdefault(label, []).append(p)
    return list(groups.values())


def _drawn_lines(group: list[Primitive]) -> int:
    """평면은 선으로 그린다. 문자만 모인 주기 묶음을 평면으로 보면 벽이 통째로 빠진다."""
    return sum(p.kind in ("line", "polyline") for p in group)


def _is_plan(group: list[Primitive], cfg: dict[str, Any]) -> bool:
    if _drawn_lines(group) < cfg["min_plan_primitives"]:
        return False
    b = union_bbox(group)
    diagonal = ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5
    return diagonal >= cfg["min_plan_diagonal_mm"]


def select_plan(
    prims: list[Primitive],
    cfg: dict[str, Any],
    frames: list[BBox],
    region: BBox | None = None,
) -> tuple[list[Primitive], list[Primitive]]:
    """(평면 요소, 평면 밖 요소)를 돌려준다. region은 mm 단위다."""
    if region is not None:
        kept, outside = [], []
        for p in prims:
            b = p.extent()
            hit = b is not None and (
                b[0] <= region[2] and b[2] >= region[0] and b[1] <= region[3] and b[3] >= region[1]
            )
            (kept if hit else outside).append(p)
        return kept, outside

    drawn = [p for p in prims if p.bbox() is not None]
    if not drawn:
        return [], list(prims)
    groups = _cluster(drawn, cfg["cluster_gap_mm"], frames)
    plans = [g for g in groups if _is_plan(g, cfg)] or [max(groups, key=_drawn_lines)]
    # 로고나 작은 안내도 묶음이 평면 후보가 되지 않게 한다. 실제 도면에서 요소 35개짜리가 나왔다.
    largest = max(_drawn_lines(g) for g in plans)
    plans = [g for g in plans if _drawn_lines(g) >= largest * cfg["plan_min_share_of_largest"]]
    # 큰 홀 안의 가구 묶음처럼 다른 평면 범위 안에 있는 묶음은 별도 평면이 아니다.
    boxes = [union_bbox(g) for g in plans]
    plans = [
        g for g, b in zip(plans, boxes)
        if not any(o is not b and _inside(b, o) and _area(o) > _area(b) for o in boxes)
    ]
    if len(plans) > 1:
        candidates = sorted(union_bbox(g) for g in plans)
        listing = ", ".join(",".join(f"{v:.0f}" for v in c) for c in candidates)
        raise RegionAmbiguousError(
            f"평면 후보가 {len(plans)}개입니다. --region으로 하나를 지정하세요. "
            f"후보(mm, x1,y1,x2,y2): {listing}",
            candidates,
        )

    chosen = plans[0]
    bounds = union_bbox(chosen)
    chosen_ids = {id(p) for p in chosen}
    kept, outside = [], []
    for p in prims:
        # 벽에서 먼 독립 기둥이나 실명도 평면 범위 안이면 평면의 일부다.
        b = p.extent()
        if id(p) in chosen_ids or (b is not None and _inside(b, bounds)):
            kept.append(p)
        else:
            outside.append(p)
    return kept, outside
