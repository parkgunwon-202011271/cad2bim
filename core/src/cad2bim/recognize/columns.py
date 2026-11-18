"""기둥 인식. 벽보다 먼저 한다. 기둥 외곽선이 벽 후보에 섞이지 않게 하기 위해서다."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon

from ..layers import LayerMap
from ..preprocess.primitive import Primitive
from .hatches import HatchIndex


@dataclass
class ColumnDraft:
    shape: str
    center: tuple[float, float]
    size: tuple[float, float]
    rotation: float
    polygon: Polygon
    prim_ids: set[str]
    handles: list[str]
    layer: str
    hatched: bool
    inferred: bool = False  # 레이어나 채움 없이 배치 규칙만으로 찾은 기둥. 항상 검토 대상


def _rect_measure(polygon: Polygon) -> tuple[tuple[float, float], float] | None:
    """사각형이면 ((폭, 깊이), 회전)."""
    if len(polygon.exterior.coords) != 5:
        return None
    box = polygon.minimum_rotated_rectangle
    if polygon.area < box.area * 0.98:
        return None
    pts = list(polygon.exterior.coords)
    w = math.dist(pts[0], pts[1])
    d = math.dist(pts[1], pts[2])
    return (w, d), math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]) % (math.pi / 2)


def _polygon_candidate(p: Primitive, cfg: dict[str, Any]) -> tuple[str, Polygon, tuple[float, float], float] | None:
    """사각형이 아닌 닫힌 도형. 모서리를 딴 ㄱ자 기둥처럼 꽉 찬 모양만 받는다."""
    if p.kind != "polyline" or not p.geometry["closed"]:
        return None
    pts = list(p.geometry["points"])
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if not 5 <= len(pts) <= cfg["polygon_max_points"]:
        return None
    polygon = Polygon(pts)
    if not polygon.is_valid or polygon.area == 0:
        return None
    box = polygon.minimum_rotated_rectangle
    if polygon.area < box.area * cfg["polygon_min_fill"]:
        return None
    ring = list(box.exterior.coords)
    size = (math.dist(ring[0], ring[1]), math.dist(ring[1], ring[2]))
    return "polygon", polygon, size, 0.0


def _candidate(p: Primitive) -> tuple[str, Polygon, tuple[float, float], float] | None:
    if p.kind == "circle":
        r = p.geometry["radius"]
        return "circle", Point(p.geometry["center"]).buffer(r), (2 * r, 2 * r), 0.0
    if p.kind == "polyline" and p.geometry["closed"]:
        pts = p.geometry["points"]
    elif p.kind == "solid":
        pts = p.geometry["points"]
    elif p.kind == "hatch" and len(p.geometry["rings"]) == 1:
        pts = p.geometry["rings"][0]
    else:
        return None
    pts = list(pts)
    if len(pts) == 5 and pts[0] == pts[4]:
        pts = pts[:4]
    if len(pts) != 4:
        return None
    polygon = Polygon(pts)
    if not polygon.is_valid or polygon.area == 0:
        return None
    measured = _rect_measure(polygon)
    return None if measured is None else ("rect", polygon, measured[0], measured[1])


def _on_grid(centers: np.ndarray, tolerance: float, min_share: float, min_spacing: float) -> bool:
    """같은 x열과 같은 y행에 다른 기둥이 있고, 서로 충분히 떨어져 있는가.

    랙 기둥이나 가구도 반복되지만 간격이 좁다. 건물 기둥 간격은 수 m 이상이다.
    """
    dx = np.abs(centers[:, None, 0] - centers[None, :, 0])
    dy = np.abs(centers[:, None, 1] - centers[None, :, 1])
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    same_column = (dx <= tolerance).any(axis=1)
    same_row = (dy <= tolerance).any(axis=1)
    if float((same_column & same_row).mean()) < min_share:
        return False
    distance = np.hypot(dx, dy)
    return float(np.median(distance.min(axis=1))) >= min_spacing


def _nearest_spacing(centers: np.ndarray) -> np.ndarray:
    distance = np.hypot(
        centers[:, None, 0] - centers[None, :, 0], centers[:, None, 1] - centers[None, :, 1]
    )
    np.fill_diagonal(distance, np.inf)
    return distance.min(axis=1)


def _is_column_line(centers: np.ndarray, cfg: dict[str, Any]) -> bool:
    """가로세로 격자가 아니라도 일정한 간격으로 줄지어 있으면 기둥 후보로 본다. 외곽 기둥이 그렇다."""
    if len(centers) < cfg["line_min_count"]:
        return False
    tolerance = cfg["grid_align_mm"]
    dx = np.abs(centers[:, None, 0] - centers[None, :, 0])
    dy = np.abs(centers[:, None, 1] - centers[None, :, 1])
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    aligned = ((dx <= tolerance).any(axis=1) | (dy <= tolerance).any(axis=1)).mean()
    gaps = _gaps_along_lines(centers, tolerance)
    if len(gaps) < 2 or float(aligned) < cfg["line_min_share"]:
        return False
    # 기둥이 빠진 칸이 있으면 간격이 기본 간격의 배수가 된다. 실제 도면의 11m와 22m가 그랬다.
    base = float(gaps.min())
    if base < cfg["grid_min_spacing_mm"]:
        return False
    ratios = gaps / base
    regular = np.abs(ratios - np.round(ratios)) <= cfg["line_multiple_tol"]
    return float(regular.mean()) >= cfg["line_min_regular_share"]


def _gaps_along_lines(centers: np.ndarray, tolerance: float) -> np.ndarray:
    """같은 x열, 같은 y행 안에서 이웃한 기둥 사이의 간격. 건물 기둥은 이 값이 일정하다."""
    gaps: list[float] = []
    for axis in (0, 1):
        other = 1 - axis
        order = np.argsort(centers[:, other])
        start = 0
        for i in range(1, len(order) + 1):
            if i == len(order) or centers[order[i], other] - centers[order[i - 1], other] > tolerance:
                row = np.sort(centers[order[start:i], axis])
                if len(row) >= 2:
                    gaps.extend(np.diff(row).tolist())
                start = i
    return np.array(gaps)


def _on_grid_lines(centers: np.ndarray, lines: np.ndarray, cfg: dict[str, Any]) -> bool:
    """이미 찾은 격자의 x선이나 y선에 놓였는가. 외곽 기둥은 중심이 조금 어긋나 허용 범위를 넓게 둔다."""
    tolerance = cfg["grid_line_tol_mm"]
    near_x = np.abs(centers[:, None, 0] - lines[None, :, 0]).min(axis=1) <= tolerance
    near_y = np.abs(centers[:, None, 1] - lines[None, :, 1]).min(axis=1) <= tolerance
    if float((near_x | near_y).mean()) < cfg["line_min_share"]:
        return False
    return float(np.median(_nearest_spacing(centers))) >= cfg["grid_min_spacing_mm"]


def _infer_from_grid(
    prims: list[Primitive], layers: LayerMap, seen: set[tuple[int, int]], cfg: dict[str, Any]
) -> list[ColumnDraft]:
    """레이어로 못 찾을 때, 같은 크기의 사각형이 격자로 반복되면 기둥 후보로 본다."""
    groups: dict[tuple[int, int], list[tuple[Primitive, Any]]] = {}
    for p in prims:
        if layers.role(p.layer) not in ("UNKNOWN", "WALL") or p.kind != "polyline":
            continue
        found = _candidate(p)
        if found is None or found[0] != "rect":
            continue
        size = found[2]
        if not all(cfg["grid_min_size_mm"] <= s <= cfg["max_size_mm"] for s in size):
            continue
        if max(size) > min(size) * cfg["grid_max_aspect"]:
            continue
        key = tuple(sorted((round(size[0] / 10), round(size[1] / 10))))
        groups.setdefault(key, []).append((p, found))

    # 기둥은 크기가 섞이고 외곽을 따라 줄지어 있기도 하다. 먼저 뚜렷한 묶음으로 격자를 찾고,
    # 그 격자선에 놓인 다른 크기를 더한다. 간격이 좁은 랙 기둥은 어느 단계에서도 통과하지 못한다.
    usable = {
        key: (members, np.array([(f[1].centroid.x, f[1].centroid.y) for _, f in members]))
        for key, members in groups.items()
        if 2 <= len(members) <= cfg["grid_max_count"]
    }
    seeds = {
        key for key, (members, centers) in usable.items()
        if len(members) >= cfg["grid_min_count"]
        and _on_grid(centers, cfg["grid_align_mm"], cfg["grid_min_share"], cfg["grid_min_spacing_mm"])
    }
    if not seeds:
        seeds = {key for key, (members, centers) in usable.items() if _is_column_line(centers, cfg)}
    # 벽은 같은 크기의 닫힌 사각형으로 반복되지 않는다. 벽 레이어에서 떨어져 반복되면 기둥으로 본다.
    seeds |= {
        key for key, (members, centers) in usable.items()
        if len(members) >= cfg["line_min_count"]
        and all(layers.role(p.layer) == "WALL" for p, _ in members)
        and float(np.median(_nearest_spacing(centers))) >= cfg["grid_min_spacing_mm"]
    }
    accepted = set(seeds)
    if seeds:
        lines = np.concatenate([usable[key][1] for key in seeds])
        for key, (_, centers) in usable.items():
            if key not in accepted and _on_grid_lines(centers, lines, cfg):
                accepted.add(key)

    drafts = []
    # 같은 자리에 크기가 다른 사각형이 겹치면 큰 쪽을 남긴다. 어느 쪽이 구조체 면인지는 도면만으로 알 수 없다.
    for key in sorted(accepted, key=lambda k: (-k[0] * k[1], k)):
        members, centers = usable[key]
        for (p, (shape, polygon, size, rotation)), center in zip(members, centers.tolist()):
            position = (round(center[0]), round(center[1]))
            if position in seen:
                continue
            seen.add(position)
            drafts.append(ColumnDraft(
                shape, tuple(center), size, rotation, polygon, {p.id}, list(p.source_handles),
                p.layer, False, inferred=True,
            ))
    return drafts


def recognize_columns(
    prims: list[Primitive], layers: LayerMap, hatches: HatchIndex, cfg: dict[str, Any]
) -> list[ColumnDraft]:
    drafts: list[ColumnDraft] = []
    seen: set[tuple[int, int]] = set()
    polygons: list[tuple[Polygon, Primitive, tuple[float, float], bool]] = []
    for p in prims:
        role = layers.role(p.layer)
        if role not in ("COLUMN", "WALL", "UNKNOWN"):
            continue
        # 해치만으로는 기둥 레이어에서만 믿는다. 개구부 사이의 벽 해치도 작은 사각형이다.
        if p.kind == "hatch" and role != "COLUMN":
            continue
        found = _candidate(p)
        if found is None and role == "COLUMN":
            found = _polygon_candidate(p, cfg)
        if found is None:
            continue
        shape, polygon, size, rotation = found
        if not all(cfg["min_size_mm"] <= s <= cfg["max_size_mm"] for s in size):
            continue
        if max(size) > min(size) * cfg["max_aspect"]:
            continue
        filled = p.kind in ("solid", "hatch")
        hatched = filled or hatches.covers(polygon, cfg["hatch_min_iou"])
        # 기둥 레이어가 아니면 채움이 있어야 믿는다. 가구·설비 사각형을 거르기 위해서다.
        if role != "COLUMN" and not hatched:
            continue
        center = (polygon.centroid.x, polygon.centroid.y)
        key = (round(center[0]), round(center[1]))
        if key in seen:  # 같은 기둥을 외곽선과 해치로 함께 그린 경우
            continue
        if shape == "polygon":
            polygons.append((polygon, p, size, hatched))
            continue
        seen.add(key)
        drafts.append(ColumnDraft(
            shape, center, size, rotation, polygon,
            {p.id}, list(p.source_handles), p.layer, hatched,
        ))
    # 사각형이 아닌 기둥은 외곽선과 안쪽 선을 겹쳐 그리는 일이 많다. 큰 것부터 받고 절반 넘게 겹치면 버린다.
    for polygon, p, size, hatched in sorted(polygons, key=lambda item: -item[0].area):
        if any(polygon.intersection(d.polygon).area >= polygon.area * cfg["polygon_max_overlap"] for d in drafts):
            continue
        centroid = polygon.centroid
        seen.add((round(centroid.x), round(centroid.y)))
        drafts.append(ColumnDraft(
            "polygon", (centroid.x, centroid.y), size, 0.0, polygon, {p.id}, list(p.source_handles), p.layer, hatched,
        ))
    # 기둥 레이어가 제대로 쓰인 도면에서 다른 레이어의 사각형은 대개 짧은 벽 조각이다.
    on_layer = [d for d in drafts if layers.role(d.layer) == "COLUMN"]
    if len(on_layer) >= cfg["layer_trust_min_count"]:
        return on_layer
    return drafts + _infer_from_grid(prims, layers, seen, cfg)
