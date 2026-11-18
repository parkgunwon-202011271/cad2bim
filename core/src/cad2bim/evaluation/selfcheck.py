"""정답 없는 도면의 자동 검증 (상세설계 8.6). 사람의 판단을 쓰지 않는다.

- 대조: 도면 안의 해치, 벽 레이어, 치수를 인식 결과와 맞춰 본다.
- 불변성: 도면을 돌리거나 레이어 이름을 가려도 결과가 유지되는지 본다.

진짜 정확도가 아니라 추정치다. 해치나 레이어가 틀린 도면에서는 기준 자체가 틀린다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString
from shapely.ops import unary_union

from ..classify import curve_share as _curve_share
from ..ir import Wall
from ..layers import LayerInfo, LayerMap
from ..preprocess.primitive import Primitive
from ..recognize import recognize
from ..recognize.hatches import _hatch_polygon
from .metrics import stage_walls
from .truth import Truth, TruthWall

_ROTATION_DEG = 30.0
_SITE_CURVE_SHARE = 0.3
_SITE_MIN_EXTENT_M = 1000.0
_EXCLUDED_HATCH_ROLES = ("FURNITURE", "GRID", "STAIR", "OTHER")


@dataclass
class SelfCheck:
    rotation_recall: float | None = None
    rotation_precision: float | None = None
    masked_recall: float | None = None
    masked_precision: float | None = None
    hatch_explained: float | None = None
    wall_in_hatch: float | None = None
    wall_layer_coverage: float | None = None
    dim_on_element: float | None = None
    guess: str = "unknown"

    def line(self) -> str:
        def fmt(value: float | None) -> str:
            return "-" if value is None else f"{value:.3f}"

        return (
            f"selfcheck: rot_recall={fmt(self.rotation_recall)} rot_precision={fmt(self.rotation_precision)} "
            f"masked_recall={fmt(self.masked_recall)} masked_precision={fmt(self.masked_precision)} "
            f"hatch_explained={fmt(self.hatch_explained)} wall_in_hatch={fmt(self.wall_in_hatch)} "
            f"wall_layer_cov={fmt(self.wall_layer_coverage)} dim_on_element={fmt(self.dim_on_element)} "
            f"guess={self.guess}"
        )


def _as_truth(walls: list[Wall]) -> Truth:
    return Truth(walls=[TruthWall(w.id, w.start, w.end, w.thickness, w.kind) for w in walls])


def _rotated_point(p: tuple[float, float], angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c)


def _rotation_stability(
    prims: list[Primitive], walls: list[Wall], layers: LayerMap, cfg: dict, patterns: dict, thresholds: dict,
) -> tuple[float | None, float | None]:
    """돌린 도면에서 찾은 벽을 되돌려 원래 결과와 비교한다."""
    if not walls:
        return None, None
    angle = math.radians(_ROTATION_DEG)
    rotated = recognize([p.rotated(angle) for p in prims], layers, cfg, patterns, walls_only=True)
    back = [
        w.model_copy(update={"start": _rotated_point(w.start, -angle), "end": _rotated_point(w.end, -angle)})
        for w in rotated.walls
    ]
    score = stage_walls(back, _as_truth(walls), thresholds)
    return score["recall"], score["precision"]


def _masked_agreement(
    prims: list[Primitive], walls: list[Wall], layers: LayerMap, cfg: dict, patterns: dict, thresholds: dict,
) -> tuple[float | None, float | None]:
    """레이어 이름을 모르는 것처럼 돌려 비교한다. 벽 레이어가 있는 도면에서만 의미가 있다."""
    if not walls or not layers.has_walls:
        return None, None
    masked = LayerMap({name: LayerInfo("UNKNOWN", "unresolved") for name in layers.layers})
    found = recognize(prims, masked, cfg, patterns, walls_only=True)
    score = stage_walls(found.walls, _as_truth(walls), thresholds)
    return score["recall"], score["precision"]


def _solids(walls: list[Wall], columns: list[Any], margin: float) -> Any:
    shapes = [
        LineString([w.start, w.end]).buffer(w.thickness / 2 + margin, cap_style="flat")
        for w in walls if w.start != w.end
    ]
    for c in columns:
        half_w, half_d = c.size[0] / 2 + margin, c.size[1] / 2 + margin
        shapes.append(shapely.box(c.center[0] - half_w, c.center[1] - half_d,
                                  c.center[0] + half_w, c.center[1] + half_d))
    return unary_union(shapes) if shapes else None


def _hatch_agreement(
    prims: list[Primitive], layers: LayerMap, solids: Any, walls: list[Wall], patterns: dict
) -> tuple[float | None, float | None]:
    known = {name.upper() for names in patterns.values() for name in names}
    shapes = []
    for p in prims:
        if p.kind != "hatch" or layers.role(p.layer) in _EXCLUDED_HATCH_ROLES:
            continue
        if layers.role(p.layer) == "HATCH" or p.geometry["pattern"].upper() in known:
            shape = _hatch_polygon(p)
            if shape is not None and not shape.is_empty:
                shapes.append(shape)
    if not shapes or solids is None:
        return None, None
    hatch = unary_union(shapes)
    explained = hatch.intersection(solids).area / hatch.area if hatch.area else None
    in_hatch = None
    lines = [LineString([w.start, w.end]) for w in walls if w.start != w.end]
    if lines:
        total = sum(line.length for line in lines)
        in_hatch = sum(line.intersection(hatch).length for line in lines) / total
    return explained, in_hatch


def _wall_layer_coverage(prims: list[Primitive], layers: LayerMap, solids: Any) -> float | None:
    """벽 레이어 선분의 길이 중 인식한 벽 안에 드는 비율. 선분 중점으로 판정한다."""
    if not layers.has_walls or solids is None:
        return None
    segments = []
    for p in prims:
        if layers.role(p.layer) != "WALL":
            continue
        if p.kind == "line":
            segments.append((p.geometry["p1"], p.geometry["p2"]))
        elif p.kind == "polyline":
            points = list(p.geometry["points"])
            if p.geometry["closed"] and points:
                points.append(points[0])
            segments.extend(zip(points, points[1:]))
    if not segments:
        return None
    array = np.array(segments, dtype=float)
    lengths = np.linalg.norm(array[:, 1] - array[:, 0], axis=1)
    middle = array.mean(axis=1)
    shapely.prepare(solids)
    inside = shapely.contains_xy(solids, middle[:, 0], middle[:, 1])
    return float(lengths[inside].sum() / lengths.sum()) if lengths.sum() else None


def _dim_on_element(prims: list[Primitive], solids: Any, margin: float) -> float | None:
    """치수의 측정점이 벽이나 기둥에 놓인 비율. 치수는 보통 구조체에 건다."""
    points = [pt for p in prims if p.kind == "dimension" for pt in (p.geometry["p1"], p.geometry["p2"])]
    if not points or solids is None:
        return None
    array = np.array(points, dtype=float)
    return float(shapely.dwithin(shapely.points(array), solids, margin).mean())


def guess_type(walls: list[Wall], extent_m: float, loops: int, dangling: int, curve_share: float) -> str:
    """도면 종류 추정. 임계값은 아직 검증되지 않은 초기값이다. 결과에 반영하지 않고 표시만 한다.

    대형 물류센터는 수백 m라 크기만으로 배치도라고 볼 수 없다.
    """
    count = len(walls)
    if count == 0:
        return "no_walls"
    if curve_share >= _SITE_CURVE_SHARE or extent_m >= _SITE_MIN_EXTENT_M:
        return "site_like"
    if loops >= count:
        return "table_like"
    if dangling >= count * 0.6:
        return "detail_like"
    return "plan_like"


def run_self_check(
    result: Any, cfg: dict, thresholds: dict, patterns: dict, loops: int
) -> SelfCheck:
    prims, layers, ir = result.pre.primitives, result.layers, result.ir
    check = SelfCheck()
    check.rotation_recall, check.rotation_precision = _rotation_stability(
        prims, ir.walls, layers, cfg, patterns, thresholds
    )
    check.masked_recall, check.masked_precision = _masked_agreement(
        prims, ir.walls, layers, cfg, patterns, thresholds
    )
    margin = cfg["walls"]["junction_tolerance_mm"]
    solids = _solids(ir.walls, ir.columns, margin)
    check.hatch_explained, check.wall_in_hatch = _hatch_agreement(prims, layers, solids, ir.walls, patterns)
    check.wall_layer_coverage = _wall_layer_coverage(prims, layers, solids)
    check.dim_on_element = _dim_on_element(prims, solids, margin)

    bbox = result.pre.plan_bbox
    extent = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 1000 if bbox else 0.0
    dangling = sum("dangling_end" in w.confidence.reasons for w in ir.walls)
    check.guess = guess_type(ir.walls, extent, loops, dangling, _curve_share(prims))
    return check


__all__ = ["SelfCheck", "guess_type", "run_self_check"]
