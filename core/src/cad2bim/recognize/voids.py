"""뚫린 곳. 사각형에 X자 대각선을 긋는 것은 보이드와 승강로를 그리는 CAD 관례다."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..layers import LayerMap
from ..preprocess.primitive import Primitive

# 격자·중심선 레이어는 빼지 않는다. X자 OPEN 표시를 중심선 레이어에 긋는 사무소가 있다(격자선은 X자로 겹치지 않는다).
_EXCLUDED = ("FURNITURE", "DIM", "DOOR", "WINDOW", "HATCH")


@dataclass
class VoidDraft:
    bounds: tuple[float, float, float, float]
    handles: set[str]


def _diagonals(prims: list[Primitive], layers: LayerMap, min_side: float) -> list[tuple[Any, ...]]:
    found = []
    for p in prims:
        if layers.role(p.layer) in _EXCLUDED:
            continue
        if p.kind == "line":
            a, b = p.geometry["p1"], p.geometry["p2"]
        elif p.kind == "polyline" and len(p.geometry["points"]) == 2 and not p.geometry["closed"]:
            a, b = p.geometry["points"]
        else:
            continue
        if abs(a[0] - b[0]) >= min_side and abs(a[1] - b[1]) >= min_side:
            found.append((a, b, p))
    return found


def _crosses(a, b, c, d) -> bool:
    """두 선분이 서로 가로지르는가."""
    def side(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0


def _iou(u, v) -> float:
    w = min(u[2], v[2]) - max(u[0], v[0])
    h = min(u[3], v[3]) - max(u[1], v[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((u[2] - u[0]) * (u[3] - u[1]) + (v[2] - v[0]) * (v[3] - v[1]) - inter)


def recognize_voids(prims: list[Primitive], layers: LayerMap, cfg: dict[str, Any]) -> list[VoidDraft]:
    """같은 사각형의 두 대각선을 찾는다. 끝점이 모서리에 맞거나, 서로 가로지르며 범위가 많이 겹쳐야 한다.

    모서리 허용 오차는 X자가 클수록 커진다. 큰 OPEN 표시는 손으로 그어 모서리에서 수백 mm 어긋나기도 한다.
    """
    base_tol = cfg["corner_tol_mm"]
    share = cfg.get("corner_tol_share", 0.0)
    lines = _diagonals(prims, layers, cfg["min_side_mm"])
    voids: list[VoidDraft] = []
    for i, (a, b, p) in enumerate(lines):
        slope = (b[1] - a[1]) / (b[0] - a[0])
        length = math.dist(a, b)
        for c, d, q in lines[i + 1:]:
            if q is p or (d[1] - c[1]) / (d[0] - c[0]) * slope >= 0:
                continue
            tolerance = max(base_tol, share * min(length, math.dist(c, d)))
            first = (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
            second = (min(c[0], d[0]), min(c[1], d[1]), max(c[0], d[0]), max(c[1], d[1]))
            square = all(abs(u - v) <= tolerance for u, v in zip(first, second))
            # 모양이 고르지 않은 실은 모서리에서 모서리로 X를 긋는다. 서로 가로지르고 범위가 많이 겹치면 받는다.
            if not square and not (_crosses(a, b, c, d) and _iou(first, second) >= cfg.get("min_overlap_iou", 2.0)):
                continue
            bounds = (min(first[0], second[0]), min(first[1], second[1]), max(first[2], second[2]), max(first[3], second[3]))
            if not any(all(math.isclose(u, v, abs_tol=tolerance) for u, v in zip(bounds, v_.bounds)) for v_ in voids):
                voids.append(VoidDraft(bounds, set(p.source_handles) | set(q.source_handles)))
    return voids
