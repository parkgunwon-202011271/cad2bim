"""뚫린 곳. 사각형에 X자 대각선을 긋는 것은 보이드와 승강로를 그리는 CAD 관례다."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..layers import LayerMap
from ..preprocess.primitive import Primitive

_EXCLUDED = ("FURNITURE", "DIM", "GRID", "DOOR", "WINDOW", "HATCH")


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


def recognize_voids(prims: list[Primitive], layers: LayerMap, cfg: dict[str, Any]) -> list[VoidDraft]:
    """같은 사각형의 두 대각선을 찾는다. 끝점이 모서리에 맞아야 한다."""
    tolerance = cfg["corner_tol_mm"]
    lines = _diagonals(prims, layers, cfg["min_side_mm"])
    keyed: dict[tuple[int, int], list[tuple[Any, ...]]] = {}
    for line in lines:
        a, b, _ = line
        center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        keyed.setdefault((round(center[0] / (tolerance * 4)), round(center[1] / (tolerance * 4))), []).append(line)

    voids: list[VoidDraft] = []
    for key, members in keyed.items():
        candidates = list(members)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    candidates.extend(keyed.get((key[0] + dx, key[1] + dy), []))
        for first in members:
            a, b, p = first
            slope = (b[1] - a[1]) / (b[0] - a[0])
            for second in candidates:
                c, d, q = second
                if q is p or (d[1] - c[1]) / (d[0] - c[0]) * slope >= 0:
                    continue
                x0, x1 = min(a[0], b[0]), max(a[0], b[0])
                y0, y1 = min(a[1], b[1]), max(a[1], b[1])
                if (
                    abs(min(c[0], d[0]) - x0) <= tolerance and abs(max(c[0], d[0]) - x1) <= tolerance
                    and abs(min(c[1], d[1]) - y0) <= tolerance and abs(max(c[1], d[1]) - y1) <= tolerance
                ):
                    bounds = (x0, y0, x1, y1)
                    if not any(all(math.isclose(u, v, abs_tol=tolerance) for u, v in zip(bounds, v_.bounds)) for v_ in voids):
                        voids.append(VoidDraft(bounds, set(p.source_handles) | set(q.source_handles)))
                    break
    return voids
