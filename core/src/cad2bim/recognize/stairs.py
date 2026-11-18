"""계단. 계단 레이어에서 일정한 간격으로 반복되는 평행선을 단으로 본다.

평면에서는 오르는 방향을 확실히 알 수 없어 항상 검토 대상이다. 레빗과 스케치업에서 고칠 수 있는 근사값이다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..layers import LayerMap
from ..preprocess.primitive import Primitive


@dataclass
class FlightDraft:
    start: tuple[float, float]  # 첫 단의 중심
    end: tuple[float, float]  # 끝 단의 중심
    width: float
    treads: int
    handles: set[str] = field(default_factory=set)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        ax, ay = self.start
        bx, by = self.end
        length = max(math.hypot(bx - ax, by - ay), 1e-9)
        nx, ny = -(by - ay) / length, (bx - ax) / length
        hw = self.width / 2
        xs = [ax + nx * hw, ax - nx * hw, bx + nx * hw, bx - nx * hw]
        ys = [ay + ny * hw, ay - ny * hw, by + ny * hw, by - ny * hw]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class StairDraft:
    flights: list[FlightDraft]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        boxes = np.array([f.bounds for f in self.flights])
        return (float(boxes[:, 0].min()), float(boxes[:, 1].min()), float(boxes[:, 2].max()), float(boxes[:, 3].max()))


def _segments(prims: list[Primitive], layers: LayerMap, min_length: float) -> list[tuple[Any, Any, Primitive]]:
    found = []
    for p in prims:
        if layers.role(p.layer) != "STAIR":
            continue
        if p.kind == "line":
            pairs = [(p.geometry["p1"], p.geometry["p2"])]
        elif p.kind == "polyline":
            points = list(p.geometry["points"])
            if p.geometry["closed"]:
                points.append(points[0])
            pairs = list(zip(points, points[1:]))
        else:
            continue
        found.extend((a, b, p) for a, b in pairs if math.dist(a, b) >= min_length)
    return found


def _flights(segments: list[tuple[Any, Any, Primitive]], cfg: dict[str, Any]) -> list[FlightDraft]:
    groups: dict[int, list[tuple[Any, Any, Primitive]]] = {}
    for item in segments:
        a, b, _ = item
        angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180.0
        groups.setdefault(round(angle / cfg["angle_bin_deg"]), []).append(item)

    flights = []
    for key, members in groups.items():
        angle = math.radians(key * cfg["angle_bin_deg"])
        ux, uy = math.cos(angle), math.sin(angle)
        nx, ny = -uy, ux
        rows = []
        for a, b, p in members:
            s = sorted(((a[0] * ux + a[1] * uy), (b[0] * ux + b[1] * uy)))
            rows.append(((a[0] + b[0]) / 2 * nx + (a[1] + b[1]) / 2 * ny, s[0], s[1], p))
        # 같은 방향의 단이라도 나란히 놓인 다른 계단이면 길이 방향 구간이 겹치지 않는다.
        columns: list[list[Any]] = []
        for row in sorted(rows, key=lambda r: r[1]):
            for column in columns:
                lo, hi = column[0][1], column[0][2]
                if min(row[2], hi) - max(row[1], lo) >= cfg["min_overlap_share"] * min(row[2] - row[1], hi - lo):
                    column.append(row)
                    break
            else:
                columns.append([row])

        def flush(items: list[Any]) -> None:
            if len(items) < cfg["min_treads"]:
                return
            lo = float(np.median([r[1] for r in items]))
            hi = float(np.median([r[2] for r in items]))
            middle = (lo + hi) / 2
            first, last = items[0][0], items[-1][0]
            flights.append(FlightDraft(
                (middle * ux + first * nx, middle * uy + first * ny),
                (middle * ux + last * nx, middle * uy + last * ny),
                hi - lo, len(items), {h for r in items for h in r[3].source_handles},
            ))

        for column in columns:
            current: list[Any] = []
            for row in sorted(column, key=lambda r: r[0]):
                if current:
                    gap = row[0] - current[-1][0]
                    overlap = min(row[2], current[-1][2]) - max(row[1], current[-1][1])
                    spacing = row[0] - current[-1][0]
                    ok = cfg["min_spacing_mm"] <= gap <= cfg["max_spacing_mm"] and overlap >= cfg["min_overlap_share"] * (row[2] - row[1])
                    if len(current) >= 2:
                        ok = ok and abs(spacing - (current[-1][0] - current[-2][0])) <= cfg["spacing_tol_mm"]
                    if gap < cfg["min_spacing_mm"] and overlap > 0:
                        continue  # 겹쳐 그린 선
                    if not ok:
                        flush(current)
                        current = []
                current.append(row)
            flush(current)
    return flights


def recognize_stairs(prims: list[Primitive], layers: LayerMap, cfg: dict[str, Any]) -> list[StairDraft]:
    """가까운 단 묶음끼리 계단 하나로 합친다."""
    flights = _flights(_segments(prims, layers, cfg["min_tread_mm"]), cfg)
    stairs: list[list[FlightDraft]] = []
    gap = cfg["join_gap_mm"]
    for flight in sorted(flights, key=lambda f: f.bounds):
        b = flight.bounds
        for stair in stairs:
            if any(
                b[0] - gap <= o.bounds[2] and b[2] + gap >= o.bounds[0] and b[1] - gap <= o.bounds[3] and b[3] + gap >= o.bounds[1]
                for o in stair
            ):
                stair.append(flight)
                break
        else:
            stairs.append([flight])
    return [StairDraft(group) for group in stairs]
