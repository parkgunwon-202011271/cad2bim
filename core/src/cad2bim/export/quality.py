"""건물 본체가 아닐 가능성이 큰 요소를 가른다. 지우지 않고 숨긴 태그로 옮긴다.

옹벽이나 경사로처럼 진짜일 수도 있고, 빠진 요소가 더 나쁘다는 사용자 답변이 있어 파일에는 남긴다.
실의 경계이거나 개구부가 달린 벽은 건물의 일부로 보고 건드리지 않는다.
"""
from __future__ import annotations

import math
from typing import Any

from shapely.geometry import LineString, MultiPoint, box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from .collada import _safe

OUTSIDE = "OUTSIDE_"
PATTERN = "PATTERN_"
FRAGMENT = "FRAGMENT_"
HIDDEN = (OUTSIDE, PATTERN, FRAGMENT)

_BODY_MARGIN_MM = 5000.0
_MIN_COLUMNS = 3
_FRAGMENT_MAX_MM = 1500.0
_TOUCH_MM = 200.0
_REPEAT_MIN = 4
_REPEAT_LENGTH_BIN_MM = 500.0
_REPEAT_GAP_TOL_MM = 100.0
_JUDGED = frozenset({"llm_outline", "sliver_filled"})  # 검토·틈 메우기로 세운 벽의 근거 표시


def _length(wall: dict[str, Any]) -> float:
    return math.dist(wall["start"], wall["end"])


def _repeated(walls: list[dict[str, Any]]) -> set[str]:
    """같은 방향과 길이로 일정한 간격을 두고 반복되는 벽. 랙, 주차선, 루버 같은 패턴이다."""
    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for wall in walls:
        angle = math.degrees(math.atan2(wall["end"][1] - wall["start"][1], wall["end"][0] - wall["start"][0])) % 180.0
        groups.setdefault((round(angle / 2.0), round(_length(wall) / _REPEAT_LENGTH_BIN_MM)), []).append(wall)
    found: set[str] = set()
    for members in groups.values():
        if len(members) < _REPEAT_MIN:
            continue
        first = members[0]
        angle = math.atan2(first["end"][1] - first["start"][1], first["end"][0] - first["start"][0])
        nx, ny = -math.sin(angle), math.cos(angle)
        rows = sorted(
            ((w["start"][0] + w["end"][0]) / 2 * nx + (w["start"][1] + w["end"][1]) / 2 * ny, w["id"]) for w in members
        )
        gaps = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
        run = [rows[0][1]]
        for k, gap in enumerate(gaps, start=1):
            if len(run) >= 2 and abs(gap - gaps[k - 2]) > _REPEAT_GAP_TOL_MM:
                if len(run) >= _REPEAT_MIN:
                    found.update(run)
                run = [rows[k - 1][1]]
            run.append(rows[k][1])
        if len(run) >= _REPEAT_MIN:
            found.update(run)
    return found


def classify(ir: dict[str, Any]) -> dict[str, str]:
    """숨길 요소의 이름과 태그 머리말. 근거가 약한 쪽으로 가르지 않고 벽 하나씩 본다."""
    walls = [w for w in ir["walls"] if _length(w) > 0]
    kept = {room_wall for room in ir.get("rooms", []) for room_wall in room.get("boundary_wall_ids", [])}
    kept |= {opening["host_wall"] for opening in ir.get("openings", [])}
    # LLM 검토가 도면을 보고 벽으로 고른 곳과 좁은 틈을 메운 벽은 추측이 아니다. 숨기면 그 자리가 다시 뚫린다.
    kept |= {w["id"] for w in walls if _JUDGED & set((w.get("confidence") or {}).get("reasons", []))}
    columns = [tuple(c["center"]) for c in ir["columns"]]
    body = None
    if len(columns) >= _MIN_COLUMNS:
        body = MultiPoint(columns).convex_hull.buffer(_BODY_MARGIN_MM)

    result: dict[str, str] = {}
    free = [w for w in walls if w["id"] not in kept and w["kind"] != "curtain"]
    if body is not None:
        for wall in free:
            if not body.intersects(LineString([wall["start"], wall["end"]])):
                result[wall["id"]] = OUTSIDE
    for name in _repeated(free):
        result.setdefault(name, PATTERN)
    if walls:
        lines = [LineString([w["start"], w["end"]]).buffer(w["thickness"] / 2 + _TOUCH_MM) for w in walls]
        tree = STRtree(lines)
        column_shapes = [
            box(c["center"][0] - c["size"][0] / 2, c["center"][1] - c["size"][1] / 2, c["center"][0] + c["size"][0] / 2, c["center"][1] + c["size"][1] / 2)
            for c in ir["columns"]
        ]
        column_union = unary_union(column_shapes) if column_shapes else None
        for index, wall in enumerate(walls):
            if wall["id"] in kept or wall["id"] in result or wall["kind"] == "curtain" or _length(wall) >= _FRAGMENT_MAX_MM:
                continue
            touching = [i for i in tree.query(lines[index], predicate="intersects").tolist() if i != index]
            if not touching and (column_union is None or not column_union.intersects(lines[index])):
                result[wall["id"]] = FRAGMENT

    if body is not None:
        for stair in ir.get("stairs", []):
            if not body.intersects(box(*stair["bounds"])):
                result[stair["id"]] = OUTSIDE
    return {_safe(name): prefix for name, prefix in result.items()}


def building_only(ir: dict[str, Any]) -> dict[str, Any]:
    """슬래브와 지붕을 계산할 때 쓰는 IR. 숨길 요소를 뺀다."""
    hidden = classify(ir)
    return {
        **ir,
        "walls": [w for w in ir["walls"] if _safe(w["id"]) not in hidden],
        "stairs": [s for s in ir.get("stairs", []) if _safe(s["id"]) not in hidden],
    }
