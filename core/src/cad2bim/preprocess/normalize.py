"""선분 정리: 동일선상 겹침 병합, 작은 틈 연결, 극소 선분 제거.

레이어·선종류·블록이 다른 선은 합치지 않는다. 문틀 선이 벽 선에 먹히면 안 된다.
폴리선은 건드리지 않는다. 닫힌 외곽 정보가 인식의 근거이기 때문이다.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

from .primitive import Primitive


@dataclass
class NormalizeStats:
    removed_short: int = 0
    merged_away: int = 0


def _canonical_angle(p: Primitive) -> float:
    (x1, y1), (x2, y2) = p.geometry["p1"], p.geometry["p2"]
    return math.atan2(y2 - y1, x2 - x1) % math.pi


def _chain(items: list, key, tolerance: float) -> list[list]:
    """정렬한 값이 tolerance 이내로 이어지는 묶음."""
    ordered = sorted(items, key=key)
    groups: list[list] = []
    for item in ordered:
        if groups and key(item) - key(groups[-1][-1]) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def _group_by_angle(lines: list[Primitive], tolerance: float) -> list[list[Primitive]]:
    groups = _chain(lines, _canonical_angle, tolerance)
    # 0과 π는 같은 방향이다.
    if len(groups) > 1 and (
        _canonical_angle(groups[0][0]) + math.pi - _canonical_angle(groups[-1][-1]) <= tolerance
    ):
        groups[0].extend(groups.pop())
    return groups


def _merge_collinear(group: list[Primitive], cfg: dict[str, Any], stats: NormalizeStats) -> list[Primitive]:
    first = group[0]
    (x1, y1), (x2, y2) = first.geometry["p1"], first.geometry["p2"]
    length = math.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length

    def offset(p: Primitive) -> float:
        mx = (p.geometry["p1"][0] + p.geometry["p2"][0]) / 2
        my = (p.geometry["p1"][1] + p.geometry["p2"][1]) / 2
        return -uy * mx + ux * my

    result: list[Primitive] = []
    for row in _chain(group, offset, cfg["collinear_offset_mm"]):
        # 띠 하나에서 시작점 순서로 훑으며 구간을 합친다.
        spans = []
        for p in row:
            a = p.geometry["p1"][0] * ux + p.geometry["p1"][1] * uy
            b = p.geometry["p2"][0] * ux + p.geometry["p2"][1] * uy
            spans.append((min(a, b), max(a, b), p))
        spans.sort(key=lambda s: s[0])

        run = [spans[0]]
        end = spans[0][1]
        runs = []
        for span in spans[1:]:
            if span[0] - end <= cfg["gap_mm"]:
                run.append(span)
                end = max(end, span[1])
            else:
                runs.append((run, end))
                run, end = [span], span[1]
        runs.append((run, end))

        n = sum(offset(p) for p in row) / len(row)
        for members, stop in runs:
            if len(members) == 1:
                result.append(members[0][2])
                continue
            start = members[0][0]
            stats.merged_away += len(members) - 1
            handles = dict.fromkeys(h for m in members for h in m[2].source_handles)
            result.append(
                replace(
                    members[0][2],
                    geometry={
                        "p1": (start * ux - n * uy, start * uy + n * ux),
                        "p2": (stop * ux - n * uy, stop * uy + n * ux),
                    },
                    source_handles=tuple(handles),
                )
            )
    return result


def normalize(prims: list[Primitive], cfg: dict[str, Any]) -> tuple[list[Primitive], NormalizeStats]:
    stats = NormalizeStats()
    others: list[Primitive] = []
    lines: dict[tuple, list[Primitive]] = defaultdict(list)
    for p in prims:
        if p.kind != "line":
            others.append(p)
            continue
        (x1, y1), (x2, y2) = p.geometry["p1"], p.geometry["p2"]
        if math.hypot(x2 - x1, y2 - y1) < cfg["min_length_mm"]:
            stats.removed_short += 1
            continue
        lines[(p.layer, p.linetype, p.block_path, p.extra.get("instance", 0))].append(p)

    tolerance = math.radians(cfg["collinear_angle_deg"])
    merged: list[Primitive] = []
    for group in lines.values():
        for same_angle in _group_by_angle(group, tolerance):
            merged.extend(_merge_collinear(same_angle, cfg, stats))
    return others + merged, stats
