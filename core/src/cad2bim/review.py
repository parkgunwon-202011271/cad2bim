"""LLM 검토(처리 단계 6). 규칙이 판단하지 못한 지점을 LLM이 도면 조각을 보고 고른다.

흐름: 의심 지점 찾기(규칙) → 지점마다 원본 도면을 잘라 그림으로(근거) → LLM이 정해진 답 중 하나를 고름
→ 빌드할 때 답대로 적용. 인식 결과(ir.json)는 바꾸지 않고, 답은 결과 폴더 _llm/decisions.json에 쌓는다.
LLM은 좌표를 만들지 않는다. 고른 답에 따라 형상은 규칙이 만든다.

지금 보는 지점: 종류를 모르는 개구부(벽선이 끊겼는데 문·창 표시를 못 찾은 곳), 외곽선인데 벽이 없는 구간.
켜는 법은 레이어 판단과 같다(CAD2BIM_LLM=anthropic|cache). cache는 저장된 답만 쓰고, 답이 없는 지점은
_llm/pending.json과 그림으로 남긴다(사람이나 다른 LLM이 답을 채울 수 있다).
"""
from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .layers.llm import mode
from .llm_client import PROVIDERS, QuotaExhausted, ask, configure, default_model, extract_json

DECISIONS = "decisions.json"
PENDING = "pending.json"
ITEMS = "review.json"  # 앱의 검토 패널이 읽는 전체 지점 목록(답이 있는 곳과 없는 곳 모두)
ANSWERS = ("wall", "door", "window", "open", "drawing_issue")
_CROP_MM = 4000.0  # 지점 둘레로 보여 줄 반경
_MM_PER_PX = 15.0  # 도면 그림 해상도

# 평면도 읽는 법. 검토하며 쌓인 판단 기준이다(실수가 나온 것부터 적는다). 지점 종류별로 필요한 것만 붙인다.
# 작은 모델은 기준을 엉뚱한 곳에 그대로 따라 하므로(기둥 모서리를 'EV 문'으로) 짧게, 종류에 맞는 것만 준다.
_NOTES_COMMON = (
    "판단 기준(한국 건축 평면도):",
    "- 아래 '근거'는 도면 데이터에서 계산한 것이라 정확하다. 그림보다 근거를 먼저 본다.",
    "- 긴 선과 점이 번갈아 나오는 일점쇄선(GRID 레이어)은 축선이다. 벽도 문도 아니니 무시한다.",
    "- 검은 사각형·해치로 채운 사각형(COLUMN 레이어)은 기둥이다.",
    "- drawing_issue는 선이 서로 어긋나 무엇인지 정말 알 수 없을 때만 쓴다.",
)
_NOTES_BY_KIND = {
    "opening": (
        "- 문은 호가 없어도 문일 수 있다: 승강로(EV) 바로 앞의 끊긴 자리는 승강기 문, 벽 속 얇은 선 한두 줄은 미닫이문이다.",
        "  근거에 DOOR 레이어 선이나 ARC(원호)가 있으면 문이다.",
        "- 상자 양쪽이 서로 다른 실(예: 화장실과 복도)이고 폭이 800~1500mm면 문이다.",
        "  양쪽이 같은 실이거나 홀·로비·통로처럼 열린 공간끼리이고 벽 레이어 선이 없으면 open이다.",
    ),
    "outline": (
        "- 외곽 빈 구간 둘레에 필로티·주차·차로·출입 광장·보행통로 같은 글자가 있고 벽·창·문 레이어 선이 없으면 open이다.",
        "- '차량 출입'처럼 글자만 있고 셔터 선(DOOR 레이어)이 없으면 open이다.",
    ),
    "corner": (
        "- 상자 안에 기둥(COLUMN)이나 벽 해치가 있으면 막혀 있다(wall).",
        "- 근거에 DOOR 레이어 선이나 ARC(원호)가 있으면 문이다.",
    ),
    "window": (
        "- 실내 창은 드물지만 있다: 회의실·사무실이 복도·공용 공간에 면한 유리벽, 오픈(보이드) 가장자리 유리.",
        "- 벽 속 얇은 두 줄(유리선)이나 근거의 WINDOW 레이어 선은 창이다. DOOR 선·ARC가 있으면 문이다.",
    ),
}


def drawing_notes(kind: str) -> str:
    """그 지점 종류에 맞는 판단 기준 글."""
    return "\n".join(_NOTES_COMMON + _NOTES_BY_KIND.get(kind, ())) + "\n\n"


def conflict(answer: str, counts: dict[str, int], kind: str = "opening") -> str | None:
    """답이 도면 데이터와 정면으로 어긋나면 그 이유(사람이 읽는 글), 아니면 None. 판단은 LLM이, 거르기는 계산으로.

    - 모서리 지점에 기둥이 있는데 벽이 아니라고 한 경우(벽 끝과 기둥 사이는 막혀 있다)
    - 상자에 문 레이어 선이나 원호가 있는데 열린 곳·벽이라고 하거나, 창 레이어 선이 있는데 벽이라고 한 경우
    - 상자에 축선·치수 말고는 아무 선도 없는데 문·창이라고 한 경우
    """
    if counts.get("COLUMN"):
        # 모서리(벽 끝과 이웃 벽 사이)에 기둥이 있으면 막힌 자리다. 그 밖의 지점은 기둥 옆 문도 있어 어느 답이든 둔다
        if kind == "corner" and answer != "wall":
            return f"모서리에 기둥이 {counts['COLUMN']}개 있어 막힌 자리인데 벽이 아니라고 했습니다."
        return None
    doors, arcs = counts.get("DOOR", 0), counts.get("ARC", 0)
    if (doors + arcs) and answer in ("open", "wall"):
        found = " · ".join(text for text, n in ((f"문선 {doors}개", doors), (f"원호 {arcs}개", arcs)) if n)
        return f"상자 안에 {found}가 있어 문 표시가 있는 자리인데 {'열린 통로라고' if answer == 'open' else '벽이라고'} 했습니다."
    if counts.get("WINDOW") and answer == "wall":
        return f"상자 안에 창선 {counts['WINDOW']}개가 있어 창 표시가 있는 자리인데 벽이라고 했습니다."
    if not any(v for k, v in counts.items() if k not in ("GRID", "DIM")) and answer in ("door", "window"):
        return f"상자 안에 축선·치수 말고는 아무 선도 없는데 {'문' if answer == 'door' else '창'}이라고 했습니다."
    return None


def consistent(answer: str, counts: dict[str, int], kind: str = "opening") -> bool:
    """답이 도면 데이터와 정면으로 어긋나면 적용하지 않는다(대기 목록에 남긴다). 이유는 conflict가 준다."""
    return conflict(answer, counts, kind) is None


_QUESTION = """건축 평면도 조각입니다. 빨간 상자는 벽선이 끊긴 자리(폭 {width:.0f}mm)이고, 규칙은 여기에 문이나 창 표시를 찾지 못했습니다.
이 자리가 무엇인지 하나만 고르세요.
- wall: 끊긴 게 아니라 벽이 이어진다(그림상 벽 표현이 이어지거나 기둥·마감선 등으로 막혀 있음)
- door: 문이 있다(문 호, 미닫이·셔터·방열문 선, 문 기호 등)
- window: 창이 있다(창틀·유리 선)
- open: 실제로 뚫린 통로·개구부다
- drawing_issue: 도면 자체가 어긋나 판단할 수 없다
JSON 하나로만 답하세요. 예: {{"answer": "door", "reason": "미닫이문 선 두 줄"}}"""


_OUTLINE_QUESTION = """건축 평면도 조각입니다. 빨간 상자는 건물 외곽선을 따라가다 벽이 그려지지 않은 구간(길이 {width:.0f}mm)입니다.
이 구간이 무엇인지 하나만 고르세요.
- wall: 벽이 있어야 한다(옆 벽이 이어지거나, 마감선·패널선 등으로 벽이 그려져 있음)
- window: 창·유리면이다
- door: 문·셔터·하역 문이다
- open: 실제로 열린 곳이다(필로티, 하역장 앞, 외부 통로 등)
- drawing_issue: 도면 자체가 어긋나 판단할 수 없다
JSON 하나로만 답하세요. 예: {{"answer": "wall", "reason": "외벽 패널선이 이어짐"}}"""
_OUTLINE_MIN_MM = 300.0  # 이보다 짧게 빈 곳은 보지 않는다
_OUTLINE_STEP_MM = 100.0
_OUTLINE_REACH_MM = 250.0  # 벽 면에서 이만큼 안이면 외곽선이 벽으로 덮였다고 본다
_SLIT_REACH_MM = 20.0  # 틈 찾기: 벽 면에 거의 붙은 것만 덮였다고 본다
_SLIT_MIN_MM = 30.0
_SLIT_STEP_MM = 10.0


@dataclass
class Spot:
    key: str  # 도면 해시 + 위치(100mm 단위)로 정한다. 인식 번호가 바뀌어도 같은 지점이다
    floor: str
    part_id: str
    opening_id: str
    center: tuple[float, float]  # IR 좌표 mm
    direction: tuple[float, float]
    width: float
    thickness: float
    kind: str = "opening"  # opening(종류 모르는 개구부) | outline(외곽선인데 벽이 없는 구간) | corner | slit | window(실내 창)
    path: tuple[tuple[float, float], ...] | None = None  # L자 모서리 틈: 벽 끝 → 두 벽 중심선이 만나는 점 → 이웃 벽 끝


def find_spots(storeys: list[Any]) -> list[Spot]:
    spots = []
    for s in storeys:
        ir = s.part.ir
        walls = {w["id"]: w for w in ir["walls"]}
        sha = ir["source"].get("sha256", "")[:16]
        for o in ir["openings"]:
            if o["type"] != "unknown" or o["host_wall"] not in walls:
                continue
            w = walls[o["host_wall"]]
            (x0, y0), (x1, y1) = w["start"], w["end"]
            length = math.hypot(x1 - x0, y1 - y0) or 1.0
            u = ((x1 - x0) / length, (y1 - y0) / length)
            c = (x0 + u[0] * o["position"], y0 + u[1] * o["position"])
            key = f"{sha}:{round(c[0] / 100)}:{round(c[1] / 100)}:{round(o['width'] / 100)}"
            spots.append(Spot(key, s.key, s.part.part_id, o["id"], c, u, o["width"], w["thickness"]))
    return spots


_WINDOW_QUESTION = """건축 평면도 조각입니다. 빨간 상자는 규칙이 '창'으로 읽은 자리(폭 {width:.0f}mm)인데, 건물 바깥벽이 아닌 실내 벽에 있습니다.
실내 창은 드물어서 확인이 필요합니다. 하나만 고르세요.
- window: 실제 창이다(창틀·유리 선, 창 기호가 있음)
- door: 창이 아니라 문이다(문 호, 미닫이·방화문 선 등)
- open: 문짝 없이 뚫린 통로·개구부다
- wall: 개구부가 아니다(벽이 이어지거나, 선반·설비 등 다른 것을 창으로 읽었음)
- drawing_issue: 도면 자체가 어긋나 판단할 수 없다
JSON 하나로만 답하세요. 예: {{"answer": "window", "reason": "창틀 두 줄과 유리선"}}"""
_INTERIOR_MM = 600.0  # 바닥 외곽선에서 이보다 안쪽 벽의 창을 실내 창으로 본다


def find_interior_windows(storeys: list[Any], export: dict[str, Any]) -> list[Spot]:
    """실내 벽에 달린 창. 진짜 창인지 문·통로를 잘못 읽은 것인지 LLM이 도면을 보고 고른다."""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    from .export.floor import floor_outlines
    from .export.quality import building_only

    spots = []
    for s in storeys:
        ir = s.part.ir
        body = building_only(ir)
        outlines = floor_outlines(body, export["slab_close_mm"], export["slab_min_area_m2"], export["slab_trim_mm"])
        if not outlines:
            continue
        edge = unary_union([o.exterior for o in outlines])
        walls = {w["id"]: w for w in body["walls"]}
        sha = ir["source"].get("sha256", "")[:16]
        for o in ir["openings"]:
            w = walls.get(o.get("host_wall"))
            if o["type"] != "window" or w is None or w["start"] == w["end"]:
                continue
            (x0, y0), (x1, y1) = w["start"][:2], w["end"][:2]
            length = math.hypot(x1 - x0, y1 - y0)
            u = ((x1 - x0) / length, (y1 - y0) / length)
            c = (x0 + u[0] * o["position"], y0 + u[1] * o["position"])
            if edge.distance(Point(c)) <= _INTERIOR_MM + w["thickness"] / 2:
                continue  # 바깥벽 창
            key = f"{sha}:w:{round(c[0] / 100)}:{round(c[1] / 100)}:{round(o['width'] / 100)}"
            spots.append(Spot(key, s.key, s.part.part_id, o["id"], c, u, o["width"], w["thickness"], "window"))
    return spots


def find_outline_spots(
    storeys: list[Any], export: dict[str, Any], reach: float = _OUTLINE_REACH_MM, min_mm: float = _OUTLINE_MIN_MM,
    max_mm: float = float("inf"), tag: str = "g", step: float = _OUTLINE_STEP_MM,
) -> list[Spot]:
    """층마다 바닥 외곽선을 따라가며 벽·기둥이 덮지 않은 구간. 외벽이 빠진 면과 벽 사이 틈을 찾는다.

    reach: 벽 면에서 이만큼 안이면 덮였다고 본다. 크게 잡으면 긴 빈 면만, 작게 잡으면 좁은 세로 틈(창·진입창 자리)도 찾는다.
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    from .export.floor import floor_outlines
    from .export.quality import building_only

    spots = []
    for s in storeys:
        ir = s.part.ir
        body = building_only(ir)
        cover = [
            LineString([w["start"], w["end"]]).buffer(w["thickness"] / 2 + reach, cap_style="round" if tag == "g" else "flat")
            for w in body["walls"] if w["start"] != w["end"]
        ]
        for c in body["columns"]:
            size = c.get("size") or [500.0, 500.0]
            size = size if isinstance(size, list) else [size, size]
            cover.append(Point(c["center"][:2]).buffer(max(size) / 2 + reach))
        if not cover:
            continue
        covered = unary_union(cover)
        thickness = sorted(w["thickness"] for w in body["walls"])[len(body["walls"]) // 2] if body["walls"] else 200.0
        sha = ir["source"].get("sha256", "")[:16]
        number = 0
        for outline in floor_outlines(body, export["slab_close_mm"], export["slab_min_area_m2"], export["slab_trim_mm"]):
            ring = list(outline.exterior.coords)
            for a, b in zip(ring, ring[1:]):
                length = math.dist(a, b)
                if length < min_mm:
                    continue
                u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
                steps = max(int(length // step), 1)
                run = None
                for k in range(steps + 1):
                    d = min(k * step, length)
                    free = not covered.contains(Point(a[0] + u[0] * d, a[1] + u[1] * d))
                    if free and run is None:
                        run = d
                    if (not free or k == steps) and run is not None:
                        stop = d if not free else length
                        if min_mm <= stop - run < max_mm and (tag == "g" or (run > 0.0 and stop < length)):
                            mid = (run + stop) / 2
                            c = (a[0] + u[0] * mid, a[1] + u[1] * mid)
                            number += 1
                            key = f"{sha}:{tag}:{round(c[0] / 100)}:{round(c[1] / 100)}:{round((stop - run) / 100)}"
                            spots.append(Spot(key, s.key, s.part.part_id, f"{tag.upper()}{number:03d}", c, u, stop - run, thickness, "outline"))
                        run = None
    return spots


def find_slits(storeys: list[Any], export: dict[str, Any], max_mm: float) -> list[Spot]:
    """외곽선에서 벽 사이 좁은 틈(최소 개구부 폭보다 좁음). 문·창일 수 없으니 판단 없이 벽으로 메운다(계산)."""
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    from .export.floor import floor_outlines
    from .export.quality import building_only

    spots = []
    for s in storeys:
        body = building_only(s.part.ir)
        walls = [w for w in body["walls"] if w["start"] != w["end"]]
        if not walls:
            continue
        near = unary_union([LineString([w["start"], w["end"]]).buffer(w["thickness"] / 2 + _SLIT_REACH_MM, cap_style="flat") for w in walls] + [
            Point(c["center"][:2]).buffer(max(c.get("size") or [500.0]) / 2 + _SLIT_REACH_MM) if isinstance(c.get("size"), list)
            else Point(c["center"][:2]).buffer((c.get("size") or 500.0) / 2 + _SLIT_REACH_MM)
            for c in body["columns"]
        ])
        thickness = sorted(w["thickness"] for w in walls)[len(walls) // 2]
        number = 0
        for outline in floor_outlines(body, export["slab_close_mm"], export["slab_min_area_m2"], export["slab_trim_mm"]):
            ring = list(outline.exterior.coords)
            for a, b in zip(ring, ring[1:]):
                length = math.dist(a, b)
                if length < _SLIT_MIN_MM:
                    continue
                u = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
                steps = max(int(length // _SLIT_STEP_MM), 1)
                run = None
                for k in range(steps + 1):
                    d = min(k * _SLIT_STEP_MM, length)
                    free = not near.contains(Point(a[0] + u[0] * d, a[1] + u[1] * d))
                    if free and run is None:
                        run = d
                    if (not free or k == steps) and run is not None:
                        stop = d if not free else length
                        # 꼭짓점에 닿는 빈 곳은 벽 끝이 만나는 모서리다(틈이 아니다).
                        if _SLIT_MIN_MM <= stop - run < max_mm and run > 0.0 and stop < length:
                            mid = (run + stop) / 2
                            c = (a[0] + u[0] * mid, a[1] + u[1] * mid)
                            number += 1
                            spots.append(Spot(f"slit:{s.key}:{number}", s.key, s.part.part_id, f"N{number:03d}", c, u,
                                              stop - run + 2 * _SLIT_REACH_MM, thickness, "slit"))
                        run = None
    return spots


_CORNER_QUESTION = """건축 평면도 조각입니다. 빨간 상자는 벽 끝과 이웃 벽 사이가 비어 있는 구간(길이 {width:.0f}mm)입니다.
이 구간이 무엇인지 하나만 고르세요.
- wall: 벽이 이어져야 한다(도면에 벽·기둥·해치가 이어지거나, 벽을 짧게 읽은 것)
- door: 문이 있다
- window: 창이 있다
- open: 실제로 열린 통로다
- drawing_issue: 도면 자체가 어긋나 판단할 수 없다
JSON 하나로만 답하세요. 예: {{"answer": "wall", "reason": "해치 벽이 모서리까지 이어짐"}}"""
_CORNER_MIN_MM = 400.0
_CORNER_MAX_MM = 1200.0
_ELBOW_MIN_MM = 30.0  # 이보다 가까운 벽 끝은 모서리 이음 자국이다


def find_corner_spots(storeys: list[Any]) -> list[Spot]:
    """벽 끝을 그 방향으로 늘렸을 때 0.4~1.2m 안에 다른 벽이 있는 곳. 벽을 짧게 읽어 생긴 모서리 틈을 찾는다."""
    from shapely.geometry import LineString

    from .export.quality import building_only

    spots = []
    for s in storeys:
        body = building_only(s.part.ir)
        walls = [w for w in body["walls"] if w["start"] != w["end"]]
        sha = s.part.ir["source"].get("sha256", "")[:16]
        lines = [LineString([w["start"], w["end"]]).buffer(w["thickness"] / 2, cap_style="flat") for w in walls]
        number = 0
        seen = set()
        for i, w in enumerate(walls):
            (x0, y0), (x1, y1) = w["start"], w["end"]
            length = math.hypot(x1 - x0, y1 - y0)
            u = ((x1 - x0) / length, (y1 - y0) / length)
            for end, (px, py), sign in ((0, (x0, y0), -1.0), (1, (x1, y1), 1.0)):
                d = (u[0] * sign, u[1] * sign)
                face = (px + d[0] * 1.0, py + d[1] * 1.0)
                ray = LineString([face, (px + d[0] * (_CORNER_MAX_MM + 1.0), py + d[1] * (_CORNER_MAX_MM + 1.0))])
                if any(lines[j].contains(ray.interpolate(0.0)) for j in range(len(walls)) if j != i):
                    continue  # 이미 다른 벽에 닿아 있다
                hits = [ray.intersection(lines[j]) for j in range(len(walls)) if j != i and ray.intersects(lines[j])]
                hits = [h for h in hits if not h.is_empty]
                if not hits:
                    continue
                gap = min(Point_distance(face, h) for h in hits)
                if not _CORNER_MIN_MM <= gap <= _CORNER_MAX_MM:
                    continue
                c = (px + d[0] * gap / 2, py + d[1] * gap / 2)
                key = f"{sha}:c:{round(c[0] / 100)}:{round(c[1] / 100)}:{round(gap / 100)}"
                if key in seen:
                    continue
                seen.add(key)
                number += 1
                spots.append(Spot(key, s.key, s.part.part_id, f"K{number:03d}", c, d, gap, w["thickness"], "corner"))
        for sp in _elbow_spots(walls, lines, sha, s.key, s.part.part_id, number):
            if sp.key not in seen:
                seen.add(sp.key)
                spots.append(sp)
    return spots


def _elbow_spots(walls: list[dict[str, Any]], lines: list[Any], sha: str, floor: str, part_id: str, number: int) -> list[Spot]:
    """두 벽 끝이 서로 비켜 있어 어느 쪽을 늘려도 상대 벽에 닿지 않는 L자 모서리 틈.

    두 벽이 거의 직각이고, 둘 다 끝에서 앞으로 늘린 중심선이 1.2m 안에서 만나면 지점으로 본다.
    """
    from shapely.geometry import LineString, Point

    ends = []
    for i, w in enumerate(walls):
        (x0, y0), (x1, y1) = w["start"], w["end"]
        length = math.hypot(x1 - x0, y1 - y0)
        u = ((x1 - x0) / length, (y1 - y0) / length)
        for p, d in (((x0, y0), (-u[0], -u[1])), ((x1, y1), u)):
            probe = Point(p[0] + d[0], p[1] + d[1])
            if any(lines[j].contains(probe) for j in range(len(walls)) if j != i):
                continue  # 이미 다른 벽에 닿아 있다
            ends.append((i, p, d))
    pairs = []
    for a in range(len(ends)):
        i, pa, da = ends[a]
        for b in range(a + 1, len(ends)):
            j, pb, db = ends[b]
            if i == j or not _ELBOW_MIN_MM <= math.dist(pa, pb) <= _CORNER_MAX_MM:
                continue
            cross = da[0] * db[1] - da[1] * db[0]
            if abs(cross) < 0.87:  # 60~120도만
                continue
            # pa + ta*da = pb + tb*db
            rx, ry = pb[0] - pa[0], pb[1] - pa[1]
            ta = (rx * db[1] - ry * db[0]) / cross
            tb = (rx * da[1] - ry * da[0]) / cross
            slack = max(walls[i]["thickness"], walls[j]["thickness"]) / 2
            if not (-slack <= ta <= _CORNER_MAX_MM and -slack <= tb <= _CORNER_MAX_MM):
                continue
            meet = (pa[0] + da[0] * ta, pa[1] + da[1] * ta)
            route = LineString([pa, meet, pb])
            if any(lines[k].intersects(route) for k in range(len(walls)) if k not in (i, j)):
                continue  # 다른 벽이 사이에 있다. 이미 막혔거나 다른 벽과의 모서리다
            pairs.append((math.dist(pa, pb), a, b, meet))
    # 벽 끝마다 가장 가까운 짝 하나만. 서로 가장 가까운 짝일 때만 지점으로 본다(계단·코어처럼 끝이 몰린 곳의 겹침을 막는다).
    best: dict[int, float] = {}
    for gap, a, b, _ in pairs:
        best[a] = min(best.get(a, gap), gap)
        best[b] = min(best.get(b, gap), gap)
    spots = []
    for gap, a, b, meet in sorted(pairs):
        if gap > best[a] or gap > best[b]:
            continue
        (i, pa, _), (j, pb, _) = ends[a], ends[b]
        c = ((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
        gap = math.dist(pa, pb)
        u = ((pb[0] - pa[0]) / gap, (pb[1] - pa[1]) / gap)
        key = f"{sha}:e:{round(c[0] / 100)}:{round(c[1] / 100)}:{round(gap / 100)}"
        number += 1
        # 최소 개구부 폭보다 좁으면 문·창일 수 없으니 판단 없이 메운다(slit). 넓으면 LLM이 고른다.
        kind = "corner" if gap >= _CORNER_MIN_MM else "slit"
        spots.append(Spot(key, floor, part_id, f"{'K' if kind == 'corner' else 'L'}{number:03d}", c, u, gap,
                          max(walls[i]["thickness"], walls[j]["thickness"]), kind, (pa, meet, pb)))
    return spots


def _past(a: tuple[float, float], b: tuple[float, float], extra: float) -> tuple[float, float]:
    """a에서 b로 가는 선을 b 너머로 extra만큼 늘린 끝점."""
    d = math.dist(a, b)
    if d < 1e-6:
        return b
    return (b[0] + (b[0] - a[0]) / d * extra, b[1] + (b[1] - a[1]) / d * extra)


def Point_distance(p: tuple[float, float], geom: Any) -> float:
    from shapely.geometry import Point

    return Point(p).distance(geom)


def load_decisions(directory: Path) -> dict[str, dict[str, Any]]:
    path = Path(directory) / DECISIONS
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.get("decisions", {}).items() if v.get("answer") in ANSWERS}


def save_decisions(directory: Path, decisions: dict[str, dict[str, Any]]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    merged = load_decisions(directory)
    merged.update(decisions)
    (directory / DECISIONS).write_text(json.dumps({"decisions": dict(sorted(merged.items()))}, ensure_ascii=False, indent=1), encoding="utf-8")


REJECTED = "rejected.json"  # LLM이 답했지만 도면 근거와 어긋나 적용하지 않은 답. 화면에서 '걸러짐'으로 보여 준다


def load_rejected(directory: Path) -> dict[str, dict[str, Any]]:
    path = Path(directory) / REJECTED
    if not path.is_file():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")).get("rejected", {}))


def save_rejected(directory: Path, rejected: dict[str, dict[str, Any]], accepted: set[str] | frozenset[str] = frozenset()) -> None:
    """걸러진 답을 더해 쓴다. 그 뒤 받아들인 답이 생긴 지점(accepted)은 지운다."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    merged = {k: v for k, v in (load_rejected(directory) | rejected).items() if k not in accepted}
    (directory / REJECTED).write_text(json.dumps({"rejected": dict(sorted(merged.items()))}, ensure_ascii=False, indent=1), encoding="utf-8")


EVIDENCE = "evidence.json"
_TEXT_REACH_MM = 4000.0  # 이 거리 안의 글자를 근거로 준다


def spot_evidence(spots: list[Spot], storeys: list[Any], drawings: dict[str, Path], out_dir: Path) -> dict[str, dict[str, Any]]:
    """지점마다 도면에서 읽은 근거 글: 둘레의 글자, 상자 안 선의 레이어 역할별 개수. 계산이라 틀리지 않는다.

    그림만 보면 '선이 없음'과 '벽선이 끊김', 그리드 일점쇄선과 셔터 점선을 헷갈린다. 판단은 LLM이 한다.
    """
    import ezdxf
    from shapely.geometry import LineString, Point, Polygon

    by_part: dict[str, list[Spot]] = {}
    for sp in spots:
        by_part.setdefault(sp.part_id, []).append(sp)
    result: dict[str, str] = {}
    for s in storeys:
        mine = by_part.get(s.part.part_id)
        if not mine or s.part.part_id not in drawings:
            continue
        ir = s.part.ir
        scale = float(ir["source"].get("unit_scale") or 1.0)
        ox, oy = ir["origin_offset"]
        roles_path = Path(out_dir) / s.part.part_id / "layer_map.json"
        roles = {n: i["role"] for n, i in json.loads(roles_path.read_text(encoding="utf-8"))["layers"].items()} if roles_path.is_file() else {}
        boxes = {}
        for sp in mine:
            u, n = sp.direction, (-sp.direction[1], sp.direction[0])
            hw, ht = sp.width / 2, sp.thickness / 2 + 150.0
            boxes[sp.key] = Polygon([(sp.center[0] + a * u[0] * hw + b * n[0] * ht, sp.center[1] + a * u[1] * hw + b * n[1] * ht)
                                     for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        lines: dict[str, dict[str, int]] = {k: {} for k in boxes}
        texts: dict[str, list[tuple[float, str]]] = {k: [] for k in boxes}
        doc = ezdxf.readfile(drawings[s.part.part_id])
        to_ir = lambda x, y: (x * scale - ox, y * scale - oy)  # noqa: E731

        def visit(e: Any, layer: str) -> None:
            kind = e.dxftype()
            if kind in ("TEXT", "MTEXT"):
                text = (e.plain_text() if kind == "MTEXT" else e.dxf.text).strip()
                if not text:
                    return
                x, y = to_ir(e.dxf.insert.x, e.dxf.insert.y)
                for key, sp in ((k, v) for k, v in ((sp.key, sp) for sp in mine)):
                    d = math.dist((x, y), sp.center)
                    if d <= _TEXT_REACH_MM:
                        texts[key].append((d, text[:30]))
                return
            if kind == "LINE":
                pts = [to_ir(e.dxf.start.x, e.dxf.start.y), to_ir(e.dxf.end.x, e.dxf.end.y)]
            elif kind == "LWPOLYLINE":
                pts = [to_ir(p[0], p[1]) for p in e.get_points()]
            elif kind in ("ARC", "CIRCLE"):
                c = to_ir(e.dxf.center.x, e.dxf.center.y)
                geom = Point(c).buffer(e.dxf.radius * scale).exterior
                pts = None
            else:
                return
            if pts is not None:
                if len(pts) < 2:
                    return
                geom = LineString(pts)
            role = roles.get(layer, "UNKNOWN")
            tag = "ARC" if kind in ("ARC", "CIRCLE") else role
            for key, poly in boxes.items():
                if poly.intersects(geom):
                    lines[key][tag] = lines[key].get(tag, 0) + 1

        for e in doc.modelspace():
            try:
                if e.dxftype() == "INSERT":
                    for v in e.virtual_entities():
                        visit(v, v.dxf.layer if v.dxf.layer != "0" else e.dxf.layer)
                else:
                    visit(e, e.dxf.layer)
            except Exception:  # 깨진 블록 하나로 근거 전체를 버리지 않는다
                continue
        rooms = [(Polygon(r["polygon"]), "·".join((r.get("names") or [])[:3]) or "이름 없는 실")
                 for r in ir.get("rooms", []) if r.get("polygon") and len(r["polygon"]) >= 3]
        for sp in mine:
            key = sp.key
            near = ", ".join(t for _, t in sorted(texts[key])[:12]) or "없음"
            counted = ", ".join(f"{k} {v}" for k, v in sorted(lines[key].items(), key=lambda kv: -kv[1])) or "아무 선도 없음"
            # 상자 양쪽(벽 두께 너머 0.6m)이 어느 실인지. 같은 실이면 통로, 다른 실이면 출입구일 가능성이 크다
            n = (-sp.direction[1], sp.direction[0])
            sides = []
            for sign in (1, -1):
                d = sp.thickness / 2 + 600.0
                probe = Point(sp.center[0] + n[0] * d * sign, sp.center[1] + n[1] * d * sign)
                sides.append(next((name for poly, name in rooms if poly.is_valid and poly.contains(probe)), "실 밖(외부 또는 인식 안 된 곳)"))
            wall_lines = lines[key].get("WALL", 0)
            result[key] = {
                "text": "\n".join([
                    "도면에서 계산한 근거(정확함):",
                    f"- 빨간 상자와 겹치는 선(레이어 역할별 개수, ARC=원호): {counted}",
                    f"- 벽 레이어 선: {wall_lines}개" + (" (상자 안에 벽선이 없다)" if not wall_lines else ""),
                    f"- 상자 한쪽: {sides[0]} / 다른 쪽: {sides[1]}" + (" (같은 실)" if sides[0] == sides[1] and not sides[0].startswith("실 밖") else ""),
                    f"- 둘레 {_TEXT_REACH_MM / 1000:.0f}m 안 글자(가까운 순): {near}",
                    "",
                ]),
                "counts": dict(lines[key]),
                "sides": sides,
            }
    return result


def render_spots(spots: list[Spot], storeys: list[Any], drawings: dict[str, Path], directory: Path, cfg: dict[str, Any]) -> dict[str, Path]:
    """지점마다 원본 도면 조각 그림. 층마다 건물 범위를 한 번 그리고 잘라 쓴다."""
    import ezdxf
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ezdxf.addons.drawing import Frontend, RenderContext
    from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration
    from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
    from PIL import Image, ImageDraw

    out = Path(directory) / "spots"
    out.mkdir(parents=True, exist_ok=True)
    by_part: dict[str, list[Spot]] = {}
    for spot in spots:
        by_part.setdefault(spot.part_id, []).append(spot)
    images: dict[str, Path] = {}
    for s in storeys:
        mine = by_part.get(s.part.part_id)
        if not mine or s.part.part_id not in drawings:
            continue
        ir = s.part.ir
        scale = float(ir["source"].get("unit_scale") or 1.0)
        ox, oy = ir["origin_offset"]
        xs = [p[0] for sp in mine for p in [sp.center]]
        ys = [p[1] for sp in mine for p in [sp.center]]
        x0, x1 = min(xs) - _CROP_MM, max(xs) + _CROP_MM
        y0, y1 = min(ys) - _CROP_MM, max(ys) + _CROP_MM
        w_px, h_px = int((x1 - x0) / _MM_PER_PX), int((y1 - y0) / _MM_PER_PX)
        doc = ezdxf.readfile(drawings[s.part.part_id])
        fig = plt.figure(figsize=(w_px / 100, h_px / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        conf = Configuration(background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.BLACK)
        Frontend(RenderContext(doc), MatplotlibBackend(ax), config=conf).draw_layout(doc.modelspace(), finalize=False)
        ax.set_xlim((x0 + ox) / scale, (x1 + ox) / scale)
        ax.set_ylim((y0 + oy) / scale, (y1 + oy) / scale)
        ax.set_aspect("auto")
        ax.axis("off")
        sheet = out / f"_{s.part.part_id}.png"
        fig.savefig(sheet, facecolor="white")
        plt.close(fig)
        page = Image.open(sheet).convert("RGB")
        to_px = lambda x, y: ((x - x0) / _MM_PER_PX, (y1 - y) / _MM_PER_PX)  # noqa: E731
        for sp in mine:
            cx, cy = to_px(*sp.center)
            r = _CROP_MM / _MM_PER_PX
            crop = page.crop((int(cx - r), int(cy - r), int(cx + r), int(cy + r))).resize((600, 600))
            draw = ImageDraw.Draw(crop)
            k = 600 / (2 * r)
            u, n = sp.direction, (-sp.direction[1], sp.direction[0])
            hw, ht = sp.width / 2 / _MM_PER_PX * k, (sp.thickness / 2 + 150) / _MM_PER_PX * k
            corners = [(300 + (a * u[0] * hw + b * n[0] * ht), 300 - (a * u[1] * hw + b * n[1] * ht)) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            draw.polygon(corners, outline=(230, 0, 0), width=3)
            path = out / f"{sp.floor}_{sp.opening_id}.png"
            crop.save(path)
            images[sp.key] = path
    return images


def ask_llm(spots: list[Spot], images: dict[str, Path], provider: str, model: str, directory: Path | None = None,
            evidence: dict[str, dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """지점마다 도면 조각과 질문을 보내 답을 받는다. 정해진 답이 아니면 버린다(대기 목록에 남는다).

    받은 답은 directory에 10개마다 저장한다. 한도가 다 되면 거기서 멈추고 받은 것까지만 돌려준다.
    """
    questions = {"outline": _OUTLINE_QUESTION, "corner": _CORNER_QUESTION, "window": _WINDOW_QUESTION}
    answers: dict[str, dict[str, Any]] = {}
    unsaved: dict[str, dict[str, Any]] = {}
    rejected: dict[str, dict[str, Any]] = {}
    for sp in spots:
        if sp.key not in images:
            continue
        try:
            info = (evidence or {}).get(sp.key) or {}
            prompt = "\n".join([drawing_notes(sp.kind) + info.get("text", ""), questions.get(sp.kind, _QUESTION).format(width=sp.width)])
            text = ask(provider, model, prompt, images[sp.key].read_bytes(), max_tokens=4000)
        except QuotaExhausted:
            break
        reply = extract_json(text)
        if isinstance(reply, dict) and reply.get("answer") in ANSWERS:
            answer = {"answer": reply["answer"], "reason": str(reply.get("reason", ""))[:200], "model": f"{provider}:{model}"}
            why = conflict(reply["answer"], info.get("counts") or {}, sp.kind)
            if why is None:
                answers[sp.key] = unsaved[sp.key] = answer
            else:  # 적용하지 않는다. 무엇을 왜 걸렀는지는 남겨 사람이 볼 수 있게 한다
                rejected[sp.key] = {**answer, "why": why}
        if directory is not None and len(unsaved) >= 10:
            save_decisions(directory, unsaved)
            unsaved = {}
    if directory is not None and unsaved:
        save_decisions(directory, unsaved)
    if directory is not None and (rejected or answers):
        save_rejected(directory, rejected, set(answers))
    return answers


def apply_decisions(storeys: list[Any], spots: list[Spot], decisions: dict[str, dict[str, Any]]) -> dict[str, int]:
    """답대로 개구부를 바꾼다. wall은 개구부를 없애고(벽이 이어짐), door·window는 종류를 정하고, 나머지는 둔다."""
    counts = {a: 0 for a in ANSWERS} | {"slit": 0}
    by_part: dict[str, dict[str, str]] = {}
    for sp in spots:
        if sp.kind in ("opening", "window") and sp.key in decisions:
            by_part.setdefault(sp.part_id, {})[sp.opening_id] = decisions[sp.key]["answer"]
    for s in storeys:
        chosen = by_part.get(s.part.part_id)
        if not chosen:
            continue
        ir = dict(s.part.ir)
        openings = []
        for o in ir["openings"]:
            answer = chosen.get(o["id"])
            if answer is None:
                openings.append(o)
                continue
            counts[answer] += 1
            if answer == "wall":
                continue
            if answer in ("door", "window") and answer != o["type"]:
                o = {**o, "type": answer, "operation": "fixed" if answer == "window" else "sliding"}
            elif answer == "open":
                o = {**o, "type": "unknown", "operation": None}  # 문짝·유리 없이 뚫린 통로
            openings.append(o)
        ir["openings"] = openings
        s.part = dataclasses.replace(s.part, ir=ir)
    for s in storeys:
        mine = [sp for sp in spots if sp.part_id == s.part.part_id and (
            (sp.kind in ("outline", "corner") and sp.key in decisions) or sp.kind == "slit")]
        if not mine:
            continue
        ir = dict(s.part.ir)
        walls, openings = list(ir["walls"]), list(ir["openings"])
        for sp in mine:
            answer = "wall" if sp.kind == "slit" else decisions[sp.key]["answer"]
            counts["slit" if sp.kind == "slit" else answer] += 1
            if answer not in ("wall", "window", "door"):
                continue
            # 외곽선 구간에 벽을 세운다. 위치·길이는 외곽선에서 계산한다(LLM은 종류만 골랐다).
            half = sp.width / 2
            wall_id = f"LW-{sp.floor}-{sp.opening_id}"
            start = [sp.center[0] - sp.direction[0] * half, sp.center[1] - sp.direction[1] * half]
            end = [sp.center[0] + sp.direction[0] * half, sp.center[1] + sp.direction[1] * half]
            if sp.path:  # L자 모서리: 두 벽을 각자 방향으로 늘려 만나게 한다. 개구부는 긴 쪽에 둔다
                pa, meet, pb = sp.path
                half_t = sp.thickness / 2  # 만나는 점 너머로 반 두께씩 늘려 모서리를 각지게 채운다
                legs = sorted([(pa, _past(pa, meet, half_t)), (_past(pb, meet, half_t), pb)], key=lambda leg: -math.dist(*leg))
                (start, end), rest = [list(legs[0][0]), list(legs[0][1])], legs[1]
                half = math.dist(start, end) / 2
                if math.dist(*rest) > 1.0:
                    walls.append({"id": wall_id + "b", "start": list(rest[0]), "end": list(rest[1]), "thickness": sp.thickness,
                                  "kind": "unknown", "confidence": {"level": "review", "score": 0.5, "reasons": ["llm_outline"]},
                                  "source_handles": []})
            walls.append({"id": wall_id, "start": start, "end": end, "thickness": sp.thickness, "kind": "unknown",
                          "confidence": {"level": "review", "score": 0.5, "reasons": ["llm_outline"]}, "source_handles": []})
            if answer in ("window", "door"):
                openings.append({"id": f"LO-{sp.floor}-{sp.opening_id}", "type": answer,
                                 "operation": "fixed" if answer == "window" else "sliding", "host_wall": wall_id,
                                 "position": half, "width": min(sp.width, 2 * half), "hinge": None, "swing_dir": None,
                                 "confidence": {"level": "review", "score": 0.5, "reasons": ["llm_outline"]}, "source_handles": []})
        ir["walls"], ir["openings"] = walls, openings
        s.part = dataclasses.replace(s.part, ir=ir)
    return counts


def _image_name(sp: Spot) -> str:
    return f"{sp.floor}_{sp.opening_id}.png"


def _write_items(directory: Path, spots: list[Spot], decisions: dict[str, dict[str, Any]], storeys: list[Any],
                 drawings: dict[str, Path], out_dir: Path, cfg: dict[str, Any]) -> None:
    """앱의 검토 패널이 읽을 전체 지점 목록을 쓴다. 답이 이미 있는 지점도 그림과 근거를 갖추게 한다.

    좌표는 아직 도면(IR) 기준이다. 건물 좌표는 층 배치와 공통 원점이 정해진 뒤 locate_items가 넣는다.
    """
    path = directory / "spots" / EVIDENCE
    evidence = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    bare = [sp for sp in spots if sp.key not in evidence or not (directory / "spots" / _image_name(sp)).is_file()]
    if bare:  # 답이 저장돼 있어 묻지 않은 지점은 그림·근거가 없을 수 있다. 사람이 볼 수 있게 만든다
        render_spots(bare, storeys, drawings, directory, cfg)
        evidence |= spot_evidence(bare, storeys, drawings, out_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(evidence, ensure_ascii=False, indent=1), encoding="utf-8")
    items = []
    rejected = load_rejected(directory)
    for sp in spots:
        proof = evidence.get(sp.key) or {}
        image = directory / "spots" / _image_name(sp)
        items.append({
            "key": sp.key, "kind": sp.kind, "floor": sp.floor, "part": sp.part_id, "opening": sp.opening_id,
            "width": round(sp.width), "thickness": round(sp.thickness),
            "center": [round(sp.center[0], 1), round(sp.center[1], 1)], "direction": [round(sp.direction[0], 4), round(sp.direction[1], 4)],
            "image": f"spots/{image.name}" if image.is_file() else None,
            "evidence": {"counts": proof.get("counts") or {}, "sides": proof.get("sides") or [], "text": proof.get("text", "")},
            "decision": decisions.get(sp.key),
            # LLM이 답했지만 도면 근거와 어긋나 적용하지 않은 답과 그 이유. LLM 답이 적용된 지점에는 없다
            "rejected": None if (decisions.get(sp.key) or {}).get("model", "user") != "user" else rejected.get(sp.key),
        })
    (directory / ITEMS).write_text(json.dumps({"answers": list(ANSWERS), "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")


def locate_items(storeys: list[Any], out_dir: Path, base: tuple[float, float]) -> int:
    """검토 지점에 건물 좌표(mm, Z 위. furniture.json과 같다)를 넣는다. 3D에서 그 자리로 가는 데 쓴다. 지점 수를 돌려준다.

    base는 모델 파일과 같은 공통 원점이다. 벽이 더해지면 원점이 바뀌므로 내보내기와 같은 값으로 불러야 한다.
    """
    path = Path(out_dir) / "_llm" / ITEMS
    if not path.is_file():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    by_part = {s.part.part_id: s for s in storeys}
    for item in data["items"]:
        s = by_part.get(item["part"])
        if s is None:
            item["position"] = None
            continue
        ox, oy = s.part.ir["origin_offset"]
        item["position"] = [
            round(item["center"][0] + ox + s.shift[0] - base[0], 1), round(item["center"][1] + oy + s.shift[1] - base[1], 1),
            round(s.elevation, 1),
        ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(data["items"])


def review(storeys: list[Any], drawings: dict[str, Path], out_dir: Path, cfg: dict[str, Any]) -> str | None:
    """LLM 검토를 돌리고 빌드 기록 한 줄을 돌려준다. 꺼져 있으면 None."""
    which = mode(cfg)
    if which == "off":
        return None
    directory = Path(out_dir) / "_llm"
    outline = find_outline_spots(storeys, cfg["export"]) if "export" in cfg else []
    # 좁은 세로 틈(0.4~1.5m): 벽 면에 거의 붙은 것만 덮였다고 보고 찾는다. 긴 빈 면 검토에 이미 걸린 곳은 뺀다.
    slots = [
        sp for sp in (find_outline_spots(storeys, cfg["export"], reach=20.0, min_mm=cfg["openings"]["min_gap_mm"], max_mm=1500.0, tag="s", step=20.0)
                      if "export" in cfg and "openings" in cfg else [])
        if not any(o.floor == sp.floor and math.dist(o.center, sp.center) < (o.width + sp.width) / 2 + 300 for o in outline)
    ]
    corners = find_corner_spots(storeys)
    windows = find_interior_windows(storeys, cfg["export"]) if "export" in cfg else []
    spots = find_spots(storeys) + outline + slots + windows + [sp for sp in corners if sp.kind != "slit"]
    slits = find_slits(storeys, cfg["export"], cfg["openings"]["min_gap_mm"]) if "export" in cfg and "openings" in cfg else []
    slits += [sp for sp in corners if sp.kind == "slit"]
    decisions = load_decisions(directory)
    images: dict[str, Path] = {}

    def ask_missing(batch: list[Spot]) -> None:
        """답이 없는 지점을 그려 근거와 함께 묻는다. 받은 답은 바로 decisions에 넣어 이번 빌드에 쓴다."""
        missing = [sp for sp in batch if sp.key not in decisions]
        if not missing:
            return
        images.update(render_spots(missing, storeys, drawings, directory, cfg))
        evidence = spot_evidence(missing, storeys, drawings, out_dir)
        (directory / "spots").mkdir(parents=True, exist_ok=True)
        path = directory / "spots" / EVIDENCE
        old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        path.write_text(json.dumps(old | evidence, ensure_ascii=False, indent=1), encoding="utf-8")
        if which in PROVIDERS:
            configure(cfg)
            decisions.update(ask_llm(missing, images, which, default_model(cfg, which), directory, evidence))

    # 묻고 나서 적용한다. 거꾸로 하면 새로 받은 답이 다음 빌드에서야 쓰인다.
    ask_missing(spots)
    counts = apply_decisions(storeys, spots + slits, decisions)
    # 답대로 세운 벽이 이웃 벽과 모서리에서 어긋날 수 있다. 세운 뒤 모서리를 한 번 더 본다.
    seen = {sp.key for sp in spots}
    again = [sp for sp in find_corner_spots(storeys) if sp.key not in seen]
    for sp in again:
        sp.opening_id = "R" + sp.opening_id  # 첫 회차와 벽 번호가 겹치지 않게
    late = [sp for sp in again if sp.kind != "slit"]
    spots += late
    ask_missing(late)
    later = apply_decisions(storeys, late + [sp for sp in again if sp.kind == "slit"], decisions)
    for k, v in later.items():
        counts[k] += v
    pending = [
        {"key": sp.key, "kind": sp.kind, "floor": sp.floor, "opening": sp.opening_id, "width": round(sp.width), "image": str(images.get(sp.key, ""))}
        for sp in spots if sp.key not in decisions
    ]
    directory.mkdir(parents=True, exist_ok=True)  # 답이 다 있으면 지난번 대기 목록이 남지 않게 비운다
    (directory / PENDING).write_text(json.dumps(pending, ensure_ascii=False, indent=1), encoding="utf-8")
    _write_items(directory, spots, decisions, storeys, drawings, out_dir, cfg)
    return f"llm_review: spots={len(spots)} decided={len(spots) - len(pending)} pending={len(pending)} " + " ".join(f"{k}={v}" for k, v in counts.items())
