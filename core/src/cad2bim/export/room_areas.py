"""실 면적을 두 기준으로 낸다: 안목(벽 안쪽)과 벽 중심선 기준.

인식한 실의 다각형은 벽과 기둥을 뺀 안쪽 영역이라 면적이 안목 기준이다(IR의 clear_area_m2).
건축 도면의 실 면적 표기는 대개 벽 중심선 기준이어서, 같은 실인데도 표기보다 10~20% 작게 나온다.
형상이 틀린 것이 아니라 재는 기준이 다른 것이므로, 중심선 기준 면적을 함께 내어 도면 표기와 바로 견줄 수 있게 한다.

중심선 기준 영역 = 실 영역 + 둘레 벽의 '실 쪽 절반' + 실에 닿은 기둥. 좌표는 새로 만들지 않고 인식한 벽·기둥에서 계산한다.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from ..preprocess import load_primitives

FILE_NAME = "rooms.csv"
_AREA_NOTE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:㎡|m2|m²)", re.I)
_TOUCH_MM = 20.0  # 벽 절반이 실에 이만큼 안으로 닿아 있으면 그 실 쪽 절반이다
_LEAK_MM = 5.0  # 실 둘레로 벽 두께 절반보다 이만큼만 더 본다(이웃 벽으로 번지지 않게)
_BEYOND_MM = 1000.0  # 벽 중심선 너머 이만큼까지는 이웃 공간으로 보고 잘라 낸다(벽선 위 기둥이 넘어간 부분)


def _half_strips(wall: dict[str, Any], extra: float = 0.0) -> list[Polygon]:
    """벽을 중심선으로 가른 두 쪽. 끝은 두께 절반만큼 늘린다(모서리에서 이웃 벽과 빈틈없이 만난다).

    extra를 주면 그만큼 더 깊고 길게 잡는다(중심선 너머를 잘라 낼 때 쓴다).
    """
    (x0, y0), (x1, y1) = wall["start"][:2], wall["end"][:2]
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return []
    half = wall["thickness"] / 2
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    nx, ny = -uy, ux
    cap, depth = half + extra, half + extra
    a = (x0 - ux * cap, y0 - uy * cap)
    b = (x1 + ux * cap, y1 + uy * cap)
    return [
        Polygon([a, b, (b[0] + nx * depth * side, b[1] + ny * depth * side), (a[0] + nx * depth * side, a[1] + ny * depth * side)])
        for side in (1, -1)
    ]


def _column_polygon(column: dict[str, Any]) -> Polygon | None:
    if column.get("polygon"):
        return Polygon(column["polygon"])
    size = column.get("size") or [0.0, 0.0]
    w, h = (size if isinstance(size, list) else [size, size])[:2]
    cx, cy = column["center"][:2]
    if column.get("shape") == "circle":
        return Point(cx, cy).buffer(w / 2)
    angle = math.radians(column.get("rotation") or 0.0)
    c, s = math.cos(angle), math.sin(angle)
    return Polygon([(cx + dx * c - dy * s, cy + dx * s + dy * c) for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))])


def centerline_region(room: dict[str, Any], walls: dict[str, dict[str, Any]], columns: list[dict[str, Any]]) -> Any:
    """벽 중심선 기준의 실 영역(shapely 도형). 둘레 벽이 없으면 안목 영역 그대로다."""
    region = Polygon(room["polygon"]).buffer(0)
    around = [walls[i] for i in room.get("boundary_wall_ids", []) if i in walls]
    if region.is_empty or not around:
        return region
    touch = region.buffer(_TOUCH_MM)
    # 실을 파고든 기둥은 실에 넣는다. 벽선 위의 기둥이 중심선을 넘는 부분은 아래에서 잘라 낸다.
    posts = [shape for shape in (_column_polygon(c) for c in columns) if shape is not None and shape.is_valid and shape.intersects(touch)]
    body = unary_union([region, *posts]).buffer(0)
    reach = max(w["thickness"] for w in around) / 2 + _LEAK_MM
    near = body.buffer(reach, join_style="mitre")
    parts = [body]
    beyond = []
    for wall in around:
        sides = _half_strips(wall)
        facing = [strip.intersection(touch).area > 1.0 for strip in sides]
        # 실에 닿은 쪽 절반만 더한다. 실 안으로 튀어나온 벽은 양쪽이 다 닿으므로 통째로 들어간다.
        parts += [strip.intersection(near) for strip, mine in zip(sides, facing) if mine]
        if sum(facing) == 1:
            # 반대쪽은 이웃 공간의 몫이다. 두께가 다른 벽이 만나는 모서리나 기둥이 중심선을 넘은 부분을 잘라 낸다.
            far = _half_strips(wall, _BEYOND_MM)
            beyond.append(far[facing.index(False)])
    grown = unary_union(parts).buffer(0)
    if beyond:
        grown = grown.difference(unary_union(beyond))
    return unary_union([grown, region]).buffer(0)  # 잘라 내기가 실 영역 자체를 줄이지는 않는다


def room_areas(ir: dict[str, Any]) -> dict[str, dict[str, float]]:
    """실 id → {clear: 안목 ㎡, centerline: 벽 중심선 기준 ㎡}."""
    walls = {w["id"]: w for w in ir["walls"]}
    result = {}
    for room in ir["rooms"]:
        if not room.get("polygon") or len(room["polygon"]) < 3:
            continue
        clear = room.get("clear_area_m2")  # 예전 결과나 손으로 만든 IR에는 없을 수 있다
        clear = float(clear) if clear is not None else round(Polygon(room["polygon"]).buffer(0).area / 1e6, 3)
        result[room["id"]] = {
            "clear": clear,
            "centerline": max(round(centerline_region(room, walls, ir.get("columns", [])).area / 1e6, 3), clear),
        }
    return result


def _written_areas(storey: Any, drawing: Path | None, cfg: dict[str, Any], cache_dir: Path | None) -> list[tuple[float, float, float]]:
    """도면에 적힌 면적 글자: (x, y, ㎡). 좌표는 IR 기준이다. 도면을 못 읽으면 빈 목록."""
    if drawing is None:
        return []
    ir = storey.part.ir
    scale = float(ir["source"].get("unit_scale") or 1.0)
    ox, oy = ir["origin_offset"]
    try:
        raw, _ = load_primitives(drawing, cfg, cache_dir)
    except Exception:  # 면적 대조는 보조 정보다. 도면을 다시 못 읽어도 면적표는 낸다
        return []
    found = []
    for p in raw:
        if p.kind != "text":
            continue
        match = _AREA_NOTE.search(p.geometry["text"])
        if match:
            x, y = p.transformed(scale, -ox, -oy).geometry["position"][:2]
            found.append((float(x), float(y), float(match.group(1))))
    return found


def write_room_areas(storeys: list[Any], drawings: dict[str, Path], out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """실 면적표(rooms.csv)를 쓴다. 도면에 면적이 하나만 적힌 실은 그 값과 차이도 적는다. 요약 숫자를 돌려준다.

    표: 층, 실 id, 이름, 안목 ㎡, 중심선 ㎡, 도면 표기 ㎡, 중심선 대비 차이 %.
    """
    rows = []
    gaps = []
    for s in storeys:
        ir = s.part.ir
        areas = room_areas(ir)
        notes = _written_areas(s, drawings.get(s.part.part_id), cfg, Path(out_dir) / "_prims")
        for room in ir["rooms"]:
            mine = areas.get(room["id"])
            if mine is None:
                continue
            region = Polygon(room["polygon"]).buffer(0)
            written = [value for x, y, value in notes if region.contains(Point(x, y))]
            # 한 실에 면적 글자가 여러 개면(부속 공간, 가구 치수) 어느 것이 그 실의 면적인지 알 수 없어 비워 둔다
            note = written[0] if len(written) == 1 and written[0] > 0 else None
            gap = round((mine["centerline"] / note - 1) * 100, 1) if note else None
            if gap is not None:
                gaps.append(abs(gap))
            rows.append([
                s.key, room["id"], " / ".join(room["names"]), f"{mine['clear']:.2f}", f"{mine['centerline']:.2f}",
                "" if note is None else f"{note:.2f}", "" if gap is None else f"{gap:+.1f}",
            ])
    path = Path(out_dir) / FILE_NAME
    with path.open("w", encoding="utf-8-sig", newline="") as handle:  # 엑셀이 한글을 바로 읽게 BOM을 붙인다
        writer = csv.writer(handle)
        writer.writerow(["층", "실 id", "이름", "안목 면적(㎡)", "중심선 면적(㎡)", "도면 표기(㎡)", "중심선 대비 차이(%)"])
        writer.writerows(rows)
    gaps.sort()
    return {
        "rooms": len(rows), "with_note": len(gaps),
        "within_3pct": sum(g <= 3 for g in gaps), "within_5pct": sum(g <= 5 for g in gaps),
        "median_gap_pct": gaps[len(gaps) // 2] if gaps else "-", "max_gap_pct": gaps[-1] if gaps else "-",
    }
