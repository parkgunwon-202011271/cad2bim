"""가구 종류마다 입체를 만든다. 바닥 모양은 도면 그대로 쓰고 높이와 부재(다리, 등받이, 상판 …)는 흔한 치수를 따른다.

평면도에는 높이가 없어 높이는 추정이다. 입체는 모두 '바닥 모양을 z0에서 z1까지 세운 기둥'이라 뷰어가 그대로 그린다.
부재마다 재질(m)을 달아 색을 나눈다: wood, wood_dark, metal, steel, fabric, ceramic, glass, panel, top, black.
"""
from __future__ import annotations

import math
from typing import Any

from shapely import affinity
from shapely.geometry import Point, Polygon

from .furniture_items import Frame, Item, basins, burners, rect_share

Part = dict[str, Any]

# 사람 시점에서 지나갈 수 없는 가구는 뷰어가 kind로 가른다(의자·탁자는 사이로 지나다닌다).
_SEAT_MM = 450.0
_TABLE_TOP = (720.0, 750.0)
_COUNTER_MM = 850.0
_PLINTH_MM = 100.0
_TOP_MM = 40.0


def _part(kind: str, material: str, polygon: Any, z0: float, z1: float, holes: list[Polygon] | None = None) -> list[Part]:
    if not isinstance(polygon, Polygon) or polygon.is_empty or z1 <= z0:
        return []
    kept = [h for h in holes or [] if isinstance(h, Polygon) and not h.is_empty and polygon.contains(h)]
    return [{"k": kind, "m": material, "polygon": polygon, "holes": kept, "z0": z0, "z1": z1}]


def _inset(polygon: Polygon, by: float) -> Polygon:
    """안으로 들인 모양. 너무 얇아 사라지면 원래 모양을 쓴다."""
    inner = polygon.buffer(-by, join_style="mitre")
    return inner if isinstance(inner, Polygon) and not inner.is_empty else polygon


def _seat_frame(item: Item) -> tuple[Frame, float, float]:
    """(앉는 방향에 맞춘 틀, 폭, 깊이). 틀의 v는 등받이 쪽을 본다. 등받이를 모르면 짧은 변 쪽으로 둔다."""
    f = item.frame
    back = item.band or f.v
    side = (-back[1], back[0])
    along_u = abs(back[0] * f.u[0] + back[1] * f.u[1]) > 0.5
    width, depth = (f.depth, f.length) if along_u else (f.length, f.depth)
    return Frame(f.center, side, back, width, depth), width, depth


def _legs(kind: str, frame: Frame, width: float, depth: float, z1: float, size: float = 35.0, inset: float = 30.0) -> list[Part]:
    parts: list[Part] = []
    for su in (-1, 1):
        for sv in (-1, 1):
            leg = frame.rect(size, size, su * (width / 2 - inset), sv * (depth / 2 - inset))
            parts += _part(kind, "metal", leg, 0.0, z1)
    return parts


def chair(item: Item) -> list[Part]:
    frame, width, depth = _seat_frame(item)
    seat_top = _SEAT_MM
    parts = _legs("chair", frame, width, depth, seat_top - 50.0)
    if item.band is None:
        return parts + _part("chair", "fabric", item.polygon, seat_top - 50.0, seat_top)
    back = min(max(item.band_mm or 45.0, 35.0), 70.0)
    parts += _part("chair", "fabric", frame.rect(width, depth - back, 0.0, -back / 2), seat_top - 50.0, seat_top)
    parts += _part("chair", "fabric", frame.rect(width, back, 0.0, depth / 2 - back / 2), seat_top - 50.0, 880.0)
    return parts


def sofa(item: Item, height: float = 820.0) -> list[Part]:
    frame, width, depth = _seat_frame(item)
    parts = _part("sofa", "metal", _inset(item.polygon, 40.0), 0.0, _PLINTH_MM)
    if item.band is None:
        return parts + _part("sofa", "fabric", item.polygon, _PLINTH_MM, 430.0)
    back = min(max(item.band_mm, 110.0), depth * 0.35)
    parts += _part("sofa", "fabric", frame.rect(width, depth - back, 0.0, -back / 2), _PLINTH_MM, 430.0)
    parts += _part("sofa", "fabric", frame.rect(width, back, 0.0, depth / 2 - back / 2), _PLINTH_MM, height)
    return parts


def stool(item: Item, height: float = _SEAT_MM) -> list[Part]:
    c = item.polygon.centroid
    kind = "stool"
    return (
        _part(kind, "metal", affinity.scale(item.polygon, 0.7, 0.7, origin=c), 0.0, 25.0)
        + _part(kind, "metal", Point(c.x, c.y).buffer(28.0, quad_segs=3), 25.0, height - 45.0)
        + _part(kind, "fabric", item.polygon, height - 45.0, height)
    )


def table(item: Item, top: tuple[float, float] = _TABLE_TOP) -> list[Part]:
    polygon = item.polygon
    f = item.frame
    parts = _part("table", "wood", polygon, *top)
    # 사각형을 다 채우지 못하거나 꼭짓점이 많으면 둥근 탁자다: 가운데 기둥과 받침.
    if rect_share(polygon) < 0.85 or len(polygon.exterior.coords) > 8:
        c = polygon.centroid
        parts += _part("table", "metal", Point(c.x, c.y).buffer(45.0, quad_segs=3), 25.0, top[0])
        parts += _part("table", "metal", affinity.scale(polygon, 0.45, 0.45, origin=c), 0.0, 25.0)
        return parts
    return parts + [dict(p, k="table") for p in _legs("table", f, f.length, f.depth, top[0], size=50.0, inset=75.0)]


def counter(item: Item, height: float = _COUNTER_MM, body: str = "panel", top: str = "top", kind: str = "counter") -> list[Part]:
    polygon = item.polygon
    return (
        _part(kind, "metal", _inset(polygon, 40.0), 0.0, _PLINTH_MM)
        + _part(kind, body, polygon, _PLINTH_MM, height - _TOP_MM)
        + _part(kind, top, polygon, height - _TOP_MM, height)
    )


def sink(item: Item, height: float = _COUNTER_MM, body: str = "steel", top: str = "steel", bowl: str = "steel") -> list[Part]:
    polygon = item.polygon
    inner = _inset(polygon, 30.0)
    bowls = [b.intersection(inner) for b in basins(item)]
    bowls = [b for b in bowls if isinstance(b, Polygon) and not b.is_empty]
    if not bowls:
        f = item.frame
        bowls = [f.rect(min(f.length * 0.6, 700.0), f.depth * 0.6)]
    floor = height - 200.0
    parts = _part("sink", "metal", _inset(polygon, 40.0), 0.0, _PLINTH_MM) + _part("sink", body, polygon, _PLINTH_MM, floor)
    parts += _part("sink", body, polygon, floor, height - _TOP_MM, bowls)
    parts += _part("sink", top, polygon, height - _TOP_MM, height, bowls)
    for b in bowls:
        parts += _part("sink", bowl, b, floor, floor + 6.0)
    return parts


def stove(item: Item, height: float = _COUNTER_MM) -> list[Part]:
    polygon = item.polygon
    parts = counter(item, height, "steel", "steel", "stove")
    parts += _part("stove", "black", _inset(polygon, 25.0), height, height + 10.0)
    rings = burners(item)
    if not rings:
        f = item.frame
        r = min(f.length, f.depth) * 0.16
        spots = [(-f.length / 4, 0.0), (f.length / 4, 0.0)] if f.length > f.depth * 1.5 else [(-f.length / 4, -f.depth / 4), (f.length / 4, -f.depth / 4), (-f.length / 4, f.depth / 4), (f.length / 4, f.depth / 4)]
        rings = [(f.center[0] + f.u[0] * du + f.v[0] * dv, f.center[1] + f.u[1] * du + f.v[1] * dv, r) for du, dv in spots]
    for x, y, r in rings:
        parts += _part("stove", "metal", Point(x, y).buffer(r, quad_segs=4), height + 10.0, height + 32.0)
    return parts


def tall(item: Item, height: float = 2000.0, material: str = "panel", kind: str = "tall") -> list[Part]:
    return _part(kind, "metal", _inset(item.polygon, 20.0), 0.0, 80.0) + _part(kind, material, item.polygon, 80.0, height)


def showcase(item: Item, height: float = 1250.0) -> list[Part]:
    polygon = item.polygon
    return (
        _part("showcase", "metal", _inset(polygon, 40.0), 0.0, _PLINTH_MM)
        + _part("showcase", "wood_dark", polygon, _PLINTH_MM, 700.0)
        + _part("showcase", "steel", _inset(polygon, 40.0), 960.0, 972.0)
        + _part("showcase", "glass", _inset(polygon, 8.0), 700.0, height - 30.0)
        + _part("showcase", "steel", polygon, height - 30.0, height)
    )


def shelf(item: Item, height: float = 1800.0) -> list[Part]:
    """열린 선반: 양옆 판과 층마다 놓은 선반 판. 사각형이 아니거나 너무 얇으면 한 덩어리로 둔다."""
    f = item.frame
    polygon = item.polygon
    if rect_share(polygon) < 0.95 or f.depth < 150.0 or f.length < 300.0:
        return _part("shelf", "wood", polygon, 0.0, height)
    side = 25.0
    parts: list[Part] = []
    for sign in (-1, 1):
        parts += _part("shelf", "wood", f.rect(side, f.depth, sign * (f.length / 2 - side / 2)), 0.0, height)
    levels = max(2, round((height - 80.0) / 380.0))
    for i in range(levels + 1):
        z = 60.0 + (height - 85.0) * i / levels
        parts += _part("shelf", "wood", f.rect(f.length - side * 2, f.depth), z, z + 25.0)
    return parts


def cabinet(item: Item, height: float = 1800.0) -> list[Part]:
    """문이 달린 수납장: 받침과 몸통."""
    return _part("cabinet", "metal", _inset(item.polygon, 30.0), 0.0, 80.0) + _part("cabinet", "wood", item.polygon, 80.0, height)


def toilet(item: Item) -> list[Part]:
    bowl = item.polygon
    parts = _part("toilet", "ceramic", _inset(bowl, 45.0), 0.0, 330.0) + _part("toilet", "ceramic", bowl, 330.0, 400.0)
    ring = _inset(bowl, 70.0)
    parts += _part("toilet", "ceramic", bowl, 400.0, 425.0, [ring] if ring is not bowl else None)
    if item.tank is not None:
        parts += _part("toilet", "ceramic", item.tank, 380.0, 800.0)
    return parts


def upper_shelves(item: Item) -> list[Part]:
    """상부 선반: 가구 위에 띄운 선반 판 둘."""
    parts: list[Part] = []
    for z in (1450.0, 1800.0):
        parts += _part("counter", "steel", item.polygon, z, z + 25.0)
    return parts


def box(item: Item, kind: str, material: str, z0: float, z1: float) -> list[Part]:
    return _part(kind, material, item.polygon, z0, z1)


def build(item: Item) -> list[Part]:
    """한 가구의 입체 목록. 이름표 규칙에 높이(z0, z1)가 있으면 그 높이를 쓴다."""
    parts = _build(item)
    return parts + upper_shelves(item) if parts and item.upper else parts


def _build(item: Item) -> list[Part]:
    rule = item.rule or {}
    z0, z1 = rule.get("z0"), rule.get("z1")
    height = float(z1) if z1 is not None else None
    material = rule.get("m")
    kind = item.kind
    match kind:
        case "chair":
            return chair(item)
        case "sofa":
            return sofa(item, height or 820.0)
        case "stool":
            return stool(item, height or _SEAT_MM)
        case "bar_stool":
            return stool(item, height or 750.0)
        case "table":
            return table(item)
        case "counter":
            return counter(item, height or _COUNTER_MM, material or "panel", "steel" if material == "steel" else "top")
        case "bar":
            return counter(item, height or 1050.0, material or "wood_dark", "wood")
        case "sink":
            return sink(item, height or _COUNTER_MM)
        case "vanity":
            return sink(item, height or _COUNTER_MM, "panel", "ceramic", "ceramic") if basins(item) else counter(item, height or _COUNTER_MM, "panel", "ceramic", "vanity")
        case "stove":
            return stove(item, height or _COUNTER_MM)
        case "tall":
            return tall(item, height or 2000.0)
        case "fridge":
            return tall(item, height or 1900.0, "steel", "fridge")
        case "showcase":
            return showcase(item, height or 1250.0)
        case "shelf":
            return shelf(item, height or (1800.0 if item.frame.length >= 900.0 else 1000.0))
        case "cabinet":
            return cabinet(item, height or 1800.0)
        case "toilet":
            if item.tank is not None or rect_share(item.polygon) < 0.93:
                return toilet(item)
            return box(item, "fixture", "ceramic", 0.0, 400.0)  # 볼 모양이 아닌 사각 기호
        case "urinal":
            return box(item, "fixture", "ceramic", float(z0) if z0 is not None else 350.0, height or 1000.0)
        case "partition":
            return box(item, "partition", "panel", 0.0, height or 1200.0)
        case "stall":
            return box(item, "partition", "panel", 100.0, 2000.0)  # 화장실 칸막이. 바닥에서 조금 띄운다
        case "fixture_low":
            return box(item, "fixture", "ceramic", 0.0, 400.0)
        case "fixture":
            return box(item, "fixture", "ceramic", float(z0 or 0.0), height or 800.0)
        case "appliance":
            return box(item, "appliance", material or "steel", float(z0 or 0.0), height or 1200.0)
        case "block":
            return box(item, "block", "panel", 0.0, height or _SEAT_MM)
    return []


def describe(kind: str) -> str:
    return _NAMES.get(kind, kind)


_NAMES = {
    "chair": "의자", "sofa": "소파", "stool": "스툴", "bar_stool": "바 의자", "table": "탁자", "counter": "조리대·카운터", "bar": "바 테이블",
    "sink": "개수대", "vanity": "세면대", "stove": "가스레인지", "tall": "키 큰 장", "fridge": "냉장고", "showcase": "진열장", "shelf": "선반",
    "cabinet": "수납장", "toilet": "변기", "urinal": "소변기", "partition": "칸막이", "stall": "화장실 칸막이",
    "fixture_low": "위생기구", "fixture": "위생기구", "appliance": "기기", "block": "기타",
}
