"""해치를 레이어와 무관하게 위치와 패턴으로 찾는다. B2처럼 레이어가 틀린 해치도 근거가 된다."""
from __future__ import annotations

from typing import Any

from shapely import STRtree
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from ..preprocess.primitive import Primitive


def _hatch_polygon(p: Primitive) -> Polygon | None:
    rings = [Polygon(r).buffer(0) for r in p.geometry["rings"] if len(r) >= 3]
    rings = [r for r in rings if not r.is_empty]
    if not rings:
        return None
    rings.sort(key=lambda r: -r.area)
    shape = rings[0]
    for ring in rings[1:]:  # 짝수-홀수 규칙. 안쪽 고리는 구멍이다.
        shape = shape.symmetric_difference(ring)
    return shape


class HatchIndex:
    def __init__(self, prims: list[Primitive], patterns: dict[str, list[str]]):
        self.by_pattern = {p.upper(): kind for kind, names in patterns.items() for p in names}
        self.items: list[tuple[Any, str, str]] = []
        for p in prims:
            if p.kind != "hatch":
                continue
            shape = _hatch_polygon(p)
            if shape is not None:
                self.items.append((shape, p.geometry["pattern"], p.id))
        self.tree = STRtree([i[0] for i in self.items]) if self.items else None
        self.excluded: set[int] = set()

    def __len__(self) -> int:
        return len(self.items)

    def covers(self, polygon: Polygon, min_iou: float) -> bool:
        """polygon과 거의 같은 해치가 있으면 그 해치를 벽 근거에서 뺀다."""
        if self.tree is None:
            return False
        for i in self.tree.query(polygon):
            shape = self.items[int(i)][0]
            union = unary_union([shape, polygon]).area
            if union and shape.intersection(polygon).area / union >= min_iou:
                self.excluded.add(int(i))
                return True
        return False

    def kinds_at(self, points: list[tuple[float, float]]) -> list[str | None]:
        """점마다 그 자리 해치의 재질. 해치가 없으면 None, 패턴을 모르면 'unknown'."""
        result: list[str | None] = []
        for xy in points:
            found = None
            if self.tree is not None:
                point = Point(xy)
                for i in self.tree.query(point):
                    i = int(i)
                    if i not in self.excluded and self.items[i][0].covers(point):
                        found = self.by_pattern.get(self.items[i][1].upper(), "unknown")
                        break
            result.append(found)
        return result
