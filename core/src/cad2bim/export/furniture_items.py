"""도면의 가구 선에서 가구 하나하나(Item)를 찾고 종류를 정한다.

가구를 그리는 방식은 도면마다 다르다. 닫힌 도형(사각형·원) 하나로 그리기도 하고 낱개 선과 호를 이어 그리기도 한다.
그래서 닫힌 도형과 '선들이 둘러싼 면'을 함께 후보로 삼고, 큰 것부터 골라 그 안의 작은 도형은 세부(details)로 붙인다.

종류는 다음 순서로 정한다. 좌표는 만들지 않는다. 바닥 모양은 언제나 도면에 그린 그대로다.
1. 이름표: 가구 안이나 곁의 글자, 가구를 가리키는 지시선 글자(config의 furniture_labels)
2. 기호: X 표시(키 큰 장), 빗금 여러 줄(유리 진열장), 사선 한 줄(수납장), 버너 원(가스레인지), 볼(개수대), 등받이 띠(의자·소파)
3. 크기: 위 둘로 정하지 못하면 크기로 가늠한다
4. 같은 모양 퍼뜨리기: 이름표가 붙은 가구와 크기·기호가 같은 가구는 같은 종류다. 조리대에 붙은 같은 깊이의 칸은 조리대다
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, polygonize

_MIN_AREA_MM2 = 0.01e6  # 이보다 작은 도형은 손잡이·기호다
_MAX_AREA_MM2 = 12e6  # 이보다 큰 도형은 가구가 아니라 구역 표시다
_NESTED_SHARE = 0.9  # 더 큰 도형에 이만큼 들어가면 그 가구의 세부다
_EXTEND_MM = 2.0  # 선 끝을 이만큼 늘여 닿을 듯 말 듯한 선을 잇는다
_TOUCH_MM = 3.0
_GRID_MM = 0.1
_PERPENDICULAR = math.sin(math.radians(8.0))
_OBLIQUE = (math.cos(math.radians(80.0)), math.cos(math.radians(10.0)))  # 상자의 변에서 10도 넘게 기운 선
_RECT_SHARE = 0.95  # 최소 사각형의 이만큼을 채우면 사각형이다
_BAND_MAX_MM = 200.0  # 가장자리 띠(등받이, 조리대 앞턱)의 최대 두께
_BACK_MM = (40.0, 130.0)  # 등받이로 보는 띠 두께. 이보다 얇으면 조리대 뒤턱, 두꺼우면 앞턱·선반이다
_ADJACENT_MM = 20.0
# 위생기구 레이어 이름. 같은 FURNITURE 역할이라도 높이와 모양이 다르다.
_FIXTURE = ("fixt", "plumb", "위생", "toilet", "변기")
_COUNTER_RUN = ("counter", "sink", "stove")
_PICTURED = ("stove", "sink")  # 그림만으로 분명한 기호


@dataclass
class Frame:
    """바닥 모양을 감싸는 가장 작은 사각형. u는 긴 변, v는 짧은 변 방향이다."""

    center: tuple[float, float]
    u: tuple[float, float]
    v: tuple[float, float]
    length: float
    depth: float

    def rect(self, along: float, across: float, du: float = 0.0, dv: float = 0.0) -> Polygon:
        """중심에서 (du, dv)만큼 옮긴 자리의 along × across 사각형."""
        cx = self.center[0] + self.u[0] * du + self.v[0] * dv
        cy = self.center[1] + self.u[1] * du + self.v[1] * dv
        hx, hy = self.u[0] * along / 2, self.u[1] * along / 2
        wx, wy = self.v[0] * across / 2, self.v[1] * across / 2
        return Polygon([(cx - hx - wx, cy - hy - wy), (cx + hx - wx, cy + hy - wy), (cx + hx + wx, cy + hy + wy), (cx - hx + wx, cy - hy + wy)])


def frame_of(polygon: Polygon) -> Frame:
    box = polygon.minimum_rotated_rectangle
    if not isinstance(box, Polygon) or box.is_empty:
        c = polygon.centroid
        return Frame((c.x, c.y), (1.0, 0.0), (0.0, 1.0), 0.0, 0.0)
    ring = list(box.exterior.coords)
    a, b = math.dist(ring[0], ring[1]), math.dist(ring[1], ring[2])
    e1 = ((ring[1][0] - ring[0][0]) / a, (ring[1][1] - ring[0][1]) / a) if a else (1.0, 0.0)
    e2 = ((ring[2][0] - ring[1][0]) / b, (ring[2][1] - ring[1][1]) / b) if b else (0.0, 1.0)
    c = box.centroid
    if a >= b:
        return Frame((c.x, c.y), e1, e2, a, b)
    return Frame((c.x, c.y), e2, e1, b, a)


def rect_share(polygon: Polygon) -> float:
    box = polygon.minimum_rotated_rectangle
    return polygon.area / box.area if box.area > 0 else 0.0


@dataclass
class Item:
    polygon: Polygon
    layer: str
    drawn: bool  # 닫힌 도형으로 그렸다(선을 이어 만든 면이 아니다)
    details: list[Polygon] = field(default_factory=list)
    slashes: list[LineString] = field(default_factory=list)
    rings: list[tuple[float, float, float]] = field(default_factory=list)  # 안에 그린 원·호 (x, y, 반지름)
    band: tuple[float, float] | None = None  # 가장자리 띠(등받이)가 붙은 쪽. 중심에서 띠 쪽을 보는 단위 벡터
    band_mm: float = 0.0
    edges: list[tuple[tuple[float, float], float]] = field(default_factory=list)  # 붙인 띠들 (방향, 두께)
    tank: Polygon | None = None  # 변기 물탱크
    kind: str = ""
    symbol: str | None = None
    upper: bool = False  # 위에 상부 선반이 달렸다
    pointed: bool = False  # 지시선이 직접 가리켰다
    rule: dict[str, Any] | None = None
    how: str = ""  # 종류를 정한 근거: label | symbol | size | same_shape | run

    @property
    def fixture(self) -> bool:
        return any(word in self.layer.lower() for word in _FIXTURE)

    @property
    def frame(self) -> Frame:
        return frame_of(self.polygon)

    @property
    def footprint(self) -> Polygon:
        """물탱크까지 넣은 바닥 모양."""
        if self.tank is None:
            return self.polygon
        merged = self.polygon.union(self.tank.buffer(_ADJACENT_MM)).buffer(0)
        return merged if isinstance(merged, Polygon) else self.polygon


# ---------------------------------------------------------------------------------------------------------------------
# 1. 후보 찾기
# ---------------------------------------------------------------------------------------------------------------------
def _extended(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """열린 선의 양 끝을 조금 늘인다. 끝이 다른 선에 닿을 듯 말 듯해도 면이 닫히게 한다."""
    if len(points) < 2 or points[0] == points[-1]:
        return points
    out = list(points)
    for end, near in ((0, 1), (-1, -2)):
        (x, y), (nx, ny) = out[end], out[near]
        d = math.hypot(x - nx, y - ny)
        if d > 1e-9:
            out[end] = (x + (x - nx) / d * _EXTEND_MM, y + (y - ny) / d * _EXTEND_MM)
    return out


def _slash_candidates(straight: list[LineString]) -> set[int]:
    """어느 끝에서도 직각으로 만나는 선이 없는 직선. 상자 안에 그은 사선(X 표시, 유리 빗금)이 여기에 든다.

    상자의 변은 끝에서 이웃 변과 직각으로 만난다. 기울어진 가구도 마찬가지라 가구 방향은 따지지 않는다.
    """
    if not straight:
        return set()
    tree = shapely.STRtree(straight)
    found = set()
    for i, line in enumerate(straight):
        (ax, ay), (bx, by) = line.coords[0], line.coords[-1]
        length = math.hypot(bx - ax, by - ay)
        if length < 100.0:
            continue
        ux, uy = (bx - ax) / length, (by - ay) / length
        square = False
        for end in ((ax, ay), (bx, by)):
            point = Point(end)
            for j in tree.query(point.buffer(_TOUCH_MM)).tolist():
                if j == i or straight[j].distance(point) > _TOUCH_MM:
                    continue
                (cx, cy), (dx, dy) = straight[j].coords[0], straight[j].coords[-1]
                other = math.hypot(dx - cx, dy - cy)
                if other > 1e-9 and abs(ux * (dx - cx) + uy * (dy - cy)) / other <= _PERPENDICULAR:
                    square = True
                    break
            if square:
                break
        if not square:
            found.add(i)
    return found


def _faces(lines: list[LineString]) -> list[Polygon]:
    """선들이 둘러싼 면. 거의 겹치는 선(소수점 아래에서만 어긋난 선)이 면을 깨지 않게 0.1mm 격자에 맞춰 잇는다."""
    if not lines:
        return []
    noded = shapely.unary_union(shapely.set_precision(shapely.multilinestrings(lines), _GRID_MM), grid_size=_GRID_MM)
    return [f for f in polygonize(noded) if f.area >= _MIN_AREA_MM2]


def _thick(polygon: Polygon) -> bool:
    """가구답게 두께가 있는 면인가. 가구 사이의 가느다란 틈과 다른 가구를 감싼 테두리는 안으로 조금만 깎아도 사라진다."""
    return polygon.buffer(-60.0).area >= polygon.area * 0.3


def _solidity(polygon: Polygon) -> float:
    hull = polygon.convex_hull.area
    return polygon.area / hull if hull > 0 else 0.0


def _fill_small_holes(face: Polygon) -> Polygon | None:
    """안에 그린 세부(볼, 서랍)가 만든 구멍은 메운다. 구멍이 절반을 넘으면 가구가 아니라 가구로 둘러싸인 바닥이다."""
    if not face.interiors:
        return face
    filled = Polygon(face.exterior)
    return filled if filled.area - face.area <= filled.area * 0.5 else None


def _whole_circles(rings: list[tuple[float, float, float, float, str]]) -> list[tuple[Polygon, str]]:
    """호 여러 개로 나눠 그린 원. 같은 중심·반지름의 호가 거의 한 바퀴를 채우면 원 하나로 본다(둥근 의자)."""
    groups: dict[tuple[int, int, int], list[Any]] = {}
    for x, y, r, sweep, layer in rings:
        group = groups.setdefault((round(x / 5.0), round(y / 5.0), round(r / 5.0)), [0.0, False, x, y, r, layer])
        group[0] += sweep
        group[1] = group[1] or sweep >= 359.0
    return [
        (Point(x, y).buffer(r, quad_segs=6), layer)
        for total, whole, x, y, r, layer in groups.values() if not whole and total >= 300.0 and r >= 100.0
    ]


def find_items(
    shapes: list[tuple[Polygon, str]], chains: list[tuple[list[tuple[float, float]], str, bool]],
    rings: list[tuple[float, float, float, float, str]] | None = None,
) -> list[Item]:
    """큰 것부터 고른 가구 목록.

    shapes: 닫힌 도형 (모양, 레이어). chains: 모든 가구 선 (점 목록, 레이어, 직선 한 토막인가).
    rings: 원과 호 (x, y, 반지름, 도는 각도, 레이어). 버너처럼 끊어 그린 원을 알아보는 데 쓴다.
    """
    rings = rings or []
    shapes = shapes + [c for c in _whole_circles(rings) if _MIN_AREA_MM2 <= c[0].area <= _MAX_AREA_MM2]
    straight, straight_layers, others, other_layers = [], [], [], []
    for points, layer, is_line in chains:
        if len(points) < 2:
            continue
        if is_line and len(points) == 2:
            straight.append(LineString(_extended(points)))
            straight_layers.append(layer)
        else:
            others.append(LineString(_extended(points)))
            other_layers.append(layer)
    # 직각 이웃은 닫힌 사각형의 변에서도 찾아야 한다. 변마다 한 토막으로 풀어 함께 본다.
    edges = [LineString([a, b]) for line in others for a, b in zip(line.coords, line.coords[1:]) if a != b]
    candidates = {i for i in _slash_candidates(straight + edges) if i < len(straight)}
    base = [line for i, line in enumerate(straight) if i not in candidates] + others
    slashes: list[LineString] = []
    if candidates:
        # 사선을 빼면 반듯한 사각형이 드러나는 경우에만 사선으로 본다. 아니면 그 선은 가구의 변이다(육각 탁자 등).
        boxes = [f for f in _faces(base) if rect_share(f) >= _RECT_SHARE]
        boxes += [polygon for polygon, _ in shapes if rect_share(polygon) >= _RECT_SHARE]
        tree = shapely.STRtree(boxes) if boxes else None
        for i in sorted(candidates):
            line = straight[i]
            if tree is not None and any(boxes[j].buffer(_TOUCH_MM * 2).contains(line) for j in tree.query(line).tolist()):
                slashes.append(line)
            else:
                base.append(line)
    faces = _faces(base)

    lines_all = straight + others
    layers_all = straight_layers + other_layers
    line_tree = shapely.STRtree(lines_all) if lines_all else None

    def face_layer(face: Polygon) -> str:
        return layers_all[int(line_tree.nearest(face.representative_point()))] if line_tree is not None else ""

    found: list[tuple[Polygon, str, bool]] = [(polygon, layer, True) for polygon, layer in shapes]
    plain: list[Polygon] = []
    wrapping: list[Polygon] = []
    for face in faces:
        filled = _fill_small_holes(face)
        if filled is None:
            continue
        if rect_share(filled) < 0.9 and _solidity(filled) < 0.8:
            wrapping.append(filled)
        elif rect_share(filled) >= 0.9 or _thick(filled):
            plain.append(filled)
    face_tree = shapely.STRtree(faces) if faces else None
    for face in wrapping:
        # 안에 그린 기구(레인지, 개수대)가 가장자리에 닿아 ㄷ자로 남은 칸: 감싼 사각형이 그 칸이다.
        # 다른 면이 그 사각형에 걸쳐 있으면(일부만 들어오면) 칸이 아니라 가구 사이의 틈이다.
        hull = face.convex_hull
        wraps = rect_share(hull) >= _RECT_SHARE and face.area >= hull.area * 0.3 and hull.area <= _MAX_AREA_MM2
        if wraps:
            shares = [faces[j].intersection(hull).area / faces[j].area for j in face_tree.query(hull).tolist()]
            wraps = all(s >= _NESTED_SHARE or s <= 1 - _NESTED_SHARE for s in shares)
        if wraps:
            plain.append(hull)
        elif _thick(face):
            plain.append(face)  # ㄱ자 조리대처럼 사각형이 아닌 가구
    # 다른 가구의 변에 걸쳐 둘로 갈린 둥근 의자: 두 조각을 합친 원을 후보로 더한다.
    halves = [f for f in plain if rect_share(f) < 0.9]
    rounds: list[tuple[Polygon, list[Polygon]]] = []
    if halves:
        half_tree = shapely.STRtree(halves)
        for i, a in enumerate(halves):
            for j in half_tree.query(a.buffer(_TOUCH_MM)).tolist():
                if j <= i or halves[j].distance(a) > _TOUCH_MM:
                    continue
                merged = a.buffer(_TOUCH_MM).union(halves[j].buffer(_TOUCH_MM)).buffer(-_TOUCH_MM)
                if isinstance(merged, Polygon) and _is_round(merged) and merged.area <= (a.area + halves[j].area) * 1.05:
                    rounds.append((merged, [a, halves[j]]))
    found += [(face, face_layer(face), False) for face in plain] + [(merged, face_layer(merged), False) for merged, _ in rounds]
    found = [c for c in found if _MIN_AREA_MM2 <= c[0].area]
    found.sort(key=lambda c: (-c[0].area, not c[2]))

    items: list[Item] = []
    for polygon, layer, drawn in found:
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
            if not isinstance(polygon, Polygon) or polygon.is_empty:
                continue
        parent = None
        for item in items:
            if not item.polygon.intersects(polygon):
                continue
            if drawn and item.drawn and item.layer != layer:
                continue  # 다른 레이어에 따로 그린 도형은 따로 본다(탁자 위의 기구 등)
            if polygon.intersection(item.polygon).area >= polygon.area * _NESTED_SHARE:
                parent = item
                break
        if parent is not None:
            if polygon.area < parent.polygon.area * 0.95:
                parent.details.append(polygon)
            elif drawn and not parent.drawn:
                parent.drawn, parent.layer = True, layer
            continue
        if polygon.area > _MAX_AREA_MM2:
            continue
        items.append(Item(polygon, layer, drawn))
    # 둥근 의자가 파먹은 자리를 메운다: 의자 조각을 더하면 둘레가 짧아지는 이웃 가구(바 테이블)는 그 조각까지가 제 모양이다.
    for seat in [item for item in items if _is_round(item.polygon) and item.details]:
        for part in seat.details:
            for item in items:
                if item is seat or item.polygon.area < seat.polygon.area * 2 or item.polygon.distance(part) > _TOUCH_MM:
                    continue
                grown = item.polygon.union(part.buffer(_TOUCH_MM)).buffer(0)
                if isinstance(grown, Polygon) and grown.exterior.length < item.polygon.exterior.length - 1.0:
                    item.polygon = grown
    for x, y, r, _, _ in rings:
        if r < 30.0:
            continue
        center = Point(x, y)
        for item in items:
            if item.polygon.contains(center):
                item.rings.append((x, y, r))
                break
    if slashes and items:
        tree = shapely.STRtree([item.polygon for item in items])
        for line in slashes:
            middle = line.interpolate(0.5, normalized=True)
            for j in tree.query(middle).tolist():
                if items[j].polygon.contains(middle):
                    items[j].slashes.append(line)
                    break
    return items


# ---------------------------------------------------------------------------------------------------------------------
# 2. 붙여 읽기: 등받이 띠, 변기 물탱크
# ---------------------------------------------------------------------------------------------------------------------
def _is_bowl(polygon: Polygon) -> bool:
    """변기 볼: 한쪽이 둥글고 길쭉한 작은 도형."""
    f = frame_of(polygon)
    return 280.0 <= f.depth <= 480.0 and 380.0 <= f.length <= 760.0 and f.length >= f.depth * 1.1 and 0.68 <= rect_share(polygon) <= 0.93


def _is_round(polygon: Polygon) -> bool:
    f = frame_of(polygon)
    return f.depth > 0 and f.length / f.depth <= 1.12 and 0.72 <= rect_share(polygon) <= 0.86


def _attach_tanks(items: list[Item], gone: set[int]) -> None:
    """변기: 볼 + 곁의 작은 사각형(물탱크)."""
    tree = shapely.STRtree([item.polygon for item in items])
    for i, item in enumerate(items):
        if i in gone or not _is_bowl(item.polygon):
            continue
        for j in tree.query(item.polygon.buffer(_ADJACENT_MM)).tolist():
            other = items[j]
            if j == i or j in gone or other.polygon.distance(item.polygon) > _ADJACENT_MM:
                continue
            f = frame_of(other.polygon)
            if rect_share(other.polygon) >= 0.9 and 100.0 <= f.depth <= 300.0 and 300.0 <= f.length <= 750.0:
                item.tank = other.polygon
                item.details += other.details
                gone.add(j)
                break


def attach_parts(items: list[Item]) -> list[Item]:
    """얇은 띠를 맞닿은 가구에 붙이고(등받이·앞턱), 물탱크를 변기 볼에 붙인다. 붙은 조각은 목록에서 빠진다.

    띠가 여러 겹이면(등받이 두 줄) 안쪽부터 차례로 붙는다. 띠가 가장 두껍게 붙은 쪽이 등받이 쪽이다.
    """
    if not items:
        return items
    gone: set[int] = set()
    _attach_tanks(items, gone)
    order = sorted(range(len(items)), key=lambda k: items[k].polygon.area)
    changed = True
    while changed:
        changed = False
        alive = [k for k in range(len(items)) if k not in gone]
        tree = shapely.STRtree([items[k].polygon for k in alive])
        for i in order:
            strip = items[i]
            if i in gone or strip.tank is not None or strip.edges:
                continue
            sf = frame_of(strip.polygon)
            if sf.depth > _BAND_MAX_MM or sf.length < 300.0 or rect_share(strip.polygon) < 0.9:
                continue
            best = None
            for n in tree.query(strip.polygon.buffer(_TOUCH_MM)).tolist():
                j = alive[n]
                host = items[j]
                if j == i or j in gone or host.tank is not None:
                    continue
                if host.polygon.distance(strip.polygon) > _TOUCH_MM or rect_share(host.polygon) < 0.9:
                    continue
                hf = frame_of(host.polygon)
                if hf.depth < 300.0 or hf.depth > 1100.0:
                    continue
                # 띠의 긴 변과 나란한 가구의 변 길이가 띠와 같아야 한다(가구 한 변을 다 덮는 띠).
                along = abs(sf.u[0] * hf.u[0] + sf.u[1] * hf.u[1])
                side = hf.length if along > 0.9 else hf.depth if along < 0.1 else None
                if side is None or not 0.9 <= sf.length / side <= 1.1:
                    continue
                merged = host.polygon.union(strip.polygon).convex_hull.minimum_rotated_rectangle
                # 붙인 뒤에도 두 조각을 합친 크기여야 한다(어긋나게 닿은 띠는 붙이지 않는다).
                if not isinstance(merged, Polygon) or merged.area > (host.polygon.area + strip.polygon.area) * 1.05:
                    continue
                if best is None or host.polygon.area > best[0]:
                    best = (host.polygon.area, j, merged)
            if best is None:
                continue
            host = items[best[1]]
            hc, sc = host.polygon.centroid, strip.polygon.centroid
            dx, dy = sc.x - hc.x, sc.y - hc.y
            # 띠는 가구의 한 변에 나란히 붙는다. 방향은 띠의 짧은 변 방향이다.
            sign = 1.0 if dx * sf.v[0] + dy * sf.v[1] >= 0 else -1.0
            host.edges.append(((sf.v[0] * sign, sf.v[1] * sign), sf.depth))
            host.polygon = best[2]
            host.details += strip.details
            host.slashes += strip.slashes
            host.drawn = host.drawn or strip.drawn
            gone.add(i)
            changed = True
    for item in items:
        if not item.edges:
            continue
        sides: list[list[Any]] = []  # [방향, 두께 합]
        for direction, thickness in item.edges:
            for side in sides:
                if side[0][0] * direction[0] + side[0][1] * direction[1] > 0.9:
                    side[1] += thickness
                    break
            else:
                sides.append([direction, thickness])
        sides.sort(key=lambda s: -s[1])
        best = sides[0]
        opposite = sum(s[1] for s in sides[1:] if s[0][0] * best[0][0] + s[0][1] * best[0][1] < -0.9)
        if best[1] >= opposite * 1.5:
            item.band, item.band_mm = best[0], best[1]
    return [item for k, item in enumerate(items) if k not in gone]


# ---------------------------------------------------------------------------------------------------------------------
# 3. 종류 정하기
# ---------------------------------------------------------------------------------------------------------------------
def burners(item: Item) -> list[tuple[float, float, float]]:
    """가스레인지 버너 (x, y, 반지름). 가구 안에 그린 지름 100~320mm의 원·호. 같은 중심의 겹친 원은 가장 큰 것 하나로 센다."""
    found: list[tuple[float, float, float]] = []
    for x, y, radius in sorted(item.rings, key=lambda ring: -ring[2]):
        if 50.0 <= radius <= 160.0 and all(math.hypot(x - a, y - b) > max(r, radius) for a, b, r in found):
            found.append((x, y, radius))
    return found


def basins(item: Item) -> list[Polygon]:
    """개수대·세면대의 볼. 가구 안에 그린, 짧은 변 220mm 이상의 둥근(모서리를 굴린) 도형. 서로 겹치지 않는 큰 것부터 둘까지.

    반듯한 사각형은 볼이 아니다(소파 방석, 서랍, 계산대 위의 기기). 볼은 타원이거나 모서리가 둥글다.
    """
    found: list[Polygon] = []
    limit = item.polygon.area * 0.75
    for detail in sorted(item.details, key=lambda d: -d.area):
        f = frame_of(detail)
        if f.depth < 220.0 or detail.area > limit or detail.area < item.polygon.area * 0.06:
            continue
        if not 0.7 <= rect_share(detail) <= 0.99:
            continue
        if any(detail.intersection(other).area > detail.area * 0.2 for other in found):
            continue
        found.append(detail)
        if len(found) == 2:
            break
    return found


def _slash_symbol(item: Item) -> str | None:
    """상자 안 사선이 뜻하는 것. X 표시 → 키 큰 장(tall), 나란한 빗금 여러 줄 → 유리(showcase), 한두 줄 → 수납(shelf)."""
    f = item.frame
    diagonal = math.hypot(f.length, f.depth)

    def oblique(line: LineString) -> bool:
        """상자의 변과 나란하지 않은 선만 사선이다. 상자 안에 가로로 그은 선(서랍, 상판 이음)은 기호가 아니다."""
        (ax, ay), (bx, by) = line.coords[0], line.coords[-1]
        along = abs((bx - ax) * f.u[0] + (by - ay) * f.u[1]) / max(line.length, 1e-9)
        return _OBLIQUE[0] <= along <= _OBLIQUE[1]

    slashes = [s for s in item.slashes if oblique(s)]
    long_ones = [s for s in slashes if s.length >= diagonal * 0.6]
    if len(long_ones) >= 2 and any(a.crosses(b) for i, a in enumerate(long_ones) for b in long_ones[i + 1:]):
        return "tall"
    if len(slashes) >= 3:
        return "showcase"
    if long_ones:
        return "shelf"
    return None


def _symbol(item: Item) -> str | None:
    if item.fixture or item.tank is not None:
        return None
    f = item.frame
    if f.depth < 250.0:
        return None
    slash = _slash_symbol(item)
    if slash is not None or f.depth < 400.0 or rect_share(item.polygon) < 0.85:
        return slash
    if 2 <= len(burners(item)) <= 8:
        return "stove"
    if basins(item) and f.length <= 2400.0:
        return "sink"
    return None


def _by_size(item: Item) -> str:
    f = item.frame
    share = rect_share(item.polygon)
    if item.fixture:
        if item.tank is not None or _is_bowl(item.polygon):
            return "toilet"
        if f.depth < 120.0:
            return "stall"  # 화장실 칸막이
        if basins(item) and f.length >= 600.0:
            return "vanity"
        if f.length <= 750.0 and item.polygon.area <= 0.3e6:
            return "fixture_low"  # 변기·소변기
        if f.depth <= 700.0 and f.length >= 1000.0:
            return "counter"  # 볼을 그리지 않은 긴 기구대
        return "fixture"
    if item.tank is not None:
        return "toilet"
    if f.depth < 100.0:
        # 선을 이어 만든 얇은 면은 가구 사이의 틈일 수 있다. 닫힌 도형으로 그린 것만 칸막이로 본다.
        return "partition" if item.drawn else ""
    if _is_round(item.polygon) or (0.6 <= share < 0.85 and f.length / max(f.depth, 1.0) <= 1.3 and len(item.polygon.exterior.coords) > 8):
        return "stool" if f.length < 600.0 else "table"
    if not item.drawn and (f.length < 300.0 or (share < 0.7 and f.length < 800.0)):
        return ""  # 선 사이에 생긴 자잘한 조각. 닫힌 도형으로 그린 것이 아니면 가구로 보지 않는다
    if share < 0.7:
        return "block"  # 세모꼴 같은 모양. 무엇인지 알 수 없어 낮은 덩어리로 둔다
    if item.band is not None and _BACK_MM[0] <= item.band_mm <= _BACK_MM[1] and 380.0 <= f.depth <= 1000.0:
        return "chair" if f.length <= 800.0 else "sofa"
    if item.band is not None and item.band_mm > _BACK_MM[1] and f.depth >= 450.0:
        return "counter"  # 앞턱(두꺼운 띠)이 붙은 것은 탁자가 아니라 조리대·작업대다
    if f.length <= 750.0:
        return "chair" if f.depth >= 300.0 else "block"
    if f.depth <= 500.0 and f.length >= 900.0:
        return "shelf"
    return "table"


def _fits(kind: str, item: Item) -> bool:
    """곁에 놓인 글자가 이 가구의 이름표일 수 있는가. 수납장 글자가 옆의 의자에 붙지 않게 모양을 따진다."""
    f = item.frame
    seat = item.band is not None and _BACK_MM[0] <= item.band_mm <= _BACK_MM[1]
    round_ = _is_round(item.polygon)
    if kind in ("cabinet", "shelf", "fridge", "showcase", "tall"):
        return not seat and not round_ and f.length >= 500.0 and f.depth >= 200.0
    if kind in ("counter", "bar", "sink", "stove", "vanity", "upper"):
        return not seat and f.length >= 500.0 and f.depth >= 250.0
    if kind == "sofa":
        return f.length >= 600.0 and f.depth >= 450.0
    if kind in ("stool", "bar_stool", "urinal"):
        return f.length <= 800.0
    if kind == "toilet":
        return item.tank is not None or f.length <= 800.0
    if kind in ("appliance", "fixture"):
        return not seat
    return True


def _signature(item: Item) -> tuple[Any, ...]:
    f = item.frame
    return (round(f.length / 25.0), round(f.depth / 25.0), _is_round(item.polygon), item.band is not None, item.tank is not None, item.symbol)


def classify(
    items: list[Item], labels: list[tuple[Point, str, bool]], rules: list[tuple[re.Pattern[str], dict[str, Any]]], reach: float,
) -> list[Item]:
    """items의 kind를 채운다. labels: (자리, 글자, 지시선인가). 종류를 정하지 못했거나 뺄 가구(skip)는 목록에서 뺀다."""
    def rule_of(text: str) -> dict[str, Any] | None:
        rule = next((r for pattern, r in rules if pattern.search(text)), None)
        return None if rule is None or rule["kind"] == "ignore" else rule

    for item in items:
        item.symbol = _symbol(item)
    # 이름표 하나는 가구 하나를 가리킨다. 지시선은 화살표 끝이 닿은 가구, 글자는 글자가 놓인 가구이거나 가장 가까운 가구다.
    # 같은 가구가 여럿이면 나머지는 아래의 '같은 모양 퍼뜨리기'가 맡는다.
    tree = shapely.STRtree([item.footprint for item in items]) if items else None
    claimed: dict[int, float] = {}
    for point, text, pointer in labels:
        rule = rule_of(text)
        if rule is None or tree is None:
            continue
        near = tree.query(point.buffer(_ADJACENT_MM if pointer else reach)).tolist()
        hit = [j for j in near if items[j].footprint.buffer(_ADJACENT_MM).contains(point)]
        if hit:
            target, distance = min(hit, key=lambda j: items[j].polygon.area), 0.0
        elif pointer:
            continue
        else:
            # 얇은 칸막이는 곁의 글자를 제 이름표로 삼지 않는다(변기 옆 칸막이가 변기가 되지 않게).
            far = [(items[j].footprint.distance(point), j) for j in near if items[j].frame.depth >= 120.0 and _fits(rule["kind"], items[j])]
            far = [f for f in far if f[0] <= reach]
            if not far:
                continue
            distance, target = min(far)
        if items[target].frame.depth < 250.0 and rule["kind"] in _COUNTER_RUN + ("bar", "upper"):
            # 조리대 앞턱 같은 얇은 띠에 떨어진 이름표는 그 띠가 붙은 큰 가구를 가리킨다.
            strip = items[target].polygon
            hosts = [
                (strip.buffer(_ADJACENT_MM).intersection(items[j].polygon).area, j) for j in tree.query(strip.buffer(_ADJACENT_MM)).tolist()
                if j != target and items[j].frame.depth >= 250.0
            ]
            if hosts and max(hosts)[0] > 0:
                target = max(hosts)[1]
        if target not in claimed or distance < claimed[target]:
            claimed[target] = distance
            items[target].rule, items[target].pointed = rule, pointer
    for item in items:
        if item.rule is not None and item.rule["kind"] == "upper":
            item.upper, item.rule = True, None  # 상부 선반은 아래 가구에 더하는 것이다. 아래 가구의 종류는 따로 정한다
            if not item.symbol:
                item.kind, item.how = "counter", "label"
                continue
        # 그림이 분명한 기호(버너, 볼)는 곁의 이름표보다 믿는다. 지시선이 옆 칸을 가리키게 그린 도면이 있다.
        if item.rule is not None and item.symbol in _PICTURED and item.rule["kind"] in _COUNTER_RUN + ("bar",) and item.rule["kind"] != item.symbol:
            item.rule = None
        if item.rule is not None:
            item.kind, item.how = item.rule["kind"], "label"
        elif item.symbol:
            item.kind, item.how = item.symbol, "symbol"
        else:
            item.kind, item.how = _by_size(item), "size"

    # 같은 모양 퍼뜨리기: 이름표가 붙은 가구와 크기·기호가 같은 가구
    named: dict[tuple[Any, ...], Item] = {}
    for item in items:
        # 지시선은 같은 가구 여럿 가운데 하나만 가리키는 것이 관례다. 곁에 적은 글자는 기기(정수기, 키오스크)처럼
        # 하나뿐인 것의 이름일 때가 많아, 기기 이름은 같은 크기의 다른 도형으로 퍼뜨리지 않는다.
        if item.how == "label" and item.kind != "skip" and (item.pointed or item.kind not in ("appliance", "fixture")):
            named.setdefault(_signature(item), item)
    for item in items:
        if item.how in ("size", "symbol"):
            twin = named.get(_signature(item))
            if twin is not None:
                item.kind, item.rule, item.how = twin.kind, twin.rule, "same_shape"
    # 조리대 줄: 조리대·개수대·레인지에 맞닿은, 깊이가 비슷한 칸은 조리대다
    if tree is not None and any(item.kind in _COUNTER_RUN for item in items):
        queue = [i for i, item in enumerate(items) if item.kind in _COUNTER_RUN + ("bar",)]
        while queue:
            host = items[queue.pop()]
            depth = host.frame.depth
            for j in tree.query(host.polygon.buffer(_ADJACENT_MM)).tolist():
                other = items[j]
                if other.how != "size" or other.kind not in ("table", "shelf", "chair", "block", "sofa", ""):
                    continue
                if other.polygon.distance(host.polygon) > _ADJACENT_MM:
                    continue
                f = other.frame
                shared = other.polygon.buffer(_ADJACENT_MM).intersection(host.polygon.buffer(_ADJACENT_MM)).area / (2 * _ADJACENT_MM)
                if f.depth < 250.0:
                    # 앞턱: 긴 변이 조리대에 맞닿은 얇은 띠. 선을 이어 만든 면만 본다(따로 그린 칸막이는 그대로 둔다).
                    if other.drawn or f.depth < 60.0 or shared < f.length * 0.8:
                        continue
                elif shared < min(depth, f.depth) * 0.5:
                    continue
                other.kind, other.how = "counter", "run"
                if host.rule is not None and host.rule.get("kind") == "counter":
                    other.rule = host.rule
                queue.append(j)
    # 바 의자: 조리대·탁자 가장자리에 걸친 둥근 의자는 높다
    solid = [item for item in items if item.kind in ("counter", "table", "bar")]
    for item in items:
        if item.kind == "stool" and item.rule is None and any(o.polygon.distance(item.polygon) <= _ADJACENT_MM and o.frame.length >= 1500.0 for o in solid):
            item.kind = "bar_stool"
    return [item for item in items if item.kind not in ("", "skip")]


def face_tables(items: list[Item], reach: float = 900.0) -> None:
    """등받이 띠가 없는 의자의 방향을 정한다. 가장 가까운 탁자를 보고 앉는다(등받이는 탁자 반대쪽)."""
    tables = [item for item in items if item.kind in ("table", "counter", "bar", "desk")]
    if not tables:
        return
    tree = shapely.STRtree([t.polygon for t in tables])
    for item in items:
        if item.kind != "chair" or item.band is not None:
            continue
        near = [tables[j] for j in tree.query(item.polygon.buffer(reach)).tolist()]
        near = [(t.polygon.distance(item.polygon), t) for t in near if t is not item]
        near = [n for n in near if n[0] <= reach]
        if not near:
            continue
        table = min(near, key=lambda n: n[0])[1]
        c = item.polygon.centroid
        target = nearest_points(table.polygon, c)[0]
        dx, dy = c.x - target.x, c.y - target.y
        if math.hypot(dx, dy) < 1.0:
            continue
        f = item.frame
        axis = f.u if abs(dx * f.u[0] + dy * f.u[1]) >= abs(dx * f.v[0] + dy * f.v[1]) else f.v
        sign = 1.0 if dx * axis[0] + dy * axis[1] >= 0 else -1.0
        item.band, item.band_mm = (axis[0] * sign, axis[1] * sign), 0.0
