"""건물 밖 요소. 대지, 지형, 데크, 외부 계단, 화단, 난간, 주차선, 옥상 정원을 만든다.

평면 한 장의 선을 레이어 역할로 읽고, 높이는 도면의 레벨 표기에서 얻는다.
- 대지경계선 안을 선으로 나눠 칸을 만든다. 레벨 표기가 든 칸은 그 높이의 평지다.
- 평지 사이의 좁고 긴 칸은 단이다. 양 끝 높이를 칸 수로 나눠 단 높이를 얻는다. 단 높이 표기가 없어도 된다.
- 나머지 땅은 높이 점과 레벨 표기로 만든 삼각망을 따른다.
레벨 표기가 없으면 평지로 둔다. 대지경계선이 없으면 땅을 만들지 않고 옥상 요소만 만든다.
좌표는 모두 층을 맞춘 뒤의 공통 좌표(mm)다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import polygonize, unary_union
from shapely.strtree import STRtree

from ..config import load_layer_dictionary
from .mesh import ELEVATION, Mesh, concat, extrude, rotation_z

_ELEVATION_NOTE = re.compile(r"\b(?:EL|GL)\s*[+＋]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_LAWN = re.compile(r"잔디|lawn|grass", re.IGNORECASE)
_PERGOLA = re.compile(r"퍼골라|파고라|pergola", re.IGNORECASE)
_LIFT = re.compile(r"^(EV|E/V|ELEV)\b|승강기|엘리베이터|elevator|lift", re.IGNORECASE)
_ROAD = re.compile(r"도로|road", re.IGNORECASE)
_EQUIPMENT = re.compile(r"실외기|차폐|설비|equipment|outdoor unit", re.IGNORECASE)

_SNAP_MM = 1.0
_LINE_EXTEND_MM = 150.0  # 건물선이나 경계선에 조금 못 미친 선도 닿게 한다
_SPOT_MARK_MM = 1500.0
_NEIGHBOUR_STEP_MM = 50.0
_RAIL_SAMPLE_MM = 300.0
_RAIL_SIMPLIFY_MM = 30.0
_BASE_MIN_GAP_MM = 3000.0
_BASE_MAX_RISE_MM = 1500.0
_LABEL_MARGIN_MM = 600.0
_FOOTPRINT_MIN_SHARE = 0.8
_MIN_CELL_MM2 = 10000.0
_CONTAINED_SHARE = 0.9
_RECTANGLE_SHARE = 0.9
_MARK_LEN_MM = 1000.0


@dataclass
class SiteShape:
    role: str
    geometry: Any  # Polygon(닫힌 선), LineString, Point
    radius: float = 0.0


@dataclass
class SiteSource:
    """평면 한 장에서 건물 밖 요소에 쓸 선과 글자."""

    key: str
    elevation: float
    outline: Any
    top: float = 0.0  # 이 층 지붕 윗면
    rooms: list[Polygon] = field(default_factory=list)
    shapes: list[SiteShape] = field(default_factory=list)
    texts: list[tuple[float, float, str, str]] = field(default_factory=list)  # x, y, 글자, 역할

    def of(self, *roles: str) -> list[SiteShape]:
        return [s for s in self.shapes if s.role in roles]


@dataclass
class SitePiece:
    name: str
    category: str
    mesh: Mesh
    volume: float | None  # 조각이 서로 닿아 스케치업에서 합쳐지는 형상은 부피를 대조하지 않는다


@dataclass
class Site:
    pieces: list[SitePiece] = field(default_factory=list)
    roof_holes: list[Polygon] = field(default_factory=list)  # 천창 자리. 공통 좌표다
    rim: list[tuple[float, float, float]] = field(default_factory=list)  # 주변 지형의 바깥 테두리
    counts: Counter = field(default_factory=Counter)

    def add(self, category: str, mesh: Mesh, checked: bool = True) -> None:
        self.counts[category] += 1
        self.pieces.append(SitePiece(f"X{len(self.pieces) + 1}", category, mesh, abs(mesh.volume) if checked else None))


def _roles() -> list[tuple[str, re.Pattern[str]]]:
    return [(entry["role"], re.compile(entry["pattern"], re.IGNORECASE)) for entry in load_layer_dictionary().get("site_roles", [])]


def read_source(
    key: str, elevation: float, primitives: list[Any], outline: Any, offset: tuple[float, float],
    top: float = 0.0, rooms: list[Polygon] | None = None,
) -> SiteSource:
    """평면의 요소를 공통 좌표로 옮겨 역할별로 모은다. 역할이 없는 레이어는 보지 않는다."""
    roles = _roles()
    dx, dy = offset
    source = SiteSource(key, elevation, outline, top, rooms or [])

    def moved(points: list[Any]) -> list[tuple[float, float]]:
        return [(p[0] + dx, p[1] + dy) for p in points]

    for p in primitives:
        role = next((name for name, pattern in roles if pattern.search(p.layer)), "")
        g = p.geometry
        if p.kind == "text":
            source.texts.append((g["position"][0] + dx, g["position"][1] + dy, g["text"].strip(), role))
        elif not role or role == "IGNORE":
            continue
        elif p.kind == "line":
            line = LineString(moved([g["p1"], g["p2"]]))
            if line.length > 0:
                source.shapes.append(SiteShape(role, line))
        elif p.kind == "polyline" and len(g["points"]) >= 2:
            points = moved(g["points"])
            shape: Any = Polygon(points) if g.get("closed") and len(points) >= 3 else LineString(points)
            if shape.is_valid and not shape.is_empty and (shape.length > 0):
                source.shapes.append(SiteShape(role, shape))
        elif p.kind == "circle":
            source.shapes.append(SiteShape(role, Point(*moved([g["center"]])[0]), g["radius"]))
    return source


def _polygons(shape: Any) -> list[Polygon]:
    return [g for g in getattr(shape, "geoms", [shape]) if isinstance(g, Polygon) and not g.is_empty and g.area > 0]


def _level_labels(source: SiteSource, area: Any, keep_out: Any, el_zero: float) -> list[tuple[Point, float, str]]:
    labels = []
    for x, y, text, role in source.texts:
        match = _ELEVATION_NOTE.search(text)
        if match is None:
            continue
        point = Point(x, y)
        if area.contains(point) and not keep_out.contains(point):
            labels.append((point, float(match.group(1)) * 1000.0 - el_zero, role))
    return labels


def _spots(source: SiteSource, el_zero: float) -> list[tuple[float, float, float]]:
    """높이 점. 글자 옆의 표시가 실제 위치다."""
    marks = [s.geometry.interpolate(0.5, normalized=True) for s in source.of("SPOT") if isinstance(s.geometry, LineString) and s.geometry.length <= _MARK_LEN_MM]
    result = []
    for x, y, text, role in source.texts:
        if role != "SPOT" or _NUMBER.fullmatch(text) is None:
            continue
        here = Point(x, y)
        near = min(marks, key=here.distance, default=None)
        if near is not None and here.distance(near) <= _SPOT_MARK_MM:
            here = near
        result.append((here.x, here.y, float(text) * 1000.0 - el_zero))
    return result


class _Terrain:
    """알려진 높이 점의 삼각망. 범위 밖은 가장 가까운 점의 높이로 잇는다."""

    def __init__(self, known: list[tuple[float, float, float]], bounds: tuple[float, float, float, float], default: float):
        unique = {(round(x), round(y)): z for x, y, z in known}
        self.default = default
        self.tri = None
        if not unique:
            return
        points = np.array(list(unique), dtype=float)
        values = np.array(list(unique.values()), dtype=float)
        x0, y0, x1, y1 = bounds
        margin = max(x1 - x0, y1 - y0)
        corners = np.array([(x0 - margin, y0 - margin), (x1 + margin, y0 - margin), (x1 + margin, y1 + margin), (x0 - margin, y1 + margin)])
        nearest = [values[int(np.argmin(np.hypot(*(points - c).T)))] for c in corners]
        self.tri = Delaunay(np.vstack([points, corners]))
        self.values = np.concatenate([values, nearest])
        self.function = LinearNDInterpolator(self.tri, self.values)

    def __call__(self, x: float, y: float) -> float:
        if self.tri is None:
            return self.default
        z = float(self.function(x, y))
        return self.default if math.isnan(z) else z


def _surface_solid(faces: list[Polygon], height: Any, base: float) -> Mesh:
    """윗면이 삼각망을 따르는 닫힌 입체. 조각끼리는 꼭짓점을 공유한다."""
    index: dict[tuple[int, int], int] = {}
    points: list[tuple[int, int]] = []

    def vertex(point: Any) -> int:
        key = (round(point[0]), round(point[1]))
        if key not in index:
            index[key] = len(points)
            points.append(key)
        return index[key]

    tops: list[tuple[int, int, int]] = []
    for face in faces:
        for triangle in shapely.constrained_delaunay_triangles(face).geoms:
            a, b, c = (vertex(p) for p in list(orient(triangle, 1.0).exterior.coords)[:3])
            if len({a, b, c}) == 3:
                tops.append((a, b, c))
    count = len(points)
    flat = np.array(points, dtype=float)
    z = np.array([height(x, y) for x, y in flat.tolist()])
    vertices = np.vstack([np.column_stack([flat, np.full(count, base)]), np.column_stack([flat, z])])
    triangles: list[tuple[int, int, int]] = []
    directed = {(a, b) for t in tops for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    for a, b, c in tops:
        triangles.append((a + count, b + count, c + count))
        triangles.append((a, c, b))
    sides = [(a, b) for a, b in directed if (b, a) not in directed]
    for a, b in sides:
        triangles.append((a, b, b + count))
        triangles.append((a, b + count, a + count))
    polygons = [([a + count, b + count, c + count], []) for a, b, c in tops]
    polygons += [([a, c, b], []) for a, b, c in tops]
    polygons += [([a, b, b + count, a + count], []) for a, b in sides]
    inner = {(min(a, b), max(a, b)) for a, b in directed if (b, a) in directed}
    soft = inner | {(a + count, b + count) for a, b in inner}
    return Mesh(vertices, np.array(triangles, dtype=np.int64), polygons, soft)


def _terrain_mesh(polygon: Polygon, terrain: _Terrain, base: float) -> Mesh:
    if terrain.tri is None:
        return extrude(polygon, terrain.default - base).transformed(np.eye(3), np.array([0.0, 0.0, base]))
    pieces: list[Polygon] = []
    points = terrain.tri.points
    for simplex in terrain.tri.simplices:
        for piece in _polygons(Polygon(points[simplex]).intersection(polygon)):
            pieces.extend(_polygons(shapely.set_precision(piece, _SNAP_MM)))
    mesh = _surface_solid(pieces, terrain, base)
    if mesh.open_edges() == 0:
        return mesh
    # 조각 경계의 반올림이 어긋나면 삼각망 안쪽 점을 포기하고 외곽 꼭짓점의 높이만 쓴다.
    return _surface_solid([polygon], terrain, base)


def _blob(center: tuple[float, float, float], radii: tuple[float, float, float]) -> Mesh:
    """둥근 덩어리. 나무와 관목에 쓴다. 안쪽 모서리는 스케치업에서 부드럽게 보인다."""
    golden = (1 + 5 ** 0.5) / 2
    points = [(-1, golden, 0), (1, golden, 0), (-1, -golden, 0), (1, -golden, 0), (0, -1, golden), (0, 1, golden), (0, -1, -golden), (0, 1, -golden), (golden, 0, -1), (golden, 0, 1), (-golden, 0, -1), (-golden, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    vertices = [np.array(p, dtype=float) / np.linalg.norm(p) for p in points]
    split: dict[tuple[int, int], int] = {}

    def middle(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in split:
            mid = vertices[a] + vertices[b]
            vertices.append(mid / np.linalg.norm(mid))
            split[key] = len(vertices) - 1
        return split[key]

    finer = []
    for a, b, c in faces:
        ab, bc, ca = middle(a, b), middle(b, c), middle(c, a)
        finer += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    shaped = np.array(vertices) * np.array(radii) + np.array(center)
    soft = {(min(a, b), max(a, b)) for t in finer for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    return Mesh(shaped, np.array(finer, dtype=np.int64), [(list(t), []) for t in finer], soft)


def _add_tree(site: Site, x: float, y: float, z: float, rng: np.random.Generator, cfg: dict[str, Any]) -> None:
    height = cfg["site_tree_height_mm"] * rng.uniform(0.8, 1.2)
    radius = height * 0.3
    site.add("tree_trunk", _flat(Point(x, y).buffer(height * 0.025, quad_segs=2), z, z + height * 0.55), checked=False)
    crowns = [_blob((x, y, z + height * 0.68), (radius, radius, radius * 1.05))]
    for _ in range(3):
        angle = rng.uniform(0.0, 2 * math.pi)
        crowns.append(_blob(
            (x + math.cos(angle) * radius * 0.55, y + math.sin(angle) * radius * 0.55, z + height * rng.uniform(0.55, 0.8)),
            (radius * 0.7, radius * 0.7, radius * 0.7),
        ))
    site.add("tree_crown", concat(crowns), checked=False)


def _add_car(site: Site, slot: Polygon, z: float, dark: bool, cfg: dict[str, Any]) -> None:
    along, _, _, _, center = _rectangle_axes(slot)
    w = cfg["site_car_width_mm"]
    a = cfg["site_car_length_mm"] / 2
    turn = rotation_z(math.atan2(along[1], along[0]))
    place = np.array([center[0], center[1], z])

    def side(points: list[tuple[float, float]], width: float) -> Mesh:
        return extrude(Polygon(points), width).transformed(ELEVATION, np.array([0.0, width / 2, 0.0])).transformed(turn, place)

    body = [(-a, 250.0), (a, 250.0), (a, 700.0), (a * 0.5, 850.0), (-a * 0.7, 850.0), (-a, 750.0)]
    cabin = [(a * 0.5, 850.0), (a * 0.2, 1400.0), (-a * 0.5, 1400.0), (-a * 0.7, 850.0)]
    site.add("car_dark" if dark else "car", side(body, w), checked=False)
    site.add("car_glass", side(cabin, w * 0.92), checked=False)
    wheels = []
    for sx in (-0.62, 0.62):
        for sy in (-0.42, 0.42):
            offset = turn @ np.array([a * sx, w * sy, 0.0]) + place
            wheels.append(_flat(Point(offset[0], offset[1]).buffer(300.0, quad_segs=3), z, z + 300.0))
    site.add("car_glass", concat(wheels), checked=False)


def _shrubs(site: Site, polygon: Polygon, z: float, rng: np.random.Generator, cfg: dict[str, Any]) -> None:
    """화단을 일정 간격으로 채운다. 칸마다 묶어 태그 하나로 끌 수 있게 한다."""
    step = cfg["site_shrub_spacing_mm"]
    x0, y0, x1, y1 = polygon.bounds
    usable = polygon.buffer(-step * 0.35)
    if usable.is_empty:
        usable = polygon
    blobs = []
    for x in np.arange(x0 + step / 2, x1, step):
        for y in np.arange(y0 + step / 2, y1, step):
            px, py = x + rng.uniform(-0.2, 0.2) * step, y + rng.uniform(-0.2, 0.2) * step
            if usable.contains(Point(px, py)):
                r = step * rng.uniform(0.45, 0.7)
                blobs.append(_blob((px, py, z + r * 0.5), (r, r, r * 0.8)))
    if blobs:
        site.add("shrub", concat(blobs[: int(cfg["site_shrub_limit"])]), checked=False)


def _planting(site: Site, boundary: Polygon, land: Any, keep_clear: Any, terrain: "_Terrain", roads: Any, cfg: dict[str, Any]) -> None:
    """대지 둘레와 넓은 빈 땅에 나무를 심는다. 대지 밖 둘레에는 가로수를 둔다. 도로 위에는 두지 않는다."""
    rng = np.random.default_rng(7)
    limit = int(cfg["site_tree_limit"])
    spacing = cfg["site_tree_spacing_mm"]
    clear = keep_clear.buffer(cfg["site_tree_clearance_mm"])
    free = land.buffer(-cfg["site_tree_inset_mm"]).difference(clear)
    spots: list[Point] = []
    if not free.is_empty:
        x0, y0, x1, y1 = free.bounds
        for row, y in enumerate(np.arange(y0, y1 + 1, spacing)):
            for x in np.arange(x0 + (row % 2) * spacing / 2, x1 + 1, spacing):
                point = Point(x, y)
                if free.contains(point):
                    spots.append(point)
    street = boundary.buffer(cfg["site_street_tree_offset_mm"], join_style="mitre").exterior
    for s in np.arange(0.0, street.length, spacing):
        point = street.interpolate(float(s))
        if roads is None or not roads.buffer(spacing * 0.2).contains(point):
            spots.append(point)
    for point in spots[:limit]:
        jitter = spacing * 0.15
        _add_tree(site, point.x + rng.uniform(-jitter, jitter), point.y + rng.uniform(-jitter, jitter), terrain(point.x, point.y), rng, cfg)


def _flat(polygon: Polygon, z0: float, z1: float) -> Mesh:
    return extrude(polygon, z1 - z0).transformed(np.eye(3), np.array([0.0, 0.0, z0]))


def _extended(line: LineString) -> LineString:
    coords = np.array(line.coords, dtype=float)
    for end, inner in ((0, 1), (-1, -2)):
        direction = coords[end] - coords[inner]
        length = float(np.hypot(*direction))
        if length > 0:
            coords[end] = coords[end] + direction / length * _LINE_EXTEND_MM
    return LineString(coords)


@dataclass
class _Cell:
    polygon: Polygon
    z: float | None = None
    kind: str = "other"  # terrain, deck, step, other
    axis: tuple[float, float] | None = None  # 단의 긴 방향
    width: float = 0.0


def _strip_shape(polygon: Polygon, max_width: float) -> tuple[tuple[float, float], float] | None:
    """직사각형에 가까운 좁고 긴 칸이면 (긴 방향, 폭)을 돌려준다."""
    rectangle = polygon.minimum_rotated_rectangle
    if polygon.area < rectangle.area * _RECTANGLE_SHARE:
        return None
    ring = np.array(rectangle.exterior.coords, dtype=float)
    edges = [ring[1] - ring[0], ring[2] - ring[1]]
    lengths = [float(np.hypot(*e)) for e in edges]
    long_side = int(lengths[1] > lengths[0])
    length, width = lengths[long_side], lengths[1 - long_side]
    if width > max_width or length < width * 1.2 or width <= 0:
        return None
    axis = edges[long_side] / length
    return (float(axis[0]), float(axis[1])), width


def _settle_steps(cells: list[_Cell], locate: Any, terrain: _Terrain) -> None:
    """단을 이어 묶고 양 끝 높이를 칸 수로 나눈다. 끝이 대지 밖이면 그 자리의 지형 높이를 쓴다."""
    free = [i for i, cell in enumerate(cells) if cell.axis is not None and cell.z is None]

    def step_from(i: int, sign: float) -> tuple[int | None, tuple[float, float]]:
        cell = cells[i]
        center = cell.polygon.minimum_rotated_rectangle.centroid
        nx, ny = -cell.axis[1], cell.axis[0]
        if nx < -1e-9 or (abs(nx) <= 1e-9 and ny < 0):
            nx, ny = -nx, -ny
        reach = cell.width / 2 + _NEIGHBOUR_STEP_MM
        point = (center.x + nx * sign * reach, center.y + ny * sign * reach)
        return locate(point), point

    def same_flight(i: int, j: int | None) -> bool:
        if j is None or cells[j].axis is None or cells[j].z is not None:
            return False
        a, b = cells[i].axis, cells[j].axis
        return abs(a[0] * b[0] + a[1] * b[1]) > 0.95

    done: set[int] = set()
    for start in free:
        if start in done:
            continue
        flight = [start]
        for sign in (1.0, -1.0):
            current = start
            while True:
                next_cell, _ = step_from(current, sign)
                if not same_flight(current, next_cell) or next_cell in flight:
                    break
                flight.insert(len(flight) if sign > 0 else 0, next_cell)
                current = next_cell

        def end_height(i: int, sign: float) -> float:
            j, point = step_from(i, sign)
            if j is not None and cells[j].z is not None:
                return float(cells[j].z)
            return terrain(*point)

        low, high = end_height(flight[0], -1.0), end_height(flight[-1], 1.0)
        for number, i in enumerate(flight, start=1):
            cells[i].z = low + (high - low) * number / (len(flight) + 1)
            done.add(i)


def _bar(p0: tuple[float, float], z0: float, p1: tuple[float, float], z1: float, depth: float, height: float) -> Mesh:
    """기울 수 있는 난간 판. 아랫변이 두 끝의 높이를 잇는다."""
    a, b = np.array(p0, dtype=float), np.array(p1, dtype=float)
    length = float(np.hypot(*(b - a)))
    nx, ny = (b - a)[::-1] * (1.0, -1.0) / length * depth / 2
    n = np.array([nx, ny])
    corners = [a + n, b + n, b - n, a - n]
    zs = [z0, z1, z1, z0]
    vertices = np.array([(*c, z) for c, z in zip(corners, zs)] + [(*c, z + height) for c, z in zip(corners, zs)], dtype=float)
    quads = [(3, 2, 1, 0), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    triangles = [t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    mesh = Mesh(vertices, np.array(triangles, dtype=np.int64), [(list(q), []) for q in quads])
    if mesh.volume < 0:
        mesh = Mesh(vertices, mesh.triangles[:, ::-1], [(list(q)[::-1], []) for q in quads])
    return mesh


def _rail_parts(line: LineString, kept: list[tuple[float, float]], cfg: dict[str, Any]) -> list[Mesh]:
    """손잡이와 기둥. 판으로 만들면 외부 공간을 가로막는 벽처럼 보인다."""
    size = cfg["site_rail_thickness_mm"]
    height = cfg["rail_height_mm"]
    parts = []
    positions: list[float] = []
    for (s0, z0), (s1, z1) in zip(kept, kept[1:]):
        a, b = line.interpolate(s0), line.interpolate(s1)
        if a.distance(b) <= 1.0:
            continue
        parts.append(_bar((a.x, a.y), z0 + height - size, (b.x, b.y), z1 + height - size, size, size))
        count = max(round((s1 - s0) / cfg["site_rail_post_spacing_mm"]), 1)
        positions += [s0 + (s1 - s0) * k / count for k in range(count + 1)]
    heights = np.interp(sorted(set(positions)), [s for s, _ in kept], [z for _, z in kept])
    half = size / 2
    for s, z in zip(sorted(set(positions)), heights.tolist()):
        p = line.interpolate(s)
        parts.append(_flat(Polygon([(p.x - half, p.y - half), (p.x + half, p.y - half), (p.x + half, p.y + half), (p.x - half, p.y + half)]), z, z + height - size))
    return parts


def _rails(site: Site, source: SiteSource, height_at: Any, cfg: dict[str, Any]) -> None:
    for shape in source.of("SITE_RAIL"):
        if isinstance(shape.geometry, Point):
            point = shape.geometry
            z = height_at(point.x, point.y)
            site.add("site_rail", _flat(point.buffer(shape.radius, quad_segs=6), z, z + cfg["site_bollard_height_mm"]))
            continue
        line = LineString(shape.geometry.exterior.coords) if isinstance(shape.geometry, Polygon) else shape.geometry
        count = max(int(line.length / _RAIL_SAMPLE_MM), 1)
        positions = [line.length * k / count for k in range(count + 1)]
        points = [line.interpolate(s) for s in positions]
        profile = LineString([(s, height_at(p.x, p.y)) for s, p in zip(positions, points)]).simplify(_RAIL_SIMPLIFY_MM)
        parts = _rail_parts(line, list(profile.coords), cfg)
        if parts:
            site.add("site_rail", concat(parts), checked=False)


def _ground(site: Site, source: SiteSource, sources: list[SiteSource], el_zero: float | None, cfg: dict[str, Any]) -> None:
    lowest = min(sources, key=lambda s: s.elevation)
    outlines = unary_union([s.outline for s in sources])
    candidates = [s.geometry for s in source.of("BOUNDARY") if isinstance(s.geometry, Polygon)]
    boundary = max(candidates, key=lambda p: p.area, default=None)
    if boundary is None or boundary.intersection(outlines).area < outlines.area * _FOOTPRINT_MIN_SHARE:
        return
    bounds = boundary.bounds
    known: list[tuple[float, float, float]] = []
    labels: list[tuple[Point, float, str]] = []
    if el_zero is not None:
        known = _spots(source, el_zero)
        labels = _level_labels(source, boundary, outlines, el_zero)
    top_known = max([z for _, _, z in known] + [z for _, z, _ in labels], default=lowest.elevation)
    # 땅보다 높은 층의 돌출부 아래까지 구멍을 내지 않는다.
    hole = unary_union([s.outline for s in sources if s.elevation <= top_known + cfg["site_level_slack_mm"]] or [lowest.outline])

    closed = [s.geometry for s in source.of("STEP") if isinstance(s.geometry, Polygon)]
    labelled = [p for p in closed if any(p.contains(point) for point, _, _ in labels)]
    blocks = [
        p for p in closed if p not in labelled
        and any(q is not p and q.area > p.area and p.intersection(q).area >= p.area * _CONTAINED_SHARE for q in closed)
    ]
    planters = [s.geometry for s in source.of("PLANTER") if isinstance(s.geometry, Polygon)]
    lines: list[Any] = [boundary.exterior, *[g.exterior for g in _polygons(hole)]]
    lines += [p.exterior for p in closed if p not in blocks]
    lines += [p.exterior for p in planters]
    lines += [_extended(s.geometry) for s in source.of("STEP") if isinstance(s.geometry, LineString)]
    noded = shapely.union_all(lines, grid_size=_SNAP_MM)
    planter_area = unary_union(planters) if planters else None

    cells: list[_Cell] = []
    for polygon in polygonize(noded):
        inside = polygon.representative_point()
        if polygon.area < _MIN_CELL_MM2 or not boundary.contains(inside) or hole.contains(inside):
            continue
        if planter_area is not None and planter_area.contains(inside):
            continue
        cells.append(_Cell(polygon))
    if not cells:
        return
    tree = STRtree([c.polygon for c in cells])

    def locate(point: tuple[float, float]) -> int | None:
        found = tree.query(Point(point), predicate="intersects")
        return int(found[0]) if len(found) else None

    land = int(np.argmax([c.polygon.area for c in cells]))
    cells[land].kind = "terrain"
    for point, z, _ in labels:
        where = locate((point.x, point.y))
        if where is None:
            continue
        if where == land:
            known.append((point.x, point.y, z))
        else:
            cells[where].z = z
            cells[where].kind = "deck"
    # 주차 구획은 표기된 바닥 높이의 평지다. 삼각망이 건물 너머로 기울지 않게 잡아 준다.
    parking = [z for _, z, role in labels if role == "PARKING_NOTE"]
    stripes = source.of("PARKING")
    if parking:
        known += [(x, y, parking[0]) for s in stripes if isinstance(s.geometry, Polygon) for x, y in s.geometry.exterior.coords]
    # 건물 둘레는 맨 아래층 바닥에 닿는다. 옆에 훨씬 높은 땅이 있는 꼭짓점은 옹벽이라 뺀다.
    for x, y in (c for g in _polygons(lowest.outline) for c in g.exterior.coords):
        if not any(z > lowest.elevation + _BASE_MAX_RISE_MM and math.hypot(x - a, y - b) < _BASE_MIN_GAP_MM for a, b, z in known):
            known.append((x, y, lowest.elevation))
    terrain = _Terrain(known, bounds, lowest.elevation)

    for i, cell in enumerate(cells):
        if i == land or cell.z is not None:
            continue
        strip = _strip_shape(cell.polygon, cfg["site_strip_max_width_mm"])
        if strip is not None:
            cell.axis, cell.width = strip
            cell.kind = "step"
    if el_zero is not None:
        _settle_steps(cells, locate, terrain)
    for i, cell in enumerate(cells):
        if i != land and cell.z is None:
            spot = cell.polygon.representative_point()
            cell.z = terrain(spot.x, spot.y)

    every_z = [c.z for i, c in enumerate(cells) if i != land] + [z for _, _, z in known] + [lowest.elevation]
    base = min(every_z) - cfg["site_base_depth_mm"]
    for i, cell in enumerate(cells):
        if i == land:
            site.add("ground", _terrain_mesh(cell.polygon, terrain, base))
        elif cell.kind == "step" and cell.width <= cfg["site_stair_max_tread_mm"]:
            site.add("exterior_stair", _flat(cell.polygon, base, cell.z))
        elif cell.kind in ("deck", "step"):
            site.add("deck", _flat(cell.polygon, base, cell.z))
        else:
            site.add("ground", _flat(cell.polygon, base, cell.z))

    land_shape = cells[land].polygon
    # 대지 밖은 도면에 없지만 끊기면 절벽으로 보이고 그림자가 떨어질 곳이 없다. 높이 점을 이어 넓혀 둔다.
    outer = boundary.buffer(cfg["site_apron_mm"], join_style="mitre")
    apron = outer.difference(boundary)
    road_strips = []
    ring = list(boundary.exterior.coords)
    for x, y, text, _ in source.texts:
        if not _ROAD.search(text) or boundary.contains(Point(x, y)):
            continue
        near = min(zip(ring, ring[1:]), key=lambda edge: LineString(edge).distance(Point(x, y)))
        strip = LineString(near).buffer(cfg["site_apron_mm"] * 1.5, cap_style="square").intersection(apron)
        road_strips.extend(p for p in _polygons(strip) if p.contains(Point(x, y)) or p.distance(Point(x, y)) < cfg["site_apron_mm"])
    roads = unary_union(road_strips) if road_strips else None
    if roads is not None:
        apron = apron.difference(roads)
        for piece in _polygons(roads):
            site.add("road", _terrain_mesh(piece, terrain, base))
    for piece in _polygons(apron):
        site.add("surroundings", _terrain_mesh(piece, terrain, base))
    site.rim = [(x, y, terrain(x, y)) for x, y in list(outer.exterior.coords)[:-1]]

    def height_at(x: float, y: float) -> float:
        where = locate((x, y))
        if where is None:
            nearest = int(tree.nearest(Point(x, y)))
            where = nearest if cells[nearest].polygon.distance(Point(x, y)) <= _LINE_EXTEND_MM else None
        if where is None or where == land:
            return terrain(x, y)
        return float(cells[where].z)

    plants = np.random.default_rng(11)
    for polygon in blocks:
        spot = polygon.representative_point()
        z = height_at(spot.x, spot.y)
        site.add("deck", _flat(polygon, z, z + cfg["site_seat_height_mm"]))
    for polygon in planters:
        near = polygon.buffer(_NEIGHBOUR_STEP_MM)
        spot = polygon.representative_point()
        sides = [c.z for i, c in enumerate(cells) if i != land and c.z is not None and c.polygon.intersects(near)]
        z = max([*sides, terrain(spot.x, spot.y)])
        site.add("planter", _flat(polygon, base, z + cfg["site_planter_height_mm"]))
        _shrubs(site, polygon, z + cfg["site_planter_height_mm"], plants, cfg)
    if stripes:
        painted = unary_union([
            (s.geometry.exterior if isinstance(s.geometry, Polygon) else s.geometry).buffer(cfg["site_stripe_width_mm"] / 2, cap_style="flat", join_style="mitre")
            for s in stripes
        ]).intersection(land_shape)
        for piece in _polygons(painted):
            spot = piece.representative_point()
            z = terrain(spot.x, spot.y)
            site.add("parking", _flat(piece, z, z + cfg["site_stripe_height_mm"]))
        # 칸 순서로 채운다. 난수를 쓰면 도면마다 몰리거나 비어 보인다.
        low, high = cfg["site_slot_min_mm"], cfg["site_slot_max_mm"]
        slots = [
            s.geometry for s in stripes if isinstance(s.geometry, Polygon)
            and low[0] <= _rectangle_axes(s.geometry)[3] <= high[0] and low[1] <= _rectangle_axes(s.geometry)[2] <= high[1]
        ]
        filled = 0
        for number, slot in enumerate(slots):
            if filled < (number + 1) * cfg["site_parking_occupancy"]:
                spot = slot.representative_point()
                _add_car(site, slot, terrain(spot.x, spot.y), filled % 2 == 1, cfg)
                filled += 1
    _rails(site, source, height_at, cfg)
    keep_clear = unary_union([hole, *[(s.geometry.exterior if isinstance(s.geometry, Polygon) else s.geometry).buffer(1000.0) for s in stripes]])
    _planting(site, boundary, land_shape, keep_clear, terrain, roads, cfg)


def _rectangle_axes(polygon: Polygon) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """가장 작은 사각형의 (긴 방향, 짧은 방향, 긴 변, 짧은 변, 중심)."""
    ring = np.array(polygon.minimum_rotated_rectangle.exterior.coords, dtype=float)
    a, b = ring[1] - ring[0], ring[2] - ring[1]
    if np.hypot(*a) < np.hypot(*b):
        a, b = b, a
    length, width = float(np.hypot(*a)), float(np.hypot(*b))
    return a / length, b / width, length, width, ring[:4].mean(axis=0)


def _skylight(site: Site, polygon: Polygon, z: float, cfg: dict[str, Any]) -> None:
    """턱 위에 박공 유리를 얹는다. 평면에 높이가 없어 가정값을 쓴다."""
    curb_top = z + cfg["site_skylight_curb_mm"]
    frame = polygon.difference(polygon.buffer(-cfg["site_skylight_curb_width_mm"], join_style="mitre"))
    for piece in _polygons(frame):
        site.add("skylight_frame", _flat(piece, z - cfg["slab_thickness_mm"], curb_top))
    along, across, length, width, center = _rectangle_axes(polygon)
    half_l, half_w = along * length / 2, across * width / 2
    corners = [center - half_l - half_w, center + half_l - half_w, center + half_l + half_w, center - half_l + half_w]
    ridge = [center - half_l, center + half_l]
    top = curb_top + cfg["site_skylight_rise_mm"]
    vertices = np.array([(*c, curb_top) for c in corners] + [(*r, top) for r in ridge], dtype=float)
    faces = [([3, 2, 1, 0], []), ([0, 1, 5, 4], []), ([2, 3, 4, 5], []), ([1, 2, 5], []), ([3, 0, 4], [])]
    triangles = [(ring[0], ring[i], ring[i + 1]) for ring, _ in faces for i in range(1, len(ring) - 1)]
    mesh = Mesh(vertices, np.array(triangles, dtype=np.int64), faces)
    if mesh.volume < 0:
        mesh = Mesh(vertices, mesh.triangles[:, ::-1], [(ring[::-1], []) for ring, _ in faces])
    site.add("skylight", mesh)
    site.roof_holes.append(polygon)


def _pergola(site: Site, polygon: Polygon, z: float, cfg: dict[str, Any]) -> None:
    along, across, length, width, center = _rectangle_axes(polygon)
    top = z + cfg["site_pergola_height_mm"]
    post = cfg["site_pergola_post_mm"]
    beam = cfg["site_pergola_beam_mm"]

    def rectangle(middle: np.ndarray, half_a: float, half_b: float) -> Polygon:
        return Polygon([tuple(middle + along * s * half_a + across * t * half_b) for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

    parts = []
    inset_a, inset_b = length / 2 - post / 2, width / 2 - post / 2
    for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        parts.append(_flat(rectangle(center + along * s * inset_a + across * t * inset_b, post / 2, post / 2), z, top - beam))
    for t in (-1, 1):
        parts.append(_flat(rectangle(center + across * t * inset_b, length / 2, post / 2), top - beam, top - beam / 2))
    spacing = cfg["site_pergola_slat_spacing_mm"]
    count = max(int(length / spacing), 1)
    for k in range(count + 1):
        offset = -length / 2 + post / 2 + (length - post) * k / count
        parts.append(_flat(rectangle(center + along * offset, post / 4, width / 2), top - beam / 2, top))
    site.add("pergola", concat(parts), checked=False)


def _near_text(polygon: Polygon, texts: list[tuple[float, float, str, str]]) -> str:
    """도형 안이나 바로 옆의 글자를 이어 붙인다."""
    near = polygon.buffer(_LABEL_MARGIN_MM)
    return " ".join(text for x, y, text, _ in texts if near.contains(Point(x, y)))


def _roof_objects(site: Site, source: SiteSource, below: Any, cfg: dict[str, Any]) -> None:
    """아래층 지붕 위, 이 층 외곽 밖에 놓인 요소. 바닥은 이 층의 바닥 높이다."""
    z = source.elevation

    def on_roof(polygon: Polygon) -> bool:
        spot = polygon.representative_point()
        return below.buffer(_LINE_EXTEND_MM).contains(spot) and not source.outline.contains(spot)

    for shape in source.of("SKYLIGHT"):
        if isinstance(shape.geometry, Polygon) and on_roof(shape.geometry):
            _skylight(site, shape.geometry, z, cfg)
    for shape in source.of("ROOF_PLANTER"):
        polygon = shape.geometry
        if not isinstance(polygon, Polygon) or not on_roof(polygon):
            continue
        inside = " ".join(text for x, y, text, _ in source.texts if polygon.contains(Point(x, y)))
        if _LAWN.search(inside):
            site.add("lawn", _flat(polygon, z, z + cfg["site_lawn_height_mm"]))
        else:
            site.add("roof_planter", _flat(polygon, z, z + cfg["site_roof_planter_height_mm"]))
            _shrubs(site, polygon, z + cfg["site_roof_planter_height_mm"], np.random.default_rng(17), cfg)
    for shape in source.of("OPEN"):
        polygon = shape.geometry
        if not isinstance(polygon, Polygon) or not on_roof(polygon):
            continue
        label = _near_text(polygon, source.texts)
        if _PERGOLA.search(label):
            _pergola(site, polygon, z, cfg)
        elif _EQUIPMENT.search(label):
            site.add("equipment", _flat(polygon, z, z + cfg["site_equipment_height_mm"]))
    for shape in source.of("FURNITURE"):
        polygon = shape.geometry
        if isinstance(polygon, Polygon) and on_roof(polygon) and polygon.area <= cfg["site_bench_max_area_mm2"]:
            site.add("bench", _flat(polygon, z, z + cfg["site_seat_height_mm"]))


def _top_of_roof(site: Site, source: SiteSource, cfg: dict[str, Any]) -> None:
    """맨 위 지붕의 파라펫과 승강기 옥탑. 평면에 없는 형상이라 가정값으로 만들고 묶음을 따로 둔다."""
    width = cfg["site_parapet_width_mm"]
    for outline in _polygons(source.outline):
        for piece in _polygons(outline.difference(outline.buffer(-width, join_style="mitre"))):
            site.add("parapet", _flat(piece, source.top, source.top + cfg["site_parapet_height_mm"]))
    for room in source.rooms:
        words = " ".join(text for x, y, text, _ in source.texts if room.contains(Point(x, y)))
        if _LIFT.search(words):
            site.add("penthouse", _flat(room, source.top, source.top + cfg["site_penthouse_height_mm"]))


def _default_ground(site: Site, sources: list[SiteSource], cfg: dict[str, Any]) -> None:
    """대지경계선이 없는 도면의 바닥. 그림자가 떨어지고 조감도가 뜨지 않게 건물 둘레만 깐다."""
    level = min(sources, key=lambda s: (s.key != "F1", abs(s.elevation)))
    outline = level.outline if not level.outline.is_empty else unary_union([s.outline for s in sources])
    if outline.is_empty:
        return
    x0, y0, x1, y1 = outline.bounds
    margin = max(cfg["site_default_margin_mm"], max(x1 - x0, y1 - y0) * 0.3)
    pad = Polygon([(x0 - margin, y0 - margin), (x1 + margin, y0 - margin), (x1 + margin, y1 + margin), (x0 - margin, y1 + margin)])
    z = level.elevation
    terrain = _Terrain([], pad.bounds, z)
    for piece in _polygons(pad.difference(outline)):
        site.add("surroundings", _flat(piece, z - cfg["site_base_depth_mm"], z))
    site.rim = [(x, y, z) for x, y in list(pad.exterior.coords)[:-1]]
    inner = pad.buffer(-margin * 0.1, join_style="mitre")
    _planting(site, inner, inner.difference(outline), outline, terrain, None, cfg)


def plan_site(sources: list[SiteSource], el_zero: float | None, cfg: dict[str, Any]) -> Site:
    """el_zero는 모델 높이 0의 EL(mm)이다. 모르면 레벨 표기를 쓰지 않는다."""
    site = Site()
    if not sources:
        return site
    ranked = sorted(
        sources, key=lambda s: (len(s.of("BOUNDARY")) > 0, sum(x.geometry.length for x in s.of("STEP", "PARKING", "PLANTER"))), reverse=True,
    )
    _ground(site, ranked[0], sources, el_zero, cfg)
    if not site.rim:
        _default_ground(site, sources, cfg)
    ordered = sorted(sources, key=lambda s: s.elevation)
    for number, source in enumerate(ordered):
        if number:
            _roof_objects(site, source, unary_union([s.outline for s in ordered[:number]]), cfg)
    _top_of_roof(site, ordered[-1], cfg)
    return site


__all__ = ["Site", "SitePiece", "SiteSource", "SiteShape", "plan_site", "read_source"]
