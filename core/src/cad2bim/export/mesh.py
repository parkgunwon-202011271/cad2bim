"""닫힌 메시. 스케치업은 부울 연산을 대신해 주지 않아 구멍까지 실제 형상으로 만든다."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient


@dataclass
class Mesh:
    vertices: np.ndarray  # (n, 3)
    triangles: np.ndarray  # (m, 3) 바깥에서 볼 때 반시계 방향
    # 원래 면. (바깥 고리, 안쪽 고리들) 꼭짓점 번호다. 스케치업에는 삼각형이 아니라 이 면을 쓴다.
    polygons: list[tuple[list[int], list[list[int]]]] = field(default_factory=list)
    # 스케치업에서 숨길 모서리. (작은 번호, 큰 번호)다. 지형의 삼각형 선이 도면처럼 보이지 않게 한다.
    soft: set[tuple[int, int]] = field(default_factory=set)

    def transformed(self, rotation: np.ndarray, translation: np.ndarray) -> "Mesh":
        return Mesh(self.vertices @ rotation.T + translation, self.triangles, self.polygons, self.soft)

    @property
    def volume(self) -> float:
        a, b, c = (self.vertices[self.triangles[:, i]] for i in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def open_edges(self) -> int:
        """닫힌 메시면 모든 모서리가 반대 방향으로 정확히 한 번씩 나온다."""
        counts: dict[tuple[int, int], int] = {}
        for tri in self.triangles.tolist():
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        return sum(count != counts.get((b, a), 0) for (a, b), count in counts.items())


def concat(meshes: list[Mesh]) -> Mesh:
    """떨어진 닫힌 조각들을 메시 하나로 묶는다. 부피는 더해지고 각 조각은 닫힌 채로 남는다."""
    offset = 0
    vertices, triangles, polygons = [], [], []
    soft: set[tuple[int, int]] = set()
    for mesh in meshes:
        vertices.append(mesh.vertices)
        triangles.append(mesh.triangles + offset)
        polygons.extend(
            ([v + offset for v in outer], [[v + offset for v in ring] for ring in inners])
            for outer, inners in mesh.polygons
        )
        soft |= {(a + offset, b + offset) for a, b in mesh.soft}
        offset += len(mesh.vertices)
    return Mesh(np.vstack(vertices), np.vstack(triangles), polygons, soft)


def _extrude(polygon: Polygon, depth: float) -> Mesh:
    """(u, v) 평면의 다각형을 w 방향으로 뽑는다. 구멍이 있어도 닫힌 메시가 된다."""
    polygon = orient(polygon, 1.0)
    index: dict[tuple[float, float], int] = {}
    points: list[tuple[float, float]] = []

    def vertex(point: tuple[float, ...]) -> int:
        key = (round(point[0], 6), round(point[1], 6))
        if key not in index:
            index[key] = len(points)
            points.append(key)
        return index[key]

    rings = [polygon.exterior, *polygon.interiors]
    loops = [[vertex(point) for point in list(ring.coords)[:-1]] for ring in rings]
    edges = [(loop[i], loop[(i + 1) % len(loop)]) for loop in loops for i in range(len(loop))]
    caps: list[tuple[int, int, int]] = []
    for triangle in shapely.constrained_delaunay_triangles(polygon).geoms:
        coords = list(orient(triangle, 1.0).exterior.coords)[:3]
        caps.append((vertex(coords[0]), vertex(coords[1]), vertex(coords[2])))

    count = len(points)
    flat = np.array(points, dtype=float)
    vertices = np.vstack([
        np.column_stack([flat, np.zeros(count)]),
        np.column_stack([flat, np.full(count, float(depth))]),
    ])
    triangles: list[tuple[int, int, int]] = []
    for a, b, c in caps:
        triangles.append((a + count, b + count, c + count))  # 윗면
        triangles.append((a, c, b))  # 아랫면
    for a, b in edges:
        triangles.append((a, b, b + count))
        triangles.append((a, b + count, a + count))
    outer, inners = loops[0], loops[1:]
    polygons = [
        ([v + count for v in outer], [[v + count for v in ring] for ring in inners]),  # 윗면
        (outer[::-1], [ring[::-1] for ring in inners]),  # 아랫면
        *(([a, b, b + count, a + count], []) for a, b in edges),
    ]
    return Mesh(vertices, np.array(triangles, dtype=np.int64), polygons)


def rotation_z(angle: float) -> np.ndarray:
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# 벽은 입면(길이, 높이)에 구멍을 내고 두께 방향으로 뽑는다. (u, v, w) → (u, -w, v)는 회전이라 면 방향이 유지된다.
ELEVATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def extrude(polygon: Polygon, depth: float) -> Mesh:
    """(u, v) 평면의 다각형을 w 방향으로 뽑는다. 구멍이 있어도 닫힌 메시가 된다.

    아주 작은 좌표 어긋남으로 삼각 분할이 실패하면(GEOS 위상 오류) 0.1mm 격자로 맞추고 1mm 이하 굴곡을 펴서 다시 한다.
    """
    try:
        return _extrude(polygon, depth)
    except shapely.errors.GEOSException:
        cleaned = shapely.simplify(shapely.set_precision(polygon.buffer(0), 0.1), 1.0)
        if cleaned.geom_type != "Polygon":
            cleaned = max(getattr(cleaned, "geoms", [cleaned]), key=lambda g: g.area)
        return _extrude(cleaned, depth)
