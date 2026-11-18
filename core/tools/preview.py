"""building.dae를 PNG 미리보기로 그린다. 깊이 버퍼로 가려짐을 처리하고 실제 모서리만 선으로 그린다.

스케치업 화면이 아니라 파일 형상을 빠르게 확인하는 그림이다. 출력은 개수뿐이다. 이름과 좌표는 찍지 않는다.
사용: python tools/preview.py building.dae 미리보기.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import collada
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1500, 1000
_LIGHT = np.array([0.45, -0.55, 0.7]) / np.linalg.norm([0.45, -0.55, 0.7])
# (방위각, 고도, 제목, 지붕 포함 여부). 지붕이 있으면 안쪽 벽과 실을 볼 수 없다.
_VIEWS = (
    (-55.0, 30.0, "남서쪽에서 본 모습", True),
    (125.0, 30.0, "북동쪽에서 본 모습", True),
    (-55.0, 50.0, "지붕을 뺀 모습", False),
)
_ON_ROOF = {"roof", "lawn", "roof_planter", "pergola", "skylight", "skylight_frame", "equipment", "bench", "parapet", "penthouse"}
# 그림자용 해 방향. 서울, 10월 중순 오후 2시 무렵이다(남서쪽 고도 37도). 스케치업 파일의 기본 시각과 맞춘다.
_SUN_AZIMUTH, _SUN_ALTITUDE = np.radians(215.0), np.radians(37.0)
_SUN = np.array([np.sin(_SUN_AZIMUTH) * np.cos(_SUN_ALTITUDE), np.cos(_SUN_AZIMUTH) * np.cos(_SUN_ALTITUDE), np.sin(_SUN_ALTITUDE)])
_SHADOW_SIZE = 2400
_SHADOW_BIAS_M = 0.08
_SHADOW_DARK = 0.62
_FAR_GAP_M = 100.0  # 화면 좌표에서 이만큼 떨어진 점 무리는 따로 본다


def _main_range(values: np.ndarray) -> tuple[float, float]:
    """점이 가장 많은 무리의 범위. 멀리 떨어진 일부 요소 때문에 건물이 점처럼 작아지지 않게 한다."""
    ordered = np.sort(values)
    cuts = np.flatnonzero(np.diff(ordered) > _FAR_GAP_M) + 1
    groups = np.split(ordered, cuts)
    main = max(groups, key=len)
    return float(main[0]), float(main[-1])


def load_meshes(path: Path, roof: bool = True, surroundings: bool = True) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    document = collada.Collada(str(path))
    meshes = []
    for bound in document.scene.objects("geometry"):
        for primitive in bound.primitives():
            if not roof and str(primitive.material.id).split("-")[0] in _ON_ROOF:
                continue
            if not surroundings and str(primitive.material.id).split("-")[0] == "surroundings":
                continue
            meshes.append((
                np.asarray(primitive.vertex, dtype=float) / 1000.0,
                np.asarray(primitive.vertex_index),
                np.array(primitive.material.effect.diffuse[:3], dtype=float),
            ))
    return meshes


def _feature_edges(vertices: np.ndarray, index: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """면이 꺾이는 모서리와 면 방향. 같은 평면을 쪼갠 삼각형 선은 뺀다."""
    _, inverse = np.unique(np.round(vertices, 3), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    triangles = inverse[index]
    a, b, c = (vertices[index[:, i]] for i in range(3))
    normal = np.cross(b - a, c - a)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    owners: dict[tuple[int, int], list[int]] = {}
    for face, (i, j, k) in enumerate(triangles.tolist()):
        for u, v in ((i, j), (j, k), (k, i)):
            owners.setdefault((min(u, v), max(u, v)), []).append(face)
    first: dict[int, int] = {}
    for position, key in enumerate(inverse.tolist()):
        first.setdefault(key, position)
    edges = [
        (vertices[first[u]], vertices[first[v]])
        for (u, v), faces in owners.items()
        if len(faces) != 2 or float(normal[faces[0]] @ normal[faces[1]]) < 0.999
    ]
    return edges, normal


def _sun_map(meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """해에서 내려다본 깊이 지도. (깊이, 오른쪽, 위, 배율, 원점)"""
    side = np.cross(_SUN, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    above = np.cross(side, _SUN)
    points = np.vstack([m[0] for m in meshes])
    flat = np.column_stack([points @ side, points @ above])
    low = flat.min(axis=0)
    scale = (_SHADOW_SIZE - 2) / max(float(np.ptp(flat, axis=0).max()), 1e-6)
    depth = np.full((_SHADOW_SIZE, _SHADOW_SIZE), np.inf)
    for vertices, index, _ in meshes:
        x = (vertices @ side - low[0]) * scale
        y = (vertices @ above - low[1]) * scale
        z = -(vertices @ _SUN)
        for a, b, c in index.tolist():
            px, py, pz = x[[a, b, c]], y[[a, b, c]], z[[a, b, c]]
            denominator = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2])
            if abs(denominator) < 1e-9:
                continue
            x0, x1 = int(px.min()), min(int(px.max()) + 1, _SHADOW_SIZE - 1)
            y0, y1 = int(py.min()), min(int(py.max()) + 1, _SHADOW_SIZE - 1)
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            w0 = ((py[1] - py[2]) * (xs - px[2]) + (px[2] - px[1]) * (ys - py[2])) / denominator
            w1 = ((py[2] - py[0]) * (xs - px[2]) + (px[0] - px[2]) * (ys - py[2])) / denominator
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if inside.any():
                region = depth[y0:y1 + 1, x0:x1 + 1]
                surface = w0 * pz[0] + w1 * pz[1] + w2 * pz[2]
                region[inside] = np.minimum(region[inside], surface[inside])
    return depth, side, above, scale, low


def render(
    meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]], azimuth: float, elevation: float, title: str,
    focus: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None, shadows: bool = True,
) -> Image.Image:
    az, el = np.radians(azimuth), np.radians(elevation)
    right = np.array([np.cos(az), np.sin(az), 0.0])
    forward = np.array([-np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), -np.sin(el)])
    up = np.cross(right, forward)
    points = np.vstack([m[0] for m in (focus or meshes)])
    sx, sy = points @ right, points @ up
    x_lo, x_hi = _main_range(sx)
    y_lo, y_hi = _main_range(sy)
    scale = min((WIDTH - 80) / max(x_hi - x_lo, 1e-6), (HEIGHT - 110) / max(y_hi - y_lo, 1e-6))
    ox = WIDTH / 2 - scale * (x_lo + x_hi) / 2
    oy = HEIGHT / 2 + 20 + scale * (y_lo + y_hi) / 2

    def project(p: np.ndarray) -> np.ndarray:
        return np.column_stack([p @ right * scale + ox, oy - p @ up * scale, p @ forward])

    depth = np.full((HEIGHT, WIDTH), np.inf)
    image = np.full((HEIGHT, WIDTH, 3), 0.94)
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for vertices, index, colour in meshes:
        edges, normal = _feature_edges(vertices, index)
        lines.extend(edges)
        projected = project(vertices)
        for face, triangle in enumerate(index):
            a, b, c = projected[triangle]
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denominator) < 1e-9:
                continue
            x0, x1 = int(max(min(a[0], b[0], c[0]), 0)), int(min(max(a[0], b[0], c[0]) + 1, WIDTH - 1))
            y0, y1 = int(max(min(a[1], b[1], c[1]), 0)), int(min(max(a[1], b[1], c[1]) + 1, HEIGHT - 1))
            if x1 < x0 or y1 < y0:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            w0 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / denominator
            w1 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / denominator
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            z = w0 * a[2] + w1 * b[2] + w2 * c[2]
            region = depth[y0:y1 + 1, x0:x1 + 1]
            visible = inside & (z < region)
            if visible.any():
                region[visible] = z[visible]
                shade = 0.62 + 0.38 * abs(float(normal[face] @ _LIGHT))
                image[y0:y1 + 1, x0:x1 + 1][visible] = colour * shade

    if shadows:
        sun_depth, side, above, sun_scale, low = _sun_map(meshes)
        rows, columns = np.nonzero(np.isfinite(depth))
        pixel_z = depth[rows, columns]
        world = np.outer((columns - ox) / scale, right) + np.outer((oy - rows) / scale, up) + np.outer(pixel_z, forward)
        map_x = np.clip(((world @ side - low[0]) * sun_scale).astype(int), 0, _SHADOW_SIZE - 1)
        map_y = np.clip(((world @ above - low[1]) * sun_scale).astype(int), 0, _SHADOW_SIZE - 1)
        blocked = sun_depth[map_y, map_x] < -(world @ _SUN) - _SHADOW_BIAS_M
        image[rows[blocked], columns[blocked]] *= _SHADOW_DARK

    for p, q in lines:
        a, b = project(np.array([p, q]))
        steps = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
        ts = np.linspace(0, 1, steps + 1)
        xs = np.round(a[0] + (b[0] - a[0]) * ts).astype(int)
        ys = np.round(a[1] + (b[1] - a[1]) * ts).astype(int)
        zs = a[2] + (b[2] - a[2]) * ts
        keep = (xs >= 0) & (xs < WIDTH) & (ys >= 0) & (ys < HEIGHT)
        xs, ys, zs = xs[keep], ys[keep], zs[keep]
        visible = zs <= depth[ys, xs] + 0.06
        image[ys[visible], xs[visible]] = 0.08

    picture = Image.fromarray((image * 255).astype(np.uint8))
    try:
        font = ImageFont.truetype("malgun.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(picture).text((24, 16), title, fill=(30, 30, 30), font=font)
    return picture


def write_preview(dae: Path, png: Path) -> int:
    """미리보기를 저장하고 삼각형 수를 돌려준다."""
    full, bare = load_meshes(dae), load_meshes(dae, roof=False)
    focus = load_meshes(dae, surroundings=False)
    sheet = Image.new("RGB", (WIDTH * len(_VIEWS), HEIGHT), (240, 240, 240))
    for number, (azimuth, elevation, title, roof) in enumerate(_VIEWS):
        sheet.paste(render(full if roof else bare, azimuth, elevation, title, focus), (WIDTH * number, 0))
    png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(png)
    return sum(len(m[1]) for m in full)


def main() -> None:
    triangles = write_preview(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"preview triangles={triangles} saved=1")


if __name__ == "__main__":
    main()
