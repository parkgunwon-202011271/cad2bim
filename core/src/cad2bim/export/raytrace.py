"""조감도 그림을 그리는 광선 추적기. 그래픽 카드 없이 CPU로 돌린다.

재질은 요소 종류로만 정한다. 건물별 값은 없다. numba와 pillow가 필요하며 없으면 호출하는 쪽에서 건너뛴다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numba import njit, prange
from PIL import Image

from .aerial import ASPECT, AerialView

# (R, G, B 선형값, 무늬). 무늬: 0 단색, 1 포장, 2 목재, 3 잔디, 4 잎, 5 아스팔트, 6 콘크리트, 7 흙과 관목
_LOOKS: dict[str, tuple[float, float, float, int]] = {
    "wall": (0.62, 0.60, 0.56, 6), "column": (0.60, 0.59, 0.56, 6), "slab": (0.34, 0.33, 0.31, 6),
    "roof": (0.46, 0.46, 0.46, 6), "parapet": (0.60, 0.59, 0.56, 6), "penthouse": (0.62, 0.60, 0.56, 6),
    "stair": (0.55, 0.54, 0.51, 0), "rail": (0.10, 0.10, 0.11, 0), "site_rail": (0.10, 0.10, 0.11, 0),
    "door": (0.08, 0.09, 0.10, 0), "ground": (0.46, 0.44, 0.41, 1), "surroundings": (0.40, 0.42, 0.37, 3),
    "road": (0.17, 0.17, 0.18, 5), "deck": (0.40, 0.25, 0.14, 2), "exterior_stair": (0.56, 0.55, 0.51, 1),
    "planter": (0.13, 0.20, 0.08, 7), "roof_planter": (0.13, 0.20, 0.08, 7), "parking": (0.82, 0.82, 0.78, 0),
    "lawn": (0.20, 0.34, 0.10, 3), "pergola": (0.25, 0.15, 0.08, 2), "skylight_frame": (0.20, 0.20, 0.22, 0),
    "equipment": (0.45, 0.45, 0.47, 0), "bench": (0.40, 0.25, 0.14, 2), "tree_crown": (0.12, 0.25, 0.08, 4),
    "tree_trunk": (0.18, 0.12, 0.08, 0), "shrub": (0.11, 0.24, 0.08, 4), "car": (0.60, 0.61, 0.63, 0),
    "car_dark": (0.09, 0.11, 0.17, 0), "car_glass": (0.02, 0.025, 0.03, 0),
}
_GLASS = {"window", "curtain", "skylight"}
_BUILDING = {"wall", "column", "window", "curtain", "door", "roof", "slab", "parapet", "penthouse"}
_LEAF_SIZE = 4
_RIM_REACH_M = 500.0


def _build_bvh(low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, ...]:
    """축 정렬 상자 계층. 자식 둘은 번호가 이어진다."""
    centers = (low + high) / 2
    order = np.arange(len(low))
    nodes_low, nodes_high, left, start, count = [], [], [], [], []

    def add() -> int:
        nodes_low.append(np.zeros(3))
        nodes_high.append(np.zeros(3))
        left.append(-1)
        start.append(0)
        count.append(0)
        return len(left) - 1

    stack = [(add(), 0, len(order))]
    while stack:
        node, begin, end = stack.pop()
        members = order[begin:end]
        nodes_low[node] = low[members].min(axis=0)
        nodes_high[node] = high[members].max(axis=0)
        if end - begin <= _LEAF_SIZE:
            start[node], count[node] = begin, end - begin
            continue
        spread = centers[members].max(axis=0) - centers[members].min(axis=0)
        axis = int(np.argmax(spread))
        middle = (end - begin) // 2
        order[begin:end] = members[np.argpartition(centers[members, axis], middle)]
        first, second = add(), add()
        left[node] = first
        stack.append((first, begin, begin + middle))
        stack.append((second, begin + middle, end))
    return (
        np.array(nodes_low), np.array(nodes_high), np.array(left, dtype=np.int32),
        np.array(start, dtype=np.int32), np.array(count, dtype=np.int32), order.astype(np.int32),
    )


@njit(cache=True, fastmath=True)
def _trace(scene, ox, oy, oz, dx, dy, dz, tmax, shadow):
    """가장 가까운 삼각형. shadow면 불투명한 면이 하나라도 있으면 바로 돌아온다."""
    low, high, left, start, count, order, v0, e1, e2, glass = scene
    ix = 1.0 / dx if abs(dx) > 1e-12 else 1e12
    iy = 1.0 / dy if abs(dy) > 1e-12 else 1e12
    iz = 1.0 / dz if abs(dz) > 1e-12 else 1e12
    best = -1
    closest = tmax
    pending = np.empty(48, dtype=np.int32)
    pending[0] = 0
    top = 1
    while top > 0:
        top -= 1
        node = pending[top]
        t0 = (low[node, 0] - ox) * ix
        t1 = (high[node, 0] - ox) * ix
        tn = min(t0, t1)
        tf = max(t0, t1)
        t0 = (low[node, 1] - oy) * iy
        t1 = (high[node, 1] - oy) * iy
        tn = max(tn, min(t0, t1))
        tf = min(tf, max(t0, t1))
        t0 = (low[node, 2] - oz) * iz
        t1 = (high[node, 2] - oz) * iz
        tn = max(tn, min(t0, t1))
        tf = min(tf, max(t0, t1))
        if tf < max(tn, 0.0) or tn > closest:
            continue
        if count[node] == 0:
            pending[top] = left[node]
            pending[top + 1] = left[node] + 1
            top += 2
            continue
        for k in range(start[node], start[node] + count[node]):
            i = order[k]
            if shadow and glass[i]:
                continue
            ax, ay, az = e1[i, 0], e1[i, 1], e1[i, 2]
            bx, by, bz = e2[i, 0], e2[i, 1], e2[i, 2]
            px, py, pz = dy * bz - dz * by, dz * bx - dx * bz, dx * by - dy * bx
            det = ax * px + ay * py + az * pz
            if abs(det) < 1e-12:
                continue
            inv = 1.0 / det
            sx, sy, sz = ox - v0[i, 0], oy - v0[i, 1], oz - v0[i, 2]
            u = (sx * px + sy * py + sz * pz) * inv
            if u < 0.0 or u > 1.0:
                continue
            qx, qy, qz = sy * az - sz * ay, sz * ax - sx * az, sx * ay - sy * ax
            v = (dx * qx + dy * qy + dz * qz) * inv
            if v < 0.0 or u + v > 1.0:
                continue
            t = (bx * qx + by * qy + bz * qz) * inv
            if 1e-4 < t < closest:
                closest = t
                best = i
                if shadow:
                    return closest, best
    return closest, best


@njit(cache=True, fastmath=True)
def _noise(x, y, z):
    """값 잡음. 무늬에 쓴다."""
    ix, iy, iz = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    fx, fy, fz = x - ix, y - iy, z - iz
    ux, uy, uz = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy), fz * fz * (3 - 2 * fz)

    def corner(a, b, c):
        h = (a * 374761393 + b * 668265263 + c * 2147483647) & 0x7fffffff
        h = ((h ^ (h >> 13)) * 1274126177) & 0x7fffffff
        return (h & 0xffff) / 65535.0

    c000, c100 = corner(ix, iy, iz), corner(ix + 1, iy, iz)
    c010, c110 = corner(ix, iy + 1, iz), corner(ix + 1, iy + 1, iz)
    c001, c101 = corner(ix, iy, iz + 1), corner(ix + 1, iy, iz + 1)
    c011, c111 = corner(ix, iy + 1, iz + 1), corner(ix + 1, iy + 1, iz + 1)
    bottom = (c000 + (c100 - c000) * ux) + ((c010 + (c110 - c010) * ux) - (c000 + (c100 - c000) * ux)) * uy
    roof = (c001 + (c101 - c001) * ux) + ((c011 + (c111 - c011) * ux) - (c001 + (c101 - c001) * ux)) * uy
    return bottom + (roof - bottom) * uz


@njit(cache=True, fastmath=True)
def _fbm(x, y, z):
    return 0.5 * _noise(x, y, z) + 0.3 * _noise(x * 2.1, y * 2.1, z * 2.1) + 0.2 * _noise(x * 4.3, y * 4.3, z * 4.3)


@njit(cache=True, fastmath=True)
def _pattern(kind, x, y, z):
    """밝기 배수. 평면에서 바로 계산한다."""
    if kind == 1:  # 포장 줄눈
        u, v = x / 0.9 % 1.0, y / 0.9 % 1.0
        line = 0.82 if (u < 0.025 or v < 0.025) else 1.0
        return line * (0.92 + 0.16 * _noise(np.floor(x / 0.9) * 7.3, np.floor(y / 0.9) * 3.1, 0.0)) * (0.92 + 0.16 * _fbm(x * 0.4, y * 0.4, 0.0))
    if kind == 2:  # 목재
        board = np.floor((x + y) / 0.14)
        gap = 0.72 if ((x + y) / 0.14 - board) < 0.08 else 1.0
        return gap * (0.82 + 0.36 * _noise(board * 3.7, 0.3, 1.1)) * (0.93 + 0.14 * _fbm(x * 3.0, y * 3.0, board))
    if kind == 3:  # 잔디
        return 0.70 + 0.60 * _fbm(x * 1.7, y * 1.7, 0.0)
    if kind == 4:  # 잎
        return 0.55 + 0.90 * _fbm(x * 3.5, y * 3.5, z * 3.5)
    if kind == 5:  # 아스팔트
        return 0.85 + 0.30 * _noise(x * 9.0, y * 9.0, 0.0)
    if kind == 6:  # 콘크리트
        return 0.90 + 0.20 * _fbm(x * 0.8 + z, y * 0.8 + z, z * 0.3)
    if kind == 7:  # 흙과 관목
        return 0.65 + 0.70 * _fbm(x * 2.5, y * 2.5, z * 2.5)
    return 1.0


@njit(cache=True, fastmath=True)
def _sky(dx, dy, dz):
    t = max(dz, 0.0) ** 0.5
    return 1.05 - 0.75 * t, 1.15 - 0.62 * t, 1.25 - 0.32 * t


@njit(cache=True, fastmath=True)
def _random(state):
    state[0] ^= (state[0] << 13) & 0xffffffff
    state[0] ^= state[0] >> 17
    state[0] ^= (state[0] << 5) & 0xffffffff
    return (state[0] & 0xffffffff) / 4294967296.0


@njit(cache=True, fastmath=True)
def _surface(scene, normals, colours, patterns, sun, samples, i, px, py, pz, dx, dy, dz, state):
    """불투명한 면 한 점의 빛. 해, 하늘, 주변 반사를 더한다."""
    nx, ny, nz = normals[i, 0], normals[i, 1], normals[i, 2]
    if nx * dx + ny * dy + nz * dz > 0:
        nx, ny, nz = -nx, -ny, -nz
    kind = patterns[i]
    if kind == 4:  # 잎은 면이 드러나지 않게 법선을 흔든다
        nx += _noise(px * 4.0, py * 4.0, pz * 4.0) - 0.5
        ny += _noise(py * 4.0, pz * 4.0, px * 4.0) - 0.5
        nz += _noise(pz * 4.0, px * 4.0, py * 4.0) - 0.5
        length = (nx * nx + ny * ny + nz * nz) ** 0.5
        nx, ny, nz = nx / length, ny / length, nz / length
    scale = _pattern(kind, px, py, pz)
    r, g, b = colours[i, 0] * scale, colours[i, 1] * scale, colours[i, 2] * scale
    ox, oy, oz = px + nx * 2e-3, py + ny * 2e-3, pz + nz * 2e-3
    # 해. 해 크기만큼 흔들어 그림자 가장자리를 부드럽게 한다
    sx = sun[0] + (_random(state) - 0.5) * 0.024
    sy = sun[1] + (_random(state) - 0.5) * 0.024
    sz = sun[2] + (_random(state) - 0.5) * 0.024
    length = (sx * sx + sy * sy + sz * sz) ** 0.5
    sx, sy, sz = sx / length, sy / length, sz / length
    light = max(nx * sx + ny * sy + nz * sz, 0.0)
    if light > 0:
        _, hit = _trace(scene, ox, oy, oz, sx, sy, sz, 1e9, True)
        if hit >= 0:
            light = 0.0
    lr, lg, lb = 2.5 * light, 2.3 * light, 2.0 * light
    # 하늘빛과 주변 반사. 코사인 가중 표본이다
    if abs(nz) < 0.9:
        tx, ty, tz = ny, -nx, 0.0
    else:
        tx, ty, tz = 0.0, nz, -ny
    length = (tx * tx + ty * ty + tz * tz) ** 0.5
    tx, ty, tz = tx / length, ty / length, tz / length
    bx, by, bz = ny * tz - nz * ty, nz * tx - nx * tz, nx * ty - ny * tx
    ar, ag, ab = 0.0, 0.0, 0.0
    for _ in range(samples):
        u, v = _random(state), _random(state)
        radius = u ** 0.5
        angle = 6.283185307 * v
        a, c = radius * np.cos(angle), radius * np.sin(angle)
        w = max(1.0 - u, 0.0) ** 0.5
        wx, wy, wz = a * tx + c * bx + w * nx, a * ty + c * by + w * ny, a * tz + c * bz + w * nz
        t, hit = _trace(scene, ox, oy, oz, wx, wy, wz, 30.0, True)
        if hit < 0:
            sr, sg, sb = _sky(wx, wy, wz)
            ar += sr
            ag += sg
            ab += sb
        else:
            fade = min(t / 30.0, 1.0) * 0.5
            ar += 0.22 + fade * 0.4
            ag += 0.21 + fade * 0.4
            ab += 0.20 + fade * 0.4
    ar, ag, ab = ar / samples, ag / samples, ab / samples
    return r * (lr + ar), g * (lg + ag), b * (lb + ab)


@njit(cache=True, fastmath=True)
def _radiance(scene, normals, colours, patterns, sun, samples, ox, oy, oz, dx, dy, dz, state):
    """화면 표본 하나. (빛 R, G, B, 첫 충돌 번호, 첫 충돌 x, y)."""
    glass = scene[9]
    lr, lg, lb = 0.0, 0.0, 0.0
    through = 1.0
    first, fx, fy = -1, ox, oy
    for bounce in range(4):
        t, hit = _trace(scene, ox, oy, oz, dx, dy, dz, 1e9, False)
        if hit < 0:
            sr, sg, sb = _sky(dx, dy, dz)
            return lr + through * sr, lg + through * sg, lb + through * sb, first, fx, fy
        px, py, pz = ox + dx * t, oy + dy * t, oz + dz * t
        if first < 0:
            first, fx, fy = hit, px, py
        if not glass[hit]:
            r, g, b = _surface(scene, normals, colours, patterns, sun, samples, hit, px, py, pz, dx, dy, dz, state)
            return lr + through * r, lg + through * g, lb + through * b, first, fx, fy
        nx, ny, nz = normals[hit, 0], normals[hit, 1], normals[hit, 2]
        cosine = nx * dx + ny * dy + nz * dz
        if cosine > 0:
            nx, ny, nz, cosine = -nx, -ny, -nz, -cosine
        fresnel = 0.20 + 0.80 * (1.0 + cosine) ** 5
        rx, ry, rz = dx - 2 * cosine * nx, dy - 2 * cosine * ny, dz - 2 * cosine * nz
        qx, qy, qz = px + nx * 2e-3, py + ny * 2e-3, pz + nz * 2e-3
        t2, seen = _trace(scene, qx, qy, qz, rx, ry, rz, 1e9, True)
        if seen < 0:
            sr, sg, sb = _sky(rx, ry, rz)
        else:
            sr, sg, sb = _surface(scene, normals, colours, patterns, sun, 1, seen, qx + rx * t2, qy + ry * t2, qz + rz * t2, rx, ry, rz, state)
        lr += through * fresnel * sr * 0.75
        lg += through * fresnel * sg * 0.9
        lb += through * fresnel * sb
        through *= (1.0 - fresnel) * 0.38
        ox, oy, oz = px + dx * 2e-3, py + dy * 2e-3, pz + dz * 2e-3
    return lr, lg, lb, first, fx, fy


@njit(parallel=True, cache=True, fastmath=True)
def _render(scene, normals, colours, patterns, sun, eye, right, up, forward, tan_v, width, height, spp, samples, center, reach):
    image = np.zeros((height, width, 3))
    ids = np.full((height, width), -1, dtype=np.int32)
    tan_h = tan_v * width / height
    for row in prange(height):
        state = np.empty(1, dtype=np.uint64)
        for column in range(width):
            state[0] = (row * 7919 + column * 104729 + 1) & 0xffffffff
            total_r, total_g, total_b = 0.0, 0.0, 0.0
            for sample in range(spp):
                sx = (2.0 * (column + _random(state)) / width - 1.0) * tan_h
                sy = (1.0 - 2.0 * (row + _random(state)) / height) * tan_v
                dx = forward[0] + right[0] * sx + up[0] * sy
                dy = forward[1] + right[1] * sx + up[1] * sy
                dz = forward[2] + right[2] * sx + up[2] * sy
                length = (dx * dx + dy * dy + dz * dz) ** 0.5
                dx, dy, dz = dx / length, dy / length, dz / length
                r, g, b, first, fx, fy = _radiance(scene, normals, colours, patterns, sun, samples, eye[0], eye[1], eye[2], dx, dy, dz, state)
                # 부지에서 멀어질수록 밝은 안개로 사라진다. 도면에 없는 주변을 지어내지 않는다.
                far = ((fx - center[0]) ** 2 + (fy - center[1]) ** 2) ** 0.5 / reach
                fade = 1.0 if first < 0 else min(max((far - 1.0) / 0.75, 0.0), 1.0)
                fade = fade * fade * (3.0 - 2.0 * fade)
                total_r += r + (2.2 - r) * fade
                total_g += g + (2.25 - g) * fade
                total_b += b + (2.3 - b) * fade
                if sample == 0:
                    ids[row, column] = first
            image[row, column, 0] = total_r / spp
            image[row, column, 1] = total_g / spp
            image[row, column, 2] = total_b / spp
    return image, ids


@njit(parallel=True, cache=True, fastmath=True)
def _primary(scene, eye, right, up, forward, tan_v, width, height):
    """화소 중심의 첫 충돌 거리와 삼각형 번호. 유리는 그대로 맞는다."""
    depth = np.full((height, width), np.inf)
    ids = np.full((height, width), -1, dtype=np.int32)
    tan_h = tan_v * width / height
    for row in prange(height):
        for column in range(width):
            sx = (2.0 * (column + 0.5) / width - 1.0) * tan_h
            sy = (1.0 - 2.0 * (row + 0.5) / height) * tan_v
            dx = forward[0] + right[0] * sx + up[0] * sy
            dy = forward[1] + right[1] * sx + up[1] * sy
            dz = forward[2] + right[2] * sx + up[2] * sy
            length = (dx * dx + dy * dy + dz * dz) ** 0.5
            t, hit = _trace(scene, eye[0], eye[1], eye[2], dx / length, dy / length, dz / length, 1e9, False)
            if hit >= 0:
                depth[row, column] = t
                ids[row, column] = hit
    return depth, ids


def _tone(image: np.ndarray) -> np.ndarray:
    """필름 곡선과 감마."""
    x = image * 0.36
    mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return (np.clip(mapped, 0.0, 1.0) ** (1 / 2.2) * 255).astype(np.uint8)


def render_passes(
    collected: list[tuple[str, dict[str, list[Any]]]], rim: list[tuple[float, float, float]], view: AerialView,
    cfg: dict[str, Any], width: int, height: int, spp: int, extra: list[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """실사화 재료. (RGB, 거리 m, 삼각형 번호, 면 법선, 건물 표시). extra는 그림에만 들어가는 메시다."""
    (scene, normals, colours, patterns, names), bases = _scene(collected, rim, extra)
    eye, right, up, forward, tan_v = _camera(view)
    image, _ = _render(
        scene, normals, colours, patterns, view.sun.astype(float), eye, right, up, forward, tan_v, width, height, spp,
        int(cfg["aerial_sky_samples"]), np.array([0.0, 0.0]), 1e9,
    )
    depth, ids = _primary(scene, eye, right, up, forward, tan_v, width, height)
    del bases
    return _tone(image), depth, ids, normals, np.array([n in _BUILDING for n in names], dtype=np.bool_)


def _camera(view: AerialView) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    eye = view.eye / 1000.0
    target = view.target / 1000.0
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return eye, right, np.cross(right, forward), forward, float(np.tan(np.radians(view.fov_degrees) / 2))


def _scene(collected: list[Any], rim: list[tuple[float, float, float]], extra: list[Any] | None = None) -> tuple[tuple[Any, ...], None]:
    pieces, names = [], []
    for mesh in extra or []:
        pieces.append(mesh.vertices[mesh.triangles] / 1000.0)
        names.extend(["city"] * len(mesh.triangles))
    for _, groups in collected:
        for category, items in groups.items():
            for _, mesh, _ in items:
                pieces.append(mesh.vertices[mesh.triangles] / 1000.0)
                names.extend([category] * len(mesh.triangles))
    if len(rim) >= 3:
        ring = np.array(rim, dtype=float) / 1000.0
        center = ring[:, :2].mean(axis=0)
        outward = ring[:, :2] - center
        outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1e-9)
        far = np.column_stack([ring[:, :2] + outward * _RIM_REACH_M, ring[:, 2]])
        next_ring, next_far = np.roll(ring, -1, axis=0), np.roll(far, -1, axis=0)
        skirt = np.concatenate([
            np.stack([ring, far, next_far], axis=1), np.stack([ring, next_far, next_ring], axis=1),
        ])
        pieces.append(skirt)
        names.extend(["surroundings"] * len(skirt))
    triangles = np.concatenate(pieces)
    kinds = sorted(set(names))
    looks = {**_LOOKS, "city": (0.50, 0.49, 0.47, 6)}
    colour_of = {k: np.array(looks.get(k, (0.5, 0.5, 0.5, 0))[:3]) for k in kinds}
    pattern_of = {k: looks.get(k, (0.5, 0.5, 0.5, 0))[3] for k in kinds}
    colours = np.array([colour_of[n] for n in names])
    patterns = np.array([pattern_of[n] for n in names], dtype=np.int32)
    glass = np.array([n in _GLASS for n in names], dtype=np.bool_)
    v0, e1, e2 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    normals = np.cross(e1, e2)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    scene = (*_build_bvh(triangles.min(axis=1), triangles.max(axis=1)), v0, e1, e2, glass)
    return (scene, normals, colours, patterns, names), None


def render_aerial(
    collected: list[tuple[str, dict[str, list[Any]]]], rim: list[tuple[float, float, float]], view: AerialView, path: Path,
    cfg: dict[str, Any], width: int | None = None, spp: int | None = None,
) -> dict[str, float]:
    """그림을 저장하고 화면 점검 값을 돌려준다. 좌표와 이름은 내보내지 않는다."""
    width = width or int(cfg["aerial_width_px"])
    height = int(width / ASPECT)
    spp = spp or int(cfg["aerial_samples_per_pixel"])
    (scene, normals, colours, patterns, names), _ = _scene(collected, rim)
    eye, right, up, forward, tan_v = _camera(view)
    reach = float(np.linalg.norm(view.eye - view.target)) / 1000.0 * tan_v * ASPECT * 0.9
    image, ids = _render(
        scene, normals, colours, patterns, view.sun.astype(float), eye, right, up, forward,
        tan_v, width, height, spp, int(cfg["aerial_sky_samples"]), (view.target / 1000.0)[:2], float(reach),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_tone(image)).save(path)
    building = np.array([n in _BUILDING for n in names], dtype=np.bool_)
    seen = ids >= 0
    return {
        "triangles": len(names), "building_share": round(float(building[ids[seen]].sum()) / ids.size, 3),
        "hit_share": round(float(seen.mean()), 3), "sun_up": int(view.sun[2] > 0), "hour": view.hour,
    }
