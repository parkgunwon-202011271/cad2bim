"""조감도 시점. 해와 카메라를 모델 형상만 보고 정한다. 건물별 값은 두지 않는다.

여덟 방향과 낮 시간대를 돌며, 카메라에 보이면서 해를 받는 입면이 가장 넓은 조합을 고른다.
출입문과 외부 공간이 보이는 쪽에 가산점을 준다. 같은 시점을 스케치업 장면과 그림에 함께 쓴다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

_FACADE = {"wall": 1.0, "column": 1.0, "window": 1.0, "curtain": 1.0, "door": 3.0}
_OUTSIDE = {"deck", "exterior_stair", "planter"}
_NOT_FOCUS = {"surroundings", "road", "tree_crown", "tree_trunk"}
_AZIMUTHS = tuple(float(a) for a in range(0, 360, 45))
_HOURS = tuple(float(h) for h in range(9, 17))
_MIN_SUN_ALTITUDE = 0.35  # 해 높이의 사인값. 약 20도다
_OUTSIDE_WEIGHT = 0.5
_FILL = 0.9
ASPECT = 1.5


@dataclass
class AerialView:
    eye: np.ndarray  # mm
    target: np.ndarray  # mm
    fov_degrees: float  # 세로 화각
    sun: np.ndarray  # 해를 향한 단위 벡터. x 동쪽, y 북쪽, z 위
    hour: float


def sun_direction(latitude: float, longitude: float, timezone: float, day: int, hour: float) -> np.ndarray:
    """지역 시계 시각의 해 방향. 시점 고르기에 쓰는 근사식이고 균시차는 무시한다."""
    declination = math.radians(23.45) * math.sin(2 * math.pi * (284 + day) / 365)
    solar = hour + (longitude - 15.0 * timezone) / 15.0
    angle = math.radians(15.0 * (solar - 12.0))
    lat = math.radians(latitude)
    east = -math.cos(declination) * math.sin(angle)
    north = math.sin(declination) * math.cos(lat) - math.cos(declination) * math.cos(angle) * math.sin(lat)
    up = math.sin(declination) * math.sin(lat) + math.cos(declination) * math.cos(angle) * math.cos(lat)
    # 지표면 기준 동, 북, 위 성분이다. 해의 시간각이 양수면 서쪽이다.
    return np.array([east, north, up])


def _triangles(mesh: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(면 법선, 면적, 무게중심)."""
    a, b, c = (mesh.vertices[mesh.triangles[:, i]] for i in range(3))
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2
    return cross / np.maximum(area[:, None] * 2, 1e-9), area, (a + b + c) / 3


def choose_view(collected: list[tuple[str, dict[str, list[Any]]]], cfg: dict[str, Any]) -> AerialView | None:
    facade, outside, focus = [], [], []
    for _, groups in collected:
        for category, items in groups.items():
            for _, mesh, _ in items:
                if category not in _NOT_FOCUS:
                    focus.append(mesh.vertices)
                if category in _FACADE or category in _OUTSIDE:
                    normal, area, center = _triangles(mesh)
                    if category in _FACADE:
                        facade.append((normal, area * _FACADE[category]))
                    else:
                        outside.append((center[:, :2], area))
    if not focus:
        return None
    points = np.vstack(focus)
    middle = (points.min(axis=0) + points.max(axis=0)) / 2
    day = int(cfg["shadow_day_of_year"])
    best = (-1.0, 0.0, 0.0)
    normals = np.vstack([n for n, _ in facade]) if facade else np.zeros((0, 3))
    weights = np.concatenate([w for _, w in facade]) if facade else np.zeros(0)
    far = float(np.linalg.norm((points.max(axis=0) - points.min(axis=0))[:2])) / 2 or 1.0
    total = float(weights.sum()) or 1.0
    for azimuth in _AZIMUTHS:
        toward = np.array([math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth)), 0.0])
        extra = 0.0
        if outside:
            seen = sum(float((a * np.clip(((c - middle[:2]) @ toward[:2]) / far + 0.5, 0.0, 1.0)).sum()) for c, a in outside)
            extra = _OUTSIDE_WEIGHT * seen / max(sum(float(a.sum()) for _, a in outside), 1.0)
        for hour in _HOURS:
            sun = sun_direction(cfg["shadow_latitude"], cfg["shadow_longitude"], cfg["shadow_timezone"], day, hour)
            if sun[2] < _MIN_SUN_ALTITUDE:
                continue
            value = float((weights * np.maximum(normals @ toward, 0.0) * np.maximum(normals @ sun, 0.0)).sum()) / total + extra
            if value > best[0]:
                best = (value, azimuth, hour)
    _, azimuth, hour = best
    sun = sun_direction(cfg["shadow_latitude"], cfg["shadow_longitude"], cfg["shadow_timezone"], day, hour)
    return AerialView(*_frame(points, azimuth, cfg["aerial_elevation_degrees"], cfg["aerial_fov_degrees"]), cfg["aerial_fov_degrees"], sun, hour)


def _frame(points: np.ndarray, azimuth: float, elevation: float, fov: float) -> tuple[np.ndarray, np.ndarray]:
    """주어진 방향에서 모든 점이 화면에 들어오는 가장 가까운 눈 위치와 목표점."""
    az, el = math.radians(azimuth), math.radians(elevation)
    back = np.array([math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)])
    forward = -back
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tan_v = math.tan(math.radians(fov) / 2) * _FILL
    tan_h = tan_v * ASPECT
    target = (points.min(axis=0) + points.max(axis=0)) / 2
    distance = 0.0
    for _ in range(3):  # 원근 때문에 중심이 조금 어긋나 몇 번 다시 맞춘다
        moved = points - target
        x, y, z = moved @ right, moved @ up, moved @ forward
        distance = float(max((np.abs(x) / tan_h - z).max(), (np.abs(y) / tan_v - z).max()))
        depth = distance + z
        sx, sy = x / depth, y / depth
        target = target + right * (sx.min() + sx.max()) / 2 * distance + up * (sy.min() + sy.max()) / 2 * distance
    return target + back * distance, target
