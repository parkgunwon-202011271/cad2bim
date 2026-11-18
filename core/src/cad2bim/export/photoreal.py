"""조감도 실사화. 3D 모델을 사진처럼 바꾼다. 모든 계산은 이 PC 안에서 끝나고 도면이나 그림을 밖으로 보내지 않는다.

모델의 깊이와 모서리를 AI에 고정 조건으로 주어 건물 형태가 흔들리지 않게 한다. AI는 재질, 하늘, 주변 도시만 칠한다.
주변 도시는 도면에 없는 연출이라 규칙으로 만든 덩어리이고, 건물별 값은 두지 않는다.
모델 파일은 미리 내려받아 두어야 하고, 없으면 건너뛴다. 실행 중에는 인터넷을 쓰지 않는다.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from shapely.geometry import Polygon, box

from .aerial import _NOT_FOCUS, AerialView
from .mesh import extrude

_PHOTO_ASPECT = 0.8  # 세로로 긴 항공 사진 구도
WIDTH, HEIGHT = 912, 1144  # SDXL이 잘 그리는 크기. 8의 배수다
_FINAL_SCALE = 1.5
_MODEL = "SG161222/RealVisXL_V4.0"
_VAE = "madebyollin/sdxl-vae-fp16-fix"
_DEPTH = "diffusers/controlnet-depth-sdxl-1.0"
_EDGE = "diffusers/controlnet-canny-sdxl-1.0"
_PROMPT = (
    "aerial drone photograph of a modern building in a dense Seoul city neighborhood, white concrete facade, large dark glass windows, "
    "rooftop garden with real plants, asphalt street with lane markings, sidewalks, surrounding apartment and office buildings "
    "stretching to the horizon, blue sky with white cumulus clouds, natural daylight, sharp architectural photography, "
    "high detail, realistic materials, 24mm lens"
)
_NEGATIVE = (
    "cartoon, illustration, 3d render, cgi, low poly, video game, painting, sketch, blurry, distorted building, warped lines, "
    "text, watermark, logo, people close up, fog, haze, oversaturated"
)


def photo_view(collected: list[tuple[str, dict[str, list[Any]]]], base: AerialView, cfg: dict[str, Any]) -> AerialView:
    """지평선이 보이는 세로 구도. 방향과 해는 조감도와 같고, 건물을 화면 아래쪽에 둔다."""
    # 사진은 건물을 크게 잡는다. 대지 끝은 화면 밖으로 나가도 된다.
    body = [
        mesh.vertices for key, groups in collected if key != "SITE" for category, items in groups.items()
        if category not in _NOT_FOCUS for _, mesh, _ in items
    ]
    points = np.vstack(body or [
        mesh.vertices for _, groups in collected for items in groups.values() for _, mesh, _ in items
    ])
    horizontal = (base.eye - base.target)[:2]
    azimuth = math.atan2(horizontal[0], horizontal[1])
    elevation = math.radians(cfg["photo_elevation_degrees"])
    fov = cfg["photo_fov_degrees"]
    back = np.array([math.sin(azimuth) * math.cos(elevation), math.cos(azimuth) * math.cos(elevation), math.sin(elevation)])
    forward = -back
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    tan_v = math.tan(math.radians(fov) / 2)
    tan_h = tan_v * _PHOTO_ASPECT
    moved = points - center
    x, y, z = moved @ right, moved @ up, moved @ forward
    fill = cfg["photo_fill"]
    distance = float(max((np.abs(x) / (tan_h * fill) - z).max(), (np.abs(y) / (tan_v * fill) - z).max()))
    target = center + up * (tan_v * distance * cfg["photo_lift"])  # 목표점을 올리면 건물이 화면 아래로 간다
    return AerialView(target + back * distance, target, fov, base.sun, base.hour)


def city_blocks(rim: list[tuple[float, float, float]], cfg: dict[str, Any]) -> list[Any]:
    """대지 둘레의 덩어리 도시. 칸을 나눠 건물을 놓고, 칸 사이는 길로 비운다. 모델과 같은 좌표(mm)다."""
    if len(rim) < 3:
        return []
    ring = np.array(rim, dtype=float)
    x0, y0, z = ring[:, 0].min(), ring[:, 1].min(), float(np.median(ring[:, 2]))
    x1, y1 = ring[:, 0].max(), ring[:, 1].max()
    block = cfg["photo_city_block_mm"]
    street = cfg["photo_city_street_mm"]
    reach = cfg["photo_city_reach_mm"]
    rng = np.random.default_rng(23)
    meshes = []
    middle = ((x0 + x1) / 2, (y0 + y1) / 2)
    keep_out = Polygon(ring[:, :2]).convex_hull.buffer(street / 2)
    for gx in np.arange(x0 - reach, x1 + reach, block):
        for gy in np.arange(y0 - reach, y1 + reach, block):
            if keep_out.intersects(box(gx, gy, gx + block, gy + block)):
                continue
            # 먼 쪽일수록 높은 건물이 섞인다. 칸을 다시 나눠 작은 건물이 빽빽한 동네처럼 만든다.
            share = min(math.hypot(gx - middle[0], gy - middle[1]) / reach, 1.0)
            cuts = int(rng.integers(1, 4))
            lot = (block - street) / cuts
            for i in range(cuts):
                for j in range(cuts):
                    height = float(rng.uniform(8000.0, 16000.0) * (1.0 + 1.5 * share) + rng.exponential(10000.0) * share * 3.0)
                    gap = float(rng.uniform(600.0, 1800.0))
                    left, low = gx + street / 2 + i * lot, gy + street / 2 + j * lot
                    meshes.append(extrude(box(left + gap, low + gap, left + lot - gap, low + lot - gap), height).transformed(np.eye(3), np.array([0.0, 0.0, z])))
    return meshes


def edge_lines(ids: np.ndarray, depth: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """종류, 깊이, 면 방향이 바뀌는 곳을 흰 선으로 그린다. 그림에서 찾지 않고 형상에서 바로 얻는다."""
    has = ids >= 0
    shaded = np.zeros((*ids.shape, 3))
    shaded[has] = normals[ids[has]]
    edges = np.zeros(ids.shape, dtype=bool)
    for axis in (0, 1):
        turned = np.abs(np.diff(shaded, axis=axis)).sum(axis=2) > 0.3
        step = np.abs(np.diff(depth, axis=axis)) > np.minimum(np.take(depth, range(depth.shape[axis] - 1), axis=axis), 1e6) * 0.01
        both = turned | step
        if axis == 0:
            edges[:-1, :] |= both
        else:
            edges[:, :-1] |= both
    return (edges * 255).astype(np.uint8)


def depth_map(depth: np.ndarray) -> np.ndarray:
    """가까울수록 밝은 깊이 그림. 먼 도시까지 구분되게 역수를 쓴다."""
    finite = np.isfinite(depth)
    inverse = np.where(finite, 1.0 / np.maximum(depth, 1e-3), 0.0)
    high = np.percentile(inverse[finite], 99.5) if finite.any() else 1.0
    return (np.clip(inverse / max(high, 1e-9), 0.0, 1.0) * 255).astype(np.uint8)


def edge_agreement(model_edges: np.ndarray, photo: np.ndarray, reach: int = 3) -> float:
    """모델 모서리 근처에서 사진에도 모서리가 나오는 비율. 건물이 뭉개지면 떨어진다."""
    from scipy.ndimage import binary_dilation, sobel

    gray = photo.astype(float).mean(axis=2)
    strength = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
    found = binary_dilation(strength > np.percentile(strength, 85), iterations=reach)
    wanted = model_edges > 0
    return float((found & wanted).sum() / max(wanted.sum(), 1))


def worker_python(cfg: dict[str, Any]) -> Path | None:
    """실사화 전용 파이썬. 없으면 실사화를 건너뛴다."""
    path = Path(os.path.expanduser(cfg["photo_python"]))
    return path if path.is_file() else None


def paint(render: np.ndarray, depth: np.ndarray, edges: np.ndarray, path: Path, cfg: dict[str, Any], prompt: str = _PROMPT) -> np.ndarray:
    """깊이와 모서리로 형태를 고정하고 사진으로 칠한다. 돌려주는 값은 키우기 전 단계의 RGB 배열이다."""
    python = worker_python(cfg)
    if python is None:
        raise FileNotFoundError("실사화 환경이 없습니다.")
    settings = {
        "model": _MODEL, "vae": _VAE, "control_models": [_DEPTH, _EDGE], "prompt": prompt, "negative": _NEGATIVE,
        "strength": cfg["photo_strength"], "control_scales": [cfg["photo_depth_scale"], cfg["photo_edge_scale"]],
        "steps": int(cfg["photo_steps"]), "guidance": cfg["photo_guidance"], "seed": int(cfg["photo_seed"]),
        "final_scale": _FINAL_SCALE, "refine_strength": cfg["photo_refine_strength"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        inputs, step, config = Path(folder) / "in.npz", Path(folder) / "first.png", Path(folder) / "set.json"
        np.savez(inputs, render=render, depth=depth, edges=edges)
        config.write_text(json.dumps(settings, ensure_ascii=False), encoding="utf-8")
        worker = Path(__file__).parents[3] / "tools" / "photo_worker.py"
        # 오류 내용에 경로가 섞일 수 있어 화면에 내보내지 않는다.
        done = subprocess.run([str(python), str(worker), str(inputs), str(path), str(step), str(config)], capture_output=True, cwd=folder)
        if done.returncode != 0:
            raise RuntimeError(f"실사화 실패: {done.returncode}")
        return np.asarray(Image.open(step).convert("RGB"))
