"""정답 스케치업 모델과 인식 결과 비교. 정답을 층 높이에서 잘라 벽 영역을 면적으로 잰다.

정답에 태그가 없어도 쓸 수 있도록 단면에서 닫힌 얇은 조각을 벽과 기둥으로 본다. 공개 도면 전용 도구이고
그림은 만들지 않는다. 출력은 숫자와 층 코드뿐이다.
"""
from __future__ import annotations

import ctypes
import json
from collections import Counter
from ctypes import byref, c_double, c_size_t, c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from shapely import polygonize, set_precision
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPoint, Polygon, box
from shapely.ops import unary_union

from ..export.floor import door_top
from ..export.skp import _load, _ok

_MM_PER_INCH = 25.4
_THIN_WIDTH_MM = 700.0
_GLASS_MM = 50.0
_MATCH_MM = 60.0
_BODY_MARGIN_MM = 1500.0
_VOTE_BIN_MM = 100.0
_VOTE_CANDIDATES = 12
_MAX_OURS_POINTS = 600
_MAX_TRUTH_POINTS = 2500


class _Point(ctypes.Structure):
    _fields_ = [("x", c_double), ("y", c_double), ("z", c_double)]


class _Transform(ctypes.Structure):
    _fields_ = [("v", c_double * 16)]


def _loop_points(api: Any, loop: c_void_p) -> np.ndarray:
    count = c_size_t()
    _ok(api.SULoopGetNumVertices(loop, byref(count)), "vertices")
    vertices = (c_void_p * count.value)()
    _ok(api.SULoopGetVertices(loop, count, vertices, byref(c_size_t())), "vertices")
    points = []
    for index in range(count.value):
        p = _Point()
        _ok(api.SUVertexGetPosition(c_void_p(vertices[index]), byref(p)), "position")
        points.append((p.x, p.y, p.z, 1.0))
    return np.array(points)


def _faces(api: Any, entities: c_void_p, transform: np.ndarray) -> list[list[np.ndarray]]:
    count = c_size_t()
    _ok(api.SUEntitiesGetNumFaces(entities, byref(count)), "faces")
    handles = (c_void_p * count.value)()
    _ok(api.SUEntitiesGetFaces(entities, count, handles, byref(c_size_t())), "faces")
    result = []
    for index in range(count.value):
        face = c_void_p(handles[index])
        outer = c_void_p()
        _ok(api.SUFaceGetOuterLoop(face, byref(outer)), "outer")
        loops = [_loop_points(api, outer)]
        inner_count = c_size_t()
        _ok(api.SUFaceGetNumInnerLoops(face, byref(inner_count)), "inner")
        inner = (c_void_p * inner_count.value)()
        _ok(api.SUFaceGetInnerLoops(face, inner_count, inner, byref(c_size_t())), "inner")
        loops += [_loop_points(api, c_void_p(inner[k])) for k in range(inner_count.value)]
        result.append([(transform @ ring.T).T[:, :3] * _MM_PER_INCH for ring in loops])
    return result


def load_truth_faces(path: Path, sdk: Path) -> list[list[np.ndarray]]:
    """가장 큰 컴포넌트의 면을 월드 좌표(mm)로 돌려준다. 곁들인 매스 모델은 뺀다."""
    api = _load(sdk)
    model, root = c_void_p(), c_void_p()
    _ok(api.SUModelCreateFromFile(byref(model), str(path).encode("utf-8")), "open")
    try:
        _ok(api.SUModelGetEntities(model, byref(root)), "root")
        loose = _faces(api, root, np.eye(4))
        groups = c_size_t()
        _ok(api.SUEntitiesGetNumGroups(root, byref(groups)), "groups")
        group_handles = (c_void_p * groups.value)()
        _ok(api.SUEntitiesGetGroups(root, groups, group_handles, byref(c_size_t())), "groups")
        for index in range(groups.value):
            inside, transform = c_void_p(), _Transform()
            _ok(api.SUGroupGetEntities(c_void_p(group_handles[index]), byref(inside)), "group")
            _ok(api.SUGroupGetTransform(c_void_p(group_handles[index]), byref(transform)), "group transform")
            loose += _faces(api, inside, np.array(transform.v).reshape(4, 4).T)
        count = c_size_t()
        _ok(api.SUEntitiesGetNumInstances(root, byref(count)), "instances")
        handles = (c_void_p * count.value)()
        _ok(api.SUEntitiesGetInstances(root, count, handles, byref(c_size_t())), "instances")
        best = loose
        for index in range(count.value):
            instance = c_void_p(handles[index])
            definition, inside, transform = c_void_p(), c_void_p(), _Transform()
            _ok(api.SUComponentInstanceGetDefinition(instance, byref(definition)), "definition")
            _ok(api.SUComponentInstanceGetTransform(instance, byref(transform)), "transform")
            _ok(api.SUComponentDefinitionGetEntities(definition, byref(inside)), "entities")
            faces = _faces(api, inside, np.array(transform.v).reshape(4, 4).T)
            if len(faces) > len(best):
                best = faces
        return best
    finally:
        api.SUModelRelease(byref(model))


def section(faces: list[list[np.ndarray]], z: float) -> list[LineString]:
    """높이 z에서 자른 선분. 구멍이 있는 면은 구멍 구간을 뺀다."""
    segments = []
    for rings in faces:
        points = []
        for ring in rings:
            a, b = ring, np.roll(ring, -1, axis=0)
            crossing = (a[:, 2] - z) * (b[:, 2] - z) < 0
            for p, q in zip(a[crossing], b[crossing]):
                t = (z - p[2]) / (q[2] - p[2])
                points.append(p[:2] + (q[:2] - p[:2]) * t)
        if len(points) < 2:
            continue
        pts = np.array(points)
        axis = pts[int(np.argmax(np.linalg.norm(pts - pts[0], axis=1)))] - pts[0]
        pts = pts[np.argsort(pts @ axis)]
        for k in range(0, len(pts) - 1, 2):
            if np.linalg.norm(pts[k + 1] - pts[k]) > 1.0:
                segments.append(LineString([pts[k], pts[k + 1]]))
    return segments


def wall_mask(segments: list[LineString]) -> Any:
    """단면에서 닫힌 조각 중 얇은 것을 벽과 기둥으로 본다."""
    if not segments:
        return Polygon()
    merged = set_precision(unary_union(segments), 1.0)
    keep = [p for p in polygonize([merged]).geoms if 2 * p.area / max(p.length, 1.0) <= _THIN_WIDTH_MM]
    return unary_union(keep) if keep else Polygon()


def our_walls(ir: dict[str, Any], z_cut: float, cfg: dict[str, Any]) -> Any:
    """인식한 벽, 기둥, 창호를 z_cut 높이의 단면으로 만든다. 문은 뚫린다.

    정답 모델은 유리도 닫힌 형상이라 단면에 잡힌다. 그래서 창과 유리벽도 단면에 넣는다.
    """
    solids = []
    for wall in ir["walls"]:
        if wall["start"] == wall["end"]:
            continue
        if wall["kind"] == "curtain":
            # 띠창 높이에서는 얇은 유리판만 단면에 잡힌다.
            in_band = cfg["window_sill_mm"] <= z_cut <= cfg["window_sill_mm"] + cfg["window_height_mm"]
            width = _GLASS_MM if in_band else wall["thickness"]
            solids.append(LineString([wall["start"], wall["end"]]).buffer(width / 2, cap_style="flat"))
            continue
        a, b = np.array(wall["start"]), np.array(wall["end"])
        length = float(np.linalg.norm(b - a))
        u = (b - a) / length
        cuts = []
        for opening in ir["openings"]:
            if opening["host_wall"] != wall["id"]:
                continue
            if opening["type"] == "window":
                continue  # 창틀과 유리가 단면에 남는다
            if z_cut > door_top(opening["width"], cfg):
                continue
            cuts.append((max(opening["position"] - opening["width"] / 2, 0.0), min(opening["position"] + opening["width"] / 2, length)))
        start = 0.0
        for lo, hi in [*sorted(cuts), (length, length)]:
            if lo > start:
                solids.append(LineString([a + u * start, a + u * lo]).buffer(wall["thickness"] / 2, cap_style="flat"))
            start = max(start, hi)
    for c in ir["columns"]:
        if c.get("polygon"):
            solids.append(Polygon(c["polygon"]))
            continue
        solids.append(box(c["center"][0] - c["size"][0] / 2, c["center"][1] - c["size"][1] / 2, c["center"][0] + c["size"][0] / 2, c["center"][1] + c["size"][1] / 2))
    return unary_union(solids) if solids else Polygon()


def _coords(shape: Any) -> np.ndarray:
    return np.array([c for g in getattr(shape, "geoms", [shape]) if not g.is_empty for c in g.exterior.coords]).reshape(-1, 2)


def _shift_candidates(ours: Any, truth: Any) -> list[np.ndarray]:
    a, b = _coords(ours), _coords(truth)
    if not len(a) or not len(b):
        return [np.zeros(2)]
    rng = np.random.default_rng(0)
    a = a[rng.choice(len(a), min(_MAX_OURS_POINTS, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(_MAX_TRUTH_POINTS, len(b)), replace=False)]
    differences = (b[None] - a[:, None]).reshape(-1, 2)
    bins = np.round(differences / _VOTE_BIN_MM).astype(np.int64)
    _, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    return [differences[inverse.ravel() == k].mean(axis=0) for k in np.argsort(-counts)[:_VOTE_CANDIDATES].tolist()]


@dataclass
class FloorScore:
    precision: float
    recall: float
    shift: np.ndarray
    truth_area: float
    ours_area: float

    @property
    def f1(self) -> float:
        return 2 * self.precision * self.recall / max(self.precision + self.recall, 1e-9)


def score(ours: Any, truth: Any, columns: list[tuple[float, float]] | None = None) -> FloorScore:
    """이동량 후보 중 F1이 가장 높은 값으로 잰다. 기둥이 있으면 건물 범위 밖 정답은 뺀다."""
    best = None
    for shift in _shift_candidates(ours, truth):
        moved = translate(ours, *shift)
        target = truth
        if columns and len(columns) >= 3:
            body = MultiPoint([(x + shift[0], y + shift[1]) for x, y in columns]).convex_hull.buffer(_BODY_MARGIN_MM)
            target = truth.intersection(body)
        precision = target.buffer(_MATCH_MM).intersection(moved).area / max(moved.area, 1.0)
        recall = target.intersection(moved.buffer(_MATCH_MM)).area / max(target.area, 1.0)
        candidate = FloorScore(precision, recall, shift, target.area, moved.area)
        if best is None or candidate.f1 > best.f1:
            best = candidate
    return best


def floor_heights_from_truth(faces: list[list[np.ndarray]], min_area_m2: float = 250.0) -> list[float]:
    """넓은 수평면이 모인 높이. 정답의 층 높이를 확인하는 데 쓴다."""
    areas: Counter = Counter()
    for rings in faces:
        ring = rings[0]
        if np.ptp(ring[:, 2]) < 1.0:
            x, y = ring[:, 0], ring[:, 1]
            areas[round(float(ring[0, 2]))] += abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2 / 1e6
    return sorted(float(z) for z, area in areas.items() if area >= min_area_m2)


def compare_model(folder: Path, truth_path: Path, height: float, z_cut: float, cfg: dict[str, Any], sdk: Path) -> list[str]:
    """build 결과를 정답과 비교한 숫자 줄."""
    out = folder / "_cad2bim_out"
    floors = json.loads((out / "building.json").read_text(encoding="utf-8"))
    faces = load_truth_faces(truth_path, sdk)
    lines = ["truth_levels_mm: " + " ".join(f"{z:.0f}" for z in floor_heights_from_truth(faces))]
    total_t = total_tp_t = total_o = total_tp_o = 0.0
    for entry in floors:
        key = entry["floor"]
        if not key[1:].isdigit():  # 지붕층(R)·부분층(F4M)은 정답 모델의 층 단면과 맞출 수 없다
            continue
        ir = json.loads((out / entry["part"] / "ir.json").read_text(encoding="utf-8"))
        number = int(key[1:])
        level = (number - 1) * height if key[0] == "F" else -number * height
        truth = wall_mask(section(faces, level + z_cut))
        ours = our_walls(ir, z_cut, cfg)
        result = score(ours, truth, [tuple(c["center"]) for c in ir["columns"]])
        total_t += result.truth_area
        total_tp_t += result.recall * result.truth_area
        total_o += result.ours_area
        total_tp_o += result.precision * result.ours_area
        lines.append(f"floor[{key}]: part={entry['part']} precision={result.precision:.3f} recall={result.recall:.3f} f1={result.f1:.3f} truth_m2={result.truth_area / 1e6:.1f} ours_m2={result.ours_area / 1e6:.1f}")
    precision = total_tp_o / max(total_o, 1.0)
    recall = total_tp_t / max(total_t, 1.0)
    lines.append(f"overall: precision={precision:.3f} recall={recall:.3f} f1={2 * precision * recall / max(precision + recall, 1e-9):.3f}")
    return lines
