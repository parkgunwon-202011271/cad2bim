"""건물 합치기. 도면별 평면에서 층을 정하고 위치를 맞춰 건물 하나로 묶는다.

출력은 숫자와 층 코드뿐이다. 파일명, 제목, 실 이름은 찍지 않는다.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .classify import SheetDictionary
from .classify.floors import (
    PARTIAL, ROOF, agreed_levels, floor_key, partial_key, plan_level, plan_title_text, read_level_marks, read_level_notes,
    roof_key,
)
from .config import load_sheet_dictionary
from .config import load_defaults
from .errors import Cad2BimError
from .evaluation.audit import OUT_DIR_NAME, _as_dxf, _check_outside_project, column_shift, find_drawings, run_audit
from .export.aerial import choose_view
from .export.collada import _collect, common_base, verify_dae, write_dae
from .export.dxf3d import verify_dxf, write_dxf
from .export.floor import floor_outlines
from .export.floors import write_editable_files
from .export.elevator import attach_elevators
from .export.furniture import write_furniture
from .export.room_areas import write_room_areas
from .export.ifc import verify_ifc, write_ifc
from .export.photoreal import HEIGHT as PHOTO_HEIGHT
from .export.photoreal import WIDTH as PHOTO_WIDTH
from .export.photoreal import city_blocks, depth_map, edge_agreement, edge_lines, paint, photo_view, worker_python
from .export.quality import building_only
from .export.site import plan_site, read_source
from .export.skp import SU5_2005, find_sdk, verify_skp, write_skp
from .preprocess import load_primitives, preprocess

# 벽 평면이 아닌 평면. 파일명이나 제목에 이 단어가 있으면 층 대표 후보에서 뺀다.
_NOT_ARCHITECTURAL = re.compile(r"천장|천정|반자|구조|설비|전기|소방|기계|조경|확대|상세|부분")
_MIN_COLUMNS_TO_ALIGN = 6
_MIN_ALIGN_SHARE = 0.5
_PREFER_ZERO_MARGIN = 0.05
_GRID_MAX_SLOPE = 0.002
_GRID_MATCH_MM = 50.0
_MIN_GRID_LINES = 2
_MIN_GRID_SHARE = 0.6
_GRID_MIN_LENGTH_SHARE = 0.5
_GRID_DEDUPE_MM = 10.0


@dataclass
class PlanPart:
    part_id: str
    floor: str
    ir: dict[str, Any]
    wall_length: float
    source: str
    grid: tuple[np.ndarray, np.ndarray] | None = None
    file_agrees: bool = False  # 파일명의 층과 평면 제목의 층이 같다
    labels: dict[str, list[tuple[float, float]]] | None = None  # 격자 기호(동그라미 안 X3 등)의 중심, mm
    level: float | None = None  # 평면에 적힌 바닥 레벨(mm, 도면 기준). 부분층 위치에 쓴다


@dataclass
class Storey:
    key: str
    part: PlanPart
    elevation: float
    height: float
    shift: tuple[float, float] = (0.0, 0.0)
    align: str = "reference"  # reference, same_origin, grid, columns, unaligned
    alternatives: int = 0
    match: float | None = None
    floor: Any = None
    others: list[PlanPart] = field(default_factory=list)
    swapped: bool = False
    site: Any = None


@dataclass
class BuildStats:
    plans: Counter = field(default_factory=Counter)


def _wall_length(ir: dict[str, Any]) -> float:
    return sum(float(np.hypot(w["end"][0] - w["start"][0], w["end"][1] - w["start"][1])) for w in ir["walls"])


def _world_columns(ir: dict[str, Any]) -> np.ndarray:
    ox, oy = ir["origin_offset"]
    return np.array([(c["center"][0] + ox, c["center"][1] + oy) for c in ir["columns"]], dtype=float).reshape(-1, 2)


def grid_lines(model: list[Any], roles: dict[str, str], ir: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """이 평면의 격자선 위치(세로선 x, 가로선 y)를 원본 좌표계의 mm로 돌려준다.

    격자 레이어에는 짧은 표시선도 섞인다. 건물을 가로지르는 긴 선만 격자선으로 본다.
    """
    scale = ir["source"]["unit_scale"]
    region = ir["source"].get("region")
    vertical: list[tuple[float, float]] = []
    horizontal: list[tuple[float, float]] = []
    for p in model:
        if p.kind != "line" or roles.get(p.layer) != "GRID":
            continue
        (x0, y0), (x1, y1) = p.geometry["p1"], p.geometry["p2"]
        if region and (max(x0, x1) < region[0] or min(x0, x1) > region[2] or max(y0, y1) < region[1] or min(y0, y1) > region[3]):
            continue
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        if dy > 0 and dx <= dy * _GRID_MAX_SLOPE:
            vertical.append(((x0 + x1) / 2 * scale, dy * scale))
        elif dx > 0 and dy <= dx * _GRID_MAX_SLOPE:
            horizontal.append(((y0 + y1) / 2 * scale, dx * scale))

    def positions(lines: list[tuple[float, float]]) -> np.ndarray:
        if not lines:
            return np.array([], dtype=float)
        longest = max(length for _, length in lines)
        kept = {round(value / _GRID_DEDUPE_MM) * _GRID_DEDUPE_MM for value, length in lines if length >= longest * _GRID_MIN_LENGTH_SHARE}
        return np.array(sorted(kept), dtype=float)

    return positions(vertical), positions(horizontal)


def _axis_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """a를 b에 맞추는 한 축의 이동량과 맞은 비율. 가장 많은 선 쌍이 가리키는 값을 쓴다."""
    if len(a) < _MIN_GRID_LINES or len(b) < _MIN_GRID_LINES:
        return 0.0, 0.0
    differences = (b[None, :] - a[:, None]).ravel()
    bins = np.round(differences / _GRID_MATCH_MM).astype(np.int64)
    values, inverse, counts = np.unique(bins, return_inverse=True, return_counts=True)
    top = counts.max()
    # 동률이면 작은 이동을 고른다. 규칙적인 격자에서 한 칸 밀린 답을 피하기 위해서다.
    candidates = [i for i in np.flatnonzero(counts == top).tolist()]
    pick = min(candidates, key=lambda i: abs(int(values[i])))
    shift = float(differences[inverse.ravel() == pick].mean())
    nearest = np.min(np.abs(a[:, None] + shift - b[None, :]), axis=1)
    small = min(len(a), len(b))
    return shift, min(float((nearest <= _GRID_MATCH_MM).sum()) / small, 1.0)


_GRID_LABEL = re.compile(r"^[A-Z]{1,2}\d{0,2}'?$|^\d{1,2}$", re.I)
_BUBBLE_MAX_RADIUS = 4.0  # 동그라미 반지름 한도. 글자 높이의 배수
_MIN_LABEL_AGREE = 2


_UP_LABEL = re.compile(r"^(UP|U\.P\.?|올라감|오름)$", re.I)
_UP_REACH_MM = 1500.0  # 계단 범위에서 이만큼 밖의 UP 글자까지 그 계단 것으로 본다


def mark_stair_up(model: list[Any], ir: dict[str, Any]) -> int:
    """평면의 'UP' 글자로 계단이 오르기 시작하는 곳을 표시한다(stair["up_at"], IR 좌표). 표시한 계단 수.

    도면이 직접 적은 방향이라 추측(윗층 바닥이 이어지는 쪽)보다 먼저 쓴다.
    """
    stairs = ir.get("stairs") or []
    if not stairs:
        return 0
    scale = ir["source"]["unit_scale"]
    ox, oy = ir["origin_offset"]
    ups = [
        (p.geometry["position"][0] * scale - ox, p.geometry["position"][1] * scale - oy)
        for p in model if p.kind == "text" and _UP_LABEL.match(p.geometry["text"].strip())
    ]
    marked = 0
    for stair in stairs:
        x0, y0, x1, y1 = stair["bounds"]
        near = [
            (math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1)), (x, y)) for x, y in ups
        ]
        near = [n for n in near if n[0] <= _UP_REACH_MM]
        if near:
            stair["up_at"] = list(min(near)[1])
            marked += 1
    return marked


def grid_labels(model: list[Any], ir: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    """격자 기호: 동그라미 안에 든 짧은 이름(X3, Y1, A, 3). 이름마다 동그라미 중심(원본 좌표 mm)을 모은다.

    격자선이 짧게만 그려진 평면(지붕층 등)도 기호의 이름으로 다른 층과 맞출 수 있다.
    """
    scale = ir["source"]["unit_scale"]
    region = ir["source"].get("region")
    circles = [(p.geometry["center"], p.geometry["radius"]) for p in model if p.kind == "circle"]
    found: dict[str, list[tuple[float, float]]] = {}
    if not circles:
        return found
    centers = np.array([c for c, _ in circles], dtype=float)
    radii = np.array([r for _, r in circles], dtype=float)
    for p in model:
        if p.kind != "text":
            continue
        label = p.geometry["text"].strip().upper()
        if not _GRID_LABEL.match(label):
            continue
        x, y = p.geometry["position"]
        if region and not (region[0] <= x <= region[2] and region[1] <= y <= region[3]):
            continue
        d = np.hypot(centers[:, 0] - x, centers[:, 1] - y)
        inside = (d <= radii) & (radii <= _BUBBLE_MAX_RADIUS * max(p.geometry["height"], 1e-9))
        if inside.any():
            k = int(np.argmin(np.where(inside, d, np.inf)))
            found.setdefault(label, []).append((float(centers[k][0] * scale), float(centers[k][1] * scale)))
    return found


def _label_axis(labels: dict[str, list[tuple[float, float]]], tol: float) -> dict[str, int]:
    """기호가 가로로 늘어서면(같은 y) 세로 격자선의 이름이라 x를 쓴다(0). 세로로 늘어서면 y를 쓴다(1)."""
    points = [(name, p) for name, ps in labels.items() for p in ps]
    axis = {}
    for name, ps in labels.items():
        row = sum(1 for other, q in points if other != name and any(abs(q[1] - p[1]) <= tol for p in ps))
        column = sum(1 for other, q in points if other != name and any(abs(q[0] - p[0]) <= tol for p in ps))
        if row != column:
            axis[name] = 0 if row > column else 1
    return axis


def _label_shift(a: PlanPart, b: PlanPart) -> tuple[tuple[float, float], float] | None:
    """같은 이름의 격자 기호로 a를 b에 맞춘다. 축마다 두 이름 이상이 같은 이동량을 가리켜야 믿는다."""
    if not a.labels or not b.labels:
        return None
    common = sorted(set(a.labels) & set(b.labels))
    axis = _label_axis({n: a.labels[n] for n in common}, _GRID_MATCH_MM * 4)
    shift = []
    for ax in (0, 1):
        votes: dict[int, set[str]] = {}
        for name in common:
            if axis.get(name) != ax:
                continue
            for p in a.labels[name]:
                for q in b.labels[name]:
                    votes.setdefault(round((q[ax] - p[ax]) / _GRID_MATCH_MM), set()).add(name)
        if not votes:
            return None
        bin_, names = max(votes.items(), key=lambda kv: (len(kv[1]), -abs(kv[0])))
        if len(names) < _MIN_LABEL_AGREE:
            return None
        shift.append(bin_ * _GRID_MATCH_MM)
    agreeing = len(common)
    return (shift[0], shift[1]), min(1.0, agreeing / max(len(a.labels), 1))


def part_title_kind(texts: list[Any], region: Any, dictionary: SheetDictionary) -> str | None:
    """나뉜 평면 하나의 제목 종류. 한 파일에 평면, 단면, 입면을 함께 그린 경우를 가른다."""
    best = None
    margin = max(region[2] - region[0], region[3] - region[1]) * 0.15 if region else 0.0
    for p in texts:
        if p.kind != "text":
            continue
        if region:
            x, y = p.geometry["position"]
            if not (region[0] - margin <= x <= region[2] + margin and region[1] - margin <= y <= region[3] + margin):
                continue
        for token in p.geometry["text"].split("/"):
            kind = dictionary.title_kind(token.strip())
            if kind is not None and (best is None or p.geometry["height"] > best[0]):
                best = (p.geometry["height"], kind)
    return best[1] if best else None


def read_parts(folder: Path, cfg: dict[str, Any], levels: dict[str, float] | None = None) -> tuple[list[PlanPart], Counter]:
    """평면마다 층을 정한다. 층을 모르는 평면은 추측하지 않고 개수만 센다."""
    out_dir = folder / OUT_DIR_NAME
    mapping = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
    paths = {p.name: p for p in find_drawings(folder)}
    parts: list[PlanPart] = []
    unfloored: list[PlanPart] = []
    dictionary = SheetDictionary(load_sheet_dictionary())
    stats: Counter = Counter()
    for drawing_id, entry in mapping.items():
        name = Path(entry["file"]).name
        if entry.get("status") != "ok" or name not in paths:
            stats["failed"] += 1
            continue
        if "plan" not in (entry.get("kind", "plan"), entry.get("by_shape", "plan")):
            stats["not_plan"] += 1
            continue
        directories = sorted(
            d for d in out_dir.iterdir()
            if d.is_dir() and (d.name == drawing_id or d.name.startswith(f"{drawing_id}.")) and (d / "ir.json").is_file()
        )
        model = None
        for directory in directories:
            ir = json.loads((directory / "ir.json").read_text(encoding="utf-8"))
            if model is None:
                model, _ = load_primitives(_as_dxf(paths[name], out_dir), cfg, out_dir / "_prims")
                if levels is not None:
                    for key, value in read_level_notes(model).items():
                        levels.setdefault(key, value)
            if len(directories) > 1 and part_title_kind(model, ir["source"].get("region"), dictionary) in ("vertical", "detail", "site", "table"):
                stats["not_plan"] += 1
                continue
            title = plan_title_text(model, ir["source"].get("region"))
            # 파일에 평면이 하나일 때만 파일명의 층을 믿는다.
            fallback = name if len(directories) == 1 else ""
            label = title or fallback
            floor = partial_key(label, floor_key(label)) or roof_key(label)
            if floor is None:
                stats["no_floor"] += 1
                if ir["walls"] and not _NOT_ARCHITECTURAL.search(name):
                    unfloored.append(PlanPart(directory.name, "F1", ir, _wall_length(ir), "assumed"))
                continue
            if _NOT_ARCHITECTURAL.search(label) or _NOT_ARCHITECTURAL.search(name):
                stats["not_architectural"] += 1
                continue
            if not ir["walls"]:
                stats["no_walls"] += 1
                continue
            stats["by_title" if title else "by_file_name"] += 1
            mark_stair_up(model, ir)
            roles = {
                name: info["role"]
                for name, info in json.loads((directory / "layer_map.json").read_text(encoding="utf-8"))["layers"].items()
            }
            parts.append(PlanPart(
                directory.name, floor, ir, _wall_length(ir), "title" if title else "file", grid_lines(model, roles, ir),
                floor_key(name) == floor, grid_labels(model, ir),
                plan_level(model, ir["source"].get("region")) if floor.endswith(PARTIAL) else None,
            ))
    # 층 표기가 전혀 없으면 단층 건물로 본다. 평면이 있는데 변환을 포기하지 않기 위해서다.
    if not parts and unfloored:
        parts.append(max(unfloored, key=lambda p: (bool(p.ir["openings"] or p.ir["rooms"]), p.wall_length)))
        stats["single_floor_assumed"] += 1
    return parts, stats


def is_full_floor(key: str) -> bool:
    """B1·F1처럼 번호로 쌓는 층인가. 지붕층(R)과 부분층(F4M)은 따로 놓는다."""
    return bool(re.fullmatch(r"[BF]\d+", key))


def elevation_of(key: str, height: float) -> float:
    """F1이 0이다. 빠진 층이 있어도 층 번호로 위치를 정한다."""
    number = int(key[1:])
    return (number - 1) * height if key[0] == "F" else -number * height


def floor_number(key: str) -> int:
    """F1은 0, F2는 1, B1은 -1. 층 사이 간격을 셀 때 쓴다."""
    return int(key[1:]) - 1 if key[0] == "F" else -int(key[1:])


def level_plan(keys: list[str], notes: dict[str, float], default: float) -> tuple[dict[str, float], dict[str, float], str]:
    """층별 (바닥 높이, 층고)와 근거. 주기의 층 바닥 높이가 둘 이상이면 그 간격을 쓴다.

    주기에 없는 층은 알려진 층 사이 간격을 그대로 이어 쓴다. F1 바닥이 0이 되게 옮긴다.
    """
    known = sorted((floor_number(key), value) for key, value in notes.items())
    step = default
    source = "default"
    if len(known) >= 2:
        steps = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(known, known[1:]) if b[0] > a[0]]
        if steps and min(steps) > 0:
            step = float(sorted(steps)[len(steps) // 2])
            source = "notes"
    absolute: dict[int, float] = dict(known) if source == "notes" else {}
    numbers = sorted({floor_number(key) for key in keys} | set(absolute))
    if not absolute:
        absolute = {0: 0.0}
    anchor, anchor_value = min(absolute.items(), key=lambda item: abs(item[0]))
    level_of = lambda n: absolute.get(n, anchor_value + (n - anchor) * step)  # noqa: E731
    zero = level_of(0)
    elevations = {key: level_of(floor_number(key)) - zero for key in keys}
    heights = {}
    for key in keys:
        n = floor_number(key)
        above = level_of(n + 1)
        heights[key] = above - level_of(n) if above > level_of(n) else step
    return elevations, heights, source


def section_levels(folder: Path, cfg: dict[str, Any]) -> dict[str, float]:
    """단면도·입면도의 층 레벨 표시(층 이름 + FH/FL 값)를 모아 층마다 합의된 값을 돌려준다. 평면은 보지 않는다."""
    out_dir = folder / OUT_DIR_NAME
    mapping = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
    paths = {p.name: p for p in find_drawings(folder)}
    marks: dict[str, list[float]] = {}
    for entry in mapping.values():
        name = Path(entry["file"]).name
        if entry.get("status") != "ok" or name not in paths or entry.get("kind", "plan") == "plan":
            continue
        model, _ = load_primitives(_as_dxf(paths[name], out_dir), cfg, out_dir / "_prims")
        for key, values in read_level_marks(model).items():
            marks.setdefault(key, []).extend(values)
    return agreed_levels(marks)


def merge_section_levels(notes: dict[str, float], sections: dict[str, float], floors: list[str]) -> str | None:
    """단면도 레벨이 평면 주기보다 많은 층을 덮으면 그것을 쓴다. 기준이 다른 값을 섞지 않기 위해 통째로 바꾼다.

    지붕 레벨은 맨 위층 다음 층의 바닥으로 넣어 맨 위층의 층고가 되게 한다.
    """
    floor_levels = {k: v for k, v in sections.items() if k != ROOF}
    if len(floor_levels) < 2 or len(floor_levels) <= len(notes):
        return None
    notes.clear()
    notes.update(floor_levels)
    above = [floor_number(k) for k in floors if is_full_floor(k) and k.startswith("F")]
    if ROOF in sections and above:
        top = max(above)
        key = f"F{top + 2}"
        if key not in notes and sections[ROOF] > notes.get(f"F{top + 1}", float("-inf")):
            notes[key] = sections[ROOF]
    return "sections"


ROOF_PARAPET_MM = 1200.0  # 지붕층 벽(파라펫) 높이 가정값. 옥상 난간 기준 높이


def _source_file(part: PlanPart) -> str:
    """평면이 나온 도면 파일. 한 파일에서 나온 평면은 D03.1, D03.2처럼 번호가 붙는다."""
    return part.part_id.split(".")[0]


def _full_set_file(by_floor: dict[str, list[PlanPart]]) -> str | None:
    """건물의 모든 층(전층) 평면을 한 벌로 담은 파일이 하나뿐이면 그 파일. 아니면 None.

    주민센터: 'B안(최종)' 파일에 1·2·3층이 다 있고, 따로 있는 2·3층 파일은 다른 버전이었다. 층마다 따로
    고르면 버전이 섞여 층이 어긋난다. 한 파일 안의 평면끼리는 같은 버전이라는 근거가 있다.
    """
    floors = [k for k in by_floor if is_full_floor(k)]
    if len(floors) < 2:
        return None
    files = {_source_file(p) for k in floors for p in by_floor[k]}
    full = [f for f in files if all(any(_source_file(p) == f for p in by_floor[k]) for k in floors)]
    return full[0] if len(full) == 1 else None


def select_storeys(parts: list[PlanPart], height: float, levels: dict[str, float] | None = None) -> list[Storey]:
    """층마다 건축 평면다운 평면을 대표로 쓴다.

    문이나 실이 있어야 건축 평면이다. 벽만 길고 둘 다 없는 평면은 구조나 마감 사본일 수 있다.
    파일명과 제목의 층이 같은 평면을 먼저 쓴다. 1층 파일에 곁들인 2층 평면은 2층 파일보다 믿을 근거가 약하다.
    그중 벽이 가장 긴 평면을 쓴다. 문과 실 개수로 고르면 부분 평면이 뽑힌다.
    """
    by_floor: dict[str, list[PlanPart]] = {}
    for part in parts:
        by_floor.setdefault(part.floor, []).append(part)
    bundle = _full_set_file(by_floor)
    storeys = []
    for key, candidates in by_floor.items():
        best = max(candidates, key=lambda p: (
            _source_file(p) == bundle,  # 모든 층을 한 벌로 담은 파일이 있으면 그 벌을 쓴다(버전이 섞이지 않게)
            bool(p.ir["openings"] or p.ir["rooms"]), p.file_agrees, p.wall_length, p.part_id,
        ))
        others = sorted(
            (p for p in candidates if p is not best and (p.ir["openings"] or p.ir["rooms"])),
            key=lambda p: -p.wall_length,
        )
        elevation = elevation_of(key, height) if is_full_floor(key) else 0.0
        storeys.append(Storey(key, best, elevation, height, alternatives=len(candidates) - 1, others=others))
    floors = [s for s in storeys if is_full_floor(s.key)]
    if levels and floors:
        elevations, heights, _ = level_plan([s.key for s in floors], levels, height)
        for storey in floors:
            storey.elevation, storey.height = elevations[storey.key], heights[storey.key]
    for partial in [s for s in storeys if s.key.endswith(PARTIAL)]:
        base = next((s for s in floors if s.key == partial.key[:-len(PARTIAL)]), None)
        if base is None:  # 기준 층이 없으면 놓을 곳이 없다
            storeys.remove(partial)
            continue
        # 부분층은 평면에 적힌 레벨로 놓는다. 기준 층과 같은 기준(도면 레벨)일 때만 쓰고, 아니면 층 중간에 둔다(가정).
        level = partial.part.level
        offset = level - levels[base.key] if level is not None and levels and base.key in levels else None
        if offset is None or not 0 < offset < base.height:
            offset = base.height / 2
        partial.elevation = base.elevation + offset
        partial.height = base.height - offset
    roof = next((s for s in storeys if s.key == ROOF), None)
    if roof is not None:
        if not floors:  # 지붕 평면만 있으면 층을 세울 수 없다
            storeys.remove(roof)
        else:
            # 지붕층은 맨 위층 위에 놓인다. 벽은 파라펫이라 높이를 가정값으로 둔다(도면에서 읽기 전까지).
            top = max(floors, key=lambda s: s.elevation + s.height)
            roof.elevation, roof.height = top.elevation + top.height, ROOF_PARAPET_MM
    return sorted(storeys, key=lambda s: s.elevation)


_PARTIAL_MARGIN_MM = 1500.0


def _opening_zones(ir: dict[str, Any], reach_mm: float) -> list[tuple[Any, dict[str, Any]]]:
    """개구부마다 (폭만큼, 벽 두께에 앞뒤 reach_mm를 더한 사각형, 개구부)."""
    from shapely.geometry import Polygon

    walls = {w["id"]: w for w in ir["walls"]}
    zones = []
    for o in ir.get("openings", []):
        w = walls.get(o.get("host_wall"))
        if w is None or w["start"] == w["end"]:
            continue
        (x0, y0), (x1, y1) = w["start"][:2], w["end"][:2]
        length = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        nx, ny = -uy, ux
        a, b = o["position"] - o["width"] / 2, o["position"] + o["width"] / 2
        d = w["thickness"] / 2 + reach_mm
        zones.append((Polygon([
            (x0 + ux * a + nx * d, y0 + uy * a + ny * d), (x0 + ux * b + nx * d, y0 + uy * b + ny * d),
            (x0 + ux * b - nx * d, y0 + uy * b - ny * d), (x0 + ux * a - nx * d, y0 + uy * a - ny * d),
        ]), o))
    return zones


def _zone_holes(wall_id: str, start: Any, end: Any, gap: Any, zones: list[tuple[Any, dict[str, Any]]]) -> list[dict[str, Any]]:
    """메운 벽이 문·창 자리에 걸치면 그 구간에 판 없는 구멍을 뚫는다(void_only)."""
    length = math.dist(start, end)
    if length <= 0:
        return []
    ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
    holes = []
    for k, (zone, o) in enumerate(zones):
        inside = zone.intersection(gap)
        if inside.is_empty or inside.area < 1000.0:
            continue
        along = [(x - start[0]) * ux + (y - start[1]) * uy for x, y in zone.exterior.coords]
        a, b = max(min(along), 0.0), min(max(along), length)
        if b - a < 50.0:
            continue
        holes.append({
            "id": f"{wall_id}~{o['id']}", "type": o["type"], "operation": None, "host_wall": wall_id,
            "position": (a + b) / 2, "width": b - a, "hinge": None, "swing_dir": None, "void_only": True,
            "confidence": {"level": "review", "score": 0.5, "reasons": ["sliver_filled"]}, "source_handles": [],
        })
    return holes


def close_slivers(storeys: list[Storey], max_gap_mm: float) -> dict[str, int]:
    """벽·기둥 사이의 좁은 빈 곳(최소 개구부 폭보다 좁음)을 벽으로 메운다. 층마다 메운 수를 돌려준다.

    틈의 종류(기둥 옆, 두 겹 벽 사이, 모서리, 비스듬한 만남)를 가리지 않는다. 벽·기둥 평면을 합쳐
    max_gap_mm 폭으로 닫았을 때 새로 채워지는 곳이 틈이다. 이보다 좁으면 문이나 창일 수 없으니 판단 없이 메운다(계산).
    """
    from shapely.geometry import LineString, Point, box
    from shapely.ops import unary_union

    r = max_gap_mm / 2
    filled: dict[str, int] = {}
    for s in storeys:
        body = building_only(s.part.ir)
        shapes = [
            LineString([w["start"], w["end"]]).buffer(w["thickness"] / 2, cap_style="flat")
            for w in body["walls"] if w["start"] != w["end"]
        ]
        for c in body["columns"]:
            size = c.get("size") or [500.0, 500.0]
            sx, sy = (size, size) if not isinstance(size, list) else size[:2]
            x, y = c["center"][:2]
            shapes.append(box(x - sx / 2, y - sy / 2, x + sx / 2, y + sy / 2))
        if not shapes:
            continue
        solid = unary_union(shapes)
        closed = solid.buffer(r, join_style="mitre", mitre_limit=2.0).buffer(-r, join_style="mitre", mitre_limit=2.0)
        gaps = closed.difference(solid)
        # 문·창 자리 앞뒤의 틈(겹벽 사이)도 메우되, 메운 벽에 같은 자리 구멍을 판 없이 뚫는다.
        # 아예 비워 두면 문 위 인방의 두 겹 사이가 가는 틈으로 보이고, 막아 버리면 문을 지나갈 수 없다.
        zones = _opening_zones(body, max_gap_mm)
        ir = dict(s.part.ir)
        openings = list(ir["openings"])
        walls = list(ir["walls"])
        count = 0
        wall_lines = [
            (LineString([w["start"], w["end"]]), (w["end"][0] - w["start"][0], w["end"][1] - w["start"][1]))
            for w in body["walls"] if w["start"] != w["end"]
        ]
        for part in getattr(gaps, "geoms", [gaps]):
            if part.is_empty or part.area < 2000.0:  # 50×40mm보다 작은 조각은 모서리 이음 자국이다
                continue
            # 틈에 닿은 벽의 방향으로 감싼다. 기울어진 최소 사각형으로 감싸면 삼각형 틈이 비스듬한 판이 된다.
            directions = [d for line, d in wall_lines if line.distance(part) <= max(r, 50.0)] or [(1.0, 0.0), (0.0, 1.0)]
            best = None
            for dx, dy in directions:
                norm = (dx * dx + dy * dy) ** 0.5
                ux, uy = dx / norm, dy / norm
                coords = list(part.exterior.coords)
                along = [x * ux + y * uy for x, y in coords]
                across = [-x * uy + y * ux for x, y in coords]
                lo_a, hi_a, lo_c, hi_c = min(along), max(along), min(across), max(across)
                width, length = hi_c - lo_c, hi_a - lo_a
                if width > length:  # 짧은 쪽이 두께다
                    ux, uy = -uy, ux
                    along, across = across, [-a for a in along]
                    lo_a, hi_a, lo_c, hi_c = min(along), max(along), min(across), max(across)
                    width, length = hi_c - lo_c, hi_a - lo_a
                if width <= max_gap_mm and length >= 30.0 and (best is None or width * length < best[0]):
                    mid_c = (lo_c + hi_c) / 2
                    start = (ux * lo_a - uy * mid_c, uy * lo_a + ux * mid_c)
                    end = (ux * hi_a - uy * mid_c, uy * hi_a + ux * mid_c)
                    best = (width * length, start, end, width)
            if best is None:
                continue
            _, start, end, width = best
            count += 1
            wall_id = f"SV-{s.key}-{count:03d}"
            walls.append({
                "id": wall_id, "start": list(start), "end": list(end),
                "thickness": max(width, 10.0) + 2.0, "kind": "unknown",
                "confidence": {"level": "review", "score": 0.5, "reasons": ["sliver_filled"]}, "source_handles": [],
            })
            openings += _zone_holes(wall_id, start, end, part, zones)
        if count:
            ir["walls"], ir["openings"] = walls, openings
            s.part = dataclasses.replace(s.part, ir=ir)
        filled[s.key] = count
    return filled


def trim_partial_floors(storeys: list[Storey]) -> dict[str, int]:
    """부분층 평면에서 기준 층 건물 범위 밖에 그린 것은 그 층에 넣지 않는다. 층마다 뺀 벽 수를 돌려준다.

    '4층 상부' 평면에 함께 그린 건물 밖 구조물은 다른 레벨(예: FH:137.60)이라 부분층 높이에 두면 공중에 뜬다.
    도면은 고치지 않는다. 빌드 기록에 남기고 결과 파일(ir.json)도 그대로 둔다.
    """
    removed: dict[str, int] = {}
    by_key = {s.key: s for s in storeys}
    for s in storeys:
        if not s.key.endswith(PARTIAL):
            continue
        base = by_key.get(s.key[:-len(PARTIAL)])
        if base is None or not base.part.ir["walls"] or s.align == "unaligned" or base.align == "unaligned":
            continue
        bx, by = base.part.ir["origin_offset"][0] + base.shift[0], base.part.ir["origin_offset"][1] + base.shift[1]
        xs = [p[0] + bx for w in base.part.ir["walls"] for p in (w["start"], w["end"])]
        ys = [p[1] + by for w in base.part.ir["walls"] for p in (w["start"], w["end"])]
        x0, x1 = min(xs) - _PARTIAL_MARGIN_MM, max(xs) + _PARTIAL_MARGIN_MM
        y0, y1 = min(ys) - _PARTIAL_MARGIN_MM, max(ys) + _PARTIAL_MARGIN_MM
        sx, sy = s.part.ir["origin_offset"][0] + s.shift[0], s.part.ir["origin_offset"][1] + s.shift[1]
        inside = lambda x, y: x0 <= x + sx <= x1 and y0 <= y + sy <= y1  # noqa: E731
        ir = dict(s.part.ir)
        walls = [w for w in ir["walls"] if inside((w["start"][0] + w["end"][0]) / 2, (w["start"][1] + w["end"][1]) / 2)]
        kept = {w["id"] for w in walls}
        removed[s.key] = len(ir["walls"]) - len(walls)
        ir["walls"] = walls
        ir["columns"] = [c for c in ir["columns"] if inside(*c["center"][:2])]
        ir["openings"] = [o for o in ir["openings"] if o.get("host_wall") in kept]
        for key in ("stairs", "voids"):
            ir[key] = [t for t in ir.get(key, []) if inside((t["bounds"][0] + t["bounds"][2]) / 2, (t["bounds"][1] + t["bounds"][3]) / 2)]
        ir["rails"] = [r for r in ir.get("rails", []) if inside((r["start"][0] + r["end"][0]) / 2, (r["start"][1] + r["end"][1]) / 2)]
        ir["rooms"] = [r for r in ir.get("rooms", []) if r["polygon"] and inside(
            sum(p[0] for p in r["polygon"]) / len(r["polygon"]), sum(p[1] for p in r["polygon"]) / len(r["polygon"]))]
        s.part = dataclasses.replace(s.part, ir=ir)
    return removed


_LAYER_GAP_MM = 450.0  # 문 뒤 나란한 벽까지 빈틈이 이보다 좁으면 같은 벽의 다른 겹으로 본다


def doors_through_layers(storeys: list[Storey]) -> dict[str, int]:
    """두 겹으로 읽힌 벽에서 문이 한 겹에만 뚫린 곳은 다른 겹에도 같은 문을 뚫는다(계산). 층마다 뚫은 수.

    도면의 벽선 두 줄을 벽 두 개로 읽으면, 문 바로 뒤에 나란한 벽(또는 창)이 남아 문을 지나갈 수 없다.
    겹친 쪽에 같은 자리의 창이 있으면 문으로 바꾼다.
    """
    added: dict[str, int] = {}
    for s in storeys:
        ir = dict(s.part.ir)
        walls = {w["id"]: w for w in ir["walls"] if w["start"] != w["end"]}
        openings = list(ir["openings"])
        count = 0
        voided: set[str] = set()  # 다른 겹의 문짝을 빈 구멍으로 바꾼 개구부. 한 문 자리에 문짝은 한 장이다
        for o in [o for o in openings if o["type"] in ("door", "unknown") and o.get("host_wall") in walls]:
            if o["id"] in voided:
                continue
            h = walls[o["host_wall"]]
            (hx0, hy0), (hx1, hy1) = h["start"][:2], h["end"][:2]
            hl = math.hypot(hx1 - hx0, hy1 - hy0)
            hu = ((hx1 - hx0) / hl, (hy1 - hy0) / hl)
            mid = o["position"]
            cx, cy = hx0 + hu[0] * mid, hy0 + hu[1] * mid
            for w in walls.values():
                if w["id"] == h["id"]:
                    continue
                (x0, y0), (x1, y1) = w["start"][:2], w["end"][:2]
                length = math.hypot(x1 - x0, y1 - y0)
                u = ((x1 - x0) / length, (y1 - y0) / length)
                if abs(u[0] * hu[1] - u[1] * hu[0]) > 0.05:
                    continue  # 나란하지 않다
                gap = abs((cx - x0) * -u[1] + (cy - y0) * u[0]) - (h["thickness"] + w["thickness"]) / 2
                if gap > _LAYER_GAP_MM:
                    continue
                along = (cx - x0) * u[0] + (cy - y0) * u[1]
                a, b = max(along - o["width"] / 2, 0.0), min(along + o["width"] / 2, length)
                if b - a < 0.5 * o["width"]:
                    continue  # 문 자리와 겹치는 길이가 짧다
                mine = [q for q in openings if q.get("host_wall") == w["id"]
                        and min(b, q["position"] + q["width"] / 2) - max(a, q["position"] - q["width"] / 2) > 0]
                passages = [q for q in mine if q["type"] in ("door", "unknown")]
                if passages:
                    if o["type"] == "door":  # 두 겹 모두 문으로 읽혔다: 이쪽 문짝만 남기고 저쪽은 빈 구멍
                        for q in passages:
                            if q["type"] == "door":
                                voided.add(q["id"])
                                openings[openings.index(q)] = {**q, "type": "unknown", "operation": None}
                                count += 1
                    continue
                openings = [q for q in openings if q not in mine]  # 같은 자리의 창은 빈 구멍으로 바꾼다
                # 문짝은 이쪽 겹에만 둔다. 저쪽 겹은 빈 구멍(종류 모름)이라 판 없이 뚫린다
                openings.append({**o, "id": f"{o['id']}~{w['id']}", "type": "unknown", "operation": None,
                                 "host_wall": w["id"], "position": (a + b) / 2, "width": b - a})
                voided.add(f"{o['id']}~{w['id']}")
                count += 1
        if count:
            ir["openings"] = openings
            s.part = dataclasses.replace(s.part, ir=ir)
        added[s.key] = count
    return added


_HANDOVER_EDGE_MM = 400.0  # 부분층 바닥판 가장자리로 보는 폭


def hand_over_to_partial(storeys: list[Storey], export: dict[str, Any]) -> dict[str, int]:
    """부분층 바닥 아래의 기준 층 벽은 부분층 바닥에서 멈춘다. 그 위는 부분층 벽(문·창 구멍 포함)이 잇는다.

    기준 층 벽이 층고 끝까지 올라가면 부분층 문 자리를 막는다(물류 4층 상부 문 15곳). 바닥판 범위는
    부분층 평면에서 계산하고, 걸친 벽은 그 경계에서 나눈다(계산). 기준 층마다 높이를 줄인 벽 수를 돌려준다.
    """
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    from shapely.affinity import translate

    changed: dict[str, int] = {}
    by_key = {s.key: s for s in storeys}
    for part in [s for s in storeys if s.key.endswith(PARTIAL)]:
        base = by_key.get(part.key[:-len(PARTIAL)])
        if base is None or part.align == "unaligned" or base.align == "unaligned":
            continue
        offset = part.elevation - base.elevation
        top = offset - export["slab_thickness_mm"]
        if top <= 0:
            continue
        outlines = floor_outlines(building_only(part.part.ir), export["slab_close_mm"], export["slab_min_area_m2"], export["slab_trim_mm"])
        if not outlines:
            continue
        sx = part.part.ir["origin_offset"][0] + part.shift[0] - base.part.ir["origin_offset"][0] - base.shift[0]
        sy = part.part.ir["origin_offset"][1] + part.shift[1] - base.part.ir["origin_offset"][1] - base.shift[1]
        cover = translate(unary_union(outlines), sx, sy).buffer(-50.0)
        # 바닥판 가장자리(외벽 자리)는 부분층 벽이 위에 있을 때만 넘긴다. 없으면 기준 층 벽이 그대로 올라가야 외벽이 뚫리지 않는다.
        above = unary_union([
            translate(LineString([w["start"][:2], w["end"][:2]]), sx, sy).buffer(w["thickness"] / 2 + 150.0, cap_style="flat")
            for w in part.part.ir["walls"] if w["start"] != w["end"]
        ])
        edge = cover.boundary.buffer(_HANDOVER_EDGE_MM)
        ir = dict(base.part.ir)
        walls, openings = [], list(ir["openings"])
        count = 0
        for w in ir["walls"]:
            (x0, y0), (x1, y1) = w["start"][:2], w["end"][:2]
            length = math.hypot(x1 - x0, y1 - y0)
            line = LineString([(x0, y0), (x1, y1)])
            inside = line.intersection(cover) if length > 0 else None
            if inside is not None and not inside.is_empty and inside.intersection(edge).length >= 0.8 * inside.length:
                inside = inside.intersection(above)  # 가장자리를 따라가는 외벽: 위에 부분층 벽이 있는 구간만
            if inside is None or inside.is_empty or inside.length < 300.0:
                walls.append(w)
                continue
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            spans = sorted(
                (min(a, b), max(a, b))
                for seg in getattr(inside, "geoms", [inside]) if seg.length >= 300.0
                for a, b in [tuple((px - x0) * ux + (py - y0) * uy for px, py in list(seg.coords)[::len(seg.coords) - 1])]
            )
            cuts, cursor = [], 0.0
            for a, b in spans:
                if a - cursor > 1.0:
                    cuts.append((cursor, a, False))
                cuts.append((max(a, cursor), b, True))
                cursor = b
            if length - cursor > 1.0:
                cuts.append((cursor, length, False))
            count += 1
            hosted = [o for o in openings if o.get("host_wall") == w["id"]]
            openings = [o for o in openings if o.get("host_wall") != w["id"]]
            for n, (a, b, low) in enumerate(cuts):
                piece = {**w, "id": w["id"] if n == 0 else f"{w['id']}~{n}",
                         "start": [x0 + ux * a, y0 + uy * a], "end": [x0 + ux * b, y0 + uy * b]}
                if low:
                    piece["top_mm"] = top
                walls.append(piece)
                for o in hosted:
                    if a <= o["position"] < b:
                        openings.append({**o, "host_wall": piece["id"], "position": o["position"] - a})
            for room in ir.get("rooms", []):
                if w["id"] in room.get("boundary_wall_ids", []):
                    room["boundary_wall_ids"] = room["boundary_wall_ids"] + [f"{w['id']}~{n}" for n in range(1, len(cuts))]
        if count:
            ir["walls"], ir["openings"] = walls, openings
            base.part = dataclasses.replace(base.part, ir=ir)
        changed[base.key] = count
    return changed


def _center(storey: Storey) -> tuple[float, float]:
    """층의 대표 위치. 기둥이 있으면 기둥 중심이고, 없으면 벽 끝점의 중앙값이다."""
    columns = _world_columns(storey.part.ir)
    ox, oy = storey.part.ir["origin_offset"]
    if len(columns) >= _MIN_COLUMNS_TO_ALIGN:
        x, y = np.median(columns, axis=0)
    else:
        points = [p for w in storey.part.ir["walls"] for p in (w["start"], w["end"])]
        x, y = (np.median(np.array(points, dtype=float), axis=0) + (ox, oy)) if points else (ox, oy)
    return float(x) + storey.shift[0], float(y) + storey.shift[1]


def _pair_shift(a: Storey, b: Storey) -> tuple[tuple[float, float], float, str] | None:
    """a를 b에 맞추는 이동량, 맞은 비율, 근거. 근거가 약하면 None."""
    if a.part.grid is not None and b.part.grid is not None:
        sx, share_x = _axis_shift(a.part.grid[0], b.part.grid[0])
        sy, share_y = _axis_shift(a.part.grid[1], b.part.grid[1])
        share = min(share_x, share_y)
        if share >= _MIN_GRID_SHARE:
            return (sx, sy), share, "grid"
    own, target = _world_columns(a.part.ir), _world_columns(b.part.ir)
    if len(own) < _MIN_COLUMNS_TO_ALIGN or len(target) < _MIN_COLUMNS_TO_ALIGN:
        # 격자선도 기둥도 모자라면 격자 기호의 이름으로 맞춘다(지붕층처럼 격자를 짧게 그린 평면).
        labelled = _label_shift(a.part, b.part)
        if labelled is None:
            return None
        shift, share = labelled
        return shift, share, "labels"
    shift, share = column_shift(own, target)
    same = float(np.mean(np.min(np.linalg.norm(own[:, None, :] - target[None, :, :], axis=2), axis=1) <= _GRID_MATCH_MM))
    if same >= share - _PREFER_ZERO_MARGIN and same >= _MIN_ALIGN_SHARE:
        return (0.0, 0.0), same, "columns"
    if share >= _MIN_ALIGN_SHARE:
        return shift, share, "columns"
    return None


def align_storeys(storeys: list[Storey]) -> None:
    """가장 잘 맞는 층끼리 차례로 잇는다. 어느 층과도 안 맞으면 옮기지 않고 표시한다.

    한 층을 기준으로 삼으면 그 층과 안 맞는 층이 모두 떨어져 나간다. 격자선은 건물 전체가 공유하므로
    기둥보다 먼저 쓴다.
    """
    if not storeys:
        return
    edges = []
    for i, a in enumerate(storeys):
        for j in range(i + 1, len(storeys)):
            found = _pair_shift(a, storeys[j])
            if found is not None:
                edges.append((found[2] == "grid", found[1], i, j, found[0], found[2]))
    edges.sort(key=lambda e: (e[0], e[1]), reverse=True)

    parent = list(range(len(storeys)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    links: dict[int, list[tuple[int, tuple[float, float], float, str]]] = {}
    for _, share, i, j, shift, method in edges:
        if find(i) != find(j):
            parent[find(i)] = find(j)
            links.setdefault(i, []).append((j, (-shift[0], -shift[1]), share, method))
            links.setdefault(j, []).append((i, shift, share, method))

    groups: dict[int, list[int]] = {}
    for index in range(len(storeys)):
        groups.setdefault(find(index), []).append(index)
    # 가장 큰 묶음을 건물로 본다. 같으면 기둥이 많은 쪽이다.
    main = max(groups.values(), key=lambda g: (len(g), max(len(storeys[i].part.ir["columns"]) for i in g)))
    root = max(main, key=lambda i: (len(storeys[i].part.ir["columns"]), storeys[i].part.part_id))

    visited = {root}
    pending = [root]
    while pending:
        current = pending.pop()
        for other, shift, share, method in links.get(current, []):
            if other in visited:
                continue
            visited.add(other)
            total = (storeys[current].shift[0] + shift[0], storeys[current].shift[1] + shift[1])
            moved = abs(total[0]) > _GRID_MATCH_MM or abs(total[1]) > _GRID_MATCH_MM
            storeys[other].shift = total if moved else (0.0, 0.0)
            storeys[other].align = method if moved else "same_origin"
            storeys[other].match = share
            pending.append(other)
    # 안 맞은 층은 같은 층의 다른 평면으로 다시 시도한다. 한 층에도 격자가 다른 사본이 있다.
    placed = [storeys[i] for i in visited]
    for index, storey in enumerate(storeys):
        if index in visited:
            continue
        storey.align = "unaligned"
        original = storey.part
        best = None
        for part in storey.others:
            storey.part = part
            for other in placed:
                found = _pair_shift(storey, other)
                if found is not None and (best is None or (found[2] == "grid", found[1]) > (best[3] == "grid", best[2])):
                    best = (part, (other.shift[0] + found[0][0], other.shift[1] + found[0][1]), found[1], found[2])
        storey.part = original
        if best is not None:
            storey.part, storey.shift, storey.match, storey.align = best
            storey.swapped = True
            placed.append(storey)
    # 끝내 못 맞춘 층은 건물 자리로 옮겨 둔다. 원래 좌표에 두면 수 km 떨어져 모델 전체를 보기 어렵다.
    # 위치는 근거가 없으므로 unaligned 표시는 그대로 남긴다.
    anchor = _center(storeys[root])
    for storey in storeys:
        if storey.align == "unaligned":
            own = _center(storey)
            storey.shift = (anchor[0] - own[0], anchor[1] - own[1])


_DEFAULT_FLOOR_HEIGHT = 3000.0


def attach_site(storeys: list[Storey], folder: Path, cfg: dict[str, Any], notes: dict[str, float]) -> dict[str, int]:
    """건물 밖 요소를 만들어 층에 달고 종류별 개수를 돌려준다. 위치를 못 맞춘 층은 쓰지 않는다."""
    out_dir = folder / OUT_DIR_NAME
    mapping = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
    paths = {p.name: p for p in find_drawings(folder)}
    export = cfg["export"]
    sources = []
    el_zero = None
    for s in storeys:
        if el_zero is None and s.key in notes:
            el_zero = notes[s.key] - s.elevation
        if s.align == "unaligned":
            continue
        ir = s.part.ir
        path = paths[Path(mapping[s.part.part_id.split(".")[0]]["file"]).name]
        pre = preprocess(_as_dxf(path, out_dir), cfg, ir["source"].get("region"), ir["source"]["unit_scale"], out_dir / "_prims")
        dx, dy = ir["origin_offset"][0] + s.shift[0], ir["origin_offset"][1] + s.shift[1]
        outlines = floor_outlines(building_only(ir), export["slab_close_mm"], export["slab_min_area_m2"], export["slab_trim_mm"])
        outline = translate(unary_union(outlines), dx, dy) if outlines else Polygon()
        rooms = [translate(Polygon(r["polygon"]), dx, dy) for r in ir.get("rooms", []) if len(r["polygon"]) >= 3]
        sources.append(read_source(s.key, s.elevation, pre.primitives, outline, (dx, dy), s.elevation + s.height, rooms))
    site = plan_site(sources, el_zero, export)
    for s in storeys:
        s.site = site
    return dict(site.counts)


def write_aerial_image(storeys: list[Storey], path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """조감도 그림. 렌더링 라이브러리가 없으면 건너뛰고 나머지 결과는 그대로 낸다."""
    if not cfg.get("aerial_enabled", True):
        return {"aerial": 0}
    try:
        from .export.raytrace import render_aerial
    except ImportError:
        return {"aerial": 0}
    collected, _ = _collect(storeys, cfg)
    view = choose_view(collected, cfg)
    if view is None:
        return {"aerial": 0}
    site = storeys[0].site
    # 메시는 공통 원점으로 옮겨져 있다. 주변 지면 테두리도 같은 좌표로 맞춘다.
    base_x, base_y = common_base(storeys)
    rim = [(x - base_x, y - base_y, z) for x, y, z in site.rim] if site is not None else []
    return {"aerial": 1, **render_aerial(collected, rim, view, path, cfg)}


def write_photo_image(storeys: list[Storey], path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """조감도 실사화. 전용 환경이나 그래픽 카드가 없으면 건너뛰고, 실패해도 다른 결과는 그대로 낸다."""
    if not cfg.get("photo_enabled") or worker_python(cfg) is None:
        return {"photo": 0}
    try:
        from .export.raytrace import render_passes
    except ImportError:
        return {"photo": 0}
    collected, _ = _collect(storeys, cfg)
    view = choose_view(collected, cfg)
    site = storeys[0].site
    if view is None or site is None or not site.rim:
        return {"photo": 0}
    base_x, base_y = common_base(storeys)
    rim = [(x - base_x, y - base_y, z) for x, y, z in site.rim]
    shot = photo_view(collected, view, cfg)
    render, depth, ids, normals, building = render_passes(collected, rim, shot, cfg, PHOTO_WIDTH, PHOTO_HEIGHT, 4, city_blocks(rim, cfg))
    edges = edge_lines(ids, depth, normals)
    try:
        first = paint(render, depth_map(depth), edges, path, cfg)
    except (RuntimeError, OSError) as error:
        return {"photo": 0, "photo_error": type(error).__name__}
    return {
        "photo": 1, "photo_building_share": round(float(building[ids[ids >= 0]].sum()) / ids.size, 3),
        "photo_edge_agreement": round(edge_agreement(edges, first), 3),
    }


def _part_drawings(folder: Path, out_dir: Path, storeys: list[Storey]) -> dict[str, Path]:
    """평면 조각 이름(D03.2) → 그 조각이 나온 DXF. 조각 이름의 점 앞이 도면 번호다."""
    mapping = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
    paths = {p.name: p for p in find_drawings(folder)}
    result = {}
    for s in storeys:
        entry = mapping.get(s.part.part_id.split(".")[0])
        name = Path(entry["file"]).name if entry else None
        if name in paths:
            result[s.part.part_id] = _as_dxf(paths[name], out_dir)
    return result


def run_build(folder: Path, height: float | None = None, jobs: int = 1) -> str:
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise Cad2BimError("입력 폴더를 찾을 수 없습니다.")
    _check_outside_project(folder)
    out_dir = folder / OUT_DIR_NAME
    if not (out_dir / "map.json").is_file():
        run_audit(folder, height or _DEFAULT_FLOOR_HEIGHT, jobs=jobs, quick=True)
    cfg = load_defaults()
    notes: dict[str, float] = {}
    parts, stats = read_parts(folder, cfg, notes)
    if not parts:
        raise Cad2BimError("층을 정할 수 있는 평면이 없습니다.")
    if not height:
        merge_section_levels(notes, section_levels(folder, cfg), [p.floor for p in parts])
    # 사용자가 준 층고가 먼저이고, 그다음이 도면 주기, 마지막이 기본값이다.
    storeys = select_storeys(parts, height or _DEFAULT_FLOOR_HEIGHT, None if height else notes)
    height_source = "user" if height else level_plan([s.key for s in storeys if is_full_floor(s.key)], notes, _DEFAULT_FLOOR_HEIGHT)[2]
    align_storeys(storeys)
    trimmed = trim_partial_floors(storeys)
    drawings = _part_drawings(folder, out_dir, storeys)
    from .review import review

    reviewed = review(storeys, drawings, out_dir, cfg)  # LLM 검토. 꺼져 있으면 아무것도 안 한다
    slivers = close_slivers(storeys, cfg["openings"]["min_gap_mm"])  # 좁은 틈은 판단 없이 메운다(계산)
    handed = hand_over_to_partial(storeys, cfg["export"])  # 부분층 바닥 아래 벽은 그 바닥에서 멈춘다(계산)
    layered = doors_through_layers(storeys)  # 두 겹 벽의 다른 겹에도 문을 뚫는다(계산)
    attach_elevators(storeys, drawings, out_dir, cfg)  # 승강로 바닥 뚫기(plan_floors)와 앱 표시에 쓴다
    site_counts = attach_site(storeys, folder, cfg, notes)
    ifc_path = out_dir / "building.ifc"
    counts = write_ifc(storeys, ifc_path, cfg["export"])
    check = verify_ifc(ifc_path)
    dae_path = out_dir / "building.dae"
    dae_counts, expected = write_dae(storeys, dae_path, cfg["export"])
    dae_check = verify_dae(dae_path, expected)
    write_furniture(storeys, drawings, out_dir, cfg)
    room_table = write_room_areas(storeys, drawings, out_dir, cfg)  # 실 면적표: 안목과 벽 중심선 기준, 도면 표기와의 차이
    if reviewed:
        from .review import locate_items

        locate_items(storeys, out_dir, common_base(storeys))  # 검토 지점을 모델과 같은 좌표로(앱의 검토 패널)
    dxf_path = out_dir / "building.dxf"
    dxf_counts, dxf_expected = write_dxf(storeys, dxf_path, cfg["export"])
    dxf_check = verify_dxf(dxf_path, dxf_expected)
    # 옛 프로그램용. R2004는 AutoCAD 2004~2006이 쓰는 형식이다.
    old_dxf_path = out_dir / "building_R2004.dxf"
    _, old_dxf_expected = write_dxf(storeys, old_dxf_path, cfg["export"], "R2004")
    old_dxf_check = verify_dxf(old_dxf_path, old_dxf_expected)
    editable = write_editable_files(storeys, out_dir, cfg["export"])
    sdk = find_sdk(cfg["export"])
    skp_lines = ["skp: unavailable"]
    if sdk is not None:
        skp_path = out_dir / "building.skp"
        old_skp_path = out_dir / "building_SU5_2005.skp"
        skp_counts, skp_expected = write_skp(storeys, skp_path, cfg["export"], sdk, (old_skp_path, SU5_2005))
        skp_check = verify_skp(skp_path, skp_expected, sdk)
        old_skp_check = verify_skp(old_skp_path, skp_expected, sdk)
        skp_lines = [
            "skp: " + " ".join(f"{k}={v}" for k, v in skp_counts.items()),
            "skp_check: " + " ".join(f"{k}={v}" for k, v in skp_check.items()),
            "skp_su5_check: " + " ".join(f"{k}={v}" for k, v in old_skp_check.items()),
        ]
    aerial = write_aerial_image(storeys, out_dir / "조감도.png", cfg["export"])
    aerial.update(write_photo_image(storeys, out_dir / "조감도_실사.png", cfg["export"]))
    (out_dir / "building.json").write_text(json.dumps([
        {
            "floor": s.key, "part": s.part.part_id, "elevation": s.elevation, "height": s.height,
            "shift": s.shift, "align": s.align, "match": s.match, "alternatives": s.alternatives,
        }
        for s in storeys
    ], indent=2), encoding="utf-8")
    lines = [
        "BUILD v1 " + " ".join(f"{k}={stats[k]}" for k in (
            "by_title", "by_file_name", "no_floor", "single_floor_assumed", "not_plan", "not_architectural", "no_walls", "failed",
        )),
        f"height_source={height_source} storeys={len(storeys)} aligned={sum(s.align != 'unaligned' for s in storeys)} "
        f"unaligned={sum(s.align == 'unaligned' for s in storeys)} swapped={sum(s.swapped for s in storeys)}",
    ]
    if reviewed:
        lines.append(reviewed)
    lines.append("slivers_filled: " + " ".join(f"{k}={v}" for k, v in slivers.items()))
    lines.append("partial_handover: " + " ".join(f"{k}={v}" for k, v in handed.items()))
    lines.append("doors_through_layers: " + " ".join(f"{k}={v}" for k, v in layered.items()))
    for key, n in trimmed.items():
        if n:
            lines.append(f"partial_outside[{key}]: walls={n} (기준 층 건물 범위 밖이라 부분층에 넣지 않음)")
    for s in storeys:
        ir = s.part.ir
        lines.append(
            f"floor[{s.key}]: elevation_m={s.elevation / 1000:.1f} align={s.align} "
            f"match={'-' if s.match is None else f'{s.match:.2f}'} alternatives={s.alternatives} "
            f"walls={len(ir['walls'])} columns={len(ir['columns'])} openings={len(ir['openings'])} rooms={len(ir['rooms'])}"
        )
    lines.append("ifc: " + " ".join(f"{k}={v}" for k, v in counts.items()))
    lines.append("ifc_check: " + " ".join(f"{k}={v}" for k, v in check.items()))
    lines.append("dae: " + " ".join(f"{k}={v}" for k, v in dae_counts.items()))
    lines.append("dae_check: " + " ".join(f"{k}={v}" for k, v in dae_check.items()))
    lines.append("dxf: " + " ".join(f"{k}={v}" for k, v in dxf_counts.items()))
    lines.append("dxf_check: " + " ".join(f"{k}={v}" for k, v in dxf_check.items()))
    lines.append("dxf_r2004_check: " + " ".join(f"{k}={v}" for k, v in old_dxf_check.items()))
    lines.extend(skp_lines)
    lines.append("room_areas: " + " ".join(f"{k}={v}" for k, v in room_table.items()))
    lines.append("editable: " + " ".join(f"{k}={v}" for k, v in editable.items()))
    lines.append("aerial: " + " ".join(f"{k}={v}" for k, v in aerial.items()))
    lines.append("site: " + (" ".join(f"{k}={v}" for k, v in sorted(site_counts.items())) or "none"))
    return "\n".join(lines) + "\n"
