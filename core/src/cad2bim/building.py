"""건물 합치기. 도면별 평면에서 층을 정하고 위치를 맞춰 건물 하나로 묶는다.

출력은 숫자와 층 코드뿐이다. 파일명, 제목, 실 이름은 찍지 않는다.
"""
from __future__ import annotations

import json
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
from .classify.floors import floor_key, plan_title_text, read_level_notes
from .config import load_sheet_dictionary
from .config import load_defaults
from .errors import Cad2BimError
from .evaluation.audit import OUT_DIR_NAME, _as_dxf, _check_outside_project, column_shift, find_drawings, run_audit
from .export.aerial import choose_view
from .export.collada import _collect, common_base, verify_dae, write_dae
from .export.dxf3d import verify_dxf, write_dxf
from .export.floor import floor_outlines
from .export.floors import write_editable_files
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
            floor = floor_key(label)
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
            roles = {
                name: info["role"]
                for name, info in json.loads((directory / "layer_map.json").read_text(encoding="utf-8"))["layers"].items()
            }
            parts.append(PlanPart(
                directory.name, floor, ir, _wall_length(ir), "title" if title else "file", grid_lines(model, roles, ir),
                floor_key(name) == floor,
            ))
    # 층 표기가 전혀 없으면 단층 건물로 본다. 평면이 있는데 변환을 포기하지 않기 위해서다.
    if not parts and unfloored:
        parts.append(max(unfloored, key=lambda p: (bool(p.ir["openings"] or p.ir["rooms"]), p.wall_length)))
        stats["single_floor_assumed"] += 1
    return parts, stats


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


def select_storeys(parts: list[PlanPart], height: float, levels: dict[str, float] | None = None) -> list[Storey]:
    """층마다 건축 평면다운 평면을 대표로 쓴다.

    문이나 실이 있어야 건축 평면이다. 벽만 길고 둘 다 없는 평면은 구조나 마감 사본일 수 있다.
    파일명과 제목의 층이 같은 평면을 먼저 쓴다. 1층 파일에 곁들인 2층 평면은 2층 파일보다 믿을 근거가 약하다.
    그중 벽이 가장 긴 평면을 쓴다. 문과 실 개수로 고르면 부분 평면이 뽑힌다.
    """
    by_floor: dict[str, list[PlanPart]] = {}
    for part in parts:
        by_floor.setdefault(part.floor, []).append(part)
    storeys = []
    for key, candidates in by_floor.items():
        best = max(candidates, key=lambda p: (bool(p.ir["openings"] or p.ir["rooms"]), p.file_agrees, p.wall_length, p.part_id))
        others = sorted(
            (p for p in candidates if p is not best and (p.ir["openings"] or p.ir["rooms"])),
            key=lambda p: -p.wall_length,
        )
        storeys.append(Storey(key, best, elevation_of(key, height), height, alternatives=len(candidates) - 1, others=others))
    if levels:
        elevations, heights, _ = level_plan([s.key for s in storeys], levels, height)
        for storey in storeys:
            storey.elevation, storey.height = elevations[storey.key], heights[storey.key]
    return sorted(storeys, key=lambda s: s.elevation)


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
        return None
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
    # 사용자가 준 층고가 먼저이고, 그다음이 도면 주기, 마지막이 기본값이다.
    storeys = select_storeys(parts, height or _DEFAULT_FLOOR_HEIGHT, None if height else notes)
    height_source = "user" if height else level_plan([s.key for s in storeys], notes, _DEFAULT_FLOOR_HEIGHT)[2]
    align_storeys(storeys)
    site_counts = attach_site(storeys, folder, cfg, notes)
    ifc_path = out_dir / "building.ifc"
    counts = write_ifc(storeys, ifc_path, cfg["export"])
    check = verify_ifc(ifc_path)
    dae_path = out_dir / "building.dae"
    dae_counts, expected = write_dae(storeys, dae_path, cfg["export"])
    dae_check = verify_dae(dae_path, expected)
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
    lines.append("editable: " + " ".join(f"{k}={v}" for k, v in editable.items()))
    lines.append("aerial: " + " ".join(f"{k}={v}" for k, v in aerial.items()))
    lines.append("site: " + (" ".join(f"{k}={v}" for k, v in sorted(site_counts.items())) or "none"))
    return "\n".join(lines) + "\n"
