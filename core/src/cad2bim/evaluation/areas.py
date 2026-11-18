"""면적표 대조. 도면 세트의 면적표를 읽어 인식한 실 면적과 비교한다 (상세설계 8.7).

정답 모델이 없을 때 인식 과정과 독립적인 근거다. 이름과 값은 PC 안에서만 비교하고, 출력은 개수와
오차뿐이다. 표에 숫자가 많으면 우연히 맞을 수 있어, 이름을 섞은 비교를 기준선으로 함께 낸다.
"""
from __future__ import annotations

import itertools
import json
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shapely.geometry import LineString, MultiPoint, Polygon, box
from shapely.ops import unary_union

from ..classify.floors import _FLOOR, floor_key, normalize_name, plan_title_floor
from ..config import load_defaults
from ..errors import Cad2BimError
from ..preprocess import load_primitives
from .audit import OUT_DIR_NAME, _as_dxf, _check_outside_project, find_drawings

_NUMBER = re.compile(r"^[\(\[]?\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:㎡|m2|m²)?\s*[\)\]]?$", re.I)
_HANGUL_NAME = re.compile(r"[가-힣]{2,}")
_MIN_AREA_M2 = 0.5
_FLOOR_AREA_HEADER = re.compile(r"바닥면적|층별면적|층면적")
_FLOOR_CELL_MAX = 8
_TOTAL_HEADER = re.compile(r"합계|소계|총계|^계$|^합$|연면적")
_COLUMN_SPAN = 0.5
_COLUMN_TOLERANCE = 2.0
_SINGLE_COLUMN_SPAN = 10.0
_SINGLE_ROW_PITCH = 5.0
_HEADER_PITCHES = 4.0
_HEADER_MAX_HEIGHT = 1.5
_SUM_RATE = 0.8
_SUM_MAX_CELLS = 16
_FORMULA = re.compile(r"\d\s*[xX×*]\s*\d|=")
_AREA_WORD = re.compile(r"면적|㎡|m2|구적", re.I)
_MAX_AREA_M2 = 500000.0
_TOLERANCES = (0.02, 0.05, 0.10)
_SHUFFLE_SEED = 20261105




@dataclass
class TableStats:
    texts: int = 0
    numbers: int = 0
    rows: int = 0
    names: int = 0
    floor_texts: int = 0
    formula_texts: int = 0
    area_word_texts: int = 0


@dataclass
class TableData:
    values: dict[str, set[float]] = field(default_factory=dict)
    per_file: list[TableStats] = field(default_factory=list)
    floor_column: dict[str, set[float]] = field(default_factory=dict)
    column_stats: Counter = field(default_factory=Counter)


def _number(text: str) -> float | None:
    match = _NUMBER.match(text.strip())
    if match is None:
        return None
    value = float(match.group(1).replace(",", ""))
    return value if _MIN_AREA_M2 <= value <= _MAX_AREA_M2 else None


def _group_rows(texts: list[Any]) -> list[list[Any]]:
    """글자를 높이로 행으로 묶는다."""
    heights = sorted(p.geometry["height"] for p in texts)
    tolerance = max(heights[len(heights) // 2], 1e-9) * 0.6
    ordered = sorted(texts, key=lambda p: p.geometry["position"][1])
    rows: list[list[Any]] = []
    previous = None
    for p in ordered:
        y = p.geometry["position"][1]
        if previous is None or y - previous > tolerance:
            rows.append([])
        rows[-1].append(p)
        previous = y
    return rows


def _is_sum(value: float, others: list[float]) -> bool:
    """값이 다른 값 2개 이상의 합이면 참. 합계 열인지 내용으로 검증한다."""
    candidates = [v for v in others if v > 0]
    if value <= 0 or len(candidates) > _SUM_MAX_CELLS:
        return False
    tolerance = max(0.02, value * 0.0005)
    return any(
        abs(sum(combo) - value) <= tolerance
        for size in range(2, len(candidates) + 1)
        for combo in itertools.combinations(candidates, size)
    )


def _columns(xs: list[float], tolerance: float) -> list[float]:
    """숫자 칸의 x를 열로 묶어 열 중심을 돌려준다."""
    groups: list[list[float]] = []
    for x in sorted(xs):
        if groups and x - groups[-1][-1] <= tolerance:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [statistics.median(g) for g in groups]


def _read_floor_group(
    group: list[tuple[list[Any], Any, str]], rows: list[list[Any]], texts: list[Any], stats: Counter
) -> dict[str, set[float]]:
    """같은 x에 놓인 층 칸들을 한 표로 보고 층 면적 열을 고른다. 확실하지 않으면 아무것도 읽지 않는다."""
    height = statistics.median(p.geometry["height"] for row, _, _ in group for p in row)
    fx = statistics.median(cell.geometry["position"][0] for _, cell, _ in group)
    tolerance = height * _COLUMN_TOLERANCE

    def numbers_of(row: list[Any]) -> list[tuple[float, float]]:
        return sorted(
            (p.geometry["position"][0], n)
            for p in row
            if p.geometry["position"][0] > fx + tolerance and (n := _number(p.geometry["text"])) is not None
        )

    cells = [(row, key, numbers_of(row)) for row, _, key in group]
    centers = _columns([x for _, _, numbers in cells for x, _ in numbers], tolerance)
    if not centers:
        stats["no_column"] += len(group)
        return {}
    spans = [b - a for a, b in zip(centers, centers[1:])]
    align = (statistics.median(spans) if spans else height * _SINGLE_COLUMN_SPAN) * _COLUMN_SPAN

    ys = sorted({row[0].geometry["position"][1] for row, _, _ in cells})
    pitch = statistics.median(b - a for a, b in zip(ys, ys[1:])) if len(ys) > 1 else height * _SINGLE_ROW_PITCH
    top, bottom = ys[-1], ys[0]

    def value_at(numbers: list[tuple[float, float]], center: float) -> float | None:
        near = [(abs(x - center), n) for x, n in numbers if abs(x - center) <= tolerance]
        return min(near)[1] if near else None

    def filled(center: float) -> bool:
        return all(value_at(numbers, center) is not None for _, _, numbers in cells)

    def headers_at(pattern: re.Pattern[str]) -> list[float]:
        found = set()
        for p in texts:
            x, y = p.geometry["position"]
            if not top < y <= top + pitch * _HEADER_PITCHES or p.geometry["height"] > height * _HEADER_MAX_HEIGHT:
                continue
            if not pattern.search(normalize_name(p.geometry["text"])):
                continue
            near = min(centers, key=lambda c: abs(c - x))
            if abs(near - x) <= align:
                found.add(near)
        return sorted(found)

    def sum_rate(center: float) -> float:
        hits = 0
        for _, _, numbers in cells:
            value = value_at(numbers, center)
            others = [n for x, n in numbers if x < center - tolerance]
            hits += value is not None and _is_sum(value, others)
        return hits / len(cells)

    named = [c for c in headers_at(_FLOOR_AREA_HEADER) if filled(c)]
    totals = [c for c in headers_at(_TOTAL_HEADER) if filled(c) and sum_rate(c) >= _SUM_RATE]
    choices = named or totals
    if len(choices) != 1:
        stats["ambiguous" if choices else "no_column"] += len(group)
        return {}
    center = choices[0]
    stats["by_area_header" if named else "by_total_column"] += 1

    values: dict[str, set[float]] = {}
    for _, key, numbers in cells:
        value = value_at(numbers, center)
        stats["column_read"] += 1
        values.setdefault(key, set()).add(value)  # type: ignore[arg-type]

    # 표 안의 소계 행과 맞으면 열을 제대로 읽었다는 독립 근거가 된다.
    floor_rows = {id(row): value_at(numbers, center) for row, _, numbers in cells}
    running = 0.0
    for row in sorted(rows, key=lambda r: -r[0].geometry["position"][1]):
        y = row[0].geometry["position"][1]
        if y > top or y < bottom - pitch * _HEADER_PITCHES:
            continue
        if id(row) in floor_rows:
            running += floor_rows[id(row)] or 0.0
        elif running and any(
            abs(p.geometry["position"][0] - fx) <= tolerance and _TOTAL_HEADER.search(normalize_name(p.geometry["text"]))
            for p in row
        ):
            value = value_at(numbers_of(row), center)
            if value is not None:
                stats["subtotal_checked"] += 1
                stats["subtotal_ok"] += abs(value - running) <= max(0.05, value * 0.001)
                running = 0.0
    return values


def read_floor_column(prims: list[Any]) -> tuple[dict[str, set[float]], Counter]:
    """층 행에서 층 면적 열의 값만 읽는다.

    가장 가까운 값을 고르면 비율과 합계까지 후보가 되어 실제보다 잘 맞아 보인다. 그래서 열을 먼저 정한다.
    1순위는 '바닥면적' 머리글이 있는 열이다. 없으면 합계 머리글이 있고 모든 층에 값이 있으며 왼쪽 값들의
    합과 일치하는 열을 쓴다. 후보가 없거나 둘 이상이면 값을 지어내지 않고 개수만 센다.
    """
    stats: Counter = Counter()
    texts = [p for p in prims if p.kind == "text" and p.geometry["text"].strip()]
    if not texts:
        return {}, stats
    stats["headers"] = sum(bool(_FLOOR_AREA_HEADER.search(normalize_name(p.geometry["text"]))) for p in texts)
    rows = _group_rows(texts)
    found: list[tuple[list[Any], Any, str]] = []
    for row in rows:
        for p in row:
            raw = p.geometry["text"]
            if _number(raw) is None and len(normalize_name(raw)) <= _FLOOR_CELL_MAX and (key := floor_key(raw)):
                found.append((row, p, key))
    stats["floor_rows"] = len(found)

    groups: list[list[tuple[list[Any], Any, str]]] = []
    for item in sorted(found, key=lambda f: f[1].geometry["position"][0]):
        x, height = item[1].geometry["position"][0], item[1].geometry["height"]
        if groups and x - groups[-1][-1][1].geometry["position"][0] <= height * _COLUMN_TOLERANCE:
            groups[-1].append(item)
        else:
            groups.append([item])

    values: dict[str, set[float]] = {}
    for group in groups:
        for key, numbers in _read_floor_group(group, rows, texts, stats).items():
            values.setdefault(key, set()).update(numbers)
    return values, stats


def read_table_rows(prims: list[Any]) -> tuple[dict[str, set[float]], TableStats]:
    """글자를 높이로 행으로 묶어, 이름 칸과 숫자 칸이 함께 있는 행을 읽는다."""
    texts = [p for p in prims if p.kind == "text" and p.geometry["text"].strip()]
    stats = TableStats(texts=len(texts))
    stats.floor_texts = sum(_FLOOR.search(p.geometry["text"]) is not None for p in texts)
    stats.formula_texts = sum(bool(_FORMULA.search(p.geometry["text"])) for p in texts)
    stats.area_word_texts = sum(bool(_AREA_WORD.search(p.geometry["text"])) for p in texts)
    if not texts:
        return {}, stats
    rows = _group_rows(texts)

    values: dict[str, set[float]] = {}
    for row in rows:
        row.sort(key=lambda p: p.geometry["position"][0])
        numbers = [n for p in row if (n := _number(p.geometry["text"])) is not None]
        names = [
            normalize_name(p.geometry["text"]) for p in row
            if _number(p.geometry["text"]) is None and _HANGUL_NAME.search(normalize_name(p.geometry["text"]))
        ]
        stats.numbers += len(numbers)
        if numbers and names:
            stats.rows += 1
            for name in names:
                values.setdefault(name, set()).update(numbers)
    stats.names = len(values)
    return values, stats


def load_tables(folder: Path, match: str, cfg: dict[str, Any]) -> TableData:
    data = TableData()
    out_dir = folder / OUT_DIR_NAME
    for path in find_drawings(folder):
        if match.casefold() not in path.name.casefold():
            continue
        dxf = _as_dxf(path, out_dir)
        model, stats = load_primitives(dxf, cfg, out_dir / "_prims")
        values, table = read_table_rows([*model, *stats.paper])
        data.per_file.append(table)
        column, column_stats = read_floor_column([*model, *stats.paper])
        data.column_stats.update(column_stats)
        for key, numbers in column.items():
            data.floor_column.setdefault(key, set()).update(numbers)
        for name, numbers in values.items():
            data.values.setdefault(name, set()).update(numbers)
    return data


def read_plan_rooms(folder: Path, match: str) -> list[tuple[str, float]]:
    """지난 audit이 저장한 결과에서 평면도의 (이름, 면적)을 읽는다."""
    out_dir = folder / OUT_DIR_NAME
    map_path = out_dir / "map.json"
    if not map_path.is_file():
        raise Cad2BimError("지난 점검 결과가 없습니다. 먼저 audit을 실행하세요.")
    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    rooms: list[tuple[str, float]] = []
    for drawing_id, entry in mapping.items():
        if entry.get("status") != "ok" or match.casefold() not in Path(entry["file"]).name.casefold():
            continue
        for directory in sorted(out_dir.glob(f"{drawing_id}*")):
            ir_path = directory / "ir.json"
            if not ir_path.is_file():
                continue
            ir = json.loads(ir_path.read_text(encoding="utf-8"))
            for room in ir["rooms"]:
                if room["names"] == ["미지정"] or len(room["names"]) != 1:
                    continue
                rooms.append((normalize_name(room["names"][0]), float(room["clear_area_m2"])))
    return rooms


_ROOF = re.compile(r"옥탑|지붕|PH", re.I)
_CLOSE_MM = 3000.0  # 개구부 정도의 틈을 메운다. 틈과 비슷하면 닫혔다가 다시 열린다


def name_kinds(names: Any) -> dict[str, int]:
    """표 이름의 종류만 센다. 층별 표인지 실별 표인지 내용을 보지 않고 가른다."""
    kinds = {"floor": 0, "total": 0, "other": 0}
    for name in names:
        if floor_key(name) is not None or _ROOF.search(name):
            kinds["floor"] += 1
        elif re.search(r"합계|소계|총계|^계$|연면적|바닥면적|건축면적", name):
            kinds["total"] += 1
        else:
            kinds["other"] += 1
    return kinds


_ESTIMATED_WALL = ("wall_layer_inferred", "wall_from_door", "layer_unresolved")


def outer_area_m2(
    ir: dict[str, Any], close_mm: float = _CLOSE_MM, use_walls: bool = True, use_columns: bool = True,
    confirmed_walls_only: bool = False,
) -> float | None:
    """벽과 기둥을 틈 메워 합친 외곽 면적. 바닥면적과 비교할 근사값이다."""
    shapes = []
    if use_walls:
        for w in ir["walls"]:
            if w["start"] == w["end"]:
                continue
            if confirmed_walls_only and any(
                r in _ESTIMATED_WALL for r in w.get("confidence", {}).get("reasons", [])
            ):
                continue
            shapes.append(LineString([w["start"], w["end"]]).buffer(w["thickness"] / 2))
    if use_columns:
        for c in ir["columns"]:
            half_w, half_d = c["size"][0] / 2, c["size"][1] / 2
            shapes.append(
                box(c["center"][0] - half_w, c["center"][1] - half_d, c["center"][0] + half_w, c["center"][1] + half_d)
            )
    if not shapes:
        return None
    closed = unary_union(shapes).buffer(close_mm).buffer(-close_mm)
    parts = closed.geoms if hasattr(closed, "geoms") else [closed]
    return sum(Polygon(p.exterior).area for p in parts) / 1e6


def column_hull_m2(ir: dict[str, Any]) -> float | None:
    """기둥 중심을 감싸는 볼록 껍질. 건물 구조체 범위의 근사값이다."""
    if len(ir["columns"]) < 3:
        return None
    return MultiPoint([tuple(c["center"]) for c in ir["columns"]]).convex_hull.area / 1e6


AREA_VARIANTS = {
    "all": lambda ir: outer_area_m2(ir),
    "tight": lambda ir: outer_area_m2(ir, close_mm=1000.0),
    "walls": lambda ir: outer_area_m2(ir, use_columns=False),
    "confirmed": lambda ir: outer_area_m2(ir, confirmed_walls_only=True),
    "columns": column_hull_m2,
}


def read_plan_variants(folder: Path, match: str) -> list[tuple[str, dict[str, float]]]:
    """평면마다 (층, 방식별 면적). 한 파일에 평면이 여럿이면 가장 큰 평면을 쓴다."""
    out_dir = folder / OUT_DIR_NAME
    mapping = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
    result = []
    for drawing_id, entry in mapping.items():
        name = Path(entry["file"]).name
        key = floor_key(name)
        if entry.get("status") != "ok" or key is None or match.casefold() not in name.casefold():
            continue
        best: dict[str, float] = {}
        for directory in sorted(out_dir.glob(f"{drawing_id}*")):
            ir_path = directory / "ir.json"
            if not ir_path.is_file():
                continue
            ir = json.loads(ir_path.read_text(encoding="utf-8"))
            for variant, function in AREA_VARIANTS.items():
                area = function(ir)
                if area:
                    best[variant] = max(best.get(variant, 0.0), area)
        if best:
            result.append((key, best))
    return result


def variant_lines(plans: list[tuple[str, dict[str, float]]], table: dict[str, set[float]]) -> list[str]:
    """방식별 중앙 비율과 10% 이내 개수, 층별 중앙 비율. 무엇이 면적을 키우는지 가른다."""
    lines = []
    for variant in AREA_VARIANTS:
        pairs = [
            (key, areas[variant]) for key, areas in plans if variant in areas and key in table
        ]
        if not pairs:
            lines.append(f"variant[{variant}]: n=0")
            continue
        ratios = [area / min(table[key], key=lambda v: abs(area - v)) for key, area in pairs]
        # 우연 기준선: 다른 층의 값 묶음과 비교한다. 값이 많으면 우연히도 잘 맞는다.
        rng = random.Random(_SHUFFLE_SEED)
        floors = sorted(table)
        chance = []
        for key, area in pairs:
            others = [f for f in floors if f != key]
            if others:
                values = table[rng.choice(others)]
                chance.append(area / min(values, key=lambda v: abs(area - v)))
        lines.append(
            f"variant[{variant}]: n={len(ratios)} median_ratio={statistics.median(ratios):.3f} "
            f"within_10pct={sum(abs(r - 1) <= 0.1 for r in ratios)} within_25pct={sum(abs(r - 1) <= 0.25 for r in ratios)} "
            f"chance_10pct={sum(abs(r - 1) <= 0.1 for r in chance)} chance_25pct={sum(abs(r - 1) <= 0.25 for r in chance)}"
        )
    by_floor: dict[str, list[tuple[float, float, float]]] = {}
    for key, areas in plans:
        if key in table and "all" in areas:
            chosen = min(table[key], key=lambda v: abs(areas["all"] - v))
            by_floor.setdefault(key, []).append((areas["all"] / chosen, areas.get("columns", 0.0) / chosen, len(table[key])))
    for key, rows in sorted(by_floor.items()):
        lines.append(
            f"floor[{key}]: plans={len(rows)} table_values={int(rows[0][2])} "
            f"all={statistics.median(r[0] for r in rows):.2f} columns={statistics.median(r[1] for r in rows):.2f}"
        )
    return lines


def read_titled_plans(
    folder: Path, match: str, cfg: dict[str, Any]
) -> tuple[list[tuple[str, dict[str, float]]], Counter]:
    """평면마다 (제목의 층, 방식별 면적). 제목이 없는 평면은 층을 추측하지 않고 개수만 센다."""
    out_dir = folder / OUT_DIR_NAME
    mapping = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
    paths = {p.name: p for p in find_drawings(folder)}
    plans: list[tuple[str, dict[str, float]]] = []
    relation: Counter = Counter()
    for drawing_id, entry in mapping.items():
        name = Path(entry["file"]).name
        if entry.get("status") != "ok" or match.casefold() not in name.casefold() or name not in paths:
            continue
        file_key = floor_key(name)
        texts = None
        for directory in sorted(out_dir.glob(f"{drawing_id}*")):
            ir_path = directory / "ir.json"
            if not ir_path.is_file():
                continue
            ir = json.loads(ir_path.read_text(encoding="utf-8"))
            if texts is None:
                texts, _ = load_primitives(_as_dxf(paths[name], out_dir), cfg, out_dir / "_prims")
            label = plan_title_floor(texts, ir["source"].get("region"))
            if label is None:
                relation["no_title"] += 1
                continue
            relation["no_file_floor" if file_key is None else "same_as_file" if file_key == label else "differs"] += 1
            areas = {variant: function(ir) for variant, function in AREA_VARIANTS.items()}
            plans.append((label, {variant: area for variant, area in areas.items() if area}))
    return plans, relation


def floor_best_lines(plans: list[tuple[str, dict[str, float]]], table: dict[str, set[float]]) -> list[str]:
    """층마다 제목이 붙은 평면 중 가장 큰 것을 대표로 표와 비교한다. 표를 보지 않고 고르는 규칙이다.

    표 값이 층마다 하나일 때만 쓴다. 평면이 아니라 층을 세므로 같은 층이 여러 번 세지지 않는다.
    """
    lines = []
    floors = sorted({key for key, _ in plans} & {key for key, values in table.items() if len(values) == 1})
    for variant in AREA_VARIANTS:
        best = {
            key: max((areas[variant] for k, areas in plans if k == key and variant in areas), default=None)
            for key in floors
        }
        known = {key: area for key, area in best.items() if area}
        if not known:
            lines.append(f"floor_best[{variant}]: floors=0")
            continue
        ratios = [area / next(iter(table[key])) for key, area in known.items()]
        rng = random.Random(_SHUFFLE_SEED)
        chance = []
        for key, area in known.items():
            others = [f for f in floors if f != key]
            if others:
                chance.append(area / next(iter(table[rng.choice(others)])))
        lines.append(
            f"floor_best[{variant}]: floors={len(ratios)} median_ratio={statistics.median(ratios):.3f} "
            f"within_10pct={sum(abs(r - 1) <= 0.1 for r in ratios)} within_25pct={sum(abs(r - 1) <= 0.25 for r in ratios)} "
            f"chance_10pct={sum(abs(r - 1) <= 0.1 for r in chance)} chance_25pct={sum(abs(r - 1) <= 0.25 for r in chance)}"
        )
    for key in floors:
        target = next(iter(table[key]))
        rows = [areas for k, areas in plans if k == key]
        cells = " ".join(
            f"{variant}={max((a[variant] for a in rows if variant in a), default=0.0) / target:.2f}"
            for variant in ("all", "columns")
        )
        lines.append(f"floor_best_row[{key}]: parts={len(rows)} {cells}")
    return lines


def read_plan_floors(folder: Path, match: str) -> list[tuple[str, float]]:
    """파일명의 층과 외곽 면적. 층을 못 읽는 평면은 뺀다. 한 파일 여러 평면은 큰 것만 쓴다."""
    out_dir = folder / OUT_DIR_NAME
    mapping = json.loads((out_dir / "map.json").read_text(encoding="utf-8"))
    floors = []
    for drawing_id, entry in mapping.items():
        name = Path(entry["file"]).name
        if entry.get("status") != "ok" or match.casefold() not in name.casefold():
            continue
        key = floor_key(name)
        if key is None:
            continue
        areas = []
        for directory in sorted(out_dir.glob(f"{drawing_id}*")):
            ir_path = directory / "ir.json"
            if ir_path.is_file():
                area = outer_area_m2(json.loads(ir_path.read_text(encoding="utf-8")))
                if area:
                    areas.append(area)
        if areas:
            floors.append((key, max(areas)))
    return floors


def _floor_table(table: dict[str, set[float]]) -> dict[str, set[float]]:
    result: dict[str, set[float]] = {}
    for name, values in table.items():
        key = floor_key(name)
        if key is not None:
            result.setdefault(key, set()).update(values)
    return result


def _error(area: float, values: set[float]) -> float:
    return min(abs(area - v) / v for v in values)


def compare_areas(rooms: list[tuple[str, float]], table: dict[str, set[float]]) -> dict[str, Any]:
    matched = [(name, area) for name, area in rooms if name in table]
    errors = [_error(area, table[name]) for name, area in matched]
    # 우연 기준선: 같은 실에 다른 이름의 값을 붙여 비교한다.
    rng = random.Random(_SHUFFLE_SEED)
    names = sorted(table)
    baseline = []
    for name, area in matched:
        others = [n for n in names if n != name]
        if others:
            baseline.append(_error(area, table[rng.choice(others)]))
    result: dict[str, Any] = {"rooms": len(rooms), "name_found": len(matched)}
    result["median_error"] = statistics.median(errors) if errors else None
    # 1보다 작으면 인식 면적이 표보다 작다. 벽을 놓쳤을 때 나타나는 방향이다.
    ratios = [area / min(table[name], key=lambda v: abs(area - v)) for name, area in matched]
    result["median_ratio"] = statistics.median(ratios) if ratios else None
    for tolerance in _TOLERANCES:
        key = f"within_{round(tolerance * 100)}pct"
        result[key] = sum(e <= tolerance for e in errors)
        result[f"chance_{key}"] = sum(e <= tolerance for e in baseline)
    return result


def run_areas(folder: Path, table_match: str, plan_match: str) -> str:
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise Cad2BimError("입력 폴더를 찾을 수 없습니다.")
    _check_outside_project(folder)
    data = load_tables(folder, table_match, load_defaults())
    if not data.per_file:
        raise Cad2BimError("--table-match에 맞는 도면이 없습니다.")
    lines = [
        f"AREA-CHECK v1 table_files={len(data.per_file)} table_names={len(data.values)} "
        f"table_values={sum(len(v) for v in data.values.values())}",
    ]
    for index, table in enumerate(data.per_file, start=1):
        lines.append(f"T{index:02d}: texts={table.texts} numbers={table.numbers} rows={table.rows} names={table.names} "
            f"floor_texts={table.floor_texts} formula_texts={table.formula_texts} area_words={table.area_word_texts}")
    def fmt(result: dict[str, Any]) -> str:
        return " ".join(
            f"{key}={f'{value:.3f}' if isinstance(value, float) else '-' if value is None else value}"
            for key, value in result.items()
        )

    kinds = name_kinds(data.values)
    lines.append(f"table_name_kinds: floor={kinds['floor']} total={kinds['total']} other={kinds['other']}")
    lines.append("rooms: " + fmt(compare_areas(read_plan_rooms(folder, plan_match), data.values)))
    floor_table = _floor_table(data.values)
    floors = read_plan_floors(folder, plan_match)
    lines.append(f"floor_keys: table={len(floor_table)} plans={len(floors)} plan_keys={len({k for k, _ in floors})}")
    lines.append("floors: " + fmt(compare_areas(floors, floor_table)))
    variants = read_plan_variants(folder, plan_match)
    lines.extend(variant_lines(variants, floor_table))
    # 바닥면적 열의 값만 쓴 비교. 앞으로의 주 지표다.
    column = data.floor_column
    per_floor = sorted(len(v) for v in column.values())
    lines.append(
        f"floor_column: headers={data.column_stats['headers']} floor_rows={data.column_stats['floor_rows']} "
        f"read={data.column_stats['column_read']} no_column={data.column_stats['no_column']} "
        f"ambiguous={data.column_stats['ambiguous']} by_area_header={data.column_stats['by_area_header']} "
        f"by_total_column={data.column_stats['by_total_column']} "
        f"subtotal_ok={data.column_stats['subtotal_ok']}/{data.column_stats['subtotal_checked']} "
        f"floors={len(column)} values_per_floor_max={per_floor[-1] if per_floor else 0}"
    )
    lines.extend(f"strict_{line}" for line in variant_lines(variants, column))
    # 파일명 대신 평면 제목의 층으로 비교한다. 앞으로의 주 지표다.
    titled, relation = read_titled_plans(folder, plan_match, load_defaults())
    lines.append(
        f"plan_titles: titled={len(titled)} no_title={relation['no_title']} same_as_file={relation['same_as_file']} "
        f"differs={relation['differs']} no_file_floor={relation['no_file_floor']}"
    )
    lines.extend(floor_best_lines(titled, column))
    return "\n".join(lines) + "\n"
