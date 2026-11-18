"""원인을 가르는 진단 숫자. 이름은 내보내지 않고 익명 번호와 허용 단어의 개수만 낸다."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from ..layers import LayerInfo, LayerMap, base_name
from ..preprocess.primitive import Primitive
from ..recognize.columns import _candidate
from ..recognize.walls import collect_segments, find_pieces, group_by_direction

_BAND_GAP_MM = 5000.0
_TOP_LAYERS = 8
# 건축 도면의 일반 용어만 센다. 프로젝트 고유 이름이 여기에 들어가면 안 된다.
_GENERIC_TOKENS = (
    "WALL", "CONC", "PANEL", "RC", "PC", "BLK", "BLOCK", "COL", "BEAM", "SLAB", "ST", "STL", "SHUT", "DOCK",
    "RACK", "FIRE", "GL", "WIN", "DR", "XREF", "PILE", "FND", "FOOT", "PAT", "HAT", "CEN", "GRID", "DIM",
    "벽", "조적", "패널", "판넬", "콘크리트", "기둥", "보", "슬래브", "철골", "셔터", "방화", "창", "문",
    "외벽", "내벽", "칸막이", "경량", "난간", "계단", "해치",
)
_TOKEN_SPLIT = re.compile(r"[-_ .]+")
_TOP_GROUPS = 6
_GROUP_SAMPLE = 1500
_GRID_ALIGN_MM = 10.0
# 표준 해치 이름만 그대로 찍는다. 그 밖의 이름은 사무소 고유일 수 있어 custom으로 센다.
_STANDARD_HATCH = re.compile(
    r"^(SOLID|ANSI\d+|ISO\d+\w*|AR-[A-Z0-9]+|DOTS|NET\d*|LINE|CROSS|ANGLE|BRICK|BRSTONE|CLAY|CORK|EARTH|"
    r"GRASS|GRATE|HEX|HONEY|INSUL|MUDST|PLAST|PLASTI|SACNCR|SQUARE|STARS|STEEL|SWAMP|TRANS|TRIANG|ZIGZAG|"
    r"DASH|DOLMIT|FLEX|GOST_\w+|GRAVEL)$",
    re.I,
)


def _bands(lo: np.ndarray, hi: np.ndarray) -> list[float]:
    """한 축에서 큰 빈틈으로 나뉘는 구간의 폭(m). 평면이 나란히 있는지 본다."""
    order = np.argsort(lo)
    bands, start, stop = [], float(lo[order[0]]), float(hi[order[0]])
    for i in order[1:]:
        if lo[i] - stop > _BAND_GAP_MM:
            bands.append((stop - start) / 1000)
            start = float(lo[i])
        stop = max(stop, float(hi[i]))
    bands.append((stop - start) / 1000)
    return bands


def layout_line(prims: list[Primitive]) -> str:
    boxes = np.array([b for b in (p.bbox() for p in prims) if b], dtype=float).reshape(-1, 4)
    if not len(boxes):
        return "layout: -"
    x = _bands(boxes[:, 0], boxes[:, 2])
    y = _bands(boxes[:, 1], boxes[:, 3])
    def fmt(values: list[float]) -> str:
        top = sorted(values, reverse=True)[:6]
        return f"n={len(values)} top_m=" + ",".join(str(round(v)) for v in top)

    longest = max((b[2] - b[0] for b in boxes), default=0.0) / 1000
    return f"layout: x_bands {fmt(x)} y_bands {fmt(y)} longest_element_m={round(longest)}"


def _tokens(name: str) -> set[str]:
    upper = base_name(name).upper()
    parts = set(_TOKEN_SPLIT.split(upper))
    return {t for t in _GENERIC_TOKENS if t in parts or (not t.isascii() and t in upper)}


def layer_lines(prims: list[Primitive], layers: LayerMap, cfg: dict[str, Any]) -> list[str]:
    """미분류 레이어가 벽처럼 그려졌는지 기하로 본다. 이름 대신 U1, U2 번호를 쓴다."""
    unknown = [n for n, i in layers.layers.items() if i.role == "UNKNOWN"]
    if not unknown:
        return ["unknown_layers: -"]
    tokens: Counter = Counter()
    prefixed = 0
    for name in unknown:
        tokens.update(_tokens(name))
        prefixed += base_name(name) != name
    token_line = " ".join(f"{k}={v}" for k, v in sorted(tokens.items())) or "-"

    view = LayerMap({
        n: i if i.role == "UNKNOWN" else LayerInfo("OTHER", i.source) for n, i in layers.layers.items()
    })
    wc = cfg["walls"]
    segments = collect_segments(prims, view, [], wc)
    total: Counter = Counter()
    for s in segments:
        total[s.layer] += s.length
    paired: Counter = Counter()
    thickness: dict[str, Counter] = defaultdict(Counter)
    for group in group_by_direction(segments, math.radians(wc["direction_tolerance_deg"])):
        for piece in find_pieces(group, wc, True):
            for layer in piece.layers:
                paired[layer] += piece.length
                thickness[layer][round(piece.thickness / 10) * 10] += piece.length
    rows = []
    for index, (layer, length) in enumerate(total.most_common(_TOP_LAYERS), start=1):
        mode = thickness[layer].most_common(1)[0][0] if thickness[layer] else "-"
        share = min(paired[layer] / length, 1.0) if length else 0.0
        rows.append(f"U{index}: len_m={round(length / 1000)} paired={share:.2f} thick_mm={mode}")
    return [
        f"unknown_layers: n={len(unknown)} xref_prefix={prefixed} tokens: {token_line}",
        *rows,
    ]


def hatch_pattern_line(prims: list[Primitive]) -> str:
    """벽 재질 추정에 쓸 해치 이름. 표준 이름 외에는 찍지 않는다."""
    counts: Counter = Counter()
    for p in prims:
        if p.kind == "hatch":
            name = str(p.geometry["pattern"]).upper()
            counts[name if _STANDARD_HATCH.match(name) else "custom"] += 1
    return "hatch_patterns: " + (" ".join(f"{k}={v}" for k, v in counts.most_common(8)) or "-")


def column_group_lines(prims: list[Primitive], layers: LayerMap) -> list[str]:
    """기둥 크기 사각형을 크기별로 묶어 격자에 놓였는지 본다. 기둥이 안 나오는 이유를 가른다."""
    groups: dict[tuple[int, int], list[tuple[str, str, tuple[float, float]]]] = defaultdict(list)
    for p in prims:
        if p.kind not in ("polyline", "solid", "hatch"):
            continue
        found = _candidate(p)
        if found is None or found[0] != "rect":
            continue
        size = found[2]
        if not all(200 <= v <= 2000 for v in size):
            continue
        key = tuple(sorted((round(size[0] / 10) * 10, round(size[1] / 10) * 10)))
        groups[key].append((p.kind, layers.role(p.layer), (found[1].centroid.x, found[1].centroid.y)))

    rows = []
    ranked = sorted(groups.items(), key=lambda item: -len(item[1]))[:_TOP_GROUPS]
    for index, ((width, depth), members) in enumerate(ranked, start=1):
        kinds = Counter(m[0] for m in members)
        roles = Counter(m[1] for m in members)
        centers = np.array([m[2] for m in members[:_GROUP_SAMPLE]])
        share = spacing = 0.0
        if len(centers) >= 2:
            dx = np.abs(centers[:, None, 0] - centers[None, :, 0])
            dy = np.abs(centers[:, None, 1] - centers[None, :, 1])
            np.fill_diagonal(dx, np.inf)
            np.fill_diagonal(dy, np.inf)
            share = float((((dx <= _GRID_ALIGN_MM).any(axis=1)) & ((dy <= _GRID_ALIGN_MM).any(axis=1))).mean())
            spacing = float(np.median(np.hypot(dx, dy).min(axis=1))) / 1000
        rows.append(
            f"CG{index}: size_mm={width}x{depth} n={len(members)} grid={share:.2f} spacing_m={spacing:.1f} "
            f"kinds={'/'.join(f'{k}:{v}' for k, v in kinds.most_common())} "
            f"roles={'/'.join(f'{k}:{v}' for k, v in roles.most_common())}"
        )
    return rows or ["CG: -"]
