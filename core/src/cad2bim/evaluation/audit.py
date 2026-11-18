"""실제 도면 점검. 사용자가 터미널에서 돌리고 출력 블록을 LLM에 붙여 넣는다 (상세설계 8.5).

출력은 허용한 항목만 골라 만든다. 파일명, 경로, 레이어명, 블록명, 문자, 좌표, 핸들,
예외 메시지는 찍지 않는다. 이런 내용은 입력 폴더 안의 결과 폴더에만 남는다.

도면이 많아도 붙여 넣을 수 있게 블록은 집계와 도면별 한 줄로 만든다.
전체 내용은 detail.txt에 두고 --show로 몇 장만 꺼내 본다. detail.txt도 같은 허용 항목뿐이다.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import random
import re
import statistics
import sys
import threading
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree

from ..classify import SheetDictionary, SheetGuess, classify_sheet, label_from_name
from ..config import config_dir, load_defaults, load_layer_dictionary, load_sheet_dictionary, load_thresholds
from ..errors import Cad2BimError, EmptyDrawingError, RegionAmbiguousError
from ..io import find_converter, require_converter
from ..pipeline import ConvertResult, convert
from ..preprocess import load_primitives
from ..timing import stage, take_record
from .benchmark import PRIMARY_METRICS, _git, config_digest, lookup, score_result
from .diagnostics import column_group_lines, hatch_pattern_line, layer_lines, layout_line
from .metrics import _betti, _wall_graph
from .selfcheck import SelfCheck, run_self_check
from .truth import load_truth

OUT_DIR_NAME = "_cad2bim_out"
OVERRIDES_NAME = "overrides.yaml"
PASTE_NAME = "paste.txt"
DETAIL_NAME = "detail.txt"
_SUFFIXES = (".dxf", ".dwg")
_SAFE_CODE = re.compile(r"^[A-Za-z0-9_]{1,40}$")
_SIZE_BUCKET_M = 5
_FULL_DETAIL_MAX = 5  # 이 이하면 블록에 전체 내용을 넣는다
_SAMPLE_SEED = 20261105
_HANGUL = re.compile("[가-힣]")
_GARBLED_MIN_MARKS = 2
_WAIT_STEP_S = 0.2
_COLUMN_MATCH_MM = 100.0
_COLUMN_VOTE_MAX = 500
_COLUMN_STACK_MIN = 8
_COLUMN_DUP_SHARE = 0.98
_COLUMN_DUP_COUNT = 2
_MAX_PARTS = 12
_COLUMN_MIN_MM = 200.0
_COLUMN_MAX_MM = 2000.0
_CHECK_FIELDS = (
    "rotation_recall", "rotation_precision", "masked_recall", "masked_precision",
    "hatch_explained", "wall_in_hatch", "wall_layer_coverage", "dim_on_element",
)
_STOP_GRACE_S = 5.0


class DrawingTimeout(Exception):
    """도면 하나가 시간 제한을 넘었다."""


def _code(value: Any) -> str:
    """코드처럼 생긴 값만 통과시킨다. 사용자 문자열이 코드 자리에 들어와도 새지 않는다."""
    text = str(value)
    return text if _SAFE_CODE.match(text) else "other"


def _counts(counter: Counter | dict) -> str:
    merged: Counter = Counter()
    for key, count in counter.items():
        merged[_code(key)] += count
    return " ".join(f"{k}={v}" for k, v in sorted(merged.items(), key=lambda x: str(x[0]))) or "-"


def _error_code(exc: BaseException) -> str:
    """예외 종류와 우리 코드의 위치만 쓴다. 메시지에는 경로나 레이어명이 있을 수 있다."""
    name = _code(type(exc).__name__)
    if isinstance(exc, Cad2BimError):
        detail = getattr(exc, "detail", "")
        return f"{name} {_code(detail)}" if detail else name
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if "cad2bim" in f.filename]
    if not frames:
        return name
    return f"{name} at {_code(Path(frames[-1].filename).stem)}:{frames[-1].lineno}"


def _bucket(values: list[float], step: int) -> Counter:
    return Counter(int(round(v / step) * step) for v in values)


def _graph(result: ConvertResult, thresholds: dict[str, Any]) -> tuple[int, int]:
    grid = thresholds["topology"]["snap_grid_mm"]
    nodes, edges = _wall_graph([(w.start, w.end) for w in result.ir.walls], grid)
    return _betti(len(nodes), edges) if nodes else (0, 0)


def _text_stats(result: ConvertResult) -> Counter:
    """변환 중 한글이 깨졌는지 개수로만 본다. 문자 내용은 찍지 않는다."""
    stats: Counter = Counter()
    for p in result.pre.primitives:
        if p.kind != "text":
            continue
        text = p.geometry["text"]
        stats["total"] += 1
        stats["hangul"] += bool(_HANGUL.search(text))
        stats["garbled"] += "\ufffd" in text or text.count("?") >= _GARBLED_MIN_MARKS
    return stats


def _column_evidence(result: ConvertResult) -> Counter:
    """기둥이 없는 이유를 가른다. 기둥 레이어가 있는지, 기둥 크기의 채운 사각형이 있는지."""
    stats: Counter = Counter()
    for p in result.pre.primitives:
        role = result.layers.role(p.layer)
        stats["on_column_layer"] += role == "COLUMN"
        box = p.bbox()
        if box is None:
            continue
        size = max(box[2] - box[0], box[3] - box[1])
        if not _COLUMN_MIN_MM <= size <= _COLUMN_MAX_MM:
            continue
        stats[f"{p.kind}_sized_{'block' if p.block_path else 'model'}"] += p.kind in (
            "solid", "hatch", "polyline", "circle",
        )
    return stats


def _layer_lines(result: ConvertResult) -> list[str]:
    layers = result.layers
    roles = Counter(i.role for i in layers.layers.values())
    prims = Counter(layers.role(p.layer) for p in result.pre.primitives)
    return [
        f"layers: total={len(layers.layers)} has_walls={layers.has_walls} {_counts(roles)}",
        f"prims_by_role: {_counts(prims)}",
    ]


def _element_lines(result: ConvertResult, thresholds: dict[str, Any]) -> list[str]:
    ir = result.ir
    walls = ir.walls
    components, loops = _graph(result, thresholds)
    lengths = [((w.end[0] - w.start[0]) ** 2 + (w.end[1] - w.start[1]) ** 2) ** 0.5 for w in walls]
    return [
        f"walls: n={len(walls)} high={sum(w.confidence.level == 'high' for w in walls)} "
        f"pieces={len(result.recognition.raw_walls)} "
        f"median_len_m={round(statistics.median(lengths) / 1000, 1) if lengths else '-'}",
        f"wall_thickness_mm: {_counts(_bucket([w.thickness for w in walls], 10))}",
        f"wall_kind: {_counts(Counter(w.kind for w in walls))}",
        f"wall_reasons: {_counts(Counter(r for w in walls for r in w.confidence.reasons))}",
        f"wall_graph: components={components} loops={loops}",
        f"columns: n={len(ir.columns)} high={sum(c.confidence.level == 'high' for c in ir.columns)} "
        f"{_counts(Counter(c.shape for c in ir.columns))}",
        f"column_size_mm: {_counts(_bucket([max(c.size) for c in ir.columns], 100))}",
        f"column_evidence: {_counts(+_column_evidence(result))}",
        f"openings: {_counts(Counter(o.type for o in ir.openings))} "
        f"high={sum(o.confidence.level == 'high' for o in ir.openings)} "
        f"reasons: {_counts(Counter(r for o in ir.openings for r in o.confidence.reasons))}",
        f"door_width_mm: {_counts(_bucket([o.width for o in ir.openings if o.type == 'door'], 100))}",
        f"door_arcs: {_counts(result.recognition.door_stats)}",
        f"rooms: n={len(ir.rooms)} high={sum(r.confidence.level == 'high' for r in ir.rooms)} "
        f"named={sum(r.names != ['미지정'] for r in ir.rooms)} "
        f"reasons: {_counts(Counter(c for r in ir.rooms for c in r.confidence.reasons))}",
        f"unsupported: {_counts(Counter(u.kind for u in ir.unsupported))}",
    ]


def _hatch_line(result: ConvertResult, patterns: dict[str, list[str]]) -> str:
    known = {name.upper() for names in patterns.values() for name in names}
    hatches = [p for p in result.pre.primitives if p.kind == "hatch"]
    recognized = sum(p.geometry["pattern"].upper() in known for p in hatches)
    return f"hatches: n={len(hatches)} known_pattern={recognized}"


def _report_ok(
    result: ConvertResult, thresholds: dict[str, Any], patterns: dict, truth: Path, check: SelfCheck,
    diagnostics: list[str],
) -> list[str]:
    pre = result.pre
    width = height = "-"
    if pre.plan_bbox:
        x0, y0, x1, y1 = pre.plan_bbox
        width = round((x1 - x0) / 1000 / _SIZE_BUCKET_M) * _SIZE_BUCKET_M
        height = round((y1 - y0) / 1000 / _SIZE_BUCKET_M) * _SIZE_BUCKET_M
    stats = pre.explode_stats
    lines = [
        f"scale: factor={pre.scale.factor:g} source={_code(pre.scale.source)} samples={pre.scale.samples}",
        f"warnings: {' '.join(sorted(_code(w) for w in result.ir.warnings)) or '-'}",
        f"plan: approx_m={width}x{height} frames={len(pre.frames)} excluded={len(pre.excluded)} "
        f"far_origin={bool(pre.origin_offset[0] or pre.origin_offset[1])}",
        f"prims: {_counts(Counter(p.kind for p in pre.primitives))}",
        f"in_blocks: {sum(bool(p.block_path) for p in pre.primitives)} "
        f"max_depth={max((len(p.block_path) for p in pre.primitives), default=0)}",
        f"normalize: merged={pre.normalize_stats.merged_away} short={pre.normalize_stats.removed_short}",
        f"skipped: hidden={stats.skipped_hidden} block_fail={len(stats.transform_failures)} "
        f"types: {_counts(stats.unsupported)}",
        *_layer_lines(result),
        _hatch_line(result, patterns),
        f"text: {_counts(_text_stats(result))}",
        *_element_lines(result, thresholds),
        check.line(),
        *diagnostics,
    ]
    if truth.is_file():
        metrics = score_result(result, load_truth(truth), thresholds)
        values = []
        for path, _ in PRIMARY_METRICS:
            value = lookup(metrics, path)
            values.append(f"{path}={'-' if value is None else f'{value:.3f}'}")
        lines.append("metrics: " + " ".join(values))
    else:
        lines.append("metrics: no_truth")
    return lines


def _short(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _row(drawing_id: str, result: ConvertResult, thresholds: dict[str, Any], check: SelfCheck) -> str:
    """도면 한 장을 한 줄로. 이상한 도면을 고른 뒤 --show로 자세히 본다."""
    ir = result.ir
    components, loops = _graph(result, thresholds)
    high = sum(w.confidence.level == "high" for w in ir.walls)
    return (
        f"{drawing_id} f={result.pre.scale.factor:g} src={_code(result.pre.scale.source)} "
        f"lay={'y' if result.layers.has_walls else 'n'} p={len(result.pre.primitives)} "
        f"w={len(ir.walls)} hi={high} c={len(ir.columns)} comp={components} loop={loops} "
        f"t={sum(p.kind == 'text' for p in result.pre.primitives)} "
        f"bl={sum(bool(p.block_path) for p in result.pre.primitives)} "
        f"rot={_short(check.rotation_recall)} mask={_short(check.masked_recall)} "
        f"wlc={_short(check.wall_layer_coverage)} hx={_short(check.hatch_explained)} "
        f"dim={_short(check.dim_on_element)} cl={_column_evidence(result)['on_column_layer']} {check.guess} "
        f"door={sum(o.type == 'door' for o in ir.openings)} win={sum(o.type == 'window' for o in ir.openings)} "
        f"gap={sum(o.type == 'unknown' for o in ir.openings)} room={len(ir.rooms)} "
        f"named={sum(r.names != ['미지정'] for r in ir.rooms)}"
    )


@dataclass
class _Totals:
    failures: dict[str, list[str]] = field(default_factory=dict)
    rows: list[str] = field(default_factory=list)
    wall_counts: list[int] = field(default_factory=list)
    high: int = 0
    split_files: int = 0
    sheet_text: Counter = field(default_factory=Counter)
    sheet_shape: Counter = field(default_factory=Counter)
    sheet_kinds: Counter = field(default_factory=Counter)
    sheet_unlabeled: int = 0
    holdout_text: Counter = field(default_factory=Counter)
    holdout_shape: Counter = field(default_factory=Counter)
    sheet_features: dict[str, list[tuple[float, ...]]] = field(default_factory=dict)
    layout_counts: Counter = field(default_factory=Counter)
    column_sets: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    no_walls: list[str] = field(default_factory=list)
    no_wall_layer: int = 0
    scale_sources: Counter = field(default_factory=Counter)
    warnings: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    unsupported: Counter = field(default_factory=Counter)
    text: Counter = field(default_factory=Counter)
    stages: Counter = field(default_factory=Counter)
    guesses: Counter = field(default_factory=Counter)
    door_arcs: Counter = field(default_factory=Counter)
    element_counts: Counter = field(default_factory=Counter)
    checks: dict[str, list[float]] = field(default_factory=dict)
    seconds: list[float] = field(default_factory=list)

    def merge(self, other: "_Totals") -> None:
        for code, ids in other.failures.items():
            self.failures.setdefault(code, []).extend(ids)
        self.rows += other.rows
        self.wall_counts += other.wall_counts
        self.high += other.high
        self.split_files += other.split_files
        self.sheet_text.update(other.sheet_text)
        self.sheet_shape.update(other.sheet_shape)
        self.sheet_kinds.update(other.sheet_kinds)
        self.sheet_unlabeled += other.sheet_unlabeled
        self.holdout_text.update(other.holdout_text)
        self.holdout_shape.update(other.holdout_shape)
        self.layout_counts.update(other.layout_counts)
        self.door_arcs.update(other.door_arcs)
        self.element_counts.update(other.element_counts)
        self.column_sets.update(other.column_sets)
        for label, rows in other.sheet_features.items():
            self.sheet_features.setdefault(label, []).extend(rows)
        self.no_walls += other.no_walls
        self.no_wall_layer += other.no_wall_layer
        for name in ("scale_sources", "warnings", "reasons", "unsupported", "text", "stages", "guesses"):
            getattr(self, name).update(getattr(other, name))
        for name, values in other.checks.items():
            self.checks.setdefault(name, []).extend(values)
        self.seconds += other.seconds

    def add_layout(self, model: list[Any], stats: Any) -> None:
        self.layout_counts["with_layout"] += stats.layouts > 0
        self.layout_counts["with_paper_text"] += any(p.kind == "text" for p in stats.paper)
        self.layout_counts["with_viewport"] += bool(stats.viewports)
        self.layout_counts["with_xclip"] += stats.xclips > 0
        self.layout_counts["clip_applied"] += stats.clip_applied
        self.layout_counts["clip_inverted"] += stats.clip_inverted
        self.layout_counts["clip_failed"] += stats.clip_failed
        self.layout_counts["clip_removed"] += stats.clip_removed
        self.layout_counts["empty_model"] += not model

    def add_sheet(self, drawing_id: str, guess: SheetGuess, label: str | None) -> None:
        self.sheet_kinds[guess.kind] += 1
        if label is None:
            self.sheet_unlabeled += 1
            return
        if _is_holdout(drawing_id):
            self.holdout_text[f"{label}>{guess.by_text}"] += 1
            self.holdout_shape[f"{label}>{guess.by_shape}"] += 1
            return
        self.sheet_text[f"{label}>{guess.by_text}"] += 1
        self.sheet_shape[f"{label}>{guess.by_shape}"] += 1
        self.sheet_features.setdefault(label, []).append(
            (guess.level_marks, guess.door_arcs, guess.grid_axes, guess.curve_share, guess.text_share)
        )

    def add(
        self, drawing_id: str, result: ConvertResult, thresholds: dict[str, Any], check: SelfCheck
    ) -> None:
        ir = result.ir
        self.rows.append(_row(drawing_id, result, thresholds, check))
        ox, oy = result.ir.origin_offset
        self.column_sets[drawing_id] = [(c.center[0] + ox, c.center[1] + oy) for c in ir.columns]
        self.guesses[check.guess] += 1
        for name in _CHECK_FIELDS:
            value = getattr(check, name)
            if value is not None:
                self.checks.setdefault(name, []).append(value)
        self.wall_counts.append(len(ir.walls))
        self.high += sum(w.confidence.level == "high" for w in ir.walls)
        if not ir.walls:
            self.no_walls.append(drawing_id)
        self.no_wall_layer += not result.layers.has_walls
        self.scale_sources[_code(result.pre.scale.source)] += 1
        self.warnings.update(_code(w) for w in ir.warnings)
        self.reasons.update(_code(r) for w in ir.walls for r in w.confidence.reasons)
        self.unsupported.update(result.pre.explode_stats.unsupported)
        self.door_arcs.update(result.recognition.door_stats)
        self.element_counts.update(Counter(o.type for o in ir.openings))
        self.element_counts["room"] += len(ir.rooms)
        self.element_counts["room_named"] += sum(r.names != ["미지정"] for r in ir.rooms)
        self.element_counts.update(Counter(u.kind for u in ir.unsupported))
        self.text.update(_text_stats(result))


def _id_ranges(ids: list[str]) -> str:
    """D01,D02,D03,D07 → D01-D03,D07"""
    ids = sorted({i.split(".")[0] for i in ids})  # D03.1, D03.2는 파일 D03 하나로 적는다
    numbers = [int(i[1:]) for i in ids]
    width = max(len(i) - 1 for i in ids) if ids else 2
    parts: list[str] = []
    start = previous = None
    for n in numbers + [None]:
        if previous is not None and n == previous + 1:
            previous = n
            continue
        if previous is not None:
            first, last = f"D{start:0{width}d}", f"D{previous:0{width}d}"
            parts.append(first if start == previous else f"{first}-{last}")
        start = previous = n
    return ",".join(parts)


def _seconds(stages: dict[str, float]) -> str:
    """느린 단계부터."""
    ordered = sorted(stages.items(), key=lambda x: -x[1])
    return " ".join(f"{_code(k)}={v:.1f}" for k, v in ordered) or "-"


def _is_holdout(drawing_id: str) -> bool:
    """짝수 번호는 시험용이다. 기준을 조정할 때 보지 않아야 수치가 부풀지 않는다."""
    return int(drawing_id[1:].split(".")[0]) % 2 == 0


def _feature_lines(totals: _Totals) -> list[str]:
    lines = []
    for label, rows in sorted(totals.sheet_features.items()):
        columns = list(zip(*rows))
        def quartiles(values: tuple[float, ...]) -> str:
            ordered = sorted(values)
            return "/".join(f"{ordered[int(q * (len(ordered) - 1))]:g}" for q in (0.25, 0.5, 0.75))

        lines.append(
            f"sheet_features[{_code(label)}] n={len(rows)} levels={quartiles(columns[0])} arcs={quartiles(columns[1])} "
            f"axes={quartiles(columns[2])} curve={quartiles(columns[3])} text_share={quartiles(columns[4])}"
        )
    return lines


def _accuracy(pairs: Counter) -> str:
    total = sum(pairs.values())
    if not total:
        return "-"
    correct = sum(n for key, n in pairs.items() if key.split(">")[0] == key.split(">")[1])
    return f"{correct / total:.3f}"


def _pairs(pairs: Counter) -> str:
    """정답>판단=개수. 종류 이름은 정해진 코드뿐이다. _code는 >를 막으므로 직접 만든다."""
    return " ".join(f"{k}={v}" for k, v in sorted(pairs.items())) or "-"


def _sheet_lines(totals: _Totals) -> list[str]:
    labeled = sum(totals.sheet_text.values())
    return [
        f"layout_summary: {_counts(+totals.layout_counts)}",
        f"sheet_kinds: {_counts(totals.sheet_kinds)}",
        f"sheet_accuracy: labeled={labeled} text={_accuracy(totals.sheet_text)} "
        f"shape={_accuracy(totals.sheet_shape)} holdout_labeled={sum(totals.holdout_text.values())} "
        f"holdout_text={_accuracy(totals.holdout_text)} holdout_shape={_accuracy(totals.holdout_shape)} "
        f"unlabeled={totals.sheet_unlabeled}",
        f"sheet_confusion_text: {_pairs(totals.sheet_text)}",
        f"sheet_confusion_shape: {_pairs(totals.sheet_shape)}",
        *_feature_lines(totals),
    ]


def column_shift(a: np.ndarray, b: np.ndarray) -> tuple[tuple[float, float], float]:
    """a를 b에 맞추는 이동량과 맞은 기둥 비율.

    실제 도면은 파일마다 원점이 달라 좌표를 그대로 비교할 수 없었다. 두 도면의 기둥 쌍이
    가장 많이 가리키는 이동량을 원점 차이로 본다.
    """
    swapped = len(a) > len(b)
    small, large = (b, a) if swapped else (a, b)
    if len(small) == 0 or len(large) > _COLUMN_VOTE_MAX:
        return (0.0, 0.0), 0.0
    shifts = (large[None, :, :] - small[:, None, :]).reshape(-1, 2)
    bins = np.round(shifts / _COLUMN_MATCH_MM).astype(np.int64)
    _, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    shift = shifts[inverse.ravel() == int(np.argmax(counts))].mean(axis=0)
    distances, _ = cKDTree(large).query(small + shift)
    share = float((distances <= _COLUMN_MATCH_MM).mean())
    return ((-float(shift[0]), -float(shift[1])) if swapped else (float(shift[0]), float(shift[1]))), share


def _column_match(a: np.ndarray, b: np.ndarray) -> float:
    """두 도면의 기둥이 원점 차이를 뺀 뒤 얼마나 겹치는가."""
    return column_shift(a, b)[1]


def column_stack_line(sets: dict[str, list[tuple[float, float]]]) -> str:
    """도면끼리 기둥 배치가 같은지 본다. 위아래 층의 기둥은 대체로 이어진다.

    같은 층을 다시 그린 도면끼리는 당연히 일치하므로 거의 같은 쌍은 따로 센다.
    규칙적인 격자끼리는 원점을 맞추면 잘 겹치므로 약한 근거다. 기둥이 맞다는 증명이 아니다.
    """
    usable = {k: np.array(v) for k, v in sets.items() if len(v) >= _COLUMN_STACK_MIN}
    if len(usable) < 2:
        return f"column_stack: drawings={len(usable)} pairs=0"
    ids = sorted(usable)
    best: dict[str, float] = {}
    duplicates = pairs = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            match = _column_match(usable[a], usable[b])
            pairs += 1
            same_count = abs(len(usable[a]) - len(usable[b])) <= _COLUMN_DUP_COUNT
            if match >= _COLUMN_DUP_SHARE and same_count:
                duplicates += 1
                continue
            best[a] = max(best.get(a, 0.0), match)
            best[b] = max(best.get(b, 0.0), match)
    values = sorted(best.values())
    median = statistics.median(values) if values else 0.0
    return (
        f"column_stack: drawings={len(usable)} pairs={pairs} duplicate_pairs={duplicates} "
        f"best_match_median={median:.3f} best_ge_50={sum(v >= 0.5 for v in values)} of={len(values)}"
    )


def plan_groups_line(sets: dict[str, list[tuple[float, float]]]) -> str:
    """기둥 배치가 거의 같은 도면을 같은 평면의 다른 버전으로 묶는다. 층 수는 이 개수로 본다.

    평균을 낼 때 같은 층을 여러 번 세면 안 된다. 기둥이 적은 도면은 묶지 못해 따로 센다.
    """
    usable = {k: np.array(v) for k, v in sets.items() if len(v) >= _COLUMN_STACK_MIN}
    parent = {k: k for k in usable}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    ids = sorted(usable)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            same_count = abs(len(usable[a]) - len(usable[b])) <= _COLUMN_DUP_COUNT
            if same_count and _column_match(usable[a], usable[b]) >= _COLUMN_DUP_SHARE:
                parent[find(a)] = find(b)
    groups: dict[str, list[str]] = {}
    for key in ids:
        groups.setdefault(find(key), []).append(key)
    listed = [g for g in groups.values() if len(g) > 1]
    text = " ".join("+".join(g) for g in sorted(listed))
    return f"plan_groups: unique={len(groups)} grouped={sum(len(g) for g in listed)} {text or '-'}"


def _summary_lines(totals: _Totals) -> list[str]:
    lines = _sheet_lines(totals)
    if totals.column_sets:
        lines.append(column_stack_line(totals.column_sets))
        lines.append(plan_groups_line(totals.column_sets))
    for code, ids in sorted(totals.failures.items(), key=lambda x: -len(x[1])):
        lines.append(f"failures: {code} n={len(ids)} ids={_id_ranges(ids)}")
    if totals.wall_counts:
        counts = totals.wall_counts
        lines += [
            f"walls_per_drawing: min={min(counts)} median={statistics.median(counts):g} max={max(counts)} "
            f"total={sum(counts)} high={totals.high}",
            f"no_walls: n={len(totals.no_walls)} ids={_id_ranges(totals.no_walls) or '-'}",
            f"no_wall_layer: {totals.no_wall_layer}",
            f"split_files: {totals.split_files} plans={len(totals.rows)}",
            f"scale_source: {_counts(totals.scale_sources)}",
            f"warnings: {_counts(totals.warnings)}",
            f"wall_reasons: {_counts(totals.reasons)}",
            f"unsupported_types: {_counts(totals.unsupported)}",
            f"text: {_counts(totals.text)}",
            f"guess: {_counts(totals.guesses)}",
            f"elements: {_counts(totals.element_counts)}",
            f"door_arcs: {_counts(totals.door_arcs)}",
            "selfcheck_median: " + " ".join(
                f"{name}={statistics.median(values):.3f}(n={len(values)})"
                for name, values in totals.checks.items()
            ),
        ]
    if totals.stages:
        lines.append("seconds_by_stage: " + _seconds(totals.stages))
    if totals.seconds:
        lines.append(f"seconds_per_drawing: median={statistics.median(totals.seconds):.1f} "
                     f"max={max(totals.seconds):.1f}")
    return lines


def _load_overrides(folder: Path) -> dict[str, dict[str, Any]]:
    path = folder / OVERRIDES_NAME
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def find_drawings(folder: Path) -> list[Path]:
    found = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUFFIXES and OUT_DIR_NAME not in p.parts
    ]
    return sorted(found, key=lambda p: str(p.relative_to(folder)).casefold())


def _check_outside_project(folder: Path) -> None:
    """프로젝트 안의 파일은 LLM 도구가 읽을 수 있다. 실제 도면은 밖에 있어야 한다."""
    project = config_dir().resolve().parent
    # data는 공개 도면과 AI 인식이 승인된 도면만 두는 곳이다.
    if folder.is_relative_to(project) and not folder.is_relative_to(project / "data"):
        raise Cad2BimError("실제 도면 폴더가 프로젝트 안에 있습니다. 프로젝트 밖으로 옮긴 뒤 실행하세요.")


def _as_dxf(path: Path, out_dir: Path) -> Path:
    """DWG는 내용 해시로 변환 결과를 보관한다. 재실행할 때마다 다시 변환하지 않기 위해서다."""
    if path.suffix.lower() != ".dwg":
        return path
    converter = require_converter()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    cache = out_dir / "_dxf" / converter.name / digest
    cached = sorted(cache.glob("*.dxf")) if cache.is_dir() else []
    if cached:
        return cached[0]
    with stage("dwg_convert"):
        return converter.to_dxf(path, cache)


def _audit_plan(
    dxf: Path, part_id: str, out_dir: Path, cfg: dict, thresholds: dict, patterns: dict, height: float,
    region: Any, unit_scale: float | None, truth: Path, quick: bool, totals: _Totals,
) -> list[str]:
    """평면 하나를 변환하고 숫자 줄을 돌려준다. 실패는 호출한 쪽에서 처리한다."""
    result = convert(
        dxf, out_dir / part_id, cfg, height, tuple(region) if region else None, unit_scale,
        cache_dir=out_dir / "_prims",
    )
    with stage("selfcheck"):
        check = (
            SelfCheck() if quick
            else run_self_check(result, cfg, thresholds, patterns, _graph(result, thresholds)[1])
        )
        diagnostics = [
            layout_line(result.pre.primitives),
            hatch_pattern_line(result.pre.primitives),
            *column_group_lines(result.pre.primitives, result.layers),
        ]
        if not quick:
            diagnostics += layer_lines(result.pre.primitives, result.layers, cfg)
    lines = _report_ok(result, thresholds, patterns, truth, check, diagnostics)
    totals.add(part_id, result, thresholds, check)
    return [f"[{part_id}] ok", *lines]


def _audit_one(
    path: Path, drawing_id: str, out_dir: Path, cfg: dict, thresholds: dict, patterns: dict,
    height: float, override: dict[str, Any], totals: _Totals, quick: bool = False,
    classify_only: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    started = time.perf_counter()
    floor = override.get("floor_height", height)
    unit_scale = override.get("unit_scale")
    truth = path.with_suffix(".truth.json")
    sheet_line: list[str] = []
    try:
        dxf = _as_dxf(path, out_dir)
        with stage("classify"):
            # 파일명은 채점 정답으로만 쓴다. 분류기에는 내용만 넘긴다.
            dictionary = SheetDictionary(load_sheet_dictionary())
            model, explode_stats = load_primitives(dxf, cfg, out_dir / "_prims")
            guess = classify_sheet(model, dictionary, explode_stats.paper)
            totals.add_layout(model, explode_stats)
            totals.add_sheet(drawing_id, guess, label_from_name(path.name, dictionary))
        sheet_line = [_sheet_line(guess), _layout_line(model, explode_stats, path, dxf)]
        if classify_only:
            if not _is_holdout(drawing_id):  # 시험용 도면의 개별 결과는 보지 않는다
                totals.rows.append(
                    f"{drawing_id} {_code(guess.kind)} text={_code(guess.by_text)} shape={_code(guess.by_shape)} "
                    f"label={_code(label_from_name(path.name, dictionary) or 'none')} "
                    f"lv={guess.level_marks} arc={guess.door_arcs} ax={guess.grid_axes} "
                    f"cv={guess.curve_share} ts={guess.text_share} "
                    f"mp={len(model)} pp={len(explode_stats.paper)} vp={len(explode_stats.viewports)} "
                    f"xc={explode_stats.xclips} rm={explode_stats.clip_removed}"
                )
            totals.seconds.append(time.perf_counter() - started)
            return [f"[{drawing_id}] classified", *sheet_line], {"status": "ok", "error": None}
        if not model and not explode_stats.paper:
            raise EmptyDrawingError("배치된 요소가 없습니다. 변환기가 내용을 잃었을 수 있습니다.")
        try:
            lines = _audit_plan(
                dxf, drawing_id, out_dir, cfg, thresholds, patterns, floor, override.get("region"),
                unit_scale, truth, quick, totals,
            )
        except RegionAmbiguousError as exc:
            lines = _audit_parts(
                dxf, drawing_id, exc, out_dir, cfg, thresholds, patterns, floor, unit_scale, quick, totals
            )
    except Exception as exc:  # 어떤 실패든 다음 도면으로 넘어가고, 내용은 찍지 않는다
        code = _error_code(exc)
        totals.failures.setdefault(code, []).append(drawing_id)
        return (
            [f"[{drawing_id}] failed {code}", *sheet_line],
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
    totals.seconds.append(time.perf_counter() - started)
    return [lines[0], *sheet_line, *lines[1:]], {
        "status": "ok", "error": None, "kind": guess.kind, "by_shape": guess.by_shape,
    }


def _layout_line(model: list[Any], stats: Any, path: Path, dxf: Path) -> str:
    paper_text = sum(p.kind == "text" for p in stats.paper)
    return (
        f"layout_info: model_prims={len(model)} layouts={stats.layouts} paper_prims={len(stats.paper)} "
        f"paper_text={paper_text} viewports={len(stats.viewports)} xclips={stats.xclips} "
        f"clip_applied={stats.clip_applied} clip_inverted={stats.clip_inverted} clip_failed={stats.clip_failed} "
        f"clip_removed={stats.clip_removed} source_kb={path.stat().st_size // 1024} dxf_kb={dxf.stat().st_size // 1024} "
        f"model_entities={stats.model_entities} hidden={stats.skipped_hidden} "
        f"model_types: {_counts(Counter(dict(stats.model_types.most_common(8))))}"
    )


def _sheet_line(guess: SheetGuess) -> str:
    return (
        f"sheet: kind={_code(guess.kind)} text={_code(guess.by_text)} shape={_code(guess.by_shape)} "
        f"level_marks={guess.level_marks} door_arcs={guess.door_arcs} grid_axes={guess.grid_axes} "
        f"curve={guess.curve_share} text_share={guess.text_share}"
    )


def _audit_parts(
    dxf: Path, drawing_id: str, exc: RegionAmbiguousError, out_dir: Path, cfg: dict, thresholds: dict,
    patterns: dict, floor: float, unit_scale: float | None, quick: bool, totals: _Totals,
) -> list[str]:
    """한 파일에 평면이 여럿이면 실패로 두지 않고 큰 것부터 각각 처리한다. 합성해 보고하지 않는다."""
    def area(box: tuple[float, float, float, float]) -> float:
        return (box[2] - box[0]) * (box[3] - box[1])

    candidates = sorted(exc.candidates, key=area, reverse=True)
    lines = [f"[{drawing_id}] split parts={len(candidates)} processed={min(len(candidates), _MAX_PARTS)}"]
    no_truth = dxf.with_suffix(".no_truth")  # 정답은 파일 단위라 평면별로 대응할 수 없다
    done = 0
    for index, box in enumerate(candidates[:_MAX_PARTS], start=1):
        part_id = f"{drawing_id}.{index}"
        region = tuple(v / exc.scale_factor for v in box)
        try:
            lines += _audit_plan(
                dxf, part_id, out_dir, cfg, thresholds, patterns, floor, region, unit_scale, no_truth, quick, totals
            )
            done += 1
        except Exception as part_error:
            code = _error_code(part_error)
            totals.failures.setdefault(code, []).append(drawing_id)
            lines.append(f"[{part_id}] failed {code}")
    if not done:
        raise exc
    totals.split_files += 1
    return lines


def _run_limited(work: Any, time_limit: float) -> tuple[Any, dict[str, float], str | None]:
    """제한 시간 안에 work를 돌린다. (결과, 단계별 시간, 초과 시 멈춘 단계).

    파이썬 코드는 스레드에 예외를 넣어 멈춘다. 외부 변환기를 기다리는 중이면 바로 멈추지
    않으므로 잠시 기다린 뒤 다음 도면으로 넘어간다. 변환기에는 자체 제한이 있다.
    """
    outcome: list[Any] = []
    worker = threading.Thread(target=lambda: outcome.append(work()), daemon=True)
    worker.start()
    deadline = time.monotonic() + time_limit
    # 짧게 나눠 기다린다. 긴 대기 중에는 Windows에서 Ctrl+C가 먹지 않는다.
    while worker.is_alive() and time.monotonic() < deadline:
        worker.join(_WAIT_STEP_S)
    stuck_at = None
    if worker.is_alive():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(worker.ident), ctypes.py_object(DrawingTimeout)
        )
        worker.join(_STOP_GRACE_S)
    record = take_record(worker.ident)
    if worker.is_alive() or not outcome:
        stuck_at = record.current or "start"
    return (outcome[0] if outcome else None), record.seconds, stuck_at


@dataclass
class _Job:
    drawing_id: str
    path: Path
    relative: str
    out_dir: Path
    height: float
    override: dict[str, Any]
    time_limit: float
    quick: bool = False
    classify_only: bool = False


def _quiet() -> None:
    """라이브러리 로그와 경고를 끈다. 도면 내용이 섞일 수 있다. 작업 프로세스에도 적용한다."""
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")


def _process(job: _Job) -> tuple[str, str, list[str], dict[str, Any], _Totals]:
    cfg, thresholds = load_defaults(), load_thresholds()
    patterns = load_layer_dictionary()["hatch_patterns"]
    totals = _Totals()

    def work() -> tuple[list[str], dict[str, Any]]:
        return _audit_one(
            job.path, job.drawing_id, job.out_dir, cfg, thresholds, patterns, job.height, job.override, totals,
            job.quick, job.classify_only,
        )

    outcome, stages, stuck_at = _run_limited(work, job.time_limit)
    if stuck_at is not None and outcome is None:
        code = f"DrawingTimeout stage={_code(stuck_at)}"
        totals.failures.setdefault(code, []).append(job.drawing_id)
        outcome = ([f"[{job.drawing_id}] failed {code}"], {"status": "failed", "error": code})
    lines, local = outcome
    totals.stages.update(stages)
    return job.drawing_id, job.relative, [*lines, f"seconds: {_seconds(stages)}"], local, totals


def _worker_failure(job: _Job, exc: BaseException) -> tuple[str, str, list[str], dict[str, Any], _Totals]:
    """작업 프로세스 자체가 죽은 경우 (메모리 부족 등)."""
    code = _error_code(exc)
    totals = _Totals()
    totals.failures[code] = [job.drawing_id]
    return (job.drawing_id, job.relative, [f"[{job.drawing_id}] failed {code}"],
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, totals)


def _progress(done: int, total: int) -> None:
    print(f"진행 {done}/{total}", end="\r", file=sys.stderr, flush=True)


def _run_jobs(jobs_list: list[_Job], jobs: int) -> tuple[list[Any], bool]:
    done: list[Any] = []
    if jobs <= 1:
        for index, job in enumerate(jobs_list, start=1):
            _progress(index, len(jobs_list))
            try:
                done.append(_process(job))
            except KeyboardInterrupt:  # 트레이스백을 찍지 않고 여기까지의 결과를 낸다
                return done, True
        return done, False

    pool = ProcessPoolExecutor(max_workers=jobs, initializer=_quiet)
    futures = {pool.submit(_process, job): job for job in jobs_list}
    try:
        for future in as_completed(futures):
            try:
                done.append(future.result())
            except Exception as exc:
                done.append(_worker_failure(futures[future], exc))
            _progress(len(done), len(jobs_list))
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        return done, True
    pool.shutdown()
    return done, False


def _select(
    drawings: list[Path], limit: int | None, match: str | None = None
) -> list[tuple[str, Path]]:
    """번호는 항상 전체 목록 기준이다. 표본을 돌려도 같은 도면은 같은 번호를 가진다."""
    width = max(2, len(str(len(drawings))))
    numbered = [(f"D{i:0{width}d}", p) for i, p in enumerate(drawings, start=1)]
    if match:
        numbered = [(i, p) for i, p in numbered if match.casefold() in p.name.casefold()]
    if limit is None or limit >= len(numbered):
        return numbered
    chosen = random.Random(_SAMPLE_SEED).sample(numbered, limit)
    return sorted(chosen, key=lambda item: item[0])


def _run(
    folder: Path, height: float, limit: int | None, time_limit: float, jobs: int, quick: bool, match: str | None,
    classify_only: bool,
) -> str:
    folder = folder.resolve()
    if not folder.is_dir():
        raise Cad2BimError("입력 폴더를 찾을 수 없습니다.")
    _check_outside_project(folder)
    drawings = find_drawings(folder)
    if not drawings:
        raise Cad2BimError("폴더에 DXF나 DWG가 없습니다.")

    overrides = _load_overrides(folder)
    out_dir = folder / OUT_DIR_NAME
    out_dir.mkdir(exist_ok=True)

    selected = _select(drawings, limit, match)
    if not selected:
        raise Cad2BimError("--match에 맞는 도면이 없습니다.")
    jobs_list = []
    for drawing_id, path in selected:
        relative = str(path.relative_to(folder))
        override = overrides.get(path.name) or overrides.get(relative) or {}
        jobs_list.append(
            _Job(drawing_id, path, relative, out_dir, height, override, time_limit, quick, classify_only)
        )

    started = time.perf_counter()
    done, interrupted = _run_jobs(jobs_list, jobs)
    print(file=sys.stderr)

    totals = _Totals()
    sections: list[list[str]] = []
    mapping: dict[str, Any] = {}
    for drawing_id, relative, lines, local, part in sorted(done, key=lambda d: d[0]):
        totals.merge(part)
        sections.append(lines)
        mapping[drawing_id] = {"file": relative, **local}

    failed = sum(len(ids) for ids in totals.failures.values())
    commit = (_git(["rev-parse", "HEAD"]) or "none")[:10]
    converter = find_converter()
    header = (
        f"CAD2BIM-AUDIT v2 commit={commit} config={config_digest()[:10]} "
        f"drawings={len(sections)} failed={failed} of_total={len(drawings)} "
        f"interrupted={interrupted} "
        f"elapsed_s={time.perf_counter() - started:.0f} "
        f"converter={converter.name if converter else 'none'} jobs={jobs}"
    )
    details = "\n\n".join("\n".join(s) for s in sections) + "\n"
    (out_dir / DETAIL_NAME).write_text(details, encoding="utf-8")
    (out_dir / "map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    body = _summary_lines(totals)
    if len(sections) <= _FULL_DETAIL_MAX:
        body += ["", details.rstrip()]
    elif totals.rows and classify_only:
        body += ["", "rows: id kind text shape label lv arc ax cv ts mp=model_prims pp=paper_prims vp=viewports xc=xclips rm=clip_removed", *totals.rows]
    elif totals.rows:
        body += ["", "rows: id f=scale src lay=wall_layer p=prims w=walls hi=high c=columns comp loop t=text bl=in_blocks rot=rotation_recall mask=masked_recall wlc=wall_layer_cov hx=hatch_explained dim=dim_on_element cl=on_column_layer guess door win gap=unknown_opening room named", *totals.rows]
    block = "\n".join([header, *body]) + "\n"
    (out_dir / PASTE_NAME).write_text(block, encoding="utf-8")
    return block


def run_audit(
    folder: Path, height: float, limit: int | None = None, time_limit: float = 180.0, jobs: int = 1,
    quick: bool = False, match: str | None = None, classify_only: bool = False,
) -> str:
    """붙여 넣을 블록을 돌려준다. 실행 중 라이브러리 로그와 경고는 끈다. 내용이 섞일 수 있다."""
    if limit is not None and limit < 1:
        raise Cad2BimError("--limit은 1 이상이어야 합니다.")
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _run(Path(folder), height, limit, time_limit, jobs, quick, match, classify_only)
    finally:
        logging.disable(logging.NOTSET)


def show_details(folder: Path, ids: list[str]) -> str:
    """지난 실행의 detail.txt에서 고른 도면만 꺼낸다. 다시 변환하지 않는다."""
    path = Path(folder).resolve() / OUT_DIR_NAME / DETAIL_NAME
    if not path.is_file():
        raise Cad2BimError("지난 점검 결과가 없습니다. 먼저 audit을 실행하세요.")
    sections = path.read_text(encoding="utf-8").strip().split("\n\n")
    by_id = {s.split("]")[0].lstrip("["): s for s in sections}
    wanted = [_code(i.strip()) for i in ids]
    return "\n\n".join(by_id.get(i, f"[{i}] not_found") for i in wanted) + "\n"
