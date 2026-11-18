"""IR과 정답 비교 (상세설계 8.2, 8.4). 도면 좌표가 정합되어 있다고 가정한다.

네 층으로 본다.
- 요소: 벽·기둥·문·창·실의 재현율·정밀도·F1. 실은 IoU 매칭과 PQ도 낸다.
- 기하: 버퍼별 완전성·정확성, 중앙 거리 오차, 방향, 접합점 일치율.
- 위상: 벽 그래프의 베티 수 오차.
- 확신도: calibration.py.

벽은 개수가 아니라 길이로 비교한다. 교차부를 관통하는 정답 벽 하나를 T자 접합에서
둘로 나눠 만들어도 틀린 것이 아니기 때문이다.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.geometry import Point as ShapelyPoint
from shapely.ops import unary_union

from ..ir import IR, Opening, Wall
from .calibration import confidence_metrics
from .truth import Point, Truth, TruthWall


@dataclass(frozen=True)
class _Segment:
    a: np.ndarray
    b: np.ndarray

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    @property
    def direction(self) -> np.ndarray:
        return (self.b - self.a) / max(self.length, 1e-9)

    def samples(self, step: float) -> np.ndarray:
        n = max(int(self.length // step), 1)
        t = (np.arange(n) + 0.5) / n
        return self.a + np.outer(t, self.b - self.a)

    def distance(self, points: np.ndarray) -> np.ndarray:
        d = self.b - self.a
        t = np.clip((points - self.a) @ d / max(float(d @ d), 1e-9), 0, 1)
        return np.linalg.norm(points - (self.a + np.outer(t, d)), axis=1)


def _seg(p1: Point, p2: Point) -> _Segment:
    return _Segment(np.asarray(p1, float), np.asarray(p2, float))


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _f1(recall: float | None, precision: float | None) -> float | None:
    """둘 다 정의되지 않으면 None. 한쪽만 없으면 전부 놓치거나 전부 틀린 경우라 0이다."""
    if recall is None and precision is None:
        return None
    if not recall or not precision:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    recall = _ratio(tp, tp + fn)
    precision = _ratio(tp, tp + fp)
    return {"tp": tp, "fp": fp, "fn": fn, "recall": recall, "precision": precision,
            "f1": _f1(recall, precision)}


def _parallel(a: _Segment, b: _Segment, max_angle: float) -> bool:
    return abs(float(a.direction @ b.direction)) >= math.cos(max_angle)


def _coverage(
    a: _Segment, others: list[_Segment], cfg: dict[str, Any], buffer: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """a의 표본점별 (가장 가까운 일치 선분 번호, 거리). 없으면 번호가 -1이다."""
    points = a.samples(cfg["sample_step_mm"])
    best = np.full(len(points), math.inf)
    owner = np.full(len(points), -1)
    max_angle = math.radians(cfg["angle_deg"])
    limit = cfg["distance_mm"] if buffer is None else buffer
    for i, other in enumerate(others):
        if not _parallel(a, other, max_angle):
            continue
        d = other.distance(points)
        hit = (d <= limit) & (d < best)
        best[hit] = d[hit]
        owner[hit] = i
    return owner, best


def _covered_length(segments: list[_Segment], others: list[_Segment], cfg: dict[str, Any], buffer: float) -> float:
    return sum(
        s.length * float((_coverage(s, others, cfg, buffer)[0] >= 0).mean()) for s in segments
    )


def _wall_metrics(ir: IR, truth: Truth, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth_segs = [_seg(w.p1, w.p2) for w in truth.walls]
    pred_segs = [_seg(w.start, w.end) for w in ir.walls]

    total = covered = thick_ok = kind_ok = aligned = 0.0
    distances: list[np.ndarray] = []
    for tw, seg in zip(truth.walls, truth_segs):
        owner, dist = _coverage(seg, pred_segs, cfg)
        piece = seg.length / len(owner)
        total += seg.length
        distances.append(dist[owner >= 0])
        for i in owner[owner >= 0]:
            pred = ir.walls[int(i)]
            covered += piece
            thick_ok += piece * (abs(pred.thickness - tw.thickness) <= cfg["thickness_mm"])
            kind_ok += piece * (pred.kind == tw.kind)
            aligned += piece * abs(float(seg.direction @ pred_segs[int(i)].direction))

    pred_total = pred_covered = 0.0
    outcomes = []
    for wall, seg in zip(ir.walls, pred_segs):
        owner, _ = _coverage(seg, truth_segs, cfg)
        share = float((owner >= 0).mean())
        pred_total += seg.length
        pred_covered += seg.length * share
        outcomes.append({"kind": "wall", "id": wall.id, "score": wall.confidence.score,
                         "level": wall.confidence.level, "correct": share >= cfg["correct_share"]})

    sweep = []
    for buffer in cfg["buffer_sweep_mm"]:
        sweep.append({
            "buffer_mm": buffer,
            "completeness": _ratio(_covered_length(truth_segs, pred_segs, cfg, buffer), total),
            "correctness": _ratio(_covered_length(pred_segs, truth_segs, cfg, buffer), pred_total),
        })

    recall = _ratio(covered, total)
    precision = _ratio(pred_covered, pred_total)
    all_distances = np.concatenate(distances) if distances else np.empty(0)
    return {
        "recall": recall,
        "precision": precision,
        "f1": _f1(recall, precision),
        "thickness_accuracy": _ratio(thick_ok, covered),
        "material_accuracy": _ratio(kind_ok, covered),
        "orientation": _ratio(aligned, total),
        "accuracy_median_mm": float(np.median(all_distances)) if all_distances.size else None,
        "buffer_sweep": sweep,
        "truth_length_mm": total,
        "pred_length_mm": pred_total,
        "covered_truth_mm": covered,
        "covered_pred_mm": pred_covered,
    }, outcomes


def _column_metrics(ir: IR, truth: Truth, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs = sorted(
        (math.dist(c.center, t.center), i, j)
        for i, c in enumerate(ir.columns)
        for j, t in enumerate(truth.columns)
    )
    used_pred: set[int] = set()
    used_truth: set[int] = set()
    for d, i, j in pairs:
        if d > cfg["distance_mm"] or i in used_pred or j in used_truth:
            continue
        p, t = ir.columns[i], truth.columns[j]
        if max(abs(a - b) for a, b in zip(p.size, t.size)) > cfg["size_mm"]:
            continue
        used_pred.add(i)
        used_truth.add(j)

    outcomes = [
        {"kind": "column", "id": c.id, "score": c.confidence.score,
         "level": c.confidence.level, "correct": i in used_pred}
        for i, c in enumerate(ir.columns)
    ]
    tp = len(used_truth)
    return (
        _prf(tp, len(ir.columns) - tp, len(truth.columns) - tp) | {"matched": tp},
        outcomes,
    )


def opening_center(opening: Opening, host: Wall) -> Point:
    t = opening.position / max(math.dist(host.start, host.end), 1e-9)
    return (
        host.start[0] + (host.end[0] - host.start[0]) * t,
        host.start[1] + (host.end[1] - host.start[1]) * t,
    )


def _same_host(
    host: Wall, truth_wall: TruthWall, center: Point, cfg: dict[str, Any], walls_cfg: dict[str, Any]
) -> bool:
    """예측 호스트가 정답 호스트의 일부면 같은 벽이다. T자 분할을 틀리게 치지 않는다."""
    seg = _seg(host.start, host.end)
    if not _parallel(seg, _seg(truth_wall.p1, truth_wall.p2), math.radians(walls_cfg["angle_deg"])):
        return False
    return float(seg.distance(np.asarray([center], float))[0]) <= cfg["host_distance_mm"]


def _opening_metrics(
    ir: IR, truth: Truth, cfg: dict[str, Any], walls_cfg: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    walls = {w.id: w for w in ir.walls}
    truth_walls = {w.id: w for w in truth.walls}

    result: dict[str, Any] = {}
    matched: set[str] = set()
    for kind, key in (("door", "doors"), ("window", "windows")):
        targets = [t for t in truth.openings if t.kind == kind]
        candidates = [o for o in ir.openings if o.type == kind]
        located = [(o, walls[o.host_wall]) for o in candidates if o.host_wall in walls]
        pairs = sorted(
            ((math.dist(opening_center(o, h), t.center), i, j)
             for i, (o, h) in enumerate(located)
             for j, t in enumerate(targets)),
            key=lambda p: p[0],
        )
        used_pred: dict[int, int] = {}
        used_truth: set[int] = set()
        for d, i, j in pairs:
            if d > cfg["distance_mm"] or i in used_pred or j in used_truth:
                continue
            target = targets[j]
            if not _same_host(located[i][1], truth_walls[target.wall], target.center, cfg, walls_cfg):
                continue
            used_pred[i] = j
            used_truth.add(j)
        matched.update(located[i][0].id for i in used_pred)

        tp = len(used_truth)
        section = _prf(tp, len(candidates) - tp, len(targets) - tp) | {"matched": tp}
        widths = [abs(located[i][0].width - targets[j].width) for i, j in used_pred.items()]
        section["width_mae_mm"] = float(np.mean(widths)) if widths else None
        if kind == "door":
            correct = 0
            for i, j in used_pred.items():
                o, t = located[i][0], targets[j]
                if (
                    o.hinge is not None
                    and o.swing_dir is not None
                    and t.hinge is not None
                    and t.swing_dir is not None
                    and math.dist(o.hinge, t.hinge) <= cfg["hinge_distance_mm"]
                    and float(np.dot(o.swing_dir, t.swing_dir)) >= cfg["swing_min_dot"]
                ):
                    correct += 1
            section["direction_accuracy"] = _ratio(correct, len(used_pred))
        result[key] = section

    outcomes = [
        {"kind": "opening", "id": o.id, "score": o.confidence.score,
         "level": o.confidence.level, "correct": o.id in matched}
        for o in ir.openings
    ]
    return result, outcomes


def _room_iou_metrics(polygons: list[Polygon], truth: Truth, threshold: float) -> dict[str, Any]:
    """영역 단위 매칭. 실명은 보지 않는다. PQ = ΣIoU / (TP + FP/2 + FN/2)."""
    by_region = {r.region_id: r.polygon for r in truth.rooms if r.polygon is not None}
    targets = list(by_region.values())
    missing = len({r.region_id for r in truth.rooms}) - len(targets)

    tp = 0
    ious: list[float] = []
    if polygons and targets:
        matrix = np.zeros((len(polygons), len(targets)))
        for i, p in enumerate(polygons):
            for j, t in enumerate(targets):
                union = p.union(t).area
                matrix[i, j] = p.intersection(t).area / union if union else 0.0
        for i, j in zip(*linear_sum_assignment(-matrix)):
            if matrix[i, j] > threshold:
                tp += 1
                ious.append(float(matrix[i, j]))

    fp = len(polygons) - tp
    fn = len(targets) + missing - tp
    denominator = tp + fp / 2 + fn / 2
    result = _prf(tp, fp, fn)
    result["mean_iou"] = float(np.mean(ious)) if ious else None
    result["pq"] = sum(ious) / denominator if denominator else None
    return result


def _room_metrics(ir: IR, truth: Truth, cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    polygons = [Polygon(r.polygon).buffer(0) for r in ir.rooms]
    correct: set[int] = set()
    hit = 0
    for t in truth.rooms:
        label = ShapelyPoint(t.label_point)
        for i, (room, polygon) in enumerate(zip(ir.rooms, polygons)):
            if not polygon.contains(label) or t.name not in room.names:
                continue
            if t.clear_area_m2 and abs(room.clear_area_m2 / t.clear_area_m2 - 1) <= cfg["area_ratio"]:
                hit += 1
                correct.add(i)
                break
    outcomes = [
        {"kind": "room", "id": r.id, "score": r.confidence.score,
         "level": r.confidence.level, "correct": i in correct}
        for i, r in enumerate(ir.rooms)
    ]
    return {
        "recall": _ratio(hit, len(truth.rooms)),
        "precision": _ratio(len(correct), len(ir.rooms)),
        "matched": hit,
        "iou": _room_iou_metrics(polygons, truth, cfg["iou_threshold"]),
    }, outcomes


# ---------------------------------------------------------------- 벽 그래프
def _wall_graph(segments: list[tuple[Point, Point]], grid: float) -> tuple[list[Point], list[tuple[int, int]]]:
    """교차점에서 쪼갠 평면 그래프. 같은 위치의 끝점은 격자로 합친다."""
    if not segments:
        return [], []
    snapped = shapely.set_precision(MultiLineString([LineString(s) for s in segments]), grid)
    noded = unary_union(snapped)
    nodes: dict[Point, int] = {}
    edges: set[tuple[int, int]] = set()
    for part in shapely.get_parts(noded):
        coords = list(part.coords)
        for a, b in zip(coords, coords[1:]):
            i = nodes.setdefault((a[0], a[1]), len(nodes))
            j = nodes.setdefault((b[0], b[1]), len(nodes))
            if i != j:
                edges.add((min(i, j), max(i, j)))
    return list(nodes), sorted(edges)


def _betti(node_count: int, edges: list[tuple[int, int]]) -> tuple[int, int]:
    """(연결 요소 수, 독립 고리 수). 실이 닫히는지를 위상으로 본다."""
    parent = list(range(node_count))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        parent[find(a)] = find(b)
    components = len({find(i) for i in range(node_count)})
    return components, len(edges) - node_count + components


def _junctions(nodes: list[Point], edges: list[tuple[int, int]]) -> list[Point]:
    """끝점과 꺾임·교차점. 직선 중간의 2차 노드는 뺀다. 벽을 나눠도 접합이 늘지 않게 한다."""
    neighbours: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        neighbours[a].append(b)
        neighbours[b].append(a)

    result = []
    for i, p in enumerate(nodes):
        around = neighbours[i]
        if len(around) == 2:
            u = np.subtract(nodes[around[0]], p)
            v = np.subtract(nodes[around[1]], p)
            cosine = float(u @ v) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-9)
            if cosine < -0.999:
                continue
        result.append(p)
    return result


def _junction_metrics(pred: list[Point], truth: list[Point], cfg: dict[str, Any]) -> dict[str, Any]:
    """헝가리안 알고리즘으로 일대일 대응한 뒤 거리 임계값별로 센다."""
    if pred and truth:
        cost = np.linalg.norm(np.asarray(pred)[:, None] - np.asarray(truth)[None, :], axis=2)
        rows, cols = linear_sum_assignment(cost)
        assigned = cost[rows, cols]
    else:
        assigned = np.empty(0)
    result = {}
    for threshold in cfg["thresholds_mm"]:
        tp = int((assigned <= threshold).sum())
        result[str(threshold)] = _prf(tp, len(pred) - tp, len(truth) - tp)
    return result


def _topology_metrics(ir: IR, truth: Truth, junction_cfg: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict, dict]:
    grid = cfg["snap_grid_mm"]
    t_nodes, t_edges = _wall_graph([(w.p1, w.p2) for w in truth.walls], grid)
    p_nodes, p_edges = _wall_graph([(w.start, w.end) for w in ir.walls], grid)
    t0, t1 = _betti(len(t_nodes), t_edges)
    p0, p1 = _betti(len(p_nodes), p_edges)
    topology = {
        "betti0_truth": t0, "betti0_pred": p0, "betti0_error": abs(t0 - p0),
        "betti1_truth": t1, "betti1_pred": p1, "betti1_error": abs(t1 - p1),
    }
    junctions = _junction_metrics(_junctions(p_nodes, p_edges), _junctions(t_nodes, t_edges), junction_cfg)
    return topology, junctions


_CLASS_TO_ROLE = {
    "wall_rc": "WALL", "wall_block": "WALL", "wall_hatch": "HATCH", "column": "COLUMN",
    "column_hatch": "HATCH", "door": "DOOR", "window": "WINDOW", "dim": "DIM",
    "room_name": "ROOM_TEXT", "room_area": "ROOM_TEXT", "furniture": "FURNITURE",
    "fixture": "FURNITURE", "grid": "GRID", "grid_tag": "GRID", "stair": "STAIR",
    "frame": "OTHER", "note": "OTHER",
}


def layer_accuracy(roles: dict[str, str], truth: Truth) -> dict[str, Any]:
    """도면에 실제로 쓰인 레이어만 센다. 정답이 여러 역할이면 그중 하나면 맞다."""
    scored = {name: role for name, role in roles.items() if name in truth.layer_classes}
    correct = sum(
        role in {_CLASS_TO_ROLE.get(c) for c in truth.layer_classes[name]}
        for name, role in scored.items()
    )
    return {"accuracy": _ratio(correct, len(scored)), "correct": correct, "total": len(scored)}


def stage_walls(raw: list[Wall], truth: Truth, thresholds: dict[str, Any]) -> dict[str, Any]:
    """잇기 전 조각만의 벽 재현율·정밀도. 손실이 조각 찾기와 잇기 중 어디서 나는지 가른다."""
    truth_segs = [_seg(w.p1, w.p2) for w in truth.walls]
    pred_segs = [_seg(w.start, w.end) for w in raw]
    cfg = thresholds["walls"]
    limit = cfg["distance_mm"]
    total = sum(s.length for s in truth_segs)
    pred_total = sum(s.length for s in pred_segs)
    return {
        "recall": _ratio(_covered_length(truth_segs, pred_segs, cfg, limit), total),
        "precision": _ratio(_covered_length(pred_segs, truth_segs, cfg, limit), pred_total),
        "pieces": len(raw),
    }


def evaluate(ir: IR, truth: Truth, thresholds: dict[str, Any]) -> dict[str, Any]:
    walls, wall_outcomes = _wall_metrics(ir, truth, thresholds["walls"])
    columns, column_outcomes = _column_metrics(ir, truth, thresholds["columns"])
    openings, opening_outcomes = _opening_metrics(ir, truth, thresholds["openings"], thresholds["walls"])
    rooms, room_outcomes = _room_metrics(ir, truth, thresholds["rooms"])
    topology, junctions = _topology_metrics(ir, truth, thresholds["junctions"], thresholds["topology"])
    outcomes = wall_outcomes + column_outcomes + opening_outcomes + room_outcomes
    return {
        "walls": walls,
        "columns": columns,
        "openings": openings,
        "rooms": rooms,
        "junctions": junctions,
        "topology": topology,
        "confidence": confidence_metrics(outcomes, thresholds["calibration"]),
        "outcomes": outcomes,
    }
