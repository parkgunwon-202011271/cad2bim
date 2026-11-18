"""규칙 기반 확신도 (상세설계 6장). 학습 없는 근거 합산이다.

없는 근거는 0점이 아니라 빠진 것으로 보고 가중치를 나머지에 다시 나눈다.
"""
from __future__ import annotations

from typing import Any

from ..ir import Confidence
from ..layers import LayerMap


def _layer_evidence(layer_names: set[str], layers: LayerMap, role: str) -> float | None:
    """역할이 사전이나 사용자로 확정되면 1, 기하로만 찾았으면 근거 없음(None)."""
    # 기하로 추정한 레이어는 근거로 치지 않는다. 같은 기하를 두 번 세는 셈이다.
    sources = [
        layers.layers[name].source for name in layer_names
        if name in layers.layers and layers.layers[name].role == role
        and layers.layers[name].source != "inferred"
    ]
    return 1.0 if sources else None


def _combine(evidence: dict[str, float | None], cfg: dict[str, Any]) -> float:
    available = {k: v for k, v in evidence.items() if v is not None}
    weight = sum(cfg["weights"][k] for k in available)
    return sum(cfg["weights"][k] * v for k, v in available.items()) / weight if weight else 0.0


def _result(score: float, reasons: list[str], cfg: dict[str, Any]) -> Confidence:
    high = score >= cfg["high_threshold"] and not reasons
    return Confidence(level="high" if high else "review", score=round(score, 3), reasons=reasons)


def score_wall(draft: Any, layers: LayerMap, has_hatches: bool, cfg: dict[str, Any]) -> Confidence:
    layer = _layer_evidence(draft.layers, layers, "WALL")
    ends_ok = all(draft.joined)
    hatched = draft.hatch_share >= cfg["hatch_min_share"]
    geometry = 0.5 * draft.on_peak + 0.5 * ends_ok
    material = (1.0 if hatched else 0.0) if has_hatches else None

    reasons = []
    if not ends_ok:
        reasons.append("dangling_end")
    if not draft.on_peak:
        reasons.append("thickness_outlier")
    if draft.single_line:
        reasons.append("single_line_wall")
    if draft.glazed:
        reasons.append("glazed_wall")
    elif draft.candidate:
        reasons.append("wall_candidate")
    elif draft.from_door:
        reasons.append("wall_from_door")
    elif any(n in layers.layers and layers.layers[n].source == "inferred" for n in draft.layers):
        reasons.append("wall_layer_inferred")
    elif layer is None and not hatched:
        reasons.append("layer_unresolved")
    if draft.glazed or draft.candidate:
        layer, material, geometry = None, None, 0.0  # 레이어와 형태가 벽이라는 근거가 아니다
    return _result(_combine({"layer": layer, "geometry": geometry, "material": material}, cfg), reasons, cfg)


def score_opening(draft: Any, cfg: dict[str, Any]) -> Confidence:
    """호나 평행선의 형태 자체가 강한 근거다. 역할이 확인된 레이어면 더 믿는다."""
    layer = 1.0 if draft.role_known else None
    geometry = 1.0 if draft.type != "unknown" else 0.0
    reasons = list(draft.reasons)
    score = _combine({"layer": layer, "geometry": geometry, "material": None}, cfg)
    return _result(score, reasons, cfg)


def score_room(draft: Any, cfg: dict[str, Any]) -> Confidence:
    reasons = []
    if draft.names == ["미지정"]:
        reasons.append("no_room_name")
    elif len(draft.names) > 1:
        reasons.append("multiple_room_names")
    if draft.area_note_ok is False:
        reasons.append("area_note_mismatch")
    layer = 1.0 if draft.has_name_role else None
    material = None if draft.area_note_ok is None else float(draft.area_note_ok)
    score = _combine({"layer": layer, "geometry": 1.0, "material": material}, cfg)
    return _result(score, reasons, cfg)


def score_column(draft: Any, layers: LayerMap, cfg: dict[str, Any]) -> Confidence:
    layer = _layer_evidence({draft.layer}, layers, "COLUMN")
    reasons = [] if layer is not None or draft.hatched else ["layer_unresolved"]
    if draft.inferred:
        reasons.append("column_inferred")
    evidence = {"layer": layer, "geometry": 1.0, "material": 1.0 if draft.hatched else 0.0}
    return _result(_combine(evidence, cfg), reasons, cfg)
