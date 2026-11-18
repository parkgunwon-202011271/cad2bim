"""도면 단위 → mm 배율 판정.

치수 표기값은 mm로 적혀 있다고 가정한다 (국내 건축 도면 관행).
표기값 / 실제 길이의 비율이 일관되면 그 값이 가장 믿을 만한 근거다.
$INSUNITS는 없거나 틀린 경우가 많아 치수가 없을 때의 대안으로만 쓴다.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from ..errors import UnitScaleError
from .primitive import Primitive

_INSUNITS_TO_MM = {1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0}


@dataclass(frozen=True)
class ScaleDecision:
    factor: float
    source: str  # user | dimension | insunits
    samples: int = 0
    warnings: tuple[str, ...] = field(default=())


def _dimension_ratios(prims: list[Primitive]) -> list[float]:
    ratios = []
    for p in prims:
        if p.kind != "dimension":
            continue
        value, measured = p.geometry["value"], p.geometry["measured"]
        if value and measured > 0:
            ratios.append(value / measured)
    return ratios


def decide_scale(
    prims: list[Primitive], insunits: int, cfg: dict[str, Any], override: float | None = None
) -> ScaleDecision:
    if override is not None:
        if override <= 0:
            raise UnitScaleError("--unit-scale는 0보다 커야 합니다.")
        return ScaleDecision(override, "user")

    from_header = _INSUNITS_TO_MM.get(insunits)
    ratios = _dimension_ratios(prims)

    if len(ratios) >= cfg["min_dimension_samples"]:
        median = statistics.median(ratios)
        tol = cfg["ratio_tolerance"]
        consistent = sum(abs(r / median - 1) <= tol for r in ratios)
        share = consistent / len(ratios)
        snapped = any(abs(median / v - 1) <= cfg["snap_tolerance"] for v in cfg["snap_values"])
        mixed = cfg["majority_share"] <= share < cfg["min_consistent_share"] and snapped
        if share < cfg["min_consistent_share"] and not mixed:
            raise UnitScaleError(
                f"치수 표기값과 실제 길이의 비율이 일정하지 않습니다 (표본 {len(ratios)}개). "
                "--unit-scale로 도면 단위 1의 mm 값을 지정하세요.",
                f"scattered_n{len(ratios)}_share{round(100 * consistent / len(ratios))}",
            )
        factor = median
        for value in cfg["snap_values"]:
            if abs(median / value - 1) <= cfg["snap_tolerance"]:
                factor = float(value)
                break
        warnings: list[str] = ["scale_mixed"] if mixed else []
        if from_header is None:
            warnings.append("insunits_missing")
        elif abs(factor / from_header - 1) > tol:
            warnings.append("insunits_conflict")
        return ScaleDecision(factor, "dimension", len(ratios), tuple(warnings))

    if from_header is not None:
        return ScaleDecision(from_header, "insunits", warnings=("scale_unverified",))

    raise UnitScaleError(
        "단위를 판정할 수 없습니다. $INSUNITS가 없고 치수 표본도 부족합니다. "
        "--unit-scale로 도면 단위 1의 mm 값을 지정하세요.",
        f"no_evidence_n{len(ratios)}",
    )
