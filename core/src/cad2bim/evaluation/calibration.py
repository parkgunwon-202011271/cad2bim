"""확신도 평가. 보정(ECE)과 선택적 예측(위험-커버리지)을 함께 본다.

우리 시스템은 high만 믿고 나머지를 사람이 검토하므로, 평균 보정 오차보다
"위험 2%에서 얼마나 많은 요소를 자동으로 넘길 수 있는가"가 더 직접적인 지표다.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def reliability_table(outcomes: Sequence[dict[str, Any]], bins: int) -> list[dict[str, Any]]:
    rows = []
    for i in range(bins):
        low, high = i / bins, (i + 1) / bins
        members = [
            o for o in outcomes
            if low <= o["score"] < high or (i == bins - 1 and o["score"] == 1.0)
        ]
        if not members:
            continue
        rows.append({
            "low": low,
            "high": high,
            "count": len(members),
            "confidence": sum(o["score"] for o in members) / len(members),
            "accuracy": sum(o["correct"] for o in members) / len(members),
        })
    return rows


def expected_calibration_error(outcomes: Sequence[dict[str, Any]], bins: int) -> float | None:
    if not outcomes:
        return None
    return sum(
        r["count"] / len(outcomes) * abs(r["accuracy"] - r["confidence"])
        for r in reliability_table(outcomes, bins)
    )


def risk_coverage_curve(outcomes: Sequence[dict[str, Any]]) -> list[dict[str, float]]:
    """점수 내림차순으로 받아들이는 범위를 넓혀 간다. 같은 점수는 함께 움직인다."""
    ordered = sorted(outcomes, key=lambda o: -o["score"])
    curve: list[dict[str, float]] = []
    errors = 0
    for i, outcome in enumerate(ordered, start=1):
        errors += not outcome["correct"]
        if i == len(ordered) or ordered[i]["score"] != outcome["score"]:
            curve.append({
                "threshold": outcome["score"],
                "coverage": i / len(ordered),
                "risk": errors / i,
            })
    return curve


def aurc(curve: Sequence[dict[str, float]]) -> float | None:
    """위험-커버리지 곡선 아래 면적. 낮을수록 좋다. 계단 적분."""
    if not curve:
        return None
    area, previous = 0.0, 0.0
    for point in curve:
        area += point["risk"] * (point["coverage"] - previous)
        previous = point["coverage"]
    return area


def coverage_at_risk(curve: Sequence[dict[str, float]], target: float) -> float | None:
    if not curve:
        return None
    return max((p["coverage"] for p in curve if p["risk"] <= target), default=0.0)


def confidence_metrics(outcomes: Sequence[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    high = [o for o in outcomes if o["level"] == "high"]
    wrong = sum(not o["correct"] for o in high)
    curve = risk_coverage_curve(outcomes)
    return {
        "high_count": len(high),
        "high_wrong": wrong,
        "high_error_rate": wrong / len(high) if high else None,
        "ece": expected_calibration_error(outcomes, cfg["bins"]),
        "aurc": aurc(curve),
        "coverage_at_target_risk": coverage_at_risk(curve, cfg["target_risk"]),
        "target_risk": cfg["target_risk"],
        "reliability": reliability_table(outcomes, cfg["bins"]),
        "risk_coverage": curve,
    }
