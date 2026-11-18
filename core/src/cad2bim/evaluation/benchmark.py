"""재현 가능한 벤치마크 실행. 매니페스트의 도면을 전부 돌려 숫자와 실행 조건을 함께 남긴다.

실제 도면에도 그대로 쓰므로 결과에는 집계 숫자만 넣는다. 파일 경로, 레이어명, 실명,
좌표는 기록하지 않는다. 도면은 매니페스트의 id로만 부른다.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from ..config import config_dir, load_defaults, load_thresholds
from ..errors import Cad2BimError
from ..pipeline import convert
from .calibration import confidence_metrics
from .metrics import evaluate, layer_accuracy, stage_walls
from .truth import load_truth

# (경로, 높을수록 좋은가). 보고와 회귀 비교에 쓰는 대표 지표.
PRIMARY_METRICS: list[tuple[str, bool]] = [
    ("walls.recall", True),
    ("walls.precision", True),
    ("walls.f1", True),
    ("walls.thickness_accuracy", True),
    ("walls.material_accuracy", True),
    ("walls.orientation", True),
    ("walls.accuracy_median_mm", False),
    ("layers.accuracy", True),
    ("stages.wall_pieces.recall", True),
    ("stages.wall_pieces.precision", True),
    ("junctions.100.f1", True),
    ("columns.f1", True),
    ("openings.doors.f1", True),
    ("openings.doors.direction_accuracy", True),
    ("openings.windows.f1", True),
    ("rooms.recall", True),
    ("rooms.iou.f1", True),
    ("rooms.iou.pq", True),
    ("topology.betti0_error", False),
    ("topology.betti1_error", False),
]
_HIGHER_IS_BETTER = dict(PRIMARY_METRICS) | {
    "confidence.high_error_rate": False,
    "confidence.ece": False,
    "confidence.aurc": False,
    "confidence.coverage_at_target_risk": True,
}
# 정밀도·재현율 집계용 (구역, tp/fp/fn 위치).
_COUNT_SECTIONS = {
    "columns": ("columns",),
    "openings.doors": ("openings", "doors"),
    "openings.windows": ("openings", "windows"),
    "rooms.iou": ("rooms", "iou"),
    "junctions.100": ("junctions", "100"),
}


@dataclass(frozen=True)
class Entry:
    id: str
    dxf: Path
    truth: Path
    split: str
    gated: bool
    unit_scale: float | None = None


@dataclass
class Manifest:
    name: str
    floor_height: float
    entries: list[Entry]
    gates: dict[str, dict[str, float]]
    digest: str


def load_manifest(path: Path) -> Manifest:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)
    root = (Path(path).parent / data.get("root", ".")).resolve()
    entries = [
        Entry(
            id=d["id"],
            dxf=root / d["dxf"],
            truth=root / d["truth"],
            split=d.get("split", "dev"),
            gated=bool(d.get("gated", True)),
            unit_scale=d.get("unit_scale"),
        )
        for d in data["drawings"]
    ]
    ids = [e.id for e in entries]
    if len(set(ids)) != len(ids):
        raise Cad2BimError("매니페스트에 중복된 도면 id가 있습니다.")
    return Manifest(
        data["name"], float(data.get("floor_height", 3000)), entries,
        data.get("gates", {}), hashlib.sha256(raw).hexdigest(),
    )


def lookup(metrics: dict[str, Any] | None, path: str) -> Any:
    """'openings.doors.f1'처럼 점으로 구분한 경로의 값. 없으면 None."""
    node: Any = metrics
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _micro(records: list[dict[str, Any]]) -> dict[str, Any]:
    """도면을 합쳐 계산한 값. 벽은 길이, 나머지는 개수를 합친다."""
    done = [r["metrics"] for r in records if r["status"] == "ok"]
    micro: dict[str, Any] = {}
    truth = sum(m["walls"]["truth_length_mm"] for m in done)
    pred = sum(m["walls"]["pred_length_mm"] for m in done)
    micro["walls.recall"] = sum(m["walls"]["covered_truth_mm"] for m in done) / truth if truth else None
    micro["walls.precision"] = sum(m["walls"]["covered_pred_mm"] for m in done) / pred if pred else None
    for name, keys in _COUNT_SECTIONS.items():
        sections = []
        for m in done:
            node = m
            for key in keys:
                node = node[key]
            sections.append(node)
        tp = sum(s["tp"] for s in sections)
        fp = sum(s["fp"] for s in sections)
        fn = sum(s["fn"] for s in sections)
        recall = tp / (tp + fn) if tp + fn else None
        precision = tp / (tp + fp) if tp + fp else None
        micro[f"{name}.recall"] = recall
        micro[f"{name}.precision"] = precision
    return micro


def _macro(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """도면별 평균. 실패한 도면은 높을수록 좋은 지표에서 0으로 센다."""
    result: dict[str, float | None] = {}
    for path, higher in PRIMARY_METRICS:
        values = []
        for record in records:
            value = lookup(record["metrics"], path)
            if value is None and record["status"] != "ok" and higher:
                value = 0.0
            if value is not None:
                values.append(float(value))
        result[path] = statistics.fmean(values) if values else None
    return result


def _bootstrap(records: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, list[float]] | None:
    """도면 단위 재표집 95% 구간. 도면이 적으면 구간이 의미 없어 내지 않는다."""
    if len(records) < cfg["bootstrap_min_drawings"]:
        return None
    rng = random.Random(cfg["bootstrap_seed"])
    intervals = {}
    for path, _ in PRIMARY_METRICS:
        values = [float(v) for r in records if (v := lookup(r["metrics"], path)) is not None]
        if not values:
            continue
        means = sorted(
            statistics.fmean(rng.choices(values, k=len(values)))
            for _ in range(cfg["bootstrap_samples"])
        )
        n = len(means)
        intervals[path] = [means[int(0.025 * n)], means[min(int(0.975 * n), n - 1)]]
    return intervals


def _not_applicable(metrics: dict[str, Any] | None, path: str) -> bool:
    """정답에도 예측에도 없는 요소(기둥 없는 도면의 기둥 등)는 판정 대상이 아니다."""
    section = lookup(metrics, path.rpartition(".")[0])
    return (
        isinstance(section, dict)
        and {"tp", "fp", "fn"} <= section.keys()
        and section["tp"] + section["fp"] + section["fn"] == 0
    )


def check_gate(records: list[dict[str, Any]], gate: dict[str, float]) -> list[dict[str, Any]]:
    """gated 도면이 기준을 모두 넘는지 본다. 값이 없으면 실패다."""
    failures = []
    for record in records:
        if not record["gated"]:
            continue
        for path, required in gate.items():
            if _not_applicable(record["metrics"], path):
                continue
            value = lookup(record["metrics"], path)
            higher = _HIGHER_IS_BETTER.get(path, True)
            passed = value is not None and (value >= required if higher else value <= required)
            if not passed:
                failures.append({"drawing": record["id"], "metric": path,
                                 "required": required, "value": value})
    return failures


def _git(args: list[str]) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=30,
                             cwd=Path(__file__).resolve().parent)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def config_digest() -> str:
    directory = config_dir()
    digest = hashlib.sha256()
    for name in ("defaults.yaml", "thresholds.yaml", "layer_dict.yaml"):
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def _provenance(manifest: Manifest) -> dict[str, Any]:
    status = _git(["status", "--porcelain"])
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status) if status is not None else None,
        "config_sha256": config_digest(),
        "manifest_sha256": manifest.digest,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("cad2bim", "ezdxf", "shapely", "scipy")},
    }


def score_result(result: Any, truth: Any, thresholds: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluate(result.ir, truth, thresholds)
    roles = {name: info.role for name, info in result.layers.layers.items()}
    metrics["layers"] = layer_accuracy(roles, truth)
    metrics["stages"] = {"wall_pieces": stage_walls(result.recognition.raw_walls, truth, thresholds)}
    return metrics


def _run_one(entry: Entry, out_dir: Path, cfg: dict, thresholds: dict, height: float) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": entry.id, "split": entry.split, "gated": entry.gated,
        "status": "ok", "error": None, "metrics": None,
    }
    started = time.perf_counter()
    try:
        result = convert(entry.dxf, out_dir / entry.id, cfg, height, unit_scale=entry.unit_scale)
    except Cad2BimError as exc:
        # 메시지에 경로나 레이어명이 들어갈 수 있어 오류 종류만 남긴다.
        record["status"] = "failed"
        record["error"] = type(exc).__name__
    else:
        record["input_sha256"] = result.ir.source.sha256
        record["warnings"] = result.ir.warnings
        record["metrics"] = score_result(result, load_truth(entry.truth), thresholds)
    record["runtime_s"] = round(time.perf_counter() - started, 3)
    return record


def _summarize(records: list[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    outcomes = [o for r in records if r["status"] == "ok" for o in r["metrics"]["outcomes"]]
    return {
        "drawings": len(records),
        "completed": sum(r["status"] == "ok" for r in records),
        "macro": _macro(records),
        "micro": _micro(records),
        "bootstrap_95": _bootstrap(records, thresholds["benchmark"]),
        "confidence": confidence_metrics(outcomes, thresholds["calibration"]),
    }


def run_benchmark(manifest_path: Path, out_dir: Path, gate: str | None = None) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    if gate is not None and gate not in manifest.gates:
        raise Cad2BimError(f"매니페스트에 관문 '{gate}'가 없습니다: {sorted(manifest.gates)}")
    cfg, thresholds = load_defaults(), load_thresholds()
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [_run_one(e, out_dir, cfg, thresholds, manifest.floor_height) for e in manifest.entries]

    # 개발용과 시험용은 절대 섞어 보고하지 않는다.
    splits = sorted({r["split"] for r in records})
    results = {
        "benchmark": manifest.name,
        "provenance": _provenance(manifest),
        "splits": {s: _summarize([r for r in records if r["split"] == s], thresholds) for s in splits},
        "drawings": records,
        "gate": None,
    }
    if gate is not None:
        failures = check_gate(records, manifest.gates[gate])
        results["gate"] = {"name": gate, "passed": not failures, "failures": failures}

    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_summary(results), encoding="utf-8")
    return results


def _cell(value: Any) -> str:
    return "-" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def render_summary(results: dict[str, Any]) -> str:
    p = results["provenance"]
    commit = (p["git_commit"] or "커밋 없음")[:10] + (" (작업 중 변경 있음)" if p["git_dirty"] else "")
    lines = [
        f"# 벤치마크: {results['benchmark']}", "",
        f"- 시각: {p['timestamp']}", f"- 코드: {commit}", f"- 설정: {p['config_sha256'][:10]}", "",
    ]
    ids = [r["id"] for r in results["drawings"]]
    lines += ["| 지표 | 평균 | " + " | ".join(ids) + " |", "|---|---|" + "---|" * len(ids)]
    for split, summary in results["splits"].items():
        lines.append(f"| **{split}** ({summary['completed']}/{summary['drawings']} 완주) | | "
                     + "| " * len(ids))
        for path, _ in PRIMARY_METRICS:
            row = [_cell(lookup(r["metrics"], path)) if r["split"] == split else ""
                   for r in results["drawings"]]
            lines.append(f"| {path} | {_cell(summary['macro'][path])} | " + " | ".join(row) + " |")
        c = summary["confidence"]
        lines.append(f"| confidence (ECE / AURC / 위험 {c['target_risk']:.0%} 커버리지) | "
                     f"{_cell(c['ece'])} / {_cell(c['aurc'])} / {_cell(c['coverage_at_target_risk'])} | "
                     + "| " * len(ids))
    if results["gate"]:
        gate = results["gate"]
        lines += ["", f"## 관문 {gate['name']}: {'통과' if gate['passed'] else '실패'}"]
        lines += [f"- {f['drawing']} {f['metric']}: {_cell(f['value'])} (기준 {f['required']})"
                  for f in gate["failures"]]
    return "\n".join(lines) + "\n"


def compare_results(old: dict[str, Any], new: dict[str, Any], tolerance: float) -> list[dict[str, Any]]:
    """도면별 대표 지표의 퇴보를 찾는다. 평균에 가려지는 후퇴를 막기 위해 도면 단위로 본다."""
    old_by_id = {r["id"]: r for r in old["drawings"]}
    regressions = []
    for record in new["drawings"]:
        before = old_by_id.get(record["id"])
        if before is None:
            continue
        for path, higher in PRIMARY_METRICS:
            a, b = lookup(before["metrics"], path), lookup(record["metrics"], path)
            if a is None:
                continue
            worse = b is None or (b < a - tolerance if higher else b > a + tolerance)
            if worse:
                regressions.append({"drawing": record["id"], "metric": path, "old": a, "new": b})
    return regressions
