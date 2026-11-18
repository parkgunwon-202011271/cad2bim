"""평가 프레임워크: 기하·위상·보정 지표와 벤치마크 실행기."""
from __future__ import annotations

import shutil

import pytest
import yaml
from conftest import DRAWINGS, NAMES, ROOT
from test_evaluation import ir_from_truth

from cad2bim.cli import main
from cad2bim.errors import Cad2BimError
from cad2bim.evaluation import compare_results, evaluate, load_manifest, load_truth, run_benchmark
from cad2bim.evaluation.benchmark import check_gate, lookup, render_summary
from cad2bim.evaluation.calibration import (
    aurc,
    confidence_metrics,
    coverage_at_risk,
    expected_calibration_error,
    risk_coverage_curve,
)
from cad2bim.evaluation.truth import Truth, TruthWall
from cad2bim.ir import Confidence

MANIFEST = ROOT / "config" / "benchmarks" / "synthetic.yaml"


@pytest.fixture
def b1() -> Truth:
    return load_truth(DRAWINGS / "B1_apt_clean_kr.truth.json")


@pytest.fixture
def a1() -> Truth:
    return load_truth(DRAWINGS / "A1_retail_clean_en.truth.json")


# ---------------------------------------------------------------- 정답 실 복원
@pytest.mark.parametrize("name", NAMES)
def test_derived_room_polygons_reproduce_truth_areas(name):
    """복원한 폴리곤이 정답 면적과 맞아야 Room IoU의 기준으로 믿을 수 있다."""
    truth = load_truth(DRAWINGS / f"{name}.truth.json")
    for room in truth.rooms:
        assert room.polygon is not None, room.name
        assert room.polygon.area / 1e6 == pytest.approx(room.clear_area_m2, abs=0.01), room.name


def test_shared_room_has_one_region(b1):
    living = next(r for r in b1.rooms if r.name == "거실")
    kitchen = next(r for r in b1.rooms if r.name == "주방")
    assert living.polygon.equals(kitchen.polygon)


# ---------------------------------------------------------------- 기하
def test_perfect_prediction_on_every_new_metric(b1, thresholds):
    m = evaluate(ir_from_truth(b1), b1, thresholds)
    assert m["walls"]["accuracy_median_mm"] == 0
    assert m["walls"]["orientation"] == pytest.approx(1)
    assert all(s["completeness"] == pytest.approx(1) for s in m["walls"]["buffer_sweep"])
    assert all(m["junctions"][t]["f1"] == 1 for t in ("50", "100", "200"))
    assert m["topology"]["betti0_error"] == 0 and m["topology"]["betti1_error"] == 0
    assert m["rooms"]["iou"]["f1"] == 1
    assert m["rooms"]["iou"]["pq"] == pytest.approx(1, abs=0.01)


def test_buffer_sweep_shows_offset_size(b1, thresholds):
    """70mm 어긋난 벽은 50mm 버퍼에서 빠지고 100mm 버퍼에서 잡혀야 한다."""
    ir = ir_from_truth(b1)
    moved = [w.model_copy(update={"start": (w.start[0] + 70, w.start[1] + 70),
                                  "end": (w.end[0] + 70, w.end[1] + 70)}) for w in ir.walls]
    m = evaluate(ir.model_copy(update={"walls": moved}), b1, thresholds)
    sweep = {s["buffer_mm"]: s["completeness"] for s in m["walls"]["buffer_sweep"]}
    assert sweep[10] == pytest.approx(0, abs=0.05)
    assert sweep[100] > 0.9
    assert m["walls"]["accuracy_median_mm"] == pytest.approx(70, abs=1)


def test_junctions_ignore_t_split_but_see_moved_corner(b1, thresholds):
    ir = ir_from_truth(b1)
    w5 = next(w for w in ir.walls if w.id == "W5")
    middle = (w5.start[0], 4800)  # W7이 닿는 T자 위치
    split = [w5.model_copy(update={"id": "a", "end": middle}),
             w5.model_copy(update={"id": "b", "start": middle})]
    others = [w for w in ir.walls if w.id != "W5"]
    m = evaluate(ir.model_copy(update={"walls": others + split, "openings": []}), b1, thresholds)
    assert m["junctions"]["100"]["f1"] == 1

    # 벽 끝을 300mm 줄이면 끝점과 접합이 어긋난다.
    w7 = next(w for w in ir.walls if w.id == "W7")
    short = w7.model_copy(update={"end": (w7.end[0] - 300, w7.end[1])})
    walls = [short if w.id == "W7" else w for w in ir.walls]
    m = evaluate(ir.model_copy(update={"walls": walls}), b1, thresholds)
    assert m["junctions"]["100"]["f1"] < 1


def wall(i, p1, p2):
    return TruthWall(f"W{i}", p1, p2, 200, "concrete")


def test_betti_counts_components_and_loops(thresholds):
    square = [wall(1, (0, 0), (1000, 0)), wall(2, (1000, 0), (1000, 1000)),
              wall(3, (1000, 1000), (0, 1000)), wall(4, (0, 1000), (0, 0))]
    alone = [wall(5, (5000, 0), (6000, 0))]
    truth = Truth(walls=square + alone)
    m = evaluate(ir_from_truth(truth), truth, thresholds)["topology"]
    assert m["betti0_truth"] == 2
    assert m["betti1_truth"] == 1


def test_gap_in_wall_is_a_topology_error(b1, thresholds):
    """길이로는 작은 틈이지만 실이 열린다. 위상 지표만 이것을 크게 본다."""
    ir = ir_from_truth(b1)
    w7 = next(w for w in ir.walls if w.id == "W7")
    short = w7.model_copy(update={"end": (w7.end[0] - 150, w7.end[1])})
    walls = [short if w.id == "W7" else w for w in ir.walls]
    m = evaluate(ir.model_copy(update={"walls": walls}), b1, thresholds)
    assert m["walls"]["recall"] > 0.99
    assert m["topology"]["betti1_error"] == 1


# ---------------------------------------------------------------- 실 매칭
def test_room_iou_rejects_half_overlap(b1, thresholds):
    ir = ir_from_truth(b1)
    target = next(r for r in ir.rooms if "침실1" in r.names)
    xs = [p[0] for p in target.polygon]
    shifted = [(x + (max(xs) - min(xs)) * 0.7, y) for x, y in target.polygon]
    rooms = [r.model_copy(update={"polygon": shifted}) if r is target else r for r in ir.rooms]
    m = evaluate(ir.model_copy(update={"rooms": rooms}), b1, thresholds)["rooms"]["iou"]
    assert (m["tp"], m["fp"], m["fn"]) == (6, 1, 1)
    assert m["pq"] == pytest.approx(6 / 7, abs=0.02)


def test_merged_rooms_lose_one_region(b1, thresholds):
    ir = ir_from_truth(b1)
    rooms = ir.rooms[:-1]
    m = evaluate(ir.model_copy(update={"rooms": rooms}), b1, thresholds)["rooms"]["iou"]
    assert m["fn"] == 1 and m["fp"] == 0


# ---------------------------------------------------------------- 확신도
def outcome(score, correct, level="review"):
    return {"kind": "wall", "id": "x", "score": score, "level": level, "correct": correct}


def test_ece_zero_when_confidence_matches_accuracy():
    outcomes = [outcome(0.75, True)] * 3 + [outcome(0.75, False)]
    assert expected_calibration_error(outcomes, 10) == pytest.approx(0)


def test_ece_detects_overconfidence():
    assert expected_calibration_error([outcome(0.95, False)] * 4, 10) == pytest.approx(0.95)


def test_score_one_lands_in_last_bin():
    assert expected_calibration_error([outcome(1.0, True)], 10) == pytest.approx(0)


def test_risk_coverage_ranks_errors_last():
    good = [outcome(0.9, True), outcome(0.8, True), outcome(0.7, True), outcome(0.1, False)]
    curve = risk_coverage_curve(good)
    assert [p["risk"] for p in curve] == [0, 0, 0, 0.25]
    assert coverage_at_risk(curve, 0.02) == pytest.approx(0.75)

    bad = [outcome(0.9, False), outcome(0.8, True), outcome(0.7, True), outcome(0.1, True)]
    bad_curve = risk_coverage_curve(bad)
    assert coverage_at_risk(bad_curve, 0.02) == 0
    assert aurc(bad_curve) > aurc(curve)


def test_tied_scores_move_together():
    curve = risk_coverage_curve([outcome(0.9, True), outcome(0.9, False)])
    assert len(curve) == 1
    assert curve[0]["risk"] == 0.5


def test_confidence_metrics_empty(thresholds):
    m = confidence_metrics([], thresholds["calibration"])
    assert m["ece"] is None and m["aurc"] is None and m["high_error_rate"] is None


def test_outcomes_cover_every_element(a1, thresholds):
    ir = ir_from_truth(a1)
    m = evaluate(ir, a1, thresholds)
    assert len(m["outcomes"]) == len(ir.walls) + len(ir.columns) + len(ir.openings) + len(ir.rooms)
    assert all(o["correct"] for o in m["outcomes"])


def test_review_level_error_is_not_a_high_error(b1, thresholds):
    ir = ir_from_truth(b1)
    review = Confidence(level="review", score=0.5)
    fake = ir.walls[0].model_copy(
        update={"id": "X", "start": (1000, 1000), "end": (1000, 3000), "confidence": review}
    )
    m = evaluate(ir.model_copy(update={"walls": [*ir.walls, fake]}), b1, thresholds)["confidence"]
    assert m["high_wrong"] == 0
    assert m["coverage_at_target_risk"] < 1


# ---------------------------------------------------------------- 벤치마크
def test_manifest_points_at_real_files():
    manifest = load_manifest(MANIFEST)
    assert {e.id for e in manifest.entries} == {"A1", "A2", "B1", "B2"}
    assert all(e.dxf.is_file() and e.truth.is_file() for e in manifest.entries)
    assert not next(e for e in manifest.entries if e.id == "A2").gated


def perfect_convert(monkeypatch, truth_for=None):
    """인식기가 완벽하다고 치고 실행기만 시험한다."""
    import cad2bim.evaluation.benchmark as bm
    from cad2bim.pipeline import convert as real

    def fake(path, out, cfg, height, unit_scale=None):
        result = real(path, out, cfg, height, unit_scale=unit_scale)
        truth = load_truth(path.with_suffix(".truth.json"))
        ir = ir_from_truth(truth)
        result.ir = result.ir.model_copy(update={
            "walls": ir.walls, "columns": ir.columns, "openings": ir.openings, "rooms": ir.rooms,
        })
        return result

    monkeypatch.setattr(bm, "convert", fake)


def test_benchmark_without_recognition_fails_gate(tmp_path, monkeypatch):
    """아무것도 찾지 못하면 관문이 실패해야 한다. 통과하면 관문이 무의미한 것이다."""
    import cad2bim.evaluation.benchmark as bm
    from cad2bim.pipeline import convert as real

    def empty(path, out, cfg, height, unit_scale=None):
        result = real(path, out, cfg, height, unit_scale=unit_scale)
        result.ir = result.ir.model_copy(update={"walls": [], "columns": []})
        return result

    monkeypatch.setattr(bm, "convert", empty)
    results = run_benchmark(MANIFEST, tmp_path, gate="m2")
    assert results["gate"]["passed"] is False
    assert {f["drawing"] for f in results["gate"]["failures"]} == {"A1", "B1", "B2"}
    assert (tmp_path / "results.json").is_file()
    assert "관문 m2: 실패" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_benchmark_perfect_recognizer_passes_gate(tmp_path, monkeypatch):
    perfect_convert(monkeypatch)
    results = run_benchmark(MANIFEST, tmp_path, gate="m3")
    assert results["gate"] == {"name": "m3", "passed": True, "failures": []}
    summary = results["splits"]["dev"]
    assert summary["completed"] == 4
    assert summary["macro"]["walls.f1"] == pytest.approx(1)
    assert summary["micro"]["openings.doors.recall"] == 1
    assert summary["confidence"]["high_error_rate"] == 0


def test_results_record_provenance(tmp_path):
    p = run_benchmark(MANIFEST, tmp_path)["provenance"]
    assert len(p["config_sha256"]) == 64 and len(p["manifest_sha256"]) == 64
    assert p["packages"]["ezdxf"]
    assert p["timestamp"]


def test_results_contain_only_aggregate_numbers(tmp_path, monkeypatch):
    """실제 도면에 쓸 때 실명, 레이어명, 파일 경로가 결과에 새면 안 된다."""
    perfect_convert(monkeypatch)
    run_benchmark(MANIFEST, tmp_path)
    text = (tmp_path / "results.json").read_text(encoding="utf-8")
    text += (tmp_path / "summary.md").read_text(encoding="utf-8")
    for forbidden in ("거실", "침실", "벽체", "A-WALL", ".dxf", "DATASET"):
        assert forbidden not in text


def small_manifest(tmp_path, drawings):
    data = {
        "name": "t", "root": str(DRAWINGS), "floor_height": 3000, "drawings": drawings,
        "gates": {"g": {"walls.recall": 0.5, "topology.betti0_error": 0}},
    }
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def entry(id_, name="A1_retail_clean_en", **extra):
    return {"id": id_, "dxf": f"{name}.dxf", "truth": f"{name}.truth.json", **extra}


def test_failed_drawing_counts_as_zero_not_skipped(tmp_path, monkeypatch):
    perfect_convert(monkeypatch)
    drawing = tmp_path / "bad"
    drawing.mkdir()
    (drawing / "x.dxf").write_text("garbage", encoding="utf-8")
    shutil.copy(DRAWINGS / "A1_retail_clean_en.truth.json", drawing / "x.truth.json")
    path = small_manifest(tmp_path, [
        entry("ok"), {"id": "bad", "dxf": str(drawing / "x.dxf"), "truth": str(drawing / "x.truth.json")},
    ])
    results = run_benchmark(path, tmp_path / "o", gate="g")
    summary = results["splits"]["dev"]
    assert summary["completed"] == 1
    assert summary["macro"]["walls.f1"] == pytest.approx(0.5)
    assert results["gate"]["passed"] is False


def test_splits_are_reported_separately(tmp_path, monkeypatch):
    perfect_convert(monkeypatch)
    path = small_manifest(tmp_path, [entry("a", split="dev"), entry("b", split="holdout")])
    results = run_benchmark(path, tmp_path / "o")
    assert set(results["splits"]) == {"dev", "holdout"}
    assert results["splits"]["holdout"]["drawings"] == 1


def test_bootstrap_only_with_enough_drawings(tmp_path, monkeypatch):
    perfect_convert(monkeypatch)
    few = run_benchmark(small_manifest(tmp_path, [entry(str(i)) for i in range(4)]), tmp_path / "o")
    assert few["splits"]["dev"]["bootstrap_95"] is None
    many = run_benchmark(small_manifest(tmp_path, [entry(str(i)) for i in range(5)]), tmp_path / "p")
    assert many["splits"]["dev"]["bootstrap_95"]["walls.f1"] == pytest.approx([1, 1])


def test_unknown_gate_and_duplicate_ids_are_errors(tmp_path):
    with pytest.raises(Cad2BimError, match="관문"):
        run_benchmark(MANIFEST, tmp_path, gate="zz")
    with pytest.raises(Cad2BimError, match="중복"):
        load_manifest(small_manifest(tmp_path, [entry("a"), entry("a")]))


def record(id_, recall, betti=0):
    return {"id": id_, "gated": True, "status": "ok",
            "metrics": {"walls": {"recall": recall}, "topology": {"betti0_error": betti}}}


def test_gate_handles_lower_is_better_and_missing_values():
    gate = {"walls.recall": 0.9, "topology.betti0_error": 0}
    assert check_gate([record("a", 0.95)], gate) == []
    assert len(check_gate([record("a", 0.95, betti=1)], gate)) == 1
    assert len(check_gate([{"id": "a", "gated": True, "metrics": None}], gate)) == 2


def test_gate_skips_absent_elements_but_not_invented_ones():
    def with_columns(tp, fp, fn):
        f1 = None if tp + fp + fn == 0 else 0.0
        return {"id": "b", "gated": True, "status": "ok",
                "metrics": {"columns": {"tp": tp, "fp": fp, "fn": fn, "f1": f1}}}

    gate = {"columns.f1": 1.0}
    assert check_gate([with_columns(0, 0, 0)], gate) == []
    assert len(check_gate([with_columns(0, 2, 0)], gate)) == 1


def test_compare_finds_per_drawing_regression_hidden_by_mean():
    old = {"drawings": [record("a", 1.0), record("b", 0.5)]}
    new = {"drawings": [record("a", 0.5), record("b", 1.0)]}
    regressions = compare_results(old, new, 0.005)
    assert [(r["drawing"], r["metric"]) for r in regressions] == [("a", "walls.recall")]


def test_compare_lower_is_better_and_tolerance():
    old = {"drawings": [record("a", 1.0, betti=0)]}
    assert compare_results(old, {"drawings": [record("a", 0.999, betti=1)]}, 0.005) == [
        {"drawing": "a", "metric": "topology.betti0_error", "old": 0, "new": 1}
    ]


def test_cli_benchmark_exit_codes(tmp_path, capsys, monkeypatch):
    assert main(["benchmark", str(MANIFEST), "--out", str(tmp_path / "a"), "--gate", "m3"]) == 0
    assert "관문 m3: 통과" in capsys.readouterr().out

    import cad2bim.evaluation.benchmark as bm
    from cad2bim.pipeline import convert as real

    def no_walls(path, out, cfg, height, unit_scale=None):
        result = real(path, out, cfg, height, unit_scale=unit_scale)
        result.ir = result.ir.model_copy(update={"walls": []})
        return result

    monkeypatch.setattr(bm, "convert", no_walls)
    assert main(["benchmark", str(MANIFEST), "--out", str(tmp_path / "b"), "--gate", "m3"]) == 1
    assert "관문 m3: 실패" in capsys.readouterr().out
    first = tmp_path / "a" / "results.json"
    assert main(["compare", str(first), str(first)]) == 0


def test_lookup_paths():
    assert lookup({"a": {"100": {"f1": 1}}}, "a.100.f1") == 1
    assert lookup({"a": {}}, "a.b") is None
    assert lookup(None, "a") is None


def test_summary_renders_without_metrics():
    results = {
        "benchmark": "t", "gate": None,
        "provenance": {"timestamp": "t", "git_commit": None, "git_dirty": None, "config_sha256": "0" * 64},
        "drawings": [{"id": "a", "split": "dev", "metrics": None}],
        "splits": {"dev": {"completed": 0, "drawings": 1,
                           "macro": dict.fromkeys(("walls.recall",), None) | _all_none(),
                           "confidence": {"target_risk": 0.02, "ece": None, "aurc": None,
                                          "coverage_at_target_risk": None}}},
    }
    assert "| a |" in render_summary(results)


def _all_none():
    from cad2bim.evaluation.benchmark import PRIMARY_METRICS

    return {path: None for path, _ in PRIMARY_METRICS}

