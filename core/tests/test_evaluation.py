"""평가 도구가 틀린 것을 틀렸다고 말하는지 확인한다. 인식기보다 먼저 믿을 수 있어야 한다."""
from __future__ import annotations

import pytest
from conftest import DRAWINGS, NAMES

from cad2bim.evaluation import evaluate, load_truth
from cad2bim.evaluation.truth import Truth
from cad2bim.ir import IR, Column, Confidence, Level, Opening, Room, SourceInfo, Wall

HIGH = Confidence(level="high", score=0.95)


def ir_from_truth(truth: Truth) -> IR:
    """정답을 그대로 IR로 옮긴다. 평가하면 모든 지표가 1이어야 한다."""
    walls = [
        Wall(id=w.id, start=w.p1, end=w.p2, thickness=w.thickness, kind=w.kind, confidence=HIGH)
        for w in truth.walls
    ]
    by_id = {w.id: w for w in truth.walls}
    openings = []
    for o in truth.openings:
        w = by_id[o.wall]
        position = abs(o.center[0] - w.p1[0]) + abs(o.center[1] - w.p1[1])
        openings.append(
            Opening(id=o.id, type=o.kind, host_wall=o.wall, position=position, width=o.width,
                    hinge=o.hinge, swing_dir=o.swing_dir, confidence=HIGH)
        )
    columns = [
        Column(id=f"C{i}", shape="rect", center=c.center, size=c.size, confidence=HIGH)
        for i, c in enumerate(truth.columns)
    ]
    regions: dict[int, list] = {}
    for r in truth.rooms:
        regions.setdefault(r.region_id, []).append(r)
    rooms = []
    for region_id, members in regions.items():
        polygon = list(members[0].polygon.exterior.coords)
        rooms.append(
            Room(id=f"R{region_id}", names=[m.name for m in members], point=members[0].label_point,
                 polygon=polygon, clear_area_m2=members[0].clear_area_m2, confidence=HIGH)
        )
    return IR(
        source=SourceInfo(file="t.dxf", sha256="0" * 64, unit_scale=1.0, unit_source="dimension"),
        level=Level(height=3000),
        walls=walls, columns=columns, openings=openings, rooms=rooms,
    )


@pytest.fixture(params=NAMES)
def truth(request) -> Truth:
    return load_truth(DRAWINGS / f"{request.param}.truth.json")


def test_truth_as_prediction_scores_perfectly(truth, thresholds):
    m = evaluate(ir_from_truth(truth), truth, thresholds)
    assert m["walls"]["recall"] == pytest.approx(1)
    assert m["walls"]["precision"] == pytest.approx(1)
    assert m["walls"]["thickness_accuracy"] == pytest.approx(1)
    assert m["walls"]["material_accuracy"] == pytest.approx(1)
    if truth.columns:
        assert m["columns"]["recall"] == 1 and m["columns"]["precision"] == 1
    assert m["openings"]["doors"]["recall"] == 1
    assert m["openings"]["doors"]["direction_accuracy"] == 1
    assert m["openings"]["windows"]["precision"] == 1
    assert m["rooms"]["recall"] == 1
    assert m["confidence"]["high_wrong"] == 0


def test_empty_prediction_scores_zero_without_crashing(truth, thresholds):
    ir = ir_from_truth(truth).model_copy(update={"walls": [], "columns": [], "openings": [], "rooms": []})
    m = evaluate(ir, truth, thresholds)
    assert m["walls"]["recall"] == 0
    assert m["walls"]["precision"] is None
    assert m["rooms"]["recall"] == 0
    assert m["confidence"]["high_error_rate"] is None


@pytest.fixture
def b1() -> Truth:
    return load_truth(DRAWINGS / "B1_apt_clean_kr.truth.json")


@pytest.fixture
def a1() -> Truth:
    return load_truth(DRAWINGS / "A1_retail_clean_en.truth.json")


def test_splitting_a_wall_at_a_t_junction_is_not_an_error(b1, thresholds):
    ir = ir_from_truth(b1)
    w5 = next(w for w in ir.walls if w.id == "W5")
    middle = (w5.start[0], 4800)
    parts = [w5.model_copy(update={"id": "W5a", "end": middle}),
             w5.model_copy(update={"id": "W5b", "start": middle})]
    openings = []
    for o in ir.openings:
        if o.host_wall == "W5":
            below = o.position < 4800
            o = o.model_copy(update={"host_wall": "W5a" if below else "W5b",
                                     "position": o.position if below else o.position - 4800})
        openings.append(o)
    split = ir.model_copy(update={
        "walls": [w for w in ir.walls if w.id != "W5"] + parts, "openings": openings,
    })
    m = evaluate(split, b1, thresholds)
    assert m["walls"]["recall"] == pytest.approx(1)
    assert m["walls"]["precision"] == pytest.approx(1)
    assert m["openings"]["doors"]["recall"] == 1


def test_missing_wall_lowers_recall_by_length_share(b1, thresholds):
    ir = ir_from_truth(b1)
    removed = next(w for w in b1.walls if w.id == "W10")
    length = abs(removed.p2[1] - removed.p1[1]) + abs(removed.p2[0] - removed.p1[0])
    total = sum(abs(w.p2[0] - w.p1[0]) + abs(w.p2[1] - w.p1[1]) for w in b1.walls)
    ir = ir.model_copy(update={"walls": [w for w in ir.walls if w.id != "W10"]})
    m = evaluate(ir, b1, thresholds)
    assert m["walls"]["recall"] == pytest.approx(1 - length / total, abs=0.01)
    assert m["walls"]["precision"] == pytest.approx(1)


def test_extra_wall_lowers_precision_not_recall(b1, thresholds):
    ir = ir_from_truth(b1)
    fake = ir.walls[0].model_copy(update={"id": "X", "start": (1000, 1000), "end": (1000, 3000)})
    m = evaluate(ir.model_copy(update={"walls": [*ir.walls, fake]}), b1, thresholds)
    assert m["walls"]["recall"] == pytest.approx(1)
    assert m["walls"]["precision"] < 1
    assert m["confidence"]["high_wrong"] == 1


def test_wall_offset_beyond_tolerance_does_not_match(b1, thresholds):
    ir = ir_from_truth(b1)
    w = ir.walls[0]
    moved = w.model_copy(update={"start": (w.start[0], w.start[1] + 150), "end": (w.end[0], w.end[1] + 150)})
    ir = ir.model_copy(update={"walls": [moved, *ir.walls[1:]]})
    assert evaluate(ir, b1, thresholds)["walls"]["recall"] < 1


def test_perpendicular_wall_does_not_cover(b1, thresholds):
    ir = ir_from_truth(b1)
    w = ir.walls[0]  # (0,0)→(10800,0)
    cross = w.model_copy(update={"start": (5000, -50), "end": (5000, 50)})
    ir = ir.model_copy(update={"walls": [cross, *ir.walls[1:]]})
    assert evaluate(ir, b1, thresholds)["walls"]["recall"] < 0.9


def test_wrong_thickness_and_material_are_counted(b1, thresholds):
    ir = ir_from_truth(b1)
    bad = [w.model_copy(update={"thickness": 150, "kind": "unknown"}) if w.id == "W1" else w
           for w in ir.walls]
    m = evaluate(ir.model_copy(update={"walls": bad}), b1, thresholds)
    assert m["walls"]["recall"] == pytest.approx(1)
    assert m["walls"]["thickness_accuracy"] < 1
    assert m["walls"]["material_accuracy"] < 1


def test_flipped_door_matches_but_direction_fails(b1, thresholds):
    ir = ir_from_truth(b1)
    flipped = [
        o.model_copy(update={"swing_dir": (-o.swing_dir[0], -o.swing_dir[1])}) if o.id == "D1" else o
        for o in ir.openings
    ]
    m = evaluate(ir.model_copy(update={"openings": flipped}), b1, thresholds)["openings"]["doors"]
    assert m["recall"] == 1
    assert m["direction_accuracy"] == pytest.approx(6 / 7)


def test_door_on_wrong_host_wall_does_not_match(b1, thresholds):
    ir = ir_from_truth(b1)
    d1 = next(o for o in ir.openings if o.id == "D1")
    wrong = d1.model_copy(update={"host_wall": "W7", "position": 3550})
    openings = [wrong if o.id == "D1" else o for o in ir.openings]
    m = evaluate(ir.model_copy(update={"openings": openings}), b1, thresholds)["openings"]["doors"]
    assert m["recall"] == pytest.approx(6 / 7)


def test_door_predicted_as_window_is_a_miss_for_both(b1, thresholds):
    ir = ir_from_truth(b1)
    openings = [o.model_copy(update={"type": "window"}) if o.id == "D1" else o for o in ir.openings]
    m = evaluate(ir.model_copy(update={"openings": openings}), b1, thresholds)["openings"]
    assert m["doors"]["recall"] == pytest.approx(6 / 7)
    assert m["windows"]["precision"] < 1


def test_column_size_error_fails_match(a1, thresholds):
    ir = ir_from_truth(a1)
    bad = [c.model_copy(update={"size": (600, 600)}) if c.id == "C0" else c for c in ir.columns]
    m = evaluate(ir.model_copy(update={"columns": bad}), a1, thresholds)["columns"]
    assert m["recall"] == pytest.approx(11 / 12)


def test_column_pairing_is_one_to_one(a1, thresholds):
    ir = ir_from_truth(a1)
    double = ir.columns[0].model_copy(update={"id": "dup"})
    m = evaluate(ir.model_copy(update={"columns": [*ir.columns, double]}), a1, thresholds)["columns"]
    assert m["recall"] == 1
    assert m["precision"] == pytest.approx(12 / 13)


def test_room_with_wrong_area_or_name_fails(b1, thresholds):
    ir = ir_from_truth(b1)
    rooms = [
        r.model_copy(update={"clear_area_m2": r.clear_area_m2 * 1.2}) if "침실1" in r.names else r
        for r in ir.rooms
    ]
    m = evaluate(ir.model_copy(update={"rooms": rooms}), b1, thresholds)["rooms"]
    assert m["recall"] == pytest.approx(7 / 8)

    rooms = [r.model_copy(update={"names": ["미지정"]}) if "침실1" in r.names else r for r in ir.rooms]
    m = evaluate(ir.model_copy(update={"rooms": rooms}), b1, thresholds)["rooms"]
    assert m["recall"] == pytest.approx(7 / 8)


def test_shared_open_room_counts_both_names(b1, thresholds):
    ir = ir_from_truth(b1)
    shared = next(r for r in ir.rooms if "거실" in r.names)
    assert set(shared.names) == {"거실", "주방"}
    assert evaluate(ir, b1, thresholds)["rooms"]["matched"] == 8
