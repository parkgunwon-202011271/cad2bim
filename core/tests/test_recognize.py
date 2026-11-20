"""벽·기둥·레이어 인식. 합성 평면의 관문과, 합성 세트에 없는 경우의 소형 도면."""
from __future__ import annotations

import copy
import json
import math

import pytest
from conftest import DRAWINGS, NAMES, ROOT

from cad2bim.config import load_layer_dictionary
from cad2bim.evaluation import evaluate, load_truth, run_benchmark
from cad2bim.evaluation.benchmark import lookup
from cad2bim.evaluation.metrics import layer_accuracy
from cad2bim.classify import curve_share as _curve_share
from cad2bim.evaluation.selfcheck import guess_type, run_self_check
from cad2bim.layers import DictionaryClassifier, apply_overrides
from cad2bim.pipeline import convert
from cad2bim.preprocess.primitive import Primitive
from cad2bim.recognize import recognize
from cad2bim.recognize.walls import Segment, _drop_repeated_treads, find_pieces

MANIFEST = ROOT / "config" / "benchmarks" / "synthetic.yaml"


def run(cfg, tmp_path, name):
    return convert(DRAWINGS / f"{name}.dxf", tmp_path, cfg, 3000)


@pytest.fixture(scope="module")
def results(cfg, tmp_path_factory):
    return {n: run(cfg, tmp_path_factory.mktemp(n), n) for n in NAMES}


@pytest.fixture(scope="module")
def metrics(results, thresholds):
    return {
        n: evaluate(r.ir, load_truth(DRAWINGS / f"{n}.truth.json"), thresholds)
        for n, r in results.items()
    }


def test_m2_gate_passes(tmp_path):
    gate = run_benchmark(MANIFEST, tmp_path, gate="m2")["gate"]
    assert gate == {"name": "m2", "passed": True, "failures": []}


@pytest.mark.parametrize("name", NAMES)
def test_walls_match_truth_exactly(metrics, name):
    """A2는 관문 밖이지만 기하만으로 같은 결과가 나와야 설계 9번 결정이 성립한다."""
    m = metrics[name]
    assert m["walls"]["recall"] == pytest.approx(1)
    assert m["walls"]["precision"] == pytest.approx(1)
    assert m["walls"]["thickness_accuracy"] == pytest.approx(1)
    assert m["walls"]["material_accuracy"] == pytest.approx(1)
    assert m["junctions"]["100"]["f1"] == pytest.approx(1)
    assert m["topology"]["betti0_error"] == 0
    assert m["topology"]["betti1_error"] == 0


@pytest.mark.parametrize("name", NAMES)
def test_high_confidence_elements_are_correct(metrics, name):
    assert metrics[name]["confidence"]["high_wrong"] == 0


def test_b2_single_line_wall_is_built_and_flagged(results):
    flagged = [w for w in results["B2_apt_partial_code"].ir.walls
               if "single_line_wall" in w.confidence.reasons]
    assert len(flagged) == 1
    wall = flagged[0]
    assert wall.confidence.level == "review"
    assert sorted((wall.start, wall.end)) == [(5400, 6600), (5400, 8400)]


def test_b2_masonry_is_found_from_mislayered_hatch(results):
    kinds = {w.kind for w in results["B2_apt_partial_code"].ir.walls}
    assert kinds == {"concrete", "masonry"}


def test_a2_is_flagged_as_unresolved(results):
    assert "layer_unresolved" in results["A2_retail_messy"].ir.warnings
    assert "layer_unresolved" not in results["A1_retail_clean_en"].ir.warnings


@pytest.mark.parametrize(
    ("name", "expected"), [("A1_retail_clean_en", 12), ("A2_retail_messy", 12),
                           ("B1_apt_clean_kr", 0), ("B2_apt_partial_code", 0)],
)
def test_column_counts(results, name, expected):
    assert len(results[name].ir.columns) == expected


def test_results_keep_source_handles(results):
    for wall in results["A1_retail_clean_en"].ir.walls:
        assert wall.source_handles


def test_layer_map_written_and_scored(results, thresholds):
    for name in ("A1_retail_clean_en", "B1_apt_clean_kr", "B2_apt_partial_code"):
        roles = {k: v.role for k, v in results[name].layers.layers.items()}
        score = layer_accuracy(roles, load_truth(DRAWINGS / f"{name}.truth.json"))
        assert score["accuracy"] == 1


@pytest.mark.parametrize(
    ("layer", "role"),
    [("A-WALL", "WALL"), ("WALL_CON", "WALL"), ("벽체", "WALL"), ("벽체해치", "HATCH"),
     ("A-WALL-PATT", "HATCH"), ("S-COLS", "COLUMN"), ("기둥해치", "HATCH"), ("A-GLAZ", "WINDOW"),
     ("창호", "WINDOW"), ("문", "DOOR"), ("DOOR", "DOOR"), ("중심선-번호", "GRID"), ("0", "UNKNOWN"),
     ("Defpoints", "OTHER"), ("도곽", "OTHER"), ("MY-LAYER", "UNKNOWN")],
)
def test_layer_dictionary(layer, role):
    p = Primitive("P1", "line", {"p1": (0, 0), "p2": (1, 0)}, layer, ("1",))
    assert DictionaryClassifier(load_layer_dictionary()).classify([p]).role(layer) == role


def test_user_override_wins(tmp_path):
    p = Primitive("P1", "line", {"p1": (0, 0), "p2": (1, 0)}, "XX", ("1",))
    layer_map = DictionaryClassifier(load_layer_dictionary()).classify([p])
    path = tmp_path / "layer_map.json"
    path.write_text(json.dumps({"layers": {"XX": {"role": "WALL"}}}), encoding="utf-8")
    updated = apply_overrides(layer_map, path)
    assert updated.role("XX") == "WALL"
    assert updated.layers["XX"].source == "user"


# ---------------------------------------------------------------- 소형 도면
def wall_lines(msp, p1, p2, t=200, layer="A-WALL", gaps=()):
    """p1→p2 벽의 양면. gaps는 시작점에서 잰 (시작, 끝) 틈."""
    (x1, y1), (x2, y2) = p1, p2
    length = math.dist(p1, p2)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    nx, ny = -uy * t / 2, ux * t / 2
    edges = [0.0]
    for a, b in gaps:
        edges += [a, b]
    edges.append(length)
    for a, b in zip(edges[::2], edges[1::2]):
        for sign in (1, -1):
            msp.add_line(
                (x1 + ux * a + sign * nx, y1 + uy * a + sign * ny),
                (x1 + ux * b + sign * nx, y1 + uy * b + sign * ny),
                dxfattribs={"layer": layer},
            )


def noise(msp):
    """평면 후보로 인정될 만큼의 주석."""
    for i in range(25):
        msp.add_text("메모", height=100, dxfattribs={"layer": "NOTE"}).set_placement((i * 300, -2000))


def room(msp, w=6000, h=4000, t=200):
    wall_lines(msp, (0, 0), (w, 0), t)
    wall_lines(msp, (w, 0), (w, h), t)
    wall_lines(msp, (w, h), (0, h), t)
    wall_lines(msp, (0, h), (0, 0), t)


def convert_built(cfg, make_dxf, tmp_path, build):
    return convert(make_dxf(build), tmp_path / "o", cfg, 3000)


def test_simple_room_closes(cfg, make_dxf, tmp_path):
    ir = convert_built(cfg, make_dxf, tmp_path, lambda d, m: (room(m), noise(m))).ir
    assert len(ir.walls) == 4
    assert all(w.confidence.reasons == ["dangling_end"] or w.confidence.reasons == [] for w in ir.walls)
    corners = {p for w in ir.walls for p in (w.start, w.end)}
    assert corners == {(0, 0), (6000, 0), (6000, 4000), (0, 4000)}


def test_door_gap_keeps_one_wall(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0), gaps=[(2000, 2900)])
        wall_lines(msp, (6000, 0), (6000, 4000))
        wall_lines(msp, (6000, 4000), (0, 4000))
        wall_lines(msp, (0, 4000), (0, 0))
        noise(msp)

    assert len(convert_built(cfg, make_dxf, tmp_path, build).ir.walls) == 4


def test_gap_wider_than_limit_stays_two_walls(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        wall_lines(msp, (0, 0), (10000, 0), gaps=[(2000, 7000)])
        wall_lines(msp, (10000, 0), (10000, 4000))
        wall_lines(msp, (10000, 4000), (0, 4000))
        wall_lines(msp, (0, 4000), (0, 0))
        noise(msp)

    walls = convert_built(cfg, make_dxf, tmp_path, build).ir.walls
    assert len([w for w in walls if w.start[1] == 0 and w.end[1] == 0]) == 2


def test_diagonal_wall(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        wall_lines(msp, (0, 0), (3000, 3000))
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert len(ir.walls) == 1
    wall = ir.walls[0]
    assert wall.thickness == pytest.approx(200)
    assert wall.start == pytest.approx((0, 0), abs=1) and wall.end == pytest.approx((3000, 3000), abs=1)


def test_dangling_wall_is_kept_for_review(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        wall_lines(msp, (2000, 1000), (2000, 2500))  # 어디에도 닿지 않음
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    free = [w for w in ir.walls if w.start[0] == pytest.approx(2000)]
    assert len(free) == 1
    assert "dangling_end" in free[0].confidence.reasons
    assert free[0].confidence.level == "review"


def test_window_at_corner_without_column_reaches_corner(cfg, make_dxf, tmp_path):
    """창이 코너에 붙어 벽 조각이 없어도 벽이 코너까지 가야 실이 닫힌다."""
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0), gaps=[(0, 1500)])
        wall_lines(msp, (6000, 0), (6000, 4000))
        wall_lines(msp, (6000, 4000), (0, 4000))
        wall_lines(msp, (0, 4000), (0, 0))
        for y in (-100, -30, 30, 100):
            msp.add_line((100, y), (1500, y), dxfattribs={"layer": "WINDOW"})
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    bottom = [w for w in ir.walls if w.start[1] == 0 and w.end[1] == 0]
    assert len(bottom) == 1
    assert min(bottom[0].start[0], bottom[0].end[0]) == pytest.approx(0)


def test_empty_gap_to_perpendicular_wall_is_not_extended(cfg, make_dxf, tmp_path):
    """틈에 아무것도 그려져 있지 않으면 벽을 지어내지 않는다."""
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0))
        wall_lines(msp, (1500, 0), (1500, 2000))  # 끝이 맞은편 벽까지 2m 비어 있음
        wall_lines(msp, (0, 4000), (6000, 4000))
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    stub = next(w for w in ir.walls if w.start[0] == pytest.approx(1500))
    assert max(stub.start[1], stub.end[1]) == pytest.approx(2000, abs=150)


def test_short_jamb_lines_make_no_fake_wall(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0), gaps=[(2000, 2900)])
        msp.add_line((2000, -100), (2000, 100), dxfattribs={"layer": "A-WALL"})
        msp.add_line((2900, -100), (2900, 100), dxfattribs={"layer": "A-WALL"})
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert [w.thickness for w in ir.walls] == [200]


def test_curved_wall_goes_to_unsupported(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        msp.add_arc((3000, 2000), 1000, 0, 90, dxfattribs={"layer": "A-WALL"})
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert [u.kind for u in ir.unsupported] == ["curved_wall"]


def test_geometric_mode_ignores_stair_treads(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        for e in list(msp):
            e.dxf.layer = "0"
        for i in range(12):
            msp.add_line((1000, 1000 + i * 280), (2500, 1000 + i * 280))
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert "layer_unresolved" in ir.warnings
    assert len(ir.walls) == 4
    assert all("layer_unresolved" in w.confidence.reasons for w in ir.walls)


def test_stair_treads_ending_on_a_rounding_boundary_stay_together(cfg):
    """끝 좌표가 구간 경계(13475 = 269.5 x 50)에 걸려도 계산 오차로 갈라지면 안 된다."""
    rows = []
    for i in range(6):
        noise = 1e-9 if i % 2 else -1e-9
        seg = Segment((12150, 7000 + i * 280), (13475 + noise, 7000 + i * 280), ("h",), "0")
        rows.append((12150.0, 13475.0 + noise, 7000.0 + i * 280, seg))
    assert _drop_repeated_treads(rows, cfg["walls"]) == []


def whole_plan_in_block(doc, msp, layer="0"):
    """외부참조를 결합한 도면처럼 평면 전체가 블록 하나에 든 경우."""
    sheet = doc.blocks.new("XREF_PLAN")
    room(sheet)
    for e in sheet:
        e.dxf.layer = layer
    sofa = doc.blocks.new("SOFA")
    sofa.add_line((0, 0), (1500, 0))
    sofa.add_line((0, 200), (1500, 200))
    sheet.add_blockref("SOFA", (2000, 2000))
    msp.add_blockref("XREF_PLAN", (0, 0))
    noise(msp)


def test_walls_inside_large_block_are_found_without_layers(cfg, make_dxf, tmp_path):
    """실제 도면에서는 요소가 거의 전부 블록 안에 있었다."""
    ir = convert_built(cfg, make_dxf, tmp_path, whole_plan_in_block).ir
    assert "layer_unresolved" in ir.warnings
    assert len(ir.walls) == 4


def test_furniture_block_inside_large_block_is_not_a_wall(cfg, make_dxf, tmp_path):
    ir = convert_built(cfg, make_dxf, tmp_path, whole_plan_in_block).ir
    assert all(w.thickness == 200 for w in ir.walls)


def test_small_block_lines_are_not_wall_candidates(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        for e in list(msp):
            e.dxf.layer = "0"
        part = doc.blocks.new("PART")
        part.add_line((0, 0), (3000, 0))
        part.add_line((0, 200), (3000, 200))
        msp.add_blockref("PART", (1000, 1000))
        noise(msp)

    assert len(convert_built(cfg, make_dxf, tmp_path, build).ir.walls) == 4


def test_column_drawn_as_solid(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        msp.add_solid([(1000, 1000), (1600, 1000), (1000, 1600), (1600, 1600)], dxfattribs={"layer": "0"})
        msp.add_solid([(3000, 1000), (3010, 1000), (3000, 1010), (3010, 1010)], dxfattribs={"layer": "0"})
        noise(msp)

    assert [c.center for c in convert_built(cfg, make_dxf, tmp_path, build).ir.columns] == [(1300, 1300)]


def test_column_drawn_only_as_hatch_on_column_layer(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        pts = [(1000, 1000), (1500, 1000), (1500, 1500), (1000, 1500)]
        msp.add_hatch(dxfattribs={"layer": "S-COLS"}).paths.add_polyline_path(pts, is_closed=True)
        noise(msp)

    assert [c.size for c in convert_built(cfg, make_dxf, tmp_path, build).ir.columns] == [(500, 500)]


def test_outline_and_hatch_make_one_column(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        pts = [(1000, 1000), (1500, 1000), (1500, 1500), (1000, 1500)]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "S-COLS"})
        msp.add_hatch(dxfattribs={"layer": "S-COLS"}).paths.add_polyline_path(pts, is_closed=True)
        noise(msp)

    assert len(convert_built(cfg, make_dxf, tmp_path, build).ir.columns) == 1


def test_column_needs_hatch_outside_column_layer(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        for e in list(msp):
            e.dxf.layer = "0"
        for x, hatched in ((1000, True), (3000, False)):
            pts = [(x, 1000), (x + 500, 1000), (x + 500, 1500), (x, 1500)]
            msp.add_lwpolyline(pts, close=True)
            if hatched:
                msp.add_hatch().paths.add_polyline_path(pts, is_closed=True)
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert [c.center for c in ir.columns] == [(1250, 1250)]


def test_circular_and_rotated_columns(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        msp.add_circle((1500, 1500), 300, dxfattribs={"layer": "S-COLS"})
        c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
        corners = [(-250, -250), (250, -250), (250, 250), (-250, 250)]
        rotated = [(4000 + x * c - y * s, 2000 + x * s + y * c) for x, y in corners]
        msp.add_lwpolyline(rotated, close=True, dxfattribs={"layer": "S-COLS"})
        noise(msp)

    columns = convert_built(cfg, make_dxf, tmp_path, build).ir.columns
    shapes = {c.shape: c for c in columns}
    assert set(shapes) == {"circle", "rect"}
    assert shapes["circle"].size == pytest.approx((600, 600))
    assert shapes["rect"].size == pytest.approx((500, 500))
    assert shapes["rect"].rotation == pytest.approx(math.radians(30))


def test_column_size_limits(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        for x, size in ((500, 100), (2000, 2500)):
            msp.add_lwpolyline(
                [(x, 500), (x + size, 500), (x + size, 500 + size), (x, 500 + size)],
                close=True, dxfattribs={"layer": "S-COLS"},
            )
        noise(msp)

    assert convert_built(cfg, make_dxf, tmp_path, build).ir.columns == []


@pytest.mark.parametrize("name", NAMES)
def test_rotation_finds_the_same_walls(results, cfg, thresholds, name):
    """인식이 방향에 의존하면 사선으로 그린 도면에서 무너진다."""
    result = results[name]
    check = run_self_check(result, cfg, thresholds, load_layer_dictionary()["hatch_patterns"], 0)
    assert check.rotation_recall == pytest.approx(1, abs=0.01)
    assert check.rotation_precision == pytest.approx(1, abs=0.01)


@pytest.mark.parametrize("name", ["A1_retail_clean_en", "B1_apt_clean_kr"])
def test_self_check_agrees_on_clean_drawings(results, cfg, thresholds, name):
    check = run_self_check(results[name], cfg, thresholds, load_layer_dictionary()["hatch_patterns"], 6)
    assert check.masked_recall == pytest.approx(1, abs=0.02)
    assert check.hatch_explained > 0.95
    assert check.wall_layer_coverage > 0.9
    assert check.guess == "plan_like"


def test_self_check_drops_when_a_wall_is_removed(results, cfg, thresholds):
    """벽을 빼면 대조 지표가 떨어져야 한다. 안 떨어지면 지표가 무의미하다."""
    result = results["A1_retail_clean_en"]
    patterns = load_layer_dictionary()["hatch_patterns"]
    before = run_self_check(result, cfg, thresholds, patterns, 6)
    longest = max(result.ir.walls, key=lambda w: abs(w.end[0] - w.start[0]) + abs(w.end[1] - w.start[1]))
    damaged = copy.copy(result)
    damaged.ir = result.ir.model_copy(update={"walls": [w for w in result.ir.walls if w is not longest]})
    after = run_self_check(damaged, cfg, thresholds, patterns, 6)
    assert after.hatch_explained < before.hatch_explained
    assert after.wall_layer_coverage < before.wall_layer_coverage


@pytest.mark.parametrize(
    ("count", "extent", "loops", "dangling", "curve", "expected"),
    [(0, 50, 0, 0, 0, "no_walls"), (44, 700, 3, 30, 0.6, "site_like"), (182, 40, 344, 10, 0, "table_like"),
     (44, 40, 0, 40, 0, "detail_like"), (73, 60, 20, 10, 0, "plan_like"),
     (204, 400, 15, 10, 0.05, "plan_like")],  # 대형 물류센터는 커도 평면이다
)
def test_guess_type(results, count, extent, loops, dangling, curve, expected):
    assert guess_type([None] * count, extent, loops, dangling, curve) == expected


def test_curve_share_separates_contours_from_wall_polylines():
    ring = [(0, 0), (5000, 0), (5000, 200), (9000, 200), (9000, 0), (12000, 0),
            (12000, 8000), (6000, 8000), (6000, 7800), (0, 7800)]
    walls = Primitive("W", "polyline", {"points": ring, "closed": True}, "A", ("1",))
    contour = [(i * 500, 3000 * math.sin(i / 3) + 40000) for i in range(30)]
    site = Primitive("C", "polyline", {"points": contour, "closed": False}, "B", ("2",))
    assert _curve_share([walls]) == 0
    assert _curve_share([walls, site]) > 0.3


def shifted_walls(result, cfg, dx, dy):
    moved = [p.transformed(1.0, dx, dy) for p in result.pre.primitives]
    found = recognize(moved, result.layers, cfg, load_layer_dictionary()["hatch_patterns"])
    return sorted(
        (round(w.start[0] - dx, 1) + 0.0, round(w.start[1] - dy, 1) + 0.0,
         round(w.end[0] - dx, 1) + 0.0, round(w.end[1] - dy, 1) + 0.0, w.thickness)
        for w in found.walls
    )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("shift", [(1000, 0), (37.3, 91.7), (-148000, -123000), (0.1, 0.2)])
def test_moving_the_drawing_does_not_change_walls(results, cfg, name, shift):
    assert shifted_walls(results[name], cfg, *shift) == shifted_walls(results[name], cfg, 0, 0)


@pytest.mark.parametrize("shift", [0.0, 0.1, 0.2, 0.3, 123.456, 148000.7])
def test_thickness_exactly_at_the_limit_is_stable(cfg, shift):
    """부동소수 오차로 경계의 포함 여부가 갈리면 위치에 따라 벽이 생기고 사라진다."""
    limit = cfg["walls"]["max_thickness_mm"]
    segments = [
        Segment((0, shift), (5000, shift), ("a",), "A"),
        Segment((0, shift + limit), (5000, shift + limit), ("b",), "A"),
    ]
    pieces = find_pieces(segments, cfg["walls"], False)
    assert len(pieces) == 1
    assert pieces[0].thickness == pytest.approx(limit)


@pytest.mark.parametrize(
    ("layer", "role"),
    [("PLAN$0$A-WALL", "WALL"), ("PLAN|벽체", "WALL"), ("X$0$CON", "WALL"), ("PLAN$1$S-COLS", "COLUMN"),
     ("PLAN$0$0", "UNKNOWN"), ("PLAN|문", "DOOR"), ("A$0$B$0$창호", "WINDOW")],
)
def test_layer_names_with_xref_prefix(layer, role):
    p = Primitive("P1", "line", {"p1": (0, 0), "p2": (1, 0)}, layer, ("1",))
    assert DictionaryClassifier(load_layer_dictionary()).classify([p]).role(layer) == role


def test_polyline_with_matching_ends_counts_as_closed(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        msp.add_lwpolyline(
            [(1000, 1000), (1500, 1000), (1500, 1500), (1000, 1500), (1000, 1000)], dxfattribs={"layer": "S-COLS"}
        )
        noise(msp)

    assert [c.size for c in convert_built(cfg, make_dxf, tmp_path, build).ir.columns] == [(500, 500)]


def column_grid(msp, spacing, count=4, size=500, layer="0"):
    for i in range(count):
        for j in range(count):
            x, y = 1000 + i * spacing, 1000 + j * spacing
            msp.add_lwpolyline(
                [(x, y), (x + size, y), (x + size, y + size), (x, y + size)], close=True, dxfattribs={"layer": layer}
            )


def test_regular_grid_of_squares_becomes_review_columns(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 40000, 40000)
        column_grid(msp, 9000)
        noise(msp)

    columns = convert_built(cfg, make_dxf, tmp_path, build).ir.columns
    assert len(columns) == 16
    assert all(c.confidence.level == "review" for c in columns)
    assert all("column_inferred" in c.confidence.reasons for c in columns)


def square(msp, x, y, w, d=None, layer="0"):
    d = d or w
    msp.add_lwpolyline([(x, y), (x + w, y), (x + w, y + d), (x, y + d)], close=True, dxfattribs={"layer": layer})


def test_mixed_column_sizes_on_one_grid(cfg, make_dxf, tmp_path):
    """실제 평면 D106. 큰 묶음만 잡히고 같은 격자의 다른 크기 기둥이 빠졌다."""
    def build(doc, msp):
        room(msp, 60000, 60000)
        column_grid(msp, 10500, count=5, size=800)
        for j in range(5):
            cx, cy = 1000 + 400 + 5 * 10500, 1000 + 400 + j * 10500
            square(msp, cx - 450, cy - 450, 900)
        for i in range(3):
            cx, cy = 1000 + 400 + i * 10500, 1000 + 400 + 5 * 10500
            square(msp, cx - 600, cy - 600, 1200)
        noise(msp)

    columns = convert_built(cfg, make_dxf, tmp_path, build).ir.columns
    assert len(columns) == 25 + 5 + 3
    assert all(c.confidence.level == "review" for c in columns)


def test_columns_in_a_single_line_without_full_grid(cfg, make_dxf, tmp_path):
    """실제 평면 D030. 외곽을 따라 줄지어 격자 조건을 통과하지 못했다."""
    def build(doc, msp):
        room(msp, 130000, 40000)
        for i in range(11):
            square(msp, 2000 + i * 11000, 2000, 1000, 1300)
        noise(msp)

    columns = convert_built(cfg, make_dxf, tmp_path, build).ir.columns
    assert len(columns) == 11


def test_column_rectangles_do_not_become_thick_walls(cfg, make_dxf, tmp_path):
    """실제 평면의 두께 1000mm 벽은 1000x1300 기둥과 개수가 같았다."""
    def build(doc, msp):
        room(msp, 130000, 40000)
        for i in range(11):
            square(msp, 2000 + i * 11000, 2000, 1000, 1300, layer="A-WALL")
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert len(ir.columns) == 11
    assert not [w for w in ir.walls if w.thickness >= 1000]


def test_columns_with_missing_bays_still_count_as_a_line(cfg, make_dxf, tmp_path):
    """실제 평면 D030. 빠진 칸 때문에 간격이 11m와 22m로 섞였다."""
    def build(doc, msp):
        room(msp, 130000, 40000)
        x = 2000
        for gap in (11000, 11000, 22000, 11000, 11000, 22000, 11000, 11000):
            square(msp, x, 2000, 1000, 1300)
            x += gap
        square(msp, x, 2000, 1000, 1300)
        noise(msp)

    assert len(convert_built(cfg, make_dxf, tmp_path, build).ir.columns) == 9


def test_small_rectangles_are_not_inferred_as_columns(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 60000, 60000)
        column_grid(msp, 9000, size=250)
        noise(msp)

    assert convert_built(cfg, make_dxf, tmp_path, build).ir.columns == []


def test_small_rectangle_on_a_column_layer_is_still_a_column(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        square(msp, 1000, 1000, 250, layer="S-COLS")
        noise(msp)

    assert len(convert_built(cfg, make_dxf, tmp_path, build).ir.columns) == 1


def test_repeated_rectangles_on_a_wall_layer_are_columns_without_alignment(cfg, make_dxf, tmp_path):
    """실제 평면 D030. 벽 레이어의 같은 크기 사각형 11개가 벽으로 잘못 들어갔다."""
    def build(doc, msp):
        room(msp, 130000, 80000)
        for i in range(11):
            square(msp, 2000 + i * 11000, 2000 + (i % 3) * 1700 + i * 37, 1000, 1300, layer="A-WALL")
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert len(ir.columns) == 11
    assert not [w for w in ir.walls if w.thickness >= 1000]


def test_same_rectangles_on_an_unknown_layer_need_alignment(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 130000, 80000)
        for i in range(11):
            square(msp, 2000 + i * 11000, 2000 + (i % 3) * 1700 + i * 37, 1000, 1300, layer="PLAN_X")
        noise(msp)

    assert convert_built(cfg, make_dxf, tmp_path, build).ir.columns == []


def test_irregular_rectangles_in_a_row_are_not_columns(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 130000, 40000)
        x = 2000
        for gap in (3500, 9000, 3200, 14000, 4100, 20000, 3300, 7000):
            square(msp, x, 2000, 1000, 1300)
            x += gap
        noise(msp)

    assert convert_built(cfg, make_dxf, tmp_path, build).ir.columns == []


def test_closely_spaced_rack_posts_are_not_columns(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 40000, 40000)
        column_grid(msp, 1200)
        noise(msp)

    assert convert_built(cfg, make_dxf, tmp_path, build).ir.columns == []


def test_few_or_scattered_squares_are_not_columns(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 40000, 40000)
        for i, (x, y) in enumerate([(1000, 1000), (7000, 3300), (13100, 8200), (20700, 14900), (27300, 21100),
                                    (31900, 25700), (35100, 30300), (38100, 36100)]):
            msp.add_lwpolyline([(x, y), (x + 500, y), (x + 500, y + 500), (x, y + 500)], close=True,
                               dxfattribs={"layer": "0"})
        noise(msp)

    assert convert_built(cfg, make_dxf, tmp_path, build).ir.columns == []


@pytest.mark.parametrize(
    ("layer", "role"),
    [("A_WIN", "WINDOW"), ("X-WIN-1", "WINDOW"), ("WINTER", "UNKNOWN"), ("A_CEN", "GRID"), ("CENTRAL", "UNKNOWN"),
     ("A_COL", "COLUMN"), ("S-COL1", "COLUMN"), ("COLOR", "UNKNOWN"), ("A_SHUTTER", "DOOR"), ("DOCK-1", "DOOR"),
     ("방화셔터", "DOOR"), ("A_CON", "WALL"), ("CONTOUR", "UNKNOWN"), ("A_BLK", "WALL"), ("샌드위치패널", "WALL"),
     ("문양", "UNKNOWN"), ("창고", "UNKNOWN")],
)
def test_abbreviations_match_only_as_words(layer, role):
    p = Primitive("P1", "line", {"p1": (0, 0), "p2": (1, 0)}, layer, ("1",))
    assert DictionaryClassifier(load_layer_dictionary()).classify([p]).role(layer) == role


def inferred_plan(doc, msp):
    room(msp)  # 사전에 있는 벽 레이어
    wall_lines(msp, (0, 7000), (60000, 7000), layer="PLAN_P1")  # 사전에 없지만 벽처럼 그린 레이어
    wall_lines(msp, (0, 7000), (0, 4000), layer="PLAN_P1")
    for i in range(30):  # 쌍을 이루지 않는 긴 선
        msp.add_line((0, 9000 + i * 2000), (60000, 9000 + i * 2000), dxfattribs={"layer": "PLAN_GRID_X"})
    noise(msp)


def test_unknown_layer_drawn_like_walls_is_promoted_for_review(cfg, make_dxf, tmp_path):
    result = convert_built(cfg, make_dxf, tmp_path, inferred_plan)
    assert result.layers.layers["PLAN_P1"].role == "WALL"
    assert result.layers.layers["PLAN_P1"].source == "inferred"
    assert result.layers.role("PLAN_GRID_X") == "UNKNOWN"
    long_wall = max(result.ir.walls, key=lambda w: abs(w.end[0] - w.start[0]))
    assert abs(long_wall.end[0] - long_wall.start[0]) == pytest.approx(60000)
    assert long_wall.confidence.level == "review"
    assert "wall_layer_inferred" in long_wall.confidence.reasons


def test_layer_with_scattered_pair_spacing_is_not_promoted(cfg, make_dxf, tmp_path):
    """실제 평면에서 두께가 90~860mm로 흩어진 레이어가 벽으로 올라왔다."""
    def build(doc, msp):
        room(msp)
        y = 7000
        for spacing in (120, 210, 330, 470, 90, 280, 410, 160, 370, 240):
            msp.add_line((0, y), (9000, y), dxfattribs={"layer": "PLAN_MISC"})
            msp.add_line((0, y + spacing), (9000, y + spacing), dxfattribs={"layer": "PLAN_MISC"})
            y += spacing + 1500
        noise(msp)

    result = convert_built(cfg, make_dxf, tmp_path, build)
    assert result.layers.role("PLAN_MISC") == "UNKNOWN"


def test_synthetic_drawings_have_no_inferred_layers(results):
    for result in results.values():
        assert not [n for n, i in result.layers.layers.items() if i.source == "inferred"]


# ---------------------------------------------------------------- 문·창·실
def test_m3_gate_passes(tmp_path):
    gate = run_benchmark(MANIFEST, tmp_path, gate="m3")["gate"]
    assert gate == {"name": "m3", "passed": True, "failures": []}


@pytest.mark.parametrize("name", NAMES)
def test_doors_windows_and_rooms_match_truth(metrics, name):
    """A2는 관문 밖이지만 기하만으로도 나와야 한다."""
    m = metrics[name]
    assert m["openings"]["doors"]["f1"] == pytest.approx(1)
    assert m["openings"]["doors"]["direction_accuracy"] == pytest.approx(1)
    assert m["openings"]["windows"]["f1"] == pytest.approx(1)
    assert m["rooms"]["recall"] == pytest.approx(1)
    assert m["rooms"]["iou"]["f1"] == pytest.approx(1)
    assert m["confidence"]["high_wrong"] == 0


def test_shared_living_kitchen_is_one_room_for_review(results):
    rooms = results["B1_apt_clean_kr"].ir.rooms
    shared = next(r for r in rooms if "거실" in r.names)
    assert set(shared.names) == {"거실", "주방"}
    assert shared.confidence.level == "review"
    assert "multiple_room_names" in shared.confidence.reasons


def test_area_note_is_not_taken_as_a_room_name(results):
    for room in results["A1_retail_clean_en"].ir.rooms:
        assert not any("㎡" in name for name in room.names)


def door(msp, hinge, width, start, end, layer="A-DOOR"):
    msp.add_arc(hinge, width, start, end, dxfattribs={"layer": layer})


def door_room(doc, msp):
    wall_lines(msp, (0, 0), (6000, 0), gaps=[(2000, 2900)])
    wall_lines(msp, (6000, 0), (6000, 4000))
    wall_lines(msp, (6000, 4000), (0, 4000))
    wall_lines(msp, (0, 4000), (0, 0))
    door(msp, (2000, 100), 900, 0, 90)
    noise(msp)


def test_swing_door_position_hinge_and_direction(cfg, make_dxf, tmp_path):
    ir = convert_built(cfg, make_dxf, tmp_path, door_room).ir
    doors = [o for o in ir.openings if o.type == "door"]
    assert len(doors) == 1
    d = doors[0]
    assert d.width == pytest.approx(900)
    assert d.hinge == pytest.approx((2000, 100))
    assert d.swing_dir == pytest.approx((0, 1))
    host = next(w for w in ir.walls if w.id == d.host_wall)
    assert host.start[1] == host.end[1] == 0
    assert d.position == pytest.approx(abs(2450 - host.start[0]))
    assert not [o for o in ir.openings if o.type == "unknown"]


def test_double_door_merges_two_arcs(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0), gaps=[(2000, 3800)])
        wall_lines(msp, (6000, 0), (6000, 4000))
        wall_lines(msp, (6000, 4000), (0, 4000))
        wall_lines(msp, (0, 4000), (0, 0))
        door(msp, (2000, 100), 900, 0, 90)
        door(msp, (3800, 100), 900, 90, 180)
        noise(msp)

    doors = [o for o in convert_built(cfg, make_dxf, tmp_path, build).ir.openings if o.type == "door"]
    assert len(doors) == 1
    assert doors[0].operation == "double"
    assert doors[0].width == pytest.approx(1800)


def test_door_arc_stats_show_where_arcs_are_lost(cfg, make_dxf, tmp_path):
    """문 호가 벽 없는 곳에 있으면 벽을 놓쳤다는 근거가 된다."""
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0), gaps=[(2000, 2900)])
        wall_lines(msp, (6000, 0), (6000, 20000))
        wall_lines(msp, (6000, 20000), (0, 20000))
        wall_lines(msp, (0, 20000), (0, 0))
        door(msp, (2000, 100), 900, 0, 90)      # 벽에 걸림
        door(msp, (3000, 10000), 900, 0, 90)    # 벽에서 먼 호
        msp.add_arc((1500, 5000), 900, 0, 90, dxfattribs={"layer": "PLAN_X"})  # 다른 레이어
        noise(msp)

    stats = convert_built(cfg, make_dxf, tmp_path, build).recognition.door_stats
    assert stats["door_layer_arcs"] == 2
    assert stats["door_layer_hosted"] == 1
    assert stats["door_layer_nohost_lt10000"] == 1
    assert stats["unknown_layer_arcs"] == 1


def test_same_center_column_keeps_the_larger_rectangle(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 60000, 60000)
        column_grid(msp, 10500, count=5, size=800)
        for j in range(5):
            cx, cy = 1400 + 5 * 10500, 1400 + j * 10500
            square(msp, cx - 450, cy - 450, 900)
            square(msp, cx - 600, cy - 600, 1200)
        noise(msp)

    columns = convert_built(cfg, make_dxf, tmp_path, build).ir.columns
    assert sorted({max(c.size) for c in columns}) == [800, 1200]


def test_door_arc_with_a_frame_offset_is_hosted_on_door_layer_only(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0), gaps=[(2000, 2900), (4000, 4900)])
        wall_lines(msp, (6000, 0), (6000, 4000))
        wall_lines(msp, (6000, 4000), (0, 4000))
        wall_lines(msp, (0, 4000), (0, 0))
        door(msp, (2000, 250), 900, 0, 90)                 # 문틀만큼 벽 면에서 떨어짐
        door(msp, (4000, 250), 900, 0, 90, layer="PLAN_X")  # 문 레이어가 아니면 믿지 않음
        noise(msp)

    doors = [o for o in convert_built(cfg, make_dxf, tmp_path, build).ir.openings if o.type == "door"]
    assert len(doors) == 1
    assert doors[0].hinge == pytest.approx((2000, 250))


def test_arc_on_korean_window_door_layer_is_a_door(cfg, make_dxf, tmp_path):
    """창호 레이어에는 문도 있다. 실제 도면에서 호 230개가 여기 있었다."""
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0), gaps=[(2000, 2900)])
        wall_lines(msp, (6000, 0), (6000, 4000))
        wall_lines(msp, (6000, 4000), (0, 4000))
        wall_lines(msp, (0, 4000), (0, 0))
        door(msp, (2000, 100), 900, 0, 90, layer="창호")
        noise(msp)

    doors = [o for o in convert_built(cfg, make_dxf, tmp_path, build).ir.openings if o.type == "door"]
    assert len(doors) == 1


def door_in_unlisted_wall(doc, msp):
    room(msp, 20000, 8000)  # 사전에 있는 벽 레이어
    wall_lines(msp, (0, 4000), (9000, 4000), layer="PLAN_P1", gaps=[(4000, 4900)])  # 사전에 없는 레이어의 벽
    door(msp, (4000, 4100), 900, 0, 90)
    noise(msp)


def test_door_arc_finds_a_wall_on_an_unlisted_layer(cfg, make_dxf, tmp_path):
    """길이가 짧아 레이어 추정 기준에 못 미쳐도 문이 근거가 된다."""
    ir = convert_built(cfg, make_dxf, tmp_path, door_in_unlisted_wall).ir
    found = [w for w in ir.walls if "wall_from_door" in w.confidence.reasons]
    assert len(found) == 1
    assert found[0].confidence.level == "review"
    assert found[0].start[1] == pytest.approx(4000) and found[0].end[1] == pytest.approx(4000)
    doors = [o for o in ir.openings if o.type == "door"]
    assert len(doors) == 1 and doors[0].host_wall == found[0].id


def test_wall_from_door_does_not_add_unknown_gaps(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 20000, 8000)
        wall_lines(msp, (0, 4000), (9000, 4000), layer="PLAN_P1", gaps=[(4000, 4900), (6500, 7500)])
        door(msp, (4000, 4100), 900, 0, 90)
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert [w for w in ir.walls if "wall_from_door" in w.confidence.reasons]
    assert not [o for o in ir.openings if o.type == "unknown"]


def test_no_door_means_no_wall_from_unlisted_layer(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 20000, 8000)
        wall_lines(msp, (0, 4000), (9000, 4000), layer="PLAN_P1", gaps=[(4000, 4900)])
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert not [w for w in ir.walls if "wall_from_door" in w.confidence.reasons]
    assert len(ir.walls) == 4


def test_door_arc_does_not_promote_a_perpendicular_or_distant_pair(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 40000, 20000)
        wall_lines(msp, (15000, 2000), (15000, 9000), layer="PLAN_P1")  # 닫힌 방향과 수직
        wall_lines(msp, (30000, 10000), (38000, 10000), layer="PLAN_P2")  # 호에서 멀다
        door(msp, (15000, 5000), 900, 0, 90)  # 수직 조각 위. 닫힌 쪽이 x, 열린 쪽이 y이므로 둘 다 허용되지만
        door(msp, (5000, 10000), 900, 0, 90)
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    found = [w for w in ir.walls if "wall_from_door" in w.confidence.reasons]
    assert not any(w.start[1] == pytest.approx(10000) for w in found)


def test_wall_already_found_is_not_duplicated_by_door_evidence(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 20000, 8000)
        wall_lines(msp, (0, 4000), (9000, 4000), gaps=[(4000, 4900)])
        wall_lines(msp, (0, 4000), (9000, 4000), layer="PLAN_P1", gaps=[(4000, 4900)])
        door(msp, (4000, 4100), 900, 0, 90)
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert not [w for w in ir.walls if "wall_from_door" in w.confidence.reasons]


def test_arc_away_from_walls_is_not_a_door(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        door(msp, (3000, 2000), 900, 0, 90)
        noise(msp)

    assert not [o for o in convert_built(cfg, make_dxf, tmp_path, build).ir.openings if o.type == "door"]


def test_large_arc_is_not_a_door(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        door(msp, (1000, 100), 2500, 0, 90)
        noise(msp)

    assert not [o for o in convert_built(cfg, make_dxf, tmp_path, build).ir.openings if o.type == "door"]


def test_unknown_gap_is_kept_for_review_and_not_filled(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0), gaps=[(2000, 3000)])
        wall_lines(msp, (6000, 0), (6000, 4000))
        wall_lines(msp, (6000, 4000), (0, 4000))
        wall_lines(msp, (0, 4000), (0, 0))
        noise(msp)

    gaps = [o for o in convert_built(cfg, make_dxf, tmp_path, build).ir.openings if o.type == "unknown"]
    assert len(gaps) == 1
    assert gaps[0].width == pytest.approx(1000)
    assert gaps[0].confidence.level == "review"
    assert "unknown_opening" in gaps[0].confidence.reasons


def test_window_lines_in_a_gap_make_a_window(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0), gaps=[(2000, 3500)])
        wall_lines(msp, (6000, 0), (6000, 4000))
        wall_lines(msp, (6000, 4000), (0, 4000))
        wall_lines(msp, (0, 4000), (0, 0))
        for y in (-100, -20, 20, 100):
            msp.add_line((2000, y), (3500, y), dxfattribs={"layer": "WINDOW"})
        noise(msp)

    windows = [o for o in convert_built(cfg, make_dxf, tmp_path, build).ir.openings if o.type == "window"]
    assert len(windows) == 1
    assert windows[0].width == pytest.approx(1500)
    assert windows[0].confidence.level == "high"


def test_column_and_junction_do_not_count_as_gaps(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        wall_lines(msp, (0, 0), (2750, 0))
        wall_lines(msp, (3250, 0), (6000, 0))
        msp.add_lwpolyline([(2750, -250), (3250, -250), (3250, 250), (2750, 250)], close=True, dxfattribs={"layer": "S-COLS"})
        wall_lines(msp, (6000, 0), (6000, 4000))
        wall_lines(msp, (6000, 4000), (0, 4000))
        wall_lines(msp, (0, 4000), (0, 0))
        noise(msp)

    assert [o for o in convert_built(cfg, make_dxf, tmp_path, build).ir.openings if o.type == "unknown"] == []


def text_at(msp, value, x, y, layer="A-ROOM"):
    msp.add_text(value, height=200, dxfattribs={"layer": layer}).set_placement((x, y))


def test_room_area_excludes_walls_and_reads_the_name(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        text_at(msp, "창고", 3000, 2000)
        text_at(msp, "(21.84㎡)", 3000, 1600)
        noise(msp)

    rooms = convert_built(cfg, make_dxf, tmp_path, build).ir.rooms
    assert len(rooms) == 1
    assert rooms[0].names == ["창고"]
    assert rooms[0].clear_area_m2 == pytest.approx(5.8 * 3.8, abs=0.01)
    assert rooms[0].confidence.level == "high"


def test_centerline_area_note_is_accepted(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        text_at(msp, "창고", 3000, 2000)
        text_at(msp, "(24.00㎡)", 3000, 1600)  # 벽 중심선 기준 6m x 4m
        noise(msp)

    room_ = convert_built(cfg, make_dxf, tmp_path, build).ir.rooms[0]
    assert "area_note_mismatch" not in room_.confidence.reasons


def _column_rect(msp, x, y, layer, w=500, h=500):
    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": layer})
    hatch = msp.add_hatch(dxfattribs={"layer": "A-WALL-PATT"})
    hatch.paths.add_polyline_path(pts, is_closed=True)


def test_hatched_wall_stub_is_not_a_column_when_a_column_layer_exists(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 30000, 20000)
        for i in range(5):
            _column_rect(msp, i * 6000, 0, "S-COLS")
        _column_rect(msp, 3000, 10000, "A-WALL", 300, 400)
        noise(msp)

    columns = convert_built(cfg, make_dxf, tmp_path, build).ir.columns
    assert len([c for c in columns if not c.wall_piece]) == 5  # 벽 조각은 벽으로 남고 기둥으로 세지 않는다
    assert len([c for c in columns if c.wall_piece]) == 1


def test_wall_layer_rectangle_can_still_be_a_column_without_a_column_layer(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 30000, 20000)
        _column_rect(msp, 3000, 10000, "A-WALL", 500, 500)
        noise(msp)

    assert len(convert_built(cfg, make_dxf, tmp_path, build).ir.columns) == 1


def _glazed_room(msp, skylight=False):
    w, h = 12000, 6000
    wall_lines(msp, (0, 0), (w, 0))
    wall_lines(msp, (w, 0), (w, h))
    wall_lines(msp, (0, h), (0, 0))
    for offset in (-50, 50):
        msp.add_line((100, h + offset), (w - 100, h + offset), dxfattribs={"layer": "A-GLAZ"})
    text_at(msp, "로비", 6000, 3000)
    if skylight:
        for offset in (0, 100):
            msp.add_line((3000, 3000 + offset), (9000, 3000 + offset), dxfattribs={"layer": "A-GLAZ"})
    noise(msp)


def test_long_glazing_between_walls_becomes_a_curtain_wall_and_closes_the_room(cfg, make_dxf, tmp_path):
    ir = convert_built(cfg, make_dxf, tmp_path, lambda d, m: _glazed_room(m)).ir
    curtain = [w for w in ir.walls if w.kind == "curtain"]
    assert len(curtain) == 1
    assert curtain[0].confidence.level == "review" and "glazed_wall" in curtain[0].confidence.reasons
    assert len(ir.rooms) == 1 and ir.rooms[0].names == ["로비"]
    assert not [o for o in ir.openings if o.type == "window" and o.host_wall == curtain[0].id]


def test_glazing_chain_around_a_corner_is_all_accepted(cfg, make_dxf, tmp_path):
    """모서리에서 유리끼리 이어지면 한쪽 끝만 벽에 닿는다."""
    def build(doc, msp):
        w, h = 12000, 8000
        wall_lines(msp, (0, 0), (0, h))
        wall_lines(msp, (0, h), (w, h))
        wall_lines(msp, (w, h), (w, 3000))
        for offset in (-50, 50):
            msp.add_line((100, offset), (9000, offset), dxfattribs={"layer": "A-GLAZ"})
            msp.add_line((9000 + offset, -offset), (w + offset, 3000 - offset), dxfattribs={"layer": "A-GLAZ"})
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert len([w for w in ir.walls if w.kind == "curtain"]) == 2


def test_four_line_glazing_becomes_one_wall_of_full_width(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        w, h = 12000, 6000
        wall_lines(msp, (0, 0), (w, 0), 300)
        wall_lines(msp, (w, 0), (w, h), 300)
        wall_lines(msp, (0, h), (0, 0), 300)
        for offset in (-150, -20, 20, 150):
            msp.add_line((150, h + offset), (w - 150, h + offset), dxfattribs={"layer": "A-GLAZ"})
        noise(msp)

    curtain = [w for w in convert_built(cfg, make_dxf, tmp_path, build).ir.walls if w.kind == "curtain"]
    assert len(curtain) == 1 and curtain[0].thickness == pytest.approx(300)
    assert curtain[0].start[1] == pytest.approx(6000)


def test_floating_glazing_is_not_a_wall(cfg, make_dxf, tmp_path):
    ir = convert_built(cfg, make_dxf, tmp_path, lambda d, m: _glazed_room(m, skylight=True)).ir
    assert len([w for w in ir.walls if w.kind == "curtain"]) == 1


def test_window_inside_a_wall_stays_an_opening(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0), gaps=[(2000, 3500)])
        wall_lines(msp, (6000, 0), (6000, 4000))
        wall_lines(msp, (6000, 4000), (0, 4000))
        wall_lines(msp, (0, 4000), (0, 0))
        for offset in (-100, 100):
            msp.add_line((2000, offset), (3500, offset), dxfattribs={"layer": "A-GLAZ"})
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert not [w for w in ir.walls if w.kind == "curtain"]
    assert [o for o in ir.openings if o.type == "window"]


@pytest.mark.parametrize("layer", ["A-ROOF", "옥상 난간", "PARAPET"])
def test_roof_layer_is_not_a_wall_even_when_lines_are_paired(cfg, make_dxf, tmp_path, layer):
    def build(doc, msp):
        room(msp)
        for i in range(4):
            y = -20000 + i * 5000
            msp.add_line((0, y), (40000, y), dxfattribs={"layer": layer})
            msp.add_line((0, y + 200), (40000, y + 200), dxfattribs={"layer": layer})
        noise(msp)

    assert all(w.start[1] >= -200 for w in convert_built(cfg, make_dxf, tmp_path, build).ir.walls)


def _panel_room(msp, thickness=50, layer="PANEL", pad=True):
    """세 벽은 벽 레이어, 한 벽은 미분류 레이어의 얇은 평행선이다."""
    w, h = 12000, 8000
    wall_lines(msp, (0, 0), (w, 0))
    wall_lines(msp, (w, 0), (w, h))
    wall_lines(msp, (0, h), (0, 0))
    for offset in (-thickness / 2, thickness / 2):
        msp.add_line((100, h + offset), (w - 100, h + offset), dxfattribs={"layer": layer})
    text_at(msp, "창고", 6000, 4000)
    noise(msp)


def _on(cfg):
    return {**cfg, "walls": {**cfg["walls"], "candidate_enabled": True}}


def test_candidate_walls_are_off_by_default(cfg, make_dxf, tmp_path):
    ir = convert_built(cfg, make_dxf, tmp_path, lambda d, m: _panel_room(m)).ir
    assert not [w for w in ir.walls if "wall_candidate" in w.confidence.reasons]


def test_thin_unknown_lines_that_close_a_room_become_a_review_wall(cfg, make_dxf, tmp_path):
    ir = convert_built(_on(cfg), make_dxf, tmp_path, lambda d, m: _panel_room(m)).ir
    found = [w for w in ir.walls if "wall_candidate" in w.confidence.reasons]
    assert len(found) == 1 and found[0].confidence.level == "review"
    assert len(ir.rooms) == 1


def test_unknown_lines_inside_a_closed_room_are_not_walls(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 12000, 8000)
        for offset in (-25, 25):
            msp.add_line((100, 4000 + offset), (11900, 4000 + offset), dxfattribs={"layer": "PANEL"})
        noise(msp)

    ir = convert_built(_on(cfg), make_dxf, tmp_path, build).ir
    assert not [w for w in ir.walls if "wall_candidate" in w.confidence.reasons]


def test_floating_unknown_pair_is_not_a_wall(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp, 12000, 8000)
        for offset in (-25, 25):
            msp.add_line((20000, 4000 + offset), (30000, 4000 + offset), dxfattribs={"layer": "PANEL"})
        noise(msp)

    ir = convert_built(_on(cfg), make_dxf, tmp_path, build).ir
    assert not [w for w in ir.walls if "wall_candidate" in w.confidence.reasons]


def test_repeated_rows_like_racks_are_not_walls(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        w, h = 30000, 20000
        wall_lines(msp, (0, 0), (w, 0))
        wall_lines(msp, (w, 0), (w, h))
        wall_lines(msp, (w, h), (0, h))
        for i in range(6):
            y = 2000 + i * 3000
            for offset in (-25, 25):
                msp.add_line((100, y + offset), (w - 100, y + offset), dxfattribs={"layer": "RACK"})
        noise(msp)

    ir = convert_built(_on(cfg), make_dxf, tmp_path, build).ir
    assert not [w for w in ir.walls if "wall_candidate" in w.confidence.reasons]


def test_wrong_area_note_sends_room_to_review(cfg, make_dxf, tmp_path):
    def build(doc, msp):
        room(msp)
        text_at(msp, "창고", 3000, 2000)
        text_at(msp, "(40.00㎡)", 3000, 1600)
        noise(msp)

    room_ = convert_built(cfg, make_dxf, tmp_path, build).ir.rooms[0]
    assert room_.confidence.level == "review"
    assert "area_note_mismatch" in room_.confidence.reasons


def test_room_without_name_is_marked_unnamed(cfg, make_dxf, tmp_path):
    room_ = convert_built(cfg, make_dxf, tmp_path, lambda d, m: (room(m), noise(m))).ir.rooms[0]
    assert room_.names == ["미지정"]
    assert "no_room_name" in room_.confidence.reasons


def test_open_room_with_a_wall_gap_does_not_close(cfg, make_dxf, tmp_path):
    """벽이 안 닫히면 실을 지어내지 않는다."""
    def build(doc, msp):
        wall_lines(msp, (0, 0), (6000, 0))
        wall_lines(msp, (6000, 0), (6000, 4000))
        text_at(msp, "복도", 3000, 2000)
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert ir.rooms == []
    assert [u.kind for u in ir.unsupported] == ["open_room"]


def test_report_draws_recognized_walls(results):
    html = results["A1_retail_clean_en"].report_path.read_text(encoding="utf-8")
    assert 'class="wall ok"' in html
    assert 'class="column ok"' in html


def test_lookup_of_stage_metric_in_benchmark(tmp_path):
    record = run_benchmark(MANIFEST, tmp_path)["drawings"][0]
    assert lookup(record["metrics"], "stages.wall_pieces.precision") == pytest.approx(1)
    assert lookup(record["metrics"], "layers.accuracy") == 1


def test_curved_wall_drawn_as_two_arcs_becomes_short_straight_walls(cfg, make_dxf, tmp_path):
    """곡선 벽(안쪽·바깥쪽 호)은 같은 각도에서 끊어 짧은 직선 벽으로 만든다. 짝이 없는 호는 지금처럼 미지원이다."""
    def build(doc, msp):
        room(msp, w=12000, h=9000)
        for radius in (2900, 3100):  # 두께 200의 사분원 벽
            msp.add_arc((6000, 4500), radius, 0, 90, dxfattribs={"layer": "A-WALL"})
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert ir.unsupported == []
    curved = [w for w in ir.walls if all(2800 < math.dist(p, (6000, 4500)) < 3200 for p in (w.start, w.end))]
    assert len(curved) == 6  # 90도를 15도씩
    assert all(abs(w.thickness - 200) <= 5 for w in curved)
    assert sum(math.dist(w.start, w.end) for w in curved) == pytest.approx(math.pi / 2 * 3000, rel=0.05)  # 현이라 호보다 조금 짧다


def test_window_lines_inside_an_unbroken_wall_make_a_window(cfg, make_dxf, tmp_path):
    """유리 외벽을 '벽 두 줄 + 그 사이 유리선'으로 그린 도면. 벽이 끊기지 않아도 유리선이 지나가는 구간이 창이다."""
    def build(doc, msp):
        room(msp)
        for y in (-30, 30):  # 아래 벽(y=0, 두께 200) 안의 유리선 두 줄
            msp.add_line((1500, y), (4500, y), dxfattribs={"layer": "A-GLAZ"})
        msp.add_line((100, 3980), (2000, 3980), dxfattribs={"layer": "A-GLAZ"})  # 한 줄뿐인 선은 창이 아니다
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    windows = [o for o in ir.openings if o.type == "window"]
    assert [(round(o.position), round(o.width)) for o in windows] == [(3000, 3000)]
    assert windows[0].confidence.reasons == ["window_in_wall"] or "window_in_wall" in windows[0].confidence.reasons
