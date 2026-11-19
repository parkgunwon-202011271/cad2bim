"""건물 합치기와 IFC. 층 판단, 위치 맞춤, 3D 출력, 내용 유출을 본다."""
from __future__ import annotations

import ezdxf
import ifcopenshell
import numpy as np
import pytest
from conftest import add_dimensions

from cad2bim.building import PlanPart, Storey, align_storeys, elevation_of, run_build, select_storeys
from cad2bim.evaluation.audit import column_shift
from cad2bim.export.ifc import verify_ifc, write_ifc

EXPORT = {
    "door_height_mm": 2100.0, "wide_door_height_mm": 2400.0, "wide_door_min_width_mm": 1500.0,
    "window_sill_mm": 900.0, "window_height_mm": 1200.0, "rail_height_mm": 1200.0, "rail_thickness_mm": 100.0,
    "slab_thickness_mm": 200.0, "slab_close_mm": 3000.0, "slab_min_area_m2": 10.0, "slab_trim_mm": 1000.0,
    "landing_thickness_mm": 200.0,
    "stair_landing_min_mm": 900.0, "stair_landing_max_mm": 2500.0, "stair_access_probe_mm": 600.0,
    "stair_access_min_share": 0.5,
}
CONF = {"level": "high", "score": 1.0, "reasons": []}


def wall(index, a, b, t=200.0):
    return {"id": f"W{index}", "start": a, "end": b, "thickness": t, "kind": "unknown", "confidence": CONF, "source_handles": []}


def column(index, x, y):
    return {"id": f"C{index}", "shape": "rect", "center": (x, y), "size": (500.0, 500.0), "rotation": 0.0, "confidence": CONF}


def plan_ir(dx=0.0, dy=0.0, columns=None, openings=()):
    walls = [
        wall(1, (dx, dy), (10000 + dx, dy)), wall(2, (10000 + dx, dy), (10000 + dx, 5000 + dy)),
        wall(3, (10000 + dx, 5000 + dy), (dx, 5000 + dy)), wall(4, (dx, 5000 + dy), (dx, dy)),
    ]
    grid = columns if columns is not None else [(i * 7000.0 + dx, j * 9000.0 + dy) for i in range(4) for j in range(3)]
    return {
        "origin_offset": (0.0, 0.0), "walls": walls, "columns": [column(i, x, y) for i, (x, y) in enumerate(grid)],
        "openings": list(openings),
        "rooms": [{"id": "R1", "names": ["SECRET실"], "polygon": [(100 + dx, 100 + dy), (9900 + dx, 100 + dy), (9900 + dx, 4900 + dy), (100 + dx, 4900 + dy)]}],
    }


def part(floor, ir=None, length=100.0, part_id="D01"):
    return PlanPart(part_id, floor, ir or plan_ir(), length, "title")


def opening(kind="door", position=5000.0, width=1000.0, host="W1", id_="O1"):
    return {"id": id_, "type": kind, "host_wall": host, "position": position, "width": width}


@pytest.mark.parametrize(("key", "z"), [("F1", 0.0), ("F2", 3000.0), ("F4", 9000.0), ("B1", -3000.0), ("B2", -6000.0)])
def test_elevation_follows_floor_number(key, z):
    assert elevation_of(key, 3000.0) == z


def test_longest_wall_plan_represents_a_floor():
    storeys = select_storeys([part("F1", length=5.0, part_id="A"), part("F1", length=9.0, part_id="B"), part("F2")], 3000.0)
    assert [s.key for s in storeys] == ["F1", "F2"]
    assert storeys[0].part.part_id == "B" and storeys[0].alternatives == 1


def test_plan_with_doors_and_rooms_beats_a_longer_plan_without_them():
    bare = plan_ir()
    bare["rooms"] = []
    rich = plan_ir(openings=[opening()])
    storeys = select_storeys([part("F1", bare, 99.0, "A"), part("F1", rich, 5.0, "B")], 3000.0)
    assert storeys[0].part.part_id == "B"


def test_grid_lines_keep_only_long_lines():
    from cad2bim.building import grid_lines
    from cad2bim.preprocess.primitive import Primitive

    def line(x0, y0, x1, y1, layer="S-GRID"):
        return Primitive(f"L{x0}{y0}{x1}{y1}", "line", {"p1": (x0, y0), "p2": (x1, y1)}, layer, ("1",))

    model = [
        line(0, 0, 0, 20000), line(8000, 0, 8000, 20000), line(4000, 0, 4000, 300),
        line(0, 0, 16000, 0), line(0, 10000, 16000, 10000), line(0, 500, 200, 500),
        line(12000, 0, 12000, 20000, "A-WALL"),
    ]
    ir = {"source": {"unit_scale": 1.0, "region": None}}
    xs, ys = grid_lines(model, {"S-GRID": "GRID", "A-WALL": "WALL"}, ir)
    assert xs.tolist() == [0.0, 8000.0] and ys.tolist() == [0.0, 10000.0]


def _grid_part(floor, xs, ys, part_id):
    return PlanPart(part_id, floor, plan_ir(columns=[(0.0, 0.0)]), 1.0, "title", (np.array(xs, float), np.array(ys, float)))


def test_floors_link_through_the_best_matching_neighbour():
    """F1과 F3은 직접 안 맞아도 F2를 거쳐 이어진다. 기준 층 하나에 묶이면 떨어져 나갈 경우다."""
    base = [0.0, 8000.0, 17000.0]
    wide = [0.0, 8000.0, 17000.0, 29000.0, 35000.0, 47000.0]
    far = [29000.0, 35000.0, 47000.0]
    storeys = [
        Storey("F1", _grid_part("F1", base, base, "A"), 0.0, 3000.0),
        Storey("F2", _grid_part("F2", [v + 1000 for v in wide], [v + 1000 for v in wide], "B"), 3000.0, 3000.0),
        Storey("F3", _grid_part("F3", [v + 5000 for v in far], [v + 5000 for v in far], "C"), 6000.0, 3000.0),
    ]
    align_storeys(storeys)
    assert all(s.align != "unaligned" for s in storeys)
    first, second, third = storeys
    assert first.part.grid[0][0] + first.shift[0] == pytest.approx(second.part.grid[0][0] + second.shift[0])
    assert third.part.grid[0][0] + third.shift[0] == pytest.approx(second.part.grid[0][3] + second.shift[0])


def test_unaligned_floor_retries_with_its_other_plan():
    base = [0.0, 8000.0, 17000.0]
    other = [0.0, 3100.0, 11300.0]
    longest = _grid_part("F3", other, other, "C")
    second = _grid_part("F3", [v + 700 for v in base], base, "D")
    storeys = [
        Storey("F1", _grid_part("F1", base, base, "A"), 0.0, 3000.0),
        Storey("F2", _grid_part("F2", base, base, "B"), 3000.0, 3000.0),
        Storey("F3", longest, 6000.0, 3000.0, others=[second]),
    ]
    align_storeys(storeys)
    assert storeys[2].part is second and storeys[2].swapped and storeys[2].align == "grid"
    assert storeys[2].shift[0] == pytest.approx(storeys[0].shift[0] - 700.0)


def test_unaligned_floor_is_brought_near_the_building():
    near_columns = [(i * 7100.0 + 300.0, 900.0) for i in range(2)]
    base = plan_ir()
    storeys = [
        Storey("F1", PlanPart("A", "F1", base, 1.0, "title"), 0.0, 3000.0),
        Storey("F2", PlanPart("B", "F2", plan_ir(3000000.0, 500000.0, columns=near_columns), 1.0, "title"), 3000.0, 3000.0),
    ]
    align_storeys(storeys)
    moved = storeys[1]
    assert moved.align == "unaligned"
    x = moved.part.ir["walls"][0]["start"][0] + moved.shift[0]
    assert abs(x) < 50000.0


def test_unrelated_floor_stays_unaligned():
    base = [0.0, 8000.0, 17000.0]
    other = [0.0, 3100.0, 11300.0]
    storeys = [
        Storey("F1", _grid_part("F1", base, base, "A"), 0.0, 3000.0),
        Storey("F2", _grid_part("F2", base, base, "B"), 3000.0, 3000.0),
        Storey("F3", _grid_part("F3", other, other, "C"), 6000.0, 3000.0),
    ]
    align_storeys(storeys)
    assert storeys[2].align == "unaligned"


def test_plan_whose_file_name_agrees_beats_a_longer_plan_from_another_floor_file():
    other_file = PlanPart("D03.3", "F2", plan_ir(openings=[opening()]), 90.0, "title", None, False)
    own_file = PlanPart("D01", "F2", plan_ir(openings=[opening()]), 50.0, "title", None, True)
    assert select_storeys([other_file, own_file], 3000.0)[0].part.part_id == "D01"


def test_partial_plan_with_many_doors_does_not_beat_the_full_plan():
    full = plan_ir(openings=[opening()])
    partial = plan_ir(openings=[opening(id_=f"O{i}", position=1000.0 + i * 1500) for i in range(5)])
    storeys = select_storeys([part("F1", full, 90.0, "A"), part("F1", partial, 10.0, "B")], 3000.0)
    assert storeys[0].part.part_id == "A"


def test_column_shift_direction_is_a_to_b():
    a = np.array([(i * 7000.0, j * 9000.0) for i in range(4) for j in range(3)])
    b = a + (250000.0, 1000.0)
    assert column_shift(a, b)[0] == pytest.approx((250000.0, 1000.0))
    assert column_shift(a[:6], b)[0] == pytest.approx((250000.0, 1000.0))
    assert column_shift(b, a[:6])[0] == pytest.approx((-250000.0, -1000.0))


def test_floors_on_different_sheets_are_moved_together():
    storeys = [Storey("F1", part("F1"), 0.0, 3000.0), Storey("F2", part("F2", plan_ir(250000.0, 1000.0), part_id="D02"), 3000.0, 3000.0)]
    align_storeys(storeys)
    assert {s.align for s in storeys} == {"reference", "columns"}
    first, second = storeys
    assert np.allclose(
        np.array(first.part.ir["columns"][0]["center"]) + first.shift,
        np.array(second.part.ir["columns"][0]["center"]) + second.shift,
    )


def test_grid_lines_align_floors_even_when_columns_differ():
    from cad2bim.building import _axis_shift

    xs = np.array([0.0, 7000.0, 15000.0, 22000.0])
    ys = np.array([0.0, 9000.0, 18000.0])
    near = plan_ir(columns=[(0.0, 0.0)])
    far = plan_ir(82000.0, 300.0, columns=[(1.0, 2.0)])
    first = PlanPart("D01", "F1", near, 1.0, "title", (xs, ys))
    second = PlanPart("D02", "F2", far, 1.0, "title", (xs + 82000.0, ys + 300.0))
    storeys = [Storey("F1", first, 0.0, 3000.0), Storey("F2", second, 3000.0, 3000.0)]
    align_storeys(storeys)
    moved = next(s for s in storeys if s.align == "grid")
    reference = next(s for s in storeys if s is not moved)
    assert np.allclose(moved.part.grid[0] + moved.shift[0], reference.part.grid[0])
    assert np.allclose(moved.part.grid[1] + moved.shift[1], reference.part.grid[1])
    assert _axis_shift(xs[:1], xs) == (0.0, 0.0)


def test_regular_grid_does_not_slip_one_bay():
    from cad2bim.building import _axis_shift

    regular = np.arange(6) * 8000.0
    shift, share = _axis_shift(regular + 500.0, regular)
    assert shift == pytest.approx(-500.0) and share == 1.0


def test_same_origin_is_not_moved():
    storeys = [Storey("F1", part("F1"), 0.0, 3000.0), Storey("F2", part("F2", part_id="D02"), 3000.0, 3000.0)]
    align_storeys(storeys)
    assert all(s.shift == (0.0, 0.0) for s in storeys)
    assert {s.align for s in storeys} == {"reference", "same_origin"}


def test_too_few_columns_is_reported_not_guessed():
    few = plan_ir(5000.0, 0.0, columns=[(0.0, 0.0), (7000.0, 0.0)])
    storeys = [Storey("F1", part("F1"), 0.0, 3000.0), Storey("F2", part("F2", few, part_id="D02"), 3000.0, 3000.0)]
    align_storeys(storeys)
    assert {s.align for s in storeys} == {"reference", "unaligned"}


def build(tmp_path, irs):
    path = tmp_path / "b.ifc"
    storeys = [Storey(k, part(k, ir), elevation_of(k, 3000.0), 3000.0) for k, ir in irs]
    return path, write_ifc(storeys, path, EXPORT)


def test_ifc_has_storeys_elements_and_passes_checks(tmp_path):
    ir = plan_ir(openings=[opening(), opening("window", 7000.0, 1500.0, "W3", "O2"), opening("unknown", 2500.0, 800.0, "W3", "O3")])
    path, counts = build(tmp_path, [("F1", ir), ("F2", plan_ir())])
    assert counts["walls"] == 8 and counts["doors"] == 1 and counts["windows"] == 1 and counts["openings"] == 3
    model = ifcopenshell.open(str(path))
    assert sorted(s.Elevation for s in model.by_type("IfcBuildingStorey")) == [0.0, 3000.0]
    assert len(model.by_type("IfcSpace")) == 2 and len(model.by_type("IfcColumn")) == 24
    check = verify_ifc(path)
    assert check["schema_errors"] == 0 and check["volume_mismatch"] == 0 and check["geometry_failed"] == 0
    assert check["volume_checked"] == 8


def test_slab_covers_the_floor_below_the_level(tmp_path):
    path, counts = build(tmp_path, [("F2", plan_ir())])
    assert counts["slabs"] == 1
    model = ifcopenshell.open(str(path))
    slab = model.by_type("IfcSlab")[0]
    assert slab.ObjectPlacement.RelativePlacement.Location.Coordinates[2] == -200.0
    assert verify_ifc(path)["schema_errors"] == 0


def test_slab_has_square_corners_and_no_spur():
    from cad2bim.export.ifc import floor_outlines

    ir = plan_ir(columns=[])
    ir["walls"].append(wall(5, (10000, 2500), (40000, 2500)))
    outlines = floor_outlines(ir, 3000.0, 10.0)
    assert len(outlines) == 1
    minx, miny, maxx, maxy = outlines[0].bounds
    assert maxx < 11000.0 and outlines[0].area == pytest.approx(10200 * 5200, rel=0.02)


def test_glazed_segment_is_a_wall_with_a_band_window_in_ifc(tmp_path):
    ir = plan_ir()
    ir["walls"][0]["kind"] = "curtain"
    path, counts = build(tmp_path, [("F1", ir)])
    assert counts["glazed_bands"] == 1 and counts["walls"] == 4
    model = ifcopenshell.open(str(path))
    assert len(model.by_type("IfcWindow")) == 1 and not model.by_type("IfcCurtainWall")
    assert verify_ifc(path)["volume_mismatch"] == 0


def test_door_cuts_the_expected_volume(tmp_path):
    path, _ = build(tmp_path, [("F1", plan_ir(openings=[opening(width=1000.0)]))])
    assert verify_ifc(path)["volume_mismatch"] == 0


def test_unknown_gap_is_cut_but_gets_no_door(tmp_path):
    path, counts = build(tmp_path, [("F1", plan_ir(openings=[opening("unknown")]))])
    model = ifcopenshell.open(str(path))
    assert counts["openings"] == 1 and not model.by_type("IfcDoor") and len(model.by_type("IfcOpeningElement")) == 1


def test_opening_outside_its_wall_is_skipped(tmp_path):
    _, counts = build(tmp_path, [("F1", plan_ir(openings=[opening(position=9900.0, width=1000.0)]))])
    assert counts["skipped_opening"] == 1 and counts["openings"] == 0


def closed(msp, x0, y0, x1, y1, layer="A-WALL"):
    msp.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], close=True, dxfattribs={"layer": layer})


def plan_dxf(path, title, dx=0.0):
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()
    for layer in ("A-WALL",):
        doc.layers.add(layer)
    for x0, y0, x1, y1 in (
        (0, 0, 20000, 200), (0, 9800, 20000, 10000), (0, 0, 200, 10000), (19800, 0, 20000, 10000), (9900, 200, 10100, 9800),
    ):
        msp.add_line((x0 + dx, y0), (x1 + dx, y0), dxfattribs={"layer": "A-WALL"})
        msp.add_line((x0 + dx, y1), (x1 + dx, y1), dxfattribs={"layer": "A-WALL"})
        msp.add_line((x0 + dx, y0), (x0 + dx, y1), dxfattribs={"layer": "A-WALL"})
        msp.add_line((x1 + dx, y0), (x1 + dx, y1), dxfattribs={"layer": "A-WALL"})
    for i in range(25):
        msp.add_text("메모", height=100).set_placement((i * 300 + dx, -2000))
    msp.add_text(title, height=400).set_placement((5000 + dx, -3000))
    add_dimensions(msp, y=-1000)
    doc.saveas(path)


def test_build_command_prints_numbers_only(tmp_path, capsys):
    plan_dxf(tmp_path / "SECRET 파일A.dxf", "SECRET 지상 1층 평면도")
    plan_dxf(tmp_path / "SECRET 파일B.dxf", "SECRET 지상 2층 평면도")
    from cad2bim.cli import main

    assert main(["build", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "storeys=2" in output and "floor[F1]" in output and "floor[F2]" in output
    assert "schema_errors=0" in output and "volume_mismatch=0" in output and "not_closed=0" in output
    assert (tmp_path / "_cad2bim_out" / "building.dae").is_file()
    assert "SECRET" not in (tmp_path / "_cad2bim_out" / "building.dxf").read_text(encoding="utf-8")
    assert "audit_errors=0" in output
    assert "SECRET" not in (tmp_path / "_cad2bim_out" / "building.dae").read_text(encoding="utf-8")
    assert "SECRET" not in output and "평면도" not in output and "파일" not in output
    model = ifcopenshell.open(str(tmp_path / "_cad2bim_out" / "building.ifc"))
    assert len(model.by_type("IfcBuildingStorey")) == 2
    assert "SECRET" not in (tmp_path / "_cad2bim_out" / "building.json").read_text(encoding="utf-8")


def test_build_without_floor_information_fails_clearly(tmp_path, capsys):
    plan_dxf(tmp_path / "이름없음.dxf", "제목")
    from cad2bim.cli import main

    assert main(["build", str(tmp_path)]) == 2
    assert "층을 정할 수" in capsys.readouterr().err


def test_a_file_with_every_floor_wins_over_single_floor_files():
    """주민센터: B안(최종) 한 파일에 1·2·3층이 다 있고, 따로 있는 2·3층 파일은 다른 버전이었다. 버전이 섞이지 않게 한 벌을 쓴다."""
    from dataclasses import replace

    def p(floor, pid, length, agrees):
        return replace(part(floor, length=length, part_id=pid), file_agrees=agrees)

    parts = [p("F1", "D03.2", 100.0, False), p("F2", "D03.1", 100.0, False), p("F3", "D03.3", 100.0, False),
             p("F2", "D01", 500.0, True), p("F3", "D02", 500.0, True)]
    chosen = {s.key: s.part.part_id for s in select_storeys(parts, 3000.0)}
    assert chosen == {"F1": "D03.2", "F2": "D03.1", "F3": "D03.3"}
    # 모든 층을 담은 파일이 없으면 예전처럼 층 파일을 먼저 쓴다
    chosen = {s.key: s.part.part_id for s in select_storeys(parts[:2] + parts[3:], 3000.0)}
    assert chosen["F2"] == "D01" and chosen["F3"] == "D02"
