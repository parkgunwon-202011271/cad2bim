"""슬래브, 지붕, 보이드, 계단. 인식과 3D 출력을 함께 본다."""
from __future__ import annotations

import ifcopenshell
import pytest
from shapely.geometry import box

from cad2bim.export.collada import read_meshes, verify_dae, write_dae
from cad2bim.export.floor import floor_outlines, plan_floors
from cad2bim.export.ifc import verify_ifc, write_ifc
from cad2bim.layers import LayerInfo, LayerMap
from cad2bim.preprocess.primitive import Primitive
from cad2bim.recognize.stairs import recognize_stairs
from cad2bim.recognize.voids import recognize_voids
from test_building import EXPORT, plan_ir, wall
from test_collada import storeys_of

STAIRS = {
    "min_tread_mm": 500.0, "min_treads": 3, "min_spacing_mm": 150.0, "max_spacing_mm": 450.0, "spacing_tol_mm": 30.0,
    "min_overlap_share": 0.5, "angle_bin_deg": 1.0, "join_gap_mm": 2500.0,
}
VOIDS = {"min_side_mm": 1000.0, "corner_tol_mm": 50.0}


def line(a, b, layer="X"):
    return Primitive(f"L{a}{b}", "line", {"p1": a, "p2": b}, layer, (f"{a}{b}",))


def layers(**roles):
    return LayerMap({name: LayerInfo(role, "dictionary") for name, role in roles.items()})


def test_x_marked_rectangle_is_a_void():
    prims = [line((0, 0), (4000, 3000)), line((0, 3000), (4000, 0))]
    voids = recognize_voids(prims, layers(X="UNKNOWN"), VOIDS)
    assert len(voids) == 1 and voids[0].bounds == (0, 0, 4000, 3000)


def test_single_diagonal_or_furniture_cross_is_not_a_void():
    assert not recognize_voids([line((0, 0), (4000, 3000))], layers(X="UNKNOWN"), VOIDS)
    cross = [line((0, 0), (4000, 3000)), line((0, 3000), (4000, 0))]
    assert not recognize_voids(cross, layers(X="FURNITURE"), VOIDS)


def test_small_cross_is_not_a_void():
    assert not recognize_voids([line((0, 0), (500, 500)), line((0, 500), (500, 0))], layers(X="UNKNOWN"), VOIDS)


def test_crosses_of_different_rectangles_do_not_pair():
    prims = [line((0, 0), (4000, 3000)), line((0, 4000), (4000, -1000))]
    assert not recognize_voids(prims, layers(X="UNKNOWN"), VOIDS)


def treads(x0, x1, y0, count, spacing=280.0, layer="S"):
    return [line((x0, y0 + i * spacing), (x1, y0 + i * spacing), layer) for i in range(count)]


def test_regular_lines_on_stair_layer_become_a_flight():
    stairs = recognize_stairs(treads(0, 1200, 0, 10), layers(S="STAIR"), STAIRS)
    assert len(stairs) == 1 and len(stairs[0].flights) == 1
    flight = stairs[0].flights[0]
    assert flight.treads == 10 and flight.width == pytest.approx(1200)


def test_two_nearby_flights_are_one_stair_and_far_ones_are_not():
    near = treads(0, 1200, 0, 8) + treads(1400, 2600, 0, 8)
    assert len(recognize_stairs(near, layers(S="STAIR"), STAIRS)) == 1
    far = treads(0, 1200, 0, 8) + treads(20000, 21200, 0, 8)
    assert len(recognize_stairs(far, layers(S="STAIR"), STAIRS)) == 2


def test_other_layers_and_uneven_lines_are_not_stairs():
    assert not recognize_stairs(treads(0, 1200, 0, 10, layer="X"), layers(X="UNKNOWN", S="STAIR"), STAIRS)
    uneven = [line((0, y), (1200, y), "S") for y in (0, 300, 900, 1000, 2000)]
    assert not recognize_stairs(uneven, layers(S="STAIR"), STAIRS)


def test_spur_is_cut_but_a_narrow_link_survives():
    two = plan_ir(columns=[])
    two["walls"] += [
        wall(5, (10000, 0), (30000, 0)), wall(6, (30000, 0), (30000, 5000)), wall(7, (30000, 5000), (10000, 5000)),
    ]
    corridor = {"origin_offset": (0, 0), "columns": [], "rooms": [], "openings": [], "walls": [
        wall(1, (0, 0), (20000, 0)), wall(2, (20000, 0), (20000, 20000)), wall(3, (20000, 20000), (0, 20000)), wall(4, (0, 20000), (0, 0)),
        wall(5, (20000, 9000), (40000, 9000)), wall(6, (20000, 10500), (40000, 10500)),
        wall(7, (40000, 0), (60000, 0)), wall(8, (60000, 0), (60000, 20000)), wall(9, (60000, 20000), (40000, 20000)), wall(10, (40000, 20000), (40000, 0)),
    ]}
    outlines = floor_outlines(corridor, 3000.0, 10.0, trim_mm=1000.0)
    assert len(outlines) == 1, "좁은 복도로 이어진 두 덩어리는 한 슬래브다"


def _with(**kwargs):
    ir = plan_ir(columns=[])
    ir.update(kwargs)
    return ir


CONF = {"level": "review", "score": 0.5, "reasons": []}


def test_void_cuts_the_slab(tmp_path):
    ir = _with(voids=[{"id": "V1", "bounds": (2000, 1000, 5000, 3000), "confidence": CONF}])
    storeys = storeys_of(("F2", ir))
    plan_floors(storeys, EXPORT)
    area = storeys[0].floor.slabs[0].area
    full = floor_outlines(ir, 3000.0, 10.0)[0].area
    assert area == pytest.approx(full - 3000 * 2000, rel=0.01)
    assert storeys[0].floor.slabs[0].interiors


def _stair():
    return {
        "id": "ST1", "bounds": (1000, 1000, 3800, 4000), "confidence": CONF,
        "flights": [
            {"start": (1600, 1500), "end": (1600, 3500), "width": 1200.0, "treads": 8},
            {"start": (3200, 1500), "end": (3200, 3500), "width": 1200.0, "treads": 8},
        ],
    }


def test_stair_cuts_upper_slab_only_and_lowest_slab_stays_whole():
    storeys = storeys_of(("F1", _with(stairs=[_stair()])), ("F2", _with(stairs=[_stair()])))
    plan_floors(storeys, EXPORT)
    assert not storeys[0].floor.slabs[0].interiors
    assert storeys[1].floor.slabs[0].interiors


def test_stair_climbs_exactly_one_storey():
    storeys = storeys_of(("F1", _with(stairs=[_stair()])), ("F2", _with()))
    plan_floors(storeys, EXPORT)
    _, boxes, landings = storeys[0].floor.stairs[0]
    assert len(boxes) == 16 and landings
    tops = [b.z1 for b in boxes]
    assert max(tops) < 3000.0 and max(tops) > 2500.0
    assert all(a <= b for a, b in zip(tops[:8], tops[1:9]))


def test_top_floor_has_no_stair_going_up():
    storeys = storeys_of(("F1", _with()), ("F2", _with(stairs=[_stair()])))
    plan_floors(storeys, EXPORT)
    assert not storeys[1].floor.stairs


def test_roof_covers_only_what_the_upper_floor_does_not():
    big = plan_ir(columns=[])
    small = {"origin_offset": (0, 0), "columns": [], "rooms": [], "openings": [], "walls": [
        wall(1, (0, 0), (5000, 0)), wall(2, (5000, 0), (5000, 5000)), wall(3, (5000, 5000), (0, 5000)), wall(4, (0, 5000), (0, 0)),
    ]}
    storeys = storeys_of(("F1", big), ("F2", small))
    plan_floors(storeys, EXPORT)
    exposed = sum(p.area for p in storeys[0].floor.roofs)
    assert exposed == pytest.approx(5000 * 5200, rel=0.1)
    assert sum(p.area for p in storeys[1].floor.roofs) == pytest.approx(5200 * 5200, rel=0.05)


def test_unaligned_floor_above_does_not_cut_the_roof():
    storeys = storeys_of(("F1", plan_ir(columns=[])), ("F2", plan_ir(columns=[])))
    storeys[1].align = "unaligned"
    plan_floors(storeys, EXPORT)
    assert storeys[0].floor.roofs


def test_outputs_contain_voids_roofs_and_stairs_and_pass_checks(tmp_path):
    ir = _with(
        stairs=[_stair()],
        voids=[{"id": "V1", "bounds": (5500, 1000, 8000, 3000), "confidence": CONF}],
    )
    storeys = storeys_of(("F1", ir), ("F2", _with(stairs=[_stair()])))
    ifc_path = tmp_path / "b.ifc"
    counts = write_ifc(storeys, ifc_path, EXPORT)
    assert counts["stairs"] == 1 and counts["roofs"] >= 1
    check = verify_ifc(ifc_path)
    assert check["schema_errors"] == 0 and check["volume_mismatch"] == 0 and check["geometry_failed"] == 0
    model = ifcopenshell.open(str(ifc_path))
    assert len(model.by_type("IfcStair")) == 1
    dae = tmp_path / "b.dae"
    dae_counts, expected = write_dae(storeys, dae, EXPORT)
    dae_check = verify_dae(dae, expected)
    assert dae_counts["stairs"] == 1 and dae_check["volume_mismatch"] == 0 and dae_check["inside_out"] == 0
    assert read_meshes(dae)["F1_ST1"].volume > 0
