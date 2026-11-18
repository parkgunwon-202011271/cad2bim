"""수정용 파일. 파일 하나 안에서 층이 나뉘는지, 블록이 없는지, 검토 표시와 위치가 맞는지 본다."""
from __future__ import annotations

import ezdxf
import pytest

from cad2bim.config import load_defaults
from cad2bim.export.floors import review_names, verify_editable_dxf, write_editable_dxf, write_editable_files
from cad2bim.export.floor import plan_floors
from cad2bim.export.skp import find_sdk, read_editable_skp
from test_building import EXPORT, opening, plan_ir
from test_collada import storeys_of
from test_floor import CONF, _stair, _with

SDK = find_sdk(load_defaults()["export"])
CFG = {**EXPORT, "sketchup_sdk_dir": str(SDK) if SDK else ""}
REVIEW = {"level": "review", "score": 0.4, "reasons": ["wall_layer_inferred"]}


def build(tmp_path, *items):
    storeys = storeys_of(*items)
    return storeys, write_editable_files(storeys, tmp_path, CFG)


def msp_of(tmp_path):
    return ezdxf.readfile(tmp_path / "building_editable.dxf").modelspace()


def test_one_file_holds_all_floors(tmp_path):
    _, total = build(tmp_path, ("F1", plan_ir()), ("F2", plan_ir()))
    assert (tmp_path / "building_editable.dxf").is_file()
    assert total["floors"] == 2 and total["dxf_floor_layers"] == 2
    assert total["dxf_errors"] == 0 and total["dxf_blocks"] == 0 and total["dxf_area_mismatch"] == 0
    assert not list(tmp_path.glob("F1*")) and not list(tmp_path.glob("floors"))


def test_dxf_is_plain_editable_lines_with_height(tmp_path):
    build(tmp_path, ("F1", plan_ir(openings=[opening()])), ("F2", plan_ir()))
    doc = ezdxf.readfile(tmp_path / "building_editable.dxf")
    msp = doc.modelspace()
    assert doc.dxfversion == "AC1018"
    assert not msp.query("INSERT") and not msp.query("3DFACE")
    walls = msp.query('LWPOLYLINE[layer=="F1_Walls"]')
    assert len(walls) == 5  # 문으로 나뉜 벽 하나가 둘이 된다
    assert all(w.closed and w.dxf.thickness == 2800.0 for w in walls)
    assert msp.query('LWPOLYLINE[layer=="F1_Doors"]')


def test_floors_sit_at_their_own_height_on_their_own_layers(tmp_path):
    build(tmp_path, ("F1", plan_ir()), ("F3", plan_ir()))
    msp = msp_of(tmp_path)
    assert {e.dxf.elevation for e in msp.query('LWPOLYLINE[layer=="F1_Walls"]')} == {0.0}
    assert {e.dxf.elevation for e in msp.query('LWPOLYLINE[layer=="F3_Walls"]')} == {6000.0}
    assert {e.dxf.elevation for e in msp.query('LWPOLYLINE[layer=="F3_Slabs"]')} == {5800.0}


def test_shifted_floors_line_up(tmp_path):
    storeys = storeys_of(("F1", plan_ir()), ("F2", plan_ir(82000.0, 300.0)))
    storeys[1].shift = (-82000.0, -300.0)
    write_editable_files(storeys, tmp_path, CFG)
    msp = msp_of(tmp_path)
    a = msp.query('LWPOLYLINE[layer=="F1_Walls"]')[0]
    b = msp.query('LWPOLYLINE[layer=="F2_Walls"]')[0]
    assert [p[:2] for p in a.get_points()] == pytest.approx([p[:2] for p in b.get_points()])


def test_low_confidence_elements_stay_in_on_a_review_layer(tmp_path):
    ir = plan_ir()
    ir["walls"][0]["confidence"] = REVIEW
    assert review_names(ir) == {"W1"}
    build(tmp_path, ("F1", ir))
    msp = msp_of(tmp_path)
    assert len(msp.query('LWPOLYLINE[layer=="F1_REVIEW_Walls"]')) == 1
    assert len(msp.query('LWPOLYLINE[layer=="F1_Walls"]')) == 3


def test_stairs_and_slab_holes_are_written_as_lines(tmp_path):
    ir = _with(stairs=[_stair()], voids=[{"id": "V1", "bounds": (5500, 1000, 8000, 3000), "confidence": CONF}])
    build(tmp_path, ("F1", ir), ("F2", _with(stairs=[_stair()])))
    msp = msp_of(tmp_path)
    assert msp.query('LWPOLYLINE[layer=="F1_REVIEW_Stairs"]')
    assert len(msp.query('LWPOLYLINE[layer=="F1_Slabs"]')) == 2  # 외곽과 보이드 구멍


def test_area_check_catches_a_changed_wall(tmp_path):
    storeys = storeys_of(("F1", plan_ir()))
    plan_floors(storeys, EXPORT)
    path = tmp_path / "f.dxf"
    areas = write_editable_dxf(storeys, path, EXPORT)
    assert verify_editable_dxf(path, areas)["wall_area_mismatch"] == 0
    assert verify_editable_dxf(path, {"F1": areas["F1"] * 2})["wall_area_mismatch"] == 1


def test_room_names_do_not_enter_the_file(tmp_path):
    build(tmp_path, ("F1", plan_ir()))
    assert "SECRET" not in (tmp_path / "building_editable.dxf").read_text(encoding="utf-8", errors="ignore")


@pytest.mark.skipif(SDK is None, reason="SketchUp SDK 없음")
def test_skp_has_one_group_per_floor_with_plain_faces_inside(tmp_path):
    ir = plan_ir(openings=[opening()])
    ir["walls"][0]["confidence"] = REVIEW
    _, total = build(tmp_path, ("F1", ir), ("F2", plan_ir()))
    assert total["skp_floor_groups"] == 2 and total["skp_nested_groups"] == 0 and total["skp_root_faces"] == 0
    assert total["skp_failed"] == 0 and total["skp_faces"] > 0
    data = read_editable_skp(tmp_path / "building_editable.skp", SDK)
    assert set(data["floors"]) == {"F1", "F2"}
    assert {"F1_Walls", "F1_REVIEW_Walls", "F2_Walls", "F1_Columns"} <= set(data["tags"])
    assert set(read_editable_skp(tmp_path / "building_editable_SU5_2005.skp", SDK)["floors"]) == {"F1", "F2"}


def test_glazed_segment_in_editable_dxf_is_wall_pieces_and_glass(tmp_path):
    ir = plan_ir()
    ir["walls"][0]["kind"] = "curtain"
    _, total = build(tmp_path, ("F1", ir))
    assert total["dxf_area_mismatch"] == 0
    msp = msp_of(tmp_path)
    assert msp.query('LWPOLYLINE[layer=="F1_Windows"]')
    assert {e.dxf.elevation for e in msp.query('LWPOLYLINE[layer=="F1_Walls"]')} >= {0.0, 2100.0}

