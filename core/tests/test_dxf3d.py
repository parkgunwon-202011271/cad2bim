"""3D DXF. 파일을 다시 읽어 구조, 닫힌 메시, 부피, 레이어, 숨긴 선, 내용 유출을 본다."""
from __future__ import annotations

import ezdxf
import pytest
from shapely.geometry import box

from cad2bim.export.dxf3d import _hidden_edge_flags, read_block_meshes, verify_dxf, write_dxf
from cad2bim.export.mesh import extrude
from test_building import EXPORT, opening, plan_ir
from test_collada import storeys_of


def write(tmp_path, *items):
    path = tmp_path / "b.dxf"
    counts, expected = write_dxf(storeys_of(*items), path, EXPORT)
    return path, counts, expected


def test_file_passes_audit_and_meshes_are_closed(tmp_path):
    ir = plan_ir(openings=[opening(), opening("window", 7000.0, 1500.0, "W3", "O2"), opening("unknown", 2500.0, 800.0, "W3", "O3")])
    path, counts, expected = write(tmp_path, ("F1", ir), ("F2", plan_ir()))
    assert counts["walls"] == 8 and counts["doors"] == 1 and counts["windows"] == 1
    check = verify_dxf(path, expected)
    assert check["audit_errors"] == 0 and check["not_closed"] == 0 and check["inside_out"] == 0
    assert check["volume_mismatch"] == 0 and check["missing"] == 0 and check["volume_checked"] > 0


def test_door_cuts_the_wall_volume(tmp_path):
    path, _, _ = write(tmp_path, ("F1", plan_ir(openings=[opening(width=1000.0)])))
    assert read_block_meshes(path)["F1_W1"].volume == pytest.approx((10000 * 2800 - 1000 * 2100) * 200, rel=1e-4)


def test_layers_are_floor_and_kind_and_units_are_mm(tmp_path):
    path, _, _ = write(tmp_path, ("F1", plan_ir(openings=[opening()])), ("F2", plan_ir()))
    doc = ezdxf.readfile(path)
    names = {layer.dxf.name for layer in doc.layers}
    assert {"F1_Walls", "F1_Doors", "F1_Columns", "F1_Slabs", "F2_Walls"} <= names
    assert doc.header["$INSUNITS"] == 4
    inserts = doc.modelspace().query("INSERT")
    assert {i.dxf.layer for i in inserts} >= {"F1_Walls", "F2_Walls"}


def test_block_faces_are_on_the_same_layer_as_the_insert(tmp_path):
    path, _, _ = write(tmp_path, ("F1", plan_ir()))
    doc = ezdxf.readfile(path)
    insert = doc.modelspace().query("INSERT")[0]
    assert {face.dxf.layer for face in doc.blocks.get(insert.dxf.name).query("3DFACE")} == {insert.dxf.layer}


def test_box_shows_only_its_twelve_real_edges():
    mesh = extrude(box(0, 0, 2000, 3000), 200)
    visible = set()
    for tri, flags in zip(mesh.triangles.tolist(), _hidden_edge_flags(mesh)):
        for bit, (u, v) in ((1, (tri[0], tri[1])), (2, (tri[1], tri[2])), (8, (tri[2], tri[0]))):
            if not flags & bit:
                visible.add((min(u, v), max(u, v)))
    assert len(visible) == 12


def test_hidden_flags_survive_the_file(tmp_path):
    path, _, _ = write(tmp_path, ("F1", plan_ir()))
    doc = ezdxf.readfile(path)
    insert = doc.modelspace().query("INSERT")[0]
    assert any(face.dxf.invisible_edges & 1 or face.dxf.invisible_edges & 2 or face.dxf.invisible_edges & 8 for face in doc.blocks.get(insert.dxf.name).query("3DFACE"))


def test_old_dxf_format_passes_the_same_checks(tmp_path):
    path = tmp_path / "old.dxf"
    _, expected = write_dxf(storeys_of(("F1", plan_ir(openings=[opening()]))), path, EXPORT, "R2004")
    assert ezdxf.readfile(path).dxfversion == "AC1018"
    check = verify_dxf(path, expected)
    assert check["audit_errors"] == 0 and check["volume_mismatch"] == 0 and check["not_closed"] == 0


def test_room_names_do_not_enter_the_file(tmp_path):
    path, _, _ = write(tmp_path, ("F1", plan_ir()))
    assert "SECRET" not in path.read_text(encoding="utf-8")
