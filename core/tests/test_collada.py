"""스케치업용 COLLADA. 파일을 다시 읽어 닫힌 메시, 부피, 층 구조, 내용 유출을 본다."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest
from shapely.geometry import box

from cad2bim.building import PlanPart, Storey, elevation_of
from cad2bim.export.collada import read_meshes, verify_dae, write_dae
from cad2bim.export.mesh import ELEVATION, extrude, rotation_z
from test_building import EXPORT, opening, plan_ir


def storeys_of(*items):
    return [
        Storey(key, PlanPart("D01", key, ir, 1.0, "title"), elevation_of(key, 3000.0), 3000.0)
        for key, ir in items
    ]


def write(tmp_path, *items):
    path = tmp_path / "b.dae"
    counts, expected = write_dae(storeys_of(*items), path, EXPORT)
    return path, counts, expected


def test_extruded_box_is_closed_with_the_right_volume():
    mesh = extrude(box(0, 0, 2000, 3000), 200)
    assert mesh.open_edges() == 0
    assert mesh.volume == pytest.approx(2000 * 3000 * 200)


def test_hole_keeps_the_mesh_closed():
    wall = box(0, 0, 5000, 3000).difference(box(1000, 900, 2000, 2100))
    mesh = extrude(wall, 200)
    assert mesh.open_edges() == 0
    assert mesh.volume == pytest.approx((5000 * 3000 - 1000 * 1200) * 200)


def test_elevation_and_rotation_do_not_turn_a_mesh_inside_out():
    assert np.linalg.det(ELEVATION) == pytest.approx(1.0)
    mesh = extrude(box(0, 0, 5000, 3000), 200).transformed(ELEVATION, np.zeros(3)).transformed(rotation_z(0.7), np.array([1.0, 2.0, 3.0]))
    assert mesh.volume == pytest.approx(5000 * 3000 * 200)


def test_wall_with_door_and_window_passes_checks(tmp_path):
    ir = plan_ir(openings=[opening(), opening("window", 7000.0, 1500.0, "W3", "O2"), opening("unknown", 2500.0, 800.0, "W3", "O3")])
    path, counts, expected = write(tmp_path, ("F1", ir), ("F2", plan_ir()))
    assert counts["walls"] == 8 and counts["doors"] == 1 and counts["windows"] == 1 and counts["openings"] == 3
    check = verify_dae(path, expected)
    assert check["not_closed"] == 0 and check["inside_out"] == 0 and check["volume_mismatch"] == 0 and check["missing"] == 0
    assert check["volume_checked"] > 0


def test_door_reaches_the_floor_and_cuts_the_wall(tmp_path):
    path, _, expected = write(tmp_path, ("F1", plan_ir(openings=[opening(width=1000.0)])))
    meshes = read_meshes(path)
    assert meshes["F1_W1"].volume == pytest.approx((10000 * 2800 - 1000 * 2100) * 200)
    assert meshes["F1_O1"].vertices[:, 2].min() == pytest.approx(0.0)


def test_unknown_gap_is_cut_but_gets_no_panel(tmp_path):
    path, counts, _ = write(tmp_path, ("F1", plan_ir(openings=[opening("unknown")])))
    assert counts["doors"] == 0 and "F1_O1" not in read_meshes(path)
    assert read_meshes(path)["F1_W1"].volume == pytest.approx((10000 * 2800 - 1000 * 2100) * 200)


def test_full_height_gap_splits_a_wall_into_closed_parts(tmp_path):
    ir = plan_ir(openings=[opening("unknown", 5000.0, 1000.0)])
    path, _, expected = write(tmp_path, ("F1", ir))
    assert verify_dae(path, expected)["not_closed"] == 0


def test_storey_is_placed_at_its_elevation(tmp_path):
    path, _, _ = write(tmp_path, ("F3", plan_ir()))
    meshes = read_meshes(path)
    assert meshes["F3_W1"].vertices[:, 2].min() == pytest.approx(6000.0)
    assert meshes["F3_W1"].vertices[:, 2].max() == pytest.approx(8800.0)
    assert meshes["F3_S1"].vertices[:, 2].max() == pytest.approx(6000.0)


def test_shift_moves_a_floor(tmp_path):
    storeys = storeys_of(("F1", plan_ir()), ("F2", plan_ir(82000.0, 300.0)))
    storeys[1].shift = (-82000.0, -300.0)
    path = tmp_path / "b.dae"
    write_dae(storeys, path, EXPORT)
    meshes = read_meshes(path)
    assert np.allclose(meshes["F1_W1"].vertices.min(axis=0)[:2], meshes["F2_W1"].vertices.min(axis=0)[:2])


def test_file_has_units_up_axis_and_floor_groups(tmp_path):
    path, _, _ = write(tmp_path, ("F1", plan_ir()), ("F2", plan_ir()))
    ns = "{http://www.collada.org/2005/11/COLLADASchema}"
    root = ET.parse(path).getroot()
    assert root.find(f"{ns}asset/{ns}unit").get("meter") == "0.001"
    assert root.find(f"{ns}asset/{ns}up_axis").text == "Z_UP"
    floors = root.find(f"{ns}library_visual_scenes/{ns}visual_scene").findall(f"{ns}node")
    assert [n.get("name") for n in floors] == ["F1", "F2"]
    assert "F1_Walls" in [n.get("name") for n in floors[0].findall(f"{ns}node")]
    tags = [child.tag.replace(ns, "") for child in root]
    assert tags == ["asset", "library_effects", "library_materials", "library_geometries", "library_visual_scenes", "scene"]


def test_room_names_do_not_enter_the_file(tmp_path):
    path, _, _ = write(tmp_path, ("F1", plan_ir()))
    assert "SECRET" not in path.read_text(encoding="utf-8")


def test_independent_reader_opens_the_file(tmp_path):
    collada = pytest.importorskip("collada")
    path, _, _ = write(tmp_path, ("F1", plan_ir(openings=[opening()])))
    document = collada.Collada(str(path))
    assert document.assetInfo.upaxis == "Z_UP"
    triangles = sum(len(primitive) for geometry in document.geometries for primitive in geometry.primitives)
    assert triangles > 0
    assert len(list(document.scene.objects("geometry"))) == len(document.geometries)


def test_glazed_segment_has_wall_below_and_above_with_glass_between(tmp_path):
    ir = plan_ir()
    ir["walls"][0]["kind"] = "curtain"
    path, counts, expected = write(tmp_path, ("F1", ir))
    meshes = read_meshes(path)
    assert counts["glazed_bands"] == 1
    assert meshes["F1_W1_1"].vertices[:, 2].max() <= 2800.0 and "F1_W1_2" in meshes
    zs = sorted((meshes["F1_W1_1"].vertices[:, 2].min(), meshes["F1_W1_2"].vertices[:, 2].min()))
    assert zs[0] == pytest.approx(0.0) and zs[1] == pytest.approx(2100.0)
    glass = meshes["F1_W1_G1"]
    assert glass.vertices[:, 2].min() == pytest.approx(900.0) and glass.vertices[:, 2].max() == pytest.approx(2100.0)
    assert verify_dae(path, expected)["not_closed"] == 0 and verify_dae(path, expected)["volume_mismatch"] == 0


def test_door_on_a_glazed_segment_interrupts_the_glass(tmp_path):
    ir = plan_ir(openings=[opening(position=5000.0, width=1000.0)])
    ir["walls"][0]["kind"] = "curtain"
    path, _, expected = write(tmp_path, ("F1", ir))
    meshes = read_meshes(path)
    assert "F1_W1_G1" in meshes and "F1_W1_G2" in meshes
    assert verify_dae(path, expected)["volume_mismatch"] == 0

