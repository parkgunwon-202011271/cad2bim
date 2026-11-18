"""층 높이 읽기, 난간, 벽 높이. 주기가 없으면 가정값으로 돌아가는지도 본다."""
from __future__ import annotations

import pytest

from cad2bim.building import PlanPart, floor_number, level_plan, select_storeys
from cad2bim.classify.floors import read_level_notes
from cad2bim.export.collada import read_meshes, write_dae
from cad2bim.export.floor import door_top, wall_top
from cad2bim.layers import DictionaryClassifier
from cad2bim.config import load_layer_dictionary
from cad2bim.preprocess.primitive import Primitive
from test_building import EXPORT, opening, plan_ir
from test_collada import storeys_of


def note(value):
    return Primitive("T", "text", {"position": (0, 0), "text": value, "height": 100.0, "rotation": 0.0, "centered": False}, "T", ("1",))


@pytest.mark.parametrize(("text", "key", "mm"), [
    ("- 3층 바닥 EL+26.50 (2층+4.5m)", "F3", 26500.0),
    ("지상 2층 바닥 FL +22.00", "F2", 22000.0),
    ("지하1층 바닥 EL-3.60", "B1", -3600.0),
])
def test_level_note_is_read(text, key, mm):
    assert read_level_notes([note(text)]) == {key: mm}


def test_plain_elevation_text_is_not_a_floor_level():
    assert read_level_notes([note("상부 데크 EL+22.00"), note("3층 평면도")]) == {}


def test_two_levels_give_the_floor_height_and_f1_is_zero():
    elevations, heights, source = level_plan(["F1", "F2", "F3"], {"F2": 22000.0, "F3": 26500.0}, 3000.0)
    assert source == "notes"
    assert elevations == {"F1": 0.0, "F2": 4500.0, "F3": 9000.0}
    assert heights == {"F1": 4500.0, "F2": 4500.0, "F3": 4500.0}


def test_uneven_floors_keep_their_own_height():
    elevations, heights, _ = level_plan(["F1", "F2", "F3"], {"F1": 0.0, "F2": 6000.0, "F3": 10000.0}, 3000.0)
    assert elevations == {"F1": 0.0, "F2": 6000.0, "F3": 10000.0}
    assert heights["F1"] == 6000.0 and heights["F2"] == 4000.0


def test_one_level_is_not_enough_and_falls_back_to_default():
    elevations, _, source = level_plan(["F1", "F2"], {"F2": 22000.0}, 3000.0)
    assert source == "default" and elevations == {"F1": 0.0, "F2": 3000.0}


def test_basement_goes_below_zero():
    assert floor_number("B1") == -1 and floor_number("F1") == 0
    elevations, _, _ = level_plan(["B1", "F1"], {}, 3000.0)
    assert elevations == {"B1": -3000.0, "F1": 0.0}


def test_select_storeys_uses_notes():
    parts = [PlanPart("A", "F2", plan_ir(), 1.0, "title"), PlanPart("B", "F3", plan_ir(), 1.0, "title")]
    storeys = select_storeys(parts, 3000.0, {"F2": 22000.0, "F3": 26500.0})
    assert storeys[1].elevation - storeys[0].elevation == 4500.0 and storeys[0].height == 4500.0


def test_wall_stops_under_the_slab_and_wide_door_is_taller():
    assert wall_top(4500.0, EXPORT) == 4300.0
    assert door_top(900.0, EXPORT) == 2100.0 and door_top(1800.0, EXPORT) == 2400.0


def test_rail_is_written_at_rail_height(tmp_path):
    ir = plan_ir()
    ir["rails"] = [{"id": "RL1", "start": (1000.0, 1000.0), "end": (4000.0, 1000.0), "confidence": {"level": "high", "score": 0.8, "reasons": []}}]
    path = tmp_path / "b.dae"
    counts, _ = write_dae(storeys_of(("F1", ir)), path, EXPORT)
    mesh = read_meshes(path)["F1_RL1"]
    assert counts["rails"] == 1 and mesh.vertices[:, 2].max() == pytest.approx(1200.0)
    assert mesh.volume == pytest.approx(3000 * 100 * 1200)


@pytest.mark.parametrize(("layer", "role"), [
    ("C-PRKG", "OTHER"), ("L-RAIL", "OTHER"), ("A-FLOR-RAIL", "RAIL"), ("A-ROOF", "RAIL"), ("A-ROOF-GLAZ", "OTHER"),
    ("옥상 난간", "RAIL"), ("A-WALL", "WALL"),
])
def test_layer_roles(layer, role):
    prims = [Primitive("L", "line", {"p1": (0, 0), "p2": (1000, 0)}, layer, ("1",))]
    assert DictionaryClassifier(load_layer_dictionary()).classify(prims).role(layer) == role
