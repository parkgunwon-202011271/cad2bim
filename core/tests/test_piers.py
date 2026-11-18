"""비정형 기둥과 짧은 벽 조각. 덮이지 않은 닫힌 도형만 살리는지, 출력까지 이어지는지 본다."""
from __future__ import annotations

import ezdxf
import pytest
from shapely.geometry import Polygon

from cad2bim.config import load_defaults
from cad2bim.export.collada import read_meshes, verify_dae, write_dae
from cad2bim.export.ifc import verify_ifc, write_ifc
from cad2bim.layers import LayerInfo, LayerMap
from cad2bim.preprocess.primitive import Primitive
from cad2bim.recognize.columns import ColumnDraft
from cad2bim.recognize.piers import find_piers
from cad2bim.recognize.walls import WallDraft
from test_building import EXPORT, plan_ir
from test_collada import storeys_of
import numpy as np

CFG = load_defaults()["walls"]
PENTAGON = [(0.0, 0.0), (600.0, 0.0), (900.0, 300.0), (900.0, 600.0), (0.0, 600.0)]


def closed(points, layer="A-WALL", name="P"):
    return Primitive(name, "polyline", {"points": points, "closed": True}, layer, (name,))


def layers():
    return LayerMap({"A-WALL": LayerInfo("WALL", "dictionary"), "S-COLS": LayerInfo("COLUMN", "dictionary"), "A-FURN": LayerInfo("FURNITURE", "dictionary")})


def wall(a, b, t=300.0):
    return WallDraft(np.array(a, float), np.array(b, float), t, "masonry", {"h"}, {"A-WALL"}, 0.0)


def test_irregular_shape_becomes_a_pier_with_its_own_outline():
    piers = find_piers([closed(PENTAGON)], layers(), [], [], CFG)
    assert len(piers) == 1 and piers[0].shape == "polygon"
    assert piers[0].polygon.area == pytest.approx(Polygon(PENTAGON).area)


def test_short_wall_stub_becomes_a_pier():
    stub = [(0.0, 0.0), (150.0, 0.0), (150.0, 300.0), (0.0, 300.0)]
    assert len(find_piers([closed(stub)], layers(), [], [], CFG)) == 1


def test_shape_already_covered_by_a_wall_is_skipped():
    inside = [(1000.0, -100.0), (2000.0, -100.0), (2000.0, 100.0), (1000.0, 100.0)]
    assert not find_piers([closed(inside)], layers(), [wall((0.0, 0.0), (5000.0, 0.0))], [], CFG)


def test_large_room_outline_and_other_layers_are_skipped():
    room = [(0.0, 0.0), (6000.0, 0.0), (6000.0, 4000.0), (0.0, 4000.0)]
    assert not find_piers([closed(room)], layers(), [], [], CFG)
    assert not find_piers([closed(PENTAGON, "A-FURN")], layers(), [], [], CFG)


def test_nested_outlines_make_one_pier():
    inner = [(100.0, 100.0), (500.0, 100.0), (500.0, 500.0), (100.0, 500.0)]
    assert len(find_piers([closed(PENTAGON, name="A"), closed(inner, "S-COLS", "B")], layers(), [], [], CFG)) == 1


def pier_ir():
    ir = plan_ir(columns=[])
    ir["columns"] = [{
        "id": "C9", "shape": "polygon", "center": (450.0, 300.0), "size": (900.0, 600.0), "rotation": 0.0, "polygon": PENTAGON,
        "confidence": {"level": "high", "score": 1.0, "reasons": []},
    }]
    return ir


def test_pier_is_written_in_every_output(tmp_path):
    dae = tmp_path / "b.dae"
    _, expected = write_dae(storeys_of(("F1", pier_ir())), dae, EXPORT)
    assert verify_dae(dae, expected)["volume_mismatch"] == 0
    assert read_meshes(dae)["F1_C9"].volume == pytest.approx(Polygon(PENTAGON).area * 2800)
    ifc = tmp_path / "b.ifc"
    write_ifc(storeys_of(("F1", pier_ir())), ifc, EXPORT)
    assert verify_ifc(ifc)["schema_errors"] == 0


def test_rectangle_on_column_layer_is_not_revived():
    big = [(0.0, 0.0), (2500.0, 0.0), (2500.0, 2500.0), (0.0, 2500.0)]
    assert not find_piers([closed(big, "S-COLS")], layers(), [], [], CFG)
    assert len(find_piers([closed(PENTAGON, "S-COLS")], layers(), [], [], CFG)) == 1


def test_wall_piece_is_written_as_wall_not_column(tmp_path):
    ir = pier_ir()
    ir["columns"][0]["wall_piece"] = True
    path = tmp_path / "b.dae"
    counts, _ = write_dae(storeys_of(("F1", ir)), path, EXPORT)
    assert counts["wall_pieces"] == 1 and counts["columns"] == 0
    assert "F1_Walls" in path.read_text(encoding="utf-8")

