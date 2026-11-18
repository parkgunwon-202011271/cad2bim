"""건물 본체가 아닌 요소 가르기. 지우지 않고 숨기는지, 건물 일부를 건드리지 않는지 본다."""
from __future__ import annotations

import ezdxf
import pytest

from cad2bim.export.floor import plan_floors
from cad2bim.export.floors import element_tags, write_editable_files
from cad2bim.export.quality import FRAGMENT, OUTSIDE, PATTERN, building_only, classify
from test_building import EXPORT, opening, plan_ir, wall
from test_collada import storeys_of
from test_floor import CONF


def base():
    return plan_ir(columns=[(0.0, 0.0), (10000.0, 0.0), (10000.0, 5000.0), (0.0, 5000.0)])


def far_wall(index=9):
    return wall(index, (60000.0, 40000.0), (90000.0, 40000.0))


def test_wall_far_from_the_building_is_outside():
    ir = base()
    ir["walls"].append(far_wall())
    assert classify(ir) == {"W9": OUTSIDE}


def test_far_wall_with_a_door_or_a_room_is_kept():
    ir = base()
    ir["walls"].append(far_wall())
    ir["openings"] = [opening(host="W9", position=1000.0)]
    assert "W9" not in classify(ir)
    ir = base()
    ir["walls"].append(far_wall())
    ir["rooms"] = [{"id": "R", "names": ["x"], "polygon": [(0, 0), (1, 0), (1, 1)], "boundary_wall_ids": ["W9"]}]
    assert "W9" not in classify(ir)


def test_repeated_rows_are_a_pattern():
    ir = base()
    ir["walls"] += [wall(20 + i, (2000.0, 1000.0 + i * 600.0), (8000.0, 1000.0 + i * 600.0), 50.0) for i in range(5)]
    assert {name: flag for name, flag in classify(ir).items()} == {f"W{20 + i}": PATTERN for i in range(5)}


def test_small_isolated_piece_is_a_fragment_but_a_touching_one_is_not():
    ir = base()
    ir["walls"].append(wall(30, (4000.0, 2500.0), (5000.0, 2500.0)))
    assert classify(ir) == {"W30": FRAGMENT}
    ir = base()
    ir["walls"].append(wall(31, (0.0, 2500.0), (1000.0, 2500.0)))
    assert classify(ir) == {}


def test_without_columns_nothing_is_called_outside():
    ir = plan_ir(columns=[])
    ir["walls"].append(far_wall())
    assert "W9" not in classify(ir)


def test_stair_outside_the_building_is_hidden_and_does_not_cut_the_slab():
    ir = base()
    far = {"id": "ST1", "bounds": (60000, 40000, 63000, 44000), "confidence": CONF, "flights": [
        {"start": (61000, 41000), "end": (61000, 43000), "width": 1200.0, "treads": 8},
    ]}
    ir["stairs"] = [far]
    assert classify(ir) == {"ST1": OUTSIDE}
    assert not building_only(ir)["stairs"]


def test_outside_wall_does_not_change_the_slab():
    clean = storeys_of(("F1", base()))
    plan_floors(clean, EXPORT)
    ir = base()
    ir["walls"].append(wall(9, (10000.0, 2500.0), (40000.0, 2500.0)))  # 건물에서 뻗어 나간 벽
    ir["walls"].append(far_wall(10))
    other = storeys_of(("F1", ir))
    plan_floors(other, EXPORT)
    assert sum(p.area for p in other[0].floor.slabs) == pytest.approx(sum(p.area for p in clean[0].floor.slabs), rel=0.02)


def test_hidden_elements_stay_in_the_file_on_a_layer_that_is_off(tmp_path):
    ir = base()
    ir["walls"].append(far_wall())
    write_editable_files(storeys_of(("F1", ir)), tmp_path, EXPORT)
    doc = ezdxf.readfile(tmp_path / "building_editable.dxf")
    assert doc.layers.get("F1_OUTSIDE_Walls").is_off()
    assert doc.layers.get("F1_Walls").is_on()
    assert len(doc.modelspace().query('LWPOLYLINE[layer=="F1_OUTSIDE_Walls"]')) == 1


def test_tag_prefers_the_hidden_kind_over_review():
    ir = base()
    wall_far = far_wall()
    wall_far["confidence"] = {"level": "review", "score": 0.1, "reasons": []}
    ir["walls"].append(wall_far)
    assert element_tags(ir)["W9"] == OUTSIDE


def test_stair_steps_form_a_sloped_slab_not_a_block():
    """단마다 한 단 높이에 계단판 두께만큼만 내려 채운다. 단 묶음 바닥까지 채우면 두 번째 묶음이
    계단참 위에 벽처럼 선다."""
    from test_floor import _stair, _with

    storeys = storeys_of(("F1", _with(stairs=[_stair()])), ("F2", _with()))
    plan_floors(storeys, EXPORT)
    _, boxes, _ = storeys[0].floor.stairs[0]
    rise = boxes[0].z1
    waist = EXPORT["landing_thickness_mm"]
    assert boxes[0].z0 == 0.0
    assert all(0 < b.z1 - b.z0 <= rise + waist + 1e-6 for b in boxes)
    # 이웃한 단은 높이가 겹쳐 한 덩어리로 이어진다(떠 있는 단이 없다).
    assert all(nxt.z0 < cur.z1 for cur, nxt in zip(boxes[:7], boxes[1:8]))
    assert boxes[8].z0 > 1000.0
