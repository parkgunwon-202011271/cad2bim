"""지붕층 평면(D). 맨 위층 위에 놓고, 바닥판이 지붕이며 그 위는 덮지 않는다."""
from __future__ import annotations

from cad2bim.building import ROOF_PARAPET_MM, PlanPart, Storey, select_storeys
from cad2bim.classify.floors import floor_key, roof_key
from cad2bim.export.floor import plan_floors
from test_building import EXPORT, part
from test_collada import storeys_of
from test_floor import _with


def test_roof_titles():
    assert roof_key("지붕층 평면도(B동)") == "R"
    assert roof_key("ROOF PLAN") == "R"
    assert roof_key("지상2층 평면도") is None
    assert floor_key("3층 지붕평면도") == "F3"  # 층 번호가 있으면 그 층이다


def test_roof_sits_on_the_top_floor():
    storeys = select_storeys([part("F1"), part("F2", part_id="D02"), part("R", part_id="D03")], 3000.0,
                             {"F1": 100000.0, "F2": 104000.0, "F3": 110000.0})
    roof = storeys[-1]
    assert roof.key == "R"
    assert roof.elevation == 4000.0 + 6000.0 and roof.height == ROOF_PARAPET_MM


def test_roof_alone_is_dropped():
    assert select_storeys([part("R")], 3000.0) == []


def test_roof_level_has_a_slab_and_no_lid_and_covers_the_floor_below():
    storeys = storeys_of(("F1", _with()))
    storeys.append(Storey("R", PlanPart("D02", "R", _with(), 1.0, "title"), 3000.0, ROOF_PARAPET_MM))
    plan_floors(storeys, EXPORT)
    top, roof = storeys
    assert roof.key == "R" and roof.floor.slabs and not roof.floor.roofs
    assert not top.floor.roofs  # 지붕층 바닥판이 덮는다


def _bubble(label, x, y):
    from cad2bim.preprocess.primitive import Primitive
    return [
        Primitive(f"c{label}{x}", "circle", {"center": (x, y), "radius": 400.0}, "G", ("1",)),
        Primitive(f"t{label}{x}", "text", {"position": (x, y), "text": label, "height": 300.0, "rotation": 0.0, "centered": True}, "G", ("1",)),
    ]


def _labelled(dx, dy, bubble_row_y):
    from test_building import plan_ir
    from cad2bim.building import PlanPart, grid_labels
    model = []
    for i in range(4):
        model += _bubble(f"X{i + 1}", dx + i * 11000.0, dy + bubble_row_y)
    for j in range(3):
        model += _bubble(f"Y{j + 1}", dx - 5000.0 - j * 7.0, dy + j * 10500.0)
    model += _bubble("1", dx + 99999.0, dy + 77777.0)  # 단면 표시 같은 다른 동그라미
    ir = {**plan_ir(), "source": {"unit_scale": 1.0}}
    return PlanPart("P", "F1", ir, 1.0, "title", None, False, grid_labels(model, ir))


def test_grid_bubbles_align_plans_even_when_bubble_rows_differ():
    from cad2bim.building import _label_shift
    roof, floor = _labelled(1_000_000.0, 400_000.0, 30000.0), _labelled(0.0, 0.0, 45000.0)
    assert set(roof.labels) >= {"X1", "X4", "Y1", "Y3"}
    assert _label_shift(roof, floor)[0] == (-1_000_000.0, -400_000.0)
