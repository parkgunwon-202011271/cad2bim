"""부분층(E). '4층 상부' 평면은 F4M이고, 평면에 적힌 레벨로 4층 안에 놓는다."""
from __future__ import annotations

import dataclasses

from cad2bim.building import is_full_floor, select_storeys
from cad2bim.classify.floors import floor_key, partial_key, plan_level
from cad2bim.preprocess.primitive import Primitive
from test_building import part


def text(value, x=0.0, y=0.0):
    return Primitive(f"T{value}{x}", "text", {"position": (x, y), "text": value, "height": 300.0, "rotation": 0.0, "centered": False}, "L", ("1",))


def test_partial_titles():
    assert partial_key("지상4층 상부 평면도(B동)", floor_key("지상4층 상부 평면도(B동)")) == "F4M"
    assert partial_key("지상4층 평면도", "F4") == "F4"
    assert partial_key("상부", None) is None
    assert is_full_floor("F4") and not is_full_floor("F4M") and not is_full_floor("R")


def test_plan_level_is_the_most_common_mark():
    assert plan_level([text("FL:135.10"), text("FL:135.10", 9), text("FL:139.90", 5), text("거실")]) == 135100.0
    assert plan_level([text("거실")]) is None


def test_partial_floor_sits_at_its_own_level_inside_the_floor_below():
    mezz = dataclasses.replace(part("F4M", part_id="D10"), level=135100.0)
    levels = {"F3": 119750.0, "F4": 129750.0, "F5": 139750.0}
    storeys = {s.key: s for s in select_storeys([part("F3"), part("F4", part_id="D09"), mezz], 3000.0, levels)}
    assert storeys["F4"].part.part_id == "D09"  # 상부 평면이 4층 평면을 밀어내지 않는다
    f4, m = storeys["F4"], storeys["F4M"]
    assert m.elevation == f4.elevation + 5350.0
    assert m.elevation + m.height == f4.elevation + f4.height


def test_partial_floor_without_a_level_goes_halfway_and_without_base_is_dropped():
    storeys = {s.key: s for s in select_storeys([part("F1"), part("F1M", part_id="D2")], 4000.0)}
    assert storeys["F1M"].elevation == 2000.0
    assert [s.key for s in select_storeys([part("F1"), part("F2M", part_id="D2")], 4000.0)] == ["F1"]


def test_partial_floor_drops_what_lies_outside_the_base_building():
    """4층 상부 평면에 함께 그린 건물 밖 구조물은 다른 레벨이라 부분층에 넣지 않는다."""
    from cad2bim.building import PlanPart, Storey, trim_partial_floors
    from test_building import plan_ir, wall

    base = plan_ir(columns=[(0.0, 0.0), (10000.0, 0.0)])
    mezz = plan_ir()
    mezz["walls"] = [wall(1, (0.0, 1000.0), (5000.0, 1000.0)), wall(2, (-40000.0, 0.0), (-35000.0, 0.0))]
    storeys = [Storey("F4", PlanPart("A", "F4", base, 1.0, "title"), 0.0, 10000.0),
               Storey("F4M", PlanPart("B", "F4M", mezz, 1.0, "title"), 5000.0, 5000.0)]
    assert trim_partial_floors(storeys) == {"F4M": 1}
    assert [w["id"] for w in storeys[1].part.ir["walls"]] == ["W1"]
    assert len(mezz["walls"]) == 2  # 원래 IR은 건드리지 않는다


def test_base_walls_under_a_partial_floor_stop_at_its_slab():
    """물류 4층 상부 문 15곳을 4층 벽이 막았다. 상부층 바닥 아래 4층 벽은 상부층 바닥에서 멈춰야 한다."""
    from cad2bim.building import PlanPart, Storey, hand_over_to_partial
    from cad2bim.config import load_defaults

    export = load_defaults()["export"]

    def ir(walls, openings=()):
        return {"source": {"sha256": "x"}, "origin_offset": [0.0, 0.0], "columns": [], "rooms": [], "stairs": [],
                "walls": walls, "openings": list(openings)}

    box = [
        {"id": f"P{i}", "start": a, "end": b, "thickness": 200.0, "kind": "unknown"}
        for i, (a, b) in enumerate([([0.0, 0.0], [5000.0, 0.0]), ([5000.0, 0.0], [5000.0, 4000.0]),
                                    ([5000.0, 4000.0], [0.0, 4000.0]), ([0.0, 4000.0], [0.0, 0.0])])
    ]
    base_walls = [{"id": "B1", "start": [-3000.0, 2000.0], "end": [3000.0, 2000.0], "thickness": 200.0, "kind": "unknown"},
                  {"id": "B2", "start": [-3000.0, -3000.0], "end": [3000.0, -3000.0], "thickness": 200.0, "kind": "unknown"}]
    door = {"id": "D1", "type": "door", "host_wall": "B1", "position": 1000.0, "width": 900.0}
    base = Storey("F4", PlanPart("D10", "F4", ir(base_walls, [door]), 1.0, "title"), 24200.0, 10000.0)
    part = Storey("F4M", PlanPart("D11", "F4M", ir(box), 1.0, "title"), 29600.0, 4600.0)
    changed = hand_over_to_partial([base, part], export)
    assert changed == {"F4": 1}
    walls = {w["id"]: w for w in base.part.ir["walls"]}
    assert "top_mm" not in walls["B2"]  # 부분층 밖 벽은 그대로
    low = [w for w in walls.values() if w.get("top_mm")]
    assert len(low) == 1 and min(low[0]["start"][0], low[0]["end"][0]) >= -60 and low[0]["top_mm"] == 5400.0 - export["slab_thickness_mm"]
    assert base.part.ir["openings"][0]["host_wall"] == "B1" and base.part.ir["openings"][0]["position"] == 1000.0


def test_base_facade_wall_keeps_full_height_where_partial_floor_has_no_wall_above():
    from cad2bim.building import PlanPart, Storey, hand_over_to_partial
    from cad2bim.config import load_defaults

    export = load_defaults()["export"]

    def ir(walls):
        return {"source": {"sha256": "x"}, "origin_offset": [0.0, 0.0], "columns": [], "rooms": [], "stairs": [], "walls": walls, "openings": []}

    # 부분층은 ㄷ자로 그려져 남쪽(y=0) 벽이 없다. 바닥판은 닫혀 계산되므로 남쪽 경계도 바닥판 가장자리다.
    part_walls = [
        {"id": f"P{i}", "start": a, "end": b, "thickness": 200.0, "kind": "unknown"}
        for i, (a, b) in enumerate([([5000.0, 0.0], [5000.0, 4000.0]), ([5000.0, 4000.0], [0.0, 4000.0]), ([0.0, 4000.0], [0.0, 0.0])])
    ]
    south = {"id": "S", "start": [0.0, 0.0], "end": [5000.0, 0.0], "thickness": 200.0, "kind": "unknown"}
    base = Storey("F4", PlanPart("D10", "F4", ir([south]), 1.0, "title"), 24200.0, 10000.0)
    part = Storey("F4M", PlanPart("D11", "F4M", ir(part_walls), 1.0, "title"), 29600.0, 4600.0)
    hand_over_to_partial([base, part], export)
    assert not any(w.get("top_mm") for w in base.part.ir["walls"])
