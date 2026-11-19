"""물류 피드백(2026-11-19): 호 없는 문, 기둥 바깥 외벽 패널, 두 겹 벽 틈, 블록 가구."""
from __future__ import annotations

import numpy as np
from shapely.geometry import box

from cad2bim.layers import LayerInfo, LayerMap
from cad2bim.preprocess.primitive import Primitive
from cad2bim.recognize.columns import ColumnDraft
from cad2bim.recognize.openings import _door_in_gap, _door_shapes
from cad2bim.recognize.walls import WallDraft, _ColumnIndex, bridge_across_columns, slit_fillers

WALLS = {"column_bridge_offset_mm": 400.0, "column_bridge_slack_mm": 300.0, "slit_max_mm": 60.0, "slit_min_overlap_mm": 1000.0}


def wall(a, b, t=150.0):
    return WallDraft(np.array(a, float), np.array(b, float), t, "unknown", set(), {"W"}, 0.0)


def column(x, y, size=900.0):
    return ColumnDraft("rect", (x, y), (size, size), 0.0, box(x - size / 2, y - size / 2, x + size / 2, y + size / 2), set(), [], "C", False)


def line(a, b, layer="D"):
    return Primitive(f"L{a}{b}", "line", {"p1": a, "p2": b}, layer, ("h",))


def test_overhead_door_drawn_as_lines_fills_the_gap_as_a_door():
    w = wall((0, 0), (10000, 0), 200.0)
    layers = LayerMap({"D": LayerInfo("DOOR", "llm")})
    doors = _door_shapes([line((3000, -25), (6000, -25)), line((3000, 50), (6000, 50))], layers)
    cfg = {"door_gap_reach_mm": 300.0, "door_gap_min_share": 0.5}
    assert _door_in_gap(w, (3000.0, 6000.0), doors, cfg) == ["h"]
    assert _door_in_gap(w, (7000.0, 9000.0), doors, cfg) is None  # 문 선이 없는 틈은 그대로 미확인


def test_panel_walls_split_at_a_column_are_joined_but_not_without_a_column():
    left, right = wall((0, -500), (10000, -500)), wall((11000, -500), (20000, -500))
    joined = bridge_across_columns([left, right], _ColumnIndex([column(10500, 0)]), WALLS)
    assert joined == 1 and left.end[0] == 11000.0
    a, b = wall((0, -500), (10000, -500)), wall((11000, -500), (20000, -500))
    assert bridge_across_columns([a, b], _ColumnIndex([column(10500, 3000)]), WALLS) == 0  # 기둥이 멀면 문일 수 있다


def test_slit_between_two_layer_panels_is_filled_over_the_overlap_only():
    outer, inner = wall((0, 0), (8000, 0), 100.0), wall((2000, 150), (12000, 150), 150.0)  # 면 사이 25mm
    fillers = slit_fillers([outer, inner], WALLS)
    assert len(fillers) == 1
    f = fillers[0]
    assert f.start[0] == 2000.0 and f.end[0] == 8000.0 and abs(f.start[1] - 62.5) < 1e-6 and f.thickness == 27.0
    assert not slit_fillers([wall((0, 0), (8000, 0), 100.0), wall((0, 400), (8000, 400), 100.0)], WALLS)


def test_loading_dock_doors_without_walls_become_door_walls_but_not_inside_a_wall():
    from cad2bim.recognize.walls import door_panel_walls

    cfg = {"door_panel_min_mm": 2000.0, "door_panel_min_gap_mm": 40.0, "door_panel_max_gap_mm": 200.0}
    layers = LayerMap({"D": LayerInfo("DOOR", "llm")})
    dock = [line((0, -25), (3000, -25)), line((0, 50), (3000, 50))]  # 75mm 간격 문짝 두 선, 벽 없음
    found = door_panel_walls(dock, layers, [], cfg)
    assert len(found) == 1 and found[0].from_door and abs(found[0].thickness - 75.0) < 1e-6
    host = wall((-5000, 0), (8000, 0), 200.0)  # 같은 축에 벽이 있으면 그 벽의 문이다
    assert not door_panel_walls(dock, layers, [host], cfg)
    assert not door_panel_walls([line((0, 0), (900, 0)), line((0, 60), (900, 60))], layers, [], cfg)  # 짧은 문짝 선


def test_new_walls_use_a_kind_the_ir_accepts():
    """문짝 벽과 틈 메움 벽도 IR 벽 종류(concrete·masonry·curtain·unknown) 안이어야 한다."""
    from cad2bim.ir.models import Wall
    from cad2bim.recognize.walls import door_panel_walls

    cfg = {"door_panel_min_mm": 2000.0, "door_panel_min_gap_mm": 40.0, "door_panel_max_gap_mm": 200.0}
    layers = LayerMap({"D": LayerInfo("DOOR", "llm")})
    panel = door_panel_walls([line((0, -25), (3000, -25)), line((0, 50), (3000, 50))], layers, [], cfg)[0]
    filler = slit_fillers([wall((0, 0), (8000, 0), 100.0), wall((0, 150), (8000, 150), 150.0)], WALLS)[0]
    allowed = set(Wall.model_fields["kind"].annotation.__args__)
    assert panel.kind in allowed and filler.kind in allowed


def test_slit_filler_stops_at_doors_in_either_panel():
    outer, inner = wall((0, 0), (8000, 0), 100.0), wall((0, 150), (8000, 150), 150.0)
    fillers = slit_fillers([outer, inner], WALLS, {1: [(3000.0, 4000.0)]})  # 안쪽 벽 3~4m에 문
    spans = sorted((round(f.start[0]), round(f.end[0])) for f in fillers)
    assert spans == [(0, 3000), (4000, 8000)]


def test_any_gap_narrower_than_an_opening_is_filled_but_wide_gaps_stay():
    """틈의 종류를 가리지 않는다: 모서리 틈, 기둥 옆 틈은 메우고 1m 틈(문일 수 있음)은 둔다."""
    from cad2bim.building import PlanPart, Storey, close_slivers
    from test_building import plan_ir

    ir = plan_ir()
    ir["columns"] = [{"id": "C1", "center": [5000.0, 200.0], "size": [400.0, 400.0]}]
    ir["walls"] = [
        {"id": "A", "start": [0.0, 0.0], "end": [4700.0, 0.0], "thickness": 200.0, "kind": "unknown"},  # 기둥까지 100mm 틈
        {"id": "B", "start": [5200.0, 0.0], "end": [9000.0, 0.0], "thickness": 200.0, "kind": "unknown"},
        {"id": "C", "start": [9150.0, 0.0], "end": [9150.0, 5000.0], "thickness": 200.0, "kind": "unknown"},  # 모서리 50mm 틈
        {"id": "D", "start": [0.0, 5000.0], "end": [3000.0, 5000.0], "thickness": 200.0, "kind": "unknown"},
        {"id": "E", "start": [4000.0, 5000.0], "end": [9150.0, 5000.0], "thickness": 200.0, "kind": "unknown"},  # 1m 틈
    ]
    s = Storey("F1", PlanPart("D01", "F1", ir, 1.0, "title"), 0.0, 3000.0)
    filled = close_slivers([s], 400.0)
    new = [w for w in s.part.ir["walls"] if w["id"].startswith("SV-")]
    assert filled["F1"] == len(new) >= 2
    xs = [((w["start"][0] + w["end"][0]) / 2, (w["start"][1] + w["end"][1]) / 2) for w in new]
    assert any(4700 <= x <= 4900 and abs(y) < 200 for x, y in xs)  # 기둥 옆
    assert any(8950 <= x <= 9300 and abs(y) < 200 for x, y in xs)  # 모서리
    assert not any(3000 < x < 4000 and abs(y - 5000) < 200 for x, y in xs)  # 1m 틈은 그대로
    assert len(ir["walls"]) == 5  # 원래 IR은 그대로


def test_triangular_gap_is_filled_along_the_walls_not_as_a_slanted_plate():
    from cad2bim.building import PlanPart, Storey, close_slivers
    from test_building import plan_ir

    ir = plan_ir()
    ir["columns"] = []
    ir["walls"] = [  # 비스듬한 벽 끝과 직교 벽 사이 삼각형 틈
        {"id": "A", "start": [0.0, 0.0], "end": [5000.0, 0.0], "thickness": 200.0, "kind": "unknown"},
        {"id": "B", "start": [5250.0, 0.0], "end": [5250.0, 5000.0], "thickness": 200.0, "kind": "unknown"},
    ]
    s = Storey("F1", PlanPart("D01", "F1", ir, 1.0, "title"), 0.0, 3000.0)
    close_slivers([s], 400.0)
    for w in [w for w in s.part.ir["walls"] if w["id"].startswith("SV-")]:
        dx, dy = w["end"][0] - w["start"][0], w["end"][1] - w["start"][1]
        assert abs(dx) < 1e-6 or abs(dy) < 1e-6  # 벽 방향(축)과 나란하다


def test_walls_added_by_review_or_gap_fill_are_never_hidden():
    """물류 북서 코어: 기둥 범위 밖 코어의 검토 벽이 '건물 밖'으로, 같은 길이 메움 벽이 '반복 무늬'로 숨겨져 다시 뚫렸다."""
    from cad2bim.export.quality import classify

    columns = [{"id": f"C{i}", "center": [x, y], "size": [500.0, 500.0]} for i, (x, y) in enumerate([(0, 0), (8000, 0), (0, 8000), (8000, 8000)])]
    judged = {"level": "review", "score": 0.5, "reasons": ["llm_outline"]}
    walls = [{"id": "LW-F2-G001", "start": [30000.0, 0.0], "end": [32300.0, 0.0], "thickness": 150.0, "kind": "unknown", "confidence": judged}]
    walls += [{"id": f"SV-F2-{i:03d}", "start": [1000.0 + i * 1000.0, 4000.0], "end": [1000.0 + i * 1000.0, 4300.0], "thickness": 100.0,
               "kind": "unknown", "confidence": {"level": "review", "score": 0.5, "reasons": ["sliver_filled"]}} for i in range(5)]
    hidden = classify({"walls": walls, "columns": columns, "openings": [], "rooms": [], "stairs": []})
    assert hidden == {}


def test_gap_fill_leaves_doorways_open():
    """문이 달린 벽과 나란한 벽 사이 띠를 그냥 메우면 문 앞이 막힌다(물류 문 77곳). 메우되 문 자리는 구멍을 남긴다."""
    from cad2bim.building import PlanPart, Storey, close_slivers

    ir = {
        "source": {"sha256": "x"}, "origin_offset": [0.0, 0.0], "columns": [], "rooms": [], "stairs": [],
        "walls": [
            {"id": "A", "start": [0.0, 0.0], "end": [6000.0, 0.0], "thickness": 200.0, "kind": "unknown"},
            {"id": "B", "start": [0.0, 400.0], "end": [6000.0, 400.0], "thickness": 200.0, "kind": "unknown"},
            {"id": "E1", "start": [0.0, 0.0], "end": [0.0, 400.0], "thickness": 200.0, "kind": "unknown"},
            {"id": "E2", "start": [6000.0, 0.0], "end": [6000.0, 400.0], "thickness": 200.0, "kind": "unknown"},
        ],
        "openings": [{"id": "D1", "type": "door", "host_wall": "A", "position": 3000.0, "width": 1000.0}],
    }
    s = Storey("F1", PlanPart("D01", "F1", ir, 1.0, "title"), 0.0, 3000.0)
    close_slivers([s], 400.0)
    added = [w for w in s.part.ir["walls"] if w["id"].startswith("SV-")]
    assert added  # 문 밖 구간은 메운다
    # 문 자리(2500~3500)를 지나는 메움 벽에는 같은 자리에 판 없는 구멍이 있어야 한다(문 위 인방 틈만 막는다).
    for w in added:
        xs = sorted([w["start"][0], w["end"][0]])
        if xs[1] <= 2500.0 + 1 or xs[0] >= 3500.0 - 1:
            continue
        holes = [o for o in s.part.ir["openings"] if o["host_wall"] == w["id"]]
        assert holes and all(o.get("void_only") for o in holes)
        lo = min(xs[0] + o["position"] - o["width"] / 2 for o in holes)
        hi = max(xs[0] + o["position"] + o["width"] / 2 for o in holes)
        assert lo <= 2500.0 + 1 and hi >= 3500.0 - 1


def test_door_in_one_layer_of_a_double_wall_is_cut_through_the_other_layer():
    from cad2bim.building import PlanPart, Storey, doors_through_layers

    ir = {
        "source": {"sha256": "x"}, "origin_offset": [0.0, 0.0], "columns": [], "rooms": [], "stairs": [],
        "walls": [
            {"id": "A", "start": [0.0, 0.0], "end": [6000.0, 0.0], "thickness": 100.0, "kind": "unknown"},
            {"id": "B", "start": [0.0, 300.0], "end": [6000.0, 300.0], "thickness": 200.0, "kind": "unknown"},
            {"id": "FAR", "start": [0.0, 2000.0], "end": [6000.0, 2000.0], "thickness": 200.0, "kind": "unknown"},
        ],
        "openings": [
            {"id": "D1", "type": "door", "host_wall": "A", "position": 3000.0, "width": 1000.0},
            {"id": "WN1", "type": "window", "host_wall": "B", "position": 3000.0, "width": 1000.0},
        ],
    }
    s = Storey("F2", PlanPart("D07", "F2", ir, 1.0, "title"), 0.0, 3000.0)
    assert doors_through_layers([s]) == {"F2": 1}
    by_host = {o["host_wall"]: o for o in s.part.ir["openings"]}
    assert by_host["B"]["type"] == "unknown" and by_host["B"]["position"] == 3000.0  # 뒤 겹은 문짝 없는 빈 구멍(창 대신)
    assert "FAR" not in by_host  # 멀리 있는 벽은 다른 벽이다


def test_double_wall_with_a_door_in_each_layer_keeps_one_leaf():
    """사진: 문 두 장이 0.3m 간격으로 마주 보고 그 사이가 비어 보였다. 한 문 자리에 문짝은 한 장이다."""
    from cad2bim.building import PlanPart, Storey, doors_through_layers

    ir = {
        "source": {"sha256": "x"}, "origin_offset": [0.0, 0.0], "columns": [], "rooms": [], "stairs": [],
        "walls": [
            {"id": "A", "start": [0.0, 0.0], "end": [6000.0, 0.0], "thickness": 100.0, "kind": "unknown"},
            {"id": "B", "start": [0.0, 300.0], "end": [6000.0, 300.0], "thickness": 100.0, "kind": "unknown"},
        ],
        "openings": [
            {"id": "D1", "type": "door", "host_wall": "A", "position": 3000.0, "width": 1000.0},
            {"id": "D2", "type": "door", "host_wall": "B", "position": 3050.0, "width": 1000.0},
        ],
    }
    s = Storey("F2", PlanPart("D07", "F2", ir, 1.0, "title"), 0.0, 3000.0)
    doors_through_layers([s])
    assert sorted(o["type"] for o in s.part.ir["openings"]) == ["door", "unknown"]
