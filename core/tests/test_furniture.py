"""가구 내보내기. 도면 선에서 가구를 찾아 종류를 정하고, 종류별 입체를 building.dae와 같은 좌표로 내는지 본다."""
from __future__ import annotations

import math
import re
from types import SimpleNamespace

import pytest
from shapely.geometry import Point, Polygon, box

from cad2bim.export import furniture
from cad2bim.export import furniture_items as items_mod
from cad2bim.export.furniture_models import build
from cad2bim.preprocess.primitive import Primitive

RULES = [
    (re.compile(r["pattern"], re.IGNORECASE), r) for r in (
        {"pattern": r"\bset\b", "kind": "ignore"},
        {"pattern": r"bar\s*chair", "kind": "bar_stool"},
        {"pattern": r"work\s*table", "kind": "counter", "m": "steel"},
        {"pattern": "수납장", "kind": "cabinet", "z1": 1800},
        {"pattern": "정수기", "kind": "appliance", "z0": 0, "z1": 1200},
    )
]


def prim(kind, layer="A-FURN", **geometry):
    return Primitive(id=f"p{kind}", kind=kind, geometry=geometry, layer=layer, source_handles=("1",))


def ring(polygon):
    points = list(polygon.exterior.coords)
    return (points, "A-FURN", False)


def lines_of(polygon):
    """사각형을 낱개 선 네 개로."""
    points = list(polygon.exterior.coords)
    return [([a, b], "A-FURN", True) for a, b in zip(points, points[1:])]


def recognise(shapes=(), chains=(), labels=(), rings=(), layer="A-FURN"):
    found = items_mod.find_items([(s, layer) for s in shapes], list(chains), list(rings))
    found = items_mod.classify(items_mod.attach_parts(found), list(labels), RULES, 900.0)
    items_mod.face_tables(found)
    return found


def test_shapes_become_segments():
    assert len(furniture._segments(prim("line", p1=(0, 0), p2=(1, 0)))) == 1
    square = prim("polyline", points=[(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
    assert len(furniture._segments(square)) == 4  # 닫힌 꺾은선은 마지막 변까지
    quarter = furniture._segments(prim("arc", center=(0, 0), radius=900, start=(900, 0), end=(0, 900)))
    assert quarter[0][0] == pytest.approx((900, 0)) and quarter[-1][1] == pytest.approx((0, 900), abs=1e-6)
    circle = furniture._segments(prim("circle", center=(0, 0), radius=300))
    assert len(circle) == math.ceil(360 / furniture._ARC_STEP_DEG)
    assert furniture._segments(prim("text", position=(0, 0), text="x", height=1, rotation=0, centered=False)) == []


def test_size_picks_the_kind_when_nothing_else_tells():
    kind = lambda polygon, layer="A-FURN": recognise([polygon], layer=layer)[0].kind  # noqa: E731
    assert kind(box(0, 0, 500, 500)) == "chair"
    assert kind(box(0, 0, 400, 1800)) == "shelf"
    assert kind(box(0, 0, 50, 1600)) == "partition"
    assert kind(box(0, 0, 1800, 700)) == "table"
    assert kind(Point(0, 0).buffer(600, quad_segs=6)) == "table"  # 둥근 탁자
    assert kind(Point(0, 0).buffer(200, quad_segs=6)) == "stool"
    assert kind(box(0, 0, 50, 1500), "A-FLOR-FIXT") == "stall"  # 화장실 칸막이
    assert kind(box(0, 0, 350, 350), "A-FLOR-FIXT") == "fixture_low"


def test_table_has_a_top_and_four_legs_and_a_round_one_a_pedestal():
    desk = build(recognise([box(0, 0, 1800, 700)])[0])
    assert [p["m"] for p in desk] == ["wood"] + ["metal"] * 4
    assert (desk[0]["z0"], desk[0]["z1"]) == (720.0, 750.0) and all(p["z0"] == 0.0 and p["z1"] == 720.0 for p in desk[1:])
    round_table = build(recognise([Point(0, 0).buffer(600, quad_segs=6)])[0])
    assert len(round_table) == 3  # 상판, 기둥, 받침


def test_strip_beside_a_seat_is_the_backrest():
    """주민센터 도면의 의자: 좌판 사각형 옆에 등받이 띠를 따로 그린다."""
    found = recognise([box(0, 0, 450, 450), box(450, 0, 550, 450)])
    assert len(found) == 1 and found[0].kind == "chair"
    chair = found[0]
    assert chair.band == pytest.approx((1.0, 0.0)) and chair.band_mm == pytest.approx(100.0)
    assert chair.polygon.bounds == pytest.approx((0, 0, 550, 450))
    parts = build(chair)
    back = max(parts, key=lambda p: p["z1"])
    assert len(parts) == 6 and back["z1"] == 880.0  # 다리 넷, 좌판, 등받이
    assert back["polygon"].bounds[0] >= 450.0  # 등받이는 띠를 그린 쪽에 선다


def test_chair_without_a_strip_faces_the_nearest_table():
    table, chair = box(0, 0, 1800, 700), box(600, 760, 1100, 1260)
    found = {item.kind: item for item in recognise([table, chair])}
    assert found["chair"].band == pytest.approx((0.0, 1.0))  # 등받이는 탁자 반대쪽
    assert len(build(found["chair"])) == 6


def test_long_seat_with_a_back_strip_is_a_sofa_and_double_strips_stack():
    """디젤 도면의 부스 소파: 등받이 띠 두 줄(40+50), 좌판, 앞턱(50)을 낱개 선으로 그린다."""
    chains = []
    for x0, x1 in ((0, 40), (40, 90), (90, 480), (480, 530)):
        chains += lines_of(box(x0, 0, x1, 1500))
    found = recognise(chains=chains)
    assert [item.kind for item in found] == ["sofa"]
    sofa = found[0]
    assert sofa.polygon.bounds == pytest.approx((0, 0, 530, 1500), abs=1.0)
    assert sofa.band == pytest.approx((-1.0, 0.0)) and sofa.band_mm == pytest.approx(90.0, abs=1.0)  # 두꺼운 쪽이 등받이


def test_lines_and_closed_shapes_give_the_same_furniture():
    drawn = recognise([box(0, 0, 1800, 700)])
    from_lines = recognise(chains=lines_of(box(0, 0, 1800, 700)))
    assert [i.kind for i in from_lines] == [i.kind for i in drawn] == ["table"]
    assert from_lines[0].polygon.area == pytest.approx(drawn[0].polygon.area, rel=1e-3)
    assert not from_lines[0].drawn and drawn[0].drawn


def test_nearly_coincident_lines_still_close_a_face():
    """소수점 아래에서만 어긋난 겹친 선이 면을 깨지 않는다(디젤 도면의 좌석 한 줄이 통째로 빠지던 문제)."""
    chains = lines_of(box(0, 0, 390, 1500)) + [([(0.0, 1e-9), (440.0, 3e-9)], "A-FURN", True), ([(0.0, 1500.0 - 2e-9), (440.0, 1500.0)], "A-FURN", True)]
    assert len(items_mod.find_items([], chains)) == 1


def test_cross_marks_a_tall_unit_hatching_a_showcase_one_slash_a_shelf():
    cell = box(0, 0, 1200, 800)
    cross = lines_of(cell) + [([(0, 0), (1200, 800)], "A-FURN", True), ([(0, 800), (1200, 0)], "A-FURN", True)]
    found = recognise(chains=cross)
    assert [item.kind for item in found] == ["tall"]  # X 표시는 상자를 넷으로 가르지 않는다
    hatch = lines_of(cell) + [([(x, 0), (x + 400, 800)], "A-FURN", True) for x in (100, 300, 500, 700)]
    assert [item.kind for item in recognise(chains=hatch)] == ["showcase"]
    assert any(p["m"] == "glass" for p in build(recognise(chains=hatch)[0]))
    slash = lines_of(cell) + [([(0, 0), (1200, 800)], "A-FURN", True)]
    assert [item.kind for item in recognise(chains=slash)] == ["shelf"]


def test_slanted_sides_of_a_hexagon_are_not_slashes():
    """끝에서 직각으로 만나는 선이 없다고 다 사선은 아니다. 빼도 사각형이 드러나지 않으면 가구의 변이다."""
    hexagon = Polygon([(600 * math.cos(math.radians(a)), 600 * math.sin(math.radians(a))) for a in range(0, 360, 60)])
    found = items_mod.find_items([], lines_of(hexagon))
    assert len(found) == 1 and found[0].polygon.area == pytest.approx(hexagon.area, rel=1e-3) and not found[0].slashes


def test_burners_make_a_stove_and_only_a_rounded_bowl_makes_a_sink():
    cell = box(0, 0, 900, 600)
    # 버너는 끊어 그린 호다. 닫힌 도형이 아니어도 중심과 반지름으로 알아본다.
    arcs = [(x, y, 110.0, 80.0, "A-FURN") for x in (250, 650) for y in (180, 420) for _ in range(4)]
    stove = recognise(chains=lines_of(cell), rings=arcs)[0]
    assert stove.kind == "stove" and len(items_mod.burners(stove)) == 4
    assert sum(1 for p in build(stove) if p["m"] == "metal" and p["z0"] > 800) == 4  # 버너 넷이 상판 위에 놓인다

    bowl = box(200, 100, 700, 500).buffer(-60).buffer(60)  # 모서리를 굴린 볼
    sink = recognise([cell, bowl])[0]
    assert sink.kind == "sink"
    assert any(p["holes"] for p in build(sink))  # 상판에 볼 자리가 뚫린다
    assert recognise([cell, box(200, 100, 700, 500)])[0].kind == "table"  # 반듯한 사각형은 볼이 아니다(계산대 위의 기기)


def test_leader_names_the_item_it_points_at_and_same_shapes_follow():
    stools = [Point(x, 0).buffer(200, quad_segs=6) for x in (0, 800, 1600)]
    found = recognise(stools, labels=[(Point(10, 10), "BAR CHAIR", True), (Point(820, 0), "SOFA&TABLE SET(6EA)", True)])
    assert [item.kind for item in found] == ["bar_stool"] * 3
    assert sorted(item.how for item in found) == ["label", "same_shape", "same_shape"]
    assert max(p["z1"] for p in build(found[0])) == 750.0


def test_drawing_evidence_beats_a_label_pointing_at_the_wrong_cell():
    """지시선이 옆 칸을 가리켜도 버너가 그려진 칸은 가스레인지다."""
    cell = box(0, 0, 900, 600)
    arcs = [(x, 300.0, 110.0, 360.0, "A-FURN") for x in (250, 650)]
    found = recognise(chains=lines_of(cell), rings=arcs, labels=[(Point(450, 50), "WORK TABLE", True)])
    assert found[0].kind == "stove" and found[0].how == "symbol"


def test_text_beside_a_chair_names_the_cabinet_not_the_chair():
    chair = [box(0, 0, 450, 450), box(450, 0, 550, 450)]
    cabinet = box(900, -300, 1400, 2100)
    found = {item.kind: item for item in recognise(chair + [cabinet], labels=[(Point(700, 200), "수납장", False)])}
    assert set(found) == {"chair", "cabinet"} and found["cabinet"].how == "label"


def test_machine_names_do_not_spread_to_same_sized_shapes():
    machines = [box(0, 0, 500, 500), box(3000, 0, 3500, 500)]
    found = recognise(machines, labels=[(Point(250, 250), "정수기", False)])
    assert sorted(item.kind for item in found) == ["appliance", "chair"]


def test_toilet_is_a_bowl_with_its_tank():
    bowl = Point(0, 0).buffer(1.0, quad_segs=8)
    from shapely import affinity

    bowl = affinity.scale(bowl, 180, 250)  # 360 × 500 타원
    tank = box(-225, 250, 225, 450)
    found = recognise([bowl, tank], layer="A-FLOR-FIXT")
    assert len(found) == 1 and found[0].kind == "toilet" and found[0].tank is not None
    parts = build(found[0])
    assert max(p["z1"] for p in parts) == 800.0  # 물탱크
    assert any(p["holes"] for p in parts)  # 좌판은 가운데가 뚫렸다


def test_floor_lines_use_model_coordinates_and_skip_other_plans(monkeypatch):
    ir = {
        "source": {"unit_scale": 1.0},
        "origin_offset": [1000.0, 2000.0],
        "walls": [{"start": [0.0, 0.0], "end": [5000.0, 0.0]}, {"start": [0.0, 0.0], "end": [0.0, 4000.0]}],
    }
    storey = SimpleNamespace(part=SimpleNamespace(ir=ir), shift=(100.0, 0.0), elevation=4500.0)
    raw = [
        prim("line", p1=(2000, 3000), p2=(3000, 3000)),  # 도면 좌표. IR로는 (1000, 1000)~(2000, 1000)
        prim("line", p1=(90000, 3000), p2=(91000, 3000)),  # 같은 도면의 다른 평면
        prim("line", layer="A-WALL", p1=(2000, 3500), p2=(3000, 3500)),  # 가구가 아닌 레이어
    ]
    monkeypatch.setattr(furniture, "load_primitives", lambda *a, **k: (raw, None))
    roles = {"A-FURN": "FURNITURE", "A-WALL": "WALL"}
    lines = furniture.floor_furniture(storey, None, roles, (500.0, 500.0), {}, None)
    # 모델 좌표 = IR + 원점 이동 + 층 이동 - 공통 원점
    assert lines == [[1000 + 1000 + 100 - 500, 1000 + 2000 - 500, 2000 + 1000 + 100 - 500, 1000 + 2000 - 500]]


def test_floor_objects_list_solids_with_materials_and_drop_their_plan_lines(monkeypatch):
    ir = {
        "source": {"unit_scale": 1.0},
        "origin_offset": [0.0, 0.0],
        "walls": [{"start": [0.0, 0.0], "end": [5000.0, 0.0]}, {"start": [0.0, 0.0], "end": [0.0, 4000.0]}],
    }
    storey = SimpleNamespace(part=SimpleNamespace(ir=ir), shift=(0.0, 0.0), elevation=0.0)
    raw = [
        prim("polyline", points=[(1000, 1000), (2800, 1000), (2800, 1700), (1000, 1700)], closed=True),  # 탁자
        prim("line", p1=(1200, 1200), p2=(2600, 1200)),  # 탁자 안에 그린 선
        prim("line", p1=(3500, 3000), p2=(4500, 3000)),  # 가구가 되지 못한 선
    ]
    monkeypatch.setattr(furniture, "load_primitives", lambda *a, **k: (raw, None))
    segments, solids, listing = furniture.floor_objects(storey, None, {"A-FURN": "FURNITURE"}, (0.0, 0.0), {}, None)
    assert segments == [[3500.0, 3000.0, 4500.0, 3000.0]]  # 입체로 만든 탁자의 선은 뺀다
    assert [s["m"] for s in solids] == ["wood"] + ["metal"] * 4 and {s["k"] for s in solids} == {"table"}
    assert listing == [{"kind": "table", "how": "size", "at": [1900.0, 1350.0]}]
