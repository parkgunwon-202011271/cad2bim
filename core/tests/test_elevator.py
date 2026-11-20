"""엘리베이터: 승강로 표시와 'EV' 글자로 찾고, 위아래 층이 같은 자리면 바닥을 뚫어 잇는다."""
from __future__ import annotations

from shapely.geometry import Point

from cad2bim.config import load_defaults
from cad2bim.export.elevator import elevator_data, find_elevators
from cad2bim.export.floor import plan_floors
from cad2bim.preprocess.primitive import Primitive
from test_building import EXPORT
from test_collada import storeys_of
from test_floor import _with

CFG = load_defaults()["export"]


def rect(x0, y0, x1, y1, layer="A-FLOR-FIXT", ident="R"):
    points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return Primitive(ident, "polyline", {"points": points, "closed": True}, layer, (ident,))


def text(x, y, value, layer="A-AREA-IDEN"):
    return Primitive(f"T{value}", "text", {"position": (x, y), "text": value, "height": 200, "rotation": 0.0, "centered": False}, layer, ("T",))


ROLES = {"A-FLOR-FIXT": "FURNITURE", "A-AREA-IDEN": "ROOM_TEXT", "A-WALL": "WALL"}


def test_marked_box_with_ev_label_is_an_elevator():
    prims = [rect(2000, 2000, 4500, 4300), text(3000, 3000, "EV"), rect(6000, 2000, 8500, 4300, ident="R2")]
    found = find_elevators(prims, ROLES, CFG)
    assert len(found) == 1 and found[0].contains(Point(3000, 3000))


def test_room_names_and_wall_outlines_are_not_elevators():
    assert not find_elevators([rect(2000, 2000, 4500, 4300), text(3000, 3000, "EPS")], ROLES, CFG)
    assert not find_elevators([rect(2000, 2000, 4500, 4300, layer="A-WALL"), text(3000, 3000, "EV")], ROLES, CFG)
    assert not find_elevators([rect(2000, 2000, 2400, 2400), text(2200, 2200, "EV")], ROLES, CFG)  # 너무 작다


def _storeys(upper_shift=0.0):
    storeys = storeys_of(("F1", _with()), ("F2", _with()), ("F3", _with()))
    shaft = rect(2000, 1000, 4500, 3300)
    found = find_elevators([shaft, text(3000, 2000, "엘리베이터")], ROLES, CFG)
    for s in storeys:
        s.elevators = list(found)
    storeys[2].shift = (upper_shift, 0.0)
    return storeys


def test_shaft_opens_the_upper_slabs_but_keeps_the_pit_floor():
    storeys = _storeys()
    plan_floors(storeys, EXPORT)
    inside = Point(3000, 2000)
    assert any(s.contains(inside) for s in storeys[0].floor.slabs)  # 맨 아래층은 피트 바닥
    assert not any(s.contains(inside) for s in storeys[1].floor.slabs)
    assert not any(s.contains(inside) for s in storeys[2].floor.slabs)


def test_shafts_that_do_not_line_up_are_not_joined():
    storeys = _storeys(upper_shift=8000.0)  # 3층 도면의 승강로가 다른 자리다
    plan_floors(storeys, EXPORT)
    shafts = elevator_data(storeys, (0.0, 0.0), CFG)
    assert sorted(s["floors"] for s in shafts) == [["F1", "F2"], ["F3"]]
    inside = Point(3000, 2000)
    assert any(s.contains(inside) for s in storeys[2].floor.slabs)  # 이어지지 않으면 뚫지 않는다


def test_furniture_label_picks_height_or_skips_wall_devices():
    import re

    from shapely.geometry import Polygon

    from cad2bim.export.furniture_items import Item, classify
    from cad2bim.export.furniture_models import build

    rules = [(re.compile(r["pattern"], re.IGNORECASE), r) for r in CFG["furniture_labels"]]
    polygon = Polygon(rect(0, 0, 500, 500).geometry["points"])

    def named(point, value):
        found = classify([Item(polygon, "A-FLOR-FIXT", True)], [(point, value, False)], rules, CFG["furniture_label_reach_mm"])
        return found[0] if found else None

    water = named(Point(250, 900), "정수기")
    assert water.kind == "appliance" and max(p["z1"] for p in build(water)) == 1200
    assert named(Point(250, 250), "비상벨") is None  # 벽에 다는 장치는 만들지 않는다
    assert named(Point(250, 250), "남자화장실").how == "size"  # 실 이름은 가구 이름표가 아니다
    assert named(Point(250, 3000), "정수기").how == "size"  # 너무 멀다
