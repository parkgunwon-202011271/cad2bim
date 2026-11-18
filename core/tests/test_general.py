"""여러 작도 방식 수용. 선 조각 문 호, 함께 그린 일람표, 영문 층 제목, 층 표기 없는 도면."""
from __future__ import annotations

import math

import pytest

from cad2bim.building import part_title_kind
from cad2bim.classify import SheetDictionary, classify_sheet
from cad2bim.classify.floors import floor_key, plan_title_floor
from cad2bim.config import load_sheet_dictionary
from cad2bim.preprocess.primitive import Primitive, faceted_door_arc


def poly(points, closed=False, layer="A-DOOR"):
    return Primitive("P", "polyline", {"points": points, "closed": closed}, layer, ("1",))


def quarter(radius=900.0, cx=1000.0, cy=2000.0, n=19, start=0.0, sweep=90.0):
    return [
        (cx + radius * math.cos(math.radians(start + sweep * i / (n - 1))), cy + radius * math.sin(math.radians(start + sweep * i / (n - 1))))
        for i in range(n)
    ]


def text(value, x=0.0, y=0.0, height=100.0):
    return Primitive(f"T{value}{x}", "text", {"position": (x, y), "text": value, "height": height, "rotation": 0.0, "centered": False}, "T", ("1",))


def test_faceted_quarter_arc_becomes_an_arc():
    arc = faceted_door_arc(poly(quarter()))
    assert arc.kind == "arc" and arc.geometry["radius"] == pytest.approx(900.0, abs=1)
    assert arc.geometry["center"] == pytest.approx((1000.0, 2000.0), abs=1)


def test_clockwise_arc_keeps_counterclockwise_order():
    arc = faceted_door_arc(poly(quarter()[::-1]))
    assert arc.geometry["start"] == pytest.approx(quarter()[0], abs=0.01)


@pytest.mark.parametrize("points", [
    quarter(radius=5000.0),  # 문 크기가 아니다
    quarter(radius=100.0),
    quarter(sweep=180.0),  # 반원
    [(0.0, 0.0), (500.0, 0.0), (1000.0, 0.0), (1500.0, 0.0), (2000.0, 0.0)],  # 직선
    [(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0), (0.0, 500.0)],  # 사각형
])
def test_other_polylines_stay_as_they_are(points):
    assert faceted_door_arc(poly(points)) is None


def test_closed_polyline_is_not_an_arc():
    assert faceted_door_arc(poly(quarter(), closed=True)) is None


@pytest.mark.parametrize(("value", "key"), [
    ("GROUND FLOOR PLAN", "F1"), ("FIRST FLOOR PLAN", "F2"), ("second floor", "F3"), ("지상 3층", "F3"), ("3F", "F3"),
])
def test_english_floor_names(value, key):
    assert floor_key(value) == key


def test_title_with_parts_reads_only_the_part_that_names_a_plan():
    prims = [text("PLAN / FIRST FLOOR PLAN / MODEL 1:1 mm (sheet 1:100)")]
    assert plan_title_floor(prims, None) == "F2"


def test_part_title_kind_tells_section_from_plan():
    dictionary = SheetDictionary(load_sheet_dictionary())
    section = [text("SECTION / SECTION A-A / MODEL 1:1 mm (sheet 1:150)", 500.0, 500.0)]
    assert part_title_kind(section, (0.0, 0.0, 1000.0, 1000.0), dictionary) == "vertical"
    assert part_title_kind(section, (5000.0, 5000.0, 6000.0, 6000.0), dictionary) is None


def test_plan_with_a_schedule_beside_it_is_a_plan():
    dictionary = SheetDictionary(load_sheet_dictionary())
    prims = [
        Primitive(f"L{i}", "line", {"p1": (0.0, i * 100.0), "p2": (5000.0, i * 100.0)}, "A-WALL", (str(i),)) for i in range(200)
    ] + [poly(quarter(cx=i * 3000.0, cy=9000.0)) for i in range(6)] + [text("DOOR SCHEDULE", 0.0, -1000.0, 300.0)]
    guess = classify_sheet(prims, dictionary)
    if guess.by_text == "table":
        assert guess.kind == "plan"
    assert guess.door_arcs == 6
