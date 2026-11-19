"""단면도·입면도 층 레벨 표시 읽기(4-b). 층 이름 글자와 가장 가까운 FH/FL 글자를 짝짓는다."""
from __future__ import annotations

from cad2bim.building import merge_section_levels
from cad2bim.classify.floors import ROOF, agreed_levels, read_level_marks
from cad2bim.preprocess.primitive import Primitive


def text(value, x, y, h=300.0):
    return Primitive(f"T{value}{x}{y}", "text", {"position": (x, y), "text": value, "height": h, "rotation": 0.0, "centered": False}, "L", ("1",))


def mark(name, level, y, x=0.0):
    return [text(name, x, y + 400), text(level, x, y)]


def test_floor_names_pair_with_the_nearest_level():
    prims = mark("지하1층", "FH:100.40", 0) + mark("1층", "FH:105.50", 5000) + mark("지붕", "FH:139.75", 40000)
    prims.append(text("FL:99.30", 90000, 0))  # 이름 없는 대지 레벨은 버린다
    assert read_level_marks(prims) == {"B1": [100400.0], "F1": [105500.0], ROOF: [139750.0]}


def test_mm_with_thousand_separator_and_far_text_is_ignored():
    prims = [text("2층", 0, 0), text("FH:6,280", 0, -400), text("3층", 0, 10000), text("FH:119.75", 50000, 10000)]
    assert read_level_marks(prims) == {"F2": [6280.0]}


def test_agreement_needs_repeats_unless_unique():
    assert agreed_levels({"F1": [105500.0, 105500.0, 99000.0], "F2": [109750.0], "F3": [1.0, 2.0]}) == {"F1": 105500.0, "F2": 109750.0}


def test_sections_replace_notes_only_when_they_cover_more_floors():
    notes = {"F1": 0.0, "F2": 4500.0, "F3": 9000.0}
    assert merge_section_levels(notes, {"F1": 105500.0, "F2": 109750.0}, ["F1", "F2", "F3"]) is None
    assert notes == {"F1": 0.0, "F2": 4500.0, "F3": 9000.0}
    notes = {}
    levels = {"B1": 100400.0, "F1": 105500.0, "F2": 109750.0, "F3": 119750.0, "F4": 129750.0, ROOF: 139750.0}
    assert merge_section_levels(notes, levels, ["B1", "F1", "F2", "F3", "F4"]) == "sections"
    assert notes["F5"] == 139750.0 and ROOF not in notes  # 지붕은 맨 위층 다음 바닥
