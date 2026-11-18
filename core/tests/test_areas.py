"""면적표 대조. 표 읽기, 채점의 정직함, 내용 유출을 본다."""
from __future__ import annotations

import pytest
import shutil
import ezdxf
from conftest import DRAWINGS

from cad2bim.cli import main
from cad2bim.evaluation.areas import (
    column_hull_m2,
    compare_areas,
    floor_best_lines,
    floor_key,
    name_kinds,
    normalize_name,
    outer_area_m2,
    plan_title_floor,
    read_floor_column,
    read_table_rows,
    variant_lines,
)
from cad2bim.preprocess.primitive import Primitive


def text(value, x, y, height=100.0):
    return Primitive(
        f"T{x}{y}", "text",
        {"position": (x, y), "text": value, "height": height, "rotation": 0.0, "centered": False},
        "T", ("1",),
    )


def test_table_rows_read_name_and_values():
    prims = [
        text("구분", 0, 300), text("면적", 500, 300),
        text("사무실", 0, 200), text("120.50", 500, 200), text("(120.50㎡)", 800, 200),
        text("창 고", 0, 100), text("1,234.5", 500, 100),
        text("비고만 있는 행", 0, 0),
    ]
    values, stats = read_table_rows(prims)
    assert values["사무실"] == {120.5}
    assert values["창고"] == {1234.5}
    assert "비고만있는행" not in values
    assert stats.rows == 2


def test_small_numbers_and_codes_are_not_areas():
    values, _ = read_table_rows([text("층수", 0, 100), text("0.3", 500, 100), text("A-101", 800, 100)])
    assert values == {}


def test_empty_drawing_reports_zero():
    values, stats = read_table_rows([])
    assert values == {} and stats.texts == 0


@pytest.mark.parametrize(("raw", "expected"), [("창 고", "창고"), ("사무실-1", "사무실1"), ("(로비)", "로비")])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


TABLE = {"사무실": {100.0}, "창고": {2000.0}, "로비": {50.0}, "복도": {300.0}, "계단실": {700.0}}


def test_exact_areas_score_far_above_chance():
    rooms = [(name, next(iter(values))) for name, values in TABLE.items()]
    result = compare_areas(rooms, TABLE)
    assert result["name_found"] == 5
    assert result["within_5pct"] == 5
    assert result["median_error"] == 0
    assert result["chance_within_5pct"] < 5


def test_median_ratio_shows_direction_of_error():
    result = compare_areas([("사무실", 50.0), ("창고", 1000.0), ("로비", 25.0)], TABLE)
    assert result["median_ratio"] == pytest.approx(0.5)


def test_wrong_areas_score_low():
    result = compare_areas([("사무실", 180.0), ("창고", 900.0)], TABLE)
    assert result["within_5pct"] == 0
    assert result["median_error"] > 0.4


def test_room_not_in_table_counts_only_in_total():
    result = compare_areas([("없는실", 10.0)], TABLE)
    assert result["rooms"] == 1 and result["name_found"] == 0
    assert result["median_error"] is None


def test_shared_name_uses_closest_value():
    result = compare_areas([("사무실", 201.0)], {"사무실": {100.0, 200.0}, "창고": {5.0}})
    assert result["within_2pct"] == 1


@pytest.mark.parametrize(
    ("text_", "key"),
    [("지상 3층 평면도", "F3"), ("3F", "F3"), ("지하1층", "B1"), ("B2F", "B2"), ("1 층", "F1"), ("계단 상세도", None)],
)
def test_floor_key(text_, key):
    assert floor_key(text_) == key


def test_name_kinds_tell_floor_tables_from_room_tables():
    kinds = name_kinds(["지상1층", "지하1층", "합계", "연면적", "사무실", "옥탑"])
    assert kinds == {"floor": 3, "total": 2, "other": 1}


def test_outer_area_of_a_closed_rectangle_of_walls():
    wall = lambda a, b: {"start": a, "end": b, "thickness": 200}
    ir = {
        "walls": [wall((0, 0), (10000, 0)), wall((10000, 0), (10000, 5000)),
                  wall((10000, 5000), (0, 5000)), wall((0, 5000), (0, 0))],
        "columns": [],
    }
    assert outer_area_m2(ir) == pytest.approx(10.2 * 5.2, abs=1.0)


def test_outer_area_closes_small_gaps():
    wall = lambda a, b: {"start": a, "end": b, "thickness": 200}
    ir = {
        "walls": [wall((0, 0), (4000, 0)), wall((5000, 0), (10000, 0)), wall((10000, 0), (10000, 5000)),
                  wall((10000, 5000), (0, 5000)), wall((0, 5000), (0, 0))],
        "columns": [],
    }
    assert outer_area_m2(ir) == pytest.approx(10.2 * 5.2, abs=1.0)


def test_confirmed_walls_exclude_estimated_ones():
    def wall(a, b, reasons=()):
        return {"start": a, "end": b, "thickness": 200, "confidence": {"reasons": list(reasons)}}

    ir = {
        "walls": [wall((0, 0), (10000, 0)), wall((10000, 0), (10000, 5000)), wall((10000, 5000), (0, 5000)),
                  wall((0, 5000), (0, 0)), wall((20000, 0), (30000, 0), ["wall_layer_inferred"])],
        "columns": [],
    }
    assert outer_area_m2(ir, confirmed_walls_only=True) == pytest.approx(10.2 * 5.2, abs=1.0)
    assert outer_area_m2(ir) > outer_area_m2(ir, confirmed_walls_only=True)


def test_column_hull_area():
    ir = {"columns": [{"center": (0, 0)}, {"center": (10000, 0)}, {"center": (10000, 5000)}, {"center": (0, 5000)}]}
    assert column_hull_m2(ir) == pytest.approx(50)
    assert column_hull_m2({"columns": [{"center": (0, 0)}]}) is None


def test_variant_lines_report_only_codes_and_numbers():
    plans = [("F1", {"all": 133.0, "columns": 100.0}), ("F2", {"all": 50.0})]
    lines = variant_lines(plans, {"F1": {100.0}})
    assert "variant[all]: n=1 median_ratio=1.330 within_10pct=0 within_25pct=0 chance_10pct=0 chance_25pct=0" in lines
    assert "floor[F1]: plans=1 table_values=1 all=1.33 columns=1.00" in lines


def test_outer_area_without_elements_is_none():
    assert outer_area_m2({"walls": [], "columns": []}) is None


def area_table():
    """머리글: 층 / 바닥면적 / 비율. 층 행마다 비율이 면적에 가까운 값이어도 고르면 안 된다."""
    rows = [text("층", 0, 500), text("바닥면적", 1000, 500), text("비율", 2000, 500)]
    for i, (floor, area, ratio) in enumerate((("지하1층", "5000.5", "4900"), ("1층", "6000.25", "5900"), ("2층", "6100", "6000"))):
        y = 400 - i * 100
        rows += [text(floor, 0, y), text(area, 1000, y), text(ratio, 2000, y)]
    return rows


def test_floor_column_reads_only_the_floor_area_column():
    values, stats = read_floor_column(area_table())
    assert values == {"B1": {5000.5}, "F1": {6000.25}, "F2": {6100.0}}
    assert stats["column_read"] == 3 and stats["no_column"] == 0


def test_floor_column_without_header_reads_nothing():
    prims = [text("층", 0, 500), text("비율", 1000, 500), text("1층", 0, 400), text("100", 1000, 400)]
    values, stats = read_floor_column(prims)
    assert values == {}
    assert stats["floor_rows"] == 1 and stats["no_column"] == 1


def test_floor_column_ignores_header_below_the_row():
    prims = [text("1층", 0, 400), text("100", 1000, 400), text("바닥면적", 1000, 300)]
    assert read_floor_column(prims)[0] == {}


def test_floor_column_rejects_a_value_far_from_the_header():
    prims = [text("바닥면적", 1000, 500), text("1층", 0, 400), text("100", 9000, 400), text("200", 12000, 400)]
    assert read_floor_column(prims)[0] == {}


def test_long_text_with_a_floor_word_is_not_a_floor_row():
    prims = [text("바닥면적", 1000, 500), text("1층 평면도 참조 바람", 0, 400), text("100", 1000, 400)]
    values, stats = read_floor_column(prims)
    assert values == {} and stats["floor_rows"] == 0


def merged_header_table(total_label="합계", broken=False):
    """합쳐진 머리글. 용도별 값, 소계, 합계 순이고 값이 없는 칸은 '-'다."""
    cells = [text("용도", 1000, 800), text("면적", 2000, 800), text(total_label, 4000, 800), text("비고", 5000, 800)]
    rows = (("지하1층", "40", "60", "100", "-"), ("1층", "200", "300", "500", "500"), ("2층", "150", "250", "400", "400"))
    for i, (floor, a, b, total, extra) in enumerate(rows):
        y = 500 - i * 100
        shown = "999" if broken else total
        cells += [text(floor, 0, y), text(a, 1000, y), text(b, 2000, y), text(shown, 4000, y), text(extra, 5000, y)]
    cells += [text("소계", 0, 200), text("1000", 4000, 200)]
    return cells


def test_total_column_is_found_in_a_merged_header_table():
    # 비고 열은 값이 빠진 층이 있어 후보가 아니다.
    values, stats = read_floor_column(merged_header_table())
    assert values == {"B1": {100.0}, "F1": {500.0}, "F2": {400.0}}
    assert stats["by_total_column"] == 1 and stats["column_read"] == 3


def test_subtotal_row_confirms_the_column():
    _, stats = read_floor_column(merged_header_table())
    assert stats["subtotal_checked"] == 1 and stats["subtotal_ok"] == 1


def test_total_column_that_is_not_a_sum_is_rejected():
    values, stats = read_floor_column(merged_header_table(broken=True))
    assert values == {} and stats["no_column"] == 3


def test_two_candidate_columns_read_nothing():
    prims = [text("합계", 3000, 500), text("소계", 4000, 500)]
    for i, floor in enumerate(("1층", "2층")):
        y = 400 - i * 100
        prims += [text(floor, 0, y), text("10", 1000, y), text("20", 2000, y), text("30", 3000, y), text("30", 4000, y)]
    values, stats = read_floor_column(prims)
    assert values == {} and stats["ambiguous"] == 2


def test_oversized_title_is_not_a_column_header():
    prims = [text("층별 바닥면적", 1000, 500, height=200.0), text("1층", 0, 400), text("100", 1000, 400)]
    assert read_floor_column(prims)[0] == {}


def test_area_command_prints_no_floor_column_values(tmp_path, capsys):
    def build(msp):
        for cell in merged_header_table():
            x, y = cell.geometry["position"]
            msp.add_text(cell.geometry["text"], height=100).set_placement((x, y))

    shutil.copy(DRAWINGS / "B1_apt_clean_kr.dxf", tmp_path / "SECRET 평면도.dxf")
    save(tmp_path / "SECRET 개요.dxf", build)
    assert main(["audit", str(tmp_path), "--match", "평면도", "--quick"]) == 0
    capsys.readouterr()
    assert main(["areas", str(tmp_path), "--table-match", "개요", "--plan-match", "평면도"]) == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "by_total_column=1" in output and "subtotal_ok=1/1" in output
    assert "SECRET" not in output and "합계" not in output and "500.0" not in output


def test_plan_title_is_read_inside_the_region_only():
    prims = [text("지상 1층 평면도", 500, -100), text("지상 2층 평면도", 5000, -100)]
    assert plan_title_floor(prims, (0, 0, 1000, 1000)) == "F1"
    assert plan_title_floor(prims, (4500, 0, 5500, 1000)) == "F2"
    assert plan_title_floor(prims, (0, 5000, 1000, 6000)) is None


def test_largest_title_wins_and_other_words_are_not_titles():
    prims = [text("3층 평면도", 0, 0, height=100.0), text("1층 평면도", 0, 100, height=300.0), text("2층 단면도", 0, 200, height=900.0)]
    assert plan_title_floor(prims, None) == "F1"


def test_floor_best_uses_the_largest_plan_of_each_floor_and_counts_floors():
    plans = [("F1", {"all": 10.0, "columns": 90.0}), ("F1", {"all": 100.0}), ("F2", {"all": 400.0})]
    lines = floor_best_lines(plans, {"F1": {100.0}, "F2": {200.0}})
    assert lines[0].startswith("floor_best[all]: floors=2 median_ratio=1.500 within_10pct=1 within_25pct=1")
    assert "floor_best_row[F1]: parts=2 all=1.00 columns=0.90" in lines


def test_floor_best_skips_floors_without_a_single_table_value():
    assert floor_best_lines([("F1", {"all": 100.0})], {"F1": {100.0, 200.0}})[0] == "floor_best[all]: floors=0"


def test_area_command_uses_plan_titles_without_printing_them(tmp_path, capsys):
    def table(msp):
        for cell in merged_header_table():
            x, y = cell.geometry["position"]
            msp.add_text(cell.geometry["text"], height=100).set_placement((x, y))

    doc = ezdxf.readfile(DRAWINGS / "B1_apt_clean_kr.dxf")
    doc.modelspace().add_text("SECRET 지상 1층 평면도", height=100).set_placement((0, 0))
    doc.saveas(tmp_path / "SECRET 평면도.dxf")
    save(tmp_path / "SECRET 개요.dxf", table)
    assert main(["audit", str(tmp_path), "--match", "평면도", "--quick"]) == 0
    capsys.readouterr()
    assert main(["areas", str(tmp_path), "--table-match", "개요", "--plan-match", "평면도"]) == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "plan_titles: titled=1 no_title=0 same_as_file=0 differs=0 no_file_floor=1" in output
    assert "floor_best[all]: floors=1" in output
    assert "SECRET" not in output and "평면도" not in output


# ---------------------------------------------------------------- 명령과 유출
def save(path, build):
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 4
    build(doc.modelspace())
    doc.saveas(path)


def table_doc(msp):
    rows = [("침실2", "15.64"), ("침실3", "11.56"), ("SECRET실", "99.99")]
    for i, (name, value) in enumerate(rows):
        msp.add_text(name, height=200).set_placement((0, i * 500))
        msp.add_text(value, height=200).set_placement((3000, i * 500))


def test_area_command_matches_rooms_without_printing_names(tmp_path, capsys):
    shutil.copy(DRAWINGS / "B1_apt_clean_kr.dxf", tmp_path / "SECRET 평면도.dxf")
    save(tmp_path / "SECRET 면적근거표.dxf", table_doc)
    assert main(["audit", str(tmp_path), "--match", "평면도", "--quick"]) == 0
    capsys.readouterr()
    assert main(["areas", str(tmp_path), "--table-match", "근거표", "--plan-match", "평면도"]) == 0
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "SECRET" not in output and "침실" not in output and "15.64" not in output
    assert "table_files=1" in output
    assert "name_found=2" in output and "within_5pct=2" in output


def test_area_command_needs_a_table_and_a_previous_audit(tmp_path, capsys):
    assert main(["areas", str(tmp_path), "--table-match", "근거표", "--plan-match", "평면도"]) == 2
    assert "table-match" in capsys.readouterr().err
    save(tmp_path / "면적근거표.dxf", table_doc)
    assert main(["areas", str(tmp_path), "--table-match", "근거표", "--plan-match", "평면도"]) == 2
    assert "먼저 audit" in capsys.readouterr().err
