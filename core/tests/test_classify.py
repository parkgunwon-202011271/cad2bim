"""도면 종류 분류. 파일명을 보지 않고 내용으로 판단하는지, 채점이 정직한지 본다."""
from __future__ import annotations

import math
import shutil

import ezdxf
import pytest
from conftest import DRAWINGS, NAMES

from cad2bim.classify import SheetDictionary, classify_sheet, label_from_name
from cad2bim.cli import main
from cad2bim.config import load_sheet_dictionary
from cad2bim.preprocess.explode import extract_primitives


@pytest.fixture(scope="module")
def dictionary() -> SheetDictionary:
    return SheetDictionary(load_sheet_dictionary())


def prims_of(build):
    doc = ezdxf.new("R2018", setup=True)
    build(doc, doc.modelspace())
    return extract_primitives(doc, 1.0)[0]


def lines(msp, count=30, length=30000):
    for i in range(count):
        msp.add_line((0, i * 500), (length, i * 500))


def title(msp, text):
    msp.add_text(text, height=500).set_placement((28000, 500))


@pytest.mark.parametrize(
    ("name", "kind"),
    [("지상 1층 평면도.dwg", "plan"), ("A-101 FLOOR PLAN.dwg", "plan"), ("북측면도.dwg", "vertical"),
     ("주단면도.dwg", "vertical"), ("교통체계도.dwg", "site"), ("전체 배치도.dwg", "site"), ("면적 근거표.dwg", "table"),
     ("창호 일람표.dwg", "table"), ("계단 상세도.dwg", "detail"), ("확대평면도.dwg", "detail"), ("배치평면도.dwg", "site"),
     ("SITE PLAN.dwg", "site"), ("도면.dwg", None), ("PLANT.dwg", None), ("A-001.dwg", None)],
)
def test_label_from_name(dictionary, name, kind):
    assert label_from_name(name, dictionary) == kind


@pytest.mark.parametrize(
    ("words", "kind"),
    [("지상 3층 평면도", "plan"), ("북측 입면도", "vertical"), ("A-A 단면도", "vertical"),
     ("전체 배치도", "site"), ("면적 근거표", "table"), ("계단 상세도", "detail")],
)
def test_title_text_decides_the_kind(dictionary, words, kind):
    guess = classify_sheet(prims_of(lambda d, m: (lines(m), title(m, words))), dictionary)
    assert guess.kind == kind
    assert guess.by_text == kind


def test_passing_mention_in_the_body_is_not_a_title(dictionary):
    def build(doc, msp):
        lines(msp)
        msp.add_text("평면도 참조", height=100).set_placement((15000, 7000))

    assert classify_sheet(prims_of(build), dictionary).by_text == "unknown"


@pytest.mark.parametrize(
    "words",
    ["대지위치", "대지면적", "범례", "개요", "현황", "교통", "도면번호", "층별 개요", "평면도 참조", "단면 상세 참조"],
)
def test_title_block_fields_and_mentions_are_not_titles(dictionary, words):
    """실제 도면에서 입면·상세도가 배치로 잘못 분류된 원인으로 본 글자들."""
    guess = classify_sheet(prims_of(lambda d, m: (lines(m), title(m, words))), dictionary)
    assert guess.by_text == "unknown"


@pytest.mark.parametrize(
    ("words", "kind"),
    [("지상 3층 평면도 S:1/300", "plan"), ("1층 평면도 (A1)", "plan"), ("북측면도 SCALE 1/200", "vertical"),
     ("주단면도-1", "vertical"), ("교통체계도", "site"), ("주차 계획도", "site"), ("FLOOR PLAN", "plan"),
     ("SECTION A", "vertical"), ("면적 근거표", "table"), ("확대 평면도", "detail")],
)
def test_title_forms_with_scale_or_number(dictionary, words, kind):
    guess = classify_sheet(prims_of(lambda d, m: (lines(m), title(m, words))), dictionary)
    assert guess.by_text == kind


def test_title_in_the_middle_of_a_very_wide_sheet_counts(dictionary):
    """실제 평면도 D106처럼 도면이 길면 제목이 가장자리에 없다."""
    def build(doc, msp):
        lines(msp, 30, 1_000_000)
        msp.add_text("지상 1층 평면도", height=2000).set_placement((500_000, 7000))

    assert classify_sheet(prims_of(build), dictionary).by_text == "plan"


def test_largest_title_wins_over_small_key_plan(dictionary):
    def build(doc, msp):
        lines(msp)
        msp.add_text("배치도", height=50).set_placement((28000, 500))
        msp.add_text("지상 1층 평면도", height=800).set_placement((15000, -1000))

    assert classify_sheet(prims_of(build), dictionary).by_text == "plan"


def test_body_mention_loses_to_real_title(dictionary):
    def build(doc, msp):
        lines(msp)
        title(msp, "북측 입면도")
        msp.add_text("평면도 참조", height=100).set_placement((15000, 7000))

    assert classify_sheet(prims_of(build), dictionary).kind == "vertical"


def contour(msp, y):
    points = [(i * 400, y + 2000 * math.sin(i / 5)) for i in range(60)]
    msp.add_lwpolyline(points)


def test_shape_finds_site_from_contours(dictionary):
    def build(doc, msp):
        for k in range(8):
            contour(msp, k * 1500)
        msp.add_line((0, 0), (500, 0))

    guess = classify_sheet(prims_of(build), dictionary)
    assert guess.by_shape == "site"
    assert guess.by_text == "unknown"
    assert guess.kind == "site"


def test_shape_finds_vertical_from_level_marks(dictionary):
    def build(doc, msp):
        lines(msp)
        for i, mark in enumerate(("1FL+0", "2FL+6000", "GL-150", "FL+12000")):
            msp.add_text(mark, height=200).set_placement((31000, i * 3000))

    guess = classify_sheet(prims_of(build), dictionary)
    assert guess.level_marks >= 3
    assert guess.by_shape == "vertical"


def test_shape_finds_table_from_dense_text(dictionary):
    def build(doc, msp):
        lines(msp, 20, 5000)
        for i in range(60):
            msp.add_text(f"항목{i}", height=100).set_placement((i % 6 * 800, i // 6 * 500))

    assert classify_sheet(prims_of(build), dictionary).by_shape == "table"


def test_shape_finds_plan_from_door_arcs(dictionary):
    def build(doc, msp):
        lines(msp)
        for i in range(4):
            msp.add_arc((i * 3000, 500), 900, 0, 90)

    guess = classify_sheet(prims_of(build), dictionary)
    assert guess.door_arcs == 4
    assert guess.by_shape == "plan"


def test_shape_finds_plan_from_two_way_grid(dictionary):
    def build(doc, msp):
        lines(msp)
        for i in range(4):
            msp.add_circle((i * 9000, -2000), 400)
            msp.add_text(f"X{i}", height=200).set_placement((i * 9000, -2000))
            msp.add_circle((-2000, i * 4500), 400)
            msp.add_text(f"Y{i}", height=200).set_placement((-2000, i * 4500))

    guess = classify_sheet(prims_of(build), dictionary)
    assert guess.grid_axes == 2
    assert guess.by_shape == "plan"


def test_one_way_grid_alone_is_not_a_plan(dictionary):
    def build(doc, msp):
        lines(msp)
        for i in range(5):
            msp.add_circle((i * 7000, -2000), 400)
            msp.add_text(f"X{i}", height=200).set_placement((i * 7000, -2000))

    guess = classify_sheet(prims_of(build), dictionary)
    assert guess.grid_axes == 1
    assert guess.by_shape == "unknown"


def test_scale_does_not_change_the_result(dictionary):
    def build(doc, msp):
        lines(msp)
        title(msp, "지상 1층 평면도")
        for i in range(4):
            msp.add_arc((i * 3000, 500), 900, 0, 90)

    prims = prims_of(build)
    small = classify_sheet([p.transformed(0.001, 0, 0) for p in prims], dictionary)
    assert small == classify_sheet(prims, dictionary)


def sheet_in_layout(doc, msp, title_text="지상 2층 평면도"):
    lines(msp, 30, 100000)
    layout = doc.layouts.new("A-101")
    layout.add_lwpolyline([(0, 0), (840, 0), (840, 594), (0, 594)], close=True)
    layout.add_text(title_text, height=8).set_placement((700, 20))
    layout.add_text("대지위치", height=3).set_placement((700, 60))
    layout.add_viewport(center=(350, 300), size=(600, 400), view_center_point=(20000, 7000), view_height=20000)


def extract(build):
    doc = ezdxf.new("R2018", setup=True)
    build(doc, doc.modelspace())
    return extract_primitives(doc, 1.0)


def test_layout_title_decides_the_kind(dictionary):
    """실제 도면의 표제란은 레이아웃에 있었다. 모델 공간만 보면 판단하지 못한다."""
    model, stats = extract(sheet_in_layout)
    assert classify_sheet(model, dictionary).by_text == "unknown"
    assert classify_sheet(model, dictionary, stats.paper).kind == "plan"


def test_layout_title_beats_model_mentions(dictionary):
    def build(doc, msp):
        sheet_in_layout(doc, msp, "북측 입면도")
        msp.add_text("지상 1층 평면도", height=500).set_placement((100, 100))

    model, stats = extract(build)
    assert classify_sheet(model, dictionary, stats.paper).kind == "vertical"


def test_empty_model_space_is_classified_from_layout(dictionary):
    def build(doc, msp):
        layout = doc.layouts.new("A-101")
        layout.add_lwpolyline([(0, 0), (840, 0), (840, 594), (0, 594)], close=True)
        layout.add_text("면적 근거표", height=8).set_placement((700, 20))

    model, stats = extract(build)
    assert model == []
    assert classify_sheet(model, dictionary, stats.paper).kind == "table"


def test_viewport_range_is_read_in_model_coordinates():
    _, stats = extract(sheet_in_layout)
    assert stats.layouts >= 1
    assert len(stats.viewports) == 1
    x0, y0, x1, y1 = stats.viewports[0]
    assert (x0 + x1) / 2 == pytest.approx(20000)
    assert (y0 + y1) / 2 == pytest.approx(7000)
    assert y1 - y0 == pytest.approx(20000)
    assert x1 - x0 == pytest.approx(30000)


def test_layout_contents_do_not_enter_recognition(tmp_path):
    from cad2bim.config import load_defaults
    from cad2bim.preprocess import preprocess

    path = tmp_path / "x.dxf"
    save(path, sheet_in_layout)
    pre = preprocess(path, load_defaults(), unit_scale=1.0)
    assert not any(p.kind == "text" for p in pre.primitives)


def test_key_plan_is_not_a_title(dictionary):
    guess = classify_sheet(prims_of(lambda d, m: (lines(m), title(m, "KEY PLAN"))), dictionary)
    assert guess.by_text == "unknown"


def test_plan_guess_from_shape_alone_is_not_used_as_final(dictionary):
    def build(doc, msp):
        lines(msp)
        for i in range(4):
            msp.add_arc((i * 3000, 500), 900, 0, 90)

    guess = classify_sheet(prims_of(build), dictionary)
    assert guess.by_shape == "plan"
    assert guess.kind == "unknown"


def test_empty_drawing_is_unknown(dictionary):
    assert classify_sheet([], dictionary).kind == "unknown"


@pytest.mark.parametrize("name", NAMES)
def test_synthetic_drawings_are_plans_by_text_and_shape(dictionary, name):
    prims, _ = extract_primitives(ezdxf.readfile(DRAWINGS / f"{name}.dxf"), 1.0)
    guess = classify_sheet(prims, dictionary)
    assert guess.kind == "plan"
    assert guess.by_shape == "plan"


# ---------------------------------------------------------------- audit 채점
def save(path, build):
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 4
    build(doc, doc.modelspace())
    doc.saveas(path)


def output(capsys):
    captured = capsys.readouterr()
    return captured.out + captured.err


def elevation(doc, msp):
    lines(msp)
    title(msp, "북측 입면도")
    for i, mark in enumerate(("1FL+0", "2FL+6000", "GL-150")):
        msp.add_text(mark, height=200).set_placement((31000, i * 3000))


def test_classifier_does_not_read_the_file_name(tmp_path, capsys):
    """내용은 입면도인데 파일명만 평면도면 오답으로 잡혀야 한다. 파일명을 봤다면 맞게 나온다."""
    save(tmp_path / "SECRET 평면도.dxf", elevation)
    assert main(["audit", str(tmp_path), "--quick"]) == 0
    text = output(capsys)
    assert "sheet_confusion_text: plan>vertical=1" in text
    assert "sheet_accuracy: labeled=1 text=0.000" in text
    assert "SECRET" not in text


def test_accuracy_is_reported_for_text_and_shape_separately(tmp_path, capsys):
    shutil.copy(DRAWINGS / "A1_retail_clean_en.dxf", tmp_path / "SECRET 평면도.dxf")
    save(tmp_path / "SECRET 측면도.dxf", elevation)
    save(tmp_path / "SECRET 무제.dxf", lambda d, m: lines(m))
    main(["audit", str(tmp_path), "--quick"])
    text = output(capsys)
    # 이름순으로 무제(D01), 측면도(D02, 시험용), 평면도(D03)
    assert "sheet_accuracy: labeled=1 text=1.000 shape=1.000 holdout_labeled=1 holdout_text=1.000" in text
    assert "unlabeled=1" in text
    assert "sheet_features[plan] n=1" in text
    assert "sheet: kind=plan" in text
    assert "SECRET" not in text and "평면도" not in text


def test_classify_only_skips_recognition(tmp_path, capsys, monkeypatch):
    import cad2bim.evaluation.audit as audit

    def forbidden(*args, **kwargs):
        raise AssertionError("변환을 돌리면 안 된다")

    monkeypatch.setattr(audit, "convert", forbidden)
    shutil.copy(DRAWINGS / "A1_retail_clean_en.dxf", tmp_path / "SECRET 평면도.dxf")
    assert main(["audit", str(tmp_path), "--classify-only"]) == 0
    text = output(capsys)
    assert "failed=0" in text and "[D01] classified" in text
    assert "sheet_accuracy: labeled=1 text=1.000 shape=1.000" in text
    assert "SECRET" not in text


def test_classify_only_lists_many_drawings_one_line_each(tmp_path, capsys):
    for i in range(6):
        shutil.copy(DRAWINGS / "A1_retail_clean_en.dxf", tmp_path / f"SECRET{i} 평면도.dxf")
    main(["audit", str(tmp_path), "--classify-only", "--jobs", "2"])
    text = output(capsys)
    assert "D01 plan text=plan shape=plan label=plan" in text
    assert "D02 " not in text  # 시험용 도면의 개별 결과는 내지 않는다
    assert "holdout_labeled=3" in text
    assert "SECRET" not in text and "평면도" not in text


def test_audit_reports_layout_numbers_without_content(tmp_path, capsys):
    save(tmp_path / "SECRET.dxf", lambda d, m: sheet_in_layout(d, m, "SECRET 평면도"))
    main(["audit", str(tmp_path), "--classify-only"])
    text = output(capsys)
    assert "layout_summary:" in text and "with_viewport=1" in text
    assert "viewports=1" in text and "SECRET" not in text


def test_drawing_that_fails_later_is_still_classified(tmp_path, capsys):
    """단위 판정에 실패해도 분류는 돼야 한다. 표나 표지가 흔히 그렇다."""
    def build(doc, msp):
        doc.header["$INSUNITS"] = 0
        lines(msp)
        title(msp, "면적 근거표")

    doc = ezdxf.new("R2018", setup=True)
    build(doc, doc.modelspace())
    doc.saveas(tmp_path / "SECRET 근거표.dxf")
    main(["audit", str(tmp_path), "--quick"])
    text = output(capsys)
    assert "UnitScaleError" in text
    assert "sheet: kind=table" in text
    assert "sheet_accuracy: labeled=1 text=1.000" in text
