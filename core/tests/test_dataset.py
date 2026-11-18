"""합성 4종 회귀 테스트. M1 완료 기준: 보고서에 그려지고 단위·축척 판정 통과."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace

import pytest
from conftest import DRAWINGS, NAMES

from cad2bim.ir import IR
from cad2bim.pipeline import build_ir, convert
from cad2bim.preprocess import preprocess
from cad2bim.report.svg import render_svg


@pytest.fixture(scope="module")
def processed(cfg):
    return {n: preprocess(DRAWINGS / f"{n}.dxf", cfg) for n in NAMES}


def truth(name: str) -> dict:
    return json.loads((DRAWINGS / f"{name}.truth.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", NAMES)
def test_scale_is_mm_from_dimensions(processed, name):
    scale = processed[name].scale
    assert scale.factor == 1.0
    assert scale.source == "dimension"
    assert scale.samples == 7


def test_a2_missing_insunits_is_reported(processed):
    assert "insunits_missing" in processed["A2_retail_messy"].warnings
    assert "insunits_missing" not in processed["A1_retail_clean_en"].warnings


@pytest.mark.parametrize("name", NAMES)
def test_frame_and_title_block_excluded_only(processed, name):
    pre = processed[name]
    assert len(pre.frames) == 1
    # 도곽, 표제란, 표제란 문자 3개, 평면 밖 주기 1개
    assert Counter(p.kind for p in pre.excluded) == {"polyline": 2, "text": 4}
    excluded_text = {p.geometry["text"] for p in pre.excluded if p.kind == "text"}
    assert not excluded_text & {r["name"] for r in truth(name)["rooms"]}


@pytest.mark.parametrize("name", NAMES)
def test_plan_covers_all_walls(processed, name):
    pre = processed[name]
    t = truth(name)
    xs = [p[0] for w in t["walls"] for p in (w["p1"], w["p2"])]
    ys = [p[1] for w in t["walls"] for p in (w["p1"], w["p2"])]
    x0, y0, x1, y1 = pre.plan_bbox
    assert x0 < min(xs) and y0 < min(ys) and x1 > max(xs) and y1 > max(ys)
    assert pre.origin_offset == (0.0, 0.0)


@pytest.mark.parametrize("name", ["A1_retail_clean_en", "A2_retail_messy"])
def test_room_labels_read_without_format_codes(processed, name):
    texts = {p.geometry["text"] for p in processed[name].primitives if p.kind == "text"}
    assert {"상가1", "남자화장실", "(68.75㎡)"} <= texts


def test_html_escapes_drawing_text(processed):
    pre = processed["A1_retail_clean_en"]
    template = next(p for p in pre.primitives if p.kind == "text")
    hostile = replace(template, geometry={**template.geometry, "text": "<script>x</script>"})
    assert "<script>x" not in render_svg([hostile], [], build_ir(pre, 3000))


def test_a2_duplicates_and_gaps_are_cleaned(processed):
    """정리 후 같은 줄의 선은 서로 겹치거나 5mm 이내로 붙어 있으면 안 된다."""
    pre = processed["A2_retail_messy"]
    assert pre.normalize_stats.merged_away > 0

    rows: dict[tuple, list[tuple[float, float]]] = {}
    for p in pre.primitives:
        if p.kind != "line" or p.layer != "0":
            continue
        (x1, y1), (x2, y2) = p.geometry["p1"], p.geometry["p2"]
        if x1 == x2:
            rows.setdefault(("v", round(x1)), []).append(tuple(sorted((y1, y2))))
        elif y1 == y2:
            rows.setdefault(("h", round(y1)), []).append(tuple(sorted((x1, x2))))
    for spans in rows.values():
        spans.sort()
        for (_, end), (start, _) in zip(spans, spans[1:]):
            assert start - end > 5.0


def test_a1_and_a2_door_swings_match(processed):
    """같은 평면이 블록(A1)과 분해 호(A2)로 그려졌다. 전개 결과가 같아야 한다."""

    def arcs(pre):
        return sorted(
            (round(a.geometry["center"][0]), round(a.geometry["center"][1]),
             round(a.geometry["start"][0]), round(a.geometry["start"][1]),
             round(a.geometry["end"][0]), round(a.geometry["end"][1]))
            for a in pre.primitives
            if a.kind == "arc"
        )

    a1, a2 = arcs(processed["A1_retail_clean_en"]), arcs(processed["A2_retail_messy"])
    assert len(a1) == 8
    assert a1 == a2


@pytest.mark.parametrize("name", ["A1_retail_clean_en", "B1_apt_clean_kr", "B2_apt_partial_code"])
def test_door_arc_center_is_truth_hinge_and_sweeps_toward_swing(processed, name):
    arcs = [p for p in processed[name].primitives if p.kind == "arc"]
    for door in (o for o in truth(name)["openings"] if o["kind"] == "door"):
        match = next(a for a in arcs if math.dist(a.geometry["center"], door["hinge"]) < 1)
        g = match.geometry
        assert g["radius"] == pytest.approx(door["width"])
        cx, cy = g["center"]
        # 호의 반시계 중점은 열리는 쪽에 있어야 한다.
        a0 = math.atan2(g["start"][1] - cy, g["start"][0] - cx)
        a1 = math.atan2(g["end"][1] - cy, g["end"][0] - cx)
        mid = a0 + ((a1 - a0) % (2 * math.pi)) / 2
        sx, sy = door["swing_dir"]
        assert math.cos(mid) * sx + math.sin(mid) * sy > 0


def test_b2_scaled_window_block_is_expanded(processed):
    pre = processed["B2_apt_partial_code"]
    refs = [p for p in pre.primitives if p.kind == "block_ref" and p.geometry["name"] == "W"]
    assert len(refs) == 8
    t = truth("B2_apt_partial_code")
    assert sorted(max(abs(r.geometry["xscale"]), abs(r.geometry["yscale"])) for r in refs) == sorted(
        o["width"] for o in t["openings"] if o["kind"] == "window"
    )
    window_lines = [p for p in pre.primitives if p.block_path == ("W",) and p.kind != "block_ref"]
    assert window_lines
    assert all(p.layer == "WIN" for p in window_lines)


def test_b2_nested_furniture_keeps_block_path(processed):
    paths = {p.block_path for p in processed["B2_apt_partial_code"].primitives}
    assert ("UNIT-FUR", "F-BED1") in paths


def test_b2_mislayered_hatch_is_kept(processed):
    hatches = [
        p for p in processed["B2_apt_partial_code"].primitives
        if p.kind == "hatch" and p.layer == "DIM"
    ]
    assert len(hatches) == 4


@pytest.mark.parametrize("name", NAMES)
def test_dimension_block_lines_do_not_become_primitives(processed, name):
    assert not any(p.block_path and p.block_path[0].startswith("*D") for p in processed[name].primitives)


@pytest.mark.parametrize("name", NAMES)
def test_convert_writes_valid_ir_and_report(cfg, tmp_path, name):
    result = convert(DRAWINGS / f"{name}.dxf", tmp_path, cfg, floor_height=3000)
    ir = IR.model_validate_json(result.ir_path.read_text(encoding="utf-8"))
    assert ir.source.file == f"{name}.dxf"
    assert len(ir.source.sha256) == 64

    html = result.report_path.read_text(encoding="utf-8")
    assert "<svg" in html and "<path" in html
    # 외부 리소스를 부르지 않아야 한다.
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in html
