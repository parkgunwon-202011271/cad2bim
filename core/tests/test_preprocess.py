"""합성 세트에 없는 함정을 소형 DXF로 시험한다."""
from __future__ import annotations

import math
import time

import ezdxf
import pytest
from conftest import DRAWINGS, NAMES, add_dimensions, add_plan

from ezdxf.xclip import XClip

from cad2bim.errors import RegionAmbiguousError, UnitScaleError, XrefMissingError
from cad2bim.preprocess import preprocess
from cad2bim.preprocess.explode import extract_primitives, parse_dimension_value
from cad2bim.preprocess.normalize import normalize
from cad2bim.preprocess.primitive import Primitive
from cad2bim.preprocess.regions import select_plan, strip_frames
from cad2bim.preprocess.units import decide_scale


def line(p1, p2, layer="W", pid="P1"):
    return Primitive(pid, "line", {"p1": p1, "p2": p2}, layer, (pid,))


# ---------------------------------------------------------------- 단위·축척
def test_metre_drawing_without_insunits_is_scaled_from_dimensions(make_dxf, cfg):
    def build(doc, msp):
        add_plan(msp, size=10.0)
        for i in range(4):
            dim = msp.add_linear_dim(base=(i * 6, -1), p1=(i * 6, 0), p2=((i + 1) * 6, 0),
                                     override={"dimlfac": 1000.0, "dimdec": 0})
            dim.render()

    pre = preprocess(make_dxf(build, units=0), cfg)
    assert pre.scale.factor == 1000.0
    assert pre.scale.source == "dimension"
    assert pre.plan_bbox[2] - pre.plan_bbox[0] > 9000


def test_wrong_insunits_loses_to_dimensions(make_dxf, cfg):
    def build(doc, msp):
        add_plan(msp, size=10.0)
        for i in range(4):
            msp.add_linear_dim(base=(i * 6, -1), p1=(i * 6, 0), p2=((i + 1) * 6, 0),
                               override={"dimlfac": 1000.0, "dimdec": 0}).render()

    pre = preprocess(make_dxf(build, units=4), cfg)
    assert pre.scale.factor == 1000.0
    assert "insunits_conflict" in pre.warnings


def dim(value, measured, pid="D"):
    return Primitive(
        pid, "dimension",
        {"p1": (0, 0), "p2": (measured, 0), "position": (0, -1), "measured": measured, "value": value},
        "DIM", (pid,),
    )


def test_scattered_dimensions_require_user_input(cfg):
    prims = [dim(6000, 6000), dim(3000, 1500), dim(1000, 5000), dim(800, 90)]
    with pytest.raises(UnitScaleError, match="--unit-scale") as info:
        decide_scale(prims, 4, cfg["units"])
    assert info.value.detail.startswith("scattered_n4_share")


def test_one_hand_edited_dimension_is_tolerated(cfg):
    prims = [dim(6000, 6000), dim(3000, 3000), dim(4500, 4500), dim(1200, 1200), dim(999, 300)]
    assert decide_scale(prims, 0, cfg["units"]).factor == 1.0


def test_majority_on_common_scale_passes_with_warning(cfg):
    """상세도가 섞여 실제 도면에서 69~73%만 일치한 경우. 1:1이면 받고 표시한다."""
    prims = [dim(6000, 6000)] * 7 + [dim(1000, 200)] * 3
    decision = decide_scale(prims, 4, cfg["units"])
    assert decision.factor == 1.0
    assert "scale_mixed" in decision.warnings


def test_majority_on_unusual_scale_still_stops(cfg):
    prims = [dim(3700, 1000)] * 7 + [dim(1000, 200)] * 3
    with pytest.raises(UnitScaleError):
        decide_scale(prims, 4, cfg["units"])


def test_no_dimensions_falls_back_to_insunits_with_warning(cfg):
    decision = decide_scale([], 5, cfg["units"])
    assert decision.factor == 10.0
    assert decision.source == "insunits"
    assert "scale_unverified" in decision.warnings


def test_no_evidence_stops(cfg):
    with pytest.raises(UnitScaleError) as info:
        decide_scale([], 0, cfg["units"])
    assert info.value.detail == "no_evidence_n0"


def test_user_scale_wins(cfg):
    prims = [dim(6000, 6000)] * 4
    assert decide_scale(prims, 4, cfg["units"], override=25.4).source == "user"


@pytest.mark.parametrize(
    ("text", "value"),
    [("6000", 6000), ("6,000", 6000), ("1,234.5", 1234.5), ("EQ", None), ("6000 (현장)", None), ("", None)],
)
def test_parse_dimension_value(text, value):
    assert parse_dimension_value(text) == value


# ---------------------------------------------------------------- 블록 전개
def test_layer_zero_inherits_insert_layer_and_nested_transform(make_dxf, cfg):
    def build(doc, msp):
        inner = doc.blocks.new("INNER")
        inner.add_line((0, 0), (100, 0))  # 레이어 0 → 상속
        inner.add_line((0, 0), (0, 50), dxfattribs={"layer": "FIXED"})
        outer = doc.blocks.new("OUTER")
        outer.add_blockref("INNER", (1000, 0))
        msp.add_blockref("OUTER", (0, 5000), dxfattribs={"layer": "FUR", "rotation": 90})

    doc = ezdxf.readfile(make_dxf(build))
    prims, _ = extract_primitives(doc, 1.0)
    lines = [p for p in prims if p.kind == "line"]
    assert {p.layer for p in lines} == {"FUR", "FIXED"}
    assert all(p.block_path == ("OUTER", "INNER") for p in lines)
    first = next(p for p in lines if p.layer == "FUR")
    assert first.geometry["p1"] == pytest.approx((0, 6000))
    assert first.geometry["p2"] == pytest.approx((0, 6100))
    # 전개된 요소는 모델 공간의 INSERT 핸들로 추적된다.
    insert_handle = doc.modelspace()[0].dxf.handle
    assert all(p.source_handles == (insert_handle,) for p in lines)


def test_mirrored_arc_is_stored_counter_clockwise(make_dxf):
    def build(doc, msp):
        block = doc.blocks.new("DOOR")
        block.add_arc((0, 0), 900, 0, 90)
        msp.add_blockref("DOOR", (0, 0), dxfattribs={"yscale": -1})

    prims, _ = extract_primitives(ezdxf.readfile(make_dxf(build)), 1.0)
    arc = next(p for p in prims if p.kind == "arc").geometry
    # 원래 1사분면 호가 y 반전으로 4사분면이 된다. 반시계면 (0,-900) → (900,0).
    assert arc["start"] == pytest.approx((0, -900), abs=1e-6)
    assert arc["end"] == pytest.approx((900, 0), abs=1e-6)


def test_non_uniform_scale_turns_arc_into_ellipse(make_dxf):
    def build(doc, msp):
        block = doc.blocks.new("B")
        block.add_arc((0, 0), 1, 0, 90)
        msp.add_blockref("B", (0, 0), dxfattribs={"xscale": 900, "yscale": 200})

    prims, _ = extract_primitives(ezdxf.readfile(make_dxf(build)), 1.0)
    ellipse = next(p for p in prims if p.kind == "ellipse").geometry
    assert ellipse["semi_major"] == pytest.approx(900)
    assert ellipse["semi_minor"] == pytest.approx(200)
    assert max(p[0] for p in ellipse["points"]) == pytest.approx(900, abs=1)


def test_bulged_polyline_keeps_true_arc(make_dxf):
    def build(doc, msp):
        msp.add_lwpolyline([(0, 0, 0, 0, 0), (1000, 0, 0, 0, 1), (1000, 1000)], format="xyseb")

    prims, _ = extract_primitives(ezdxf.readfile(make_dxf(build)), 1.0)
    assert sorted(p.kind for p in prims) == ["arc", "line"]


def test_spline_is_flattened(make_dxf):
    def build(doc, msp):
        msp.add_spline([(0, 0), (1000, 800), (2000, 0), (3000, 800)])

    prims, _ = extract_primitives(ezdxf.readfile(make_dxf(build)), 1.0)
    assert prims[0].kind == "polyline"
    assert len(prims[0].geometry["points"]) > 8


def test_unsupported_entities_are_counted(make_dxf):
    prims, stats = extract_primitives(
        ezdxf.readfile(make_dxf(lambda d, m: m.add_point((0, 0)))), 1.0
    )
    assert not prims
    assert stats.unsupported["POINT"] == 1


def test_hidden_layers_are_skipped(make_dxf):
    def build(doc, msp):
        doc.layers.add("OFF").off()
        doc.layers.add("FROZEN").freeze()
        msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "OFF"})
        msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "FROZEN"})
        msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "ON"})

    prims, stats = extract_primitives(ezdxf.readfile(make_dxf(build)), 1.0)
    assert [p.layer for p in prims] == ["ON"]
    assert stats.skipped_hidden == 2


def test_linetype_resolved_from_layer(make_dxf):
    def build(doc, msp):
        doc.layers.add("GRID", linetype="CENTER")
        msp.add_line((0, 0), (1, 0), dxfattribs={"layer": "GRID"})

    prims, _ = extract_primitives(ezdxf.readfile(make_dxf(build)), 1.0)
    assert prims[0].linetype == "CENTER"


def test_xref_stops_with_message(make_dxf):
    def build(doc, msp):
        doc.add_xref_def("walls.dwg", "WALLS")
        msp.add_blockref("WALLS", (0, 0))

    doc = ezdxf.readfile(make_dxf(build))
    with pytest.raises(XrefMissingError, match="WALLS"):
        extract_primitives(doc, 1.0)


# ---------------------------------------------------------------- 빠른 전개
def signature(prims):
    """순서를 뺀 내용 비교용. 부동소수 오차는 반올림으로 흡수한다."""
    def rounded(value):
        if isinstance(value, float):
            return round(value, 6) + 0.0
        if isinstance(value, (list, tuple)):
            return tuple(rounded(v) for v in value)
        return value

    return sorted(
        repr((p.kind, p.layer, p.block_path, p.linetype, p.color, p.source_handles,
              sorted((k, rounded(v)) for k, v in p.geometry.items())))
        for p in prims
    )


def assert_same_expansion(doc):
    fast, fast_stats = extract_primitives(doc, 1.0, fast=True)
    slow, slow_stats = extract_primitives(doc, 1.0, fast=False)
    assert signature(fast) == signature(slow)
    assert fast_stats.skipped_hidden == slow_stats.skipped_hidden


@pytest.mark.parametrize("name", NAMES)
def test_fast_expansion_matches_library_on_dataset(name):
    assert_same_expansion(ezdxf.readfile(DRAWINGS / f"{name}.dxf"))


def test_fast_expansion_matches_library_on_tricky_blocks(make_dxf):
    def build(doc, msp):
        doc.layers.add("HIDDEN").off()
        inner = doc.blocks.new("INNER", base_point=(100, 50))
        inner.add_line((0, 0), (500, 0))
        inner.add_line((0, 0), (0, 500), dxfattribs={"layer": "HIDDEN"})
        inner.add_lwpolyline([(0, 0), (300, 0), (300, 300)], close=True, dxfattribs={"layer": "P"})
        inner.add_lwpolyline([(0, 0, 0, 0, 1), (300, 0)], format="xyseb")
        inner.add_arc((0, 0), 300, 10, 80)
        inner.add_circle((50, 50), 25)
        inner.add_text("T", height=50).set_placement((10, 10))
        inner.add_mtext("M", dxfattribs={"char_height": 50, "insert": (20, 20)})
        inner.add_ellipse((0, 0), major_axis=(100, 0), ratio=0.5)
        hatch = inner.add_hatch()
        hatch.paths.add_polyline_path([(0, 0), (100, 0), (100, 100)], is_closed=True)
        inner.add_solid([(0, 0), (10, 0), (0, 10), (10, 10)])
        inner.add_attdef("TAG", (0, 0))
        outer = doc.blocks.new("OUTER", base_point=(-30, 70))
        outer.add_blockref("INNER", (1000, 0), dxfattribs={"rotation": 30, "xscale": 2, "yscale": 2})
        outer.add_blockref("INNER", (0, 1000), dxfattribs={"xscale": -1})
        outer.add_line((0, 0), (9, 9))
        msp.add_blockref("OUTER", (5000, 0), dxfattribs={"rotation": 45, "layer": "L1"})
        msp.add_blockref("OUTER", (0, 5000), dxfattribs={"yscale": -1, "xscale": 1, "layer": "L2"})
        msp.add_blockref("OUTER", (0, 9000), dxfattribs={"xscale": 3, "yscale": 1})  # 비균등
        msp.add_blockref(
            "INNER", (0, 0), dxfattribs={"column_count": 2, "row_count": 2, "column_spacing": 900, "row_spacing": 900}
        )

    assert_same_expansion(ezdxf.readfile(make_dxf(build)))


def test_cached_primitives_match_a_fresh_read(cfg, tmp_path):
    path = DRAWINGS / "B2_apt_partial_code.dxf"
    fresh = preprocess(path, cfg)
    first = preprocess(path, cfg, cache_dir=tmp_path)
    second = preprocess(path, cfg, cache_dir=tmp_path)
    assert second.source.doc is None
    assert second.source.sha256 == fresh.source.sha256
    assert signature(second.primitives) == signature(fresh.primitives) == signature(first.primitives)
    assert second.scale == fresh.scale


def test_broken_cache_is_rebuilt(cfg, tmp_path):
    path = DRAWINGS / "B2_apt_partial_code.dxf"
    preprocess(path, cfg, cache_dir=tmp_path)
    for file in tmp_path.glob("*.pkl"):
        file.write_bytes(b"broken")
    assert preprocess(path, cfg, cache_dir=tmp_path).primitives


# ---------------------------------------------------------------- 블록 잘라내기
def clipped_doc(build):
    doc = ezdxf.new("R2018", setup=True)
    build(doc, doc.modelspace())
    return doc


def base_block(doc):
    base = doc.blocks.new("BASE")
    base.add_line((0, 0), (1000, 0), dxfattribs={"layer": "W"})
    base.add_line((5000, 0), (6000, 0), dxfattribs={"layer": "W"})
    base.add_text("밖의 글자", height=10).set_placement((5000, 50))
    base.add_text("안의 글자", height=10).set_placement((200, 10))
    return base


def test_clip_keeps_only_the_visible_part():
    def build(doc, msp):
        base_block(doc)
        insert = msp.add_blockref("BASE", (0, 0))
        XClip(insert).set_block_clipping_path([(100, -50), (400, 50)])

    prims, stats = extract_primitives(clipped_doc(build), 1.0)
    lines = [p for p in prims if p.kind == "line"]
    assert len(lines) == 1
    assert sorted((lines[0].geometry["p1"], lines[0].geometry["p2"])) == [(100, 0), (400, 0)]
    assert [p.geometry["text"] for p in prims if p.kind == "text"] == ["안의 글자"]
    assert stats.clip_applied == 1 and stats.clip_removed > 0


def test_same_base_inserted_twice_gives_only_the_two_windows():
    """실제 도면 구조. 요소 수가 삽입 횟수의 배수로 늘던 원인이다."""
    def build(doc, msp):
        base_block(doc)
        for offset, window in ((0, [(100, -50), (400, 50)]), (20000, [(5100, -50), (5300, 50)])):
            XClip(msp.add_blockref("BASE", (offset, 0))).set_block_clipping_path(window)

    prims, _ = extract_primitives(clipped_doc(build), 1.0)
    lines = [p for p in prims if p.kind == "line"]
    assert len(lines) == 2
    assert max(max(p.geometry["p1"][0], p.geometry["p2"][0]) for p in lines) == pytest.approx(25300)


def test_clip_follows_rotation_and_position():
    def build(doc, msp):
        base_block(doc)
        insert = msp.add_blockref("BASE", (10000, 0), dxfattribs={"rotation": 90})
        XClip(insert).set_block_clipping_path([(100, -50), (400, 50)])

    prims, _ = extract_primitives(clipped_doc(build), 1.0)
    line = next(p for p in prims if p.kind == "line").geometry
    assert sorted((line["p1"][1], line["p2"][1])) == pytest.approx([100, 400])
    assert line["p1"][0] == pytest.approx(10000)


def test_clip_cuts_polylines_and_nested_blocks():
    def build(doc, msp):
        inner = doc.blocks.new("INNER")
        inner.add_lwpolyline([(0, 0), (1000, 0), (1000, 1000)])
        far = doc.blocks.new("FAR")
        far.add_line((0, 0), (10, 0))
        outer = doc.blocks.new("OUTER")
        outer.add_blockref("INNER", (0, 0))
        outer.add_blockref("FAR", (90000, 0))
        XClip(msp.add_blockref("OUTER", (0, 0))).set_block_clipping_path([(0, -10), (500, 10)])

    prims, _ = extract_primitives(clipped_doc(build), 1.0)
    assert not any(p.block_path[-1:] == ("FAR",) for p in prims if p.kind == "line")
    polylines = [p.geometry["points"] for p in prims if p.kind == "polyline"]
    assert polylines == [[(0, 0), (500, 0)]]


def test_clip_on_inner_block_applies():
    def build(doc, msp):
        base_block(doc)
        outer = doc.blocks.new("OUTER")
        XClip(outer.add_blockref("BASE", (0, 0))).set_block_clipping_path([(100, -50), (400, 50)])
        msp.add_blockref("OUTER", (0, 0))

    prims, stats = extract_primitives(clipped_doc(build), 1.0)
    assert stats.clip_applied == 1
    assert len([p for p in prims if p.kind == "line"]) == 1


def test_disabled_clip_keeps_everything():
    def build(doc, msp):
        base_block(doc)
        clip = XClip(msp.add_blockref("BASE", (0, 0)))
        clip.set_block_clipping_path([(100, -50), (400, 50)])
        clip.disable_clipping()

    prims, stats = extract_primitives(clipped_doc(build), 1.0)
    assert len([p for p in prims if p.kind == "line"]) == 2
    assert stats.clip_applied == 0


def test_inverted_clip_is_counted_and_not_applied():
    def build(doc, msp):
        base_block(doc)
        clip = XClip(msp.add_blockref("BASE", (0, 0)))
        clip.set_block_clipping_path([(100, -50), (400, 50)])
        clip.invert_clipping_path(ignore_acad_compatibility=True)

    prims, stats = extract_primitives(clipped_doc(build), 1.0)
    assert len([p for p in prims if p.kind == "line"]) == 2
    assert stats.clip_inverted == 1 and stats.clip_applied == 0


def test_skipping_outside_blocks_matches_full_expansion():
    """밖의 블록을 건너뛰는 최적화가 결과를 바꾸면 안 된다."""
    def build(doc, msp):
        base_block(doc)
        part = doc.blocks.new("PART")
        part.add_line((0, 0), (300, 0))
        part.add_circle((100, 100), 50)
        wrap = doc.blocks.new("WRAP")
        wrap.add_blockref("BASE", (0, 0))
        wrap.add_blockref("PART", (150, 0))
        wrap.add_blockref("PART", (7000, 0))
        wrap.add_blockref("PART", (450, -20))  # 걸침
        XClip(msp.add_blockref("WRAP", (0, 0))).set_block_clipping_path([(100, -50), (500, 200)])

    doc = clipped_doc(build)
    fast, _ = extract_primitives(doc, 1.0, fast=True)
    slow, _ = extract_primitives(doc, 1.0, fast=False)
    assert signature(fast) == signature(slow)


# ---------------------------------------------------------------- 인코딩
def test_cp949_text_and_mtext_codes(make_dxf):
    def build(doc, msp):
        doc.header["$DWGCODEPAGE"] = "ANSI_949"
        doc.encoding = "cp949"
        msp.add_text("벽체 거실", height=100)
        msp.add_mtext(r"{\W0.9;\L거실}", dxfattribs={"char_height": 100})

    path = make_dxf(build, version="R2000")
    assert "벽체".encode("cp949") in path.read_bytes()
    prims, _ = extract_primitives(ezdxf.readfile(path), 1.0)
    assert [p.geometry["text"] for p in prims] == ["벽체 거실", "거실"]


# ---------------------------------------------------------------- 정규화
def test_normalize_merges_overlap_and_small_gap(cfg):
    prims = [
        line((0, 0), (1000, 0), pid="a"),
        line((500, 0), (1500, 0), pid="b"),
        line((1503, 0), (2000, 0), pid="c"),
    ]
    out, stats = normalize(prims, cfg["normalize"])
    assert len(out) == 1
    assert out[0].geometry["p1"] == pytest.approx((0, 0))
    assert out[0].geometry["p2"] == pytest.approx((2000, 0))
    assert set(out[0].source_handles) == {"a", "b", "c"}
    assert stats.merged_away == 2


def test_normalize_keeps_real_opening_gap(cfg):
    out, _ = normalize([line((0, 0), (1000, 0)), line((1900, 0), (3000, 0))], cfg["normalize"])
    assert len(out) == 2


def test_normalize_does_not_merge_across_layers(cfg):
    out, _ = normalize([line((0, 0), (1000, 0), "W"), line((1000, 0), (2000, 0), "D")], cfg["normalize"])
    assert len(out) == 2


def test_normalize_keeps_parallel_wall_faces(cfg):
    out, _ = normalize([line((0, 0), (1000, 0)), line((0, 200), (1000, 200))], cfg["normalize"])
    assert len(out) == 2


def test_normalize_removes_tiny_lines(cfg):
    out, stats = normalize([line((0, 0), (0.5, 0))], cfg["normalize"])
    assert out == []
    assert stats.removed_short == 1


def test_normalize_handles_reversed_and_wrapped_direction(cfg):
    # 거의 0°와 거의 180° 선은 같은 방향이다.
    a = line((0, 0), (1000, 0.5), pid="a")
    b = line((2000, 1.0), (1000, 0.5), pid="b")
    out, _ = normalize([a, b], cfg["normalize"])
    assert len(out) == 1


def test_normalize_diagonal_wall(cfg):
    d = math.sqrt(0.5)
    out, _ = normalize(
        [line((0, 0), (1000 * d, 1000 * d)), line((1002 * d, 1002 * d), (2000 * d, 2000 * d))],
        cfg["normalize"],
    )
    assert len(out) == 1
    assert out[0].geometry["p2"] == pytest.approx((2000 * d, 2000 * d))


# ---------------------------------------------------------------- 영역
def closed_rect(msp, x0, y0, x1, y1):
    msp.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], close=True)


def test_outer_wall_rectangle_is_not_a_frame_without_title_box(make_dxf, cfg):
    def build(doc, msp):
        closed_rect(msp, 0, 0, 10000, 8000)
        closed_rect(msp, 200, 200, 9800, 7800)
        add_plan(msp, 1000, 1000, 5000)
        add_dimensions(msp, 3000, y=-1000)

    pre = preprocess(make_dxf(build), cfg)
    assert pre.frames == []
    assert not pre.excluded


def test_two_plans_require_region(make_dxf, cfg):
    def build(doc, msp):
        add_plan(msp, 0, 0)
        add_plan(msp, 50000, 0)
        add_dimensions(msp, 3000, y=-1000)

    path = make_dxf(build)
    with pytest.raises(RegionAmbiguousError) as info:
        preprocess(path, cfg)
    assert len(info.value.candidates) == 2

    pre = preprocess(path, cfg, region=(45000, -2000, 70000, 12000))
    assert pre.plan_bbox[0] == pytest.approx(50000)
    assert pre.excluded


def test_two_framed_plans_each_lose_their_title_box(make_dxf, cfg):
    def sheet(msp, x):
        closed_rect(msp, x, -5000, x + 20000, 15000)
        closed_rect(msp, x + 15000, -5000, x + 20000, -3000)
        msp.add_text("제목", height=200).set_placement((x + 15500, -4000))
        add_plan(msp, x + 2000, 0)

    def build(doc, msp):
        sheet(msp, 0)
        sheet(msp, 21000)  # 도곽끼리 1m 간격. 군집 간격보다 가깝다
        add_dimensions(msp, 3000, y=-1000)

    with pytest.raises(RegionAmbiguousError) as info:
        preprocess(make_dxf(build), cfg)
    assert len(info.value.candidates) == 2


def test_adjoining_sheets_without_title_box_are_split(make_dxf, cfg):
    def build(doc, msp):
        for x in (0, 40000, 80000):
            closed_rect(msp, x, -5000, x + 40000, 25000)
            add_plan(msp, x + 3000, 2000)
            msp.add_linear_dim(base=(x + 3000, -1000), p1=(x + 3000, 0), p2=(x + 9000, 0), override={"dimlfac": 1.0, "dimdec": 0}).render()

    with pytest.raises(RegionAmbiguousError) as info:
        preprocess(make_dxf(build), cfg)
    assert len(info.value.candidates) == 3


def test_same_size_rooms_are_not_sheets_when_content_is_outside(make_dxf, cfg):
    def build(doc, msp):
        for x in (0, 10000):
            closed_rect(msp, x, 0, x + 8000, 8000)
            msp.add_linear_dim(base=(x + 1000, 1000), p1=(x + 1000, 2000), p2=(x + 4000, 2000), override={"dimlfac": 1.0, "dimdec": 0}).render()
        add_plan(msp, 0, 12000)
        add_dimensions(msp, 3000, y=-1000)

    assert not preprocess(make_dxf(build), cfg).frames


def test_loose_annotation_cluster_is_not_a_second_plan(make_dxf, cfg):
    def build(doc, msp):
        add_plan(msp, 0, 0)
        msp.add_text("메모", height=200).set_placement((40000, 0))
        add_dimensions(msp, 3000, y=-1000)

    pre = preprocess(make_dxf(build), cfg)
    assert [p.geometry["text"] for p in pre.excluded if p.kind == "text"] == ["메모"]


def test_text_only_cluster_does_not_replace_small_plan(make_dxf, cfg):
    def build(doc, msp):
        add_plan(msp, 0, 0, lines=8)
        for i in range(30):
            msp.add_text("주기", height=200).set_placement((40000 + i * 300, 0))

    pre = preprocess(make_dxf(build), cfg)
    assert sum(p.kind == "line" for p in pre.primitives) == 8
    assert not any(p.kind == "text" for p in pre.primitives)


def test_cluster_inside_large_hall_is_part_of_the_plan(make_dxf, cfg):
    """벽에서 먼 가구 묶음이 두 번째 평면이 되거나 평면에서 빠지면 안 된다."""
    def build(doc, msp):
        size = 30000
        for i in range(6):
            d = i * 40
            msp.add_line((0, d), (size, d))
            msp.add_line((0, size - d), (size, size - d))
            msp.add_line((d, 0), (d, size))
            msp.add_line((size - d, 0), (size - d, size))
        add_plan(msp, 10000, 10000, size=8000)
        msp.add_circle((5000, 5000), 300)

    pre = preprocess(make_dxf(build), cfg)
    assert not pre.excluded
    assert sum(p.kind == "circle" for p in pre.primitives) == 1


def test_frame_detection_scales_to_many_rectangles(cfg):
    """실제 도면에는 닫힌 사각형이 수만 개다. 사각형마다 전체를 훑으면 멈춘 것처럼 보인다."""
    prims = [
        Primitive(
            f"P{i}", "polyline",
            {"points": [(x, 0), (x + 100, 0), (x + 100, 100), (x, 100)], "closed": True},
            "F", (str(i),),
        )
        for i, x in enumerate(range(0, 40000 * 200, 200))
    ]
    started = time.perf_counter()
    kept, excluded, frames = strip_frames(prims, cfg["regions"])
    assert time.perf_counter() - started < 5
    assert len(kept) == 40000 and not excluded and not frames


def test_element_without_vertices_does_not_stop_plan_selection(cfg):
    """실제 도면에서 나온 KeyError. 꼭짓점 없는 폴리선은 범위가 없어 평면 밖으로 본다."""
    lines = [
        Primitive(f"L{i}", "line", {"p1": (0, i * 200), "p2": (6000, i * 200)}, "W", (str(i),))
        for i in range(25)
    ]
    empty = Primitive("E", "polyline", {"points": [], "closed": False}, "W", ("e",))
    kept, outside = select_plan([*lines, empty], cfg["regions"], [])
    assert len(kept) == 25 and outside == [empty]
    kept, outside = select_plan([*lines, empty], cfg["regions"], [], region=(-1, -1, 7000, 6000))
    assert empty in outside


def test_clustering_scales_with_long_grid_lines(cfg):
    """긴 선이 많으면 후보 쌍이 폭증한다. 실제 도면에서 가장 느렸던 단계다."""
    prims = [
        Primitive(f"G{i}", "line", {"p1": (0, i * 100), "p2": (2_000_000, i * 100)}, "W", (str(i),))
        for i in range(300)
    ]
    prims += [
        Primitive(f"S{i}", "line", {"p1": (i * 50, 100), "p2": (i * 50 + 40, 100)}, "W", (f"s{i}",))
        for i in range(40000)
    ]
    started = time.perf_counter()
    kept, outside = select_plan(prims, cfg["regions"], [])
    assert time.perf_counter() - started < 20
    assert len(kept) == len(prims) and not outside


def sheet_with_strip(msp, in_block=False):
    """실제 도면에서 본 구조. 겹친 도곽선과 세로로 긴 표제란."""
    target = msp
    closed_rect(target, 0, 0, 40000, 30000)
    closed_rect(target, 500, 500, 39500, 29500)
    target.add_line((32000, 500), (32000, 29500))  # 표제란 경계
    for i in range(12):
        target.add_line((32000, 2000 * i + 1000), (39500, 2000 * i + 1000))
        target.add_text("표제", height=200).set_placement((33000, 2000 * i + 1200))
    add_plan(target, 8000, 8000, lines=60)
    add_dimensions(target, 3000, y=6000)


def test_nested_frame_lines_and_tall_title_strip_are_removed(make_dxf, cfg):
    pre = preprocess(make_dxf(lambda d, m: sheet_with_strip(m)), cfg)
    assert len(pre.frames) == 1
    assert not any(p.kind == "text" for p in pre.primitives)
    assert not any(p.kind == "polyline" for p in pre.primitives)
    assert sum(p.kind == "line" for p in pre.primitives) == 60
    assert pre.plan_bbox[2] < 32000


def test_frame_block_contents_are_removed_with_it(make_dxf, cfg):
    def build(doc, msp):
        block = doc.blocks.new("SHEET")
        closed_rect(block, 0, 0, 40000, 30000)
        closed_rect(block, 35000, 0, 40000, 2000)
        block.add_text("제목", height=200).set_placement((35500, 1000))
        block.add_line((36000, 25000), (39000, 25000))  # 블록에 든 범례선
        msp.add_blockref("SHEET", (0, 0))
        add_plan(msp, 8000, 8000, lines=60)
        add_dimensions(msp, 3000, y=6000)

    pre = preprocess(make_dxf(build), cfg)
    assert sum(p.kind == "line" for p in pre.primitives) == 60


def test_plan_inside_frame_block_is_not_erased(make_dxf, cfg):
    def build(doc, msp):
        block = doc.blocks.new("WHOLE")
        closed_rect(block, 0, 0, 40000, 30000)
        closed_rect(block, 35000, 0, 40000, 2000)
        block.add_text("제목", height=200).set_placement((35500, 1000))
        add_plan(block, 8000, 8000, lines=60)
        msp.add_blockref("WHOLE", (0, 0))
        add_dimensions(msp, 3000, y=6000)

    pre = preprocess(make_dxf(build), cfg)
    assert sum(p.kind == "line" for p in pre.primitives) == 60


def test_long_wall_in_the_middle_is_not_a_title_boundary(make_dxf, cfg):
    def build(doc, msp):
        closed_rect(msp, 0, 0, 40000, 30000)
        closed_rect(msp, 35000, 0, 40000, 2000)
        msp.add_text("제목", height=200).set_placement((35500, 1000))
        msp.add_line((20000, 500), (20000, 29500), dxfattribs={"layer": "A-WALL"})
        add_plan(msp, 21000, 8000, lines=60)
        add_dimensions(msp, 3000, y=6000)

    pre = preprocess(make_dxf(build), cfg)
    assert sum(p.kind == "line" for p in pre.primitives) == 61


def test_tiny_group_next_to_a_big_plan_is_not_a_second_plan(make_dxf, cfg):
    """실제 도면에서 요소 35개 묶음이 평면 후보가 되어 파일이 둘로 나뉘었다."""
    def build(doc, msp):
        add_plan(msp, 0, 0, lines=800)
        add_plan(msp, 50000, 0, lines=25)
        add_dimensions(msp, 3000, y=-1000)

    pre = preprocess(make_dxf(build), cfg)
    assert sum(p.kind == "line" for p in pre.primitives) == 800


def test_two_comparable_plans_still_require_region(make_dxf, cfg):
    def build(doc, msp):
        add_plan(msp, 0, 0, lines=800)
        add_plan(msp, 50000, 0, lines=500)
        add_dimensions(msp, 3000, y=-1000)

    with pytest.raises(RegionAmbiguousError):
        preprocess(make_dxf(build), cfg)


def test_far_coordinates_are_moved_near_origin(make_dxf, cfg):
    def build(doc, msp):
        add_plan(msp, 152_300, 48_700)


    pre = preprocess(make_dxf(build), cfg)
    ox, oy = pre.origin_offset
    assert (ox, oy) == (157_000.0, 54_000.0)  # 평면 중심을 1m 단위로 반올림
    x0, y0, x1, y1 = pre.plan_bbox
    assert max(abs(x0), abs(x1), abs(y0), abs(y1)) < 10_000
