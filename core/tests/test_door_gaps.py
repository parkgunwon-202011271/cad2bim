"""문틀만 그린 문 자리, 짧은 창, ㄱ자 기둥.

입면이 '기둥 | 유리 | 문틀 | 양개 문 | 문틀 | 짧은 유리 | 기둥'으로 이어지는 평면이다. 문 양옆에 벽선이 없어
예전에는 문도 벽도 만들지 않았고, 800mm 유리는 최소 길이에 걸려 빠졌다. 둘 다 바닥부터 천장까지 뚫린 틈이 됐다.
"""
from __future__ import annotations

import ezdxf
import pytest
from test_recognize import convert_built, noise

from cad2bim.pipeline import convert

T = 300  # 벽 두께


def rect(msp, x0, y0, x1, y1, layer):
    msp.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], close=True, dxfattribs={"layer": layer})


def glazing(msp, x0, x1):
    """창호 선 네 줄(틀 양면과 유리 두 줄)."""
    for y in (-T / 2, -20, 20, T / 2):
        msp.add_line((x0, y), (x1, y), dxfattribs={"layer": "A-GLAZ"})


def facade(msp):
    rect(msp, 0, -200, 400, 200, "S-COLS")
    glazing(msp, 400, 2800)
    rect(msp, 2800, -T / 2, 3000, T / 2, "A-WALL")  # 문틀
    # 양개 문: 경첩은 문틀 면에, 문짝은 벽 방향으로 닫히고 아래로 열린다.
    msp.add_arc((3000, -T / 2), 900, 270, 0, dxfattribs={"layer": "A-DOOR"})
    msp.add_arc((4800, -T / 2), 900, 180, 270, dxfattribs={"layer": "A-DOOR"})
    msp.add_line((3000, -T / 2), (3000, -T / 2 - 900), dxfattribs={"layer": "A-DOOR"})
    msp.add_line((4800, -T / 2), (4800, -T / 2 - 900), dxfattribs={"layer": "A-DOOR"})
    rect(msp, 4800, -T / 2, 5000, T / 2, "A-WALL")  # 문틀
    glazing(msp, 5000, 5800)  # 800mm. 유리벽 최소 길이(1000mm)보다 짧다
    rect(msp, 5800, -200, 6200, 200, "S-COLS")
    # 뒤쪽 실. 평면으로 인정받을 만큼 벽을 둔다.
    for x0, y0, x1, y1 in ((0, 200, 300, 6000), (5900, 200, 6200, 6000), (0, 5700, 6200, 6000)):
        rect(msp, x0, y0, x1, y1, "A-WALL")
    noise(msp)


@pytest.fixture(scope="module")
def facade_ir(cfg, tmp_path_factory):
    """세 시험이 같은 평면을 쓰므로 한 번만 변환한다."""
    path = tmp_path_factory.mktemp("facade") / "f.dxf"
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 4
    facade(doc.modelspace())
    doc.saveas(path)
    return convert(path, path.parent / "o", cfg, 3000).ir


def _wall_at(ir, x, y=0.0):
    for w in ir.walls:
        (x0, y0), (x1, y1) = w.start, w.end
        if abs(y0 - y) <= 1 and abs(y1 - y) <= 1 and min(x0, x1) <= x <= max(x0, x1):
            yield w


def test_double_door_between_jambs_is_built(facade_ir):
    doors = [o for o in facade_ir.openings if o.type == "door"]
    assert len(doors) == 1
    door = doors[0]
    assert door.operation == "double"
    assert door.width == pytest.approx(1800, abs=1)
    host = next(w for w in facade_ir.walls if w.id == door.host_wall)
    assert "wall_from_door" in host.confidence.reasons  # 벽선 없는 자리를 메웠으니 검토 대상


def test_filled_wall_does_not_overlap_glazing(facade_ir):
    walls = list(_wall_at(facade_ir, 1500))
    assert [w.kind for w in walls] == ["curtain"]


def test_short_glazing_between_jamb_and_column_is_kept(facade_ir):
    walls = list(_wall_at(facade_ir, 5400))
    assert [w.kind for w in walls] == ["curtain"]


def test_door_arc_in_open_air_is_not_walled(cfg, make_dxf, tmp_path):
    """반대쪽 문짝 끝이 아무 데도 닿지 않으면 메우지 않는다."""
    def build(doc, msp):
        rect(msp, 0, -200, 400, 200, "S-COLS")
        glazing(msp, 400, 2800)
        rect(msp, 2800, -T / 2, 3000, T / 2, "A-WALL")
        msp.add_arc((3000, -T / 2), 900, 270, 0, dxfattribs={"layer": "A-DOOR"})
        for x0, y0, x1, y1 in ((0, 200, 300, 6000), (0, 5700, 6200, 6000)):
            rect(msp, x0, y0, x1, y1, "A-WALL")
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert not list(_wall_at(ir, 3450))


def test_glazing_between_walls_leaves_no_slit(cfg, make_dxf, tmp_path):
    """벽 | 유리 | 벽이 한 줄로 이어지면 사이에 틈이 없어야 한다. 예전에는 유리벽이 이웃 벽 외곽에 끝을 맞춰
    두께 절반(150mm)짜리 틈이 남아 밖이 보였다."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    from test_recognize import wall_lines

    # 유리 구간은 벽 잇기 한계(max_bridge_mm 4000)보다 길어야 창이 아닌 유리벽이 된다.
    def build(doc, msp):
        wall_lines(msp, (0, 0), (3000, 0), T)
        glazing(msp, 3000, 8000)
        wall_lines(msp, (8000, 0), (11000, 0), T)
        wall_lines(msp, (11000, 0), (11000, 5000), T)
        wall_lines(msp, (11000, 5000), (0, 5000), T)
        wall_lines(msp, (0, 5000), (0, 0), T)
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    line = [w for w in ir.walls if abs(w.start[1]) <= 1 and abs(w.end[1]) <= 1]
    assert any(w.kind == "curtain" for w in line)
    solid = unary_union([LineString([w.start, w.end]).buffer(w.thickness / 2, cap_style="flat") for w in line])
    assert solid.geom_type == "Polygon"  # 한 덩어리. 틈이 있으면 여러 조각이 된다
    assert solid.bounds[0] <= 150 and solid.bounds[2] >= 10850


def test_cubicle_door_hinged_at_the_outer_wall_is_not_a_door_in_that_wall(cfg, make_dxf, tmp_path):
    """화장실 칸막이 문. 경첩이 바깥벽 안쪽 면 바로 옆이고 문짝 선(열린 자리)이 바깥벽을 따라 그려져 있다.
    예전에는 그 선을 닫힌 자리로 보고 바깥벽에 문을 뚫었다."""
    from test_recognize import wall_lines

    def build(doc, msp):
        wall_lines(msp, (0, 0), (8000, 0), T)
        wall_lines(msp, (8000, 0), (8000, 5000), T)
        wall_lines(msp, (8000, 5000), (0, 5000), T)
        wall_lines(msp, (0, 5000), (0, 0), T)
        hinge = (4400, T / 2 + 150)  # 벽 면에서 150mm. 문틀 허용(200mm) 안이라 벽에 걸릴 수 있다
        msp.add_arc(hinge, 650, 0, 90, dxfattribs={"layer": "A-DOOR"})
        msp.add_line(hinge, (hinge[0] + 650, hinge[1]), dxfattribs={"layer": "A-DOOR"})  # 바깥벽과 나란한 문짝 선
        rect(msp, 4385, T / 2, 4415, hinge[1], "A-FLOR-FIXT")  # 칸막이 문틀
        rect(msp, 4385, hinge[1] + 650, 4415, hinge[1] + 850, "A-FLOR-FIXT")
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    assert not [o for o in ir.openings if o.type == "door"]


def test_rail_drawn_as_two_faces_is_one_rail(cfg):
    """보이드 난간을 안쪽·바깥쪽 면 두 닫힌 선(50mm 간격)으로 그렸다. 난간은 변마다 하나다."""
    from cad2bim.preprocess.primitive import Primitive
    from cad2bim.recognize import _pair_rails

    def ring(x0, y0, x1, y1, ident):
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        p = Primitive(ident, "polyline", {"points": pts[:-1], "closed": True}, "A-FLOR-RAIL", (ident,))
        return [(a, b, p) for a, b in zip(pts, pts[1:])]

    rails = _pair_rails(ring(2650, 10150, 17350, 15850, "O") + ring(2700, 10200, 17300, 15800, "I"), cfg["rails"])
    assert len(rails) == 4
    bottom = next(r for r in rails if abs(r[0][1] - r[1][1]) < 1 and r[0][1] < 12000)
    assert bottom[0][1] == pytest.approx(10175)  # 두 면의 가운데
    assert sorted((bottom[0][0], bottom[1][0])) == pytest.approx([2650, 17350])
    lone = _pair_rails(ring(0, 0, 4000, 3000, "A"), cfg["rails"])
    assert len(lone) == 4  # 짝이 없으면 그대로


def test_notched_column_on_column_layer_is_kept(cfg, make_dxf, tmp_path):
    """모서리를 딴 ㄱ자 기둥. 사각형만 받으면 빠지고, 벽이 절반 넘게 덮어 벽 조각 찾기에서도 빠졌다."""
    def build(doc, msp):
        for x0, y0, x1, y1 in ((0, 0, 6000, 200), (0, 3800, 6000, 4000), (0, 0, 200, 4000), (5800, 0, 6000, 4000),
                               (3050, 200, 3250, 3800)):  # 기둥으로 들어오는 칸막이벽
            rect(msp, x0, y0, x1, y1, "A-WALL")
        msp.add_lwpolyline(
            [(3400, 4100), (3400, 3600), (2900, 3600), (2900, 4000), (3000, 4000), (3000, 4100)],
            close=True, dxfattribs={"layer": "S-COLS"},
        )
        noise(msp)

    ir = convert_built(cfg, make_dxf, tmp_path, build).ir
    notched = [c for c in ir.columns if c.shape == "polygon" and not c.wall_piece]
    assert len(notched) == 1
    assert len(notched[0].polygon) == 6
