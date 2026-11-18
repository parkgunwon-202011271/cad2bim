"""가구 선 내보내기. 도형을 선분으로 바꾸고 building.dae와 같은 좌표로 옮기는지 본다."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from cad2bim.export import furniture
from cad2bim.preprocess.primitive import Primitive


def prim(kind, layer="A-FURN", **geometry):
    return Primitive(id=f"p{kind}", kind=kind, geometry=geometry, layer=layer, source_handles=("1",))


def test_shapes_become_segments():
    assert len(furniture._segments(prim("line", p1=(0, 0), p2=(1, 0)))) == 1
    square = prim("polyline", points=[(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
    assert len(furniture._segments(square)) == 4  # 닫힌 꺾은선은 마지막 변까지
    quarter = furniture._segments(prim("arc", center=(0, 0), radius=900, start=(900, 0), end=(0, 900)))
    assert quarter[0][0] == pytest.approx((900, 0)) and quarter[-1][1] == pytest.approx((0, 900), abs=1e-6)
    circle = furniture._segments(prim("circle", center=(0, 0), radius=300))
    assert len(circle) == math.ceil(360 / furniture._ARC_STEP_DEG)
    assert furniture._segments(prim("text", position=(0, 0), text="x", height=1, rotation=0, centered=False)) == []


def test_outline_size_and_layer_pick_the_kind_of_furniture():
    from shapely.geometry import Point, box

    kinds = lambda polygon, fixture=False: [k for k, *_ in furniture._solids(polygon, fixture)]  # noqa: E731
    assert kinds(box(0, 0, 500, 500)) == ["chair"]
    assert kinds(box(0, 0, 400, 1800)) == ["shelf"]
    assert kinds(box(0, 0, 50, 1600)) == ["partition"]
    desk = furniture._solids(box(0, 0, 1800, 700), False)
    assert [k for k, *_ in desk] == ["table"] * 5  # 상판과 다리 넷
    assert desk[0][2:] == furniture._TABLE_TOP and all(z0 == 0.0 for _, _, z0, _ in desk[1:])
    assert len(furniture._solids(Point(0, 0).buffer(600, quad_segs=6), False)) == 2  # 둥근 탁자는 가운데 기둥 하나
    assert furniture._solids(box(0, 0, 400, 600), True)[0][3] == 400.0  # 변기
    assert kinds(box(0, 0, 600, 2400), True) == ["counter"]  # 세면대
    assert furniture._solids(box(0, 0, 50, 1500), True)[0][2] == 100.0  # 칸막이는 바닥에서 띄운다


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
