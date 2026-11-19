"""실 면적 두 기준(안목, 벽 중심선). 도면의 면적 표기는 대개 중심선 기준이다."""
from __future__ import annotations

import pytest

from cad2bim.export.room_areas import centerline_region, room_areas


def wall(wall_id, start, end, thickness=200.0):
    return {"id": wall_id, "start": list(start), "end": list(end), "thickness": thickness}


def box_ir(thickness=200.0):
    """벽 중심선이 6m × 4m인 방 하나. 실 다각형은 벽 안쪽이다."""
    half = thickness / 2
    return {
        "walls": [
            wall("W1", (0, 0), (6000, 0), thickness), wall("W2", (6000, 0), (6000, 4000), thickness),
            wall("W3", (6000, 4000), (0, 4000), thickness), wall("W4", (0, 4000), (0, 0), thickness),
        ],
        "columns": [],
        "rooms": [{
            "id": "R1", "names": ["방"], "clear_area_m2": (6000 - thickness) * (4000 - thickness) / 1e6,
            "polygon": [[half, half], [6000 - half, half], [6000 - half, 4000 - half], [half, 4000 - half]],
            "boundary_wall_ids": ["W1", "W2", "W3", "W4"],
        }],
    }


def test_centerline_area_is_the_area_inside_the_wall_axes():
    areas = room_areas(box_ir())["R1"]
    assert areas["clear"] == pytest.approx(5.8 * 3.8)
    assert areas["centerline"] == pytest.approx(6.0 * 4.0, abs=0.01)


def test_walls_of_different_thickness_each_add_their_own_half():
    ir = box_ir()
    ir["walls"][1]["thickness"] = 400.0  # 오른쪽 벽만 두껍다. 중심선은 그대로이고 실 안쪽 면이 100mm 들어온다
    ir["rooms"][0]["polygon"] = [[100, 100], [5800, 100], [5800, 3900], [100, 3900]]
    ir["rooms"][0]["clear_area_m2"] = 5.7 * 3.8
    assert room_areas(ir)["R1"]["centerline"] == pytest.approx(24.0, abs=0.02)


def test_a_column_that_bites_into_the_room_counts_in_the_centerline_area():
    ir = box_ir()
    ir["columns"] = [{"id": "C1", "shape": "rect", "center": [300.0, 300.0], "size": [400.0, 400.0], "rotation": 0.0, "polygon": None}]
    ir["rooms"][0]["polygon"] = [[500, 100], [5900, 100], [5900, 3900], [100, 3900], [100, 500], [500, 500]]
    ir["rooms"][0]["clear_area_m2"] = 5.8 * 3.8 - 0.4 * 0.4
    assert room_areas(ir)["R1"]["centerline"] == pytest.approx(24.0, abs=0.01)


def test_only_the_half_of_a_shared_wall_that_faces_the_room_is_added():
    ir = box_ir()
    ir["walls"].append(wall("W5", (6000, 0), (12000, 0)))  # 이웃 방으로 이어지는 벽. 이 방의 둘레가 아니다
    assert room_areas(ir)["R1"]["centerline"] == pytest.approx(24.0, abs=0.01)
    region = centerline_region(ir["rooms"][0], {w["id"]: w for w in ir["walls"]}, [])
    assert region.bounds == pytest.approx((0.0, 0.0, 6000.0, 4000.0), abs=6.0)  # 벽 중심선을 넘지 않는다


def test_a_room_without_boundary_walls_keeps_its_clear_area():
    ir = box_ir()
    ir["rooms"][0]["boundary_wall_ids"] = []
    areas = room_areas(ir)["R1"]
    assert areas["centerline"] == pytest.approx(areas["clear"])
