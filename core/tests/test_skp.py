"""스케치업 파일. 공식 SDK로 다시 열어 그룹, 태그, 닫힌 형상, 부피를 본다. SDK가 없으면 건너뛴다."""
from __future__ import annotations

import pytest

from cad2bim.config import load_defaults
from cad2bim.export.skp import SU5_2005, find_sdk, read_groups, read_shadows, read_view, verify_skp, write_skp
from test_building import EXPORT, opening, plan_ir
from test_collada import storeys_of
from test_floor import _stair, _with, CONF

SDK = find_sdk(load_defaults()["export"])
pytestmark = pytest.mark.skipif(SDK is None, reason="SketchUp SDK 없음")


def write(tmp_path, *items):
    path = tmp_path / "b.skp"
    counts, expected = write_skp(storeys_of(*items), path, EXPORT, SDK)
    return path, counts, expected


def test_file_reopens_with_closed_groups_and_right_volumes(tmp_path):
    ir = plan_ir(openings=[opening(), opening("window", 7000.0, 1500.0, "W3", "O2"), opening("unknown", 2500.0, 800.0, "W3", "O3")])
    path, counts, expected = write(tmp_path, ("F1", ir), ("F2", plan_ir()))
    assert counts["fill_failed"] == 0 and counts["walls"] == 8
    check = verify_skp(path, expected, SDK)
    assert check["not_closed"] == 0 and check["volume_mismatch"] == 0 and check["missing"] == 0
    assert check["wrong_tag"] == 0 and check["no_faces"] == 0 and check["groups"] > 8


def test_each_element_is_its_own_group_on_a_floor_tag(tmp_path):
    path, _, _ = write(tmp_path, ("F1", plan_ir(openings=[opening()])), ("F2", plan_ir()))
    groups = read_groups(path, SDK)
    assert groups["F1_W1"]["tag"] == "F1_Walls"
    assert groups["F1_O1"]["tag"] == "F1_Doors"
    assert groups["F2_W1"]["tag"] == "F2_Walls"
    assert groups["F1_C1"]["tag"] == "F1_Columns"


def test_door_hole_stays_in_the_wall_volume(tmp_path):
    path, _, _ = write(tmp_path, ("F1", plan_ir(openings=[opening(width=1000.0)])))
    assert read_groups(path, SDK)["F1_W1"]["volume_mm3"] == pytest.approx((10000 * 2800 - 1000 * 2100) * 200, rel=1e-3)


def test_wall_faces_are_whole_not_triangles(tmp_path):
    path, _, _ = write(tmp_path, ("F1", plan_ir()))
    assert read_groups(path, SDK)["F1_W1"]["faces"] == 6


def test_void_stair_and_roof_are_written(tmp_path):
    ir = _with(stairs=[_stair()], voids=[{"id": "V1", "bounds": (5500, 1000, 8000, 3000), "confidence": CONF}])
    path, counts, expected = write(tmp_path, ("F1", ir), ("F2", _with(stairs=[_stair()])))
    assert counts["fill_failed"] == 0 and counts["stairs"] == 1 and counts["roofs"] >= 1
    assert verify_skp(path, expected, SDK)["volume_mismatch"] == 0
    assert read_groups(path, SDK)["F1_ST1"]["tag"] == "F1_Stairs"


def test_polygon_that_cannot_be_a_face_falls_back_to_triangles(tmp_path, monkeypatch):
    import cad2bim.export.skp as skp

    original = skp._fill
    calls = {"polygon": 0}

    def fails_on_first_wall(api, entities, mesh, triangles=False):
        if not triangles and calls["polygon"] == 0:
            calls["polygon"] = 1
            raise skp.SketchUpError("face")
        return original(api, entities, mesh, triangles)

    monkeypatch.setattr(skp, "_fill", fails_on_first_wall)
    path, counts, expected = write(tmp_path, ("F1", plan_ir()))
    assert counts["fill_retried"] == 1 and counts["fill_failed"] == 0
    assert verify_skp(path, expected, SDK)["volume_mismatch"] == 0


def test_old_format_is_written_and_marked_as_sketchup_5(tmp_path):
    from cad2bim.export.skp import SU5_2005

    new, old = tmp_path / "b.skp", tmp_path / "old.skp"
    _, expected = write_skp(storeys_of(("F1", plan_ir(openings=[opening()]))), new, EXPORT, SDK, (old, SU5_2005))
    assert old.is_file() and old.read_bytes() != new.read_bytes()
    check = verify_skp(old, expected, SDK)
    assert check["volume_mismatch"] == 0 and check["not_closed"] == 0 and check["missing"] == 0


def test_names_contain_no_room_text(tmp_path):
    path, _, _ = write(tmp_path, ("F1", plan_ir()))
    assert not any("SECRET" in name for name in read_groups(path, SDK))
    assert b"SECRET" not in path.read_bytes()


def test_saved_file_has_the_sun_above_the_horizon(tmp_path):
    """해가 지평선 아래면 스케치업 화면 전체가 밤처럼 어두워진다. 위치 값만 봐서는 알 수 없다."""
    path, legacy = tmp_path / "b.skp", tmp_path / "old.skp"
    write_skp(storeys_of(("F1", plan_ir())), path, load_defaults()["export"], SDK, (legacy, SU5_2005))
    for name in (path, legacy):
        shadows = read_shadows(name, SDK)
        assert shadows["DisplayShadows"] == 1.0
        assert shadows["SunZ"] > 0.3


def test_saved_file_opens_at_the_aerial_view(tmp_path):
    path, legacy = tmp_path / "b.skp", tmp_path / "old.skp"
    write_skp(storeys_of(("F1", plan_ir())), path, load_defaults()["export"], SDK, (legacy, SU5_2005))
    data = read_view(path, SDK)
    assert data["scenes"] == 1
    assert data["eye_mm"][2] > 3000.0

