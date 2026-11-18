"""정답 모델 비교. 자기 자신과 비교하면 만점이고, 벽을 빼거나 더하면 점수가 떨어져야 한다."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, box

from cad2bim.config import load_defaults
from cad2bim.evaluation.model_compare import load_truth_faces, our_walls, score, section, wall_mask
from cad2bim.export.mesh import extrude
from cad2bim.export.skp import find_sdk, write_skp
from test_building import EXPORT, opening, plan_ir, wall
from test_collada import storeys_of

SDK = find_sdk(load_defaults()["export"])
Z = 1200.0


def box_faces(x0, y0, x1, y1, z0=0.0, z1=3000.0):
    mesh = extrude(box(x0, y0, x1, y1), z1 - z0).transformed(np.eye(3), np.array([0.0, 0.0, z0]))
    return [[mesh.vertices[outer]] for outer, _ in mesh.polygons]


def test_section_cuts_a_box_at_the_height():
    segments = section(box_faces(0, 0, 4000, 200), Z)
    assert len(segments) == 4 and sum(s.length for s in segments) == pytest.approx(8400)
    assert section(box_faces(0, 0, 4000, 200), 5000.0) == []


def test_thin_closed_piece_is_a_wall_and_a_big_room_is_not():
    wall_part = wall_mask(section(box_faces(0, 0, 4000, 200), Z))
    assert wall_part.area == pytest.approx(4000 * 200)
    assert wall_mask(section(box_faces(0, 0, 8000, 8000), Z)).is_empty


def test_identical_walls_score_full_and_shift_is_found():
    truth = box(5000, 7000, 15000, 7200).union(box(5000, 7000, 5200, 12000)).union(box(14800, 7000, 15000, 12000))
    ours = truth.buffer(0)
    from shapely.affinity import translate

    result = score(translate(ours, -5000, -7000), truth)
    assert result.precision == pytest.approx(1.0) and result.recall == pytest.approx(1.0)
    assert result.shift == pytest.approx((5000, 7000), abs=1)


def test_extra_wall_lowers_precision_and_missing_wall_lowers_recall():
    truth = box(0, 0, 10000, 200).union(box(0, 0, 200, 8000)).union(box(9800, 0, 10000, 8000))
    extra = truth.union(box(3000, 3000, 7000, 3200))
    assert score(extra, truth).precision < 0.95
    missing = box(0, 0, 10000, 200).union(box(0, 0, 200, 8000))
    assert score(missing, truth).recall < 0.8


def test_door_opens_the_section_but_a_window_stays():
    plain = our_walls(plan_ir(columns=[]), 1200.0, EXPORT).area
    door = our_walls(plan_ir(columns=[], openings=[opening("door", 5000.0, 1000.0, "W1", "D")]), 1200.0, EXPORT).area
    window = our_walls(plan_ir(columns=[], openings=[opening("window", 7000.0, 1000.0, "W3", "G")]), 1200.0, EXPORT).area
    assert plain - door == pytest.approx(1000 * 200, rel=0.01)
    assert window == pytest.approx(plain)
    assert our_walls(plan_ir(columns=[], openings=[opening("door", 5000.0, 1000.0, "W1", "D")]), 2500.0, EXPORT).area == pytest.approx(plain)


@pytest.mark.skipif(SDK is None, reason="SketchUp SDK 없음")
def test_own_skp_as_truth_scores_near_full(tmp_path):
    ir = plan_ir(columns=[])
    path = tmp_path / "t.skp"
    write_skp(storeys_of(("F1", ir)), path, EXPORT, SDK)
    faces = load_truth_faces(path, SDK)
    truth = wall_mask(section(faces, Z))
    result = score(our_walls(ir, Z, EXPORT), truth)
    assert result.recall > 0.9 and result.precision > 0.9
