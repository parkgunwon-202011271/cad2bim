"""실사화 재료와 점검. AI 자체는 그래픽 카드가 필요해 여기서 돌리지 않는다."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from cad2bim.config import load_defaults
from cad2bim.export.aerial import choose_view
from cad2bim.export.mesh import extrude
from cad2bim.export.photoreal import city_blocks, depth_map, edge_agreement, edge_lines, photo_view, worker_python

CFG = load_defaults()["export"]
RIM = [(0.0, 0.0, 0.0), (30000.0, 0.0, 0.0), (30000.0, 20000.0, 0.0), (0.0, 20000.0, 0.0)]


def block():
    return [("F1", {"wall": [("W1", extrude(box(0, 0, 20000, 10000), 9000.0), None)]})]


def test_city_never_touches_the_site():
    site = Polygon([p[:2] for p in RIM])
    meshes = city_blocks(RIM, CFG)
    assert meshes
    assert all(not site.contains(Polygon(m.vertices[:3, :2])) for m in meshes[:200])
    for mesh in meshes[:200]:
        x0, y0 = mesh.vertices[:, :2].min(axis=0)
        x1, y1 = mesh.vertices[:, :2].max(axis=0)
        assert not box(x0, y0, x1, y1).intersects(site)


def test_city_is_the_same_every_run():
    a, b = city_blocks(RIM, CFG), city_blocks(RIM, CFG)
    assert len(a) == len(b) and np.array_equal(a[0].vertices, b[0].vertices)


def test_no_rim_means_no_city():
    assert city_blocks([], CFG) == []


def test_photo_view_keeps_the_sun_and_side_of_the_aerial_view():
    collected = block()
    base = choose_view(collected, CFG)
    shot = photo_view(collected, base, CFG)
    assert np.array_equal(shot.sun, base.sun)
    assert (shot.eye - shot.target)[:2] @ (base.eye - base.target)[:2] > 0


def test_depth_map_is_brighter_when_closer():
    depth = np.array([[10.0, 100.0], [np.inf, 50.0]])
    picture = depth_map(depth).astype(int)
    assert picture[0, 0] > picture[1, 1] > picture[0, 1] > picture[1, 0]


def test_edges_follow_a_depth_step():
    depth = np.full((20, 20), 100.0)
    depth[:, 10:] = 50.0
    ids = np.zeros((20, 20), dtype=np.int32)
    lines = edge_lines(ids, depth, np.array([[0.0, 0.0, 1.0]]))
    assert lines[:, 9].all() and not lines[:, 2].any()


def test_agreement_drops_when_the_shape_is_gone():
    wanted = np.zeros((60, 60), dtype=np.uint8)
    wanted[:, 30] = 255
    good = np.zeros((60, 60, 3), dtype=np.uint8)
    good[:, 30:] = 200
    blurred = np.zeros((60, 60, 3), dtype=np.uint8)
    blurred[:, 5:] = 200  # 모서리가 다른 곳으로 옮겨 갔다
    assert edge_agreement(wanted, good) > edge_agreement(wanted, blurred)


def test_missing_worker_is_reported_not_raised():
    assert worker_python({"photo_python": "~/없는경로/python.exe"}) is None
