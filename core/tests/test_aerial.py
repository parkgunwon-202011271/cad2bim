"""조감도 시점. 건물별 값 없이 어떤 모델에서도 시점이 나오는지 본다."""
from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import LineString, Point, box

from cad2bim.config import load_defaults
from cad2bim.export.aerial import ASPECT, choose_view, sun_direction
from cad2bim.export.collada import _collect
from cad2bim.export.mesh import extrude
from cad2bim.export.site import SiteShape, SiteSource, plan_site
from test_building import plan_ir
from test_collada import storeys_of

CFG = load_defaults()["export"]


def source(shapes=(), outline=None):
    return SiteSource("F1", 0.0, outline if outline is not None else box(0, 0, 10000, 10000), 3000.0, [], list(shapes), [])


def block(width=20000.0, depth=10000.0, height=9000.0):
    return [("F1", {"wall": [("W1", extrude(box(0, 0, width, depth), height), None)]})]


def on_screen(view, points):
    forward = (view.target - view.eye) / np.linalg.norm(view.target - view.eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    moved = points - view.eye
    depth = moved @ forward
    x, y = (moved @ right) / depth, (moved @ up) / depth
    tan_v = math.tan(math.radians(view.fov_degrees) / 2)
    return np.abs(y).max() <= tan_v and np.abs(x).max() <= tan_v * ASPECT


def test_sun_is_high_at_noon_and_below_the_horizon_at_night():
    noon = sun_direction(37.5, 127.0, 9.0, 288, 12.5)
    assert noon[2] > 0.6 and noon[1] < 0 and abs(noon[0]) < 0.2
    assert sun_direction(37.5, 127.0, 9.0, 288, 22.0)[2] < 0


def test_morning_sun_is_in_the_east():
    assert sun_direction(37.5, 127.0, 9.0, 288, 9.0)[0] > 0


@pytest.mark.parametrize("size", [(20000.0, 10000.0, 9000.0), (160000.0, 90000.0, 21000.0), (8000.0, 6000.0, 3000.0)])
def test_whole_building_stays_on_screen_at_any_size(size):
    collected = block(*size)
    view = choose_view(collected, CFG)
    assert view is not None and view.sun[2] > 0
    assert on_screen(view, collected[0][1]["wall"][0][1].vertices)


def test_camera_faces_the_sunlit_side():
    view = choose_view(block(), CFG)
    assert (view.eye - view.target)[:2] @ view.sun[:2] > 0


def test_no_geometry_gives_no_view():
    assert choose_view([], CFG) is None


def test_plan_without_a_boundary_still_gets_ground_and_trees():
    site = plan_site([source()], None, CFG)
    assert site.counts["surroundings"] >= 1 and site.counts["tree_crown"] >= 1 and len(site.rim) == 4


def test_trees_stay_off_the_building():
    site = plan_site([source()], None, CFG)
    outline = box(0, 0, 10000, 10000)
    for piece in site.pieces:
        if piece.category == "tree_trunk":
            x, y = piece.mesh.vertices[:, :2].mean(axis=0)
            assert not outline.contains(Point(x, y))


def test_parking_slots_are_filled_evenly():
    boundary = box(-5000, -5000, 40000, 30000)
    slots = [SiteShape("PARKING", box(12000 + 2500 * k, 12000, 14500 + 2500 * k, 17000)) for k in range(5)]
    site = plan_site([source([SiteShape("BOUNDARY", boundary), *slots])], None, CFG)
    assert site.counts["car"] + site.counts["car_dark"] == 3


def test_no_road_is_drawn_unless_the_drawing_names_one():
    shapes = [SiteShape("BOUNDARY", box(-5000, -5000, 40000, 30000)), SiteShape("STEP", LineString([(0, 0), (1, 0)]))]
    assert plan_site([source(shapes)], None, CFG).counts["road"] == 0


def test_real_plan_gets_a_view():
    collected, _ = _collect(storeys_of(("F1", plan_ir())), load_defaults()["export"])
    assert choose_view(collected, CFG) is not None
