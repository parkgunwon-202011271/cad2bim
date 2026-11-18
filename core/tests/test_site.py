"""건물 밖 요소. 레벨 표기에서 높이를 얻고, 표기가 없어도 멈추지 않으며, 파일까지 이어지는지 본다."""
from __future__ import annotations

import pytest
from shapely.geometry import LineString, Point, box

from cad2bim.config import load_defaults
from cad2bim.export.collada import read_meshes, verify_dae, write_dae
from cad2bim.export.floor import plan_floors
from cad2bim.export.site import Site, SiteShape, SiteSource, plan_site
from test_building import EXPORT, plan_ir
from test_collada import storeys_of

CFG = load_defaults()["export"]
BUILDING = box(0, 0, 10000, 10000)
BOUNDARY = box(-1000, -1000, 30000, 12000)


def source(shapes=(), texts=(), key="F1", elevation=0.0, outline=BUILDING, top=3000.0, rooms=()):
    return SiteSource(key, elevation, outline, top, list(rooms), list(shapes), list(texts))


def closed(polygon, role):
    return SiteShape(role, polygon)


def stepped_yard():
    """낮은 마당과 높은 데크 사이에 단 세 개(선 네 개)가 있다."""
    low = box(10000, 0, 16000, 10000)
    high = box(17200, 0, 24000, 10000)
    lines = [SiteShape("STEP", LineString([(16300 + 300 * k, 0), (16300 + 300 * k, 10000)])) for k in range(3)]
    lines += [SiteShape("STEP", LineString([(16000, y), (17200, y)])) for y in (0, 10000)]
    return [closed(BOUNDARY, "BOUNDARY"), closed(low, "STEP"), closed(high, "STEP"), *lines], [
        (12000.0, 5000.0, "하부 마당 EL+100.00", "LEVEL"),
        (20000.0, 5000.0, "상부 데크 EL+102.00", "LEVEL"),
    ]


def z_top(mesh):
    return float(mesh.vertices[:, 2].max())


def tops(site, category):
    return sorted(round(z_top(p.mesh)) for p in site.pieces if p.category == category)


def test_steps_divide_the_height_between_two_levels():
    shapes, texts = stepped_yard()
    site = plan_site([source(shapes, texts)], 100000.0, CFG)
    assert 0.0 in [float(t) for t in tops(site, "deck")] and 2000.0 in [float(t) for t in tops(site, "deck")]
    assert tops(site, "exterior_stair") == pytest.approx([400, 800, 1200, 1600], abs=1)


def test_wide_steps_are_seats_not_stairs():
    low = box(10000, 0, 14000, 10000)
    high = box(15800, 0, 24000, 10000)
    lines = [SiteShape("STEP", LineString([(14900, 0), (14900, 10000)]))]
    lines += [SiteShape("STEP", LineString([(14000, y), (15800, y)])) for y in (0, 10000)]
    texts = [(12000.0, 5000.0, "EL+100.00", "LEVEL"), (20000.0, 5000.0, "EL+101.35", "LEVEL")]
    site = plan_site([source([closed(BOUNDARY, "BOUNDARY"), closed(low, "STEP"), closed(high, "STEP"), *lines], texts)], 100000.0, CFG)
    assert not tops(site, "exterior_stair")
    assert {450, 900} <= set(tops(site, "deck"))


def test_open_end_of_steps_meets_the_terrain():
    deck = box(10000, 0, 16000, 10000)
    lines = [SiteShape("STEP", LineString([(16000 + 300 * k, 2000), (16000 + 300 * k, 3500)])) for k in range(1, 4)]
    rails = [SiteShape("STEP", LineString([(16000, 2000), (17200, 2000)])), SiteShape("STEP", LineString([(16000, 3500), (17200, 3500)]))]
    boundary = box(-1000, -1000, 17200, 12000)
    shapes = [closed(boundary, "BOUNDARY"), closed(deck, "STEP"), *lines, *rails, SiteShape("SPOT", LineString([(17000, 2800), (17100, 2800)]))]
    texts = [(12000.0, 5000.0, "EL+100.00", "LEVEL"), (17050.0, 2800.0, "99.00", "SPOT")]
    site = plan_site([source(shapes, texts, outline=BUILDING)], 99000.0, CFG)
    assert site.counts["exterior_stair"] >= 1
    assert all(0 < t < 1000 for t in tops(site, "exterior_stair"))


def test_without_level_notes_the_ground_is_flat():
    shapes, _ = stepped_yard()
    site = plan_site([source(shapes)], None, CFG)
    assert site.counts["ground"] >= 1
    assert max(z_top(p.mesh) for p in site.pieces if p.category in ("ground", "deck")) == pytest.approx(0.0, abs=1)


def test_no_boundary_makes_no_ground():
    site = plan_site([source([closed(box(20000, 0, 21000, 1000), "PLANTER")])], None, CFG)
    assert not site.counts["ground"] and not site.counts["planter"]


def test_boundary_away_from_the_building_is_ignored():
    site = plan_site([source([closed(box(100000, 0, 130000, 30000), "BOUNDARY")])], None, CFG)
    assert not site.counts["ground"]


def test_terrain_inner_edges_are_soft_but_its_rim_is_not():
    shapes = [closed(BOUNDARY, "BOUNDARY"), *[SiteShape("SPOT", LineString([(x, y), (x + 200, y)])) for x, y in ((-1000, -1000), (30000, 12000))]]
    texts = [(x + 100, y, value, "SPOT") for x, y, value in ((-1000, -1000, "100.0"), (30000, 12000, "102.0"))]
    ground = next(p for p in plan_site([source(shapes, texts)], 100000.0, CFG).pieces if p.category == "ground").mesh
    assert ground.soft
    half = len(ground.vertices) // 2  # 아래 절반이 바닥면, 위 절반이 윗면이다
    assert all((a < half) == (b < half) for a, b in ground.soft)  # 옆면의 세로 모서리는 숨기지 않는다
    outside = {(a, b) for ring, _ in ground.polygons if len(ring) == 4 for a, b in zip(ring, ring[1:])}
    assert not {(min(a, b), max(a, b)) for a, b in outside} & ground.soft


def test_terrain_follows_spot_heights_and_is_closed():
    shapes = [closed(BOUNDARY, "BOUNDARY"), *[SiteShape("SPOT", LineString([(x, y), (x + 200, y)])) for x, y in ((-1000, -1000), (30000, -1000), (30000, 12000), (-1000, 12000))]]
    texts = [(x + 100, y, value, "SPOT") for x, y, value in ((-1000, -1000, "100.0"), (30000, -1000, "101.0"), (30000, 12000, "102.0"), (-1000, 12000, "101.0"))]
    site = plan_site([source(shapes, texts)], 100000.0, CFG)
    ground = next(p for p in site.pieces if p.category == "ground").mesh
    assert ground.open_edges() == 0 and ground.volume > 0
    assert z_top(ground) == pytest.approx(2000.0, abs=50)


def roofed():
    lower = source(key="F1", outline=box(0, 0, 20000, 20000), top=4000.0)
    upper_outline = box(0, 15000, 20000, 20000)
    shapes = [
        closed(box(2000, 5000, 12000, 11000), "SKYLIGHT"),
        closed(box(1000, 1000, 5000, 4000), "ROOF_PLANTER"),
        closed(box(6000, 1000, 9000, 4000), "ROOF_PLANTER"),
        closed(box(14000, 2000, 18000, 6000), "OPEN"),
        closed(box(14000, 8000, 15000, 9000), "OPEN"),
        closed(box(10000, 1000, 12000, 1500), "FURNITURE"),
    ]
    texts = [
        (3000.0, 2500.0, "잔디마당", ""), (16000.0, 4000.0, "퍼골라 쉼터", ""), (14500.0, 8500.0, "실외기", ""),
    ]
    upper = source(shapes, texts, "F2", 4000.0, upper_outline, 8000.0)
    return [lower, upper]


def test_roof_objects_stand_on_the_upper_floor_level():
    site = plan_site(roofed(), None, CFG)
    assert site.counts["lawn"] == 1 and site.counts["roof_planter"] == 1
    assert site.counts["pergola"] == 1 and site.counts["equipment"] == 1 and site.counts["bench"] == 1
    assert tops(site, "lawn") == [4000 + round(CFG["site_lawn_height_mm"])]
    assert tops(site, "bench") == [4000 + round(CFG["site_seat_height_mm"])]
    assert site.counts["skylight"] == 1 and len(site.roof_holes) == 1


def test_unlabelled_open_outline_stays_out():
    lower, upper = roofed()
    upper.texts = []
    site = plan_site([lower, upper], None, CFG)
    assert not site.counts["pergola"] and not site.counts["equipment"]


def test_lift_room_gets_a_penthouse_and_top_roof_a_parapet():
    room = box(1000, 1000, 3000, 3000)
    site = plan_site([source(texts=[(2000.0, 2000.0, "EV", "")], rooms=[room])], None, CFG)
    assert site.counts["parapet"] == 1 and site.counts["penthouse"] == 1
    assert tops(site, "penthouse") == [3000 + round(CFG["site_penthouse_height_mm"])]


def test_skylight_cuts_the_roof_and_site_is_written(tmp_path):
    storeys = storeys_of(("F1", plan_ir()))
    site = Site(roof_holes=[box(1000, 1000, 2000, 2000)])
    site.add("planter", extrude_box())
    storeys[0].site = site
    path = tmp_path / "b.dae"
    _, expected = write_dae(storeys, path, EXPORT)
    assert verify_dae(path, expected)["volume_mismatch"] == 0
    assert "SITE_X1" in read_meshes(path)
    plan_floors(storeys, EXPORT)
    assert any(len(p.interiors) == 1 for p in storeys[0].floor.roofs)


def extrude_box():
    from cad2bim.export.mesh import extrude

    return extrude(box(0, 0, 1000, 1000), 500.0)


def test_circle_rail_is_a_bollard():
    shapes, texts = stepped_yard()
    shapes.append(SiteShape("SITE_RAIL", Point(12000, 5000), 100.0))
    site = plan_site([source(shapes, texts)], 100000.0, CFG)
    assert tops(site, "site_rail") == [round(CFG["site_bollard_height_mm"])]

