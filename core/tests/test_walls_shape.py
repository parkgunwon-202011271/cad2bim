"""이름 없는 도면의 벽 후보 고르기. 표준 이름이 있으면 동작이 바뀌지 않아야 한다."""
from __future__ import annotations

from cad2bim.config import load_defaults
from cad2bim.layers import LayerInfo, LayerMap, drop_weak_walls
from cad2bim.preprocess.primitive import Primitive
from cad2bim.recognize.walls import collect_segments, infer_walls_by_shape

CFG = load_defaults()["walls"]


def line(name, a, b, layer):
    return Primitive(name, "line", {"p1": a, "p2": b}, layer, (name,))


def double_line(x0, y0, x1, y1, layer, tag, gap=200.0):
    if y0 == y1:
        return [line(tag + "a", (x0, y0), (x1, y1), layer), line(tag + "b", (x0, y0 + gap), (x1, y1 + gap), layer)]
    return [line(tag + "a", (x0, y0), (x1, y1), layer), line(tag + "b", (x0 + gap, y0), (x1 + gap, y1), layer)]


def walls(layer="L1"):
    return (
        double_line(0, 0, 12000, 0, layer, "s") + double_line(0, 8000, 12000, 8000, layer, "n")
        + double_line(0, 0, 0, 8000, layer, "w") + double_line(12000, 0, 12000, 8000, layer, "e")
    )


def furniture(layer="L2"):
    items = []
    for k in range(30):
        x = 500.0 + k * 300.0
        items += double_line(x, 1000, x + 200, 1000, layer, f"f{k}", 100.0)
    return items


def unknown(*names):
    return LayerMap({n: LayerInfo("UNKNOWN", "unresolved") for n in names})


def test_long_pairs_pick_the_wall_layer_and_skip_furniture():
    prims = walls() + furniture()
    assert infer_walls_by_shape(prims, unknown("L1", "L2"), CFG) == {"L1"}


def test_best_layer_is_kept_when_all_score_below_the_threshold():
    prims = walls() + [line(f"x{k}", (0.0, 20000.0 + k * 900.0), (60000.0, 20000.0 + k * 900.0), "L1") for k in range(12)]
    assert infer_walls_by_shape(prims, unknown("L1"), CFG) == {"L1"}


def test_named_walls_are_left_alone():
    layers = LayerMap({"A-WALL": LayerInfo("WALL", "dictionary"), "L2": LayerInfo("UNKNOWN", "unresolved")})
    assert infer_walls_by_shape(walls("A-WALL") + walls("L2"), layers, CFG) == set()


def test_only_chosen_layers_become_wall_candidates():
    prims = walls() + furniture()
    layers = unknown("L1", "L2")
    layers.shape_walls = {"L1"}
    assert {s.layer for s in collect_segments(prims, layers, [], CFG)} == {"L1"}


def test_tiny_wall_named_layer_stops_blocking_shape_detection():
    prims = walls("L1") + [line("b", (0.0, 0.0), (300.0, 0.0), "BLOCK")]
    layers = LayerMap({"L1": LayerInfo("UNKNOWN", "unresolved"), "BLOCK": LayerInfo("WALL", "dictionary")})
    assert drop_weak_walls(layers, prims, CFG["weak_wall_max_share"], CFG["weak_wall_min_length_mm"]) == {"BLOCK"}
    assert layers.role("BLOCK") == "UNKNOWN" and not layers.has_walls


def test_real_wall_layer_is_not_dropped():
    layers = LayerMap({"A-WALL": LayerInfo("WALL", "dictionary")})
    assert drop_weak_walls(layers, walls("A-WALL"), CFG["weak_wall_max_share"], CFG["weak_wall_min_length_mm"]) == set()
    assert layers.role("A-WALL") == "WALL"
