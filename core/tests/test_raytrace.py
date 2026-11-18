"""조감도 그림. 그림이 저장되고 건물이 화면에 보이며 빛과 그늘이 구분되는지 본다."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

pytest.importorskip("numba")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from cad2bim.config import load_defaults  # noqa: E402
from cad2bim.export.aerial import choose_view  # noqa: E402
from cad2bim.export.mesh import extrude  # noqa: E402
from cad2bim.export.raytrace import render_aerial  # noqa: E402

CFG = load_defaults()["export"]


def scene():
    ground = extrude(box(-30000, -30000, 50000, 40000), 200.0).transformed(np.eye(3), np.array([0.0, 0.0, -200.0]))
    return [
        ("F1", {"wall": [("W1", extrude(box(0, 0, 20000, 10000), 9000.0), None)]}),
        ("SITE", {"surroundings": [("X1", ground, None)]}),
    ]


def test_picture_is_saved_and_shows_the_building(tmp_path):
    path = tmp_path / "a.png"
    collected = scene()
    result = render_aerial(collected, [], choose_view(collected, CFG), path, CFG, 150, 1)
    assert Image.open(path).size == (150, 100)
    assert result["building_share"] > 0.05 and result["sun_up"] == 1


def test_light_and_shade_are_distinct(tmp_path):
    path = tmp_path / "a.png"
    collected = scene()
    render_aerial(collected, [], choose_view(collected, CFG), path, CFG, 300, 2)
    picture = np.asarray(Image.open(path).convert("L")).astype(float)
    assert picture.max() - picture.min() > 60
