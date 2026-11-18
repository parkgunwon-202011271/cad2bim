"""미리보기 도구. 그림이 저장되고, 멀리 떨어진 요소가 있어도 건물이 작아지지 않아야 한다."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from cad2bim.export.collada import write_dae
from test_building import EXPORT, plan_ir, wall
from test_collada import storeys_of

pytest.importorskip("collada")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

spec = importlib.util.spec_from_file_location("preview", Path(__file__).parents[1] / "tools" / "preview.py")
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def share(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(int)
    return float((np.abs(a[60:] - 240).max(axis=2) > 12).mean())


def test_preview_is_saved_with_three_views(tmp_path):
    dae, png = tmp_path / "b.dae", tmp_path / "b.png"
    write_dae(storeys_of(("F1", plan_ir())), dae, EXPORT)
    assert preview.write_preview(dae, png) > 0
    assert Image.open(png).size == (preview.WIDTH * 3, preview.HEIGHT)
    assert share(png) > 0.03


def test_far_element_does_not_shrink_the_building(tmp_path):
    ir = plan_ir()
    ir["walls"].append(wall(50, (3000000.0, 0.0), (3001000.0, 0.0)))
    dae, png = tmp_path / "b.dae", tmp_path / "b.png"
    write_dae(storeys_of(("F1", ir)), dae, EXPORT)
    preview.write_preview(dae, png)
    assert share(png) > 0.03


def test_shadow_darkens_the_ground(tmp_path):
    wall_block = np.array([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 3), (1, 0, 3), (1, 1, 3), (0, 1, 3),
    ], dtype=float)
    faces = np.array([
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ])
    ground = np.array([(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)], dtype=float)
    meshes = [(wall_block, faces, np.array([.8, .8, .8])), (ground, np.array([(0, 1, 2), (0, 2, 3)]), np.array([.8, .8, .8]))]
    lit = np.asarray(preview.render(meshes, -55.0, 30.0, "", shadows=False)).astype(int)
    shaded = np.asarray(preview.render(meshes, -55.0, 30.0, "")).astype(int)
    assert (lit - shaded).max() > 30


def test_roof_can_be_left_out(tmp_path):
    dae = tmp_path / "b.dae"
    write_dae(storeys_of(("F1", plan_ir())), dae, EXPORT)
    assert len(preview.load_meshes(dae, roof=False)) < len(preview.load_meshes(dae))
