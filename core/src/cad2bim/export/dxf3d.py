"""건물을 3D DXF로 쓴다. 요소마다 블록 하나이고, 레이어는 층과 종류로 나눈다.

면은 3DFACE라 대부분의 CAD와 스케치업이 읽을 수 있다. 같은 평면을 쪼갠 삼각형 선은 숨긴다.
이름은 층 코드와 요소 번호뿐이다.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import ezdxf
import numpy as np

from .collada import _GROUPS, _VOLUME_TOLERANCE, _collect
from .mesh import Mesh

_ACI = {
    "wall": 8, "curtain": 4, "column": 5, "slab": 9, "roof": 251, "stair": 30, "rail": 250, "door": 3, "window": 4,
    "ground": 253, "surroundings": 254, "deck": 32, "exterior_stair": 252, "planter": 94, "parking": 7, "site_rail": 250, "lawn": 82,
    "roof_planter": 94, "pergola": 34, "skylight": 4, "skylight_frame": 251, "equipment": 252, "bench": 34,
    "parapet": 8, "penthouse": 8, "road": 250, "tree_crown": 94, "tree_trunk": 34, "shrub": 94, "car": 9, "car_dark": 250, "car_glass": 250,
}
_GLASS_TRANSPARENCY = 0.55
_COPLANAR = 0.9999
_READ_DECIMALS = 3


def _hidden_edge_flags(mesh: Mesh) -> list[int]:
    """삼각형마다 숨길 모서리 표시. 양쪽 면이 같은 평면인 모서리와 겹친 넷째 모서리를 숨긴다."""
    a, b, c = (mesh.vertices[mesh.triangles[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    owners: dict[tuple[int, int], list[int]] = {}
    triangles = mesh.triangles.tolist()
    for face, tri in enumerate(triangles):
        for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            owners.setdefault((min(u, v), max(u, v)), []).append(face)

    def flat(u: int, v: int) -> bool:
        faces = owners[(min(u, v), max(u, v))]
        return len(faces) == 2 and float(normals[faces[0]] @ normals[faces[1]]) >= _COPLANAR

    flags = []
    for tri in triangles:
        value = 4  # 3DFACE의 셋째와 넷째 꼭짓점이 같아 생기는 길이 0 모서리
        value |= 1 if flat(tri[0], tri[1]) else 0
        value |= 2 if flat(tri[1], tri[2]) else 0
        value |= 8 if flat(tri[2], tri[0]) else 0
        flags.append(value)
    return flags


def write_dxf(storeys: list[Any], path: Path, cfg: dict[str, Any], version: str = "R2010") -> tuple[dict[str, int], dict[str, float]]:
    """(종류별 개수, 블록 이름별 기대 부피 mm3)를 돌려준다."""
    collected, counts = _collect(storeys, cfg)
    doc = ezdxf.new(version, setup=False)
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    msp = doc.modelspace()
    expected: dict[str, float] = {}
    faces = Counter()
    for key, groups in collected:
        for category, items in groups.items():
            layer = f"{key}_{_GROUPS[category]}"
            entry = doc.layers.add(layer, color=_ACI[category])
            if category in ("curtain", "window"):
                entry.transparency = _GLASS_TRANSPARENCY
            for name, mesh, volume in items:
                block_name = f"{key}_{name}"
                block = doc.blocks.new(block_name)
                for tri, flags in zip(mesh.triangles.tolist(), _hidden_edge_flags(mesh)):
                    p0, p1, p2 = (tuple(float(v) for v in mesh.vertices[i]) for i in tri)
                    block.add_3dface([p0, p1, p2, p2], dxfattribs={"layer": layer, "invisible_edges": flags})
                    faces["faces"] += 1
                msp.add_blockref(block_name, (0.0, 0.0, 0.0), dxfattribs={"layer": layer})
                if volume is not None:
                    expected[block_name] = volume
    doc.saveas(path)
    for name in ("walls", "curtain_walls", "columns", "slabs", "roofs", "stairs", "rails", "doors", "windows", "openings", "skipped_opening"):
        counts.setdefault(name, 0)
    return {**counts, **faces}, expected


def read_block_meshes(path: Path) -> dict[str, Mesh]:
    """쓴 파일을 다시 읽어 블록별 메시로 만든다. 같은 좌표는 같은 꼭짓점으로 합친다."""
    doc = ezdxf.readfile(path)
    used = {insert.dxf.name for insert in doc.modelspace().query("INSERT")}
    meshes = {}
    for name in used:
        index: dict[tuple[float, float, float], int] = {}
        points: list[tuple[float, float, float]] = []
        triangles = []
        for face in doc.blocks.get(name).query("3DFACE"):
            tri = []
            for corner in (face.dxf.vtx0, face.dxf.vtx1, face.dxf.vtx2):
                key = tuple(round(float(v), _READ_DECIMALS) for v in corner)
                if key not in index:
                    index[key] = len(points)
                    points.append(key)
                tri.append(index[key])
            triangles.append(tri)
        meshes[name] = Mesh(np.array(points, dtype=float).reshape(-1, 3), np.array(triangles, dtype=np.int64).reshape(-1, 3))
    return meshes


def verify_dxf(path: Path, expected: dict[str, float]) -> dict[str, int]:
    """파일 구조 오류, 닫힌 메시, 안팎, 부피를 개수로 돌려준다."""
    doc = ezdxf.readfile(path)
    auditor = doc.audit()
    meshes = read_block_meshes(path)
    return {
        "audit_errors": len(auditor.errors),
        "meshes": len(meshes),
        "not_closed": sum(mesh.open_edges() > 0 for mesh in meshes.values()),
        "inside_out": sum(mesh.volume <= 0 for mesh in meshes.values()),
        "volume_checked": len(expected),
        "volume_mismatch": sum(
            abs(meshes[name].volume - volume) > volume * _VOLUME_TOLERANCE
            for name, volume in expected.items() if name in meshes
        ),
        "missing": sum(name not in meshes for name in expected),
    }
