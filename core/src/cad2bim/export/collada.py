"""건물을 COLLADA(.dae)로 쓴다. 스케치업에서 가져오는 첫 경로다.

층 아래에 종류별 묶음을 두어 가져온 뒤에도 층과 종류로 고를 수 있게 한다. 이름은 층 코드와 요소 번호뿐이다.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Point, Polygon, box

from .floor import door_top, glazing_band, plan_floors, wall_height, wall_top
from .ifc import _opening_span
from .mesh import ELEVATION, Mesh, concat, extrude, rotation_z

_NS = "http://www.collada.org/2005/11/COLLADASchema"
_PANEL_MM = 50.0
_CIRCLE_SIDES = 24
_VOLUME_TOLERANCE = 0.005

# 이름, (R, G, B), 불투명도
MATERIALS = {
    "wall": ((0.80, 0.78, 0.73), 1.0),
    "curtain": ((0.50, 0.80, 0.90), 0.45),
    "column": ((0.25, 0.35, 0.75), 1.0),
    "slab": ((0.90, 0.88, 0.84), 1.0),
    "roof": ((0.62, 0.62, 0.64), 1.0),
    "stair": ((0.70, 0.55, 0.40), 1.0),
    "rail": ((0.35, 0.35, 0.38), 1.0),
    "door": ((0.15, 0.60, 0.33), 1.0),
    "window": ((0.35, 0.75, 0.90), 0.45),
    "ground": ((0.74, 0.71, 0.64), 1.0),
    "surroundings": ((0.82, 0.82, 0.80), 1.0),
    "deck": ((0.72, 0.57, 0.40), 1.0),
    "exterior_stair": ((0.78, 0.77, 0.73), 1.0),
    "planter": ((0.42, 0.56, 0.32), 1.0),
    "parking": ((0.96, 0.96, 0.93), 1.0),
    "site_rail": ((0.35, 0.35, 0.38), 1.0),
    "lawn": ((0.50, 0.68, 0.36), 1.0),
    "roof_planter": ((0.42, 0.56, 0.32), 1.0),
    "pergola": ((0.52, 0.36, 0.22), 1.0),
    "skylight": ((0.60, 0.85, 0.95), 0.4),
    "skylight_frame": ((0.55, 0.55, 0.57), 1.0),
    "equipment": ((0.62, 0.62, 0.65), 1.0),
    "bench": ((0.62, 0.46, 0.30), 1.0),
    "parapet": ((0.70, 0.69, 0.66), 1.0),
    "penthouse": ((0.80, 0.78, 0.73), 1.0),
    "road": ((0.32, 0.32, 0.34), 1.0),
    "tree_crown": ((0.30, 0.48, 0.24), 1.0),
    "tree_trunk": ((0.36, 0.26, 0.18), 1.0),
    "shrub": ((0.30, 0.48, 0.24), 1.0),
    "car": ((0.78, 0.79, 0.81), 1.0),
    # 짙은 남색. 검은 유리(car_glass)와 같은 색이면 옆 칸 차와 붙어 한 덩어리로 보인다.
    "car_dark": ((0.17, 0.25, 0.45), 1.0),
    "car_glass": ((0.08, 0.09, 0.11), 1.0),
}
_GROUPS = {
    "wall": "Walls", "curtain": "CurtainWalls", "column": "Columns", "slab": "Slabs", "roof": "Roofs",
    "stair": "Stairs", "rail": "Rails", "door": "Doors", "window": "Windows",
    "ground": "Ground", "surroundings": "Surroundings", "deck": "Decks", "exterior_stair": "ExteriorStairs", "planter": "Planters", "parking": "Parking",
    "site_rail": "SiteRails", "lawn": "Lawn", "roof_planter": "RoofPlanters", "pergola": "Pergola",
    "skylight": "Skylights", "skylight_frame": "SkylightFrames", "equipment": "Equipment", "bench": "Benches",
    "parapet": "Parapets", "penthouse": "Penthouses", "road": "Roads",
    "tree_crown": "TreeCrowns", "tree_trunk": "TreeTrunks", "shrub": "Shrubs", "car": "Cars", "car_dark": "DarkCars", "car_glass": "CarGlass",
}
SITE_KEY = "SITE"


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", text)


def _wall_mesh(
    length: float, thickness: float, height: float, holes: list[tuple[float, float, float, float]]
) -> tuple[Mesh | None, float]:
    """(길이 x, 높이 z) 입면에 구멍을 뚫어 뽑는다. 벽 중심이 원점, 바닥이 z=0이다."""
    shape: Any = box(-length / 2, 0.0, length / 2, height)
    for hole in holes:
        shape = shape.difference(box(*hole))
    return shape, shape.area * thickness


def _wall_parts(shape: Any) -> list[Polygon]:
    return [g for g in getattr(shape, "geoms", [shape]) if isinstance(g, Polygon) and g.area > 0]


def _panel(width: float, height: float, sill: float) -> Mesh:
    mesh = extrude(box(-width / 2, sill, width / 2, sill + height), _PANEL_MM)
    return mesh.transformed(ELEVATION, np.array([0.0, _PANEL_MM / 2, 0.0]))


def common_base(storeys: list[Any]) -> tuple[float, float]:
    """모든 층이 함께 쓰는 원점. 층별 파일을 합쳐도 위치가 맞아야 한다."""
    world = [
        (w["start"][0] + s.part.ir["origin_offset"][0] + s.shift[0], w["start"][1] + s.part.ir["origin_offset"][1] + s.shift[1])
        for s in storeys for w in s.part.ir["walls"]
    ]
    if not world:
        return 0.0, 0.0
    return (
        round(sum(p[0] for p in world) / len(world) / 1000.0) * 1000.0,
        round(sum(p[1] for p in world) / len(world) / 1000.0) * 1000.0,
    )


def _collect(storeys: list[Any], cfg: dict[str, Any]) -> tuple[list[tuple[str, dict[str, list[tuple[str, Mesh, float | None]]]]], Counter]:
    counts: Counter = Counter()
    plan_floors(storeys, cfg)
    base_x, base_y = common_base(storeys)

    result = []
    for s in storeys:
        ir = s.part.ir
        dx = ir["origin_offset"][0] + s.shift[0] - base_x
        dy = ir["origin_offset"][1] + s.shift[1] - base_y
        z = s.elevation
        groups: dict[str, list[tuple[str, Mesh, float | None]]] = {}
        by_wall: dict[str, list[dict[str, Any]]] = {}
        for opening in ir["openings"]:
            by_wall.setdefault(opening["host_wall"], []).append(opening)

        for wall in ir["walls"]:
            x0, y0 = wall["start"]
            x1, y1 = wall["end"]
            length = math.hypot(x1 - x0, y1 - y0)
            if length <= 0:
                continue
            angle = math.atan2(y1 - y0, x1 - x0)
            rotation = rotation_z(angle)
            center = np.array([(x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, z])
            holes = []
            for opening in by_wall.get(wall["id"], []):
                span = _opening_span(opening, length)
                if span is None:
                    counts["skipped_opening"] += 1
                    continue
                kind = opening["type"]
                sill = cfg["window_sill_mm"] if kind == "window" else 0.0
                top = min(
                    sill + cfg["window_height_mm"] if kind == "window" else door_top(span[1] - span[0], cfg),
                    wall_height(wall, s.height, cfg),
                )
                if top <= sill:
                    counts["skipped_opening"] += 1
                    continue
                x_lo, x_hi = span[0] - length / 2, span[1] - length / 2
                holes.append((x_lo, sill, x_hi, top))
                counts["openings"] += 1
                if kind == "unknown" or opening.get("void_only"):
                    continue  # 미확인 틈은 비워 둔다. 문이나 창을 지어내지 않는다. 겹벽 메움의 구멍도 판이 없다
                panel = extrude(box(x_lo, sill, x_hi, top), _PANEL_MM).transformed(ELEVATION, np.array([0.0, _PANEL_MM / 2, 0.0]))
                category = "door" if kind == "door" else "window"
                groups.setdefault(category, []).append((_safe(opening["id"]), panel.transformed(rotation, center), None))
                counts[f"{category}s"] += 1
            category = "curtain" if wall["kind"] == "curtain" else "wall"
            if category == "curtain":
                # 띠창: 벽 사이에 얇은 유리를 넣는다. 문이 있으면 유리에서 뺀다.
                low, high = glazing_band(s.height, cfg)
                glass = box(-length / 2, low, length / 2, high)
                for hole in holes:
                    glass = glass.difference(box(*hole))
                for number, pane in enumerate(_wall_parts(glass), start=1):
                    panel = extrude(pane, _PANEL_MM).transformed(ELEVATION, np.array([0.0, _PANEL_MM / 2, 0.0]))
                    groups.setdefault("window", []).append(
                        (f"{_safe(wall['id'])}_G{number}", panel.transformed(rotation, center), pane.area * _PANEL_MM)
                    )
                holes = [*holes, (-length / 2, low, length / 2, high)]
                category = "wall"
                counts["glazed_bands"] += 1
            shape, _ = _wall_mesh(length, wall["thickness"], wall_height(wall, s.height, cfg), holes)
            parts = _wall_parts(shape)
            for number, part in enumerate(parts):
                mesh = extrude(part, wall["thickness"]).transformed(ELEVATION, np.array([0.0, wall["thickness"] / 2, 0.0]))
                suffix = f"_{number + 1}" if len(parts) > 1 else ""
                groups.setdefault(category, []).append(
                    (f"{_safe(wall['id'])}{suffix}", mesh.transformed(rotation, center), part.area * wall["thickness"])
                )
            counts["walls"] += 1

        for column in ir["columns"]:
            if column["shape"] == "polygon":
                # 그린 모양 그대로다. 좌표가 이미 층 좌표라 옮기기만 한다.
                outline = Polygon(column["polygon"])
                mesh = extrude(outline, wall_top(s.height, cfg)).transformed(np.eye(3), np.array([dx, dy, z]))
                kind = "wall" if column.get("wall_piece") else "column"
                groups.setdefault(kind, []).append((_safe(column["id"]), mesh, outline.area * wall_top(s.height, cfg)))
                counts["wall_pieces" if column.get("wall_piece") else "columns"] += 1
                continue
            if column["shape"] == "circle":
                outline = Point(0.0, 0.0).buffer(column["size"][0] / 2, quad_segs=_CIRCLE_SIDES // 4)
            else:
                outline = box(-column["size"][0] / 2, -column["size"][1] / 2, column["size"][0] / 2, column["size"][1] / 2)
            mesh = extrude(outline, wall_top(s.height, cfg)).transformed(
                rotation_z(column.get("rotation", 0.0)), np.array([column["center"][0] + dx, column["center"][1] + dy, z])
            )
            groups.setdefault("column", []).append((_safe(column["id"]), mesh, outline.area * wall_top(s.height, cfg)))
            counts["columns"] += 1

        for rail in s.floor.rails:  # 바닥 위에 선 부분만(floor.plan_floors)
            (x0, y0), (x1, y1) = rail["start"], rail["end"]
            length = math.hypot(x1 - x0, y1 - y0)
            if length <= 0:
                continue
            depth, top = cfg["rail_thickness_mm"], cfg["rail_height_mm"]
            mesh = extrude(box(-length / 2, -depth / 2, length / 2, depth / 2), top).transformed(
                rotation_z(math.atan2(y1 - y0, x1 - x0)), np.array([(x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, z])
            )
            groups.setdefault("rail", []).append((_safe(rail["id"]), mesh, length * depth * top))
            counts["rails"] += 1
        thickness = cfg["slab_thickness_mm"]
        for number, outline in enumerate(s.floor.slabs, start=1):
            mesh = extrude(outline, thickness).transformed(np.eye(3), np.array([dx, dy, z - thickness]))
            groups.setdefault("slab", []).append((f"S{number}", mesh, outline.area * thickness))
            counts["slabs"] += 1
        for number, outline in enumerate(s.floor.roofs, start=1):
            mesh = extrude(outline, thickness).transformed(np.eye(3), np.array([dx, dy, z + s.height - thickness]))
            groups.setdefault("roof", []).append((f"R{number}", mesh, outline.area * thickness))
            counts["roofs"] += 1
        for stair_id, boxes, landings in s.floor.stairs:
            pieces = [
                extrude(box(-b.length / 2, -b.width / 2, b.length / 2, b.width / 2), b.z1 - b.z0).transformed(
                    rotation_z(b.angle), np.array([b.center[0] + dx, b.center[1] + dy, b.z0])
                )
                for b in boxes
            ]
            pieces += [
                extrude(polygon, z1 - z0).transformed(np.eye(3), np.array([dx, dy, z0]))
                for polygon, z0, z1 in landings
            ]
            if pieces:
                volume = sum(b.length * b.width * (b.z1 - b.z0) for b in boxes) + sum(p.area * (z1 - z0) for p, z0, z1 in landings)
                groups.setdefault("stair", []).append((_safe(stair_id), concat(pieces), volume))
                counts["stairs"] += 1
        result.append((s.key, groups))
    site = getattr(storeys[0], "site", None) if storeys else None
    if site is not None and site.pieces:
        # 층과 나눠 한 묶음으로 둔다. 건물만 보고 싶으면 이 묶음을 끈다.
        groups = {}
        move = np.array([-base_x, -base_y, 0.0])
        for piece in site.pieces:
            groups.setdefault(piece.category, []).append((piece.name, piece.mesh.transformed(np.eye(3), move), piece.volume))
        result.append((SITE_KEY, groups))
        counts["site_pieces"] = len(site.pieces)
    return result, counts


def _text(parent: ET.Element, tag: str, value: str, **attrs: str) -> ET.Element:
    node = ET.SubElement(parent, tag, attrs)
    node.text = value
    return node


def _numbers(values: np.ndarray) -> str:
    return " ".join(f"{v:.3f}".rstrip("0").rstrip(".") if v != 0 else "0" for v in values.ravel().tolist())


def _add_geometry(library: ET.Element, identifier: str, mesh: Mesh, material: str) -> None:
    geometry = ET.SubElement(library, "geometry", id=identifier)
    node = ET.SubElement(geometry, "mesh")
    a, b, c = (mesh.vertices[mesh.triangles[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    def source(name: str, values: np.ndarray) -> None:
        element = ET.SubElement(node, "source", id=f"{identifier}-{name}")
        _text(element, "float_array", _numbers(values), id=f"{identifier}-{name}-array", count=str(values.size))
        accessor = ET.SubElement(
            ET.SubElement(element, "technique_common"), "accessor",
            source=f"#{identifier}-{name}-array", count=str(len(values)), stride="3",
        )
        for axis in "XYZ":
            ET.SubElement(accessor, "param", name=axis, type="float")

    source("positions", mesh.vertices)
    source("normals", normals)
    vertices = ET.SubElement(node, "vertices", id=f"{identifier}-vertices")
    ET.SubElement(vertices, "input", semantic="POSITION", source=f"#{identifier}-positions")
    triangles = ET.SubElement(node, "triangles", material=material, count=str(len(mesh.triangles)))
    ET.SubElement(triangles, "input", semantic="VERTEX", source=f"#{identifier}-vertices", offset="0")
    ET.SubElement(triangles, "input", semantic="NORMAL", source=f"#{identifier}-normals", offset="1")
    faces = np.arange(len(mesh.triangles))
    indices = np.column_stack([
        mesh.triangles[:, 0], faces, mesh.triangles[:, 1], faces, mesh.triangles[:, 2], faces,
    ])
    _text(triangles, "p", " ".join(map(str, indices.ravel().tolist())))


def write_dae(storeys: list[Any], path: Path, cfg: dict[str, Any]) -> tuple[dict[str, int], dict[str, float]]:
    """(종류별 개수, 형상 이름별 기대 부피 mm3)를 돌려준다."""
    collected, counts = _collect(storeys, cfg)
    ET.register_namespace("", _NS)
    root = ET.Element("COLLADA", xmlns=_NS, version="1.4.1")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "contributor")
    _text(asset, "created", "2026-01-01T00:00:00Z")
    _text(asset, "modified", "2026-01-01T00:00:00Z")
    ET.SubElement(asset, "unit", name="millimeter", meter="0.001")
    _text(asset, "up_axis", "Z_UP")

    effects = ET.SubElement(root, "library_effects")
    materials = ET.SubElement(root, "library_materials")
    for name, (colour, alpha) in MATERIALS.items():
        effect = ET.SubElement(effects, "effect", id=f"{name}-fx")
        phong = ET.SubElement(ET.SubElement(ET.SubElement(effect, "profile_COMMON"), "technique", sid="common"), "lambert")
        _text(ET.SubElement(phong, "diffuse"), "color", f"{colour[0]} {colour[1]} {colour[2]} 1")
        _text(ET.SubElement(phong, "transparency"), "float", str(alpha))
        ET.SubElement(ET.SubElement(materials, "material", id=f"{name}-mat", name=name), "instance_effect", url=f"#{name}-fx")

    geometries = ET.SubElement(root, "library_geometries")
    scenes = ET.SubElement(root, "library_visual_scenes")
    scene = ET.SubElement(scenes, "visual_scene", id="Building", name="Building")
    expected: dict[str, float] = {}
    for key, groups in collected:
        storey = ET.SubElement(scene, "node", id=f"{key}", name=f"{key}")
        for category, items in groups.items():
            group_name = f"{key}_{_GROUPS[category]}"
            group = ET.SubElement(storey, "node", id=group_name, name=group_name)
            for name, mesh, volume in items:
                identifier = f"{key}_{name}"
                _add_geometry(geometries, identifier, mesh, f"{category}-sym")
                element = ET.SubElement(group, "node", id=f"{identifier}-node", name=identifier)
                instance = ET.SubElement(element, "instance_geometry", url=f"#{identifier}")
                common = ET.SubElement(ET.SubElement(instance, "bind_material"), "technique_common")
                ET.SubElement(common, "instance_material", symbol=f"{category}-sym", target=f"#{category}-mat")
                if volume is not None:
                    expected[identifier] = volume
    ET.SubElement(ET.SubElement(root, "scene"), "instance_visual_scene", url="#Building")
    # 스키마 순서: asset, effects, materials, geometries, visual_scenes, scene
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    for name in ("walls", "curtain_walls", "columns", "slabs", "roofs", "stairs", "rails", "doors", "windows", "openings", "skipped_opening"):
        counts.setdefault(name, 0)
    return dict(counts), expected


def read_meshes(path: Path) -> dict[str, Mesh]:
    """쓴 파일을 다시 읽는다. 검증은 메모리의 메시가 아니라 파일 내용으로 한다."""
    root = ET.parse(path).getroot()
    tag = lambda name: f"{{{_NS}}}{name}"  # noqa: E731
    meshes = {}
    for geometry in root.iter(tag("geometry")):
        mesh = geometry.find(tag("mesh"))
        positions = mesh.find(f"{tag('source')}/{tag('float_array')}")
        vertices = np.array(positions.text.split(), dtype=float).reshape(-1, 3)
        triangles = mesh.find(tag("triangles"))
        indices = np.array(triangles.find(tag("p")).text.split(), dtype=np.int64).reshape(-1, 6)[:, [0, 2, 4]]
        meshes[geometry.get("id")] = Mesh(vertices, indices)
    return meshes


def verify_dae(path: Path, expected: dict[str, float]) -> dict[str, int]:
    """닫힌 메시인지, 안쪽이 안쪽인지, 부피가 맞는지 개수로 돌려준다."""
    meshes = read_meshes(path)
    not_closed = sum(mesh.open_edges() > 0 for mesh in meshes.values())
    inside_out = sum(mesh.volume <= 0 for mesh in meshes.values())
    mismatch = sum(
        abs(meshes[name].volume - volume) > volume * _VOLUME_TOLERANCE
        for name, volume in expected.items() if name in meshes
    )
    return {
        "meshes": len(meshes), "not_closed": not_closed, "inside_out": inside_out,
        "volume_checked": len(expected), "volume_mismatch": mismatch,
        "missing": sum(name not in meshes for name in expected),
    }
