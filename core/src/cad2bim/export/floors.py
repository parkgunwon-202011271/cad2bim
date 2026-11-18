"""수정용 파일. 파일 하나 안에서 층이 나뉘고, 바로 고칠 수 있게 한다.

DXF는 블록 없이 평면 선(LWPOLYLINE)에 높이(thickness)를 준다. 레이어는 층_종류이고 층은 실제 높이에 놓는다.
SKP는 층마다 그룹 하나이고 그 안은 면과 선이다. 확신이 낮은 요소는 층_REVIEW_종류로 나눈다.
빠진 요소가 더 나쁘다는 사용자 답변에 따라 빼지 않고 켜 둔 채로 넣는다.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import ezdxf
from shapely.geometry import Polygon

from .collada import _collect, _safe, common_base
from .dxf3d import _ACI
from .floor import door_top, glazing_band, wall_top
from .quality import HIDDEN, classify
from .skp import find_sdk, read_editable_skp, read_shadows, write_editable_skp

_LEAF_MM = 50.0
_AREA_TOLERANCE = 0.01


def review_names(ir: dict[str, Any]) -> set[str]:
    """확신이 낮은 요소. 계단과 보이드는 항상 검토 대상이다."""
    names = set()
    for key in ("walls", "columns", "openings", "stairs", "voids"):
        for element in ir.get(key, []):
            if element.get("confidence", {}).get("level") == "review":
                names.add(_safe(element["id"]))
    return names


def element_tags(ir: dict[str, Any]) -> dict[str, str]:
    """요소 이름별 태그 머리말. 건물 밖으로 가른 것이 먼저이고, 그다음이 검토 대상이다."""
    tags = {name: "REVIEW_" for name in review_names(ir)}
    tags.update(classify(ir))
    return tags


def _rectangle(cx: float, cy: float, angle: float, length: float, width: float) -> list[tuple[float, float]]:
    c, s = math.cos(angle), math.sin(angle)
    return [
        (cx + c * x - s * y, cy + s * x + c * y)
        for x, y in ((-length / 2, -width / 2), (length / 2, -width / 2), (length / 2, width / 2), (-length / 2, width / 2))
    ]


def _add_floor(doc: Any, storey: Any, cfg: dict[str, Any], base: tuple[float, float]) -> float:
    """한 층을 모델 공간에 넣고 벽 평면 면적을 돌려준다."""
    msp = doc.modelspace()
    ir = storey.part.ir
    dx = ir["origin_offset"][0] + storey.shift[0] - base[0]
    dy = ir["origin_offset"][1] + storey.shift[1] - base[1]
    height = storey.height
    top_of_wall = wall_top(height, cfg)
    floor_z = storey.elevation

    def layer(name: str, category: str, flag: str) -> str:
        full = f"{storey.key}_{flag}{name}"
        if full not in doc.layers:
            entry = doc.layers.add(full, color=_ACI[category])
            if flag in HIDDEN:
                entry.off()
        return full

    def outline(points: list[tuple[float, float]], name: str, category: str, flag: str, depth: float, z: float = 0.0) -> None:
        msp.add_lwpolyline(
            points, close=True,
            dxfattribs={"layer": layer(name, category, flag), "thickness": depth, "elevation": floor_z + z},
        )

    tags = element_tags(ir)
    wall_area = 0.0
    by_wall: dict[str, list[dict[str, Any]]] = {}
    for opening in ir["openings"]:
        by_wall.setdefault(opening["host_wall"], []).append(opening)
    for wall in ir["walls"]:
        (x0, y0), (x1, y1) = wall["start"], wall["end"]
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 0:
            continue
        angle = math.atan2(y1 - y0, x1 - x0)
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        glazed = wall["kind"] == "curtain"
        name, category = "Walls", "wall"
        cuts = []
        for opening in sorted(by_wall.get(wall["id"], []), key=lambda o: o["position"]):
            half = opening["width"] / 2
            lo, hi = max(opening["position"] - half, 0.0), min(opening["position"] + half, length)
            if hi <= lo:
                continue
            cuts.append((lo, hi))
            middle = (lo + hi) / 2
            cx, cy = x0 + ux * middle + dx, y0 + uy * middle + dy
            flag = tags.get(_safe(opening["id"]), "")
            if opening["type"] == "door":
                outline(_rectangle(cx, cy, angle, hi - lo, _LEAF_MM), "Doors", "door", flag, door_top(hi - lo, cfg))
            elif opening["type"] == "window":
                outline(
                    _rectangle(cx, cy, angle, hi - lo, _LEAF_MM), "Windows", "window", flag,
                    cfg["window_height_mm"], cfg["window_sill_mm"],
                )
        start = 0.0
        segments = []
        for lo, hi in cuts:
            if lo > start:
                segments.append((start, lo))
            start = max(start, hi)
        if start < length:
            segments.append((start, length))
        for lo, hi in segments:
            middle = (lo + hi) / 2
            corners = _rectangle(x0 + ux * middle + dx, y0 + uy * middle + dy, angle, hi - lo, wall["thickness"])
            flag = tags.get(_safe(wall["id"]), "")
            if glazed:
                # 띠창: 창대 벽, 유리, 인방 벽을 겹쳐 놓는다.
                low, high = glazing_band(height, cfg)
                outline(corners, name, category, flag, low)
                outline(corners, name, category, flag, top_of_wall - high, high)
                outline(
                    _rectangle(x0 + ux * middle + dx, y0 + uy * middle + dy, angle, hi - lo, _LEAF_MM),
                    "Windows", "window", flag, high - low, low,
                )
                wall_area += 2 * (hi - lo) * wall["thickness"]
            else:
                outline(corners, name, category, flag, top_of_wall)
                wall_area += (hi - lo) * wall["thickness"]

    for column in ir["columns"]:
        flag = tags.get(_safe(column["id"]), "")
        cx, cy = column["center"][0] + dx, column["center"][1] + dy
        if column["shape"] == "polygon":
            points = [(x + dx, y + dy) for x, y in column["polygon"]]
            if column.get("wall_piece"):
                outline(points, "Walls", "wall", flag, top_of_wall)
                wall_area += Polygon(column["polygon"]).area
            else:
                outline(points, "Columns", "column", flag, top_of_wall)
        elif column["shape"] == "circle":
            msp.add_circle(
                (cx, cy, floor_z), column["size"][0] / 2,
                dxfattribs={"layer": layer("Columns", "column", flag), "thickness": top_of_wall},
            )
        else:
            outline(_rectangle(cx, cy, column.get("rotation", 0.0), column["size"][0], column["size"][1]), "Columns", "column", flag, top_of_wall)

    for rail in storey.floor.rails:  # 바닥 위에 선 부분만(floor.plan_floors)
        (x0, y0), (x1, y1) = rail["start"], rail["end"]
        length = math.hypot(x1 - x0, y1 - y0)
        if length > 0:
            outline(
                _rectangle((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, math.atan2(y1 - y0, x1 - x0), length, cfg["rail_thickness_mm"]),
                "Rails", "rail", "", cfg["rail_height_mm"],
            )

    def rings(polygon: Polygon) -> list[list[tuple[float, float]]]:
        return [
            [(x + dx, y + dy) for x, y in list(ring.coords)[:-1]]
            for ring in [polygon.exterior, *polygon.interiors]
        ]

    thickness = cfg["slab_thickness_mm"]
    for polygon in storey.floor.slabs:
        for ring in rings(polygon):
            outline(ring, "Slabs", "slab", "", thickness, -thickness)
    for polygon in storey.floor.roofs:
        for ring in rings(polygon):
            outline(ring, "Roofs", "roof", "", thickness, height - thickness)
    for stair_id, boxes, landings in storey.floor.stairs:
        flag = tags.get(_safe(stair_id), "REVIEW_")
        for b in boxes:
            outline(
                _rectangle(b.center[0] + dx, b.center[1] + dy, b.angle, b.length, b.width),
                "Stairs", "stair", flag, b.z1 - b.z0, b.z0 - floor_z,
            )
        for polygon, z0, z1 in landings:
            for ring in rings(polygon):
                outline(ring, "Stairs", "stair", flag, z1 - z0, z0 - floor_z)
    return wall_area


def write_editable_dxf(storeys: list[Any], path: Path, cfg: dict[str, Any]) -> dict[str, float]:
    """파일 하나에 모든 층을 쓰고 층별 벽 평면 면적을 돌려준다."""
    doc = ezdxf.new("R2004", setup=False)
    doc.header["$INSUNITS"] = 4
    doc.header["$MEASUREMENT"] = 1
    base = common_base(storeys)
    areas = {storey.key: _add_floor(doc, storey, cfg, base) for storey in storeys}
    doc.saveas(path)
    return areas


def verify_editable_dxf(path: Path, wall_areas: dict[str, float]) -> dict[str, int]:
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    measured = {key: 0.0 for key in wall_areas}
    for entity in msp.query("LWPOLYLINE"):
        key = entity.dxf.layer.split("_")[0]
        if entity.dxf.layer.endswith("Walls") and key in measured:
            measured[key] += Polygon([p[:2] for p in entity.get_points()]).area
    custom = [b.name for b in doc.blocks if not b.name.lower().startswith(("*model", "*paper"))]
    return {
        "audit_errors": len(doc.audit().errors),
        "blocks": len(custom),
        "inserts": len(msp.query("INSERT")),
        "floor_layers": len({layer.dxf.name.split("_")[0] for layer in doc.layers if "_" in layer.dxf.name}),
        "wall_area_mismatch": sum(
            abs(measured[key] - area) > max(area * _AREA_TOLERANCE, 1.0) for key, area in wall_areas.items()
        ),
    }


def write_editable_files(storeys: list[Any], folder: Path, cfg: dict[str, Any]) -> dict[str, int]:
    """수정용 DXF와, SDK가 있으면 SKP를 쓴다. 숫자만 돌려준다."""
    collected, _ = _collect(storeys, cfg)  # plan_floors도 여기서 끝난다
    dxf_path = folder / "building_editable.dxf"
    areas = write_editable_dxf(storeys, dxf_path, cfg)
    check = verify_editable_dxf(dxf_path, areas)
    tags = {storey.key: element_tags(storey.part.ir) for storey in storeys}
    kinds = [flag for per_floor in tags.values() for flag in per_floor.values()]
    total = {
        "floors": len(storeys), "dxf_errors": check["audit_errors"], "dxf_blocks": check["blocks"] + check["inserts"],
        "dxf_floor_layers": check["floor_layers"], "dxf_area_mismatch": check["wall_area_mismatch"],
        "review_elements": kinds.count("REVIEW_"),
        "hidden_outside": kinds.count("OUTSIDE_"), "hidden_pattern": kinds.count("PATTERN_"),
        "hidden_fragment": kinds.count("FRAGMENT_"),
    }
    sdk = find_sdk(cfg)
    if sdk is None:
        return {**total, "skp": 0}
    skp_path = folder / "building_editable.skp"
    result = write_editable_skp(collected, tags, skp_path, folder / "building_editable_SU5_2005.skp", sdk, HIDDEN, cfg)
    data = read_editable_skp(skp_path, sdk)
    legacy_path = folder / "building_editable_SU5_2005.skp"
    return {
        "skp_sun_up": int(read_shadows(skp_path, sdk)["SunZ"] > 0),
        "skp_su5_sun_up": int(read_shadows(legacy_path, sdk)["SunZ"] > 0),
        **total, "skp": 1, "skp_floor_groups": len(data["floors"]),
        "skp_faces": sum(f["faces"] for f in data["floors"].values()),
        "skp_nested_groups": sum(f["nested"] for f in data["floors"].values()),
        "skp_root_faces": data["root_faces"], "skp_retried": result["retried"], "skp_failed": result["failed"],
    }
