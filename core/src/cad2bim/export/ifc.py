"""건물을 IFC로 쓴다. Revit 없이 3D 형상을 검증하기 위한 첫 출력이다 (상세설계 8장 L3).

층고, 문 높이, 창 높이는 평면에 없는 값이라 설정의 가정값을 쓴다.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.geom
import ifcopenshell.guid
import ifcopenshell.util.shape
import ifcopenshell.validate

from .floor import door_top, floor_outlines, glazing_band, plan_floors, wall_top  # noqa: F401  floor_outlines는 테스트와 다른 모듈이 쓴다

_VOLUME_TOLERANCE = 0.01
_VOID_MARGIN_MM = 100.0
_GLASS_MM = 50.0


def _polygon(points: list[Any]) -> Any:
    from shapely.geometry import Polygon

    return Polygon(points)


class _Writer:
    def __init__(self) -> None:
        self.model = ifcopenshell.api.run("project.create_file", version="IFC4")
        project = self._root("IfcProject", "Project")
        ifcopenshell.api.run("unit.assign_unit", self.model)  # 기본값은 mm
        parent = ifcopenshell.api.run("context.add_context", self.model, context_type="Model")
        self.body = ifcopenshell.api.run(
            "context.add_context", self.model, context_type="Model", context_identifier="Body",
            target_view="MODEL_VIEW", parent=parent,
        )
        self.site = self._root("IfcSite", "Site")
        self.building = self._root("IfcBuilding", "Building")
        ifcopenshell.api.run("aggregate.assign_object", self.model, products=[self.site], relating_object=project)
        ifcopenshell.api.run("aggregate.assign_object", self.model, products=[self.building], relating_object=self.site)
        self.site.ObjectPlacement = self.placement(0.0, 0.0)
        self.building.ObjectPlacement = self.placement(0.0, 0.0, 0.0, 0.0, self.site.ObjectPlacement)

    def _root(self, cls: str, name: str) -> Any:
        return ifcopenshell.api.run("root.create_entity", self.model, ifc_class=cls, name=name)

    def _axis(self, x: float, y: float, z: float = 0.0, angle: float = 0.0) -> Any:
        m = self.model
        return m.createIfcAxis2Placement3D(
            m.createIfcCartesianPoint((float(x), float(y), float(z))),
            m.createIfcDirection((0.0, 0.0, 1.0)),
            m.createIfcDirection((math.cos(angle), math.sin(angle), 0.0)),
        )

    def placement(self, x: float, y: float, z: float = 0.0, angle: float = 0.0, parent: Any = None) -> Any:
        return self.model.createIfcLocalPlacement(parent, self._axis(x, y, z, angle))

    def solid(self, profile: Any, depth: float) -> Any:
        m = self.model
        return m.createIfcExtrudedAreaSolid(profile, self._axis(0.0, 0.0), m.createIfcDirection((0.0, 0.0, 1.0)), float(depth))

    def box_profile(self, x_dim: float, y_dim: float) -> Any:
        m = self.model
        origin = m.createIfcAxis2Placement2D(m.createIfcCartesianPoint((0.0, 0.0)), m.createIfcDirection((1.0, 0.0)))
        return m.createIfcRectangleProfileDef("AREA", None, origin, float(x_dim), float(y_dim))

    def profile(self, polygon: Any, dx: float, dy: float) -> Any:
        """구멍이 있어도 되는 다각형 단면."""
        m = self.model

        def curve(ring: Any) -> Any:
            points = [(float(x + dx), float(y + dy)) for x, y in ring.coords]
            return m.createIfcPolyline([m.createIfcCartesianPoint(p) for p in points])

        if polygon.interiors:
            return m.createIfcArbitraryProfileDefWithVoids(
                "AREA", None, curve(polygon.exterior), [curve(r) for r in polygon.interiors]
            )
        return m.createIfcArbitraryClosedProfileDef("AREA", None, curve(polygon.exterior))

    def solid_at(self, profile: Any, depth: float, x: float, y: float, z: float, angle: float) -> Any:
        m = self.model
        return m.createIfcExtrudedAreaSolid(profile, self._axis(x, y, z, angle), m.createIfcDirection((0.0, 0.0, 1.0)), float(depth))

    def shape(self, product: Any, solid: Any) -> None:
        m = self.model
        rep = m.createIfcShapeRepresentation(self.body, "Body", "SweptSolid", solid if isinstance(solid, list) else [solid])
        product.Representation = m.createIfcProductDefinitionShape(None, None, [rep])

    def storey(self, name: str, elevation: float) -> Any:
        storey = self._root("IfcBuildingStorey", name)
        storey.Elevation = float(elevation)
        storey.ObjectPlacement = self.placement(0.0, 0.0, elevation, 0.0, self.building.ObjectPlacement)
        ifcopenshell.api.run("aggregate.assign_object", self.model, products=[storey], relating_object=self.building)
        return storey


def _opening_span(opening: dict[str, Any], length: float) -> tuple[float, float] | None:
    half = opening["width"] / 2
    start, end = opening["position"] - half, opening["position"] + half
    if start < -1.0 or end > length + 1.0 or end - start <= 0:
        return None
    return max(start, 0.0), min(end, length)


def write_ifc(storeys: list[Any], path: Path, cfg: dict[str, Any]) -> dict[str, int]:
    """층 목록을 IFC로 쓰고 종류별 개수를 돌려준다."""
    plan_floors(storeys, cfg)
    writer = _Writer()
    m = writer.model
    counts: Counter = Counter()
    # 원점에서 먼 좌표는 뷰어에서 흔들린다. 기준 층의 중심을 원점으로 옮긴다.
    world = [
        (w["start"][0] + s.part.ir["origin_offset"][0] + s.shift[0], w["start"][1] + s.part.ir["origin_offset"][1] + s.shift[1])
        for s in storeys for w in s.part.ir["walls"]
    ]
    base_x = round(sum(p[0] for p in world) / len(world) / 1000.0) * 1000.0 if world else 0.0
    base_y = round(sum(p[1] for p in world) / len(world) / 1000.0) * 1000.0 if world else 0.0

    for s in storeys:
        ir = s.part.ir
        dx = ir["origin_offset"][0] + s.shift[0] - base_x
        dy = ir["origin_offset"][1] + s.shift[1] - base_y
        storey = writer.storey(s.key, s.elevation)
        anchor = storey.ObjectPlacement
        contained: list[Any] = []
        hosts: dict[str, tuple[Any, Any, float, float]] = {}

        for wall in ir["walls"]:
            x0, y0 = wall["start"]
            x1, y1 = wall["end"]
            length = math.hypot(x1 - x0, y1 - y0)
            if length <= 0:
                counts["skipped_zero_wall"] += 1
                continue
            angle = math.atan2(y1 - y0, x1 - x0)
            entity = writer._root("IfcWall", wall["id"])
            entity.ObjectPlacement = writer.placement((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, 0.0, angle, anchor)
            writer.shape(entity, writer.solid(writer.box_profile(length, wall["thickness"]), wall_top(s.height, cfg)))
            contained.append(entity)
            hosts[wall["id"]] = (entity, entity.ObjectPlacement, length, wall["thickness"])
            counts["walls"] += 1
            if wall["kind"] == "curtain":
                # 창호 선만 있는 구간은 벽에 띠창을 낸다. 층고 전체를 유리로 만들지 않는다.
                low, high = glazing_band(s.height, cfg)
                void = writer._root("IfcOpeningElement", f"{wall['id']}-band")
                void.ObjectPlacement = writer.placement(0.0, 0.0, low, 0.0, entity.ObjectPlacement)
                writer.shape(void, writer.solid(writer.box_profile(length, wall["thickness"] + _VOID_MARGIN_MM), high - low))
                m.createIfcRelVoidsElement(ifcopenshell.guid.new(), None, None, None, entity, void)
                filler = writer._root("IfcWindow", f"{wall['id']}-glass")
                filler.OverallWidth, filler.OverallHeight = float(length), float(high - low)
                filler.ObjectPlacement = writer.placement(0.0, 0.0, low, 0.0, entity.ObjectPlacement)
                writer.shape(filler, writer.solid(writer.box_profile(length, _GLASS_MM), high - low))
                m.createIfcRelFillsElement(ifcopenshell.guid.new(), None, None, None, void, filler)
                contained.append(filler)
                counts["glazed_bands"] += 1

        for column in ir["columns"]:
            entity = writer._root("IfcColumn", column["id"])
            if column["shape"] == "polygon":
                entity.ObjectPlacement = writer.placement(0.0, 0.0, 0.0, 0.0, anchor)
                writer.shape(entity, writer.solid(writer.profile(_polygon(column["polygon"]), dx, dy), wall_top(s.height, cfg)))
                contained.append(entity)
                counts["columns"] += 1
                continue
            entity.ObjectPlacement = writer.placement(
                column["center"][0] + dx, column["center"][1] + dy, 0.0, column.get("rotation", 0.0), anchor
            )
            if column["shape"] == "circle":
                origin = m.createIfcAxis2Placement2D(m.createIfcCartesianPoint((0.0, 0.0)), m.createIfcDirection((1.0, 0.0)))
                profile = m.createIfcCircleProfileDef("AREA", None, origin, column["size"][0] / 2)
            else:
                profile = writer.box_profile(column["size"][0], column["size"][1])
            writer.shape(entity, writer.solid(profile, wall_top(s.height, cfg)))
            contained.append(entity)
            counts["columns"] += 1

        for opening in ir["openings"]:
            host = hosts.get(opening["host_wall"])
            span = _opening_span(opening, host[2]) if host else None
            if span is None:
                counts["skipped_opening"] += 1
                continue
            wall_entity, wall_place, length, thickness = host
            kind = opening["type"]
            sill = cfg["window_sill_mm"] if kind == "window" else 0.0
            top = min(
                sill + cfg["window_height_mm"] if kind == "window" else door_top(span[1] - span[0], cfg),
                wall_top(s.height, cfg),
            )
            if top <= sill:
                counts["skipped_opening"] += 1
                continue
            width = span[1] - span[0]
            center = (span[0] + span[1]) / 2 - length / 2
            void = writer._root("IfcOpeningElement", opening["id"])
            void.ObjectPlacement = writer.placement(center, 0.0, sill, 0.0, wall_place)
            writer.shape(void, writer.solid(writer.box_profile(width, thickness + _VOID_MARGIN_MM), top - sill))
            m.createIfcRelVoidsElement(ifcopenshell.guid.new(), None, None, None, wall_entity, void)
            counts["openings"] += 1
            if kind == "unknown":
                continue  # 미확인 틈은 비워 둔다. 문이나 창을 지어내지 않는다
            filler = writer._root("IfcDoor" if kind == "door" else "IfcWindow", opening["id"])
            filler.OverallWidth, filler.OverallHeight = float(width), float(top - sill)
            filler.ObjectPlacement = writer.placement(center, 0.0, sill, 0.0, wall_place)
            writer.shape(filler, writer.solid(writer.box_profile(width, thickness), top - sill))
            m.createIfcRelFillsElement(ifcopenshell.guid.new(), None, None, None, void, filler)
            contained.append(filler)
            counts["doors" if kind == "door" else "windows"] += 1

        for rail in ir.get("rails", []):
            (x0, y0), (x1, y1) = rail["start"], rail["end"]
            length = math.hypot(x1 - x0, y1 - y0)
            if length <= 0:
                continue
            entity = writer._root("IfcRailing", rail["id"])
            entity.ObjectPlacement = writer.placement(
                (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, 0.0, math.atan2(y1 - y0, x1 - x0), anchor
            )
            writer.shape(entity, writer.solid(writer.box_profile(length, cfg["rail_thickness_mm"]), cfg["rail_height_mm"]))
            contained.append(entity)
            counts["rails"] += 1
        thickness = cfg["slab_thickness_mm"]
        for outline in s.floor.slabs:
            slab = writer._root("IfcSlab", f"S-{s.key}-{counts['slabs'] + 1}")
            slab.ObjectPlacement = writer.placement(0.0, 0.0, -thickness, 0.0, anchor)
            writer.shape(slab, writer.solid(writer.profile(outline, dx, dy), thickness))
            contained.append(slab)
            counts["slabs"] += 1
        for outline in s.floor.roofs:
            roof = writer._root("IfcSlab", f"R-{s.key}-{counts['roofs'] + 1}")
            roof.PredefinedType = "ROOF"
            roof.ObjectPlacement = writer.placement(0.0, 0.0, s.height - thickness, 0.0, anchor)
            writer.shape(roof, writer.solid(writer.profile(outline, dx, dy), thickness))
            contained.append(roof)
            counts["roofs"] += 1
        for stair_id, boxes, landings in s.floor.stairs:
            items = [
                writer.solid_at(
                    writer.box_profile(b.length, b.width), b.z1 - b.z0,
                    b.center[0] + dx, b.center[1] + dy, b.z0 - s.elevation, b.angle,
                )
                for b in boxes
            ]
            items += [
                writer.solid_at(writer.profile(polygon, dx, dy), z1 - z0, 0.0, 0.0, z0 - s.elevation, 0.0)
                for polygon, z0, z1 in landings
            ]
            if not items:
                continue
            stair = writer._root("IfcStair", stair_id)
            stair.ObjectPlacement = writer.placement(0.0, 0.0, 0.0, 0.0, anchor)
            writer.shape(stair, items)
            contained.append(stair)
            counts["stairs"] += 1

        spaces = []
        for room in ir["rooms"]:
            points = [(float(x + dx), float(y + dy)) for x, y in room["polygon"]]
            if points[0] != points[-1]:
                points.append(points[0])
            if len(points) < 4:
                counts["skipped_room"] += 1
                continue
            space = writer._root("IfcSpace", room["id"])
            space.LongName = " / ".join(room["names"])
            space.ObjectPlacement = writer.placement(0.0, 0.0, 0.0, 0.0, anchor)
            curve = m.createIfcPolyline([m.createIfcCartesianPoint(p) for p in points])
            writer.shape(space, writer.solid(m.createIfcArbitraryClosedProfileDef("AREA", None, curve), s.height))
            spaces.append(space)
            counts["spaces"] += 1
        if spaces:
            ifcopenshell.api.run("aggregate.assign_object", writer.model, products=spaces, relating_object=storey)
        if contained:
            ifcopenshell.api.run("spatial.assign_container", writer.model, products=contained, relating_structure=storey)
    writer.model.write(str(path))
    for key in ("walls", "curtain_walls", "slabs", "roofs", "stairs", "rails", "columns", "doors", "windows", "openings", "spaces", "skipped_opening", "skipped_room", "skipped_zero_wall"):
        counts.setdefault(key, 0)
    return dict(counts)


def _expected_wall_volume(wall: Any) -> float | None:
    """벽 몸체 부피에서 겹치지 않게 합친 개구부 부피를 뺀다. 단위는 m3."""
    rep = wall.Representation.Representations[0].Items[0]
    profile = rep.SweptArea
    volume = profile.XDim * profile.YDim * rep.Depth
    openings = []
    for rel in wall.HasOpenings:
        void = rel.RelatedOpeningElement
        item = void.Representation.Representations[0].Items[0]
        x = void.ObjectPlacement.RelativePlacement.Location.Coordinates[0]
        z = void.ObjectPlacement.RelativePlacement.Location.Coordinates[2]
        openings.append((x - item.SweptArea.XDim / 2, x + item.SweptArea.XDim / 2, z, z + item.Depth))
    # 같은 벽의 개구부가 겹치면 단순 합이 맞지 않아 대조하지 않는다.
    openings.sort()
    for a, b in zip(openings, openings[1:]):
        if a[1] > b[0] and min(a[3], b[3]) > max(a[2], b[2]):
            return None
    for x0, x1, z0, z1 in openings:
        volume -= (x1 - x0) * profile.YDim * (z1 - z0)
    return volume / 1e9


def verify_ifc(path: Path) -> dict[str, int]:
    """다시 읽어 스키마 오류와 벽 부피를 확인한다. 개수만 돌려준다."""
    model = ifcopenshell.open(str(path))
    logger = ifcopenshell.validate.json_logger()
    level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        ifcopenshell.validate.validate(model, logger)
    finally:
        logging.disable(level)
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    checked = mismatch = failed = skipped = 0
    for wall in model.by_type("IfcWall"):
        expected = _expected_wall_volume(wall)
        if expected is None:
            skipped += 1
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, wall)
        except RuntimeError:
            failed += 1
            continue
        actual = ifcopenshell.util.shape.get_volume(shape.geometry)
        checked += 1
        mismatch += abs(actual - expected) > max(expected * _VOLUME_TOLERANCE, 1e-6)
    return {
        "schema_errors": len(logger.statements),
        "volume_checked": checked,
        "volume_mismatch": mismatch,
        "geometry_failed": failed,
        "overlap_skipped": skipped,
    }
