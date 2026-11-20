"""모델 공간을 Primitive 목록으로 푼다. 블록은 재귀 전개한다.

DIMENSION은 선으로 풀지 않는다. 치수 블록 안의 선이 벽 후보가 되면 안 되기 때문이다.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

import shapely
from ezdxf import bbox as ezbbox
from ezdxf import path as ezpath
from ezdxf.document import Drawing
from ezdxf.entities import DXFEntity
from ezdxf.lldxf.const import DXFError
from ezdxf.tools.text import plain_mtext as ezdxf_plain_mtext
from ezdxf.xclip import XClip
from shapely.geometry import LineString, Polygon

from ..errors import XrefMissingError
from .primitive import Primitive

_MAX_BLOCK_DEPTH = 16
_CLOSE_TOL = 1e-6
_CLIP_MARGIN = 1.0
_NUMBER = re.compile(r"\d{1,3}(,\d{3})+(\.\d+)?|\d+(\.\d+)?")


@dataclass
class ExplodeStats:
    skipped_hidden: int = 0
    unsupported: Counter = field(default_factory=Counter)
    transform_failures: list[str] = field(default_factory=list)
    # 실무 도면은 표제란과 "어디를 보여 줄지"를 모델 공간이 아니라 레이아웃에 둔다.
    paper: list[Primitive] = field(default_factory=list)
    viewports: list[tuple[float, float, float, float]] = field(default_factory=list)  # 모델 좌표 범위
    layouts: int = 0
    xclips: int = 0  # 일부만 보이도록 잘라낸 블록 삽입
    model_entities: int = 0
    model_types: Counter = field(default_factory=Counter)
    clip_applied: int = 0
    clip_inverted: int = 0  # 안쪽을 지우는 잘라내기. 라이브러리 표현이 불명확해 적용하지 않는다
    clip_failed: int = 0
    clip_removed: int = 0  # 잘라내기로 사라진 요소 수


@dataclass(frozen=True)
class _Parent:
    layer: str
    linetype: str
    color: int
    handle: str
    block_path: tuple[str, ...] = ()
    depth: int = 0
    instance: int = 0  # 가장 안쪽 블록 삽입의 번호. 0은 모델 공간이다
    clip_box: tuple[float, float, float, float] | None = None  # 이 밖의 요소는 만들 필요가 없다


def _xy(v: Any) -> tuple[float, float]:
    return (float(v[0]), float(v[1]))


def _polyline(points: list[tuple[float, float]], closed: bool) -> dict[str, Any]:
    """닫힘 표시 없이 끝점을 시작점에 맞춘 폴리선이 실무 도면에 흔하다."""
    if not closed and len(points) >= 4 and math.dist(points[0], points[-1]) <= _CLOSE_TOL:
        return {"points": points[:-1], "closed": True}
    return {"points": points, "closed": closed}


def parse_dimension_value(text: str) -> float | None:
    """'6000', '6,000', '6000.5'만 숫자로 본다. 그 외 표기는 근거로 쓰지 않는다."""
    s = text.strip()
    if not _NUMBER.fullmatch(s):
        return None
    return float(s.replace(",", ""))


def _is_clipped(e: DXFEntity) -> bool:
    try:
        return e.has_extension_dict and "ACAD_FILTER" in e.get_extension_dict()
    except (DXFError, AttributeError, TypeError):
        return False


def _boxes_touch(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] <= b[2] + _CLIP_MARGIN and a[2] >= b[0] - _CLIP_MARGIN and a[1] <= b[3] + _CLIP_MARGIN and a[3] >= b[1] - _CLIP_MARGIN


def _intersect_boxes(
    a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    if a is None:
        return tuple(b)
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


def _apply_clip(prims: list[Primitive], polygon: Polygon) -> list[Primitive]:
    """잘라내기 범위 안쪽만 남긴다. 걸친 선은 범위까지 자르고, 그 밖의 요소는 겹치면 남긴다."""
    if not prims:
        return prims
    shapely.prepare(polygon)
    kept: list[Primitive] = []
    for p in prims:
        box = p.extent()
        if box is None:
            kept.append(p)
            continue
        if p.kind == "line":
            pieces = _clip_path([p.geometry["p1"], p.geometry["p2"]], polygon)
            if pieces is None:
                kept.append(p)
            else:
                kept.extend(
                    replace(p, id=f"{p.id}c{i}", geometry={"p1": part[0], "p2": part[-1]})
                    for i, part in enumerate(pieces)
                )
        elif p.kind == "polyline":
            points = list(p.geometry["points"])
            if p.geometry["closed"] and points:
                points = [*points, points[0]]
            pieces = _clip_path(points, polygon)
            if pieces is None:
                kept.append(p)
            else:
                kept.extend(
                    replace(p, id=f"{p.id}c{i}", geometry={"points": part, "closed": False})
                    for i, part in enumerate(pieces)
                )
        elif polygon.intersects(shapely.box(*box)):
            kept.append(p)
    return kept


def _clip_path(points: list[tuple[float, float]], polygon: Polygon) -> list[list[tuple[float, float]]] | None:
    """완전히 안쪽이면 None, 아니면 안쪽 조각들."""
    if len(points) < 2:
        return None
    path = LineString(points)
    if polygon.contains(path):
        return None
    inside = path.intersection(polygon)
    pieces = []
    for part in shapely.get_parts(inside):
        if part.geom_type == "LineString" and part.length > 0:
            pieces.append([(float(x), float(y)) for x, y in part.coords])
    return pieces


def _viewport_range(e: DXFEntity) -> tuple[float, float, float, float] | None:
    """뷰포트가 보여 주는 모델 좌표 범위. 번호 1은 종이 자체라 뺀다."""
    width, height = e.dxf.get("width", 0.0), e.dxf.get("height", 0.0)
    view_height = e.dxf.get("view_height", 0.0)
    if e.dxf.get("id", 2) == 1 or width <= 0 or height <= 0 or view_height <= 0:
        return None
    cx, cy = _xy(e.dxf.get("view_center_point", (0.0, 0.0)))
    half_h = view_height / 2
    half_w = half_h * width / height
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


class _Extractor:
    def __init__(self, doc: Drawing, sagitta: float, fast: bool = True):
        self.doc = doc
        self.sagitta = sagitta
        self.fast = fast
        self.prims: list[Primitive] = []
        self.stats = ExplodeStats()
        self._next = 0
        self._next_instance = 0
        self._block_boxes: dict[str, tuple[float, float, float, float] | None] = {}
        self._hidden_layers = {
            layer.dxf.name.casefold()
            for layer in doc.layers
            if layer.is_off() or layer.is_frozen()
        }

    # -- 공통 ---------------------------------------------------------------
    def _resolve(self, e: DXFEntity, parent: _Parent | None) -> _Parent:
        """블록 상속 규칙을 적용한 이 엔티티의 레이어·선종류·색."""
        layer = e.dxf.get("layer", "0")
        if parent is not None and layer == "0":
            layer = parent.layer

        linetype = e.dxf.get("linetype", "BYLAYER")
        if linetype.upper() == "BYBLOCK" and parent is not None:
            linetype = parent.linetype
        elif linetype.upper() in ("BYLAYER", "BYBLOCK"):
            linetype = (
                self.doc.layers.get(layer).dxf.linetype
                if self.doc.layers.has_entry(layer)
                else "Continuous"
            )

        color = e.dxf.get("color", 256)
        if color == 0 and parent is not None:
            color = parent.color

        return _Parent(
            layer=layer,
            linetype=linetype,
            color=color,
            handle=parent.handle if parent is not None else str(e.dxf.handle),
            block_path=parent.block_path if parent is not None else (),
            depth=parent.depth if parent is not None else 0,
            instance=parent.instance if parent is not None else 0,
            clip_box=parent.clip_box if parent is not None else None,
        )

    def _add(self, kind: str, geometry: dict[str, Any], ctx: _Parent) -> None:
        self._next += 1
        self.prims.append(
            Primitive(
                id=f"P{self._next:06d}",
                kind=kind,
                geometry=geometry,
                layer=ctx.layer,
                source_handles=(ctx.handle,),
                block_path=ctx.block_path,
                linetype=ctx.linetype,
                color=ctx.color,
                extra={"instance": ctx.instance},
            )
        )

    # -- 엔티티 ---------------------------------------------------------------
    def _visible(self, e: DXFEntity, parent: _Parent | None) -> _Parent | None:
        ctx = self._resolve(e, parent)
        if ctx.layer.casefold() in self._hidden_layers or e.dxf.get("invisible", 0):
            self.stats.skipped_hidden += 1
            return None
        return ctx

    def visit(self, e: DXFEntity, parent: _Parent | None = None) -> None:
        ctx = self._visible(e, parent)
        if ctx is None:
            return

        handler = getattr(self, f"_on_{e.dxftype().lower()}", None)
        if handler is None:
            self.stats.unsupported[e.dxftype()] += 1
            return
        handler(e, ctx)

    def _on_line(self, e: DXFEntity, ctx: _Parent) -> None:
        self._add("line", {"p1": _xy(e.dxf.start), "p2": _xy(e.dxf.end)}, ctx)

    def _on_lwpolyline(self, e: DXFEntity, ctx: _Parent) -> None:
        if e.has_arc:
            # 호를 원형으로 유지하려고 선분과 호로 푼다.
            for sub in e.virtual_entities():
                self.visit(sub, ctx)
            return
        points = [_xy(p) for p in e.vertices_in_wcs()]
        self._add("polyline", _polyline(points, bool(e.closed)), ctx)

    def _on_polyline(self, e: DXFEntity, ctx: _Parent) -> None:
        if not (e.is_2d_polyline or e.is_3d_polyline):
            self.stats.unsupported["POLYLINE_MESH"] += 1
            return
        if e.has_arc:
            for sub in e.virtual_entities():
                self.visit(sub, ctx)
            return
        points = [_xy(p) for p in e.points_in_wcs()]
        self._add("polyline", _polyline(points, bool(e.is_closed)), ctx)

    def _on_solid(self, e: DXFEntity, ctx: _Parent) -> None:
        """채워진 사각형. 기둥을 이렇게 그리는 도면이 있다. 벽 선 후보로는 쓰지 않는다."""
        points: list[tuple[float, float]] = []
        for v in e.wcs_vertices():
            xy = _xy(v)
            if not points or points[-1] != xy:
                points.append(xy)
        if len(points) >= 3:
            self._add("solid", {"points": points}, ctx)

    def _on_trace(self, e: DXFEntity, ctx: _Parent) -> None:
        self._on_solid(e, ctx)

    def _on_arc(self, e: DXFEntity, ctx: _Parent) -> None:
        start, end = _xy(e.start_point), _xy(e.end_point)
        if e.dxf.extrusion.z < 0:
            # 뒤집힌 OCS에서는 WCS로 볼 때 시계 방향이다. 항상 반시계로 저장한다.
            start, end = end, start
        center = _xy(e.ocs().to_wcs(e.dxf.center))
        self._add(
            "arc", {"center": center, "radius": float(e.dxf.radius), "start": start, "end": end}, ctx
        )

    def _on_circle(self, e: DXFEntity, ctx: _Parent) -> None:
        center = _xy(e.ocs().to_wcs(e.dxf.center))
        self._add("circle", {"center": center, "radius": float(e.dxf.radius)}, ctx)

    def _on_ellipse(self, e: DXFEntity, ctx: _Parent) -> None:
        major = float(e.dxf.major_axis.magnitude)
        points = [_xy(p) for p in ezpath.make_path(e).flattening(self.sagitta)]
        self._add(
            "ellipse",
            {
                "center": _xy(e.dxf.center),
                "semi_major": major,
                "semi_minor": major * float(e.dxf.ratio),
                "points": points,
            },
            ctx,
        )

    def _on_spline(self, e: DXFEntity, ctx: _Parent) -> None:
        points = [_xy(p) for p in ezpath.make_path(e).flattening(self.sagitta)]
        self._add("polyline", {"points": points, "closed": bool(e.closed)}, ctx)

    def _on_text(self, e: DXFEntity, ctx: _Parent) -> None:
        align, placement, _ = e.get_placement()  # 정렬된 문자는 정렬점, 왼쪽 정렬은 삽입점
        self._add(
            "text",
            {
                "position": _xy(e.ocs().to_wcs(placement)),
                "text": e.plain_text().strip(),
                "height": float(e.dxf.height),
                "rotation": math.radians(e.dxf.get("rotation", 0.0)),
                "centered": align.name.endswith("CENTER"),
            },
            ctx,
        )

    def _on_attrib(self, e: DXFEntity, ctx: _Parent) -> None:
        self._on_text(e, ctx)

    def _on_mtext(self, e: DXFEntity, ctx: _Parent) -> None:
        self._add(
            "text",
            {
                "position": _xy(e.dxf.insert),
                "text": e.plain_text().strip(),
                "height": float(e.dxf.char_height),
                "rotation": math.radians(e.get_rotation()),
                "centered": e.dxf.get("attachment_point", 1) in (2, 5, 8),
            },
            ctx,
        )

    def _on_multileader(self, e: DXFEntity, ctx: _Parent) -> None:
        """지시선 글자. 화살표 끝마다 leader 요소 하나를 만든다. 글자가 가리키는 대상(가구 등)의 이름표로 쓴다.

        글자 자리가 아니라 화살표 끝을 위치로 둔다. kind가 text가 아니므로 실 이름·제목을 찾는 단계는 이 글자를 보지 않는다.
        """
        context = getattr(e, "context", None)
        mtext = getattr(context, "mtext", None)
        if mtext is None:
            return
        text = ezdxf_plain_mtext(mtext.default_content).strip()
        if not text:
            return
        for leader in context.leaders:
            for line in leader.lines:
                if not line.vertices:
                    continue
                self._add(
                    "leader",
                    {
                        "position": _xy(line.vertices[0]),
                        "text": text,
                        "height": float(context.char_height or 0.0),
                        "rotation": 0.0,
                        "centered": False,
                    },
                    ctx,
                )

    def _on_hatch(self, e: DXFEntity, ctx: _Parent) -> None:
        rings = []
        for p in ezpath.from_hatch(e):
            ring = [_xy(v) for v in p.flattening(self.sagitta)]
            if len(ring) >= 3:
                rings.append(ring)
        if not rings:
            self.stats.unsupported["HATCH_EMPTY"] += 1
            return
        self._add(
            "hatch",
            {"rings": rings, "pattern": e.dxf.pattern_name, "solid": bool(e.dxf.solid_fill)},
            ctx,
        )

    def _on_dimension(self, e: DXFEntity, ctx: _Parent) -> None:
        kind = e.dxf.dimtype & 7
        if kind not in (0, 1):  # 선형·정렬만 축척 근거로 쓴다
            return
        p1, p2 = e.dxf.defpoint2, e.dxf.defpoint3
        if kind == 0:
            angle = math.radians(e.dxf.get("angle", 0.0))
            measured = abs((p2.x - p1.x) * math.cos(angle) + (p2.y - p1.y) * math.sin(angle))
        else:
            measured = math.hypot(p2.x - p1.x, p2.y - p1.y)
        self._add(
            "dimension",
            {
                "p1": _xy(p1),
                "p2": _xy(p2),
                "position": _xy(e.dxf.defpoint),
                "measured": measured,
                "value": self._dimension_value(e),
            },
            ctx,
        )

    @staticmethod
    def _dimension_value(e: DXFEntity) -> float | None:
        # 헤더 $DIMLFAC는 치수 스타일이 덮어쓰므로 믿지 않고, 렌더링된 문자를 읽는다.
        block = e.get_geometry_block()
        if block is not None:
            for sub in block:
                if sub.dxftype() in ("TEXT", "MTEXT"):
                    value = parse_dimension_value(sub.plain_text())
                    if value is not None:
                        return value
        return parse_dimension_value(e.dxf.get("text", ""))

    def _clip_polygon(self, e: DXFEntity) -> Polygon | None:
        """이 삽입의 잘라내기 범위. 없거나 적용할 수 없으면 None이다."""
        if not _is_clipped(e):
            return None
        try:
            xclip = XClip(e)
            if not (xclip.has_clipping_path and xclip.is_clipping_enabled):
                return None
            if xclip.is_inverted_clip:
                self.stats.clip_inverted += 1
                return None
            vertices = [_xy(v) for v in xclip.get_wcs_clipping_path().vertices]
            polygon = Polygon(vertices).buffer(0) if len(vertices) >= 3 else None
        except (DXFError, AttributeError, TypeError, ValueError):
            polygon = None
        if polygon is None or polygon.is_empty:
            self.stats.clip_failed += 1
            return None
        self.stats.clip_applied += 1
        return polygon

    def _block_box(self, name: str) -> tuple[float, float, float, float] | None:
        """블록 좌표계의 범위. 한 번만 계산한다."""
        if name not in self._block_boxes:
            box = None
            try:
                found = ezbbox.extents(self.doc.blocks.get(name), fast=True)
                if found.has_data:
                    box = (found.extmin.x, found.extmin.y, found.extmax.x, found.extmax.y)
            except (DXFError, AttributeError, TypeError, ValueError):
                box = None
            self._block_boxes[name] = box
        return self._block_boxes[name]

    def _insert_is_outside(self, e: DXFEntity, box: tuple[float, float, float, float]) -> bool:
        """안쪽 블록 전체가 잘라내기 밖이면 펼치지 않는다. 큰 외부참조에서 시간을 크게 줄인다."""
        own = self._block_box(e.dxf.name)
        if own is None or e.mcount > 1:
            return False
        corners = [(own[0], own[1]), (own[2], own[1]), (own[2], own[3]), (own[0], own[3])]
        points = [_xy(v) for v in e.matrix44().transform_vertices(corners)]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        return not _boxes_touch((min(xs), min(ys), max(xs), max(ys)), box)

    def _expand_uniform(self, e: DXFEntity, block: Any, ref: _Parent, skipped: Any) -> None:
        """균등 축척 삽입. 선과 폴리선은 복사 없이 계산한다. 큰 블록에서 복사 비용이 가장 크다.

        비균등 축척은 호가 타원이 되는 등 변환이 달라 라이브러리 경로를 그대로 쓴다.
        """
        matrix = e.matrix44()
        for sub in block:
            kind = sub.dxftype()
            if kind == "ATTDEF":  # 라이브러리 전개도 속성 정의는 내보내지 않는다
                continue
            if kind == "LINE":
                ctx = self._visible(sub, ref)
                if ctx is not None:
                    p1, p2 = _xy(matrix.transform(sub.dxf.start)), _xy(matrix.transform(sub.dxf.end))
                    if ref.clip_box is None or _boxes_touch(
                        (min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])), ref.clip_box
                    ):
                        self._add("line", {"p1": p1, "p2": p2}, ctx)
            elif kind == "LWPOLYLINE" and not sub.has_arc:
                ctx = self._visible(sub, ref)
                if ctx is not None:
                    points = [_xy(matrix.transform(v)) for v in sub.vertices_in_wcs()]
                    self._add("polyline", _polyline(points, bool(sub.closed)), ctx)
            else:
                try:
                    clone = sub.copy()
                    clone.transform(matrix)
                except (DXFError, NotImplementedError) as exc:
                    skipped(sub, type(exc).__name__)
                    continue
                if kind == "INSERT" and ref.clip_box is not None and self._insert_is_outside(clone, ref.clip_box):
                    continue
                self.visit(clone, ref)

    def _on_insert(self, e: DXFEntity, ctx: _Parent) -> None:
        if e.mcount > 1:
            for single in e.multi_insert():
                self._expand(single, ctx)
            return
        self._expand(e, ctx)

    def _expand(self, e: DXFEntity, ctx: _Parent) -> None:
        name = e.dxf.name
        self.stats.xclips += _is_clipped(e)
        block = self.doc.blocks.get(name)
        if block is None:
            self.stats.unsupported["INSERT_NO_BLOCK"] += 1
            return
        if block.block_record.is_xref or block.block.is_xref:
            raise XrefMissingError(
                f"외부참조 '{name}'가 도면에 포함되어 있지 않습니다. "
                "벽이 통째로 빠질 수 있어 멈춥니다. 외부참조를 결합(bind)한 도면을 입력하세요."
            )
        if ctx.depth >= _MAX_BLOCK_DEPTH:
            self.stats.unsupported["INSERT_TOO_DEEP"] += 1
            return

        clip = self._clip_polygon(e)
        clip_box = ctx.clip_box
        if clip is not None:
            clip_box = _intersect_boxes(clip_box, clip.bounds)
        start = len(self.prims)
        self._next_instance += 1
        ref = _Parent(
            layer=ctx.layer,
            linetype=ctx.linetype,
            color=ctx.color,
            handle=ctx.handle,
            block_path=(*ctx.block_path, name),
            depth=ctx.depth + 1,
            instance=self._next_instance,
            clip_box=clip_box,
        )
        self._add(
            "block_ref",
            {
                "position": _xy(e.ocs().to_wcs(e.dxf.insert)),
                "name": name,
                "rotation": math.radians(e.dxf.get("rotation", 0.0)),
                "xscale": float(e.dxf.get("xscale", 1.0)),
                "yscale": float(e.dxf.get("yscale", 1.0)),
            },
            ref,
        )

        def skipped(entity: DXFEntity, reason: str) -> None:
            self.stats.transform_failures.append(f"{name}/{entity.dxftype()}: {reason}")

        if self.fast and e.has_uniform_scaling:
            self._expand_uniform(e, block, ref, skipped)
        else:
            for sub in e.virtual_entities(skipped_entity_callback=skipped):
                self.visit(sub, ref)
        for attrib in e.attribs:
            self.visit(attrib, ref)
        if clip is not None:
            before = len(self.prims) - start
            self.prims[start:] = _apply_clip(self.prims[start:], clip)
            self.stats.clip_removed += max(before - (len(self.prims) - start), 0)


def extract_primitives(
    doc: Drawing, sagitta: float, fast: bool = True
) -> tuple[list[Primitive], ExplodeStats]:
    ex = _Extractor(doc, sagitta, fast)
    for e in doc.modelspace():
        ex.stats.model_entities += 1
        ex.stats.model_types[e.dxftype()] += 1
        ex.visit(e)
    model, ex.prims = ex.prims, []
    for layout in doc.layouts:
        if layout.is_modelspace:
            continue
        ex.stats.layouts += 1
        for e in layout:
            if e.dxftype() == "VIEWPORT":
                view = _viewport_range(e)
                if view is not None:
                    ex.stats.viewports.append(view)
                continue
            try:
                ex.visit(e)
            except XrefMissingError:  # 레이아웃의 외부참조는 표제란 정도라 전체를 멈추지 않는다
                ex.stats.unsupported["PAPER_XREF"] += 1
    ex.stats.paper = ex.prims
    return model, ex.stats
