"""평면 오버레이 SVG. 원본 Primitive 위에 인식 결과를 겹쳐 그린다.

좌표는 그룹 변환 없이 직접 뒤집는다. y축 반전 그룹 안에서는 글자도 뒤집히기 때문이다.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from html import escape

from ..ir import IR
from ..preprocess.primitive import BBox, Primitive, union_bbox

_PADDING_MM = 500.0
_CONFIDENCE_CLASS = {"high": "ok", "review": "review"}


def _fmt(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


class _Canvas:
    def __init__(self, bbox: BBox):
        self.x0 = bbox[0] - _PADDING_MM
        self.y1 = bbox[3] + _PADDING_MM
        self.width = bbox[2] - bbox[0] + 2 * _PADDING_MM
        self.height = bbox[3] - bbox[1] + 2 * _PADDING_MM
        self.stroke = max(self.width, self.height) / 1500

    def xy(self, p: tuple[float, float]) -> tuple[str, str]:
        return _fmt(p[0] - self.x0), _fmt(self.y1 - p[1])

    def path(self, points: Iterable[tuple[float, float]], closed: bool = False) -> str:
        parts = []
        for i, p in enumerate(points):
            x, y = self.xy(p)
            parts.append(f"{'L' if i else 'M'}{x} {y}")
        return " ".join(parts) + (" Z" if closed else "")


def _arc_svg(c: _Canvas, g: dict) -> str:
    (cx, cy), r = g["center"], g["radius"]
    a0 = math.atan2(g["start"][1] - cy, g["start"][0] - cx)
    a1 = math.atan2(g["end"][1] - cy, g["end"][0] - cx)
    sweep = (a1 - a0) % (2 * math.pi)
    x0, y0 = c.xy(g["start"])
    x1, y1 = c.xy(g["end"])
    large = 1 if sweep > math.pi else 0
    # 화면 y가 뒤집혀 반시계 호는 sweep-flag 0이다.
    return f'<path d="M{x0} {y0} A{_fmt(r)} {_fmt(r)} 0 {large} 0 {x1} {y1}"/>'


def _primitive_svg(c: _Canvas, p: Primitive) -> str:
    g = p.geometry
    match p.kind:
        case "line":
            return f'<path d="{c.path([g["p1"], g["p2"]])}"/>'
        case "polyline":
            return f'<path d="{c.path(g["points"], g["closed"])}"/>'
        case "ellipse":
            return f'<path d="{c.path(g["points"])}"/>'
        case "solid":
            return f'<path class="hatch" d="{c.path(g["points"], True)}"/>'
        case "arc":
            return _arc_svg(c, g)
        case "circle":
            x, y = c.xy(g["center"])
            return f'<circle cx="{x}" cy="{y}" r="{_fmt(g["radius"])}"/>'
        case "hatch":
            d = " ".join(c.path(ring, True) for ring in g["rings"])
            return f'<path class="hatch" d="{d}"/>'
        case "text":
            x, y = c.xy(g["position"])
            size = _fmt(max(g["height"], c.stroke * 4))
            anchor = ' text-anchor="middle"' if g["centered"] else ""
            return f'<text x="{x}" y="{y}" font-size="{size}"{anchor}>{escape(g["text"])}</text>'
        case "dimension":
            return f'<path class="dim" d="{c.path([g["p1"], g["p2"]])}"/>'
        case _:
            return ""


def _ir_svg(c: _Canvas, ir: IR) -> list[str]:
    out: list[str] = []
    for w in ir.walls:
        cls = _CONFIDENCE_CLASS[w.confidence.level]
        out.append(
            f'<path class="wall {cls}" stroke-width="{_fmt(w.thickness)}" '
            f'd="{c.path([w.start, w.end])}"><title>{escape(w.id)}</title></path>'
        )
    for col in ir.columns:
        cls = _CONFIDENCE_CLASS[col.confidence.level]
        x, y = c.xy((col.center[0] - col.size[0] / 2, col.center[1] + col.size[1] / 2))
        out.append(
            f'<rect class="column {cls}" x="{x}" y="{y}" width="{_fmt(col.size[0])}" '
            f'height="{_fmt(col.size[1])}"><title>{escape(col.id)}</title></rect>'
        )
    walls = {w.id: w for w in ir.walls}
    for o in ir.openings:
        host = walls.get(o.host_wall)
        if host is None:
            continue
        length = math.dist(host.start, host.end) or 1.0
        t = o.position / length
        center = (
            host.start[0] + (host.end[0] - host.start[0]) * t,
            host.start[1] + (host.end[1] - host.start[1]) * t,
        )
        x, y = c.xy(center)
        cls = _CONFIDENCE_CLASS[o.confidence.level]
        out.append(
            f'<circle class="opening {cls}" cx="{x}" cy="{y}" r="{_fmt(o.width / 2)}">'
            f"<title>{escape(o.id)}</title></circle>"
        )
    for r in ir.rooms:
        cls = _CONFIDENCE_CLASS[r.confidence.level]
        label = escape(", ".join(r.names))
        out.append(
            f'<path class="room {cls}" d="{c.path(r.polygon, True)}"><title>{label}</title></path>'
        )
    return out


def render_svg(primitives: list[Primitive], excluded: list[Primitive], ir: IR) -> str:
    bbox = union_bbox(primitives + excluded) or (0.0, 0.0, 1000.0, 1000.0)
    c = _Canvas(bbox)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_fmt(c.width)} {_fmt(c.height)}" '
        f'style="--sw:{_fmt(c.stroke)}">',
        '<g class="layer-excluded">',
        *(s for p in excluded if (s := _primitive_svg(c, p))),
        "</g>",
        '<g class="layer-source">',
        *(s for p in primitives if (s := _primitive_svg(c, p))),
        "</g>",
        '<g class="layer-ir">',
        *_ir_svg(c, ir),
        "</g>",
        "</svg>",
    ]
    return "\n".join(parts)
