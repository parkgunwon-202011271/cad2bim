"""테스트용 가상 DXF 도면과 정답(JSON) 생성기.

실제 도면 없이 변환 코드를 개발·테스트하기 위한 데이터입니다.
레이아웃(A, B) x 도면 작성 습관(레이어 체계, 블록 사용, 지저분함)을 조합해 만듭니다.

사용법:
    python generate.py --out ../out
"""
import argparse
import json
import math
import random
from pathlib import Path

import ezdxf
from ezdxf import colors
from ezdxf.math import Vec2
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from layouts import LAYOUTS, RC, BLOCK

# ---------------------------------------------------------------- 레이어 체계
# 요소 종류(class) -> 레이어명. 정답 파일에도 그대로 기록해 레이어 분류기 테스트에 쓴다.
LAYER_SCHEMES = {
    "en": {
        "wall_rc": "A-WALL-CONC", "wall_block": "A-WALL-BLCK", "wall_hatch": "A-WALL-PATT",
        "column": "S-COLS", "column_hatch": "S-COLS-PATT", "door": "A-DOOR", "window": "A-GLAZ",
        "dim": "A-ANNO-DIMS", "room_name": "A-AREA-IDEN", "room_area": "A-AREA-IDEN",
        "furniture": "A-FURN", "fixture": "P-FIXT", "grid": "S-GRID", "grid_tag": "S-GRID-IDEN",
        "stair": "A-FLOR-STRS", "frame": "A-ANNO-TTLB", "note": "A-ANNO-NOTE",
    },
    "kr": {
        "wall_rc": "벽체-RC", "wall_block": "벽체-조적", "wall_hatch": "벽체해치",
        "column": "기둥", "column_hatch": "기둥해치", "door": "문", "window": "창호",
        "dim": "치수", "room_name": "실명", "room_area": "면적",
        "furniture": "가구", "fixture": "위생기구", "grid": "중심선", "grid_tag": "중심선-번호",
        "stair": "계단", "frame": "도곽", "note": "주기",
    },
    "kr_code": {
        "wall_rc": "CON", "wall_block": "BLK", "wall_hatch": "HAT",
        "column": "COL", "column_hatch": "COL", "door": "DOOR", "window": "WIN",
        "dim": "DIM", "room_name": "TEXT", "room_area": "TEXT",
        "furniture": "FUR", "fixture": "FUR", "grid": "CEN", "grid_tag": "CEN",
        "stair": "STAIR", "frame": "FORM", "note": "TEXT",
    },
    "messy": {
        "wall_rc": "0", "wall_block": "0", "wall_hatch": "0",
        "column": "1", "column_hatch": "1", "door": "0", "window": "2",
        "dim": "DIM", "room_name": "3", "room_area": "3",
        "furniture": "0", "fixture": "0", "grid": "Defpoints", "grid_tag": "Defpoints",
        "stair": "0", "frame": "도곽", "note": "3",
    },
}

LAYER_COLORS = {"wall_rc": 7, "wall_block": 7, "wall_hatch": 8, "column": 7, "column_hatch": 8,
                "door": 3, "window": 4, "dim": 2, "room_name": 7, "room_area": 7, "furniture": 9,
                "fixture": 9, "grid": 1, "grid_tag": 1, "stair": 6, "frame": 7, "note": 7}

# ---------------------------------------------------------------- 변형(도면 작성 습관)
VARIANTS = {
    # name: layout, layer scheme, options
    "A1_retail_clean_en": dict(layout="A", scheme="en", use_blocks=True, walls_as="lines",
                               block_names="en", insunits=4, quirks=[]),
    "A2_retail_messy": dict(layout="A", scheme="messy", use_blocks=False, walls_as="lines",
                            block_names="en", insunits=0,
                            quirks=["explode", "duplicate_lines", "gaps", "entity_colors"]),
    "B1_apt_clean_kr": dict(layout="B", scheme="kr", use_blocks=True, walls_as="polylines",
                            block_names="kr", insunits=4, quirks=[]),
    "B2_apt_partial_code": dict(layout="B", scheme="kr_code", use_blocks=True, walls_as="mixed",
                                block_names="code", insunits=4,
                                quirks=["single_line_partition", "mislayered_hatch",
                                        "nested_furniture", "scaled_window_block"]),
}

QUIRK_NOTES = {
    "explode": "문·창·가구가 블록이 아니라 분해된 선과 호로 그려져 있음",
    "duplicate_lines": "벽 선 일부가 같은 자리에 두 번 그려져 있음",
    "gaps": "벽 선 일부가 3mm 틈으로 끊겨 있음",
    "entity_colors": "레이어 색이 아니라 객체별 색을 지정함",
    "single_line_partition": "현관 칸막이(W10)가 두께 없이 중심선 한 줄로만 그려져 있음",
    "mislayered_hatch": "조적벽 해치가 DIM 레이어에 잘못 들어가 있음",
    "nested_furniture": "가구 블록이 다른 블록(UNIT-FUR) 안에 중첩되어 있음",
    "scaled_window_block": "창 블록 하나(W)를 비균등 축척으로 늘려 모든 창에 사용함",
}


# ---------------------------------------------------------------- 기하 계산
def wall_frame(p1, p2):
    a, b = Vec2(p1), Vec2(p2)
    u = (b - a).normalize()
    n = u.orthogonal()  # +90도
    return a, b, u, n


def wall_rect(p1, p2, t, extend):
    a, b, u, n = wall_frame(p1, p2)
    a2, b2 = a - u * extend, b + u * extend
    h = n * (t / 2)
    return Polygon([a2 + h, b2 + h, b2 - h, a2 - h])


def opening_rect(wall, center, width):
    _, p1, p2, t, _ = wall
    a, b, u, n = wall_frame(p1, p2)
    c = Vec2(center)
    h = n * (t / 2 + 5)
    w = u * (width / 2)
    return Polygon([c - w + h, c + w + h, c + w - h, c - w - h])


def build_geometry(L):
    walls = {w[0]: w for w in L["walls"]}
    cs = L["column_size"]
    cols = [box(x - cs / 2, y - cs / 2, x + cs / 2, y + cs / 2) for x, y in L["columns"]]
    col_union = unary_union(cols) if cols else Polygon()
    op_polys = [opening_rect(walls[o[2]], o[3], o[4]) for o in L["openings"]]
    op_union = unary_union(op_polys)

    def mat_union(mat):
        polys = [wall_rect(w[1], w[2], w[3], 100) for w in L["walls"] if w[4] == mat]
        return unary_union(polys) if polys else Polygon()

    rc_closed = mat_union(RC)
    blk_closed = mat_union(BLOCK).difference(rc_closed)
    rc = rc_closed.difference(op_union).difference(col_union)
    blk = blk_closed.difference(op_union).difference(col_union)
    closed_all = unary_union([rc_closed, blk_closed, col_union])
    return dict(walls=walls, cols=cols, rc=rc, blk=blk, closed_all=closed_all)


def rings(geom):
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    out = []
    for p in polys:
        if p.is_empty:
            continue
        out.append(list(p.exterior.coords))
        out.extend(list(r.coords) for r in p.interiors)
    return out


def ring_segments(ring):
    return [(ring[i], ring[i + 1]) for i in range(len(ring) - 1)]


def rooms_truth(L, g):
    xs, ys = L["grid_x"], L["grid_y"]
    footprint = box(min(xs), min(ys), max(xs), max(ys))
    free = footprint.difference(g["closed_all"])
    regions = [free] if free.geom_type == "Polygon" else list(free.geoms)
    regions = [r for r in regions if r.area > 1e5]
    out = []
    for name, pt in L["rooms"]:
        idx = next(i for i, r in enumerate(regions) if r.contains(Point(pt)))
        out.append({"name": name, "label_point": list(pt), "region_id": idx,
                    "clear_area_m2": round(regions[idx].area / 1e6, 2)})
    shared = {}
    for r in out:
        shared.setdefault(r["region_id"], []).append(r["name"])
    for r in out:
        others = [n for n in shared[r["region_id"]] if n != r["name"]]
        if others:
            r["shares_region_with"] = others
    return out


# ---------------------------------------------------------------- 블록 정의
def block_name(kind, size, style):
    w, t = size
    if style == "en":
        return {"door": f"DOOR_SGL_{w}", "window": f"WIN_{w}x{t}"}[kind]
    if style == "kr":
        return {"door": f"문-{w}", "window": f"창-{w}x{t}"}[kind]
    return {"door": f"D{w}", "window": "W"}[kind]


FIXTURE_NAMES = {
    "en": {"WC": "WC", "LAV": "LAV", "BED_Q": "BED_QUEEN", "BED_S": "BED_SINGLE", "SOFA": "SOFA_3P",
           "TABLE4": "DINING_4P", "COUNTER": "KITCHEN_CNT"},
    "kr": {"WC": "양변기", "LAV": "세면기", "BED_Q": "침대-퀸", "BED_S": "침대-싱글", "SOFA": "소파",
           "TABLE4": "식탁-4인", "COUNTER": "주방가구"},
    "code": {"WC": "F-WC", "LAV": "F-LAV", "BED_Q": "F-BED1", "BED_S": "F-BED2", "SOFA": "F-SOFA",
             "TABLE4": "F-TBL", "COUNTER": "F-KIT"},
}


def fixture_shapes(kind):
    """가구·위생기구 형상: (type, data) 리스트. 원점 = 삽입점."""
    if kind == "WC":
        return [("rect", (-220, -150, 220, 120)), ("ellipse", ((0, -350), 200, 260))]
    if kind == "LAV":
        return [("rect", (-300, -230, 300, 230)), ("ellipse", ((0, 0), 220, 150))]
    if kind == "BED_Q":
        return [("rect", (-750, -1000, 750, 1000)), ("rect", (-650, 650, -50, 950)),
                ("rect", (50, 650, 650, 950))]
    if kind == "BED_S":
        return [("rect", (-500, -1000, 500, 1000)), ("rect", (-400, 650, 400, 950))]
    if kind == "SOFA":
        return [("rect", (-1200, -450, 1200, 450)), ("rect", (-1200, 250, 1200, 450)),
                ("rect", (-1200, -450, -1000, 450)), ("rect", (1000, -450, 1200, 450))]
    if kind == "TABLE4":
        out = [("rect", (-700, -400, 700, 400))]
        for x in (-1000, 1000):
            for y in (-200, 200):
                out.append(("circle", ((x, y), 220)))
        return out
    if kind == "COUNTER":
        return [("rect", (-300, -1000, 300, 1000)), ("circle", ((0, 300), 180)),
                ("circle", ((0, -400), 120)), ("circle", ((0, -700), 120))]
    raise KeyError(kind)


def add_shapes(target, shapes, attribs, xform=None):
    """shapes를 target(블록 또는 모델공간)에 그린다. xform: (insert, rot_deg) 이면 직접 변환(분해 상태)."""
    def tp(x, y):
        if not xform:
            return (x, y)
        (ix, iy), r = xform
        c, s = math.cos(math.radians(r)), math.sin(math.radians(r))
        return (ix + x * c - y * s, iy + x * s + y * c)

    for kind, d in shapes:
        if kind == "rect":
            x0, y0, x1, y1 = d
            target.add_lwpolyline([tp(x0, y0), tp(x1, y0), tp(x1, y1), tp(x0, y1)], close=True,
                                  dxfattribs=attribs)
        elif kind == "circle":
            (cx, cy), r = d
            target.add_circle(tp(cx, cy), r, dxfattribs=attribs)
        elif kind == "ellipse":
            (cx, cy), rx, ry = d
            rot = xform[1] if xform else 0
            if ry > rx:  # DXF 타원은 장축 기준이므로 축을 바꾼다
                rx, ry, rot = ry, rx, rot + 90
            major = Vec2(rx, 0).rotate_deg(rot)
            target.add_ellipse(tp(cx, cy), major_axis=(major.x, major.y), ratio=ry / rx,
                               dxfattribs=attribs)


# ---------------------------------------------------------------- 도면 작성
class Writer:
    def __init__(self, name, cfg, seed=7):
        self.name, self.cfg = name, cfg
        self.L = LAYOUTS[cfg["layout"]]
        self.layers = LAYER_SCHEMES[cfg["scheme"]]
        self.quirks = set(cfg["quirks"])
        self.rng = random.Random(seed)
        self.doc = ezdxf.new("R2018", setup=True)
        self.doc.header["$INSUNITS"] = cfg["insunits"]
        self.msp = self.doc.modelspace()
        self.g = build_geometry(self.L)
        self.truth_layers = {}
        self._setup_layers()
        self._setup_styles()

    # -- 공통
    def _setup_layers(self):
        for cls, lname in self.layers.items():
            self.truth_layers.setdefault(lname, set()).add(cls)
            if lname not in self.doc.layers:
                lay = self.doc.layers.add(lname, color=LAYER_COLORS[cls])
                if cls == "grid":
                    lay.dxf.linetype = "CENTER"

    def _setup_styles(self):
        if "KR" not in self.doc.styles:
            self.doc.styles.add("KR", font="malgun.ttf")
        ds = self.doc.dimstyles.new("KR-100")
        ds.dxf.dimtxt = 250
        ds.dxf.dimasz = 150
        ds.dxf.dimexo = 150
        ds.dxf.dimexe = 100
        ds.dxf.dimgap = 80
        ds.dxf.dimdec = 0
        ds.dxf.dimtxsty = "KR"
        ds.set_arrows(blk=ezdxf.ARROWS.architectural_tick)

    def attr(self, cls, **extra):
        a = {"layer": self.layers[cls]}
        if "entity_colors" in self.quirks:
            a["color"] = self.rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
        a.update(extra)
        return a

    # -- 벽
    def draw_walls(self):
        g = self.g
        rc_segs = []
        for mat, geom in ((RC, g["rc"]), (BLOCK, g["blk"])):
            cls = "wall_rc" if mat == RC else "wall_block"
            for ring in rings(geom):
                segs = ring_segments(ring)
                if mat == BLOCK:  # RC와 맞닿는 면은 이미 그려져 있으므로 생략
                    segs = [s for s in segs if not any(
                        LineString(s).within(LineString(r).buffer(1)) for r in rc_segs)]
                else:
                    rc_segs.extend(segs)
                self._emit_wall_segments(segs, cls, mat)
            self._hatch(geom, "wall_hatch" if not (mat == BLOCK and "mislayered_hatch" in self.quirks)
                        else "dim", "AR-CONC" if mat == RC else "ANSI31",
                        20 if mat == RC else 30)

        if "single_line_partition" in self.quirks:
            w = g["walls"]["W10"]
            self.msp.add_line(w[1], w[2], dxfattribs=self.attr("wall_block"))

    def _emit_wall_segments(self, segs, cls, mat):
        mode = self.cfg["walls_as"]
        if mode == "mixed":
            mode = "polylines" if mat == RC else "lines"
        if "single_line_partition" in self.quirks and mat == BLOCK:
            w10 = wall_rect(self.g["walls"]["W10"][1], self.g["walls"]["W10"][2], 100, 0).buffer(2)
            segs = [s for s in segs if not LineString(s).within(w10)]
        if mode == "polylines":
            for chain in self._chain(segs):
                closed = Vec2(chain[0]).isclose(Vec2(chain[-1]), abs_tol=1e-6)
                pts = chain[:-1] if closed else chain
                self.msp.add_lwpolyline(pts, close=closed, dxfattribs=self.attr(cls))
            return
        for a, b in segs:
            if "gaps" in self.quirks and LineString([a, b]).length > 1500 and self.rng.random() < 0.15:
                m = (Vec2(a) + Vec2(b)) / 2
                d = (Vec2(b) - Vec2(a)).normalize() * 1.5
                self.msp.add_line(a, m - d, dxfattribs=self.attr(cls))
                self.msp.add_line(m + d, b, dxfattribs=self.attr(cls))
                continue
            self.msp.add_line(a, b, dxfattribs=self.attr(cls))
            if "duplicate_lines" in self.quirks and self.rng.random() < 0.1:
                self.msp.add_line(a, b, dxfattribs=self.attr(cls))

    @staticmethod
    def _chain(segs):
        segs = [list(s) for s in segs]
        chains = []
        while segs:
            a, b = segs.pop(0)
            chain = [a, b]
            grown = True
            while grown:
                grown = False
                for i, (c, d) in enumerate(segs):
                    if Vec2(c).isclose(Vec2(chain[-1]), abs_tol=1e-6):
                        chain.append(d); segs.pop(i); grown = True; break
            chains.append(chain)
        return chains

    def _hatch(self, geom, cls, pattern, scale):
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for p in polys:
            if p.is_empty:
                continue
            h = self.msp.add_hatch(dxfattribs=self.attr(cls))
            h.set_pattern_fill(pattern, scale=scale)
            h.paths.add_polyline_path(list(p.exterior.coords)[:-1], is_closed=True, flags=1)
            for r in p.interiors:
                h.paths.add_polyline_path(list(r.coords)[:-1], is_closed=True, flags=0)

    # -- 기둥
    def draw_columns(self):
        for c in self.g["cols"]:
            pts = list(c.exterior.coords)[:-1]
            self.msp.add_lwpolyline(pts, close=True, dxfattribs=self.attr("column"))
            self._hatch(c, "column_hatch", "ANSI31", 15)

    # -- 문, 창
    def draw_openings(self):
        truth = []
        walls = self.g["walls"]
        explode = "explode" in self.quirks or not self.cfg["use_blocks"]
        for o in self.L["openings"]:
            oid, kind, wid, center, width = o[:5]
            w = walls[wid]
            a, b, u, n = wall_frame(w[1], w[2])
            c = Vec2(center)
            t = w[3]
            rec = {"id": oid, "kind": kind, "wall": wid, "center": list(center), "width": width}
            if kind == "door":
                hinge_end, swing = o[5], Vec2(o[6])
                # hinge_end: -1이면 좌표값이 작은 쪽 문틀
                ends = [c - u * (width / 2), c + u * (width / 2)]
                ends.sort(key=lambda p: (p.x, p.y))
                hinge_axis = ends[0] if hinge_end < 0 else ends[1]
                other_axis = ends[1] if hinge_end < 0 else ends[0]
                face = swing * (t / 2)
                hinge, other = hinge_axis + face, other_axis + face
                rec.update(hinge=[round(hinge.x), round(hinge.y)], swing_dir=list(o[6]))
                self._door(hinge, other, swing, width, t, explode)
            else:
                self._window(c, u, width, t, explode)
            truth.append(rec)
        return truth

    def _door(self, hinge, other, swing, width, t, explode):
        d = (other - hinge).normalize()
        leaf_end = hinge + swing * width
        if explode:
            at = self.attr("door")
            self.msp.add_line(hinge, leaf_end, dxfattribs=at)
            a1, a2 = d.angle_deg, swing.angle_deg
            if (a2 - a1) % 360 > 180:
                a1, a2 = a2, a1
            self.msp.add_arc(hinge, width, a1, a2, dxfattribs=self.attr("door"))
            return
        name = block_name("door", (width, t), self.cfg["block_names"])
        if name not in self.doc.blocks:
            blk = self.doc.blocks.new(name)
            blk.add_line((0, 0), (0, width), dxfattribs={"layer": "0"})
            blk.add_arc((0, 0), width, 0, 90, dxfattribs={"layer": "0"})
        rot = d.angle_deg
        local_y = d.orthogonal()
        mirror = local_y.dot(swing) < 0
        self.msp.add_blockref(name, hinge, dxfattribs=self.attr(
            "door", rotation=rot, yscale=-1 if mirror else 1))

    def _window(self, c, u, width, t, explode):
        n = u.orthogonal()
        offs = [t / 2, t / 10, -t / 10, -t / 2]
        if explode:
            for o in offs:
                p = c + n * o
                self.msp.add_line(p - u * (width / 2), p + u * (width / 2),
                                  dxfattribs=self.attr("window"))
            return
        if "scaled_window_block" in self.quirks:
            name = "W"
            if name not in self.doc.blocks:
                blk = self.doc.blocks.new(name)
                for o in (0.5, 0.1, -0.1, -0.5):
                    blk.add_line((-0.5, o), (0.5, o), dxfattribs={"layer": "0"})
            self.msp.add_blockref(name, c, dxfattribs=self.attr(
                "window", rotation=u.angle_deg, xscale=width, yscale=t))
            return
        name = block_name("window", (width, t), self.cfg["block_names"])
        if name not in self.doc.blocks:
            blk = self.doc.blocks.new(name)
            for o in offs:
                blk.add_line((-width / 2, o), (width / 2, o), dxfattribs={"layer": "0"})
        self.msp.add_blockref(name, c, dxfattribs=self.attr("window", rotation=u.angle_deg))

    # -- 가구, 위생기구, 계단
    def draw_fixtures(self):
        names = FIXTURE_NAMES[self.cfg["block_names"]]
        explode = "explode" in self.quirks or not self.cfg["use_blocks"]
        nested = "nested_furniture" in self.quirks
        truth = []
        container = None
        if nested:
            container = self.doc.blocks.new("UNIT-FUR")
        for kind, pt, rot in self.L["fixtures"]:
            cls = "fixture" if kind in ("WC", "LAV") else "furniture"
            truth.append({"type": kind, "insert": list(pt), "rotation": rot})
            if explode:
                add_shapes(self.msp, fixture_shapes(kind), self.attr(cls), xform=(pt, rot))
                continue
            name = names[kind]
            if name not in self.doc.blocks:
                add_shapes(self.doc.blocks.new(name), fixture_shapes(kind), {"layer": "0"})
            if nested and cls == "furniture":
                container.add_blockref(name, pt, dxfattribs={"layer": self.layers[cls],
                                                             "rotation": rot})
            else:
                self.msp.add_blockref(name, pt, dxfattribs=self.attr(cls, rotation=rot))
        if nested:
            self.msp.add_blockref("UNIT-FUR", (0, 0), dxfattribs=self.attr("furniture"))
        return truth

    def draw_stair(self):
        s = self.L["stair"]
        if not s:
            return
        at = lambda: self.attr("stair")
        mid = (s["x0"] + s["x1"]) / 2
        ytop = s["y0"] + s["tread"] * s["n"]
        for i in range(s["n"] + 1):
            y = s["y0"] + i * s["tread"]
            self.msp.add_line((s["x0"], y), (mid - 50, y), dxfattribs=at())
            self.msp.add_line((mid + 50, y), (s["x1"], y), dxfattribs=at())
        self.msp.add_lwpolyline([(mid - 50, s["y0"]), (mid - 50, ytop), (mid + 50, ytop),
                                 (mid + 50, s["y0"])], dxfattribs=at())
        q = (s["x0"] + mid) / 2
        self.msp.add_lwpolyline([(q, s["y0"] + 100), (q, ytop - 200)], dxfattribs=at())
        self.msp.add_lwpolyline([(q - 120, ytop - 400), (q, ytop - 200), (q + 120, ytop - 400)],
                                dxfattribs=at())
        self.msp.add_text("UP", height=200, dxfattribs=self.attr("stair", style="KR")).set_placement(
            (q, s["y0"] - 250), align=ezdxf.enums.TextEntityAlignment.MIDDLE_CENTER)

    # -- 주석
    def draw_annotations(self, rooms):
        from ezdxf.enums import TextEntityAlignment as TA
        xs, ys = self.L["grid_x"], self.L["grid_y"]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        mtext = "explode" in self.quirks  # 지저분한 도면은 MTEXT + 서식 코드
        for r in rooms:
            x, y = r["label_point"]
            if mtext:
                self.msp.add_mtext(r"{\W0.9;\L" + r["name"] + "}",
                                   dxfattribs=self.attr("room_name", char_height=300,
                                                        insert=(x, y), attachment_point=5))
            else:
                self.msp.add_text(r["name"], height=300, dxfattribs=self.attr(
                    "room_name", style="KR")).set_placement((x, y), align=TA.MIDDLE_CENTER)
            if r.get("shares_region_with"):  # 벽 없이 트인 공간(LDK)은 면적을 따로 적지 않음
                continue
            self.msp.add_text(f"({r['clear_area_m2']:.2f}㎡)", height=200, dxfattribs=self.attr(
                "room_area", style="KR")).set_placement((x, y - 400), align=TA.MIDDLE_CENTER)

        # 그리드 + 번호
        ext = 2500
        for i, x in enumerate(xs, 1):
            self.msp.add_line((x, y0 - ext), (x, y1 + 1500), dxfattribs=self.attr("grid"))
            self._grid_tag(f"X{i}", (x, y0 - ext - 450))
        for i, y in enumerate(ys, 1):
            self.msp.add_line((x0 - ext, y), (x1 + 1500, y), dxfattribs=self.attr("grid"))
            self._grid_tag(f"Y{i}", (x0 - ext - 450, y))

        # 치수: 그리드 간격 + 전체
        da = self.attr("dim")
        for a, b in zip(xs, xs[1:]):
            self.msp.add_linear_dim(base=(a, y0 - 1200), p1=(a, y0), p2=(b, y0),
                                    dimstyle="KR-100", dxfattribs=da).render()
        self.msp.add_linear_dim(base=(x0, y0 - 1900), p1=(x0, y0), p2=(x1, y0),
                                dimstyle="KR-100", dxfattribs=da).render()
        for a, b in zip(ys, ys[1:]):
            self.msp.add_linear_dim(base=(x0 - 1200, a), p1=(x0, a), p2=(x0, b), angle=90,
                                    dimstyle="KR-100", dxfattribs=da).render()
        self.msp.add_linear_dim(base=(x0 - 1900, y0), p1=(x0, y0), p2=(x0, y1), angle=90,
                                dimstyle="KR-100", dxfattribs=da).render()

        # 도곽 + 표제란 + 주기
        fx0, fy0, fx1, fy1 = x0 - 5500, y0 - 6500, x1 + 6500, y1 + 3500
        fa = self.attr("frame")
        self.msp.add_lwpolyline([(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)], close=True,
                                dxfattribs=fa)
        tx0 = fx1 - 7000
        self.msp.add_lwpolyline([(tx0, fy0), (fx1, fy0), (fx1, fy0 + 2400), (tx0, fy0 + 2400)],
                                close=True, dxfattribs=self.attr("frame"))
        for i, line in enumerate([self.L["title"], "SCALE 1/100", "도면번호 A-101"]):
            self.msp.add_text(line, height=280, dxfattribs=self.attr("frame", style="KR")).set_placement(
                (tx0 + 300, fy0 + 1900 - i * 650))
        self.msp.add_text("※ 본 도면은 개발 테스트용 가상 도면임", height=220, dxfattribs=self.attr(
            "note", style="KR")).set_placement((fx0 + 400, fy0 + 500))

    def _grid_tag(self, label, c):
        from ezdxf.enums import TextEntityAlignment as TA
        self.msp.add_circle(c, 400, dxfattribs=self.attr("grid_tag"))
        self.msp.add_text(label, height=280, dxfattribs=self.attr("grid_tag", style="KR")).set_placement(
            c, align=TA.MIDDLE_CENTER)

    # -- 실행
    def run(self, out_dir: Path):
        rooms = rooms_truth(self.L, self.g)
        self.draw_walls()
        self.draw_columns()
        openings = self.draw_openings()
        fixtures = self.draw_fixtures()
        self.draw_stair()
        self.draw_annotations(rooms)
        dxf_path = out_dir / f"{self.name}.dxf"
        self.doc.saveas(dxf_path, encoding="utf-8")

        truth = {
            "file": dxf_path.name,
            "layout": self.L["id"],
            "description": self.L["title"],
            "units": "mm" if self.cfg["insunits"] == 4 else "미지정(실제는 mm)",
            "quirks": [{"id": q, "note": QUIRK_NOTES[q]} for q in self.cfg["quirks"]],
            "layer_classes": {k: sorted(v) for k, v in sorted(self.truth_layers.items())},
            "walls": [{"id": w[0], "p1": list(w[1]), "p2": list(w[2]), "thickness": w[3],
                       "material": w[4]} for w in self.L["walls"]],
            "columns": [{"center": list(c), "size": self.L["column_size"]} for c in self.L["columns"]],
            "openings": openings,
            "rooms": rooms,
            "fixtures": fixtures,
            "counts": {
                "walls": len(self.L["walls"]),
                "columns": len(self.L["columns"]),
                "doors": sum(o["kind"] == "door" for o in openings),
                "windows": sum(o["kind"] == "window" for o in openings),
                "rooms": len(rooms),
                "room_regions": len({r["region_id"] for r in rooms}),
            },
        }
        if self.cfg["quirks"] and "mislayered_hatch" in self.cfg["quirks"]:
            truth["layer_classes"].setdefault("DIM", [])
            if "wall_hatch" not in truth["layer_classes"]["DIM"]:
                truth["layer_classes"]["DIM"] = sorted(set(truth["layer_classes"]["DIM"]) |
                                                       {"wall_hatch(잘못 배치)"})
        (out_dir / f"{self.name}.truth.json").write_text(
            json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
        return dxf_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, cfg in VARIANTS.items():
        p = Writer(name, cfg).run(out)
        print("wrote", p)


if __name__ == "__main__":
    main()
