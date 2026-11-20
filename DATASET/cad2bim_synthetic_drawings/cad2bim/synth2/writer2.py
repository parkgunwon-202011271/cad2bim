"""절차 생성 평면(procedural.py)을 실무 도면처럼 DXF로 쓴다. 기존 생성기의 벽·해치·문·창·치수·도곽을 그대로 쓴다.

사용: python writer2.py --n 20 --seed 1 --out ../out2
  건물마다 폴더 하나, 층마다 'A-10N 지상N층 평면도.dxf'와 정답('*.truth.json'), 건물 정보('building.json').

더한 것:
- 문·창 종류: 양개문, 미닫이(호 없음), 승강기 문(호 없음), 점검문, 셔터(점선), 실내 유리, 넓은 열린 통로
- 계단 여러 개(UP/DN), 샤프트 상자(EV·PD·EPS, X 표시), FL 레벨 표기
- 레이어 체계: 영문(외부참조 접두사 포함), 회사 코드식, 한글, 약어, 거의 다 레이어 0
- 작성 습관: 블록 분해, 중복선, 3mm 틈, 객체별 색, 출력 안 하는 레이어의 잡선, 지시선 주기
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import ezdxf
from ezdxf.enums import TextEntityAlignment as TA
from ezdxf.math import Vec2
from shapely.geometry import Point, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth"))
import generate as base  # noqa: E402

import procedural  # noqa: E402

EXTRA = {"shutter": "door", "shaft": "note", "level": "note", "noplot": "note", "leader": "note"}
SCHEMES = {
    "en": dict(base.LAYER_SCHEMES["en"], shutter="A-DOOR-SHUT", shaft="A-FLOR-SHFT", level="A-ANNO-LEVL",
               noplot="NOPLOT", leader="A-ANNO-LEAD"),
    "kr": dict(base.LAYER_SCHEMES["kr"], shutter="셔터", shaft="샤프트", level="레벨", noplot="NOPLOT", leader="지시선"),
    "kr_code": dict(base.LAYER_SCHEMES["kr_code"], shutter="SHUT", shaft="SHFT", level="LEVL", noplot="NOPLOT",
                    leader="LEAD"),
    "company": {
        "wall_rc": "AA-WAXM-CONC", "wall_block": "AA-WAXM-SGPN-GLASSWOOL", "wall_hatch": "AA-WAXM-PATT",
        "column": "AA-COLS-CONC", "column_hatch": "AA-COLS-PATT", "door": "AA-DOOR", "window": "AA-WIND",
        "dim": "AA-DIMS", "room_name": "AA-ROOM-NAME", "room_area": "AA-ROOM-AREA", "furniture": "AA-EQPM",
        "fixture": "AA-TOIL", "grid": "AA-GRID", "grid_tag": "AA-GRID-TAG", "stair": "AA-STRS",
        "frame": "AA-FORM", "note": "AA-XXXX-TXT4", "shutter": "AA-DOOR-SHUT", "shaft": "AA-SHFT",
        "level": "AA-MKXS-LEVL", "noplot": "NOPLOT", "leader": "AA-LEAD",
    },
    "messy": dict(base.LAYER_SCHEMES["messy"], shutter="0", shaft="1", level="3", noplot="NOPLOT", leader="3"),
}
XREF_PREFIX = "XR_A_PLAN_B$0$"


class Writer2(base.Writer):
    def __init__(self, name, layout, cfg, seed):
        self.name, self.cfg = name, cfg
        self.L = layout
        self.layers = dict(SCHEMES[cfg["scheme"]])
        if cfg.get("xref"):
            self.layers = {k: (XREF_PREFIX + v if v not in ("0", "Defpoints") else v) for k, v in self.layers.items()}
        self.quirks = set(cfg["quirks"])
        self.rng = random.Random(seed)
        self.doc = ezdxf.new("R2018", setup=True)
        self.doc.header["$INSUNITS"] = cfg["insunits"]
        self.msp = self.doc.modelspace()
        self.g = base.build_geometry(self.L)
        self.truth_layers = {}
        self._setup_layers()
        self._setup_styles()

    def _setup_layers(self):
        for cls, lname in self.layers.items():
            self.truth_layers.setdefault(lname, set()).add(cls)
            if lname not in self.doc.layers:
                color = base.LAYER_COLORS.get(cls, base.LAYER_COLORS.get(EXTRA.get(cls, "note"), 7))
                lay = self.doc.layers.add(lname, color=color)
                if cls == "grid":
                    lay.dxf.linetype = "CENTER"
                if cls == "noplot":
                    lay.dxf.plot = 0

    # -- 문·창·열림
    def draw_openings(self):
        truth = []
        walls = self.g["walls"]
        explode = "explode" in self.quirks or not self.cfg["use_blocks"]
        for o in self.L["openings"]:
            oid, kind, wid, center, width = o[:5]
            w = walls[wid]
            a, b, u, n = base.wall_frame(w[1], w[2])
            c, t = Vec2(center), w[3]
            rec = {"id": oid, "kind": kind, "label": procedural.LABEL_OF[kind], "wall": wid,
                   "center": [round(c.x), round(c.y)], "width": width}
            if kind in ("door", "access_door", "double_door"):
                hinge_end, swing = o[5], Vec2(o[6])
                ends = sorted([c - u * (width / 2), c + u * (width / 2)], key=lambda p: (p.x, p.y))
                face = swing * (t / 2)
                if kind == "double_door":
                    half = width / 2
                    self._door(ends[0] + face, c + face, swing, half, t, explode)
                    self._door(ends[1] + face, c + face, swing, half, t, explode)
                else:
                    hinge = (ends[0] if hinge_end < 0 else ends[1]) + face
                    other = (ends[1] if hinge_end < 0 else ends[0]) + face
                    self._door(hinge, other, swing, width, t, explode)
                    rec["hinge"] = [round(hinge.x), round(hinge.y)]
                rec["swing_dir"] = [swing.x, swing.y]
            elif kind == "sliding":
                for k, off in ((-1, t / 6), (1, -t / 6)):  # 두 짝이 겹쳐 미끄러진다(호 없음)
                    p0 = c + u * (k * width / 4) + n * off
                    self.msp.add_lwpolyline([p0 - u * (width / 4 + 50) - n * 20, p0 + u * (width / 4 + 50) - n * 20,
                                             p0 + u * (width / 4 + 50) + n * 20, p0 - u * (width / 4 + 50) + n * 20],
                                            close=True, dxfattribs=self.attr("door"))
            elif kind == "ev_door":  # 승강기 문: 얇은 두 줄과 문틀만 있고 호가 없다
                for off in (-t / 2 - 60, -t / 2 - 120):
                    p0 = c + n * off
                    self.msp.add_line(p0 - u * (width / 2), p0 + u * (width / 2), dxfattribs=self.attr("door"))
                for s in (-1, 1):
                    q = c + u * (s * width / 2)
                    self.msp.add_line(q - n * (t / 2 + 150), q + n * (t / 2), dxfattribs=self.attr("door"))
            elif kind == "shutter":  # 셔터: 개구부를 따라 점선
                self.msp.add_line(c - u * (width / 2), c + u * (width / 2),
                                  dxfattribs=self.attr("shutter", linetype="DASHED", ltscale=20))
                self.msp.add_text("SH", height=200, dxfattribs=self.attr("shutter", style="KR")).set_placement(
                    c + n * (t / 2 + 400), align=TA.MIDDLE_CENTER)
            elif kind in ("window", "glass"):
                self._window(c, u, width, t, explode)
            # open: 그리지 않는다(벽만 끊김)
            truth.append(rec)
        return truth

    # -- 계단·샤프트·레벨
    def draw_stair(self):
        for s in self.L.get("stairs", []):
            at = lambda: self.attr("stair")  # noqa: E731
            mid = (s["x0"] + s["x1"]) / 2
            ytop = s["y0"] + s["tread"] * s["n"]
            for i in range(s["n"] + 1):
                y = s["y0"] + i * s["tread"]
                self.msp.add_line((s["x0"], y), (mid - 50, y), dxfattribs=at())
                self.msp.add_line((mid + 50, y), (s["x1"], y), dxfattribs=at())
            self.msp.add_lwpolyline([(mid - 50, s["y0"]), (mid - 50, ytop), (mid + 50, ytop), (mid + 50, s["y0"])],
                                    dxfattribs=at())
            for label, xq, flag in (("UP", (s["x0"] + mid) / 2, s.get("up", True)), ("DN", (mid + s["x1"]) / 2, s.get("down", True))):
                if not flag:
                    continue
                self.msp.add_lwpolyline([(xq, s["y0"] + 100), (xq, ytop - 200)], dxfattribs=at())
                self.msp.add_text(label, height=200, dxfattribs=self.attr("stair", style="KR")).set_placement(
                    (xq, s["y0"] - 250), align=TA.MIDDLE_CENTER)

    def draw_shafts(self):
        for s in self.L.get("shafts", []):
            x0, y0, x1, y1 = s["box"]
            a = self.attr("shaft")
            self.msp.add_lwpolyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], close=True, dxfattribs=a)
            self.msp.add_line((x0, y0), (x1, y1), dxfattribs=self.attr("shaft"))
            self.msp.add_line((x0, y1), (x1, y0), dxfattribs=self.attr("shaft"))
            self.msp.add_text(s["kind"], height=220, dxfattribs=self.attr("shaft", style="KR")).set_placement(
                ((x0 + x1) / 2, (y0 + y1) / 2 + 150), align=TA.MIDDLE_CENTER)

    def draw_levels(self):
        for text, (x, y) in self.L.get("levels", []):
            self.msp.add_circle((x, y), 250, dxfattribs=self.attr("level"))
            self.msp.add_line((x - 250, y), (x + 250, y), dxfattribs=self.attr("level"))
            self.msp.add_line((x, y - 250), (x, y + 250), dxfattribs=self.attr("level"))
            self.msp.add_text(text, height=250, dxfattribs=self.attr("level", style="KR")).set_placement((x + 450, y - 120))

    def draw_noise(self):
        """출력 안 하는 레이어의 잡선, 지시선 주기. 실무 도면의 잡음."""
        xs, ys = self.L["grid_x"], self.L["grid_y"]
        for _ in range(self.rng.randint(5, 20)):
            x, y = self.rng.uniform(xs[0], xs[-1]), self.rng.uniform(ys[0], ys[-1])
            self.msp.add_line((x, y), (x + self.rng.uniform(-3000, 3000), y + self.rng.uniform(-3000, 3000)),
                              dxfattribs=self.attr("noplot"))
        notes = ["THK 200 RC", "방화구획", "마감: 에폭시", "천장고 CH:2,700", "배수구", "점검구 600x600"]
        for _ in range(self.rng.randint(2, 6)):
            x, y = self.rng.uniform(xs[0], xs[-1]), self.rng.uniform(ys[0], ys[-1])
            self.msp.add_line((x, y), (x + 1200, y + 900), dxfattribs=self.attr("leader"))
            self.msp.add_text(self.rng.choice(notes), height=200, dxfattribs=self.attr("leader", style="KR")).set_placement(
                (x + 1300, y + 900))

    def run(self, out_dir: Path):
        rooms = safe_rooms(self.L, self.g)
        self.draw_walls()
        self.draw_columns()
        openings = self.draw_openings()
        fixtures = self.draw_fixtures()
        self.draw_stair()
        self.draw_shafts()
        self.draw_levels()
        self.draw_noise()
        self.draw_annotations(rooms)
        path = out_dir / f"{self.name}.dxf"
        self.doc.saveas(path, encoding="utf-8")
        truth = {
            "file": path.name, "floor": self.L["floor"], "title": self.L["title"], "scheme": self.cfg["scheme"],
            "xref_prefix": bool(self.cfg.get("xref")), "units": "mm" if self.cfg["insunits"] == 4 else "미지정(실제는 mm)",
            "quirks": sorted(self.quirks),
            "layer_classes": {k: sorted(v) for k, v in sorted(self.truth_layers.items())},
            "walls": [{"id": w[0], "p1": list(w[1]), "p2": list(w[2]), "thickness": w[3], "material": w[4]} for w in self.L["walls"]],
            "columns": [{"center": list(c), "size": self.L["column_size"]} for c in self.L["columns"]],
            "openings": openings, "pilotis": self.L.get("pilotis", []), "rooms": rooms, "fixtures": fixtures,
            "stairs": self.L.get("stairs", []), "shafts": self.L.get("shafts", []),
        }
        (out_dir / f"{self.name}.truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
        return path


def safe_rooms(L, g):
    """실 이름 위치가 벽에 걸리면 버린다(정답이 틀어지지 않게)."""
    xs, ys = L["grid_x"], L["grid_y"]
    free = box(min(xs), min(ys), max(xs), max(ys)).difference(g["closed_all"])
    regions = [free] if free.geom_type == "Polygon" else list(free.geoms)
    regions = [r for r in regions if r.area > 1e5]
    out = []
    for name, pt in L["rooms"]:
        idx = next((i for i, r in enumerate(regions) if r.contains(Point(pt))), None)
        if idx is None:
            continue
        out.append({"name": name, "label_point": list(pt), "region_id": idx, "clear_area_m2": round(regions[idx].area / 1e6, 2)})
    return out


def style_for(rng) -> dict:
    scheme = rng.choice(["en", "en", "kr", "kr_code", "company", "company", "messy"])
    quirks = rng.sample(["explode", "duplicate_lines", "gaps", "entity_colors"], rng.randint(0, 3))
    return dict(scheme=scheme, xref=scheme in ("en", "company") and rng.random() < 0.5,
                use_blocks=rng.random() < 0.7, walls_as=rng.choice(["lines", "polylines", "mixed"]),
                block_names=rng.choice(["en", "kr", "code"]), insunits=rng.choice([4, 4, 4, 0]), quirks=quirks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kind", default=None, choices=[None, "office", "warehouse"])
    ap.add_argument("--out", default="../out2")
    ap.add_argument("--bench", action="store_true", help="평가 틀(cad2bim benchmark)용 정답과 매니페스트도 쓴다")
    args = ap.parse_args()
    root = Path(args.out)
    for k in range(args.n):
        seed = args.seed + k
        made = procedural.building(seed, args.kind)
        b = made["building"]
        folder = root / b["id"]
        folder.mkdir(parents=True, exist_ok=True)
        cfg = style_for(random.Random(seed * 7919))  # 한 건물은 한 사무소가 그린 것처럼 같은 체계
        for i, L in enumerate(made["layouts"], 1):
            floor = L["floor"]
            title = "지붕층 평면도" if floor == "R" else f"지상{floor[1:]}층 평면도"
            L["title"] = title
            name = f"A-1{i:02d} {title}"
            Writer2(name, L, cfg, seed * 31 + i).run(folder)
        (folder / "building.json").write_text(json.dumps({**b, "style": cfg}, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", folder, b["kind"], b["floors"], "floors", cfg["scheme"], "xref" if cfg["xref"] else "")
        if args.bench:
            bench_truth(folder)
    if args.bench:
        manifest(root)



def bench_truth(folder: Path) -> None:
    """평가 틀용 정답(*.bench.json): 문·창 종류를 door/window로 묶고, 문짝 없는 열린 통로는 뺀다."""
    for f in folder.glob("*.truth.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        d["openings"] = [{**o, "kind": o["label"]} for o in d["openings"] if o["label"] in ("door", "window")]
        f.with_name(f.name.replace(".truth.json", ".bench.json")).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def manifest(root: Path) -> None:
    """cad2bim benchmark 매니페스트. 지붕층은 파라펫뿐이라 뺀다."""
    lines = ["# synth2 평가 세트(자동 생성). 학습·튜닝에 쓰지 않은 건물이다.", "name: synth2", f"root: {root.resolve().as_posix()}",
             "floor_height: 3000", "", "drawings:"]
    for t in sorted(root.glob("S*/*.bench.json")):
        if "지붕층" in t.name:
            continue
        rel = t.relative_to(root).as_posix()
        dxf = rel.replace(".bench.json", ".dxf")
        lines.append(f"  - {{id: {t.parent.name}_{t.name.split()[0]}, dxf: '{dxf}', truth: '{rel}', split: test, gated: false}}")
    lines += ["", "gates: {}"]
    (root / "synth2_bench.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
