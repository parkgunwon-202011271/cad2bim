"""절차 생성 평면: 여러 층으로 된 가상 건물을 무작위로 만든다. 실제 도면과 무관한 지어낸 평면이다.

좌표는 mm, 벽은 중심선. 층마다 기존 생성기(synth/generate.py)의 레이아웃 형식(dict)을 만들고, 그 위에
새 키를 더한다: stairs(여러 개), shafts(EV·PD·EPS 상자), levels(FL 표기), floor(층 키), building(건물 정보).

문·창·열림 종류(opening kind)와 학습 정답(label):
  door(여닫이), double_door(양개), sliding(미닫이), ev_door(승강기 문, 호 없음), access_door(점검문),
  shutter(셔터, 점선) -> door
  window(외벽 창), glass(실내 유리벽) -> window
  open(문짝 없는 넓은 통로) -> open
  필로티: 외벽이 없는 외곽 구간 (building.pilotis에 기록) -> open
오늘(2026-11-19) 모델들이 헷갈린 경우(호 없는 문, 실내 유리, 넓은 통로, 필로티, 기둥 모서리)를 일부러 섞는다.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

RC, BLOCK = "RC", "BLOCK"
LABEL_OF = {
    "door": "door", "double_door": "door", "sliding": "door", "ev_door": "door", "access_door": "door",
    "shutter": "door", "window": "window", "glass": "window", "open": "open",
}


@dataclass
class Plan:
    """한 층 평면을 쌓는 도우미. 벽·문을 추가하면 id를 붙여 준다."""
    floor: str
    title: str
    grid_x: list[float]
    grid_y: list[float]
    column_size: int
    walls: list = field(default_factory=list)
    openings: list = field(default_factory=list)
    rooms: list = field(default_factory=list)
    fixtures: list = field(default_factory=list)
    stairs: list = field(default_factory=list)
    shafts: list = field(default_factory=list)
    levels: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    pilotis: list = field(default_factory=list)

    def wall(self, p1, p2, t, mat) -> str:
        wid = f"W{len(self.walls) + 1}"
        self.walls.append((wid, tuple(map(float, p1)), tuple(map(float, p2)), t, mat))
        return wid

    def opening(self, kind, wid, center, width, hinge=None, swing=None) -> str:
        prefix = {"window": "WN", "glass": "GL", "open": "OP", "shutter": "SH"}.get(kind, "D")
        oid = f"{prefix}{len(self.openings) + 1}"
        rec = (oid, kind, wid, tuple(map(float, center)), float(width))
        if hinge is not None:
            rec += (hinge, swing)
        self.openings.append(rec)
        return oid

    def layout(self, building: dict) -> dict:
        return {
            "id": f"{building['id']}_{self.floor}", "title": self.title, "floor": self.floor,
            "grid_x": self.grid_x, "grid_y": self.grid_y, "column_size": self.column_size,
            "columns": self.columns, "walls": self.walls, "openings": self.openings, "rooms": self.rooms,
            "fixtures": self.fixtures, "stair": None, "stairs": self.stairs, "shafts": self.shafts,
            "levels": self.levels, "pilotis": self.pilotis, "building": building,
        }


def _place_door(plan: Plan, rng, wid, center, width, into, kind="door"):
    """into: 문이 열리는 쪽 법선 (dx, dy)."""
    plan.opening(kind, wid, center, width, rng.choice([-1, 1]), into)


def _core(plan: Plan, rng, x0, y0, w, h, floor_level, top: bool, style: dict):
    """코어: 계단실, 승강로(1~2대), 승강장 홀, 남·여 화장실, PD·EPS 샤프트. 층마다 같은 자리다."""
    t_core, t_part = style["core_t"], style["part_t"]
    x1, y1 = x0 + w, y0 + h
    stair_h = 6000
    ev_w, ev_h = 2600 * style["ev_count"], 2600
    # 코어 외곽은 건물 외벽과 겹칠 수 있다. 안쪽 벽만 그린다(외벽은 따로).
    ys = y0 + stair_h
    wall_stair_top = plan.wall((x0, ys), (x1, ys), t_core, RC)
    plan.stairs.append({"x0": x0 + 300, "x1": x1 - 300, "y0": y0 + 900, "tread": 280, "n": 15,
                        "up": not top, "down": floor_level > 1 or style.get("basement", False)})
    plan.rooms.append(("계단실", (x0 + w / 2, y0 + 600)))
    _place_door(plan, rng, wall_stair_top, (x0 + w * 0.75, ys), 1000, (0, -1))
    # 승강로 + 승강장 홀
    ey0, ey1 = ys + 2400, ys + 2400 + ev_h
    wall_ev_front = plan.wall((x0, ey0), (x0 + ev_w, ey0), t_core, RC)
    plan.wall((x0 + ev_w, ey0), (x0 + ev_w, ey1), t_core, RC)
    plan.wall((x0, ey1), (x0 + ev_w, ey1), t_core, RC)
    for k in range(style["ev_count"]):
        cx = x0 + 1300 + k * 2600
        plan.opening("ev_door", wall_ev_front, (cx, ey0), 1100)
        plan.shafts.append({"kind": "EV", "box": (x0 + k * 2600 + 150, ey0 + 150, x0 + (k + 1) * 2600 - 150, ey1 - 150)})
    plan.rooms.append(("EV홀", (x0 + w / 2, ys + 1200)))
    # PD·EPS 샤프트(작은 상자 + 점검문)
    sx0 = x0 + ev_w + 300
    if sx0 + 1600 < x1:
        pd_wall = plan.wall((sx0, ey0), (sx0 + 1600, ey0), t_part, BLOCK)
        plan.wall((sx0 + 800, ey0), (sx0 + 800, ey0 + 1000), t_part, BLOCK)
        plan.wall((sx0, ey0), (sx0, ey0 + 1000), t_part, BLOCK)
        plan.wall((sx0 + 1600, ey0), (sx0 + 1600, ey0 + 1000), t_part, BLOCK)
        plan.wall((sx0, ey0 + 1000), (sx0 + 1600, ey0 + 1000), t_part, BLOCK)
        plan.opening("access_door", pd_wall, (sx0 + 400, ey0), 600, rng.choice([-1, 1]), (0, -1))
        plan.opening("access_door", pd_wall, (sx0 + 1200, ey0), 600, rng.choice([-1, 1]), (0, -1))
        plan.shafts.append({"kind": "PD", "box": (sx0 + 80, ey0 + 80, sx0 + 720, ey0 + 920)})
        plan.shafts.append({"kind": "EPS", "box": (sx0 + 880, ey0 + 80, sx0 + 1520, ey0 + 920)})
    # 화장실(위쪽)
    ty = ey1 + 300
    if y1 - ty > 3000:
        mid = x0 + w / 2
        wall_toilet = plan.wall((x0, ty), (x1, ty), t_part, BLOCK)
        plan.wall((mid, ty), (mid, y1), t_part, BLOCK)
        kind = rng.choice(["door", "sliding"])
        for cx, name in ((x0 + w * 0.25, "남자화장실"), (x0 + w * 0.75, "여자화장실")):
            if kind == "door":
                _place_door(plan, rng, wall_toilet, (cx, ty), 900, (0, 1))
            else:
                plan.opening("sliding", wall_toilet, (cx, ty), 900)
            plan.rooms.append((name, (cx, ty + 1600)))
            plan.fixtures.append(("WC", (cx - 500, y1 - 700), 0))
            plan.fixtures.append(("LAV", (cx + 600, ty + 600), 0))
    return x1, y1


def _grid(rng, kind):
    if kind == "logistics_xl":  # 초대형 물류센터: 110~240m
        sx, sy = rng.choice([11000, 12000]), rng.choice([10000, 11000, 12000])
        nx, ny = rng.randint(10, 20), rng.randint(5, 8)
        return [i * sx for i in range(nx + 1)], [j * sy for j in range(ny + 1)]
    if kind == "residential":  # 복도식 공동주택: 세대 폭 7.2~9m
        sx = rng.choice([7200, 8100, 9000])
        return [i * sx for i in range(rng.randint(5, 9) + 1)], [0, 9000, 10800]
    if kind == "warehouse":
        sx, sy = rng.choice([9000, 10000, 11000]), rng.choice([9000, 10000, 11000])
        nx, ny = rng.randint(5, 8), rng.randint(3, 5)
    else:
        sx, sy = rng.choice([6000, 7200, 8100, 8400]), rng.choice([6000, 7200, 8100])
        nx, ny = rng.randint(4, 7), rng.randint(2, 3)
        while sy * ny < 14000:
            ny += 1
    return [i * sx for i in range(nx + 1)], [j * sy for j in range(ny + 1)]


def building(seed: int, kind: str | None = None) -> dict:
    """건물 하나: 층마다 레이아웃(dict) 목록과 건물 정보."""
    rng = random.Random(seed)
    kind = kind or rng.choice(["office", "office", "public", "public", "warehouse", "residential", "residential", "logistics_xl"])
    base = {"public": "office", "logistics_xl": "warehouse", "residential": "office"}.get(kind, kind)
    gx, gy = _grid(rng, kind)
    variant, kind = kind, base
    floors = rng.randint(2, 4) if kind == "office" else rng.randint(2, 3)
    height = rng.choice([3900, 4200, 4500]) if kind == "office" else rng.choice([7500, 9000, 10000])
    if variant == "residential":
        floors, height = rng.randint(2, 5), rng.choice([2800, 2900])
    b = {"id": f"S{seed:04d}", "kind": variant, "floors": floors, "floor_height": height,
         "name": {"office": "가상 업무 시설", "public": "가상 공공 시설", "warehouse": "가상 물류창고",
                  "logistics_xl": "가상 대형 물류센터", "residential": "가상 공동주택"}[variant]}
    style = {
        "ext_t": 250 if kind == "office" else 150, "ext_mat": RC if kind == "office" else BLOCK,
        "core_t": 200, "part_t": rng.choice([100, 150]), "ev_count": rng.choice([1, 2]),
        "col": 600 if kind == "office" else 800,
    }
    W, H = gx[-1], gy[-1]
    core_w = gx[1] if gx[1] >= 6000 else gx[2]
    pilotis = kind == "office" and variant != "residential" and rng.random() < (0.8 if variant == "public" else 0.6)
    double = kind == "warehouse" or rng.random() < 0.4  # 겹벽 외벽(패널 + 안쪽 벽)
    setback = kind == "office" and floors >= 3 and rng.random() < 0.5
    layouts = []
    for f in range(1, floors + 1):
        top = f == floors
        x_end = W - (gx[-1] - gx[-2]) if (setback and top) else W
        p = Plan(f"F{f}", f"가상 {b['name']} 지상{f}층 평면도", gx, gy, style["col"])
        p.columns = [(x, y) for x in gx for y in gy if x <= x_end]
        level = (f - 1) * height
        p.levels.append((f"FL:{level / 1000:+.2f}".replace("+", ""), (gx[1] + 1500, gy[1] - 1500)))
        # 외벽(필로티 구간과 도크 셔터는 열어 둔다)
        piloti_span = (gx[2], gx[min(4, len(gx) - 2)]) if (pilotis and f == 1) else None
        south = []
        if piloti_span:
            south += [((0, 0), (piloti_span[0], 0)), ((piloti_span[1], 0), (x_end, 0))]
            p.pilotis.append({"p1": [piloti_span[0], 0], "p2": [piloti_span[1], 0]})
            p.rooms.append(("필로티", ((piloti_span[0] + piloti_span[1]) / 2, gy[1] / 2)))
            # 필로티 안쪽 벽: 첫 그리드 줄에 맞춘다
            inner = p.wall((piloti_span[0], gy[1]), (piloti_span[1], gy[1]), style["ext_t"], style["ext_mat"])
            p.wall((piloti_span[0], 0), (piloti_span[0], gy[1]), style["ext_t"], style["ext_mat"])
            p.wall((piloti_span[1], 0), (piloti_span[1], gy[1]), style["ext_t"], style["ext_mat"])
            _place_door(p, rng, inner, ((piloti_span[0] + piloti_span[1]) / 2, gy[1]), 1800, (0, 1), "double_door")
        elif f == 1 and variant != "residential" and rng.random() < (0.7 if variant == "public" else 0.4) and len(gx) > 4:
            # 바깥으로 열린 면(보행통로·광장 쪽): 외벽 한 칸을 비우고 필로티처럼 '트임' 정답으로 기록한다
            a, c = gx[2], gx[3]
            south += [((0, 0), (a, 0)), ((c, 0), (x_end, 0))]
            p.pilotis.append({"p1": [a, 0], "p2": [c, 0]})
            p.levels.append((rng.choice(["출입 광장", "보행통로 (비가림)", "주차장", "차로 6,000"]), ((a + c) / 2, -2500)))
        else:
            south.append(((0, 0), (x_end, 0)))
        ext = [p.wall(a, c, style["ext_t"], style["ext_mat"]) for a, c in south]
        east = p.wall((x_end, 0), (x_end, H), style["ext_t"], style["ext_mat"])
        north = p.wall((x_end, H), (0, H), style["ext_t"], style["ext_mat"])
        west = p.wall((0, H), (0, 0), style["ext_t"], style["ext_mat"])
        core_h = H if kind == "office" else min(H, 16000)  # 창고는 코어가 한쪽 끝에만 있다
        _core(p, rng, 0, 0, core_w, core_h, f, top, style)
        p.wall((core_w, 0), (core_w, core_h), style["core_t"], RC)
        if core_h < H:
            p.wall((0, core_h), (core_w, core_h), style["core_t"], RC)
        core_door_wall = p.walls[-1][0]
        _place_door(p, rng, core_door_wall, (core_w, gy[1] + 1500 if len(gy) > 2 else H / 2), 1000, (1, 0))
        if variant == "residential":
            _residential_rooms(p, rng, core_w, x_end, H, gx, style)
        elif kind == "office":
            _office_rooms(p, rng, core_w, x_end, H, gx, style, f, public=variant == "public")
        else:
            _warehouse_rooms(p, rng, core_w, x_end, H, gx, style, f, ext)
        # 외벽 창
        for wid, axis in ((north, "x"), (east, "y"), (west, "y"), *[(e, "x") for e in ext]):
            w = next(w for w in p.walls if w[0] == wid)
            (ax, ay), (bx, by) = w[1], w[2]
            span = abs(bx - ax) if axis == "x" else abs(by - ay)
            for k in range(1, int(span // 4000)):
                if rng.random() < (0.75 if kind == "office" else 0.25):
                    c = (min(ax, bx) + k * 4000, ay) if axis == "x" else (ax, min(ay, by) + k * 4000)
                    if (axis == "x" and c[0] < core_w + 500) or (wid == west and c[1] < 6500):
                        continue  # 코어(계단·승강로) 앞에는 창을 내지 않는다
                    if any(o[2] == wid and abs(o[3][0] - c[0]) + abs(o[3][1] - c[1]) < 2600 for o in p.openings):
                        continue  # 문·셔터와 겹치지 않게
                    p.opening("window", wid, c, rng.choice([1500, 1800, 2400]))
        if kind == "office" and not piloti_span and f == 1:
            # 정문(양개문)과 출입 홀
            sx = gx[2] + (gx[3] - gx[2]) / 2 if len(gx) > 3 else core_w + 3000
            _place_door(p, rng, ext[0], (sx, 0), 1800, (0, 1), "double_door")
        roughen(p, rng, style, double, ext + [east, north, west], (x_end / 2, H / 2))
        layouts.append(p.layout(b))
    layouts.append(_roof(b, gx, gy, core_w, W if not setback else W, style))
    return {"building": b, "layouts": layouts}


def _office_rooms(p: Plan, rng, core_w, x_end, H, gx, style, f, public=False):
    """복도 양쪽에 실을 나눈다. 회의실은 복도에 실내 유리, 로비는 복도로 넓게 트인다."""
    cy = H / 2
    cw = rng.choice([2000, 2400])
    t = style["part_t"]
    south_wall = p.wall((core_w, cy - cw / 2), (x_end, cy - cw / 2), t, BLOCK)
    north_wall = p.wall((core_w, cy + cw / 2), (x_end, cy + cw / 2), t, BLOCK)
    p.rooms.append(("복도", ((core_w + x_end) / 2, cy)))
    xs = [x for x in gx if core_w < x < x_end]
    bays = [core_w] + xs + [x_end]
    names = ["사무실", "회의실", "자료실", "상담실", "다목적실", "창고", "직원휴게실"]
    if public:
        names = ["민원실", "주민자치위원회실", "다목적홀", "북카페", "작은도서관", "프로그램실", "소회의실", "창고"]
    for side, wall_id, y0, y1, into in (("S", south_wall, 0, cy - cw / 2, (0, -1)), ("N", north_wall, cy + cw / 2, H, (0, 1))):
        # 두 칸씩 묶은 실도 만든다
        i = 0
        while i < len(bays) - 1:
            j = min(i + rng.choice([1, 1, 2]), len(bays) - 1)
            xa, xb = bays[i], bays[j]
            if j < len(bays) - 1:
                p.wall((xb, y0), (xb, y1), t, BLOCK)
            name = rng.choice(names)
            if f == 1 and side == "S" and i == 1:
                # 로비: 복도로 넓게 트인 통로(문짝 없음)
                name = "로비"
                p.opening("open", wall_id, ((xa + xb) / 2, y1 if into == (0, -1) else y0), min(xb - xa - 1200, 3600))
            elif xb - xa > 4000 and rng.random() < (0.4 if public else 0.25):
                # 문짝 없이 트인 통로(1.2~3m). 실제 도면(주민센터)의 틀린 판정 7곳이 전부 이런 '트임'이었다
                p.opening("open", wall_id, (xa + rng.uniform(1500, xb - xa - 1500), y1 if into == (0, -1) else y0),
                          rng.choice([1200, 1600, 2000, 3000]))
            else:
                door_x = xa + rng.uniform(900, max(1000, xb - xa - 900))
                kind = "double_door" if xb - xa > 12000 and rng.random() < 0.5 else "door"
                width = 1800 if kind == "double_door" else rng.choice([900, 1000])
                _place_door(p, rng, wall_id, (door_x, y1 if into == (0, -1) else y0), width, into, kind)
                if name == "회의실" and rng.random() < 0.8:
                    gx_c = xa + (xb - xa) * 0.7 if door_x < (xa + xb) / 2 else xa + (xb - xa) * 0.3
                    if abs(gx_c - door_x) > 1800:
                        p.opening("glass", wall_id, (gx_c, y1 if into == (0, -1) else y0), rng.choice([1800, 2400, 3000]))
            p.rooms.append((name, ((xa + xb) / 2, (y0 + y1) / 2)))
            if name in ("회의실", "다목적실"):
                p.fixtures.append(("TABLE4", ((xa + xb) / 2, (y0 + y1) / 2 - 900), 0))
            i = j


def _residential_rooms(p: Plan, rng, core_w, x_end, H, gx, style):
    """복도식 공동주택: 북쪽 복도(1.8m), 남쪽에 세대 반복. 세대마다 현관문, 침실·욕실 문, 발코니 미닫이 창."""
    t = style["part_t"]
    cy = H - 1800
    corridor = p.wall((core_w, cy), (x_end, cy), 200, RC)
    p.rooms.append(("복도", ((core_w + x_end) / 2, cy + 900)))
    xs = [x for x in gx if core_w < x < x_end]
    for a, c in zip([core_w] + xs, xs + [x_end]):
        if c < x_end:
            p.wall((c, 0), (c, cy), 200, RC)  # 세대 경계벽
        _place_door(p, rng, corridor, (a + 1200, cy), 900, (0, -1))  # 현관문
        bed_y = cy - 3600
        bw = p.wall((a, bed_y), (a + (c - a) * 0.55, bed_y), t, BLOCK)
        p.wall((a + (c - a) * 0.55, bed_y), (a + (c - a) * 0.55, cy), t, BLOCK)
        _place_door(p, rng, bw, (a + (c - a) * 0.3, bed_y), 900, (0, 1))
        p.rooms.append(("침실", (a + (c - a) * 0.27, bed_y + 1800)))
        bath = p.wall((a + (c - a) * 0.55, cy - 2200), (c, cy - 2200), t, BLOCK)
        p.opening("sliding" if rng.random() < 0.5 else "door", bath, (a + (c - a) * 0.75, cy - 2200), 800,
                  *((rng.choice([-1, 1]), (0, 1)) if True else ()))
        p.rooms.append(("욕실", (a + (c - a) * 0.78, cy - 1100)))
        p.rooms.append(("거실", ((a + c) / 2, 2500)))
        bal = p.wall((a, 1500), (c, 1500), t, BLOCK)  # 발코니 창(미닫이 큰 창)
        p.opening("window", bal, ((a + c) / 2, 1500), min(c - a - 1200, 4800))
        p.rooms.append(("발코니", ((a + c) / 2, 750)))


def _warehouse_rooms(p: Plan, rng, core_w, x_end, H, gx, style, f, ext):
    """큰 창고 공간, 1층 남쪽은 도크 셔터, 한쪽 끝은 사무 공간(작은 실)."""
    if f == 1:
        w = next(w for w in p.walls if w[0] == ext[0])
        xs = [x for x in gx if core_w < x < x_end]
        for a, b in zip([core_w] + xs, xs + [x_end]):
            if rng.random() < 0.7:
                p.opening("shutter", ext[0], ((a + b) / 2, 0), rng.choice([3000, 3600, 4500]))
        p.rooms.append(("하역장", ((core_w + x_end) / 2, 3000)))
    # 사무 공간: 동쪽 끝 한 칸
    ox = x_end - (gx[-1] - gx[-2])
    t = style["part_t"]
    ow = p.wall((ox, H - 7000), (ox, H), t, BLOCK)
    oy = p.wall((ox, H - 7000), (x_end, H - 7000), t, BLOCK)
    _place_door(p, rng, oy, (ox + 1500, H - 7000), 1000, (0, 1))
    p.opening("glass", oy, (ox + 5000, H - 7000), 3000)
    p.rooms.append(("사무실", ((ox + x_end) / 2, H - 3500)))
    p.rooms.append(("창고", ((core_w + ox) / 2, H / 2)))
    if f == 1 and len(gx) > 4:
        # 하역장과 창고 사이: 넓게 트인 통로(문짝 없음)가 있는 칸막이
        dy = 7000
        wall = p.wall((core_w, dy), (ox, dy), style["part_t"], BLOCK)
        xs = [x for x in gx if core_w < x < ox]
        for a, c in zip([core_w] + xs, xs + [ox]):
            if c - a > 6000 and rng.random() < 0.6:
                p.opening("open", wall, ((a + c) / 2, dy), min(c - a - 1500, 5000))


def _roof(b, gx, gy, core_w, W, style) -> dict:
    """지붕층: 둘레 파라펫과 계단탑. 바닥은 지붕이다."""
    p = Plan("R", f"가상 {b['name']} 지붕층 평면도", gx, gy, 0)
    t = 200
    for a, c in (((0, 0), (W, 0)), ((W, 0), (W, gy[-1])), ((W, gy[-1]), (0, gy[-1])), ((0, gy[-1]), (0, 0))):
        p.wall(a, c, t, RC)
    p.wall((0, 6000 + 300), (core_w, 6000 + 300), 200, RC)
    p.wall((core_w, 0), (core_w, 6000 + 300), 200, RC)
    d = p.walls[-1][0]
    p.opening("door", d, (core_w, 3000), 1000, 1, (1, 0))
    p.stairs.append({"x0": 300, "x1": core_w - 300, "y0": 900, "tread": 280, "n": 15, "up": False, "down": True})
    p.rooms.append(("계단탑", (core_w / 2, 600)))
    p.rooms.append(("지붕", ((core_w + W) / 2, gy[-1] / 2)))
    return p.layout(b)


def _inward(p1, p2, center):
    """벽 p1→p2의 법선 중 건물 중심 쪽."""
    (ax, ay), (bx, by) = p1, p2
    L = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 or 1.0
    n = (-(by - ay) / L, (bx - ax) / L)
    mx, my = (ax + bx) / 2, (ay + by) / 2
    if (center[0] - mx) * n[0] + (center[1] - my) * n[1] < 0:
        n = (-n[0], -n[1])
    return n


def roughen(p: Plan, rng, style, double: bool, ext_ids: list, center):
    """실무 도면의 지저분함: 겹벽 외벽, 기둥 옆에서 짧게 끊긴 칸막이, 모서리에서 비켜 끝난 외벽.

    정답은 그대로다(끊긴 곳도 벽이 이어져야 하는 곳, 겹벽 사이도 벽). 검토 지점의 '벽' 예시를 만든다.
    """
    walls = {w[0]: w for w in p.walls}
    if double:
        gap = rng.choice([150, 200, 250, 300])
        for wid in ext_ids:
            w = walls.get(wid)
            if w is None:
                continue
            n = _inward(w[1], w[2], center)
            d = w[3] / 2 + gap + 100
            q1 = (w[1][0] + n[0] * d, w[1][1] + n[1] * d)
            q2 = (w[2][0] + n[0] * d, w[2][1] + n[1] * d)
            inner = p.wall(q1, q2, 200, RC)
            for o in [o for o in p.openings if o[2] == wid]:
                c = (o[3][0] + n[0] * d, o[3][1] + n[1] * d)
                if o[1] in ("door", "double_door", "access_door") and len(o) > 5:
                    p.opening(o[1], inner, c, o[4], o[5], o[6])
                else:
                    p.opening(o[1], inner, c, o[4])
    # 칸막이 끝을 기둥 옆에서 짧게(15~50cm), 외벽 끝을 모서리에서 비켜(30~70cm)
    used = {o[2] for o in p.openings}
    out = []
    for w in p.walls:
        wid, a, c, th, mat = w
        L = ((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2) ** 0.5
        cut = 0.0
        if mat == BLOCK and wid not in ext_ids and rng.random() < 0.35:
            cut = rng.uniform(150, 500)
        elif wid in ext_ids and rng.random() < 0.25:
            cut = rng.uniform(300, 700)
        if cut and L > 3 * cut + 2000:
            u = ((c[0] - a[0]) / L, (c[1] - a[1]) / L)
            if rng.random() < 0.5:
                a = (a[0] + u[0] * cut, a[1] + u[1] * cut)
                shift = cut
            else:
                c = (c[0] - u[0] * cut, c[1] - u[1] * cut)
                shift = 0.0
            if wid in used and shift:  # 문 위치가 벽 시작점 기준이 아니라 좌표라 그대로 둬도 된다
                pass
        out.append((wid, a, c, th, mat))
    p.walls[:] = out
