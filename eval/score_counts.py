"""Doors/windows after the full build: rules only vs rules + ML (model 1), on the 30 held-out synthetic plans.

사용: python eval/score_counts.py <빌드 폴더(rules/, ml/ 아래 S50xx)> <합성 시험 폴더(synth2_bench.yaml)>
Counts (AECV-Bench style: exact match, MAPE) and location F1 (predicted centre within max(600 mm, width/2) of a
same-class truth centre, one-to-one). Openings that only cut a hole (void_only) are not counted.
"""
import glob
import json
import math
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
root = sys.argv[1]
import yaml
_syn = sys.argv[2]
_man = yaml.safe_load(open(os.path.join(_syn, "synth2_bench.yaml"), encoding="utf-8"))
KEEP = set()  # 발표에 쓴 평면 30장(매니페스트)만 센다
for _it in next(v for v in _man.values() if isinstance(v, list)):
    _t = json.load(open(os.path.join(_syn, _it["truth"]), encoding="utf-8"))
    KEEP.add(f"{_it['dxf'].split('/')[0]}_{_t['floor']}")


def built_openings(ir):
    ox, oy = ir.get("origin_offset", [0.0, 0.0])
    walls = {w["id"]: w for w in ir["walls"]}
    out = []
    for o in ir["openings"]:
        if o.get("void_only") or o["type"] not in ("door", "window"):
            continue
        w = walls.get(o.get("host_wall"))
        if not w:
            continue
        (x1, y1), (x2, y2) = w["start"], w["end"]
        L = math.hypot(x2 - x1, y2 - y1) or 1.0
        t = o["position"] / L
        out.append({"type": o["type"], "x": x1 + (x2 - x1) * t + ox, "y": y1 + (y2 - y1) * t + oy, "width": o["width"]})
    return out


def plans(variant):
    rows = []
    for b in sorted(glob.glob(os.path.join(root, variant, "S50*"))):
        out = os.path.join(b, "_cad2bim_out")
        if not os.path.isfile(os.path.join(out, "building.json")):
            continue
        truths = {json.load(open(t, encoding="utf-8"))["floor"]: t for t in glob.glob(os.path.join(b, "*.bench.json"))}
        for s in json.load(open(os.path.join(out, "building.json"), encoding="utf-8")):
            path = os.path.join(out, s["part"], "ir_built.json")
            if s["floor"] not in truths or not os.path.isfile(path):
                continue
            truth = json.load(open(truths[s["floor"]], encoding="utf-8"))
            if f"{os.path.basename(b)}_{s['floor']}" not in KEEP:
                continue
            rows.append({"id": f"{os.path.basename(b)}_{s['floor']}", "pred": built_openings(json.load(open(path, encoding="utf-8"))),
                         "truth": [{"type": o["label"], "x": o["center"][0], "y": o["center"][1], "width": float(o["width"])}
                                   for o in truth["openings"] if o["label"] in ("door", "window")]})
    return rows


def f1(rows, cls):
    tp = fp = fn = 0
    for r in rows:
        P = [p for p in r["pred"] if p["type"] == cls]
        T = [t for t in r["truth"] if t["type"] == cls]
        pairs = sorted((math.hypot(p["x"] - t["x"], p["y"] - t["y"]), i, j) for i, p in enumerate(P) for j, t in enumerate(T)
                       if math.hypot(p["x"] - t["x"], p["y"] - t["y"]) <= max(600.0, t["width"] / 2))
        up, ut = set(), set()
        for _, i, j in pairs:
            if i not in up and j not in ut:
                up.add(i)
                ut.add(j)
        tp += len(up)
        fp += len(P) - len(up)
        fn += len(T) - len(up)
    return 2 * tp / max(2 * tp + fp + fn, 1), tp, fp, fn


def counts(rows, cls):
    exact = ape = 0
    for r in rows:
        p = sum(o["type"] == cls for o in r["pred"])
        t = sum(o["type"] == cls for o in r["truth"])
        exact += p == t
        ape += abs(p - t) / max(t, 1)
    return exact / len(rows), 100 * ape / len(rows)


result = {}
for v in ("rules", "ml"):
    rows = plans(v)
    result[v] = {"plans": len(rows)}
    for cls in ("door", "window"):
        F, tp, fp, fn = f1(rows, cls)
        ex, mape = counts(rows, cls)
        result[v][cls] = {"f1": round(F, 3), "tp": tp, "fp": fp, "fn": fn, "exact": round(ex, 3), "mape": round(mape, 1)}
    print(v, json.dumps(result[v], ensure_ascii=False))
json.dump(result, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "counts_ours.json"), "w", encoding="utf-8"), indent=1)
