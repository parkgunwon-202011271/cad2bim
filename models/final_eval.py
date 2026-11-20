"""Compare spot models: current (synthetic100+logistics) vs + new synthetic with open passages. Usage: final_eval.py <S>"""
import ast, glob, json, sys, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from cad2bim.review_ml import features, LABELS
sys.stdout.reconfigure(encoding="utf-8")
S = sys.argv[1]
def norm(r):
    r = dict(r)
    for k in ("counts", "sides"):
        if isinstance(r.get(k), str): r[k] = ast.literal_eval(r[k])
    return r
def load(p): return [norm(r) for f in sorted(glob.glob(p)) for r in json.load(open(f, encoding="utf-8")) if r.get("label") in LABELS]
X = lambda rows: np.array([features(r["kind"], float(r["width"]), float(r["thickness"]), r) for r in rows], float)
s2, s3, s4, lg = load(f"{S}/ml/feat2_S*.json"), load(f"{S}/ml/feat3_S*.json"), load(f"{S}/ml/feat4_S*.json"), load(f"{S}/ml/logistics.json")
meta = json.load(open(f"{S}/eval_jumin.json", encoding="utf-8"))
ev = json.load(open("C:/Users/pw710/OneDrive/Desktop/새 폴더 (2)/data/public/주민센터/_cad2bim_out/_llm/spots/evidence.json", encoding="utf-8"))
mine = "open door door door open open open door open open door open open open open open open window window window door wall wall wall".split()
jumin = [{"kind": x["kind"], "width": x["width"], "thickness": 200.0, **(ev.get(x["key"]) or {}), "label": l} for x, l in zip(meta, mine)]
def fit(rows): return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=0).fit(X(rows), [r["label"] for r in rows])
def sc(m, rows):
    y = [r["label"] for r in rows]; p = m.predict(X(rows)); labs = sorted(set(y))
    return {"acc": round(float(np.mean(np.array(y) == p)), 3), "macroF1": round(f1_score(y, p, labels=labs, average="macro", zero_division=0), 3), "n": len(y)}
print("spots: synth100", len(s2), "expanded", len(s4), "new synth", len(s3), dict(__import__("collections").Counter(r["label"] for r in s3)), "logistics", len(lg))
for name, rows in (("A current: synth100+logistics", s2 + lg), ("B synth100+new+logistics", s2 + s3 + lg), ("E synth100+new+expanded+logistics", s2 + s3 + s4 + lg)):
    m = fit(rows); print(name, "| 주민센터", sc(m, jumin))
    if name[0] in "BE": joblib.dump(m, f"C:/Users/pw710/cad2bim_local/models/spot_model_{name[0]}.joblib", compress=3)
for name, rows in (("C synth100 only", s2), ("D synth100+new", s2 + s3), ("F synth100+new+expanded", s2 + s3 + s4)):
    m = fit(rows); print(name, "| 주민센터", sc(m, jumin), "| logistics (held out)", sc(m, lg))
