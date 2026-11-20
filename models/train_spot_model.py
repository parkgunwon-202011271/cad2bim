"""Train model 1 (spot judgment) with the repo's feature code and save it. Usage: python train_spot_model.py <scratchpad> <out.joblib>"""
import ast, glob, json, sys
import numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from cad2bim.review_ml import features, LABELS
S, out = sys.argv[1], sys.argv[2]
def norm(r):
    r = dict(r)
    for k in ("counts", "sides"):
        if isinstance(r.get(k), str): r[k] = ast.literal_eval(r[k])
    return r
rows = [norm(r) for f in sorted(glob.glob(f"{S}/ml/feat2_S*.json")) for r in json.load(open(f, encoding="utf-8")) if r.get("label") in LABELS]
rows += [norm(r) for r in json.load(open(f"{S}/ml/logistics.json", encoding="utf-8")) if r.get("label") in LABELS]
X = np.array([features(r["kind"], float(r["width"]), float(r["thickness"]), r) for r in rows], float)
y = np.array([r["label"] for r in rows])
# same features as the scratch train.py?
sys.argv = [sys.argv[0], S]; exec(open(f"{S}/ml/train.py", encoding="utf-8").read().split("log = [")[0])
old = np.array([feats({**r, "width": float(r["width"]), "thickness": float(r["thickness"])}) for r in rows[:200]], float)
print("same features as experiments:", np.allclose(old, X[:200]))
m = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=0).fit(X, y)
joblib.dump(m, out, compress=3)
print("trained on", len(y), "spots; saved", out)
