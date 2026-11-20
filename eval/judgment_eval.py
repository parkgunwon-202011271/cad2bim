"""애매한 지점 판정 평가: 우리 ML(models/spot_model.joblib) 대 LLM, F1.

사용: python eval/judgment_eval.py
A. 합성: feat2(시드 1000~1099)로 학습한 모델을 다른 합성 30동(feat_S*)에서 평가 (규칙 대용 기준선 포함)
B. 주민센터 24곳: 우리 모델과 LLM 답(eval/results/llm_answers_jumin.txt)을 검토자 정답과 비교
   근거는 평가 당시 LLM에 준 것을 eval/jumin_evidence.json 에 저장해 두었다(새로 인식하면 지점이 조금 달라진다).
"""
import ast, glob, json, re, sys
from pathlib import Path
import joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from cad2bim.review_ml import features, LABELS

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).resolve().parent.parent
def norm(r):
    r = dict(r)
    for k in ("counts", "sides"):
        if isinstance(r.get(k), str): r[k] = ast.literal_eval(r[k])
    return r
def load(p): return [norm(r) for f in sorted(glob.glob(str(p))) for r in json.load(open(f, encoding="utf-8")) if r.get("label") in LABELS]
def X(rows): return np.array([features(r["kind"], float(r["width"]), float(r["thickness"]), r) for r in rows], float)
def rule(r):  # 학습 없는 기준선: 근거만 보고 정한다
    c = r["counts"]
    if c.get("ARC") or c.get("DOOR"): return "door"
    if c.get("WINDOW"): return "window"
    return "open" if r["kind"] in ("opening", "outline") and not c.get("WALL") else "wall"
def report(name, y, p):
    labs = [l for l in LABELS if l in set(y)]
    per = dict(zip(LABELS, f1_score(y, p, labels=LABELS, average=None, zero_division=0).round(3)))
    out = {"accuracy": round(float(np.mean(np.array(y) == np.array(p))), 3),
           "macro_f1": round(f1_score(y, p, labels=labs, average="macro", zero_division=0), 3), "per_class": {k: float(v) for k, v in per.items()}}
    print(f"  {name:28s} {out}"); return out
T = R / "models" / "training"
train, test = load(T / "feat2_S*.json"), load(T / "feat_S*.json")
yt = [r["label"] for r in test]
syn = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=0).fit(X(train), [r["label"] for r in train])
print("A. 합성 시험 지점", len(test))
res = {"synthetic": {"rule": report("규칙(기준선)", yt, [rule(r) for r in test]), "ml": report("우리 ML", yt, syn.predict(X(test)))}}
meta = json.load(open(R / "eval" / "jumin_spots.json", encoding="utf-8"))
ev = json.load(open(R / "eval" / "jumin_evidence.json", encoding="utf-8"))  # LLM에 준 것과 같은 근거(평가 당시 저장)
truth = "open door door door open open open door open open door open open open open open open window window window door wall wall wall".split()
jumin = [{"kind": x["kind"], "width": x["width"], "thickness": 200.0, **(ev.get(x["key"]) or {"counts": {}})} for x in meta]
model = joblib.load(R / "models" / "spot_model.joblib")
print("B. 주민센터 24곳")
res["jumin"] = {"ours": report("우리 (models/spot_model.joblib)", truth, model.predict(X(jumin)))}
llm, name = {}, None
for line in open(R / "eval" / "results" / "llm_answers_jumin.txt", encoding="utf-8"):
    m = re.match(r"== (\S+):", line)
    if m: name = m.group(1); llm[name] = []
    m = re.match(r"#\d+ +\S+ +mine=\S+ +(\S+)", line)
    if m and name: llm[name].append(m.group(1))
for n, ans in llm.items():
    res["jumin"][n] = report(n, truth, [a if a in LABELS else "none" for a in ans])
json.dump(res, open(R / "eval" / "results" / "judgment.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
