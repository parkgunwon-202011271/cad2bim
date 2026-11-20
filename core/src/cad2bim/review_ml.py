"""애매한 지점 판정을 학습 모델로 (처리 단계 6의 ML 판정). LLM 대신 쓴다. 비용 0, 외부로 보내는 것 없음.

켜는 법: CAD2BIM_LLM=ml 과 모델 파일(CAD2BIM_SPOT_MODEL 또는 config의 review_ml.path).
모델: RandomForest. 입력은 LLM에 주던 것과 같은 근거(상자와 겹치는 선의 레이어 역할별 개수, 양쪽 실,
가까운 글자)와 지점 종류·폭·두께. 학습은 정답이 있는 합성 도면(및 사람이 판단한 도면)으로 한다.
LLM과 같은 거르기(consistent)를 거치고, 확신도(min_prob) 미만이면 답하지 않는다(대기 목록에 남는다).
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

ROLES = ["WALL", "DOOR", "WINDOW", "COLUMN", "GRID", "ARC", "HATCH", "STAIR", "RAIL", "UNKNOWN", "OTHER", "DIM", "FURNITURE", "ROOM_TEXT"]
KINDS = ["opening", "outline", "corner", "window"]
WORDS = ["필로티", "주차", "차로", "도로", "광장", "통로", "보행", "홀", "로비", "복도", "EV", "엘리베이터", "계단", "화장실", "PD", "EPS", "TPS",
         "OPEN", "오픈", "VOID", "출입", "셔터", "도크", "회의", "실", "창고", "UP", "DN", "기계", "전기", "PS"]
LABELS = ["wall", "door", "window", "open"]
_NEAR = re.compile(r"글자\(가까운 순\): (.*)")


def features(kind: str, width: float, thickness: float, evidence: dict[str, Any]) -> list[float]:
    """지점 하나의 특징. 학습 스크립트도 이 함수를 쓴다(학습과 사용이 같아야 한다)."""
    counts = evidence.get("counts") or {}
    sides = evidence.get("sides") or []
    outside = sum(s.startswith("실 밖") for s in sides)
    same = int(len(sides) == 2 and sides[0] == sides[1] and not sides[0].startswith("실 밖"))
    m = _NEAR.search(evidence.get("text", ""))
    near = (m.group(1) if m else "").lower()
    v = [float(kind == k) for k in KINDS] + [width, math.log1p(width), thickness]
    v += [counts.get(k, 0) for k in ROLES] + [sum(counts.values()), outside, same]
    v += [float(w.lower() in near) for w in WORDS]
    return v


def model_path(cfg: dict[str, Any]) -> Path | None:
    env = os.environ.get("CAD2BIM_SPOT_MODEL", "").strip()
    if env:
        return Path(env)
    conf = cfg.get("review_ml") or {}
    if not conf.get("path"):
        return None
    path = Path(conf["path"])
    if not path.is_absolute():  # 저장소 기준 경로(config 폴더의 부모)
        from .config import config_dir

        path = config_dir().parent / path
    return path if path.is_file() else None


def decide(spots: list[Any], evidence: dict[str, dict[str, Any]], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """지점마다 모델의 답. 근거와 어긋나는 답(consistent)과 확신이 낮은 답은 버린다."""
    from .review import consistent  # 순환 import를 피한다

    path = model_path(cfg)
    if path is None or not spots:
        return {}
    import joblib  # ML 판정을 켤 때만 필요하다(pip install -e core[ml])

    model = joblib.load(path)
    min_prob = float((cfg.get("review_ml") or {}).get("min_prob", 0.5))
    rows = [(sp, evidence.get(sp.key) or {}) for sp in spots]
    proba = model.predict_proba([features(sp.kind, sp.width, sp.thickness, ev) for sp, ev in rows])
    answers = {}
    for (sp, ev), p in zip(rows, proba):
        best = int(p.argmax())
        answer = str(model.classes_[best])
        if p[best] < min_prob or not consistent(answer, ev.get("counts") or {}, sp.kind):
            continue
        answers[sp.key] = {"answer": answer, "reason": f"ML p={p[best]:.2f}", "model": f"ml:{path.name}"}
    return answers
