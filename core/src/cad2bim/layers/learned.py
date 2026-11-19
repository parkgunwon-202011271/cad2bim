"""학습한 예시로 레이어 역할 채우기 (처리 단계 3-a의 학습 모델). 사전이 '모름'으로 둔 레이어만 본다.

사전은 벽·계단·창처럼 중요한 이름을 잘 맞히고, 모르는 레이어는 대부분 주기·표기·대지라 '기타'다.
그래서 사전이 먼저고, 사전이 모를 때만 예시와 가장 비슷한 레이어의 역할을 쓴다(2026-11-19 주민센터 39개:
사전만 24, 사전+예시 36). 비용이 들지 않고 외부로 보내는 것이 없다.

모델: 레이어 이름 글자 조각(2~4글자, TF-IDF)과 도형 통계(종류별 비율, 개수, 평균 선 길이)로
가까운 예시 k개를 찾아 다수결. 확신도(같은 답 비율)가 기준 미만이거나 도형 종류와 어긋나면 쓰지 않는다.

예시: 공개 도면에서 만든 것은 config/layer_examples.json(저장소). 비공개 도면에서 만든 것은 저장소 밖
파일에 두고 환경 변수 CAD2BIM_LAYER_EXAMPLES(여러 개는 ;로 구분)로 알려 준다. 레이어 이름도 도면 정보다.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from . import LayerMap, base_name
from .llm import role_fits

SOURCE = "learned"
_SHAPED = frozenset({"WALL", "COLUMN", "DOOR", "WINDOW", "STAIR", "RAIL"})
_KINDS = ("line", "polyline", "arc", "circle", "text", "hatch", "dimension", "block_ref", "ellipse", "solid")


def _grams(name: str) -> Counter:
    text = " " + re.sub(r"[_\-$|.]+", " ", name.lower()).strip() + " "
    return Counter(text[i:i + n] for n in (2, 3, 4) for i in range(len(text) - n + 1))


def _stats(stats: dict[str, Any]) -> list[float]:
    kinds = stats.get("kinds") or {}
    total = max(sum(kinds.values()), 1)
    mean = stats.get("mean_line_mm") or 0.0
    return [kinds.get(k, 0) / total for k in _KINDS] + [math.log1p(total) / 10.0, math.log1p(mean) / 10.0]


class LayerExamples:
    """예시 레이어(이름, 통계, 역할)로 가장 비슷한 것을 찾는다."""

    def __init__(self, examples: list[dict[str, Any]], k: int = 3):
        self.examples = [e for e in examples if e.get("role") and e["role"] != "UNKNOWN"]
        self.k = k
        docs = [_grams(e["name"]) for e in self.examples]
        df = Counter(g for d in docs for g in d)
        n = max(len(docs), 1)
        self.idf = {g: math.log((1 + n) / (1 + c)) + 1.0 for g, c in df.items()}
        self.vectors = [self._vector(d) for d in docs]
        self.stats = [_stats(e) for e in self.examples]

    def _vector(self, grams: Counter) -> dict[str, float]:
        v = {g: c * self.idf.get(g, 0.0) for g, c in grams.items() if g in self.idf}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {g: x / norm for g, x in v.items()}

    def predict(self, name: str, stats: dict[str, Any]) -> tuple[str, float] | None:
        """(역할, 확신도). 예시가 없으면 None."""
        if not self.examples:
            return None
        q = self._vector(_grams(name))
        s = _stats(stats)
        scored = []
        for vec, st, ex in zip(self.vectors, self.stats, self.examples):
            text = sum(w * vec.get(g, 0.0) for g, w in q.items())
            shape = 1.0 - min(1.0, math.dist(s, st) / 2.0)
            scored.append((0.7 * text + 0.3 * shape, ex["role"]))
        top = sorted(scored, reverse=True)[: self.k]
        votes = Counter(role for _, role in top)
        role, count = votes.most_common(1)[0]
        return role, count / len(top)


def load_examples(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if Path(path).is_file():
            rows += json.loads(Path(path).read_text(encoding="utf-8")).get("layers", [])
    return rows


def example_paths(cfg: dict[str, Any], config_dir: Path) -> list[Path]:
    learned = cfg.get("learned_layers") or {}
    # CAD2BIM_LAYER_EXAMPLES_ONLY=1이면 환경 변수로 준 예시만 쓴다(같은 도면의 예시를 빼고 공정하게 재기 위해)
    only = os.environ.get("CAD2BIM_LAYER_EXAMPLES_ONLY", "").strip() in ("1", "on", "true")
    paths = [] if only else [config_dir / p for p in learned.get("examples", [])]
    paths += [Path(p) for p in os.environ.get("CAD2BIM_LAYER_EXAMPLES", "").split(";") if p.strip()]
    return paths


def apply_learned_roles(layer_map: LayerMap, cfg: dict[str, Any], config_dir: Path, skip: set[str] | None = None) -> int:
    """사전이 '모름'으로 둔 레이어에 예시의 역할을 넣는다. 바꾼 수. 꺼져 있으면 0."""
    learned = cfg.get("learned_layers") or {}
    enabled = os.environ.get("CAD2BIM_LEARNED_LAYERS", "").strip().lower()
    if not (enabled in ("1", "on", "true") or (not enabled and learned.get("enabled"))):
        return 0
    model = LayerExamples(load_examples(example_paths(cfg, config_dir)), int(learned.get("k", 3)))
    threshold = float(learned.get("min_confidence", 0.6))
    changed = 0
    for name, info in layer_map.layers.items():
        if info.role != "UNKNOWN" or info.source not in ("unresolved",) or name in (skip or set()):
            continue  # 모양으로 벽 후보가 된 레이어(skip)는 건드리지 않는다
        found = model.predict(base_name(name), info.stats or {})
        if found is None:
            continue
        role, confidence = found
        # 형상이 생기는 역할은 틀리면 3D에 가짜 벽·계단이 생긴다. 가까운 예시가 모두 같을 때만 쓴다
        # (주민센터: 직원 동선 표시 'A-ANNO-CIRC-STAF'가 STAIR와 글자가 비슷해 계단이 될 뻔했다).
        need = 1.0 if role in _SHAPED else threshold
        if confidence < need or role == "UNKNOWN" or not role_fits(role, info.stats or {}):
            continue
        info.role, info.source = role, SOURCE
        changed += 1
    return changed


def export_examples(layer_map_paths: list[Path], target: Path, roles_from: str = "final") -> int:
    """만든 결과의 layer_map.json들에서 예시를 뽑아 저장한다(이름·통계·역할). '모름'은 뺀다."""
    rows: dict[str, dict[str, Any]] = {}
    for path in layer_map_paths:
        for name, info in json.loads(Path(path).read_text(encoding="utf-8"))["layers"].items():
            if info["role"] == "UNKNOWN":
                continue
            short = base_name(name)
            row = rows.setdefault(short, {"name": short, "role": info["role"], "kinds": Counter(), "mean_line_mm": None})
            row["kinds"].update((info.get("stats") or {}).get("kinds") or {})
            row["mean_line_mm"] = row["mean_line_mm"] or (info.get("stats") or {}).get("mean_line_mm")
    out = [{**r, "kinds": dict(r["kinds"])} for r in rows.values()]
    Path(target).write_text(json.dumps({"source": roles_from, "layers": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    return len(out)
