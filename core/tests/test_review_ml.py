"""ML 판정(CAD2BIM_LLM=ml). 실제 모델 파일 없이 가짜 모델로 시험한다."""
from __future__ import annotations

import numpy as np
import pytest

from cad2bim import review_ml
from cad2bim.layers import llm
from cad2bim.review import Spot


class Fake:
    classes_ = np.array(["door", "open", "wall", "window"])

    def __init__(self, rows):
        self.rows = rows

    def predict_proba(self, X):
        assert len(X[0]) == len(review_ml.features("opening", 900.0, 200.0, {}))
        return np.array(self.rows[: len(X)])


def spot(key, kind="opening"):
    return Spot(key=key, kind=kind, floor="F1", part_id="D01", opening_id="U-1", center=(0.0, 0.0),
                direction=(1.0, 0.0), width=900.0, thickness=200.0)


def test_ml_is_a_valid_mode(monkeypatch):
    monkeypatch.setenv("CAD2BIM_LLM", "ml")
    assert llm.mode({}) == "ml"


def test_without_a_model_file_nothing_is_decided(monkeypatch):
    monkeypatch.delenv("CAD2BIM_SPOT_MODEL", raising=False)
    assert review_ml.decide([spot("a")], {}, {}) == {}


def test_confident_and_consistent_answers_only(monkeypatch, tmp_path):
    import joblib

    monkeypatch.setenv("CAD2BIM_SPOT_MODEL", str(tmp_path / "m.joblib"))
    rows = [[0.9, 0.05, 0.05, 0.0],   # a: 문, 확신 → 씀
            [0.4, 0.3, 0.3, 0.0],     # b: 확신 낮음 → 대기
            [0.0, 0.1, 0.9, 0.0]]     # c: 벽이라 했지만 문 호(ARC)가 있음 → 근거와 어긋나 버림
    monkeypatch.setattr(joblib, "load", lambda path: Fake(rows))
    evidence = {"a": {"counts": {"WALL": 2}}, "b": {"counts": {}}, "c": {"counts": {"ARC": 1, "DOOR": 2}}}
    out = review_ml.decide([spot("a"), spot("b"), spot("c")], evidence, {"review_ml": {"min_prob": 0.5}})
    assert list(out) == ["a"] and out["a"]["answer"] == "door" and out["a"]["model"] == "ml:m.joblib"


def test_features_read_near_text_and_sides():
    ev = {"counts": {"WALL": 3, "ARC": 1}, "sides": ["실 밖(외부 또는 인식 못 한 곳)", "로비"],
          "text": "- 둘레 4m 안 글자(가까운 순): 필로티, 주차장\n"}
    v = review_ml.features("outline", 3000.0, 250.0, ev)
    names = [f"k_{k}" for k in review_ml.KINDS] + ["w", "logw", "t"] + review_ml.ROLES + ["total", "outside", "same"] + review_ml.WORDS
    f = dict(zip(names, v))
    assert f["k_outline"] == 1 and f["WALL"] == 3 and f["ARC"] == 1 and f["total"] == 4 and f["outside"] == 1
    assert f["필로티"] == 1 and f["주차"] == 1 and f["계단"] == 0
    assert f["logw"] == pytest.approx(np.log1p(3000.0))
