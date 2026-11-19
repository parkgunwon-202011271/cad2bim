"""학습 예시로 레이어 역할 채우기: 사전이 모르는 레이어만, 형상 역할은 만장일치만, 기본은 꺼짐."""
from __future__ import annotations

import json

from cad2bim.layers import LayerInfo, LayerMap
from cad2bim.layers.learned import LayerExamples, apply_learned_roles

EXAMPLES = [
    {"name": "A-ANNO-NOTE", "role": "OTHER", "kinds": {"text": 9}},
    {"name": "A-ANNO-SYMB", "role": "OTHER", "kinds": {"polyline": 4, "hatch": 4}},
    {"name": "A-ANNO-TTLB", "role": "OTHER", "kinds": {"text": 6, "line": 3}},
    {"name": "A-FLOR-STRS", "role": "STAIR", "kinds": {"line": 136}},
    {"name": "A-STAIR", "role": "STAIR", "kinds": {"line": 40}},
    {"name": "A-ANNO-STAF", "role": "OTHER", "kinds": {"polyline": 1, "hatch": 1}},
]


def write(tmp_path, rows):
    (tmp_path / "ex.json").write_text(json.dumps({"layers": rows}), encoding="utf-8")
    return {"learned_layers": {"enabled": True, "examples": ["ex.json"], "k": 3, "min_confidence": 0.6}}


def test_unknown_annotation_layer_becomes_other_and_known_layers_stay(tmp_path):
    cfg = write(tmp_path, EXAMPLES)
    layers = LayerMap({
        "A-ANNO-CIRC": LayerInfo("UNKNOWN", "unresolved", stats={"kinds": {"polyline": 10, "hatch": 10}}),
        "A-WALL": LayerInfo("WALL", "dictionary", stats={"kinds": {"line": 50}}),
    })
    assert apply_learned_roles(layers, cfg, tmp_path) == 1
    assert layers.role("A-ANNO-CIRC") == "OTHER" and layers.layers["A-ANNO-CIRC"].source == "learned"
    assert layers.role("A-WALL") == "WALL"


def test_shaped_role_needs_every_neighbour_to_agree(tmp_path):
    """직원 동선 표시가 STAIR와 글자가 비슷해 계단이 될 뻔했다. 형상 역할은 만장일치일 때만."""
    cfg = write(tmp_path, EXAMPLES)
    model = LayerExamples(EXAMPLES)
    role, confidence = model.predict("A-ANNO-CIRC-STAF", {"kinds": {"line": 20}})
    layers = LayerMap({"A-ANNO-CIRC-STAF": LayerInfo("UNKNOWN", "unresolved", stats={"kinds": {"line": 20}})})
    apply_learned_roles(layers, cfg, tmp_path)
    if role == "STAIR" and confidence < 1.0:
        assert layers.role("A-ANNO-CIRC-STAF") == "UNKNOWN"


def test_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CAD2BIM_LEARNED_LAYERS", raising=False)
    cfg = write(tmp_path, EXAMPLES)
    cfg["learned_layers"]["enabled"] = False
    layers = LayerMap({"A-ANNO-CIRC": LayerInfo("UNKNOWN", "unresolved", stats={"kinds": {"polyline": 1}})})
    assert apply_learned_roles(layers, cfg, tmp_path) == 0 and layers.role("A-ANNO-CIRC") == "UNKNOWN"
