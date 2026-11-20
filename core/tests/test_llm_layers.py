"""LLM 레이어 역할(3-a). 네트워크 없이 저장된 답으로 시험한다."""
from __future__ import annotations

import json

import pytest

from cad2bim.errors import Cad2BimError
from cad2bim.layers import LayerInfo, LayerMap, drop_weak_walls
from cad2bim.layers import llm
from cad2bim.preprocess.primitive import Primitive


def info(role, source="dictionary", **stats):
    return LayerInfo(role, source, None, {"count": 1, "kinds": {"line": 1}, "mean_line_mm": 100.0, "blocks": [], **stats})


def line(layer, length):
    return Primitive(id=f"{layer}{length}", kind="line", geometry={"p1": (0, 0), "p2": (length, 0)}, layer=layer, source_handles=("1",))


def cache(tmp_path, roles):
    (tmp_path / llm.CACHE_NAME).write_text(json.dumps({"models": ["m"], "roles": roles}), encoding="utf-8")


def test_off_by_default_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("CAD2BIM_LLM", raising=False)
    layers = LayerMap({"X$0$SGPN": info("UNKNOWN", "unresolved")})
    cache(tmp_path, {"SGPN": "WALL"})
    assert llm.apply_llm_roles(layers, tmp_path, {}) == 0
    assert layers.layers["X$0$SGPN"].role == "UNKNOWN"


def test_cache_mode_applies_stored_answers_by_base_name(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD2BIM_LLM", "cache")
    layers = LayerMap({
        "XR$0$AA-SGPN": info("UNKNOWN", "unresolved"),
        "$1$XR$0$AA-SGPN": info("UNKNOWN", "unresolved"),
        "토목옹벽-f": info("WALL"),  # 사전은 '벽'으로 벽이라 했다
        "A-NEW": info("UNKNOWN", "unresolved"),  # 답이 없으면 그대로
    })
    cache(tmp_path, {"AA-SGPN": "WALL", "토목옹벽-f": "OTHER"})
    assert llm.apply_llm_roles(layers, tmp_path, {}) == 3
    assert [layers.layers[n].role for n in ("XR$0$AA-SGPN", "$1$XR$0$AA-SGPN", "토목옹벽-f", "A-NEW")] == ["WALL", "WALL", "OTHER", "UNKNOWN"]
    assert layers.layers["XR$0$AA-SGPN"].source == "llm"


def test_evidence_merges_xref_copies_and_has_no_coordinates():
    layers = LayerMap({
        "F$0$A-WALL": info("WALL", count=2, kinds={"line": 2}, mean_line_mm=1000.0, blocks=["F$0$마감"]),
        "A-WALL": info("WALL", count=1, kinds={"line": 1}, mean_line_mm=4000.0),
    })
    ev = llm.layer_evidence(layers)
    assert list(ev) == ["A-WALL"]
    assert ev["A-WALL"] == {"count": 3, "kinds": {"line": 3}, "mean_line_mm": 2000, "blocks": ["마감"]}


def test_parse_keeps_only_asked_names_and_known_roles():
    text = '설명 {"A-WALL": "wall", "A-X": "SPACESHIP", "B": "OTHER"} 끝'
    assert llm.parse_roles(text, {"A-WALL", "A-X"}) == {"A-WALL": "WALL"}
    with pytest.raises(Cad2BimError):
        llm.parse_roles("없음", {"A"})


def test_save_merges_with_existing_answers(tmp_path):
    llm.save_cache(tmp_path, {"A": "WALL"}, "m1")
    llm.save_cache(tmp_path, {"B": "DOOR"}, "m2")
    data = json.loads((tmp_path / llm.CACHE_NAME).read_text(encoding="utf-8"))
    assert data == {"models": ["m1", "m2"], "roles": {"A": "WALL", "B": "DOOR"}}


def test_bad_mode_is_an_error(monkeypatch):
    monkeypatch.setenv("CAD2BIM_LLM", "gpt")
    with pytest.raises(Cad2BimError):
        llm.mode({})


def test_weak_wall_check_trusts_llm_walls():
    """선이 적은 이름뿐인 벽은 버리지만, LLM이 통계를 보고 정한 벽은 남긴다."""
    prims = [line("A-WALL", 100.0), line("SITE", 900000.0)]
    named = LayerMap({"A-WALL": info("WALL"), "SITE": info("UNKNOWN", "unresolved")})
    assert drop_weak_walls(named, prims, 0.01, 5000.0) == {"A-WALL"}
    trusted = LayerMap({"A-WALL": info("WALL", "llm"), "SITE": info("UNKNOWN", "unresolved")})
    assert drop_weak_walls(trusted, prims, 0.01, 5000.0) == set()
    assert trusted.layers["A-WALL"].role == "WALL"


def test_drawing_specific_answer_wins_over_layer_name_answer(tmp_path):
    """레이어 '0'은 도면마다 쓰임이 다르다. 지붕층 평면에서만 벽(파라펫)이다."""
    from cad2bim.layers import LayerInfo, LayerMap

    (tmp_path / llm.CACHE_NAME).write_text(json.dumps({"models": ["t"], "roles": {"0": "OTHER", "abcd:0": "WALL"}}), encoding="utf-8")
    roof = LayerMap({"0": LayerInfo("UNKNOWN", "unresolved")})
    other = LayerMap({"0": LayerInfo("UNKNOWN", "unresolved")})
    llm.apply_llm_roles(roof, tmp_path, {"llm": {"enabled": True}}, "abcd")
    llm.apply_llm_roles(other, tmp_path, {"llm": {"enabled": True}}, "ffff")
    assert roof.role("0") == "WALL" and other.role("0") == "OTHER"


def test_gemini_is_a_provider_and_needs_a_key(monkeypatch):
    from cad2bim import llm_client

    monkeypatch.setenv("CAD2BIM_LLM", "gemini")
    assert llm.mode({}) == "gemini"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(Cad2BimError):
        llm_client.ask("gemini", "m", "q")
    with pytest.raises(Cad2BimError):
        llm_client.ask("nobody", "m", "q")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)  # 개발 PC의 .env에 모델이 적혀 있어도 설정값을 확인한다
    assert llm_client.default_model({"llm": {"models": {"gemini": "g"}}}, "gemini") == "g"
    assert llm_client.extract_json('설명 ```json\n{"answer": "wall"}\n```') == {"answer": "wall"}


def test_env_file_fills_missing_keys_only(tmp_path, monkeypatch):
    """회사 키는 .env에만 둔다. 이미 설정된 환경 변수가 먼저다."""
    import os

    from cad2bim.envfile import load_env

    env = tmp_path / ".env"
    env.write_text('# 주석\nOPENAI_API_KEY="sk-test"\nOPENAI_BASE_URL=\nOPENAI_MODEL=gpt-x\nCAD2BIM_LLM=openai\n', encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "already-set")
    monkeypatch.delenv("CAD2BIM_LLM", raising=False)
    try:
        loaded = load_env(env)
        assert sorted(loaded) == ["CAD2BIM_LLM", "OPENAI_API_KEY"]  # 빈 값은 넣지 않고, 이미 있는 값은 덮지 않는다
        assert os.environ["OPENAI_API_KEY"] == "sk-test" and os.environ["OPENAI_MODEL"] == "already-set"
    finally:  # load_env가 넣은 값은 monkeypatch가 되돌리지 않는다. 남으면 뒤 테스트가 LLM을 부른다
        for name in ("CAD2BIM_LLM", "OPENAI_API_KEY"):
            os.environ.pop(name, None)


def test_openai_provider_reads_model_from_env_and_needs_a_key(monkeypatch):
    from cad2bim import llm_client

    monkeypatch.setenv("CAD2BIM_LLM", "openai")
    assert llm.mode({}) == "openai"
    monkeypatch.setenv("OPENAI_MODEL", "company-model")
    assert llm_client.default_model({}, "openai") == "company-model"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(Cad2BimError):
        llm_client.ask("openai", "company-model", "q")


def test_llm_role_that_contradicts_the_layer_shapes_is_ignored(tmp_path):
    """작은 모델이 계단 선 레이어를 '실 이름 글자'로 바꿔 계단이 사라졌다. 도형 종류와 어긋난 답은 쓰지 않는다."""
    from cad2bim.layers import LayerInfo, LayerMap

    (tmp_path / llm.CACHE_NAME).write_text(json.dumps({"models": ["t"], "roles": {"A-FLOR-STRS": "ROOM_TEXT", "A-NOTE": "ROOM_TEXT"}}), encoding="utf-8")
    layers = LayerMap({
        "A-FLOR-STRS": LayerInfo("STAIR", "dictionary", stats={"kinds": {"line": 33}}),
        "A-NOTE": LayerInfo("UNKNOWN", "unresolved", stats={"kinds": {"text": 9, "line": 1}}),
    })
    llm.apply_llm_roles(layers, tmp_path, {"llm": {"enabled": True, "provider": "cache"}})
    assert layers.role("A-FLOR-STRS") == "STAIR" and layers.role("A-NOTE") == "ROOM_TEXT"
