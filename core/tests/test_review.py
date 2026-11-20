"""LLM 검토(6). 의심 지점 찾기, 답 적용, 꺼짐·저장된 답만 쓰기."""
from __future__ import annotations

import json

from cad2bim import review
from cad2bim.building import PlanPart, Storey
from test_building import plan_ir


def storey(openings):
    ir = plan_ir()
    ir["source"] = {"sha256": "abcdef0123456789zz", "unit_scale": 1.0}
    ir["walls"] = [{"id": "W1", "start": [0.0, 0.0], "end": [10000.0, 0.0], "thickness": 200.0, "kind": "unknown"}]
    ir["openings"] = openings
    return Storey("F2", PlanPart("D07", "F2", ir, 1.0, "title"), 0.0, 3000.0)


def gap(oid, pos, kind="unknown"):
    return {"id": oid, "type": kind, "host_wall": "W1", "position": pos, "width": 1200.0}


def test_spots_are_unknown_openings_keyed_by_place():
    spots = review.find_spots([storey([gap("U1", 2000.0), gap("D1", 5000.0, "door")])])
    assert [s.opening_id for s in spots] == ["U1"]
    assert spots[0].key == "abcdef0123456789:20:0:12" and spots[0].center == (2000.0, 0.0)


def test_answers_change_openings_without_touching_the_ir_file_data():
    s = storey([gap("U1", 2000.0), gap("U2", 5000.0), gap("U3", 8000.0)])
    original = s.part.ir
    spots = review.find_spots([s])
    keys = {sp.opening_id: sp.key for sp in spots}
    counts = review.apply_decisions([s], spots, {keys["U1"]: {"answer": "wall"}, keys["U2"]: {"answer": "door"}})
    assert counts["wall"] == 1 and counts["door"] == 1
    assert [(o["id"], o["type"]) for o in s.part.ir["openings"]] == [("U2", "door"), ("U3", "unknown")]
    assert len(original["openings"]) == 3  # 원래 IR은 그대로


def test_off_does_nothing_and_cache_leaves_a_pending_list(tmp_path, monkeypatch):
    s = storey([gap("U1", 2000.0)])
    monkeypatch.setenv("CAD2BIM_LLM", "off")
    assert review.review([s], {}, tmp_path, {}) is None
    monkeypatch.setenv("CAD2BIM_LLM", "cache")
    monkeypatch.setattr(review, "render_spots", lambda *a, **k: {})
    line = review.review([s], {}, tmp_path, {})
    assert "pending=1" in line
    pending = json.loads((tmp_path / "_llm" / review.PENDING).read_text(encoding="utf-8"))
    review.save_decisions(tmp_path / "_llm", {pending[0]["key"]: {"answer": "window", "reason": "창틀"}})
    s2 = storey([gap("U1", 2000.0)])
    assert "decided=1 pending=0" in review.review([s2], {}, tmp_path, {})
    assert s2.part.ir["openings"][0]["type"] == "window"


def test_item_list_for_the_app_keeps_decided_spots_and_gets_building_coordinates(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD2BIM_LLM", "cache")
    monkeypatch.setattr(review, "render_spots", lambda *a, **k: {})
    s = storey([gap("U1", 2000.0), gap("U2", 5000.0)])
    s.shift, s.elevation = (300.0, -50.0), 4500.0
    key = review.find_spots([s])[0].key
    review.save_decisions(tmp_path / "_llm", {key: {"answer": "door", "reason": "문 호", "model": "openai:test"}})
    review.review([s], {}, tmp_path, {})
    assert review.locate_items([s], tmp_path, (1000.0, 0.0)) == 2
    data = json.loads((tmp_path / "_llm" / review.ITEMS).read_text(encoding="utf-8"))
    decided, waiting = data["items"]  # 답이 있는 곳도 목록에 남는다(대기 목록에는 없다)
    assert decided["decision"]["answer"] == "door" and waiting["decision"] is None
    assert decided["kind"] == "opening" and decided["floor"] == "F2" and decided["width"] == 1200
    ox, oy = s.part.ir["origin_offset"]
    assert decided["position"] == [2000.0 + ox + 300.0 - 1000.0, 0.0 + oy - 50.0, 4500.0]  # 도면 좌표 + 원점 + 층 이동 - 공통 원점
    assert data["answers"] == list(review.ANSWERS)


def test_outline_without_a_wall_becomes_a_spot_and_a_wall_answer_closes_it():
    ir = plan_ir()
    ir["source"] = {"sha256": "abcdef0123456789zz", "unit_scale": 1.0}
    ir["columns"] = []
    t = 200.0
    ir["walls"] = [  # 10m x 8m 상자에서 위쪽 벽 가운데 4m가 비었다
        {"id": "W1", "start": [0.0, 0.0], "end": [10000.0, 0.0], "thickness": t, "kind": "unknown"},
        {"id": "W2", "start": [10000.0, 0.0], "end": [10000.0, 8000.0], "thickness": t, "kind": "unknown"},
        {"id": "W3", "start": [0.0, 8000.0], "end": [3000.0, 8000.0], "thickness": t, "kind": "unknown"},
        {"id": "W4", "start": [7000.0, 8000.0], "end": [10000.0, 8000.0], "thickness": t, "kind": "unknown"},
        {"id": "W5", "start": [0.0, 0.0], "end": [0.0, 8000.0], "thickness": t, "kind": "unknown"},
    ]
    ir["openings"] = []
    s = Storey("F1", PlanPart("D01", "F1", ir, 1.0, "title"), 0.0, 3000.0)
    export = {"slab_close_mm": 3000.0, "slab_min_area_m2": 10.0, "slab_trim_mm": 1000.0}
    spots = review.find_outline_spots([s], export)
    assert len(spots) == 1 and spots[0].kind == "outline"
    assert 3000.0 <= spots[0].width <= 4000.0 and abs(spots[0].center[1] - 8000.0) < 200
    counts = review.apply_decisions([s], spots, {spots[0].key: {"answer": "wall"}})
    assert counts["wall"] == 1 and s.part.ir["walls"][-1]["id"].startswith("LW-")
    assert len(ir["walls"]) == 5  # 원래 IR은 그대로


def test_narrow_slit_on_the_outline_is_closed_without_asking():
    ir = plan_ir()
    ir["source"] = {"sha256": "abcdef0123456789zz", "unit_scale": 1.0}
    ir["columns"] = []
    t = 200.0
    ir["walls"] = [  # 위쪽 벽 사이에 150mm 틈
        {"id": "W1", "start": [0.0, 0.0], "end": [10000.0, 0.0], "thickness": t, "kind": "unknown"},
        {"id": "W2", "start": [10000.0, 0.0], "end": [10000.0, 8000.0], "thickness": t, "kind": "unknown"},
        {"id": "W3", "start": [0.0, 8000.0], "end": [4925.0, 8000.0], "thickness": t, "kind": "unknown"},
        {"id": "W4", "start": [5075.0, 8000.0], "end": [10000.0, 8000.0], "thickness": t, "kind": "unknown"},
        {"id": "W5", "start": [0.0, 0.0], "end": [0.0, 8000.0], "thickness": t, "kind": "unknown"},
    ]
    ir["openings"] = []
    s = Storey("F1", PlanPart("D01", "F1", ir, 1.0, "title"), 0.0, 3000.0)
    export = {"slab_close_mm": 3000.0, "slab_min_area_m2": 10.0, "slab_trim_mm": 1000.0}
    slits = review.find_slits([s], export, 400.0)
    assert len(slits) == 1 and slits[0].kind == "slit" and abs(slits[0].center[0] - 5000.0) < 30
    counts = review.apply_decisions([s], slits, {})
    assert counts["slit"] == 1 and s.part.ir["walls"][-1]["id"].startswith("LW-")


def test_wall_ending_short_of_a_neighbour_becomes_a_corner_spot_and_wall_extends():
    ir = plan_ir()
    ir["source"] = {"sha256": "abcdef0123456789zz", "unit_scale": 1.0}
    ir["columns"] = []
    ir["walls"] = [  # 세로 벽이 가로 벽 600mm 앞에서 끝난다
        {"id": "A", "start": [0.0, 0.0], "end": [6000.0, 0.0], "thickness": 200.0, "kind": "unknown"},
        {"id": "B", "start": [3000.0, 700.0], "end": [3000.0, 5000.0], "thickness": 100.0, "kind": "unknown"},
    ]
    ir["openings"] = []
    s = Storey("F2", PlanPart("D07", "F2", ir, 1.0, "title"), 0.0, 3000.0)
    corners = review.find_corner_spots([s])
    assert len(corners) == 1 and corners[0].kind == "corner" and 550 <= corners[0].width <= 650
    review.apply_decisions([s], corners, {corners[0].key: {"answer": "wall"}})
    added = s.part.ir["walls"][-1]
    assert added["id"].startswith("LW-") and min(added["start"][1], added["end"][1]) <= 101 and max(added["start"][1], added["end"][1]) >= 699


def test_l_corner_where_two_wall_ends_miss_each_other_is_closed_by_two_legs():
    """물류 2층 도크 옆: 세로 벽은 가로 벽 끝보다 바깥에서 끝나 어느 쪽을 늘려도 서로 닿지 않는다."""
    ir = plan_ir()
    ir["source"] = {"sha256": "abcdef0123456789zz", "unit_scale": 1.0}
    ir["columns"] = []
    ir["walls"] = [
        {"id": "A", "start": [0.0, 550.0], "end": [0.0, 3400.0], "thickness": 100.0, "kind": "unknown"},
        {"id": "B", "start": [600.0, 0.0], "end": [6000.0, 0.0], "thickness": 200.0, "kind": "unknown"},
    ]
    ir["openings"] = []
    s = Storey("F2", PlanPart("D07", "F2", ir, 1.0, "title"), 0.0, 3000.0)
    corners = review.find_corner_spots([s])
    assert len(corners) == 1 and corners[0].path is not None
    review.apply_decisions([s], corners, {corners[0].key: {"answer": "wall"}})
    added = [w for w in s.part.ir["walls"] if w["id"].startswith("LW-")]
    points = {tuple(round(v) for v in p) for w in added for p in (w["start"], w["end"])}
    assert points == {(0, 550), (0, -100), (-100, 0), (600, 0)}  # 만나는 점 너머로 반 두께씩 늘려 모서리가 각지다


def test_narrow_l_corner_is_filled_without_asking():
    """0.4m보다 좁은 L자 모서리 틈은 문·창일 수 없으니 판단 없이(slit) 메운다."""
    ir = plan_ir()
    ir["source"] = {"sha256": "abcdef0123456789zz", "unit_scale": 1.0}
    ir["columns"] = []
    ir["walls"] = [  # 물류 3층 코어 북서 모서리: 서쪽 벽 끝과 북쪽 벽 끝이 0.3m 어긋남
        {"id": "A", "start": [0.0, -4800.0], "end": [0.0, 0.0], "thickness": 200.0, "kind": "unknown"},
        {"id": "B", "start": [300.0, 100.0], "end": [3000.0, 100.0], "thickness": 150.0, "kind": "unknown"},
    ]
    ir["openings"] = []
    s = Storey("F3", PlanPart("D08", "F3", ir, 1.0, "title"), 0.0, 3000.0)
    corners = review.find_corner_spots([s])
    assert [sp.kind for sp in corners] == ["slit"]
    review.apply_decisions([s], corners, {})
    assert any(w["id"].startswith("LW-F3-L") for w in s.part.ir["walls"])


def test_window_in_an_interior_wall_is_asked_and_can_become_a_door():
    """실내 벽의 창은 드물다. 진짜 창인지 LLM이 도면을 보고 고르고, 문이면 문으로 바꾼다."""
    from cad2bim.config import load_defaults

    export = load_defaults()["export"]
    ir = plan_ir()
    ir["source"] = {"sha256": "abcdef0123456789zz", "unit_scale": 1.0}
    ir["columns"] = []
    ir["walls"] = [
        {"id": f"E{i}", "start": a, "end": b, "thickness": 200.0, "kind": "unknown"}
        for i, (a, b) in enumerate([([0.0, 0.0], [10000.0, 0.0]), ([10000.0, 0.0], [10000.0, 8000.0]),
                                    ([10000.0, 8000.0], [0.0, 8000.0]), ([0.0, 8000.0], [0.0, 0.0])])
    ] + [{"id": "IN", "start": [5000.0, 0.0], "end": [5000.0, 8000.0], "thickness": 150.0, "kind": "unknown"}]
    ir["openings"] = [
        {"id": "OUTWIN", "type": "window", "host_wall": "E0", "position": 2000.0, "width": 1200.0},
        {"id": "INWIN", "type": "window", "host_wall": "IN", "position": 4000.0, "width": 900.0},
    ]
    s = Storey("F1", PlanPart("D06", "F1", ir, 1.0, "title"), 0.0, 3000.0)
    spots = review.find_interior_windows([s], export)
    assert [sp.opening_id for sp in spots] == ["INWIN"]
    review.apply_decisions([s], spots, {spots[0].key: {"answer": "door"}})
    assert {o["id"]: o["type"] for o in s.part.ir["openings"]} == {"OUTWIN": "window", "INWIN": "door"}


def test_llm_review_stops_at_the_daily_quota_and_keeps_what_it_got(tmp_path, monkeypatch):
    """무료 등급은 하루 한도가 있다. 다 되면 더 묻지 않고, 받은 답은 저장해 다음에 남은 것만 묻는다."""
    from cad2bim.llm_client import QuotaExhausted

    calls = []

    def fake_ask(provider, model, prompt, image=None, max_tokens=0):
        calls.append(1)
        if len(calls) > 2:
            raise QuotaExhausted("하루 한도")
        return '{"answer": "wall", "reason": "벽선 이어짐"}'

    monkeypatch.setattr(review, "ask", fake_ask)
    spots = [review.Spot(f"k{i}", "F1", "D01", f"U{i}", (0.0, 0.0), (1.0, 0.0), 900.0, 200.0) for i in range(5)]
    images = {}
    for sp in spots:
        path = tmp_path / f"{sp.key}.png"
        path.write_bytes(b"png")
        images[sp.key] = path
    got = review.ask_llm(spots, images, "gemini", "m", tmp_path)
    assert set(got) == {"k0", "k1"} and len(calls) == 3
    assert set(review.load_decisions(tmp_path)) == {"k0", "k1"}


def test_answers_that_contradict_the_drawing_data_are_held_back():
    """판단은 LLM, 거르기는 계산. 어긋난 답은 적용하지 않고 대기 목록에 남긴다."""
    assert review.consistent("door", {"COLUMN": 1, "WINDOW": 4}, "corner") is False  # 모서리 기둥은 벽만
    assert review.consistent("wall", {"COLUMN": 1, "WINDOW": 4}, "corner") is True
    assert review.consistent("door", {"COLUMN": 1, "DOOR": 1}, "opening") is True  # 기둥 옆 문은 있을 수 있다
    assert review.consistent("open", {"DOOR": 1, "WALL": 1}, "opening") is False  # 문 선이 있는데 열린 곳
    assert review.consistent("wall", {"WINDOW": 2}, "window") is False  # 창 선이 있는데 벽
    assert review.consistent("door", {"GRID": 1}, "outline") is False  # 축선뿐인데 문


def test_answers_received_now_are_applied_in_the_same_build(tmp_path, monkeypatch):
    """버그: 적용한 뒤에 물어서, API로 새로 받은 답은 다음 빌드에서야 쓰였다."""
    s = storey([gap("U1", 2000.0)])
    monkeypatch.setenv("CAD2BIM_LLM", "gemini")
    monkeypatch.setattr(review, "render_spots", lambda spots, *a, **k: {sp.key: tmp_path / "x.png" for sp in spots})
    monkeypatch.setattr(review, "spot_evidence", lambda *a, **k: {})
    monkeypatch.setattr(review, "default_model", lambda cfg, provider: "m")
    monkeypatch.setattr(review, "ask_llm", lambda spots, *a, **k: {sp.key: {"answer": "door", "reason": "문"} for sp in spots})
    line = review.review([s], {}, tmp_path, {})
    assert "pending=0" in line and s.part.ir["openings"][0]["type"] == "door"


def test_an_answer_that_contradicts_the_drawing_is_kept_aside_with_the_reason(tmp_path, monkeypatch):
    """LLM의 답이 도면 근거와 어긋나면 적용하지 않는다. 무엇을 왜 걸렀는지는 남겨 화면에서 보여 준다."""
    s = storey([gap("U1", 2000.0), gap("U2", 5000.0)])
    first, second = review.find_spots([s])
    image = tmp_path / "spot.png"
    image.write_bytes(b"png")
    evidence = {first.key: {"text": "", "counts": {"DOOR": 1, "ARC": 2}}, second.key: {"text": "", "counts": {"WALL": 2}}}
    monkeypatch.setattr(review, "ask", lambda *a, **k: '{"answer": "open", "reason": "열린 공간"}')
    directory = tmp_path / "_llm"
    answers = review.ask_llm([first, second], {first.key: image, second.key: image}, "gemini", "m", directory, evidence)
    assert list(answers) == [second.key]  # 문 표시가 없는 자리의 '열린 통로'는 받아들인다
    kept = review.load_rejected(directory)[first.key]
    assert kept["answer"] == "open" and kept["model"] == "gemini:m" and "문선 1개" in kept["why"] and "원호 2개" in kept["why"]
    assert review.conflict("door", {"DOOR": 1}) is None and review.consistent("door", {"DOOR": 1})
    assert "창선 4개" in review.conflict("wall", {"WINDOW": 4, "WALL": 2}, "window")
    assert "기둥" in review.conflict("door", {"COLUMN": 1}, "corner") and review.conflict("door", {"COLUMN": 1}) is None
    assert "아무 선도 없는데" in review.conflict("window", {"GRID": 1})
    # 다음에 받아들일 답이 오면 걸러진 기록은 지운다
    monkeypatch.setattr(review, "ask", lambda *a, **k: '{"answer": "door", "reason": "문 호"}')
    review.ask_llm([first], {first.key: image}, "gemini", "m", directory, evidence)
    assert first.key not in review.load_rejected(directory)


def test_item_list_carries_the_rejected_answer_until_an_ai_answer_is_applied(tmp_path, monkeypatch):
    monkeypatch.setenv("CAD2BIM_LLM", "cache")
    monkeypatch.setattr(review, "render_spots", lambda *a, **k: {})
    s = storey([gap("U1", 2000.0)])
    key = review.find_spots([s])[0].key
    review.save_rejected(tmp_path / "_llm", {key: {"answer": "open", "reason": "r", "model": "gemini:m", "why": "문선이 있습니다."}})
    review.review([s], {}, tmp_path, {})
    item = json.loads((tmp_path / "_llm" / review.ITEMS).read_text(encoding="utf-8"))["items"][0]
    assert item["decision"] is None and item["rejected"]["answer"] == "open" and item["rejected"]["why"] == "문선이 있습니다."
