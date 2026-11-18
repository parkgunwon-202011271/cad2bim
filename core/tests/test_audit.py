"""실제 도면용 audit. 핵심은 어떤 경로로도 도면 내용이 출력에 섞이지 않는 것이다."""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import ezdxf
import pytest
import yaml
from conftest import DRAWINGS, add_plan

import cad2bim.evaluation.audit as audit
from cad2bim.io import dwg
from cad2bim.cli import main

@pytest.fixture(autouse=True)
def no_installed_converter(monkeypatch):
    """개발 PC에 변환기를 설치해도 테스트 결과가 달라지지 않게 한다."""
    monkeypatch.delenv("ODA_FILE_CONVERTER", raising=False)
    monkeypatch.delenv("LIBREDWG_DWG2DXF", raising=False)
    monkeypatch.setattr(dwg, "_installed_libredwg", lambda: None)
    monkeypatch.setattr(dwg, "_installed_oda", lambda: None)
    monkeypatch.setattr(dwg.shutil, "which", lambda name: None)


BLOCK_START = "아래부터 복사"
BLOCK_END = "여기까지 복사"


def save(path, build) -> None:
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 4
    build(doc, doc.modelspace())
    doc.saveas(path)


def normal_drawing(doc, msp):
    block = doc.blocks.new("SECRETBLOCK")
    block.add_line((0, 0), (500, 0), dxfattribs={"layer": "SECRETBLOCKLAYER"})
    msp.add_blockref("SECRETBLOCK", (1000, 1000))
    for i in range(25):
        msp.add_line((0, i * 200), (6000, i * 200), dxfattribs={"layer": "SECRETLAYER"})
    msp.add_text("SECRETTEXT", height=100, dxfattribs={"layer": "SECRETTEXTLAYER"}).set_placement((100, 100))
    # 지원하지 않는 엔티티가 있어도 이름 외 내용은 나오지 않아야 한다.
    msp.add_point((10, 10), dxfattribs={"layer": "SECRETPOINTLAYER"})


def xref_drawing(doc, msp):
    doc.add_xref_def("SECRETXREF.dwg", "SECRETXREF")
    msp.add_blockref("SECRETXREF", (0, 0))


def two_plans(doc, msp):
    add_plan(msp, 0, 0)
    add_plan(msp, 50000, 0)


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "SECRETDIR"
    root.mkdir()
    save(root / "SECRETFILE.dxf", normal_drawing)
    save(root / "SECRETXREFFILE.dxf", xref_drawing)
    save(root / "SECRETTWOPLANS.dxf", two_plans)
    (root / "SECRETBAD.dxf").write_text("SECRETGARBAGE", encoding="utf-8")
    (root / "SECRETDWG.dwg").write_bytes(b"SECRETDWGBYTES")
    return root


def output(capsys) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


def test_no_drawing_content_in_terminal_or_paste_file(folder, capsys):
    assert main(["audit", str(folder)]) == 0
    text = output(capsys)
    assert "SECRET" not in text
    assert "SECRET" not in (folder / audit.OUT_DIR_NAME / audit.PASTE_NAME).read_text(encoding="utf-8")


def test_failures_are_reported_by_code_and_run_continues(folder, capsys):
    main(["audit", str(folder)])
    text = output(capsys)
    assert "drawings=5 failed=3" in text
    for code in ("Cad2BimError", "XrefMissingError", "ConverterNotAvailableError"):
        assert code in text
    assert "RegionAmbiguousError" not in text
    assert "split parts=2 processed=2" in text
    assert text.count("] ok") == 3


def test_local_map_keeps_names_for_the_user(folder):
    audit.run_audit(folder, 3000)
    mapping = json.loads((folder / audit.OUT_DIR_NAME / "map.json").read_text(encoding="utf-8"))
    files = {v["file"] for v in mapping.values()}
    assert "SECRETFILE.dxf" in files
    assert any("SECRETXREF" in (v["error"] or "") for v in mapping.values())


def test_unexpected_exception_and_library_logs_do_not_leak(folder, monkeypatch, capsys):
    def boom(*args, **kwargs):
        logging.getLogger("ezdxf").error("SECRETLOG")
        raise ValueError("SECRETMESSAGE at C:\\SECRETPATH")

    monkeypatch.setattr(audit, "convert", boom)
    assert main(["audit", str(folder)]) == 0
    text = output(capsys)
    assert "SECRET" not in text
    assert "ValueError" in text


def test_real_location_of_our_bug_is_reported(folder, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise KeyError("SECRETLAYER")

    monkeypatch.setattr(audit, "_report_ok", boom)
    main(["audit", str(folder)])
    text = output(capsys)
    assert "SECRET" not in text
    assert "KeyError at audit:" in text


def test_top_level_errors_hide_the_path(tmp_path, capsys):
    assert main(["audit", str(tmp_path / "SECRETMISSING")]) == 2
    assert "SECRET" not in output(capsys)


def test_refuses_folder_inside_project(capsys):
    assert main(["audit", str(DRAWINGS)]) == 2
    assert "프로젝트 밖" in output(capsys)


def test_empty_folder(tmp_path, capsys):
    assert main(["audit", str(tmp_path)]) == 2
    assert "DXF" in output(capsys)


def test_synthetic_drawing_with_truth_prints_numbers_only(tmp_path, capsys):
    shutil.copy(DRAWINGS / "B1_apt_clean_kr.dxf", tmp_path)
    shutil.copy(DRAWINGS / "B1_apt_clean_kr.truth.json", tmp_path)
    assert main(["audit", str(tmp_path)]) == 0
    text = output(capsys)
    assert "metrics: walls.recall=1.000" in text
    assert "walls: n=10" in text
    for leaked in ("B1_apt", "침실", "거실", "벽체", "창호", "문-900"):
        assert leaked not in text


def test_without_truth_reports_no_truth(tmp_path, capsys):
    shutil.copy(DRAWINGS / "A1_retail_clean_en.dxf", tmp_path)
    main(["audit", str(tmp_path)])
    text = output(capsys)
    assert "metrics: no_truth" in text
    assert "A-WALL" not in text and "상가" not in text


def test_block_is_marked_for_copying(tmp_path, capsys):
    shutil.copy(DRAWINGS / "A1_retail_clean_en.dxf", tmp_path)
    main(["audit", str(tmp_path)])
    text = output(capsys)
    assert text.index(BLOCK_START) < text.index("CAD2BIM-AUDIT") < text.index(BLOCK_END)


def test_overrides_region_fixes_two_plan_file_without_printing_it(folder, capsys):
    overrides = {"SECRETTWOPLANS.dxf": {"region": [45000, -2000, 70000, 12000]}}
    (folder / audit.OVERRIDES_NAME).write_text(yaml.safe_dump(overrides), encoding="utf-8")
    main(["audit", str(folder)])
    text = output(capsys)
    assert "RegionAmbiguousError" not in text
    assert "SECRET" not in text


def test_rerun_ignores_its_own_output(folder):
    audit.run_audit(folder, 3000)
    shutil.copy(folder / "SECRETFILE.dxf", folder / audit.OUT_DIR_NAME / "copy.dxf")
    assert len(audit.find_drawings(folder)) == 5


def many_copies(folder, count):
    for i in range(count):
        shutil.copy(DRAWINGS / "A1_retail_clean_en.dxf", folder / f"SECRET{i:03d}.dxf")


def test_many_drawings_give_one_line_each(tmp_path, capsys):
    many_copies(tmp_path, 6)
    (tmp_path / "SECRETBAD.dxf").write_text("x", encoding="utf-8")
    assert main(["audit", str(tmp_path)]) == 0
    text = output(capsys)
    assert "SECRET" not in text and "상가" not in text
    assert "failures: Cad2BimError n=1 ids=D07" in text
    assert "walls_per_drawing: min=8 median=8 max=8 total=48" in text
    assert "D01 f=1 src=dimension lay=y" in text
    assert "wall_thickness_mm" not in text  # 자세한 내용은 --show로


def test_show_prints_only_chosen_drawings_without_rerun(tmp_path, capsys):
    many_copies(tmp_path, 6)
    main(["audit", str(tmp_path)])
    capsys.readouterr()
    for path in tmp_path.glob("*.dxf"):
        path.unlink()
    assert main(["audit", str(tmp_path), "--show", "D02,D99,벽체"]) == 0
    text = output(capsys)
    assert "[D02] ok" in text and "wall_thickness_mm" in text
    assert "[D99] not_found" in text and "[other] not_found" in text
    assert "[D01]" not in text


def test_show_needs_previous_run(tmp_path, capsys):
    assert main(["audit", str(tmp_path), "--show", "D01"]) == 2
    assert "먼저 audit" in output(capsys)


def test_limit_keeps_numbers_stable(tmp_path, capsys):
    many_copies(tmp_path, 8)
    assert main(["audit", str(tmp_path), "--limit", "3"]) == 0
    text = output(capsys)
    assert "drawings=3 failed=0 of_total=8" in text
    first = audit._select(audit.find_drawings(tmp_path), 3)
    assert first == audit._select(audit.find_drawings(tmp_path), 3)
    full = dict(audit._select(audit.find_drawings(tmp_path), None))
    assert all(full[i] == p for i, p in first)


def test_id_ranges():
    assert audit._id_ranges(["D01", "D02", "D03", "D07"]) == "D01-D03,D07"
    assert audit._id_ranges(["D099", "D100", "D101"]) == "D099-D101"
    assert audit._id_ranges([]) == ""


def test_dwg_is_converted_once_and_cached(tmp_path, monkeypatch, capsys):
    (tmp_path / "SECRET 한글[1].dwg").write_bytes(b"fake")
    (tmp_path / "SECRET 한글[1].truth.json").unlink(missing_ok=True)
    calls = []

    real_run = dwg.subprocess.run

    def fake_converter(args, **kwargs):
        if args[0] != "oda.exe":  # git 같은 다른 호출은 그대로 둔다
            return real_run(args, **kwargs)
        calls.append(args)
        source, target = Path(args[1]), Path(args[2])
        assert (source / "input.dwg").is_file()  # 원래 이름은 변환기에 넘기지 않는다
        shutil.copy(DRAWINGS / "A1_retail_clean_en.dxf", target / "input.dxf")

    monkeypatch.setattr(dwg, "_installed_oda", lambda: "oda.exe")
    monkeypatch.setattr(dwg.subprocess, "run", fake_converter)
    assert main(["audit", str(tmp_path)]) == 0
    assert main(["audit", str(tmp_path)]) == 0
    assert len(calls) == 1
    assert calls[0][4:8] == ["DXF", "0", "1", "*.DWG"]
    text = output(capsys)
    assert "SECRET" not in text and "[D01] ok" in text


def test_converter_without_output_is_a_failure(tmp_path, monkeypatch, capsys):
    (tmp_path / "SECRET.dwg").write_bytes(b"fake")
    monkeypatch.setattr(dwg, "_installed_oda", lambda: "oda.exe")
    real_run = dwg.subprocess.run
    monkeypatch.setattr(
        dwg.subprocess, "run",
        lambda args, **kwargs: None if args[0] == "oda.exe" else real_run(args, **kwargs),
    )
    assert main(["audit", str(tmp_path)]) == 0
    text = output(capsys)
    assert "failures: Cad2BimError n=1" in text and "SECRET" not in text


def test_libredwg_is_preferred_and_called_with_temp_names(tmp_path, monkeypatch, capsys):
    (tmp_path / "SECRET 한글.dwg").write_bytes(b"fake")
    real_run = dwg.subprocess.run
    calls = []

    def fake(args, **kwargs):
        if args[0] != "dwg2dxf.exe":
            return real_run(args, **kwargs)
        calls.append(args)
        assert args[1:3] == ["-y", "-o"]
        assert args[4].endswith("input.dwg")
        shutil.copy(DRAWINGS / "B1_apt_clean_kr.dxf", args[3])

    monkeypatch.setattr(dwg, "_installed_libredwg", lambda: "dwg2dxf.exe")
    monkeypatch.setattr(dwg, "_installed_oda", lambda: "oda.exe")
    monkeypatch.setattr(dwg.subprocess, "run", fake)
    assert main(["audit", str(tmp_path)]) == 0
    text = output(capsys)
    assert len(calls) == 1
    assert "converter=libredwg" in text and "SECRET" not in text


def test_empty_conversion_output_is_a_failure(tmp_path, monkeypatch, capsys):
    (tmp_path / "SECRET.dwg").write_bytes(b"fake")
    real_run = dwg.subprocess.run

    def fake(args, **kwargs):
        if args[0] != "dwg2dxf.exe":
            return real_run(args, **kwargs)
        Path(args[3]).write_bytes(b"")

    monkeypatch.setattr(dwg, "_installed_libredwg", lambda: "dwg2dxf.exe")
    monkeypatch.setattr(dwg.subprocess, "run", fake)
    main(["audit", str(tmp_path)])
    text = output(capsys)
    assert "failures: Cad2BimError n=1" in text and "SECRET" not in text


def test_text_stats_count_without_printing(tmp_path, capsys):
    def build(doc, msp):
        for i in range(25):
            msp.add_line((0, i * 200), (6000, i * 200), dxfattribs={"layer": "A-WALL"})
        for value, x in (("거실", 100), ("???? SECRET", 900), ("ROOM", 1800)):
            msp.add_text(value, height=100).set_placement((x, 100))

    save(tmp_path / "x.dxf", build)
    main(["audit", str(tmp_path)])
    text = output(capsys)
    assert "text: garbled=1 hangul=1 total=3" in text
    assert "SECRET" not in text and "거실" not in text


def test_slow_drawing_times_out_and_run_continues(tmp_path, monkeypatch, capsys):
    many_copies(tmp_path, 2)
    real = audit.convert
    calls = []

    def slow_first(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            with audit.stage("walls"):
                while True:
                    time.sleep(0.01)
        return real(*args, **kwargs)

    monkeypatch.setattr(audit, "convert", slow_first)
    assert main(["audit", str(tmp_path), "--time-limit", "0.5"]) == 0
    text = output(capsys)
    assert "failed=1" in text and "DrawingTimeout" in text
    assert "[D02] ok" in text
    assert "SECRET" not in text


def test_ctrl_c_prints_partial_block_without_traceback(tmp_path, monkeypatch, capsys):
    many_copies(tmp_path, 3)
    real = audit._run_limited
    calls = []

    def interrupt_second(work, limit):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(work, limit)

    monkeypatch.setattr(audit, "_run_limited", interrupt_second)
    assert main(["audit", str(tmp_path)]) == 0
    text = output(capsys)
    assert "drawings=1 failed=0 of_total=3 interrupted=True" in text
    assert "Traceback" not in text and "SECRET" not in text


def test_stage_seconds_are_reported(tmp_path, capsys):
    many_copies(tmp_path, 1)
    main(["audit", str(tmp_path)])
    text = output(capsys)
    assert "seconds_by_stage:" in text
    assert "walls=" in text and "frames=" in text


def test_block_has_self_check_numbers(tmp_path, capsys):
    shutil.copy(DRAWINGS / "A1_retail_clean_en.dxf", tmp_path)
    main(["audit", str(tmp_path)])
    text = output(capsys)
    assert "selfcheck: rot_recall=1.000" in text
    assert "guess: plan_like=1" in text
    assert "selfcheck_median:" in text


def rows_of(text):
    return sorted(
        " ".join(w for w in line.split() if not w.startswith("seconds"))
        for line in text.splitlines() if line.startswith("D0")
    )


def test_parallel_run_gives_the_same_rows_and_leaks_nothing(tmp_path, capsys):
    for name in ("A1_retail_clean_en", "A2_retail_messy", "B1_apt_clean_kr", "B2_apt_partial_code"):
        for i in range(2):
            shutil.copy(DRAWINGS / f"{name}.dxf", tmp_path / f"SECRET{name}{i}.dxf")
    (tmp_path / "SECRETBAD.dxf").write_text("x", encoding="utf-8")
    assert main(["audit", str(tmp_path)]) == 0
    serial = output(capsys)
    assert main(["audit", str(tmp_path), "--jobs", "3"]) == 0
    parallel = output(capsys)
    assert "jobs=3" in parallel and "SECRET" not in parallel
    assert rows_of(serial) == rows_of(parallel)
    ids = [row.split()[0] for row in rows_of(parallel)]
    assert len(ids) == len(set(ids)) == 8  # 실제 결과에서 한 줄이 두 번 보인 적이 있다
    assert "failures: Cad2BimError n=1" in parallel


def test_second_run_uses_cache_and_gives_the_same_rows(tmp_path, capsys):
    many_copies(tmp_path, 6)
    main(["audit", str(tmp_path)])
    first = output(capsys)
    main(["audit", str(tmp_path)])
    second = output(capsys)
    assert rows_of(first) == rows_of(second)
    assert " read=" not in second.split("seconds_by_stage:")[1].split("\n")[0]


def test_diagnostics_never_print_layer_names(tmp_path, capsys):
    def build(doc, msp):
        for layer, y in (("SECRET_STL_PC", 0), ("SECRETOTHER", 1000)):
            msp.add_line((0, y), (8000, y), dxfattribs={"layer": layer})
            msp.add_line((0, y + 200), (8000, y + 200), dxfattribs={"layer": layer})
        for i in range(25):
            msp.add_line((0, 2000 + i * 200), (6000, 2000 + i * 200), dxfattribs={"layer": "SECRETTHIRD"})

    save(tmp_path / "x.dxf", build)
    main(["audit", str(tmp_path)])
    text = output(capsys)
    assert "SECRET" not in text
    assert "unknown_layers: n=3" in text
    assert "U1: len_m=" in text and "layout: x_bands" in text
    assert "STL=1" in text and "PC=1" in text


def test_generic_token_list_has_no_project_words():
    from cad2bim.evaluation.diagnostics import _GENERIC_TOKENS, _tokens

    assert len(set(_GENERIC_TOKENS)) == len(_GENERIC_TOKENS)
    assert _tokens("PLAN$0$A-WALL-CONC") == {"WALL", "CONC"}
    assert _tokens("SECRETWORD") == set()


def test_stage_seconds_do_not_double_count(tmp_path, capsys):
    many_copies(tmp_path, 1)
    main(["audit", str(tmp_path)])
    line = next(l for l in output(capsys).splitlines() if l.startswith("seconds_by_stage:"))
    assert "selfcheck=" in line


def test_quick_mode_skips_self_check_but_keeps_results(tmp_path, capsys):
    many_copies(tmp_path, 6)
    main(["audit", str(tmp_path)])
    full = output(capsys)
    main(["audit", str(tmp_path), "--quick"])
    quick = output(capsys)

    def walls(text):
        return [" ".join(line.split()[:9]) for line in text.splitlines() if line.startswith("D0")]

    assert walls(full) == walls(quick)
    assert " rot=- " in quick


def test_two_plan_file_is_split_and_reported_per_plan(tmp_path, capsys):
    def build(doc, msp):
        for x in (0, 50000):
            for i in range(25):
                msp.add_line((x, i * 400), (x + 8000, i * 400), dxfattribs={"layer": "SECRET"})
                msp.add_line((x, i * 400 + 200), (x + 8000, i * 400 + 200), dxfattribs={"layer": "SECRET"})

    save(tmp_path / "SECRET.dxf", build)
    assert main(["audit", str(tmp_path)]) == 0
    text = output(capsys)
    assert "split_files: 1 plans=2" in text
    assert "[D01.1] ok" in text and "[D01.2] ok" in text
    assert "failed=0" in text and "SECRET" not in text
    assert (tmp_path / audit.OUT_DIR_NAME / "D01.1" / "ir.json").is_file()
    assert main(["audit", str(tmp_path), "--show", "D01"]) == 0
    assert "[D01.2] ok" in output(capsys)


def test_failed_parts_count_the_file_once():
    assert audit._id_ranges(["D03", "D03", "D04"]) == "D03-D04"
    assert audit._id_ranges(["D03.1", "D03.2", "D07"]) == "D03,D07"


def test_match_selects_by_file_name_and_keeps_numbers(tmp_path, capsys):
    for name in ("SECRET_a.dxf", "SECRET_평면도_1.dxf", "SECRET_b.dxf", "SECRET_평면도_2.dxf"):
        shutil.copy(DRAWINGS / "A1_retail_clean_en.dxf", tmp_path / name)
    assert main(["audit", str(tmp_path), "--match", "평면도"]) == 0
    text = output(capsys)
    assert "drawings=2 failed=0 of_total=4" in text
    assert "[D03] ok" in text and "[D04] ok" in text and "[D01]" not in text
    assert "SECRET" not in text and "평면도" not in text


def test_match_with_no_result_is_an_error_without_echo(tmp_path, capsys):
    shutil.copy(DRAWINGS / "A1_retail_clean_en.dxf", tmp_path / "a.dxf")
    assert main(["audit", str(tmp_path), "--match", "SECRET"]) == 2
    assert "SECRET" not in output(capsys)


def test_column_stack_separates_stacked_random_and_duplicate_floors():
    grid = [(x * 9000.0, y * 9000.0) for x in range(4) for y in range(4)]
    upper = grid[:12] + [(x + 9000.0, 50000.0) for x in (0.0, 4000.0)]
    scattered = [(i * 3700.0 + 411.0, i * 2900.0 + 977.0) for i in range(16)]
    line = audit.column_stack_line({"D01": grid, "D02": list(grid), "D03": upper, "D04": scattered})
    assert "drawings=4 pairs=6 duplicate_pairs=1" in line
    assert "best_ge_50=3 of=4" in line


def test_column_stack_ignores_a_different_origin():
    """실제 도면은 파일마다 원점이 달랐다. 같은 층인데 0으로 나오면 안 된다."""
    grid = [(x * 9000.0, y * 7000.0) for x in range(5) for y in range(4)]
    moved = [(x + 123456.0, y - 98765.0) for x, y in grid]
    line = audit.column_stack_line({"D01": grid, "D02": moved})
    assert "duplicate_pairs=1" in line


def test_plan_groups_find_versions_of_the_same_floor():
    a = [(x * 11000.0, y * 11000.0 + 300.0 * ((x * y) % 2)) for x in range(5) for y in range(4)]
    b = [(x + 5000.0, y - 7000.0) for x, y in a]
    other = [(x * 11000.0 + 2000.0 * (y % 2), y * 9000.0) for x in range(3) for y in range(5)]
    line = audit.plan_groups_line({"D01": a, "D02": b, "D03": other, "D04": [(0.0, 0.0)]})
    assert line == "plan_groups: unique=2 grouped=2 D01+D02"


def test_column_stack_needs_enough_columns():
    assert audit.column_stack_line({"D01": [(0.0, 0.0)] * 3}) == "column_stack: drawings=0 pairs=0"


def test_layout_line_reports_why_a_model_is_empty(tmp_path, capsys):
    def build(doc, msp):
        doc.layers.add("SECRETOFF").off()
        doc.blocks.new("SECRETB").add_line((0, 0), (9, 0))
        msp.add_blockref("SECRETB", (0, 0), dxfattribs={"layer": "SECRETOFF"})

    save(tmp_path / "SECRET.dxf", build)
    main(["audit", str(tmp_path), "--classify-only"])
    text = output(capsys)
    assert "model_entities=1 hidden=1" in text
    assert "INSERT=1" in text and "SECRET" not in text


def test_empty_model_space_gets_its_own_failure_code(tmp_path, capsys):
    """실제 도면 D105. 단위 실패로 표시되면 원인을 잘못 찾게 된다."""
    save(tmp_path / "SECRET.dxf", lambda d, m: d.blocks.new("SECRETB").add_line((0, 0), (9, 0)))
    main(["audit", str(tmp_path)])
    text = output(capsys)
    assert "failures: EmptyDrawingError n=1" in text
    assert "UnitScaleError" not in text and "SECRET" not in text


def test_column_groups_and_hatch_names_print_numbers_only(tmp_path, capsys):
    def build(doc, msp):
        for i in range(4):
            for j in range(4):
                x, y = i * 9000, j * 7000
                msp.add_lwpolyline(
                    [(x, y), (x + 800, y), (x + 800, y + 800), (x, y + 800)], close=True, dxfattribs={"layer": "SECRET1"}
                )
        for i in range(25):
            msp.add_line((0, i * 1000), (30000, i * 1000), dxfattribs={"layer": "SECRET2"})
        msp.add_hatch().paths.add_polyline_path([(0, 0), (900, 0), (900, 900)], is_closed=True)
        custom = msp.add_hatch()
        custom.paths.add_polyline_path([(0, 0), (900, 0), (900, 900)], is_closed=True)
        custom.set_pattern_fill("SECRETPATTERN")

    save(tmp_path / "x.dxf", build)
    main(["audit", str(tmp_path), "--quick"])
    text = output(capsys)
    assert "CG1: size_mm=800x800 n=16 grid=1.00 spacing_m=7.0 kinds=polyline:16 roles=UNKNOWN:16" in text
    assert "hatch_patterns:" in text and "SOLID=1" in text and "custom=1" in text
    assert "SECRET" not in text


def test_code_filter_blocks_free_text():
    assert audit._code("dangling_end") == "dangling_end"
    assert audit._code("벽체 레이어") == "other"
    assert audit._code("C:\\x") == "other"
