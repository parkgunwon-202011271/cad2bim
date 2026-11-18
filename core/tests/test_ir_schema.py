from __future__ import annotations

import json
import socket

import pytest
from conftest import ROOT
from pydantic import ValidationError
from pytest_socket import SocketBlockedError

from cad2bim.cli import main
from cad2bim.ir import IR, Level, SourceInfo, ir_json_schema


def minimal() -> IR:
    return IR(
        source=SourceInfo(file="a.dxf", sha256="0" * 64, unit_scale=1.0, unit_source="dimension"),
        level=Level(height=3000),
    )


def test_round_trip():
    ir = minimal()
    assert IR.model_validate_json(ir.model_dump_json()) == ir


def test_unknown_fields_are_rejected():
    data = minimal().model_dump(mode="json")
    data["wals"] = []
    with pytest.raises(ValidationError):
        IR.model_validate(data)


def test_committed_schema_matches_models():
    """schema/ir.schema.json은 C# 쪽 계약이다. 모델이 바뀌면 다시 생성해야 한다."""
    committed = json.loads((ROOT / "schema" / "ir.schema.json").read_text(encoding="utf-8"))
    assert committed == ir_json_schema()


def test_network_is_blocked_during_tests():
    """보안 테스트의 전제. 이게 깨지면 외부 통신 없음을 보증하지 못한다."""
    with pytest.raises(SocketBlockedError):
        socket.socket()


def test_cli_reports_input_errors_without_traceback(tmp_path, capsys):
    code = main(["convert", str(tmp_path / "없음.dxf"), "--out", str(tmp_path / "o")])
    assert code == 2
    assert "파일이 없습니다" in capsys.readouterr().err
