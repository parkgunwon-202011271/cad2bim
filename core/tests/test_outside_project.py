"""입력 폴더 보호. 프로젝트 안에서는 data만 허용한다."""
from __future__ import annotations

import pytest

from cad2bim.config import config_dir
from cad2bim.errors import Cad2BimError
from cad2bim.evaluation.audit import _check_outside_project

PROJECT = config_dir().resolve().parent


def test_project_folder_is_refused():
    with pytest.raises(Cad2BimError):
        _check_outside_project(PROJECT / "core")


def test_project_root_is_refused():
    with pytest.raises(Cad2BimError):
        _check_outside_project(PROJECT)


def test_data_folder_is_allowed():
    _check_outside_project(PROJECT / "data" / "public" / "주민센터")


def test_outside_folder_is_allowed(tmp_path):
    _check_outside_project(tmp_path)
