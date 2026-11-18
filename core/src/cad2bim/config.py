"""config/ 폴더의 YAML을 읽는다. 수치는 코드에 직접 쓰지 않는다."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def config_dir() -> Path:
    env = os.environ.get("CAD2BIM_CONFIG")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config" / "defaults.yaml"
        if candidate.is_file():
            return candidate.parent
    raise FileNotFoundError("config/defaults.yaml을 찾을 수 없습니다. CAD2BIM_CONFIG를 지정하세요.")


def load_yaml(name: str, directory: Path | None = None) -> dict[str, Any]:
    path = (directory or config_dir()) / name
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_defaults(directory: Path | None = None) -> dict[str, Any]:
    return load_yaml("defaults.yaml", directory)


def load_layer_dictionary(directory: Path | None = None) -> dict[str, Any]:
    return load_yaml("layer_dict.yaml", directory)


def load_sheet_dictionary(directory: Path | None = None) -> dict[str, Any]:
    return load_yaml("sheet_dict.yaml", directory)


def load_thresholds(directory: Path | None = None) -> dict[str, Any]:
    return load_yaml("thresholds.yaml", directory)
