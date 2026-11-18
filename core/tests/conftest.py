from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import ezdxf
import pytest

from cad2bim.config import load_defaults, load_thresholds

ROOT = Path(__file__).resolve().parents[2]
DRAWINGS = ROOT / "DATASET" / "cad2bim_synthetic_drawings" / "cad2bim" / "out"
NAMES = [
    "A1_retail_clean_en",
    "A2_retail_messy",
    "B1_apt_clean_kr",
    "B2_apt_partial_code",
]


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_defaults()


@pytest.fixture(scope="session")
def thresholds() -> dict:
    return load_thresholds()


def add_dimensions(msp, length: float = 6000.0, count: int = 3, y: float = 0.0) -> None:
    """정상 치수 count개. 렌더링까지 해서 표기값이 블록에 들어간다."""
    for i in range(count):
        dim = msp.add_linear_dim(
            base=(i * length, y - length / 5), p1=(i * length, y), p2=((i + 1) * length, y),
            override={"dimlfac": 1.0, "dimdec": 0},
        )
        dim.render()


def add_plan(msp, x: float = 0.0, y: float = 0.0, size: float = 10000.0, lines: int = 25) -> None:
    """평면 후보로 인정될 만큼 선이 있는 작은 묶음."""
    for i in range(lines):
        yy = y + i * size / lines
        msp.add_line((x, yy), (x + size, yy), dxfattribs={"layer": "WALL"})


@pytest.fixture
def make_dxf(tmp_path: Path) -> Callable[..., Path]:
    """build(doc, msp)로 그린 도면을 저장하고 경로를 돌려준다."""

    def _make(build: Callable, name: str = "t.dxf", version: str = "R2018", units: int = 4) -> Path:
        doc = ezdxf.new(version, setup=True)
        doc.header["$INSUNITS"] = units
        build(doc, doc.modelspace())
        path = tmp_path / name
        doc.saveas(path)
        return path

    return _make
