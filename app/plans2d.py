"""입력 도면을 2D PNG로 그린다. 코어가 만들어 둔 DXF 캐시를 다시 쓴다.

표제란의 학번·사업자번호처럼 숫자만 8자리 이상인 글자는 기본으로 그리지 않는다.
시연 화면에 개인 식별 번호가 나오지 않게 하기 위해서다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import ezdxf
from ezdxf import recover
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .jobs import APP_DIR_NAME, OUT_DIR_NAME

_ID_LIKE = re.compile(r"^\s*\d{8,}\s*$")
_WIDTH_IN = 16.0
_DPI = 150

if sys.platform == "win32":
    ezdxf.options.set("core", "default_font", "malgun.ttf")


def _plain_text(entity) -> str:
    kind = entity.dxftype()
    if kind in ("TEXT", "ATTRIB", "ATTDEF"):
        return entity.dxf.get("text", "") or ""
    if kind == "MTEXT":
        return entity.plain_text()
    return ""


def _hide_id_numbers(entity, properties) -> None:
    if _ID_LIKE.match(_plain_text(entity)):
        properties.is_visible = False


def render_plan(dxf_path: Path, png_path: Path, hide_id_numbers: bool = True) -> None:
    doc, _ = recover.readfile(str(dxf_path))
    fig = Figure(figsize=(_WIDTH_IN, _WIDTH_IN * 0.7))
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    config = Configuration(
        background_policy=BackgroundPolicy.WHITE, color_policy=ColorPolicy.BLACK, lineweight_scaling=0.6,
    )
    frontend = Frontend(RenderContext(doc), MatplotlibBackend(ax), config=config)
    if hide_id_numbers:
        frontend.push_property_override_function(_hide_id_numbers)
    frontend.draw_layout(doc.modelspace(), finalize=True)
    # 도면 비율대로 그림 크기를 맞춘다. 가로로 긴 시트가 작게 나오지 않게 한다.
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    if x1 > x0 and y1 > y0:
        ratio = min(max((y1 - y0) / (x1 - x0), 0.2), 1.5)
        fig.set_size_inches(_WIDTH_IN, _WIDTH_IN * ratio)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=_DPI, facecolor="white")


def render_inputs(out_dir: Path) -> list[dict]:
    """결과 폴더의 입력 도면을 그린다. 이미 그린 것은 다시 그리지 않는다."""
    from cad2bim.evaluation.audit import _as_dxf, find_drawings

    folder = out_dir.parent
    target = out_dir / APP_DIR_NAME / "plans"
    plans = []
    for i, drawing in enumerate(find_drawings(folder), start=1):
        png = target / f"{i:02d}.png"
        if not png.is_file() or png.stat().st_mtime < drawing.stat().st_mtime:
            render_plan(_as_dxf(drawing, out_dir), png)
        plans.append({"name": drawing.stem, "path": png})
    return plans


if __name__ == "__main__":
    # 사용: python -m app.plans2d <입력 폴더>
    folder = Path(sys.argv[1]).resolve()
    for plan in render_inputs(folder / OUT_DIR_NAME):
        print(plan["path"])
