"""생성한 DXF를 PNG로 렌더링해 눈으로 확인한다. (가짜 도면 전용)

사용법: python preview.py ../out
"""
import sys
from pathlib import Path

import ezdxf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

PREVIEW_FONT = "NotoSansCJK-Regular.ttc"  # 미리보기에서만 한글 글꼴로 대체


def render(dxf_path: Path, png_path: Path):
    doc = ezdxf.readfile(dxf_path)
    for st in doc.styles:
        st.dxf.font = PREVIEW_FONT
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_axes([0, 0, 1, 1])
    cfg = Configuration(background_policy=BackgroundPolicy.WHITE,
                        color_policy=ColorPolicy.BLACK)
    Frontend(RenderContext(doc), MatplotlibBackend(ax), config=cfg).draw_layout(doc.modelspace())
    ax.set_title(dxf_path.stem, fontsize=14)
    fig.savefig(png_path, dpi=110, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "../out")
    for p in sorted(out.glob("*.dxf")):
        png = p.with_suffix(".png")
        render(p, png)
        print("rendered", png)
