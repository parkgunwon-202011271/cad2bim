"""report.html. 외부 리소스 없이 파일 하나로 열린다. 도면 내용이 밖으로 나가지 않아야 한다."""
from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path

from ..ir import IR
from ..preprocess import Preprocessed
from .svg import render_svg

_STYLE = """
:root { color-scheme: light dark; --bg:#fff; --fg:#1a1a1a; --muted:#666; --line:#444; --faint:#bbb;
        --hatch:rgba(90,110,140,.22); --ok:#1a7f4b; --review:#d9a400; --panel:#f4f4f2; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#17181a; --fg:#e8e8e6; --muted:#a0a0a0; --line:#c8c8c8; --faint:#555;
          --hatch:rgba(150,170,200,.25); --panel:#222427; } }
* { box-sizing: border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--fg);
       font:14px/1.5 system-ui,"Malgun Gothic",sans-serif; }
h1 { font-size:18px; margin:0 0 12px; }
.meta { display:flex; flex-wrap:wrap; gap:8px 24px; background:var(--panel); padding:12px;
        border-radius:6px; margin-bottom:12px; }
.meta div { min-width:130px; } .meta dt { color:var(--muted); font-size:12px; } .meta dd { margin:0; }
.warn { color:var(--review); font-weight:600; }
.controls { margin-bottom:8px; } .controls label { margin-right:16px; }
svg { width:100%; height:auto; max-height:80vh; display:block; background:var(--bg); }
svg path, svg circle, svg rect { fill:none; stroke:var(--line); stroke-width:var(--sw); }
svg path.hatch { fill:var(--hatch); stroke:none; }
svg path.dim { stroke:var(--faint); }
svg text { fill:var(--fg); stroke:none; font-family:system-ui,"Malgun Gothic",sans-serif; }
.layer-excluded * { stroke:var(--faint); opacity:.45; }
.layer-excluded text { fill:var(--faint); }
.layer-ir .wall { stroke-opacity:.45; stroke-linecap:butt; }
.layer-ir .ok { stroke:var(--ok); } .layer-ir .review { stroke:var(--review); }
.layer-ir .column.ok, .layer-ir .room.ok { fill:var(--ok); fill-opacity:.2; }
.layer-ir .column.review, .layer-ir .room.review { fill:var(--review); fill-opacity:.25; }
"""

_SCRIPT = """
for (const box of document.querySelectorAll('input[data-layer]')) {
  box.addEventListener('change', () => {
    document.querySelector('.' + box.dataset.layer).style.display = box.checked ? '' : 'none';
  });
}
"""


def _item(label: str, value: object, cls: str = "") -> str:
    return (
        f'<div><dt>{escape(label)}</dt><dd class="{cls}">{escape(str(value))}</dd></div>'
    )


def render_html(pre: Preprocessed, ir: IR) -> str:
    kinds = Counter(p.kind for p in pre.primitives)
    bbox = pre.plan_bbox
    extent = f"{(bbox[2] - bbox[0]) / 1000:.1f} × {(bbox[3] - bbox[1]) / 1000:.1f} m" if bbox else "-"
    review = sum(
        e.confidence.level == "review" for e in (*ir.walls, *ir.columns, *ir.openings, *ir.rooms)
    )
    items = [
        _item("파일", ir.source.file),
        _item("축척(도면 단위 → mm)", f"{pre.scale.factor:g} ({pre.scale.source})"),
        _item("평면 범위", extent),
        _item("도곽", f"{len(pre.frames)}개"),
        _item("원점 이동", f"{ir.origin_offset[0]:g}, {ir.origin_offset[1]:g}"),
        _item("정규화", f"병합 {pre.normalize_stats.merged_away}, 극소 제거 {pre.normalize_stats.removed_short}"),
        _item("요소", ", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))),
        _item("벽·기둥·개구부·실", f"{len(ir.walls)}·{len(ir.columns)}·{len(ir.openings)}·{len(ir.rooms)}"),
        _item("검토 필요", review, "warn" if review else ""),
        _item("미지원 엔티티", dict(pre.explode_stats.unsupported) or "없음"),
    ]
    if ir.warnings:
        items.append(_item("경고", ", ".join(ir.warnings), "warn"))
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CAD2BIM 보고서</title><style>{_STYLE}</style></head>
<body>
<h1>{escape(ir.source.file)}</h1>
<dl class="meta">{''.join(items)}</dl>
<div class="controls">
<label><input type="checkbox" data-layer="layer-source" checked> 원본</label>
<label><input type="checkbox" data-layer="layer-ir" checked> 인식 결과</label>
<label><input type="checkbox" data-layer="layer-excluded" checked> 제외된 요소</label>
</div>
{render_svg(pre.primitives, pre.excluded, ir)}
<script>{_SCRIPT}</script>
</body></html>
"""


def write_report(path: Path, pre: Preprocessed, ir: IR) -> None:
    path.write_text(render_html(pre, ir), encoding="utf-8")
