"""큰 도면에서의 속도 측정. 합성 도면을 타일처럼 반복해 실제 도면 규모를 흉내 낸다.

    python core/tools/stress.py --tiles 15 --source B1_apt_clean_kr --out <임시폴더>

실제 도면을 쓰지 않으므로 결과를 어디든 공유해도 된다.
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

import ezdxf

from cad2bim.config import load_defaults
from cad2bim.pipeline import convert
from cad2bim.timing import take_record

import threading

DRAWINGS = Path(__file__).resolve().parents[2] / "DATASET" / "cad2bim_synthetic_drawings" / "cad2bim" / "out"
_SKIP = {"DIMENSION"}  # 치수 블록은 복사하지 않는다. 표본 3개면 축척 판정에 충분하다


def build(source: str, tiles: int, out: Path, in_block: bool = False) -> Path:
    doc = ezdxf.readfile(DRAWINGS / f"{source}.dxf")
    msp = doc.modelspace()
    originals = list(msp)
    x0, y0, x1, y1 = -5500, -6500, 24500, 15500  # 도곽을 넘지 않는 충분한 간격이 아니어도 된다
    step_x, step_y = 12000 if source.startswith("B") else 20000, 10000 if source.startswith("B") else 14000
    frames = [e for e in originals if e.dxf.layer in ("도곽", "FORM", "A-ANNO-TTLB")]
    for e in frames:
        msp.delete_entity(e)
    originals = [e for e in originals if e not in frames and e.is_alive]
    for i in range(tiles):
        for j in range(tiles):
            if i == j == 0:
                continue
            for e in originals:
                if e.dxftype() in _SKIP:
                    continue
                clone = e.copy()
                msp.add_entity(clone)
                clone.translate(i * step_x, j * step_y, 0)
    if in_block:
        # 실제 도면처럼 평면 전체를 블록 하나에 넣는다. 치수는 축척 판정용으로 남긴다.
        sheet = doc.blocks.new("WHOLE_PLAN")
        for e in list(msp):
            if e.dxftype() in _SKIP:
                continue
            msp.unlink_entity(e)
            sheet.add_entity(e)
        msp.add_blockref("WHOLE_PLAN", (0, 0))
    path = out / f"stress_{source}_{tiles}{'_block' if in_block else ''}.dxf"
    doc.saveas(path)
    return path


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="B1_apt_clean_kr")
    parser.add_argument("--tiles", type=int, default=15)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--in-block", action="store_true")
    parser.add_argument("--top", type=int, default=18)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    path = build(args.source, args.tiles, args.out, args.in_block)
    cfg = load_defaults()
    profiler = cProfile.Profile() if args.profile else None
    started = time.perf_counter()
    if profiler:
        profiler.enable()
    result = convert(path, args.out / "result", cfg, 3000)
    if profiler:
        profiler.disable()
    total = time.perf_counter() - started
    seconds = take_record(threading.get_ident()).seconds
    print(f"prims={len(result.pre.primitives)} walls={len(result.ir.walls)} columns={len(result.ir.columns)} "
          f"total={total:.1f}s")
    print(" ".join(f"{k}={v:.1f}" for k, v in sorted(seconds.items(), key=lambda x: -x[1])))
    if profiler:
        pstats.Stats(profiler).sort_stats("tottime").print_stats(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
