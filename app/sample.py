"""브라우저 미리보기용 샘플(app/web/sample)을 만든다. 공개 도면(주민센터)만 쓴다.

사용: python -m app.sample
"""
from __future__ import annotations

import json
import shutil
import sys

from . import jobs
from .plans2d import render_inputs

SOURCE = jobs.PROJECT / "data" / "public" / "주민센터"
TARGET = jobs.PROJECT / "app" / "web" / "sample"


def main() -> int:
    def show(event: dict) -> None:
        print(event, file=sys.stderr)

    out_dir = jobs.run_build(SOURCE, show)
    plans = render_inputs(out_dir)
    if TARGET.exists():
        shutil.rmtree(TARGET)
    (TARGET / "plans").mkdir(parents=True)
    shutil.copy2(out_dir / "building.dae", TARGET / "building.dae")
    if (out_dir / "furniture.json").is_file():
        shutil.copy2(out_dir / "furniture.json", TARGET / "furniture.json")
    meta_plans = []
    for plan in plans:
        name = f"plans/{plan['path'].name}"
        shutil.copy2(plan["path"], TARGET / name)
        meta_plans.append({"name": plan["name"], "file": name})
    summary = jobs.parse_summary((out_dir / jobs.APP_DIR_NAME / "build.txt").read_text(encoding="utf-8"))
    (TARGET / "summary.json").write_text(
        json.dumps({"summary": summary, "plans": meta_plans}, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"샘플 저장: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
