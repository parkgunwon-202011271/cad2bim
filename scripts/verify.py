"""설치 확인. 공개 도면(주민센터)을 변환해 기준값과 비교한다.

사용: python scripts/verify.py
기준값은 메인 PC(2026-11-18, LibreDWG 0.14.8829)에서 얻은 값이다.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import jobs  # noqa: E402
from app.plans2d import render_inputs  # noqa: E402

SOURCE = jobs.PROJECT / "data" / "public" / "주민센터"
EXPECTED_FLOORS = 3
EXPECTED = {"walls": "90", "columns": "68", "openings": "72", "doors": "35", "windows": "25", "spaces": "31"}
EXPECTED_DAE_SHA256 = "63a8f7939bb5f018"  # 앞 16자리. 다르면 숫자는 같아도 형상이 조금 다를 수 있다


def main() -> int:
    print("주민센터 변환 중 (1분 안팎)...")
    out_dir = jobs.run_build(SOURCE, lambda event: None)
    summary = jobs.parse_summary((out_dir / jobs.APP_DIR_NAME / "build.txt").read_text(encoding="utf-8"))
    ok = True

    floors = len(summary["floors"])
    ok &= _check("층 수", floors, EXPECTED_FLOORS)
    for key, value in EXPECTED.items():
        ok &= _check(key, summary["totals"].get(key), value)

    digest = hashlib.sha256((out_dir / "building.dae").read_bytes()).hexdigest()[:16]
    same = digest == EXPECTED_DAE_SHA256
    print(f"  {'같음' if same else '다름'}  3D 모델 파일 해시 {digest} (기준 {EXPECTED_DAE_SHA256})")

    plans = render_inputs(out_dir)
    ok &= _check("2D 도면 그림", len(plans), 3)

    print("재현 성공" if ok else "재현 실패: 위에서 '다름'인 항목을 확인하세요.")
    return 0 if ok else 1


def _check(label: str, got, expected) -> bool:
    same = str(got) == str(expected)
    print(f"  {'같음' if same else '다름'}  {label}: {got} (기준 {expected})")
    return same


if __name__ == "__main__":
    raise SystemExit(main())
