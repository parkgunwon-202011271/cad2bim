"""설치 확인. 공개 도면(주민센터)을 변환해 기준값과 비교한다.

사용: python scripts/verify.py
기준값은 메인 PC(2026-11-18, LibreDWG 0.14.8829)에서 얻은 값이다.
결과 폴더(_cad2bim_out)에 지난 인식 결과가 있으면 build가 그것을 다시 쓴다. 인식 규칙을 바꾼 뒤에는 지우고 돌린다.
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
# 2026-11-19: 문틀만 그린 문 자리 메우기, 짧은 창, ㄱ자 기둥 인식으로 벽 +4, 문 +4, 실 +3이 됐다.
# 같은 날 계단(계단참·경사판·구멍 위치)과 난간(바닥 위만)을 고쳐 3D 해시가 바뀌었다.
# 같은 날 칸막이 문을 바깥벽 문으로 잘못 붙이던 것을 고쳐 문 -6(그 자리는 창 +2), 승강로를 뚫고, 두 선으로 그린 난간을 하나로 합쳤다.
EXPECTED = {"walls": "94", "columns": "68", "openings": "71", "doors": "33", "windows": "27", "spaces": "34"}
EXPECTED_DAE_SHA256 = "e214ffe87551d693"  # 앞 16자리. 다르면 숫자는 같아도 형상이 조금 다를 수 있다


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
