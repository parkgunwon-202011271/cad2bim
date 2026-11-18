"""DWG → DXF 변환 어댑터. DXF 입력이 표준 경로이고 DWG는 이 뒤에 숨긴다.

- LibreDWG(GPL-3.0): 기본값. 별도 프로세스로 호출한다. 재현을 위해 고정 버전을 저장소의
  third_party/libredwg에 함께 두고 가장 먼저 쓴다. 고객에게 함께 배포하려면 GPL 의무를 검토한다.
- ODA File Converter: 비회원은 비상업 용도만 허용된다 (상세설계 9장). 사용자가 설치했을 때만 쓴다.

두 변환기 모두 임시 폴더의 input.dwg로 호출한다. 큰 폴더를 훑지 않고, 파일명의 한글과
특수문자가 변환기에 전달되지 않게 하기 위해서다. 종료 코드는 믿지 않고 결과 파일로 판단한다.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from ..errors import Cad2BimError, ConverterNotAvailableError

_TIMEOUT_S = 600
_DEFAULT_LIBREDWG_DIR = r"C:\libredwg"
# 저장소에 함께 둔 고정 버전. PC마다 변환 결과가 같도록 설치본보다 먼저 쓴다.
_BUNDLED_LIBREDWG = Path(__file__).resolve().parents[4] / "third_party" / "libredwg" / "dwg2dxf.exe"


class DwgConverter(Protocol):
    name: str

    def to_dxf(self, dwg: Path, out_dir: Path) -> Path: ...


def _installed_libredwg() -> str | None:
    if _BUNDLED_LIBREDWG.is_file():
        return str(_BUNDLED_LIBREDWG)
    pattern = os.path.join(_DEFAULT_LIBREDWG_DIR, "**", "dwg2dxf.exe")
    found = sorted(glob.glob(pattern, recursive=True))
    return found[0] if found else None


def _installed_oda() -> str | None:
    """기본 설치 위치에서 찾는다. 이 프로그램은 설치해도 PATH에 들어가지 않는다."""
    for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        if root:
            found = sorted(glob.glob(os.path.join(root, "ODA", "ODAFileConverter*", "ODAFileConverter.exe")))
            if found:
                return found[-1]
    return None


class _TempFolderConverter:
    def _command(self, source: Path, target: Path) -> list[str]:
        raise NotImplementedError

    def to_dxf(self, dwg: Path, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "in", Path(tmp) / "out"
            source.mkdir()
            target.mkdir()
            shutil.copyfile(dwg, source / "input.dwg")
            subprocess.run(self._command(source, target), capture_output=True, timeout=_TIMEOUT_S)
            produced = target / "input.dxf"
            if not produced.is_file() or produced.stat().st_size == 0:
                raise Cad2BimError(f"DWG 변환 결과가 없습니다: {dwg.name}")
            out = out_dir / (dwg.stem + ".dxf")
            shutil.move(str(produced), out)
        return out


class LibreDwgConverter(_TempFolderConverter):
    name = "libredwg"

    def __init__(self, executable: str):
        self.executable = executable

    def _command(self, source: Path, target: Path) -> list[str]:
        return [self.executable, "-y", "-o", str(target / "input.dxf"), str(source / "input.dwg")]


class OdaFileConverter(_TempFolderConverter):
    name = "oda"

    def __init__(self, executable: str, version: str = "ACAD2018"):
        self.executable = executable
        self.version = version

    def _command(self, source: Path, target: Path) -> list[str]:
        # 인자 순서: 입력 폴더, 출력 폴더, 버전, 형식, 재귀 여부, 감사 여부, 파일 필터
        return [self.executable, str(source), str(target), self.version, "DXF", "0", "1", "*.DWG"]


def find_converter() -> DwgConverter | None:
    """라이선스 제약이 적은 LibreDWG를 먼저 찾는다."""
    libredwg = (
        os.environ.get("LIBREDWG_DWG2DXF") or _installed_libredwg() or shutil.which("dwg2dxf")
    )
    if libredwg:
        return LibreDwgConverter(libredwg)
    oda = (
        os.environ.get("ODA_FILE_CONVERTER") or shutil.which("ODAFileConverter") or _installed_oda()
    )
    return OdaFileConverter(oda) if oda else None


def require_converter() -> DwgConverter:
    converter = find_converter()
    if converter is None:
        raise ConverterNotAvailableError(
            "DWG를 읽을 변환기가 없습니다. LibreDWG를 C:\\libredwg에 풀거나 "
            "LIBREDWG_DWG2DXF에 dwg2dxf.exe 경로를 지정하세요."
        )
    return converter
