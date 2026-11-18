"""DXF 읽기. 인코딩은 ezdxf가 $DWGCODEPAGE와 파일 버전을 보고 처리한다."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import ezdxf
from ezdxf import recover
from ezdxf.document import Drawing

from ..errors import Cad2BimError


@dataclass
class DxfSource:
    path: Path
    sha256: str
    doc: Drawing | None  # 캐시에서 읽으면 None이다. 전개 후에는 쓰지 않는다
    insunits: int
    recovered: bool = False


def read_dxf(path: Path) -> DxfSource:
    path = Path(path)
    if not path.is_file():
        raise Cad2BimError(f"파일이 없습니다: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    recovered = False
    try:
        doc = ezdxf.readfile(path)
    except OSError as exc:
        raise Cad2BimError(f"DXF 파일이 아닙니다: {path.name}") from exc
    except ezdxf.DXFStructureError:
        try:
            doc, auditor = recover.readfile(path)
        except ezdxf.DXFStructureError as exc:
            raise Cad2BimError(f"DXF를 읽을 수 없습니다: {path.name} ({exc})") from exc
        if auditor.has_errors:
            raise Cad2BimError(f"DXF 손상을 복구하지 못했습니다: {path.name}")
        recovered = True
    return DxfSource(path, digest, doc, int(doc.header.get("$INSUNITS", 0)), recovered)
