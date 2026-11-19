""".env 파일 읽기. API 키 같은 비밀 값을 코드·설정·저장소 밖에 두기 위해서다.

저장소 맨 위의 .env를 읽어 환경 변수로 넣는다. 이미 설정된 환경 변수가 먼저다(덮어쓰지 않는다).
형식은 한 줄에 '이름=값'. 빈 줄과 #으로 시작하는 줄은 건너뛴다. 값의 앞뒤 따옴표는 뗀다.
.env는 .gitignore에 있어 올라가지 않는다. 양식은 .env.example에 있다.
"""
from __future__ import annotations

import os
from pathlib import Path

NAME = ".env"


def find_env_file(start: Path | None = None) -> Path | None:
    """start(없으면 이 파일 위치)에서 위로 올라가며 .env를 찾는다. .git이 있는 곳(저장소 맨 위)에서 멈춘다."""
    here = Path(start or Path(__file__)).resolve()
    for folder in [here, *here.parents]:
        candidate = folder / NAME
        if candidate.is_file():
            return candidate
        if (folder / ".git").exists():
            return None
    return None


def load_env(path: Path | None = None) -> list[str]:
    """.env의 값을 환경 변수로 넣고, 넣은 이름을 돌려준다(값은 돌려주지 않는다)."""
    path = path or find_env_file()
    if path is None or not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if not name or not value or name in os.environ:
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded
