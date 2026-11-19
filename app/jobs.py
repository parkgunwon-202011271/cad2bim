"""앱의 변환 작업. 코어 CLI(`cad2bim.cli build`)를 별도 프로세스로 실행하고 진행 상황을 읽는다.

코어는 고치지 않는다. 앱은 입력 폴더를 준비하고, 출력 글자를 읽어 화면에 넘길 뿐이다.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

OUT_DIR_NAME = "_cad2bim_out"
APP_DIR_NAME = "_app"
SUFFIXES = {".dwg", ".dxf"}
PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT / "core"

_PROGRESS = re.compile(r"진행 (\d+)/(\d+)")
_PAIR = re.compile(r"([A-Za-z_]+)=([\w.\-]+)")

Emit = Callable[[dict], None]


def jobs_root() -> Path:
    """파일을 끌어다 놓았을 때 모아 둘 곳. 프로젝트와 OneDrive 밖의 로컬 폴더다."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "cad2bim" / "jobs"


def prepare_input(paths: list[str]) -> Path:
    """폴더 하나는 그대로 쓰고, 파일은 작업 폴더에 모은다. 코어 build는 폴더만 받는다."""
    items = [Path(p) for p in paths if p]
    if not items:
        raise ValueError("끌어다 놓은 항목의 경로를 받지 못했습니다. '폴더 열기'를 써 주세요.")
    folders = [p for p in items if p.is_dir()]
    if folders:
        if len(items) > 1:
            raise ValueError("폴더는 하나만 놓아 주세요. 여러 도면은 파일로 함께 놓으면 됩니다.")
        return folders[0].resolve()
    files = [p for p in items if p.is_file() and p.suffix.lower() in SUFFIXES]
    if not files:
        raise ValueError("DWG 또는 DXF 파일이 없습니다.")
    if len({f.name.casefold() for f in files}) != len(files):
        raise ValueError("이름이 같은 도면 파일이 있습니다. 파일 이름을 다르게 바꾼 뒤 다시 놓아 주세요.")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job = jobs_root() / stamp
    n = 1
    while job.exists():
        n += 1
        job = jobs_root() / f"{stamp}-{n}"
    job.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, job / f.name)
    return job


def count_drawings(folder: Path) -> int:
    return sum(
        1 for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUFFIXES and OUT_DIR_NAME not in p.parts
    )


def parse_summary(text: str) -> dict:
    """build 출력에서 화면에 보일 숫자만 고른다. 이름과 경로는 출력에 없다."""
    floors, totals = [], {}
    for line in text.splitlines():
        m = re.match(r"floor\[(\w+)\]:\s*(.*)", line)
        if m:
            floors.append({"key": m.group(1), **dict(_PAIR.findall(m.group(2)))})
        elif line.startswith("ifc:"):
            totals = dict(_PAIR.findall(line))
    return {"floors": floors, "totals": totals}


def run_build(folder: Path, emit: Emit) -> Path:
    """build를 실행하고 결과 폴더를 돌려준다. 실패하면 RuntimeError."""
    jobs = max(1, min(4, (os.cpu_count() or 2) - 1))
    cmd = [sys.executable, "-m", "cad2bim.cli", "build", str(folder), "--jobs", str(jobs)]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    # LLM은 CAD2BIM_LLM(.env 또는 환경 변수)로 직접 켤 때만 쓴다. 키가 있다고 자동으로 켜지 않는다.
    # 기본은 규칙 기반(cache: 저장된 답이 있으면 쓰고, 외부로 아무것도 보내지 않는다).
    if not env.get("CAD2BIM_LLM"):
        env["CAD2BIM_LLM"] = "cache"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        cmd, cwd=CORE, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags,
    )
    errors: list[str] = []

    def read_stderr() -> None:
        # 진행 표시는 \r로 덮어쓰므로 줄바꿈과 \r을 모두 경계로 본다.
        buf = b""
        assert proc.stderr is not None
        while chunk := proc.stderr.read(64):
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                handle(raw.decode("utf-8", "replace").strip())
        handle(buf.decode("utf-8", "replace").strip())

    def handle(line: str) -> None:
        if not line:
            return
        m = _PROGRESS.search(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            emit({"type": "progress", "stage": "read", "done": done, "total": total})
            if done == total:
                emit({"type": "progress", "stage": "model"})
        else:
            errors.append(line)

    reader = threading.Thread(target=read_stderr, daemon=True)
    reader.start()
    assert proc.stdout is not None
    stdout = proc.stdout.read().decode("utf-8", "replace")
    proc.wait()
    reader.join(timeout=5)
    if proc.returncode != 0:
        # 트레이스백에는 경로가 있을 수 있어 마지막 '오류:' 줄만 보인다.
        message = next((e for e in reversed(errors) if e.startswith("오류:")), None)
        raise RuntimeError(message or f"변환에 실패했습니다 (종료 코드 {proc.returncode}).")
    out_dir = folder / OUT_DIR_NAME
    app_dir = out_dir / APP_DIR_NAME
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "build.txt").write_text(stdout, encoding="utf-8")
    return out_dir


def load_review(out_dir: Path) -> dict:
    """검토 패널에 보일 지점 목록(_llm/review.json). 없거나 읽지 못하면 빈 목록. 그림 경로는 결과 폴더 기준이다."""
    path = Path(out_dir) / "_llm" / "review.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"answers": [], "items": []}
    # review.json의 답은 마지막 빌드에 적용된 것이다. 그 뒤 사람이 고친 답(decisions.json)을 겹쳐,
    # 지금 답(decision)과 모델에 반영된 답(applied)이 다르면 화면이 '적용 안 된 변경'으로 보여 준다.
    current = _read_decisions(out_dir)
    items = []
    for item in data.get("items", []):
        image = Path(out_dir) / "_llm" / item["image"] if item.get("image") else None
        items.append({
            **item, "image": image if image is not None and image.is_file() else None,
            "applied": (item.get("decision") or {}).get("answer"), "decision": current.get(item["key"]),
        })
    return {"answers": data.get("answers", []), "items": items}


def _read_decisions(out_dir: Path) -> dict:
    try:
        data = json.loads((Path(out_dir) / "_llm" / "decisions.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(data.get("decisions", {}))


def set_review_decision(out_dir: Path, key: str, answer: str | None) -> dict | None:
    """검토 지점의 답을 사람이 정한다(model: "user"). answer가 None이면 사람 답을 지우고 AI 답으로 되돌린다.

    AI가 골랐던 답은 ai에 남겨 화면에서 'AI: 문 → 사람: 벽'처럼 보여 준다. 저장 형식은 코어(review.py)와 같다.
    모델에는 다음 빌드에서 반영된다. 바뀐 뒤의 답(없으면 None)을 돌려준다. 모르는 지점·답이면 ValueError.
    """
    review = load_review(out_dir)
    if key not in {item["key"] for item in review["items"]}:
        raise ValueError("이 결과에 없는 검토 지점입니다.")
    if answer is not None and answer not in review["answers"]:
        raise ValueError("고를 수 없는 답입니다.")
    decisions = _read_decisions(out_dir)
    now = decisions.get(key)
    ai = now.get("ai") if now and now.get("model") == "user" else now
    if answer is None:
        entry = ai
    else:
        entry = {"answer": answer, "reason": "", "model": "user"}
        if ai:
            entry["ai"] = {k: ai.get(k, "") for k in ("answer", "reason", "model")}
    if entry:
        decisions[key] = entry
    else:
        decisions.pop(key, None)
    directory = Path(out_dir) / "_llm"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "decisions.json").write_text(
        json.dumps({"decisions": dict(sorted(decisions.items()))}, ensure_ascii=False, indent=1), encoding="utf-8",
    )
    return entry


def resolve_result(folder: Path) -> Path:
    """입력 폴더나 결과 폴더 어느 쪽을 골라도 결과 폴더를 찾는다."""
    folder = Path(folder)
    for candidate in (folder, folder / OUT_DIR_NAME):
        if (candidate / "building.dae").is_file():
            return candidate
    raise ValueError("이 폴더에 변환 결과(building.dae)가 없습니다.")
