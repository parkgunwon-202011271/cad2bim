"""화면(JS)에서 부르는 Python API. pywebview가 window.pywebview.api로 연결한다.

밑줄로 시작하지 않는 메서드만 JS에 열린다. 오래 걸리는 일은 스레드에서 하고,
진행 상황은 window.cad2bimEvent(...)로 화면에 보낸다.
"""
from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

import webview

from . import jobs
from .plans2d import render_inputs


class Api:
    def __init__(self) -> None:
        self._window: webview.Window | None = None
        self._busy = threading.Lock()
        self._result: Path | None = None

    def _bind(self, window: webview.Window) -> None:
        self._window = window

    def _emit(self, event: dict) -> None:
        if self._window is not None:
            payload = json.dumps(event, ensure_ascii=False)
            self._window.evaluate_js(f"window.cad2bimEvent && window.cad2bimEvent({payload})")

    # ---- JS에서 부르는 메서드 ----

    def choose_folder(self) -> bool:
        """도면 폴더를 골라 변환한다. 끌어다 놓기가 안 될 때의 대안이다."""
        picked = self._pick_folder()
        if picked:
            self.convert_paths([picked])
        return bool(picked)

    def open_result(self) -> bool:
        """이미 변환한 폴더를 연다. 입력 폴더나 _cad2bim_out 어느 쪽이든 된다."""
        picked = self._pick_folder()
        if not picked:
            return False
        try:
            self._result = jobs.resolve_result(Path(picked))
        except ValueError as exc:
            self._emit({"type": "error", "message": str(exc)})
            return False
        self._emit({"type": "done"})
        return True

    def convert_paths(self, paths: list[str]) -> None:
        if not self._busy.acquire(blocking=False):
            self._emit({"type": "error", "message": "이미 변환 중입니다. 끝난 뒤 다시 놓아 주세요."})
            return
        threading.Thread(target=self._convert, args=(paths,), daemon=True).start()

    def get_model(self) -> dict:
        """마지막 결과의 3D 모델, 요약, 2D 도면 그림."""
        if self._result is None:
            return {}
        out_dir = self._result
        build = out_dir / jobs.APP_DIR_NAME / "build.txt"
        summary = jobs.parse_summary(build.read_text(encoding="utf-8")) if build.is_file() else {}
        try:
            plans = [
                {"name": p["name"], "src": _data_url(p["path"])} for p in render_inputs(out_dir)
            ]
        except Exception:  # 2D 그림이 없어도 3D는 보여 준다
            plans = []
        return {
            "dae": (out_dir / "building.dae").read_text(encoding="utf-8"),
            "summary": summary,
            "plans": plans,
        }

    # ---- 내부 ----

    def _pick_folder(self) -> str | None:
        if self._window is None:
            return None
        picked = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        return picked[0] if picked else None

    def _convert(self, paths: list[str]) -> None:
        try:
            folder = jobs.prepare_input(paths)
            self._emit({"type": "started", "count": jobs.count_drawings(folder)})
            self._result = jobs.run_build(folder, self._emit)
            self._emit({"type": "progress", "stage": "plans"})
            render_inputs(self._result)
            self._emit({"type": "done"})
        except Exception as exc:  # 화면에 보일 문장만 넘긴다
            message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
            self._emit({"type": "error", "message": message})
        finally:
            self._busy.release()


def _data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
