"""cad2bim 데스크톱 앱. 실행: python -m app

화면은 app/web(HTML/CSS/JS)이고, 창은 pywebview(Windows에서는 Edge WebView2)다.
외부 통신은 없다. three.js도 app/web/vendor에 함께 둔다.
"""
from __future__ import annotations

from pathlib import Path

import webview
from webview.dom import DOMEventHandler

from .api import Api

WEB = Path(__file__).resolve().parent / "web"


def _on_drop(api: Api, event: dict) -> None:
    files = event.get("dataTransfer", {}).get("files", [])
    paths = [f["pywebviewFullPath"] for f in files if f.get("pywebviewFullPath")]
    api.convert_paths(paths)


def _ignore(_event: dict) -> None:
    pass


def _bind(window: webview.Window, api: Api) -> None:
    api._bind(window)
    # 끌어다 놓은 파일의 실제 경로는 Python 쪽 drop 처리기에서만 받을 수 있다.
    window.dom.document.events.dragover += DOMEventHandler(_ignore, True, True)
    window.dom.document.events.drop += DOMEventHandler(lambda e: _on_drop(api, e), True, True)


def main() -> None:
    api = Api()
    window = webview.create_window(
        "cad2bim", str(WEB / "index.html"), js_api=api, width=1440, height=900, min_size=(960, 640),
    )
    webview.start(_bind, (window, api), http_server=True)


if __name__ == "__main__":
    main()
