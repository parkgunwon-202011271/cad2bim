"""LLM 호출 한 곳. 레이어 판단과 검토가 함께 쓴다. 공급자는 anthropic(Claude), gemini, openai다.

키는 환경 변수에서만 읽는다(ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY). 저장소 맨 위 .env에 적어 두면
실행할 때 환경 변수로 읽는다(envfile.py, .gitignore 대상). 코드·설정 파일에는 적지 않는다.
openai는 OpenAI 호환 주소(회사 게이트웨이, OCI 등)를 OPENAI_BASE_URL로 받는다. 모델은 OPENAI_MODEL이 설정보다 먼저다.
답은 JSON 하나를 기대한다. 꺼내지 못하면 None을 돌려주고, 부르는 쪽은 그 지점을 대기 목록에 남긴다.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from .errors import Cad2BimError

PROVIDERS = ("anthropic", "gemini", "openai")
_RETRIES = 4  # 일시적인 실패(과부하·분당 한도)는 잠시 쉬고 다시 묻는다
_GEMINI: Any = None
_OPENAI: Any = None
_LAST_CALL = 0.0
_PACE_S = 0.0  # 호출 사이 최소 간격(초). 분당 한도를 넘지 않게 configure()로 정한다


class QuotaExhausted(Cad2BimError):
    """한도(하루 한도 등)가 다 됐다. 부르는 쪽은 더 묻지 않고, 남은 지점은 대기 목록에 남긴다."""


def configure(cfg: dict[str, Any]) -> None:
    """설정의 분당 호출 한도로 호출 간격을 정한다(llm.requests_per_minute)."""
    global _PACE_S
    rpm = float(os.environ.get("CAD2BIM_LLM_RPM") or (cfg.get("llm") or {}).get("requests_per_minute") or 0)  # 환경 변수가 먼저
    _PACE_S = 60.0 / rpm * 1.05 if rpm > 0 else 0.0


def default_model(cfg: dict[str, Any], provider: str) -> str:
    env = {"openai": "OPENAI_MODEL", "gemini": "GEMINI_MODEL", "anthropic": "ANTHROPIC_MODEL"}.get(provider)
    if env and os.environ.get(env):
        return os.environ[env]
    llm = cfg.get("llm") or {}
    models = llm.get("models") or {}
    if provider in models:
        return str(models[provider])
    if llm.get("model") and llm.get("provider", "anthropic") == provider:
        return str(llm["model"])
    raise Cad2BimError(f"{provider} 모델 이름이 없습니다. .env의 {provider.upper()}_MODEL 또는 설정 llm.models에 적으세요.")


def extract_json(text: str) -> Any:
    """답에서 JSON 객체 하나를 꺼낸다. 앞뒤 설명이나 코드 블록 표시가 붙어 있어도 된다."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def ask(provider: str, model: str, prompt: str, image_png: bytes | None = None, max_tokens: int = 2000) -> str:
    """질문(과 그림 한 장)을 보내고 답 글을 돌려준다."""
    if provider == "anthropic":
        call = lambda: _ask_anthropic(model, prompt, image_png, max_tokens)  # noqa: E731
    elif provider == "gemini":
        call = lambda: _ask_gemini(model, prompt, image_png, max_tokens)  # noqa: E731
    elif provider == "openai":
        call = lambda: _ask_openai(model, prompt, image_png, max_tokens)  # noqa: E731
    else:
        raise Cad2BimError(f"LLM 공급자를 알 수 없습니다: {provider} ({', '.join(PROVIDERS)})")
    global _LAST_CALL
    for attempt in range(_RETRIES):
        wait = _PACE_S - (time.monotonic() - _LAST_CALL)
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL = time.monotonic()
        try:
            return call()
        except Cad2BimError:
            raise
        except Exception as exc:  # 네트워크·한도 오류
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            text = str(exc)
            if code == 429 and ("PerDay" in text or "per day" in text.lower()):
                raise QuotaExhausted(f"LLM 하루 한도가 다 됐습니다({provider}:{model}). 남은 지점은 대기 목록에 둡니다.") from exc
            permanent = isinstance(code, int) and 400 <= code < 500 and code != 429  # 요청이 잘못됐다: 다시 물어도 같다
            if permanent:
                raise Cad2BimError(f"LLM 호출에 실패했습니다({provider}): {type(exc).__name__}: {text[:200]}") from exc
            if attempt == _RETRIES - 1:
                if code == 429:  # 분당 한도를 계속 넘는다: 이번 빌드에서는 그만 묻는다
                    raise QuotaExhausted(f"LLM 한도에 계속 걸립니다({provider}:{model}): {text[:150]}") from exc
                raise Cad2BimError(f"LLM 호출에 실패했습니다({provider}): {type(exc).__name__}: {text[:200]}") from exc
            time.sleep(30 if code == 429 else 2 ** attempt * 5)  # 분당 한도는 다음 분까지 기다린다
    raise AssertionError("unreachable")


def _ask_anthropic(model: str, prompt: str, image_png: bytes | None, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise Cad2BimError("Claude를 쓰려면 anthropic 패키지가 필요합니다: pip install anthropic") from exc
    import base64

    content: list[dict[str, Any]] = []
    if image_png is not None:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                     "data": base64.b64encode(image_png).decode("ascii")}})
    content.append({"type": "text", "text": prompt})
    response = anthropic.Anthropic().messages.create(
        model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason in ("max_tokens", "refusal"):
        raise Cad2BimError(f"LLM 답이 끝나지 않았습니다: {response.stop_reason}")
    return "".join(block.text for block in response.content if block.type == "text")


def _ask_gemini(model: str, prompt: str, image_png: bytes | None, max_tokens: int) -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise Cad2BimError("Gemini를 쓰려면 google-genai 패키지가 필요합니다: pip install google-genai") from exc
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise Cad2BimError("Gemini 키가 없습니다. 환경 변수 GEMINI_API_KEY를 설정하세요.")
    contents: list[Any] = []
    if image_png is not None:
        contents.append(types.Part.from_bytes(data=image_png, mime_type="image/png"))
    contents.append(prompt)
    global _GEMINI
    if _GEMINI is None:  # 한 번 만들어 계속 쓴다. 임시 객체로 부르면 요청 전에 연결이 닫힌다
        _GEMINI = genai.Client()
    response = _GEMINI.models.generate_content(
        model=model, contents=contents,
        # JSON 전용 응답 옵션은 SDK 기본 API(v1)가 받지 않는다. 질문에서 JSON만 요구하고 답에서 꺼낸다.
        config=types.GenerateContentConfig(max_output_tokens=max_tokens, temperature=0.0),
    )
    return response.text or ""


def _ask_openai(model: str, prompt: str, image_png: bytes | None, max_tokens: int) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise Cad2BimError("OpenAI 호환 API를 쓰려면 openai 패키지가 필요합니다: pip install openai") from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise Cad2BimError("OpenAI 키가 없습니다. .env 또는 환경 변수 OPENAI_API_KEY를 설정하세요.")
    import base64

    global _OPENAI
    if _OPENAI is None:
        _OPENAI = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL") or None)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_png is not None:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")}})
    # 답 길이 한도는 넘기지 않는다. 모델·게이트웨이마다 받는 이름(max_tokens, max_completion_tokens)이 달라서다.
    response = _OPENAI.chat.completions.create(model=model, messages=[{"role": "user", "content": content}])
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise Cad2BimError("LLM 답이 끝나지 않았습니다: length")
    return choice.message.content or ""
