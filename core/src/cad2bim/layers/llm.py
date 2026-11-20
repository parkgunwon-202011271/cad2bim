"""LLM으로 레이어 역할 정하기 (처리 단계 3-a). 켜야만 쓰고, 답은 저장해 두고 다시 쓴다.

보내는 것은 외부참조 접두사를 뗀 레이어 이름과 통계(도형 수·종류·평균 선 길이·블록 이름)뿐이다.
좌표와 글자 내용은 보내지 않는다. LLM은 이름표만 붙이고 형상은 규칙이 계산한다.

켜는 법: config의 llm.enabled 또는 환경 변수 CAD2BIM_LLM.
- anthropic: 저장된 답에 없는 레이어만 Claude에 묻는다. ANTHROPIC_API_KEY와 anthropic 패키지가 필요하다.
- gemini: 같은 일을 Gemini에 묻는다. GEMINI_API_KEY와 google-genai 패키지가 필요하다.
- cache: 저장된 답만 쓴다. 외부 통신이 없다. 답이 없는 레이어는 사전 결과를 그대로 둔다.
- off: 끈다.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ..errors import Cad2BimError
from ..llm_client import PROVIDERS, QuotaExhausted, ask, configure, default_model, extract_json
from . import ROLES, LayerMap, base_name

CACHE_NAME = "layer_roles.json"
SOURCE = "llm"

_ROLE_GUIDE = """\
WALL: 건물 벽의 선(콘크리트·조적·패널·샌드위치패널·칸막이벽)
COLUMN: 기둥
DOOR: 문·셔터
WINDOW: 창·유리·커튼월
ROOM_TEXT: 실 이름·면적 글자
DIM: 치수
HATCH: 해치(벽·기둥 채움 무늬 등)
GRID: 그리드선·그리드 번호
STAIR: 건물 안의 계단. L- 접두사(조경) 레이어, 외부 계단·스탠드 좌석·화단 단 같은 조경 단은 STAIR가 아니라 OTHER
RAIL: 난간·파라펫
FURNITURE: 가구·위생기구·설비 기구
OTHER: 그 밖의 모든 것. 도곽·주기·기호·지시선·레벨 표시·마감선·단열선·재료 표시,
       그리고 건물 밖 토목·대지 요소(옹벽·경사로·주차·도로·등고선·조경)는 벽처럼 생겨도 OTHER
UNKNOWN: 이름과 통계로 판단할 수 없음"""

_PROMPT = """건축 CAD 평면도의 레이어 목록입니다. 각 레이어의 역할을 아래에서 하나씩 고르세요.

{guide}

레이어(이름: 통계). kinds는 도형 종류별 개수, mean_line_mm은 선의 평균 길이(mm), blocks는 속한 블록 이름입니다.
{layers}

JSON 객체 하나로만 답하세요. 키는 레이어 이름 그대로, 값은 역할 이름입니다. 예: {{"A-WALL": "WALL"}}"""


def mode(cfg: dict[str, Any]) -> str:
    """off | anthropic | gemini | cache. 환경 변수가 설정보다 앞선다."""
    env = os.environ.get("CAD2BIM_LLM", "").strip().lower()
    if env:
        value = env
    else:
        llm = cfg.get("llm") or {}
        value = str(llm.get("provider", "anthropic")).lower() if llm.get("enabled") else "off"
    if value not in ("off", "cache", "ml", *PROVIDERS):
        raise Cad2BimError(f"CAD2BIM_LLM 값을 알 수 없습니다: {value} (off, cache, ml, anthropic, gemini, openai)")
    return value


def layer_evidence(layer_map: LayerMap) -> dict[str, dict[str, Any]]:
    """접두사를 뗀 이름별로 통계를 합친다. 같은 레이어가 외부참조 여러 곳에서 와도 한 번만 묻는다."""
    merged: dict[str, dict[str, Any]] = {}
    for name, info in layer_map.layers.items():
        stats = info.stats or {}
        entry = merged.setdefault(base_name(name), {"count": 0, "kinds": {}, "line_total_mm": 0.0, "lines": 0, "blocks": set()})
        entry["count"] += stats.get("count", 0)
        for kind, n in (stats.get("kinds") or {}).items():
            entry["kinds"][kind] = entry["kinds"].get(kind, 0) + n
        lines = (stats.get("kinds") or {}).get("line", 0)
        if stats.get("mean_line_mm") and lines:
            entry["line_total_mm"] += stats["mean_line_mm"] * lines
            entry["lines"] += lines
        entry["blocks"].update(b for b in stats.get("blocks", []) if b)
    result = {}
    for name, e in sorted(merged.items()):
        result[name] = {
            "count": e["count"],
            "kinds": e["kinds"],
            "mean_line_mm": round(e["line_total_mm"] / e["lines"]) if e["lines"] else None,
            "blocks": sorted(base_name(b) for b in e["blocks"])[:5],
        }
    return result


def load_cache(directory: Path) -> dict[str, str]:
    path = Path(directory) / CACHE_NAME
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: role for name, role in data.get("roles", {}).items() if role in ROLES}


def save_cache(directory: Path, roles: dict[str, str], model: str) -> None:
    """다른 작업이 그사이 쓴 답과 합쳐 통째로 바꾼다(여러 도면을 동시에 처리하므로)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    merged = load_cache(directory)
    merged.update(roles)
    path = directory / CACHE_NAME
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    models = sorted(set(old.get("models", [])) | {model})
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"models": models, "roles": dict(sorted(merged.items()))}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def parse_roles(text: str, names: set[str]) -> dict[str, str]:
    """답에서 JSON 객체를 꺼낸다. 묻지 않은 이름과 모르는 역할은 버린다."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise Cad2BimError("LLM 답에 JSON이 없습니다.")
    data = json.loads(text[start:end + 1])
    return {name: str(role).upper() for name, role in data.items() if name in names and str(role).upper() in ROLES}


def ask_layers(evidence: dict[str, dict[str, Any]], provider: str, model: str, batch: int) -> dict[str, str]:
    """레이어를 batch개씩 묶어 묻는다. 꺼내지 못한 묶음은 건너뛴다(사전 결과가 남는다)."""
    names = sorted(evidence)
    roles: dict[str, str] = {}
    for i in range(0, len(names), batch):
        chunk = {n: evidence[n] for n in names[i:i + batch]}
        lines = "\n".join(f"{n}: {json.dumps(s, ensure_ascii=False)}" for n, s in chunk.items())
        try:
            text = ask(provider, model, _PROMPT.format(guide=_ROLE_GUIDE, layers=lines), max_tokens=16000)
        except QuotaExhausted:
            break  # 한도가 다 됐다: 사전 결과로 두고 다음에 다시 묻는다
        if extract_json(text) is None:
            continue
        roles.update(parse_roles(text, set(chunk)))
    return roles


_TEXT_KINDS = ("text",)
_SHAPE_KINDS = ("line", "polyline", "arc", "circle", "block_ref", "hatch", "solid", "ellipse", "spline")


def role_fits(role: str, stats: dict[str, Any]) -> bool:
    """LLM이 고른 역할이 레이어의 도형 종류와 정면으로 어긋나지 않는가(계산으로 거른다).

    작은 모델이 실제 계단 선 레이어(A-FLOR-STRS)를 '실 이름 글자'로 바꿔 계단이 사라진 일이 있었다.
    글자 역할은 글자가 절반 넘게, 치수는 치수가 있어야 하고, 형상 역할은 선·도형이 있어야 한다.
    """
    kinds = stats.get("kinds") or {}
    total = sum(kinds.values())
    if not total:
        return True  # 통계가 없으면 거를 근거가 없다
    text = sum(kinds.get(k, 0) for k in _TEXT_KINDS)
    shapes = sum(kinds.get(k, 0) for k in _SHAPE_KINDS)
    if role == "ROOM_TEXT":
        return text * 2 >= total
    if role == "DIM":
        return kinds.get("dimension", 0) > 0 or shapes > 0
    if role in ("WALL", "COLUMN", "DOOR", "WINDOW", "STAIR", "RAIL", "HATCH", "GRID", "FURNITURE"):
        return shapes > 0
    return True


def apply_llm_roles(layer_map: LayerMap, directory: Path, cfg: dict[str, Any], drawing: str | None = None) -> int:
    """LLM의 역할로 사전 결과를 바꾼다. 바꾼 레이어 수를 돌려준다. 꺼져 있으면 아무것도 안 한다.

    답은 레이어 이름으로 저장해 모든 도면에 쓴다. '0'처럼 도면마다 쓰임이 다른 레이어는
    '도면 해시 앞 16자:레이어' 키로 그 도면만의 답을 둘 수 있고, 있으면 그것이 먼저다.
    """
    which = mode(cfg)
    if which == "off":
        return 0
    llm = cfg.get("llm") or {}
    evidence = layer_evidence(layer_map)
    known = load_cache(directory)
    missing = {n: s for n, s in evidence.items() if n not in known}
    if missing and which in PROVIDERS:
        model = default_model(cfg, which)
        configure(cfg)
        answered = ask_layers(missing, which, model, int(llm.get("max_layers_per_call", 150)))
        save_cache(directory, answered, f"{which}:{model}")
        known.update(answered)
    changed = 0
    for name, info in layer_map.layers.items():
        role = known.get(f"{drawing}:{base_name(name)}") if drawing else None
        role = role or known.get(base_name(name))
        if role is None or role == "UNKNOWN" or not role_fits(role, info.stats or {}):
            continue
        if role != info.role:
            changed += 1
            if role != "WALL":
                info.material = None
        info.role, info.source = role, SOURCE
    return changed
