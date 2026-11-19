"""레이어 분류. 사전으로 역할을 정하고, 모르면 UNKNOWN으로 두어 기하로 넘긴다."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..errors import Cad2BimError
from ..preprocess.primitive import Primitive

ROLES = frozenset({
    "WALL", "COLUMN", "DOOR", "WINDOW", "ROOM_TEXT", "DIM", "HATCH", "GRID", "STAIR", "RAIL",
    "FURNITURE", "OTHER", "UNKNOWN",
})


@dataclass
class LayerInfo:
    role: str
    source: str  # dictionary | user | llm | inferred | unresolved
    material: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerMap:
    layers: dict[str, LayerInfo]
    shape_walls: set[str] = field(default_factory=set)  # 이름이 없을 때 모양으로 고른 벽 후보 레이어

    def role(self, layer: str) -> str:
        info = self.layers.get(layer)
        return info.role if info else "UNKNOWN"

    def material(self, layer: str) -> str | None:
        info = self.layers.get(layer)
        return info.material if info else None

    @property
    def has_walls(self) -> bool:
        return any(i.role == "WALL" for i in self.layers.values())

    @property
    def has_named_walls(self) -> bool:
        """이름으로 벽을 찾았는가. 모양으로 찾은 도면은 문과 창 레이어도 이름을 믿을 수 없다."""
        return any(i.role == "WALL" and i.source != "inferred" for i in self.layers.values())

    def to_json(self) -> dict[str, Any]:
        return {
            "has_walls": self.has_walls,
            "shape_walls": sorted(self.shape_walls),
            "layers": {
                name: {"role": i.role, "source": i.source, "material": i.material, "stats": i.stats}
                for name, i in sorted(self.layers.items())
            },
        }


class LayerClassifier(Protocol):
    """MVP에서 LLM 구현을 추가한다. 입력은 마스킹한 레이어명과 통계로 제한한다."""

    def classify(self, prims: list[Primitive]) -> LayerMap: ...


def _stats(prims: list[Primitive]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Primitive]] = defaultdict(list)
    for p in prims:
        grouped[p.layer].append(p)
    result = {}
    for layer, members in grouped.items():
        lengths = []
        for p in members:
            if p.kind == "line":
                (x1, y1), (x2, y2) = p.geometry["p1"], p.geometry["p2"]
                lengths.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        result[layer] = {
            "count": len(members),
            "kinds": dict(Counter(p.kind for p in members)),
            "mean_line_mm": round(sum(lengths) / len(lengths), 1) if lengths else None,
            "blocks": sorted({p.block_path[-1] for p in members if p.block_path})[:10],
            "colors": sorted({p.color for p in members})[:10],
        }
    return result


_XREF_PREFIX = re.compile(r"^.*(?:\$\d+\$|\|)")


def base_name(layer: str) -> str:
    """결합된 외부참조의 접두사를 뗀다. 파일$0$A-WALL, 파일|A-WALL → A-WALL"""
    return _XREF_PREFIX.sub("", layer) or layer


class DictionaryClassifier:
    def __init__(self, dictionary: dict[str, Any]):
        self.roles = [(r["role"], re.compile(r["pattern"], re.I)) for r in dictionary["roles"]]
        self.materials = [(m["kind"], re.compile(m["pattern"], re.I)) for m in dictionary["materials"]]

    def classify(self, prims: list[Primitive]) -> LayerMap:
        layers = {}
        for name, stats in _stats(prims).items():
            short = base_name(name)
            role = next((r for r, pattern in self.roles if pattern.search(short)), None)
            material = None
            if role == "WALL":
                material = next((k for k, pattern in self.materials if pattern.search(short)), None)
            layers[name] = LayerInfo(role or "UNKNOWN", "dictionary" if role else "unresolved",
                                     material, stats)
        return LayerMap(layers)


def drop_weak_walls(layer_map: LayerMap, prims: list[Primitive], min_share: float, min_length_mm: float) -> set[str]:
    """이름만 벽이고 선이 거의 없는 레이어는 믿지 않는다. BLOCK처럼 우연히 맞은 이름 하나가 형상 판단을 막는다."""
    lengths: Counter = Counter()
    for p in prims:
        if p.kind == "line":
            (x1, y1), (x2, y2) = p.geometry["p1"], p.geometry["p2"]
            lengths[p.layer] += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        elif p.kind == "polyline":
            points = p.geometry["points"]
            lengths[p.layer] += sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(points, points[1:]))
    total = sum(lengths.values())
    named = [name for name, info in layer_map.layers.items() if info.role == "WALL"]
    # 사용자나 LLM이 통계를 보고 정한 벽은 선이 적어도 믿는다.
    weak = {
        name for name in named
        if layer_map.layers[name].source not in ("user", "llm") and lengths[name] < max(total * min_share, min_length_mm)
    }
    if len(weak) == len(named):  # 벽 레이어가 일부만 약하면 도면이 원래 그렇게 나뉜 것이다
        for name in weak:
            layer_map.layers[name].role = "UNKNOWN"
            layer_map.layers[name].source = "unresolved"
        return weak
    return set()


def promote_inferred_walls(layer_map: LayerMap, names: set[str]) -> None:
    """기하로 벽처럼 보이는 미분류 레이어를 벽으로 올린다. 이름 근거가 아니므로 표시해 둔다."""
    for name in names:
        info = layer_map.layers[name]
        if info.role == "UNKNOWN":
            info.role = "WALL"
            info.source = "inferred"


def apply_overrides(layer_map: LayerMap, path: Path) -> LayerMap:
    """사용자가 고친 layer_map.json을 반영한다. 도면에 없는 레이어는 무시한다."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, entry in data["layers"].items():
        current = layer_map.layers.get(name)
        if current is None:
            continue
        if entry["role"] not in ROLES:
            raise Cad2BimError(f"layer_map의 역할을 알 수 없습니다: {entry['role']}")
        if entry["role"] != current.role or entry.get("source") == "user":
            current.role = entry["role"]
            current.source = "user"
            current.material = entry.get("material", current.material)
    return layer_map
