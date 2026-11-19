"""전체 변환 흐름. 평면도 한 장을 IR과 보고서로 만든다."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import config_dir, load_layer_dictionary
from .io import require_converter
from .ir import IR, Level, SourceInfo
from .layers import DictionaryClassifier, LayerMap, apply_overrides, drop_weak_walls, promote_inferred_walls
from .layers.learned import apply_learned_roles
from .layers.llm import apply_llm_roles
from .preprocess import Preprocessed, preprocess
from .preprocess.primitive import BBox
from .recognize import Recognition, recognize
from .recognize.walls import infer_wall_layers, infer_walls_by_shape
from .report import write_report
from .timing import stage


@dataclass
class ConvertResult:
    ir: IR
    pre: Preprocessed
    ir_path: Path
    report_path: Path
    layers: LayerMap | None = None
    recognition: Recognition | None = None


def build_ir(pre: Preprocessed, height: float, recognition: Recognition | None = None) -> IR:
    found = {}
    warnings = list(pre.warnings)
    if recognition is not None:
        found = {
            "walls": recognition.walls,
            "columns": recognition.columns,
            "openings": recognition.openings,
            "rooms": recognition.rooms,
            "stairs": recognition.stairs,
            "voids": recognition.voids,
            "rails": recognition.rails,
            "unsupported": recognition.unsupported,
        }
        warnings += recognition.warnings
    return IR(
        source=SourceInfo(
            file=pre.source.path.name,
            sha256=pre.source.sha256,
            region=pre.region,
            unit_scale=pre.scale.factor,
            unit_source=pre.scale.source,
        ),
        origin_offset=pre.origin_offset,
        level=Level(height=height),
        warnings=warnings,
        **found,
    )


def convert(
    path: Path,
    out_dir: Path,
    cfg: dict[str, Any],
    floor_height: float,
    region: BBox | None = None,
    unit_scale: float | None = None,
    layer_map: Path | None = None,
    cache_dir: Path | None = None,
    llm_dir: Path | None = None,
) -> ConvertResult:
    path = Path(path)
    if path.suffix.lower() == ".dwg":
        with tempfile.TemporaryDirectory() as tmp:
            dxf = require_converter().to_dxf(path, Path(tmp))
            pre = preprocess(dxf, cfg, region, unit_scale, cache_dir)
    else:
        pre = preprocess(path, cfg, region, unit_scale, cache_dir)

    dictionary = load_layer_dictionary()
    layers = DictionaryClassifier(dictionary).classify(pre.primitives)
    # 켜져 있을 때만 LLM이 레이어 역할을 정한다. 답은 llm_dir에 저장해 다시 쓴다. 사용자 수정이 그보다 앞선다.
    apply_llm_roles(layers, llm_dir if llm_dir is not None else out_dir.parent / "_llm", cfg, pre.source.sha256[:16])
    if layer_map is not None:
        layers = apply_overrides(layers, layer_map)
    with stage("infer_layers"):
        drop_weak_walls(layers, pre.primitives, cfg["walls"]["weak_wall_max_share"], cfg["walls"]["weak_wall_min_length_mm"])
        promote_inferred_walls(layers, infer_wall_layers(pre.primitives, layers, cfg["walls"]))
        layers.shape_walls = infer_walls_by_shape(pre.primitives, layers, cfg["walls"])
        # 모양으로도 벽을 못 찾은 '모름' 레이어만 학습 예시로 채운다(켜져 있을 때만, 비용 없음).
        # 먼저 채우면 '모름'이 '기타'가 되어 모양으로 벽을 찾는 단계가 그 레이어를 못 본다(물류 벽 1,198→1,029).
        apply_learned_roles(layers, cfg, config_dir(), skip=layers.shape_walls)
    recognition = recognize(pre.primitives, layers, cfg, dictionary["hatch_patterns"])

    ir = build_ir(pre, floor_height, recognition)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layer_map.json").write_text(
        json.dumps(layers.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ir_path = out_dir / "ir.json"
    ir_path.write_text(
        json.dumps(ir.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = out_dir / "report.html"
    with stage("report"):
        write_report(report_path, pre, ir)
    return ConvertResult(ir, pre, ir_path, report_path, layers, recognition)
