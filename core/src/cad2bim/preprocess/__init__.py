"""DXF → 정규화된 Primitive 목록, 단위, 대상 영역."""
from __future__ import annotations

import hashlib
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import RegionAmbiguousError
from ..io import DxfSource, read_dxf
from ..timing import stage
from .explode import ExplodeStats, extract_primitives
from .normalize import NormalizeStats, normalize
from .primitive import BBox, Primitive, union_bbox
from .regions import line_cells, select_plan, strip_frames
from .units import ScaleDecision, decide_scale


@dataclass
class Preprocessed:
    source: DxfSource
    primitives: list[Primitive]
    excluded: list[Primitive]  # 도곽·표제란·평면 밖. 보고서에 흐리게 그린다
    scale: ScaleDecision
    origin_offset: tuple[float, float]
    plan_bbox: BBox | None
    region: BBox | None
    frames: list[BBox]
    explode_stats: ExplodeStats
    normalize_stats: NormalizeStats
    warnings: list[str] = field(default_factory=list)


def _origin_offset(bbox: BBox | None, cfg: dict[str, Any]) -> tuple[float, float]:
    """원점에서 먼 도면만 옮긴다. 가까운 도면은 원본 좌표를 유지한다."""
    if bbox is None:
        return (0.0, 0.0)
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    if max(abs(cx), abs(cy)) < cfg["far_threshold_mm"]:
        return (0.0, 0.0)
    step = cfg["offset_round_mm"]
    return (round(cx / step) * step, round(cy / step) * step)


def _code_fingerprint() -> str:
    """전개 코드가 바뀌면 캐시를 버린다."""
    digest = hashlib.sha256(sys.version.encode())
    here = Path(__file__).parent
    for name in ("explode.py", "primitive.py"):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()


def _read_and_explode(
    path: Path, sagitta: float, cache_dir: Path | None
) -> tuple[DxfSource, list[Primitive], ExplodeStats]:
    """읽기와 전개가 시간 대부분을 차지하므로 결과를 저장해 재실행 때 건너뛴다."""
    cache_file = None
    if cache_dir is not None:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        key = hashlib.sha256(f"{digest}|{_code_fingerprint()}|{sagitta}".encode()).hexdigest()[:24]
        cache_file = cache_dir / f"{key}.pkl"
        if cache_file.is_file():
            try:
                raw, stats, insunits, recovered = pickle.loads(cache_file.read_bytes())
            except (pickle.PickleError, EOFError, AttributeError, ImportError, ValueError):
                pass  # 깨진 캐시는 무시하고 다시 만든다
            else:
                return DxfSource(Path(path), digest, None, insunits, recovered), raw, stats
    with stage("read"):
        source = read_dxf(path)
    with stage("explode"):
        raw, stats = extract_primitives(source.doc, sagitta)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(pickle.dumps((raw, stats, source.insunits, source.recovered)))
    return source, raw, stats


def preprocess(
    path: Path,
    cfg: dict[str, Any],
    region: BBox | None = None,
    unit_scale: float | None = None,
    cache_dir: Path | None = None,
) -> Preprocessed:
    """region은 원본 CAD 좌표(도면 단위)로 받는다. 사용자가 CAD에서 읽은 값이기 때문이다."""
    source, raw, stats = _read_and_explode(path, cfg["curves"]["flatten_sagitta_mm"], cache_dir)

    with stage("scale"):
        scale = decide_scale(raw, source.insunits, cfg["units"], unit_scale)
        in_mm = raw if scale.factor == 1.0 else [p.transformed(scale.factor, 0.0, 0.0) for p in raw]
    region_mm = tuple(v * scale.factor for v in region) if region else None

    with stage("frames"):
        kept, frame_parts, frames = strip_frames(in_mm, cfg["regions"])
    with stage("plan"):
        try:
            # 선으로 그린 칸이 도곽으로 잡혔으면(사각형 도곽이 없을 때) 칸마다 도면 하나로 본다.
            cells = frames if frames and line_cells(in_mm, cfg["regions"])[0] == frames else None
            plan, outside = select_plan(kept, cfg["regions"], frames, region_mm, cells)
        except RegionAmbiguousError as exc:
            exc.scale_factor = scale.factor
            raise

    plan_bbox = union_bbox(plan)
    ox, oy = _origin_offset(plan_bbox, cfg["origin"])
    if ox or oy:
        plan = [p.transformed(1.0, -ox, -oy) for p in plan]
        plan_bbox = union_bbox(plan)
    excluded = [p.transformed(1.0, -ox, -oy) for p in frame_parts + outside]
    frames = [(f[0] - ox, f[1] - oy, f[2] - ox, f[3] - oy) for f in frames]

    with stage("normalize"):
        primitives, normalize_stats = normalize(plan, cfg["normalize"])

    warnings = list(scale.warnings)
    if source.recovered:
        warnings.append("dxf_recovered")
    if stats.transform_failures:
        warnings.append("block_transform_failed")
    return Preprocessed(
        source=source,
        primitives=primitives,
        excluded=excluded,
        scale=scale,
        origin_offset=(ox, oy),
        plan_bbox=plan_bbox,
        region=region,
        frames=frames,
        explode_stats=stats,
        normalize_stats=normalize_stats,
        warnings=warnings,
    )


def load_primitives(
    path: Path, cfg: dict[str, Any], cache_dir: Path | None = None
) -> tuple[list[Primitive], ExplodeStats]:
    """축척 판정 전의 요소. 분류는 단위를 몰라도 해야 한다."""
    _, raw, stats = _read_and_explode(path, cfg["curves"]["flatten_sagitta_mm"], cache_dir)
    return raw, stats


__all__ = ["Preprocessed", "Primitive", "load_primitives", "preprocess"]
