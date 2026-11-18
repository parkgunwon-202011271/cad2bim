"""중간 표현(IR). Python과 C# 애드인의 유일한 계약이며 이 모델이 스키마의 원본이다.

단위는 mm, 각도는 라디안, 좌표는 origin_offset 적용 후 값이다.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1"

Point = tuple[float, float]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Confidence(_Model):
    level: Literal["high", "review"]
    score: float = Field(ge=0, le=1)
    reasons: list[str] = []


class SourceInfo(_Model):
    file: str
    sha256: str
    region: tuple[float, float, float, float] | None = None
    unit_scale: float = Field(gt=0, description="도면 단위 → mm 배율")
    unit_source: Literal["user", "dimension", "insunits"]


class Level(_Model):
    name: str = "1F"
    elevation: float = 0
    height: float = Field(gt=0)


class Wall(_Model):
    id: str
    start: Point
    end: Point
    thickness: float = Field(gt=0)
    kind: Literal["concrete", "masonry", "curtain", "unknown"]
    confidence: Confidence
    source_handles: list[str] = []


class Column(_Model):
    id: str
    shape: Literal["rect", "circle", "polygon"]
    center: Point
    size: tuple[float, float]
    rotation: float = 0
    polygon: list[Point] | None = Field(default=None, description="shape가 polygon일 때 외곽")
    wall_piece: bool = Field(default=False, description="벽 레이어의 짧은 조각. 벽으로 표시한다")
    confidence: Confidence
    source_handles: list[str] = []


class Opening(_Model):
    id: str
    type: Literal["door", "window", "unknown"]
    operation: Literal["swing", "double", "sliding", "fixed"] | None = None
    host_wall: str
    position: float = Field(ge=0, description="호스트 벽 시작점에서 개구부 중심까지의 거리")
    width: float = Field(gt=0)
    hinge: Point | None = None
    swing_dir: Point | None = None
    confidence: Confidence
    source_handles: list[str] = []


class Room(_Model):
    id: str
    names: list[str]
    point: Point
    polygon: list[Point] = Field(min_length=3, description="순면적 외곽")
    clear_area_m2: float = Field(ge=0)
    boundary_wall_ids: list[str] = []
    confidence: Confidence
    source_handles: list[str] = []


class Flight(_Model):
    start: Point
    end: Point
    width: float = Field(gt=0)
    treads: int = Field(ge=1)


class Stair(_Model):
    id: str
    bounds: tuple[float, float, float, float]
    flights: list[Flight]
    confidence: Confidence
    source_handles: list[str] = []


class Rail(_Model):
    id: str
    start: Point
    end: Point
    confidence: Confidence
    source_handles: list[str] = []


class Void(_Model):
    id: str
    bounds: tuple[float, float, float, float]
    confidence: Confidence
    source_handles: list[str] = []


class Unsupported(_Model):
    kind: str
    handles: list[str] = []
    bbox: tuple[float, float, float, float] | None = None
    note: str | None = None


class IR(_Model):
    schema_version: str = SCHEMA_VERSION
    source: SourceInfo
    units: Literal["mm"] = "mm"
    origin_offset: Point = (0.0, 0.0)
    level: Level
    walls: list[Wall] = []
    columns: list[Column] = []
    openings: list[Opening] = []
    rooms: list[Room] = []
    stairs: list[Stair] = []
    voids: list[Void] = []
    rails: list[Rail] = []
    unsupported: list[Unsupported] = []
    warnings: list[str] = Field(default=[], description="도면 전체에 대한 문자열 코드")


def ir_json_schema() -> dict[str, Any]:
    return IR.model_json_schema()
