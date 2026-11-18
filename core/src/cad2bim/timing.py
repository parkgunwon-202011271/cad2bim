"""단계별 시간 기록. 실제 도면은 내용을 볼 수 없으므로 어디가 느린지 숫자로 안다."""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageRecord:
    current: str | None = None
    seconds: dict[str, float] = field(default_factory=dict)
    depth: int = 0


_records: dict[int, StageRecord] = {}


@contextmanager
def stage(name: str) -> Iterator[None]:
    record = _records.setdefault(threading.get_ident(), StageRecord())
    if record.depth:  # 안쪽 단계는 바깥 단계에 포함한다. 합계가 실제 시간과 맞아야 한다
        yield
        return
    record.current = name
    record.depth += 1
    started = time.perf_counter()
    try:
        yield
    finally:
        record.depth -= 1
        record.seconds[name] = record.seconds.get(name, 0.0) + time.perf_counter() - started


def take_record(thread_id: int) -> StageRecord:
    """그 스레드의 기록을 꺼낸다. 다음 도면과 섞이지 않게 지운다."""
    return _records.pop(thread_id, StageRecord())
