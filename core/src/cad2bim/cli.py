"""명령줄 진입점."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_defaults, load_thresholds
from .errors import Cad2BimError
from .evaluation import compare_results, evaluate, load_truth, run_audit, run_benchmark, show_details
from .ir import IR, ir_json_schema
from .pipeline import convert


def _region(text: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(v) for v in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("x1,y1,x2,y2 숫자 4개가 필요합니다") from exc
    if len(values) != 4 or values[0] >= values[2] or values[1] >= values[3]:
        raise argparse.ArgumentTypeError("x1,y1,x2,y2 형식이고 x1<x2, y1<y2여야 합니다")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cad2bim")
    commands = parser.add_subparsers(dest="command", required=True)

    c = commands.add_parser("convert", help="DXF/DWG를 IR과 보고서로 변환")
    c.add_argument("input", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--floor-height", type=float, default=3000.0)
    c.add_argument("--region", type=_region, help="x1,y1,x2,y2 (원본 CAD 좌표)")
    c.add_argument("--unit-scale", type=float, help="도면 단위 1의 mm 값")
    c.add_argument("--layer-map", type=Path, help="사용자가 고친 layer_map.json")

    e = commands.add_parser("eval", help="IR 하나를 정답과 비교")
    e.add_argument("ir", type=Path)
    e.add_argument("truth", type=Path)
    e.add_argument("--detail", action="store_true", help="요소별 정오와 곡선도 출력")

    b = commands.add_parser("benchmark", help="매니페스트의 도면을 전부 평가")
    b.add_argument("manifest", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--gate", help="통과 기준 이름 (예: m2). 실패하면 종료 코드 1")

    d = commands.add_parser("compare", help="두 results.json의 도면별 퇴보 확인")
    d.add_argument("old", type=Path)
    d.add_argument("new", type=Path)

    a = commands.add_parser("audit", help="실제 도면 폴더 점검. 붙여 넣어도 되는 숫자만 출력")
    a.add_argument("folder", type=Path)
    a.add_argument("--floor-height", type=float, default=3000.0)
    a.add_argument("--limit", type=int, help="무작위로 고른 N장만 실행")
    a.add_argument("--time-limit", type=float, default=180.0, help="도면 한 장의 제한 시간(초)")
    a.add_argument("--jobs", type=int, default=1, help="동시에 처리할 도면 수")
    a.add_argument("--match", help="파일명에 이 글자가 든 도면만 실행. 출력에 찍히지 않음")
    a.add_argument("--classify-only", action="store_true", help="도면 종류만 분류하고 벽·기둥 인식은 건너뜀")
    a.add_argument("--quick", action="store_true", help="회전·마스킹 검증을 건너뛰어 빠르게 실행")
    a.add_argument("--show", help="지난 결과에서 고른 도면만 자세히 출력 (예: D05,D17)")

    r = commands.add_parser("areas", help="면적표와 인식한 실 면적을 대조. 숫자만 출력")
    r.add_argument("folder", type=Path)
    r.add_argument("--table-match", required=True, help="면적표 파일명에 든 글자")
    r.add_argument("--plan-match", required=True, help="평면도 파일명에 든 글자")

    u = commands.add_parser("build", help="DWG 폴더를 건물 하나의 IFC로 변환. 숫자만 출력")
    u.add_argument("folder", type=Path)
    u.add_argument("--floor-height", type=float, help="없으면 도면 주기에서 읽고, 못 읽으면 3000")
    u.add_argument("--jobs", type=int, default=1)

    m = commands.add_parser("compare-model", help="build 결과를 정답 SKP와 비교. 숫자만 출력")
    m.add_argument("folder", type=Path)
    m.add_argument("truth", type=Path)
    m.add_argument("--floor-height", type=float, default=3000.0)
    m.add_argument("--z-cut", type=float, default=1200.0)

    s = commands.add_parser("schema", help="IR JSON 스키마 출력")
    s.add_argument("--out", type=Path)
    return parser


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(args: argparse.Namespace) -> int:
    if args.command == "convert":
        result = convert(
            args.input, args.out, load_defaults(), args.floor_height, args.region,
            args.unit_scale, args.layer_map,
        )
        pre = result.pre
        print(f"축척 {pre.scale.factor:g} ({pre.scale.source}), 요소 {len(pre.primitives)}개")
        print(f"벽 {len(result.ir.walls)}, 기둥 {len(result.ir.columns)}")
        for code in result.ir.warnings:
            print(f"경고: {code}")
        print(f"IR: {result.ir_path}")
        print(f"보고서: {result.report_path}")
    elif args.command == "eval":
        ir = IR.model_validate_json(args.ir.read_text(encoding="utf-8"))
        metrics = evaluate(ir, load_truth(args.truth), load_thresholds())
        if not args.detail:
            metrics.pop("outcomes")
            metrics["confidence"].pop("reliability")
            metrics["confidence"].pop("risk_coverage")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.command == "benchmark":
        results = run_benchmark(args.manifest, args.out, args.gate)
        print((args.out / "summary.md").read_text(encoding="utf-8"), end="")
        if results["gate"] and not results["gate"]["passed"]:
            return 1
    elif args.command == "audit":
        try:
            if args.show:
                block = show_details(args.folder, args.show.split(","))
            else:
                block = run_audit(args.folder, args.floor_height, args.limit, args.time_limit, args.jobs, args.quick, args.match, args.classify_only)
        except Cad2BimError:
            raise
        except Exception as exc:  # 트레이스백에는 경로가 있을 수 있다
            raise Cad2BimError(f"점검을 끝내지 못했습니다: {type(exc).__name__}") from None
        print("상세 결과는 입력 폴더의 _cad2bim_out에 있습니다. 그 폴더 내용은 붙여 넣지 마세요.")
        print("----- 아래부터 복사 -----")
        print(block, end="")
        print("----- 여기까지 복사 -----")
    elif args.command == "build":
        from .building import run_build

        print(run_build(args.folder, args.floor_height, args.jobs), end="")
    elif args.command == "compare-model":
        from .evaluation.model_compare import compare_model
        from .export.skp import find_sdk

        cfg = load_defaults()["export"]
        sdk = find_sdk(cfg)
        if sdk is None:
            raise Cad2BimError("SketchUp SDK가 없어 정답 SKP를 읽을 수 없습니다.")
        print("\n".join(compare_model(args.folder.resolve(), args.truth, args.floor_height, args.z_cut, cfg, sdk)))
    elif args.command == "areas":
        from .evaluation.areas import run_areas

        logging.disable(logging.CRITICAL)
        try:
            print(run_areas(args.folder, args.table_match, args.plan_match), end="")
        except Cad2BimError:
            raise
        except Exception as exc:  # 트레이스백에는 경로가 있을 수 있다
            raise Cad2BimError(f"대조를 끝내지 못했습니다: {type(exc).__name__}") from None
        finally:
            logging.disable(logging.NOTSET)
    elif args.command == "compare":
        tolerance = load_thresholds()["benchmark"]["regression_tolerance"]
        regressions = compare_results(_read_json(args.old), _read_json(args.new), tolerance)
        for r in regressions:
            print(f"퇴보: {r['drawing']} {r['metric']}: {r['old']} → {r['new']}")
        print(f"퇴보 {len(regressions)}건")
        return 1 if regressions else 0
    else:
        text = json.dumps(ir_json_schema(), ensure_ascii=False, indent=2) + "\n"
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    from .envfile import load_env

    load_env()  # 저장소 맨 위 .env의 키(LLM 등)를 환경 변수로. 이미 설정된 값이 먼저다
    try:
        return _run(args)
    except Cad2BimError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
