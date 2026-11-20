#!/bin/sh
# 합성 시험 도면 30장 생성 → 규칙만 / 규칙+ML 로 조립 → 문·창 개수와 위치 F1 채점
# 사용: sh eval/run_counts.sh <작업 폴더>   (저장소 맨 위에서 실행, venv 활성화 상태)
set -e
W="${1:-_eval_work}"; R="$(pwd)"
(cd DATASET/cad2bim_synthetic_drawings/cad2bim/synth2 && python writer2.py --n 10 --seed 5000 --out "$R/$W/synth_eval" --bench)
for v in rules ml; do
  for b in "$W"/synth_eval/S50*; do
    n=$(basename "$b"); mkdir -p "$W/$v/$n"; cp "$b"/*.dxf "$b"/*.bench.json "$W/$v/$n/"
    if [ "$v" = ml ]; then M=ml; else M=off; fi
    (cd core && CAD2BIM_LLM=$M python -m cad2bim.cli build "$R/$W/$v/$n" > /dev/null)
  done
done
python eval/score_counts.py "$W" "$W/synth_eval"
